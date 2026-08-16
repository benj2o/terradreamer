"""Re-run the HEADLINE P3 configs with per-observation predictions, then score
the triggers.

WHY A NARROWED RE-RUN AND NOT THE WHOLE TABLE
-----------------------------------------------
The published Tier-1 run is 1540 rows over five (aggregation, fold mode) views
and took 173.7 min on 7 workers. Most of that is spent where the trigger
question has nothing to ask:

* ``fold_mode=loco`` is 115 folds of two to five rows each. A contingency table
  over a fold that holds three observations is not a contingency table, and the
  fold-clustered interval this module reports would be a jackknife over 115
  degenerate cells.
* ``aggregation=cell_mean`` is 16 rows per forecast row at ``grid_cell``
  feature level -- the RESOLUTION axis, not the decision axis. A trigger is a
  statement about a place, and the place here is the cube.
* ``aggregation=cube_p90`` is the secondary target.
* ``alpha_rule=fixed_alpha_D`` is kept in the published table because that is
  what the 2026-08-12 run was computed under, but nested CV is the rule the
  Tier-1 headline is read on.

So this runs exactly the headline configs:

    aggregation   cube_mean          (feature_level pooled, by CONTEXT_LEVEL)
    fold_mode     cube, spatial_block
    alpha_rule    nested_cv, not_a_ridge
    horizons      all four
    model kinds   all nine, all nine encoder views

WHAT IT WRITES, AND WHAT IT REFUSES TO CLAIM
----------------------------------------------
The table it produces is a SUBSET table. It is written under its own name and
``p3_forecast``'s completeness assertions are NOT run on it, because it would
fail them and should: a table missing the fixed-alpha rows and three of the five
views is not the published table. What IS checked, and is the point of writing
it at all, is that every scoring column it shares with
``p3_tier1_results.csv`` is bit-identical -- which is the evidence that turning
the predictions on changed nothing.

    .venv/bin/python -m scripts.rerun_p3_predictions --tile 32UNU --n-jobs 7
"""

from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np

#: the config key a row is identified by, in both tables
KEY = ["delta_days", "aggregation", "fold_mode", "encoder", "feature_set",
       "model_kind", "estimator", "alpha_rule", "feature_base"]

#: what must not have moved. Scores and fold bookkeeping -- NOT the paired or
#: margin columns, which are computed ACROSS the rows of a table and legitimately
#: differ when the table holds a different set of rows.
SCORING_COLUMNS = [
    "r2_mean", "r2_std", "r2_min", "r2_max", "r2_ci_lo", "r2_ci_hi",
    "rmse_mean", "mae_mean", "r2_pooled", "r2_pooled_ci_lo", "r2_pooled_ci_hi",
    "skill_vs_persistence", "skill_vs_persistence_ci_lo",
    "skill_vs_persistence_ci_hi", "rmse_pooled", "mae_pooled", "medae_pooled",
    "medae_persistence", "skill_vs_persistence_medae", "sse_share_top1pct",
    "n_folds", "n_folds_empty", "n_folds_nan", "n_rows_pooled",
    "n_rows_test_total", "n_train_median", "effective_n", "D",
    "alpha_median", "alpha_min", "alpha_max", "n_folds_alpha_at_grid_edge",
    "n_retained", "n_cubes", "n_feature_rows", "per_fold_r2",
]


def compare_against_published(new, old) -> None:
    """Every scoring column of every shared row, bit for bit."""
    import pandas as pd

    if old is None:
        print("\n[p3-pred] no published Tier-1 CSV on disk; the bit-identity "
              "check is SKIPPED rather than reported as passing")
        return
    cols = [c for c in SCORING_COLUMNS if c in new.columns and c in old.columns]
    m = new[KEY + cols].merge(old[KEY + cols], on=KEY, how="inner",
                              suffixes=("_new", "_old"))
    print(f"\n[p3-pred] {len(m)} rows shared with the published Tier-1 table "
          f"({len(new)} in this run, {len(old)} published); comparing "
          f"{len(cols)} scoring columns")
    assert len(m), (
        "no row of this run matches a published row on "
        f"{KEY}. Either the key changed or the run did not cover the "
        "published configs -- either way the comparison is not being made."
    )
    bad = []
    for c in cols:
        a = m[f"{c}_new"].to_numpy()
        b = m[f"{c}_old"].to_numpy()
        if a.dtype.kind == "f" and b.dtype.kind == "f":
            same = np.array_equal(a, b, equal_nan=True)
            worst = float(np.nanmax(np.abs(a - b))) if not same else 0.0
        else:
            same, worst = bool((a == b).all()), float("nan")
        if not same:
            bad.append((c, int((~((a == b) | (pd.isna(a) & pd.isna(b)))).sum()),
                        worst))
    if bad:
        print("[p3-pred] *** THESE COLUMNS MOVED ***")
        for c, n, w in bad:
            print(f"    {c:<32} {n:>5} row(s), worst |diff| {w:.3e}")
    assert not bad, (
        f"{len(bad)} scoring column(s) differ from the published table. "
        "emit_predictions must not change a number; a run that writes the "
        "artefact and a run that does not have to produce the same scores."
    )
    print(f"[p3-pred] every one of {len(cols)} scoring columns is "
          f"BIT-IDENTICAL on all {len(m)} shared rows")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tile", default="32UNU")
    ap.add_argument("--out", default=None, help="cache root; default data/scaled_<tile>")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=0)
    ap.add_argument("--horizons", default="")
    ap.add_argument("--fold-modes", default="cube,spatial_block")
    ap.add_argument("--encoders", default="")
    ap.add_argument("--csv-name", default="p3_tier1_subset_results.csv",
                    help="the SUBSET table. Deliberately not the published "
                         "name -- it is missing rows the published one has.")
    ap.add_argument("--predictions-name", default="p3_tier1_predictions.csv")
    ap.add_argument("--triggers-name", default="p3_tier1_triggers.csv")
    ap.add_argument("--published-csv", default=None)
    ap.add_argument("--no-screen", action="store_true")
    ap.add_argument("--skip-triggers", action="store_true")
    args = ap.parse_args()

    import pandas as pd

    from data.loader import load_cube
    from encoders.manifest import (assert_strata_present, assert_weather_join,
                                   build_manifest)
    from probes import p3_forecast as p3
    from probes import p3_triggers as tg

    out_root = args.out or os.path.join("data", f"scaled_{args.tile}")
    cube_dir = os.path.join(out_root, "raw")
    emb_dir = os.path.join(out_root, "embeddings")
    emb_dir_cir = os.path.join(out_root, p3.CIR_EMB_DIRNAME)
    mask_dir = os.path.join(out_root, "masks")
    for d, what in ((cube_dir, "cubes"), (emb_dir, "RGB embeddings"),
                    (emb_dir_cir, "colour-infrared embeddings"),
                    (mask_dir, "masks")):
        assert os.path.isdir(d), f"no {what} at {d}"

    encoders = (tuple(args.encoders.split(",")) if args.encoders
                else p3.ENCODER_VIEWS_ALL)
    horizons = (tuple(int(x) for x in args.horizons.split(","))
                if args.horizons else p3.HORIZONS)
    fold_modes = tuple(args.fold_modes.split(","))
    alpha_rules = (p3.ALPHA_RULE_TUNED, p3.ALPHA_RULE_NA)

    paths = sorted(glob.glob(os.path.join(cube_dir, "*.nc")))
    print(f"[p3-pred] {len(paths)} cubes at {cube_dir}")
    print(f"[p3-pred] SUBSET: aggregation=cube_mean, fold_modes={fold_modes}, "
          f"alpha_rules={alpha_rules}, horizons={horizons}, "
          f"{len(encoders)} encoder views, all model kinds")

    t0 = time.time()
    manifest = build_manifest([load_cube(p, verbose=False) for p in paths],
                              verbose=False)
    assert_strata_present(manifest)
    join = assert_weather_join(manifest, cube_dir, verbose=False)
    assert max(join["max_abs_diff"].values()) == 0.0, join
    print(f"[p3-pred] manifest {manifest.shape}, "
          f"{manifest.cube_id.nunique()} cubes, {time.time() - t0:.0f}s")

    pred_path = os.path.join(out_root, args.predictions_name)
    t0 = time.time()
    df, data = p3.run_p3(manifest, cube_dir, encoders=encoders,
                         horizons=horizons, aggregations=("cube_mean",),
                         fold_modes=fold_modes, alpha_rules=alpha_rules,
                         k=args.k, emb_dir=emb_dir, emb_dir_cir=emb_dir_cir,
                         mask_dir=mask_dir,
                         plausibility_screen=not args.no_screen,
                         n_jobs=args.n_jobs or max(1, (os.cpu_count() or 2) - 1),
                         emit_predictions=True, predictions_path=pred_path,
                         log_path=os.path.join(out_root,
                                               "p3_tier1_predictions_run.log"),
                         verbose=True)
    mins = (time.time() - t0) / 60
    print(f"\n[p3-pred] run_p3: {df.shape[0]} rows x {df.shape[1]} cols in "
          f"{mins:.1f} min")

    csv = p3.results_path(args.csv_name, root=out_root)
    df.to_csv(csv, index=False)
    print(f"[p3-pred] wrote {csv} -- A SUBSET TABLE. It is missing the "
          "fixed_alpha_D rows and three of the five views, so the "
          "completeness assertions are NOT run on it and it must not be read "
          "as the published Tier-1 table.")

    # The invariants that ARE meaningful on a subset: they are about how a row
    # was scored, not about which rows exist.
    #
    # The MARGIN-derived checks are not among them, and that is a consequence
    # of the scope rather than an oversight. An unfitted baseline takes the
    # ridge control at the FIXED penalty rule (add_margins' documented
    # fallback), and this run does not compute the fixed-alpha rows -- so
    # ``margin_over_control`` has nothing to point at, add_margins refuses the
    # table, and everything that reads its columns
    # (assert_separability_is_paired, assert_control_identical_across_views)
    # has nothing to read. The PAIRED columns run_p3 already attached are still
    # on every row, and the check that matters about them -- that no comparison
    # was made over a partial intersection -- is made directly below. Nothing
    # downstream reads the margins: the trigger table is built from the
    # predictions.
    p3.assert_climatology_rows_labelled(df)
    p3.assert_mi_flagged_and_single_frame(df)
    p3.assert_effective_n_counts_cubes(df)
    p3.assert_plausibility_screen_declared(df, required=not args.no_screen)
    assert int(df.paired_rows_lost_max.max()) == 0, (
        f"a paired comparison lost up to {int(df.paired_rows_lost_max.max())} "
        "held-out row(s) to the intersection: the two rows were not scored on "
        "the same observations, so their difference is not paired"
    )
    assert (df.separability_test.astype(str).str.contains("paired")).all()
    print("[p3-pred] the four row-level invariants PASS on the subset table, "
          "and no paired comparison lost a held-out row")

    old_csv = args.published_csv or os.path.join(out_root,
                                                 "p3_tier1_results.csv")
    old = pd.read_csv(old_csv) if os.path.exists(old_csv) else None
    compare_against_published(df, old)

    if args.skip_triggers:
        return
    path = pred_path if os.path.exists(pred_path) else pred_path + ".gz"
    pred = tg.load_predictions(path, verbose=True)
    tr = tg.trigger_metrics(pred, verbose=True)
    tpath = os.path.join(out_root, args.triggers_name)
    tr.to_csv(tpath, index=False)
    print(f"[p3-pred] wrote {tpath} ({tr.shape[0]} rows x {tr.shape[1]} cols)")

    print("\n\n" + "#" * 92)
    print("# THE TRIGGER TABLES -- lead time across the columns")
    print("#" * 92)
    for mode in fold_modes:
        for lv in tg.TRIGGER_LEVELS:
            for m in ("hit_rate", "false_alarm_rate", "csi", "pss"):
                tg.print_trigger_table(tr, level=lv, fold_mode=mode, metric=m)


if __name__ == "__main__":
    main()
