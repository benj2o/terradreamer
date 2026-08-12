"""Re-run the whole P3 table under the four Tier-1 corrections, on 115 cubes.

WHY THIS IS A SCRIPT AND NOT A NOTEBOOK CELL
---------------------------------------------
The 2026-08-12 P3 run is published: a 460-row CSV, an archived notebook and a
set of claims in log.md that rest on it. This run changes four things at once --
the penalty rule, the separability test, the feature base and the encoder roster
-- and the only useful output is a COMPARISON against that table. A comparison
written by hand each time is a comparison that drifts, so it is code, and the
diff is the artefact rather than something a reader has to reconstruct.

It also runs for hours. A script can be launched in the background, writes its
own verbatim stdout, and does not lose its output to a notebook kernel.

WHAT CHANGES, AND WHAT EACH CHANGE IS FOR
-------------------------------------------
1. ``alpha_rule``      the ridge additionally runs under a SELECTED penalty
                       (``p2_deltas.select_ridge_alpha``, nested CV on the
                       training fold). alpha = D spans 79 to 11536 across these
                       views, and the narrow end is the band-matched baseline
                       every network is measured against. BOTH rules are kept.
2. paired separability every "X beats Y" is the paired per-fold difference with
                       a fold-clustered interval, never two marginal CIs.
3. ``feature_base``    [NDVI(t), weather] under every model row, so a row
                       answers "what does this representation add beyond current
                       NDVI and weather" -- which is what removes
                       ``raw_features``' unearned advantage from holding
                       NDVI_mean(t) while no other row did.
4. nine encoder views  5 RGB + 4 colour-infrared, with the plausibility screen
                       APPLIED. The headline is the paired difference between
                       each ``_cir`` row and its own ``_rgb`` twin: same weights,
                       same frames, same read-out, only the bands differ.

WHAT IT REFUSES TO DO
----------------------
Nothing is fine-tuned and no encoder is imported: this reads the frozen caches
under ``data/scaled_<tile>/{embeddings,embeddings_cir}`` and the cubes. It never
writes to ``data/raw`` or ``data/phase1_2``, and it writes its table beside the
cubes it was computed from.

    .venv/bin/python -m scripts.rerun_p3_tier1 --tile 32UNU --n-jobs 7
"""

from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np


def _fmt(x, w=7, p=3):
    return f"{x:+{w}.{p}f}" if np.isfinite(x) else " " * (w - 3) + "nan"


def _cell(df, **q):
    """The single row matching a query, or None. Never the first of several."""
    if df is None:
        return None
    sub = df
    for k, v in q.items():
        if k not in sub.columns:
            return None
        sub = sub[sub[k] == v]
    if len(sub) != 1:
        return None
    return sub.iloc[0]


def compare_against_published(new, old, args) -> None:
    """Every headline beside its 2026-08-12 value.

    ``old`` may be None -- the published CSV is a build artefact and a fresh
    checkout will not have it. The new numbers are still printed; only the
    comparison column is dropped, and it says so rather than printing zeros.
    """
    from probes import p3_forecast as p3

    print("\n" + "=" * 92)
    print("THE PUBLISHED ROW, RE-MEASURED -- cube_mean / cube folds / ridge, "
          "pooled out-of-fold R2")
    print("=" * 92)
    if old is None:
        print("  no published CSV on disk; the 'was' column is omitted rather "
              "than filled with zeros")
    print("  'was' is the 2026-08-12 run: fixed alpha = D, no shared base, no "
          "plausibility screen, 5 views.")
    print(f"  'now' is this run at feature_base={args.report_base}, "
          f"alpha_rule={args.report_rule}, screen applied.\n")

    encs = [e for e in p3.ENCODER_VIEWS_ALL]
    header = f"  {'row':<34}" + "".join(f"{('D=' + str(H)):>18}"
                                        for H in p3.HORIZONS)
    print(header)
    rows = [(e, "embedding") for e in encs] + \
           [(p3.BASELINE_ENCODER, p3.BAND_MATCHED_BASELINE)]
    for enc, fs in rows:
        cells = []
        for H in p3.HORIZONS:
            n = _cell(new, delta_days=H, aggregation="cube_mean",
                      fold_mode="cube", estimator="linear",
                      alpha_rule=args.report_rule,
                      feature_base=args.report_base, encoder=enc,
                      feature_set=fs)
            o = _cell(old, delta_days=H, aggregation="cube_mean",
                      fold_mode="cube", estimator="linear", encoder=enc,
                      feature_set=fs)
            cells.append(f"{_fmt(n.r2_pooled) if n is not None else '    n/a'}"
                         f" ({_fmt(o.r2_pooled) if o is not None else '  --  '})")
        tag = "  <- BAND-MATCHED" if fs == p3.BAND_MATCHED_BASELINE else ""
        print(f"  {enc + '/' + fs:<34}" + "".join(f"{c:>18}" for c in cells) + tag)
    for kind in ("persistence", "climatology_proxy", "weather_only",
                 "observation", "permutation", "horizon_only"):
        cells = []
        for H in p3.HORIZONS:
            q = dict(delta_days=H, aggregation="cube_mean", fold_mode="cube",
                     model_kind=kind)
            if kind in ("weather_only", "observation", "permutation"):
                q.update(estimator="linear")
                n = _cell(new, **q, alpha_rule=args.report_rule,
                          feature_base=(args.report_base
                                        if kind == "weather_only"
                                        else p3.FEATURE_BASE_NONE))
            else:
                n = _cell(new, **q)
            o = _cell(old, **q)
            cells.append(f"{_fmt(n.r2_pooled) if n is not None else '    n/a'}"
                         f" ({_fmt(o.r2_pooled) if o is not None else '  --  '})")
        print(f"  {kind:<34}" + "".join(f"{c:>18}" for c in cells))


def print_penalty_effect(df) -> None:
    """What the penalty rule alone did, by design width. The reason for (a)."""
    from probes import p3_forecast as p3

    print("\n" + "=" * 92)
    print("WHAT THE PENALTY RULE ALONE DID -- nested_cv minus fixed_alpha_D, "
          "same rows, same folds")
    print("  alpha = D is set by the WIDTH of the design, and the widths here "
          "differ by 146x.")
    print("=" * 92)
    key = ["encoder", "feature_set", "model_kind", "feature_base"]
    sub = df[(df.aggregation == "cube_mean") & (df.fold_mode == "cube")
             & (df.estimator == "linear") & (~df.is_control)]
    if not len(sub):
        print("  (no ridge rows)")
        return
    for base in p3.FEATURE_BASES:
        s = sub[sub.feature_base == base]
        if not len(s):
            continue
        print(f"\n  feature_base = {base}")
        print(f"  {'row':<40} {'D':>7} {'alpha(cv) med':>14} "
              + "".join(f"{('D=' + str(H)):>12}" for H in p3.HORIZONS))
        for name, g in s.groupby(key[:2]):
            fixed = g[g.alpha_rule == p3.ALPHA_RULE_FIXED]
            tuned = g[g.alpha_rule == p3.ALPHA_RULE_TUNED]
            if not len(fixed) or not len(tuned):
                continue
            D = int(fixed.D.iloc[0])
            med = float(np.median(tuned.alpha_median.to_numpy()))
            deltas = []
            for H in p3.HORIZONS:
                f = fixed[fixed.delta_days == H]
                t = tuned[tuned.delta_days == H]
                deltas.append(_fmt(float(t.r2_pooled.iloc[0])
                                   - float(f.r2_pooled.iloc[0]), 12, 3)
                              if len(f) and len(t) else f"{'n/a':>12}")
            print(f"  {'/'.join(name):<40} {D:>7d} {med:>14g} "
                  + "".join(deltas))


def print_base_effect(df) -> None:
    """What the shared base alone did. The reason for (c)."""
    from probes import p3_forecast as p3

    print("\n" + "=" * 92)
    print("WHAT THE SHARED BASE ALONE DID -- +[NDVI(t)] minus no base, same "
          "rows, same folds, same penalty")
    print("  Before it, ONE row (raw_features) held NDVI_mean(t) and no other "
          "did.")
    print("=" * 92)
    sub = df[(df.aggregation == "cube_mean") & (df.fold_mode == "cube")
             & (df.estimator == "linear")
             & (df.alpha_rule == p3.ALPHA_RULE_TUNED) & (~df.is_control)]
    print(f"  {'row':<40} " + "".join(f"{('D=' + str(H)):>12}"
                                      for H in p3.HORIZONS))
    for name, g in sub.groupby(["encoder", "feature_set"]):
        a = g[g.feature_base == p3.FEATURE_BASE_SHARED]
        b = g[g.feature_base == p3.FEATURE_BASE_NONE]
        if not len(a) or not len(b):
            continue
        cells = []
        for H in p3.HORIZONS:
            x, y = a[a.delta_days == H], b[b.delta_days == H]
            cells.append(_fmt(float(x.r2_pooled.iloc[0])
                              - float(y.r2_pooled.iloc[0]), 12, 3)
                         if len(x) and len(y) else f"{'n/a':>12}")
        print(f"  {'/'.join(name):<40} " + "".join(cells))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tile", default="32UNU")
    ap.add_argument("--out", default=None,
                    help="cache root; default data/scaled_<tile>")
    ap.add_argument("--old-csv", default=None,
                    help="the published P3 CSV to diff against")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=0)
    ap.add_argument("--horizons", default="",
                    help="comma-separated; default p3.HORIZONS")
    ap.add_argument("--aggregations", default="",
                    help="comma-separated; default p3.AGGREGATIONS")
    ap.add_argument("--encoders", default="",
                    help="comma-separated; default all nine views")
    ap.add_argument("--max-cubes", type=int, default=0,
                    help="SMOKE TEST ONLY: strided subset of the cubes, so the "
                         "whole path can be exercised in minutes. Any table it "
                         "produces is a shape check, never a result -- the "
                         "effective n is CUBES and this changes it.")
    ap.add_argument("--no-screen", action="store_true",
                    help="do NOT apply the plausibility screen (it is still "
                         "declared on every row, as False)")
    ap.add_argument("--report-base", default=None)
    ap.add_argument("--report-rule", default=None)
    ap.add_argument("--csv-name", default="p3_tier1_results.csv")
    args = ap.parse_args()

    import pandas as pd

    from data.loader import load_cube
    from encoders import TIER_A, TIER_A_CIR
    from encoders.manifest import (assert_strata_present, assert_weather_join,
                                   build_manifest)
    from encoders.pipeline import audit_embeddings
    from probes import p3_forecast as p3

    args.report_base = args.report_base or p3.FEATURE_BASE_SHARED
    args.report_rule = args.report_rule or p3.ALPHA_RULE_TUNED

    out_root = args.out or os.path.join("data", f"scaled_{args.tile}")
    cube_dir = os.path.join(out_root, "raw")
    emb_dir = os.path.join(out_root, "embeddings")
    emb_dir_cir = os.path.join(out_root, p3.CIR_EMB_DIRNAME)
    mask_dir = os.path.join(out_root, "masks")
    for d, what in ((cube_dir, "cubes"), (emb_dir, "RGB embeddings"),
                    (emb_dir_cir, "colour-infrared embeddings"),
                    (mask_dir, "masks")):
        assert os.path.isdir(d), (
            f"no {what} at {d}. Phase 1.7 builds the RGB cache and Phase 1.9 "
            "(notebooks/phase1_9_cir_encoding.ipynb) builds the colour-infrared "
            "one; neither is produced by a probe."
        )

    encoders = (tuple(args.encoders.split(",")) if args.encoders
                else p3.ENCODER_VIEWS_ALL)
    horizons = (tuple(int(x) for x in args.horizons.split(","))
                if args.horizons else p3.HORIZONS)
    aggregations = (tuple(args.aggregations.split(",")) if args.aggregations
                    else p3.AGGREGATIONS)

    paths = sorted(glob.glob(os.path.join(cube_dir, "*.nc")))
    if args.max_cubes:
        # STRIDED, not the first n: the filenames sort by window-start date, so
        # a prefix shares a narrow slice of the season and the proxy climatology
        # is not identifiable on it.
        stride = max(1, len(paths) // args.max_cubes)
        paths = paths[::stride][:args.max_cubes]
        print(f"[p3-tier1] *** SMOKE TEST: {len(paths)} cubes of "
              f"{len(glob.glob(os.path.join(cube_dir, '*.nc')))}. The effective "
              "n is CUBES, so no number below is a result. ***")
    cube_ids = {os.path.basename(p) for p in paths}
    print(f"[p3-tier1] {len(paths)} cubes at {cube_dir}")
    print(f"[p3-tier1] {len(glob.glob(os.path.join(emb_dir, '*.npz')))} rgb "
          f"embeddings, "
          f"{len(glob.glob(os.path.join(emb_dir_cir, '*.npz')))} cir "
          f"embeddings, {len(glob.glob(os.path.join(mask_dir, '*.npz')))} masks")
    print(f"[p3-tier1] encoder views ({len(encoders)}): {list(encoders)}")

    # --- 1. both caches are whole ----------------------------------------
    want_rgb = [e for e in encoders if not p3.is_cir(e)]
    want_cir = [e for e in encoders if p3.is_cir(e)]
    for d, want in ((emb_dir, want_rgb), (emb_dir_cir, want_cir)):
        if not want:
            continue
        audit = audit_embeddings(d, cube_ids=cube_ids, verbose=False)
        missing = [(c, e) for c in sorted(cube_ids) for e in want
                   if not os.path.exists(
                       os.path.join(d, f"{os.path.splitext(c)[0]}__{e}.npz"))]
        assert not missing, (
            f"{len(missing)} (cube, encoder) pair(s) missing from {d}, e.g. "
            f"{missing[:3]}. A per-encoder comparison over a cache with holes "
            "is a comparison over DIFFERENT cubes."
        )
        print(f"[p3-tier1] {d}: complete for {len(want)} view(s) x "
              f"{len(cube_ids)} cubes ({len(audit.current)} usable files, "
              f"{len(audit.duplicates)} duplicate(s), "
              f"{len(audit.unreadable)} unreadable)")
    assert set(want_cir) <= set(TIER_A_CIR), (
        f"unknown colour-infrared view(s) {sorted(set(want_cir) - set(TIER_A_CIR))}"
    )
    assert set(want_rgb) <= set(TIER_A), (
        f"unknown RGB view(s) {sorted(set(want_rgb) - set(TIER_A))}"
    )

    # --- 2. the manifest, rebuilt fresh ----------------------------------
    t0 = time.time()
    samples = [load_cube(p, verbose=False) for p in paths]
    manifest = build_manifest(samples, verbose=False)
    assert_strata_present(manifest)
    join = assert_weather_join(manifest, cube_dir, verbose=False)
    assert max(join["max_abs_diff"].values()) == 0.0, join
    print(f"[p3-tier1] manifest {manifest.shape}, "
          f"{manifest.cube_id.nunique()} cubes, "
          f"years {sorted(manifest.year.unique().tolist())}, "
          f"weather join 0 rows off their own day, {time.time() - t0:.0f}s")
    del samples

    # --- 3. the run -------------------------------------------------------
    t0 = time.time()
    n_jobs = args.n_jobs or max(1, (os.cpu_count() or 2) - 1)
    df, data = p3.run_p3(manifest, cube_dir, encoders=encoders,
                         horizons=horizons, aggregations=aggregations,
                         k=args.k, emb_dir=emb_dir, emb_dir_cir=emb_dir_cir,
                         mask_dir=mask_dir,
                         plausibility_screen=not args.no_screen,
                         n_jobs=n_jobs,
                         log_path=os.path.join(out_root, "p3_tier1_run.log"),
                         verbose=True)
    df = p3.add_margins(df, verbose=True)
    mins = (time.time() - t0) / 60
    print(f"\n[p3-tier1] run_p3: {df.shape[0]} rows x {df.shape[1]} cols in "
          f"{mins:.1f} min on {n_jobs} workers (fold-level parallelism; it "
          "changes wall-clock, never a number)")

    # --- 4. the table, WRITTEN BEFORE it is asserted -----------------------
    # A run that ends in an exception after four hours leaves nothing to
    # diagnose. A run that writes the table and then refuses it leaves the
    # evidence on disk, and the assertions below re-check the CSV as well as
    # the frame, so nothing is trusted because it was written.
    csv = p3.results_path(args.csv_name, root=out_root)
    df.to_csv(csv, index=False)
    print(f"\n[p3-tier1] wrote {csv} ({df.shape[0]} rows x {df.shape[1]} cols) "
          "-- the invariants run NEXT, on this table and on the CSV")
    for name, table in (("retention", data.retention),
                        ("survival", data.survival),
                        ("tolerance", data.tolerance),
                        ("outliers", data.outliers)):
        p = os.path.join(out_root, f"p3_tier1_{name}.csv")
        table.to_csv(p, index=False)
        print(f"[p3-tier1] wrote {p}")

    # --- 5. the invariants ------------------------------------------------
    p3.assert_results_complete(df, encoders=encoders, horizons=horizons)
    p3.assert_baselines_present(df)
    p3.assert_controls_present(df)
    p3.assert_control_identical_across_views(df)
    p3.assert_climatology_rows_labelled(df)
    p3.assert_mi_flagged_and_single_frame(df)
    p3.assert_effective_n_counts_cubes(df)
    p3.assert_retention_shrinks(data.retention, verbose=False)
    p3.assert_alpha_rules_present(df)
    p3.assert_shared_base_present(df)
    p3.assert_plausibility_screen_declared(df, required=not args.no_screen)
    if want_cir:
        p3.assert_cir_twins_present(df, encoders=encoders)
    p3.assert_separability_is_paired(df)
    print("\n[p3-tier1] all THIRTEEN table invariants PASS")
    print(f"  rows {df.shape[0]} x cols {df.shape[1]}")
    print(f"  horizons        {sorted(df.delta_days.unique())}")
    print(f"  fold modes      {sorted(df.fold_mode.unique())}")
    print(f"  aggregations    {sorted(df.aggregation.unique())}")
    print(f"  encoder views   {sorted(df.encoder.unique())}")
    print(f"  band composites {sorted(df.band_composite.unique())}")
    print(f"  alpha rules     {sorted(df.alpha_rule.unique())}")
    print(f"  feature bases   {sorted(df.feature_base.unique())}")
    print(f"  estimators      {sorted(df.estimator.unique())}")
    print(f"  screen applied  {bool(df.plausibility_screen.iloc[0])}, "
          f"{int(df.n_rows_dropped_implausible.max())} row(s) dropped at the "
          "worst horizon")

    # --- 6. and again on the CSV, which is what anyone else reads ---------
    back = pd.read_csv(csv)
    assert back.shape == df.shape, (back.shape, df.shape)
    p3.assert_separability_is_paired(back)
    p3.assert_plausibility_screen_declared(back, required=not args.no_screen)
    p3.assert_alpha_rules_present(back)
    p3.assert_shared_base_present(back)
    p3.assert_control_identical_across_views(back)
    p3.assert_climatology_rows_labelled(back)
    print("[p3-tier1] every invariant re-checked ON THE CSV, not on the "
          "in-memory table")

    # --- 7. the results ---------------------------------------------------
    print("\n\n" + "#" * 92)
    print("# THE HEADLINE THIS RUN EXISTS FOR")
    print("#" * 92)
    for base in p3.FEATURE_BASES:
        for rule in p3.ALPHA_RULES:
            p3.print_cir_vs_rgb(df, feature_base=base, alpha_rule=rule)

    print("\n\n" + "#" * 92)
    print("# DOES ANY ENCODER SEPARABLY BEAT THE BAND-MATCHED BASELINE?")
    print("#" * 92)
    for mode in sorted(df.fold_mode.unique()):
        for base in p3.FEATURE_BASES:
            for rule in p3.ALPHA_RULES:
                p3.print_separability(df, fold_mode=mode, feature_base=base,
                                      alpha_rule=rule)

    print("\n\n" + "#" * 92)
    print("# THE FULL TABLES")
    print("#" * 92)
    for mode in sorted(df.fold_mode.unique()):
        for base in p3.FEATURE_BASES:
            for rule in p3.ALPHA_RULES:
                p3.print_headlines(df, fold_mode=mode, feature_base=base,
                                   alpha_rule=rule)
    p3.print_controls(df[df.aggregation == "cube_mean"])
    for agg in [a for a in aggregations if a != "cube_mean"]:
        p3.print_headlines(df, aggregation=agg, fold_mode="cube",
                           feature_base=p3.FEATURE_BASE_SHARED,
                           alpha_rule=p3.ALPHA_RULE_TUNED)
    for mode in sorted(df.fold_mode.unique()):
        p3.print_severity_table(df, fold_mode=mode,
                                feature_base=p3.FEATURE_BASE_SHARED,
                                alpha_rule=p3.ALPHA_RULE_TUNED)

    print_penalty_effect(df)
    print_base_effect(df)

    old_csv = args.old_csv or os.path.join(out_root, "p3_forecast_results.csv")
    old = pd.read_csv(old_csv) if os.path.exists(old_csv) else None
    if old is None:
        print(f"\n[p3-tier1] no published CSV at {old_csv}; the comparison "
              "column is omitted rather than filled with zeros")
    compare_against_published(df, old, args)
    print(f"\n[p3-tier1] DONE in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
