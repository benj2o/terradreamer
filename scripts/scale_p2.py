"""Run the full P2 table on the SCALED cube subset, and diff it against the 20.

WHY THIS IS A SCRIPT AND NOT A NOTEBOOK CELL
---------------------------------------------
The 20-cube P2 run is published: a 600-row CSV under ``data/phase1_6/results/``,
an archived notebook, and a set of claims in log.md that rest on it. The only
thing worth doing at 115 cubes is COMPARING, and a comparison written by hand
each time is a comparison that drifts. This script re-runs the identical probe
against the identical fold machinery and prints every headline beside its
20-cube value, so the delta is the output rather than something a reader has to
reconstruct.

WHY P2 NEEDED A NEW CACHE AND P4 DID NOT
-----------------------------------------
``scripts/scale_p4.py`` could scale by pointing at more cubes, because P4 reads
no embeddings. P2 reads the frozen encoder cache, which existed for exactly the
20 cubes in ``data/raw``. Phase 1.7 built the 115-cube cache under
``data/scaled_<tile>/{embeddings,masks}``; this script consumes it. Nothing here
runs a network, and ``data/raw`` and ``data/phase1_2`` are never written to.

WHAT IS CHECKED BEFORE ANY NUMBER IS PRODUCED
-----------------------------------------------
1. The scaled cache is COMPLETE -- 115 x 5, no holes. A cube missing one encoder
   turns a per-encoder comparison into a comparison over different cubes.
2. The scaled cache REPRODUCES the published one on the 20 cubes they share
   (``encoders.pipeline.assert_caches_agree``). Without this the two tables are
   not two measurements of one quantity, and diffing them is meaningless.
3. The gap axes still disagree on the real pairs
   (``p2_deltas.assert_gap_axes_disagree``), on 6x the data.

    python -m scripts.scale_p2 --tile 32UNU
"""

from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np


def _fmt(x, w=7, p=3):
    return f"{x:+{w}.{p}f}" if np.isfinite(x) else " " * (w - 3) + "nan"


def compare_headlines(new, old, primary: dict, delta_primary: dict) -> None:
    """Print every headline at both sample sizes, side by side.

    ``old`` may be None -- the 20-cube CSV is a build artefact and a fresh
    checkout will not have it. The scaled numbers are still printed; only the
    comparison column is dropped, and it says so rather than printing zeros.
    """
    import pandas as pd   # noqa: F401  (frames are passed in)

    def cell(df, **q):
        if df is None:
            return None
        sub = df
        for k, v in q.items():
            sub = sub[sub[k] == v]
        return None if len(sub) != 1 else sub.iloc[0]

    n_new = int(new.n_cubes.iloc[0])
    n_old = int(old.n_cubes.iloc[0]) if old is not None else 0
    head = f"{n_new} cubes" + (f"   vs {n_old} cubes" if old is not None else "")

    # ---- Gate K2 ----------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"GATE K2 -- reconstruction floor at "
          f"{primary['aggregation']}/{primary['feature_level']}/"
          f"{primary['fold_mode']}   ({head})")
    print("=" * 78)
    print(f"  {'encoder':<24} {'R2':>8} {'CI width':>9} | {'was':>8} {'width':>8} "
          f"| {'paired CI on vs-raw':>22}  separable?")
    sel = new[(new.part == "A_reconstruction")
              & (new.aggregation == primary["aggregation"])
              & (new.feature_level == primary["feature_level"])
              & (new.fold_mode == primary["fold_mode"])
              & (new.feature_set == "embedding")].sort_values("score_mean",
                                                              ascending=False)
    for r in sel.itertuples():
        o = cell(old, part="A_reconstruction", aggregation=primary["aggregation"],
                 feature_level=primary["feature_level"],
                 fold_mode=primary["fold_mode"], feature_set="embedding",
                 encoder=r.encoder)
        w_new = r.score_ci_hi - r.score_ci_lo
        was = f"{_fmt(o.score_mean)} {o.score_ci_hi - o.score_ci_lo:8.3f}" if o is not None else " " * 17
        print(f"  {r.encoder:<24} {_fmt(r.score_mean, 8)} {w_new:9.3f} | {was} "
              f"| [{_fmt(r.k2_margin_ci_lo)},{_fmt(r.k2_margin_ci_hi)}]  "
              f"{'SEPARABLE' if r.k2_separable else 'not separable'}")

    # ---- the delta probe --------------------------------------------------
    p = delta_primary
    for dt in ("sign", "magnitude"):
        print("\n" + "=" * 78)
        print(f"DELTA PROBE -- {p['aggregation']} / {dt} / {p['feature_level']} "
              f"/ {p['readout']} / {p['fold_mode']}   ({head})")
        print("=" * 78)
        sub = new[(new.part == "B_delta") & (new.delta_target == dt)
                  & (new.aggregation == p["aggregation"])
                  & (new.feature_level == p["feature_level"])
                  & (new.readout == p["readout"])
                  & (new.fold_mode == p["fold_mode"])
                  & (new.feature_set.isin(("embedding", "raw_rgb_only"))
                     | new.is_control)].sort_values("score_mean", ascending=False)
        print(f"  {'row':<34} {'rho':>8} {'CI':>18} {'margin':>8} | {'was rho':>8} "
              f"{'was margin':>10}")
        for r in sub.itertuples():
            o = cell(old, part="B_delta", delta_target=dt,
                     aggregation=p["aggregation"], feature_level=p["feature_level"],
                     readout=p["readout"], fold_mode=p["fold_mode"],
                     encoder=r.encoder, feature_set=r.feature_set)
            name = (f"{r.encoder}/{r.feature_set}"
                    if r.feature_set not in ("embedding",) else r.encoder)
            tag = ("  <- CONTROL" if r.is_control
                   else ("  [si_comparable=False]" if not r.si_comparable else ""))
            was = (f"{_fmt(o.score_mean, 8)} {_fmt(o.margin_over_control, 10)}"
                   if o is not None else " " * 19)
            print(f"  {name:<34} {_fmt(r.score_mean, 8)} "
                  f"[{_fmt(r.score_ci_lo)},{_fmt(r.score_ci_hi)}] "
                  f"{_fmt(r.margin_over_control, 8)} | {was}{tag}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", default="32UNU")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=0, help="0 = cpu_count - 1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--old-csv", default=None,
                    help="the 20-cube CSV to diff against; default is the "
                         "phase1_6 results path")
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="tolerance for the cache reproduction check")
    ap.add_argument("--skip-reproduction-check", action="store_true",
                    help="NOT recommended. Without it the two tables are not "
                         "known to measure the same quantity.")
    args = ap.parse_args()

    import pandas as pd

    from data.loader import load_cube
    from encoders import TIER_A
    from encoders.manifest import (assert_strata_present, assert_weather_join,
                                   build_manifest)
    from encoders.pipeline import (assert_caches_agree,
                                   assert_embeddings_complete, audit_embeddings)
    from probes import p2_deltas as p2

    out_root = args.out or os.path.join("data", f"scaled_{args.tile}")
    cube_dir = os.path.join(out_root, "raw")
    emb_dir = os.path.join(out_root, "embeddings")
    mask_dir = os.path.join(out_root, "masks")
    for d, what in ((cube_dir, "cubes"), (emb_dir, "embeddings"), (mask_dir, "masks")):
        assert os.path.isdir(d), (
            f"no {what} at {d}. Phase 1.7 "
            "(notebooks/phase1_7_scaled_encoding.ipynb) builds this cache; it "
            "is not produced by any probe."
        )

    paths = sorted(glob.glob(os.path.join(cube_dir, "*.nc")))
    cube_ids = {os.path.basename(p) for p in paths}
    print(f"[scale-p2] {len(paths)} cubes at {cube_dir}")
    print(f"[scale-p2] {len(glob.glob(os.path.join(emb_dir, '*.npz')))} embeddings, "
          f"{len(glob.glob(os.path.join(mask_dir, '*.npz')))} masks")

    # --- 1. the cache is whole -------------------------------------------
    audit = audit_embeddings(emb_dir, cube_ids=cube_ids, verbose=True)
    assert_embeddings_complete(audit, cube_ids, TIER_A)
    missing = [c for c in sorted(cube_ids)
               if not os.path.exists(os.path.join(
                   mask_dir, f"{os.path.splitext(c)[0]}__masks.npz"))]
    assert not missing, (
        f"{len(missing)} cube(s) have no cached per-pixel mask, e.g. "
        f"{missing[:3]}. Common-masking is mandatory in P2 and cannot be "
        "approximated from clear_frac."
    )

    # --- 2. it reproduces the published cache on the shared cubes ---------
    old_emb = os.path.join("data", "phase1_2", "embeddings")
    shared = sorted(cube_ids & {os.path.basename(p)
                                for p in glob.glob(os.path.join("data", "raw", "*.nc"))})
    if args.skip_reproduction_check:
        print("[scale-p2] WARNING: reproduction check SKIPPED. The scaled table "
              "is not known to measure the same quantity as the published one.")
    elif shared and os.path.isdir(old_emb):
        print(f"\n[scale-p2] reproduction check on {len(shared)} shared cubes")
        agree = assert_caches_agree(old_emb, emb_dir, shared, TIER_A,
                                    tol=args.tol, verbose=True)
        print(f"[scale-p2] the scaled cache REPRODUCES the published one "
              f"(worst pooled diff {agree['max_abs_pooled']:.3g})")
    else:
        print("[scale-p2] no shared cubes or no 20-cube cache on disk; the "
              "reproduction check could not run and is reported as NOT DONE.")

    # --- 3. the manifest, and the axis guard on 6x the data ---------------
    t0 = time.time()
    samples = [load_cube(p, verbose=False) for p in paths]
    manifest = build_manifest(samples, verbose=False)
    assert_strata_present(manifest)
    assert_weather_join(manifest, cube_dir, verbose=False)
    print(f"\n[scale-p2] manifest {manifest.shape}, "
          f"{manifest.cube_id.nunique()} cubes, "
          f"years {sorted(manifest.year.unique().tolist())}, "
          f"{time.time() - t0:.0f}s")
    pairs = p2.pair_index(manifest, verbose=True)
    p2.assert_gap_axes_disagree(pairs, verbose=True)

    # --- 4. the run -------------------------------------------------------
    t0 = time.time()
    n_jobs = args.n_jobs or max(1, (os.cpu_count() or 2) - 1)
    df = p2.run_p2(manifest, cube_dir, k=args.k, emb_dir=emb_dir,
                   mask_dir=mask_dir, n_jobs=n_jobs,
                   log_path=os.path.join(out_root, "p2_scaled_run.log"),
                   verbose=False)
    df = p2.add_margins(df, verbose=False)
    df = p2.add_k2_verdicts(df, verbose=False)
    print(f"[scale-p2] run_p2: {len(df)} rows in {(time.time() - t0) / 60:.1f} min "
          f"on {n_jobs} workers (fold-level; changes wall-clock, never a number)")

    ranking = p2.structural_hypothesis(df, verbose=False)
    p2.assert_results_complete(df)
    p2.assert_k2_verdict_recorded(df)
    p2.assert_control_identical_across_views(df)
    p2.assert_degenerate_control_present(df)
    p2.assert_mi_flagged_and_excluded(df, ranking)
    p2.assert_effective_n_counts_cubes(df)
    print("[scale-p2] all six table invariants PASS")

    csv = p2.results_path("p2_scaled_results.csv", root=out_root)
    df.to_csv(csv, index=False)
    back = pd.read_csv(csv)
    assert back.shape == df.shape
    print(f"[scale-p2] wrote {csv} ({df.shape[0]} rows x {df.shape[1]} cols)")

    # --- 5. the comparison this run exists to make ------------------------
    old_csv = args.old_csv or p2.results_path()
    old = pd.read_csv(old_csv) if os.path.exists(old_csv) else None
    if old is None:
        print(f"\n[scale-p2] no 20-cube CSV at {old_csv}; printing scaled "
              "numbers alone. Re-run the Phase 1.6 notebook to restore it.")
    compare_headlines(df, old, p2.K2_PRIMARY, p2.STRUCTURAL_PRIMARY)

    # --- 6. the three questions the 20-cube run could not answer ----------
    print("\n" + "=" * 78)
    print("WHAT THE SCALE-UP WAS FOR")
    print("=" * 78)

    sel = df[(df.part == "A_reconstruction")
             & (df.aggregation == p2.K2_PRIMARY["aggregation"])
             & (df.feature_level == p2.K2_PRIMARY["feature_level"])
             & (df.fold_mode == p2.K2_PRIMARY["fold_mode"])
             & (df.feature_set == "embedding")
             & (df.encoder != "raw_features")]
    n_sep = int(sel.k2_separable.sum())
    print(f"1. K2 separability: {n_sep}/{len(sel)} encoders now separable from "
          f"the baseline\n   (20-cube run: 1/4, and that one was the "
          "multi-image control)")

    p = p2.STRUCTURAL_PRIMARY
    mag = df[(df.part == "B_delta") & (df.delta_target == "magnitude")
             & (df.aggregation == p["aggregation"])
             & (df.feature_level == p["feature_level"])
             & (df.readout == p["readout"])
             & (df.fold_mode == p["fold_mode"]) & (~df.is_control)]
    beat = mag[mag.margin_over_control > 0]
    ctrl = df[(df.part == "B_delta") & (df.delta_target == "magnitude")
              & (df.aggregation == p["aggregation"])
              & (df.feature_level == p["feature_level"])
              & (df.readout == p["readout"])
              & (df.fold_mode == p["fold_mode"]) & (df.is_control)]
    print(f"\n2. MAGNITUDE vs the gap-length control: {len(beat)}/{len(mag)} rows "
          "now beat it")
    if len(ctrl) == 1:
        print(f"   control {_fmt(ctrl.iloc[0].score_mean)} "
              f"[{_fmt(ctrl.iloc[0].score_ci_lo)},{_fmt(ctrl.iloc[0].score_ci_hi)}]"
              "   (20-cube run: +0.209, and it beat EVERY encoder)")

    print(f"\n3. Structural hypothesis: order stable across fold modes = "
          f"{ranking['order_stable_across_fold_modes']}, "
          f"supported = {ranking['supported']}")
    for mode, order in ranking["order_by_fold_mode"].items():
        print(f"   {mode:<14} {' > '.join(order)}")
    print("   (20-cube run: NOT determinable -- dinov2 and satlas SI swapped "
          "between cube and loco)")


if __name__ == "__main__":
    main()
