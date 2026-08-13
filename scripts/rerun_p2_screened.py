"""Re-run scaled P2 with P4's shared plausibility screen, without overwriting P2.

This is the P2 analogue of ``scripts/rerun_p3_tier1``: scientific logic stays
in the canonical probe, while this thin runner validates frozen inputs, opts
into the corrected row set, writes a distinctly named artefact, asserts it
both in memory and after CSV round-trip, and diffs it against the published
115-cube table.

    .venv/bin/python -m scripts.rerun_p2_screened --tile 32UNU --n-jobs 7
"""

from __future__ import annotations

import argparse
import glob
import os
import time


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tile", default="32UNU")
    ap.add_argument("--out", default=None,
                    help="cache root; default data/scaled_<tile>")
    ap.add_argument("--old-csv", default=None,
                    help="unscreened scaled CSV; default <out>/p2_scaled_results.csv")
    ap.add_argument("--csv-name", default="p2_screened_results.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=0,
                    help="0 = cpu_count - 1")
    ap.add_argument("--max-cubes", type=int, default=0,
                    help="SMOKE TEST ONLY: strided cube subset")
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="scaled-vs-published cache reproduction tolerance")
    ap.add_argument("--skip-reproduction-check", action="store_true",
                    help="not recommended; records that cache equivalence was not checked")
    ap.add_argument("--overwrite", action="store_true",
                    help="explicitly replace an existing screened CSV")
    args = ap.parse_args()

    import pandas as pd

    from data.loader import load_cube
    from encoders import TIER_A
    from encoders.manifest import (assert_strata_present, assert_weather_join,
                                   build_manifest)
    from encoders.pipeline import (assert_caches_agree,
                                   assert_embeddings_complete,
                                   audit_embeddings)
    from probes import p2_deltas as p2
    from scripts.scale_p2 import compare_headlines

    out_root = args.out or os.path.join("data", f"scaled_{args.tile}")
    cube_dir = os.path.join(out_root, "raw")
    emb_dir = os.path.join(out_root, "embeddings")
    mask_dir = os.path.join(out_root, "masks")
    for path, label in ((cube_dir, "cubes"), (emb_dir, "embeddings"),
                        (mask_dir, "masks")):
        assert os.path.isdir(path), f"no {label} at {path}"

    csv_name = args.csv_name
    if args.max_cubes and csv_name == "p2_screened_results.csv":
        csv_name = "p2_screened_smoke_results.csv"
    csv = p2.results_path(csv_name, root=out_root)
    assert args.overwrite or not os.path.exists(csv), (
        f"{csv} already exists. Published/reviewed artefacts are not replaced "
        "implicitly; pass --overwrite only for an intentional rerun."
    )

    paths = sorted(glob.glob(os.path.join(cube_dir, "*.nc")))
    if args.max_cubes:
        stride = max(1, len(paths) // args.max_cubes)
        paths = paths[::stride][:args.max_cubes]
        print(f"[p2-screened] *** SMOKE TEST: {len(paths)} strided cubes; "
              "these numbers are not results ***")
    cube_ids = {os.path.basename(p) for p in paths}
    assert cube_ids, f"no .nc cubes at {cube_dir}"
    print(f"[p2-screened] {len(cube_ids)} cubes at {cube_dir}")

    audit = audit_embeddings(emb_dir, cube_ids=cube_ids, verbose=True)
    assert_embeddings_complete(audit, cube_ids, TIER_A)
    missing_masks = [
        cube for cube in sorted(cube_ids)
        if not os.path.exists(os.path.join(
            mask_dir, f"{os.path.splitext(cube)[0]}__masks.npz"))
    ]
    assert not missing_masks, (
        f"{len(missing_masks)} cubes lack common-mask caches, e.g. "
        f"{missing_masks[:3]}"
    )

    old_emb = os.path.join("data", "phase1_2", "embeddings")
    published_cubes = {
        os.path.basename(p)
        for p in glob.glob(os.path.join("data", "raw", "*.nc"))
    }
    shared = sorted(cube_ids & published_cubes)
    if args.skip_reproduction_check:
        print("[p2-screened] WARNING: cache reproduction check SKIPPED")
    elif shared and os.path.isdir(old_emb):
        agree = assert_caches_agree(old_emb, emb_dir, shared, TIER_A,
                                    tol=args.tol, verbose=True)
        print("[p2-screened] scaled cache reproduces the published cache "
              f"(worst pooled difference {agree['max_abs_pooled']:.3g})")
    else:
        print("[p2-screened] cache reproduction check NOT RUN: no shared "
              "published cubes/cache")

    t0 = time.time()
    samples = [load_cube(path, verbose=False) for path in paths]
    manifest = build_manifest(samples, verbose=False)
    assert_strata_present(manifest)
    join = assert_weather_join(manifest, cube_dir, verbose=False)
    assert max(join["max_abs_diff"].values()) == 0.0, join
    del samples
    print(f"[p2-screened] manifest {manifest.shape}, "
          f"{manifest.cube_id.nunique()} cubes, built in {time.time() - t0:.0f}s")

    n_jobs = args.n_jobs or max(1, (os.cpu_count() or 2) - 1)
    t0 = time.time()
    df = p2.run_p2(
        manifest, cube_dir, k=args.k, emb_dir=emb_dir, mask_dir=mask_dir,
        n_jobs=n_jobs, plausibility_screen=True,
        log_path=os.path.join(out_root, "p2_screened_run.log"),
        verbose=False,
    )
    df = p2.add_margins(df, verbose=False)
    df = p2.add_k2_verdicts(df, verbose=False)
    print(f"[p2-screened] run_p2: {df.shape[0]} rows x {df.shape[1]} cols "
          f"in {(time.time() - t0) / 60:.1f} min on {n_jobs} workers")

    # Match Tier-1: preserve the expensive table before assertions, then trust
    # it only after the in-memory and round-trip checks below.
    df.to_csv(csv, index=False)
    print(f"[p2-screened] wrote {csv}; invariants run next")

    ranking = p2.structural_hypothesis(df, verbose=False)
    p2.assert_results_complete(df)
    p2.assert_k2_verdict_recorded(df)
    p2.assert_control_identical_across_views(df)
    p2.assert_degenerate_control_present(df)
    p2.assert_mi_flagged_and_excluded(df, ranking)
    p2.assert_effective_n_counts_cubes(df)
    p2.assert_plausibility_screen_declared(df, required=True)

    back = pd.read_csv(csv)
    assert back.shape == df.shape, (back.shape, df.shape)
    p2.assert_results_complete(back)
    p2.assert_k2_verdict_recorded(back)
    p2.assert_control_identical_across_views(back)
    p2.assert_plausibility_screen_declared(back, required=True)
    print("[p2-screened] all invariants pass in memory and on the CSV")

    print(f"[p2-screened] screen geometry: "
          f"{int(back.n_implausible_frames.iloc[0])} implausible frames; "
          f"{int(back.n_pairs_dropped_implausible.iloc[0])} pairs dropped; "
          f"{int(back.n_pairs.iloc[0])} pairs scored")

    old_csv = args.old_csv or os.path.join(out_root, "p2_scaled_results.csv")
    old = pd.read_csv(old_csv) if os.path.exists(old_csv) else None
    if old is None:
        print(f"[p2-screened] no unscreened comparison CSV at {old_csv}")
    compare_headlines(df, old, p2.K2_PRIMARY, p2.STRUCTURAL_PRIMARY)


if __name__ == "__main__":
    main()
