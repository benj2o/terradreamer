"""Run the full P4 table on a LARGER cube subset than data/raw holds.

WHY THIS DOES NOT WRITE TO data/raw
------------------------------------
``data/raw`` is shared across phases and Phase 1.2's embedding cache is keyed to
exactly the 20 cubes in it. Dropping 115 cubes there would make
``build_manifest`` return 115 cubes' worth of rows while the cache still held
20 x 5 ``.npz``, and ``assert_embeddings_complete`` would fail every phase that
reads embeddings -- P1 today, P2 tomorrow. P4 is the one probe that reads no
embeddings at all, so it is also the only one that can scale its cube count
independently, and it takes ``cube_dir`` as a parameter precisely so it can.

So the scaled subset lives under ``data/scaled_<tile>/raw`` and nothing else in
the project sees it. ``data/raw`` stays the 20-cube set the caches match.

WHAT THIS IS FOR
----------------
P4's Stage A proxy ceiling is CUBE-limited, and cubes are cheap. Measured on
30TVN: 87 cubes give a fold-clustered CI width of 0.105 against 20 cubes' 0.450.
On 32UNU the 20-cube run put the fold-clustered interval across zero for all 54
weather rows -- not a null result, an underpowered one. This is the cheap fix.

It is NOT a fix for H1. That is year-limited (``crossed`` mode clamps its fold
count to the number of years, and GreenEarthNet has four), and no cube count
changes it. See docs/DECISIONS.md, 2026-08-10.

    python -m scripts.scale_p4 --tile 32UNU --n 115
"""

from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np


def download(tile: str, split: str, n: int, out_dir: str,
             verbose: bool = True) -> list:
    """Non-overlapping cubes for one tile. Skips what is already on disk.

    Calls the library functions rather than ``data.download_greenearthnet``'s
    CLI, whose ``n <= 20`` assertion is a deliberate toy-load guard on the
    interactive path and is left in place.
    """
    from data.download_greenearthnet import (list_cubes, s3fs_client,
                                             select_non_overlapping)

    os.makedirs(out_dir, exist_ok=True)
    s3 = s3fs_client()
    keys = list_cubes(tile=tile, split=split, s3=s3)
    chosen = select_non_overlapping(keys, n=n)
    if verbose:
        print(f"[dl] {tile}/{split}: {len(keys)} listed, {len(chosen)} "
              f"non-overlapping selected (asked {n})")
        if len(chosen) < n:
            print(f"[dl] NOTE: capped at {len(chosen)} by the 64 px "
                  "non-overlap rule, not by the listing")

    t0, paths, fetched = time.time(), [], 0
    for key in chosen:
        path = os.path.join(out_dir, os.path.basename(key))
        if not os.path.exists(path):
            tmp = path + ".partial"
            s3.download(key, tmp)
            os.replace(tmp, path)
            fetched += 1
        paths.append(path)
    if verbose:
        mb = sum(os.path.getsize(p) for p in paths) / 1e6
        print(f"[dl] {len(paths)} cubes on disk ({fetched} new), {mb:.0f} MB, "
              f"{time.time() - t0:.0f}s -> {out_dir}")
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", default="32UNU")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=115)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=0, help="0 = cpu_count - 1")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import pandas as pd

    from data.loader import load_cube
    from encoders.manifest import (assert_strata_present, assert_weather_join,
                                   build_manifest)
    from probes import p4_ceiling as p4

    out_root = args.out or os.path.join("data", f"scaled_{args.tile}")
    cube_dir = os.path.join(out_root, "raw")
    n_jobs = args.n_jobs or max(1, (os.cpu_count() or 2) - 1)

    download(args.tile, args.split, args.n, cube_dir)

    t0 = time.time()
    paths = sorted(glob.glob(os.path.join(cube_dir, "*.nc")))
    samples = [load_cube(p, verbose=False) for p in paths]
    manifest = build_manifest(samples, verbose=False)
    assert_strata_present(manifest)
    assert_weather_join(manifest, cube_dir)
    print(f"[build] manifest {manifest.shape}, {manifest.cube_id.nunique()} "
          f"cubes, years {sorted(manifest.year.unique().tolist())}, "
          f"{time.time() - t0:.0f}s")

    t0 = time.time()
    df, data, info = p4.run_p4(manifest, cube_dir, k=args.k, n_jobs=n_jobs,
                               verbose=False)
    print(f"[run] run_p4: {len(df)} rows in {(time.time() - t0) / 60:.1f} min "
          f"on {n_jobs} workers")

    p4.assert_results_complete(df)
    p4.assert_stage_b_ran_or_deferred(df, info)

    csv = os.path.join(out_root, "p4_scaled_results.csv")
    df.to_csv(csv, index=False)
    back = pd.read_csv(csv)
    assert back.shape == df.shape
    print(f"[run] wrote {csv} ({df.shape[0]} rows x {df.shape[1]} cols)")

    # --- the comparison this run exists to make ---------------------------
    print("\n" + "=" * 78)
    print(f"P4 Stage A at {manifest.cube_id.nunique()} cubes -- "
          "cube_mean / cube / linear / weather_full8")
    print("=" * 78)
    sub = df[(df.target == "cube_mean") & (df.fold_mode == "cube")
             & (df.estimator == "linear") & (df.feature_set == "weather_full8")]
    for _, r in sub.iterrows():
        print(f"  {r.model_kind:<25} r2_vs_clim {r.r2_vs_climatology_mean:+.3f}  "
              f"[{r.r2_vs_climatology_ci_lo:+.3f}, {r.r2_vs_climatology_ci_hi:+.3f}]"
              f"  width {r.r2_vs_climatology_ci_hi - r.r2_vs_climatology_ci_lo:.3f}"
              f"  margin {r.margin_over_control:+.3f}")

    w = df[df.model_kind == "weather"]
    crosses = ((w.r2_vs_climatology_ci_lo <= 0)
               & (w.r2_vs_climatology_ci_hi >= 0)).sum()
    print(f"\nweather rows whose interval still includes zero: "
          f"{int(crosses)}/{len(w)}   (20-cube run: 54/54)")
    print(f"weather rows at or below the observation control: "
          f"{int((w.margin_over_control <= 0).sum())}/{len(w)}   "
          "(20-cube run: 33/54)")

    lin_doy = df[(df.model_kind == "doy") & (df.estimator == "linear")]
    print(f"\nlinear day-of-year sanity control, worst |r2_vs_clim|: "
          f"{lin_doy.r2_vs_climatology_mean.abs().max():.3f}   "
          "(20-cube run: 0.019)")
    perm = df[df.model_kind == "permutation"]
    print(f"permutation control, max r2_vs_clim: "
          f"{perm.r2_vs_climatology_mean.max():+.3f}   "
          "(must stay <= 0: the pipeline never invents skill)")


if __name__ == "__main__":
    main()
