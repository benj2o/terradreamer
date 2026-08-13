"""Re-run scaled P4 with its shared plausibility screen, preserving old tables.

The scientific path remains ``probes.p4_ceiling``. This Tier-1-style runner
validates inputs, explicitly opts into target-row screening, writes a new CSV,
checks every invariant again after CSV round-trip, and prints the screened
headline beside the published unscreened 115-cube result.

    .venv/bin/python -m scripts.rerun_p4_screened --tile 32UNU --n-jobs 7
"""

from __future__ import annotations

import argparse
import glob
import os
import time


def compare_headlines(new, old) -> None:
    """Print the predeclared cell-mean/HGB proxy cells and both controls."""
    modes = ("cube", "loco")
    kinds = ("weather", "observation", "doy")
    print("\n" + "=" * 92)
    print("P4 SCREENED VS UNSCREENED -- cell_mean / HGB / weather_full8 / Stage A")
    print("=" * 92)
    print(f"  {'mode/kind':<28} {'screened R2':>12} {'obs margin':>11} "
          f"{'DOY margin':>11} | {'unscreened R2':>13} {'obs margin':>11} "
          f"{'DOY margin':>11}")
    for mode in modes:
        for kind in kinds:
            query = ((new.stage == "A") & (new.target == "cell_mean")
                     & (new.fold_mode == mode) & (new.estimator == "hgb")
                     & (new.feature_set == "weather_full8")
                     & (new.model_kind == kind))
            current = new[query]
            assert len(current) == 1, (
                f"screened headline is not unique for {mode}/{kind}: "
                f"{len(current)} rows"
            )
            r = current.iloc[0]
            prior = None
            if old is not None:
                q_old = ((old.stage == "A") & (old.target == "cell_mean")
                         & (old.fold_mode == mode) & (old.estimator == "hgb")
                         & (old.feature_set == "weather_full8")
                         & (old.model_kind == kind))
                match = old[q_old]
                prior = match.iloc[0] if len(match) == 1 else None
            old_text = (
                f"{prior.r2_vs_climatology_mean:+13.3f} "
                f"{prior.margin_over_control:+11.3f} "
                f"{prior.margin_over_doy:+11.3f}"
                if prior is not None else f"{'n/a':>37}"
            )
            print(f"  {mode + '/' + kind:<28} "
                  f"{r.r2_vs_climatology_mean:+12.3f} "
                  f"{r.margin_over_control:+11.3f} "
                  f"{r.margin_over_doy:+11.3f} | {old_text}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tile", default="32UNU")
    ap.add_argument("--out", default=None,
                    help="cache root; default data/scaled_<tile>")
    ap.add_argument("--old-csv", default=None,
                    help="unscreened scaled CSV; default <out>/p4_scaled_results.csv")
    ap.add_argument("--csv-name", default="p4_screened_results.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=0,
                    help="0 = cpu_count - 1")
    ap.add_argument("--max-cubes", type=int, default=0,
                    help="SMOKE TEST ONLY: strided cube subset")
    ap.add_argument("--download", action="store_true",
                    help="explicitly download missing cubes before the run")
    ap.add_argument("--split", default="train",
                    help="download split when --download is used")
    ap.add_argument("--n", type=int, default=115,
                    help="requested non-overlapping cubes when --download is used")
    ap.add_argument("--overwrite", action="store_true",
                    help="explicitly replace an existing screened CSV")
    args = ap.parse_args()

    import pandas as pd

    from data.loader import load_cube
    from encoders.manifest import (assert_strata_present, assert_weather_join,
                                   build_manifest)
    from probes import p4_ceiling as p4

    out_root = args.out or os.path.join("data", f"scaled_{args.tile}")
    cube_dir = os.path.join(out_root, "raw")
    csv_name = args.csv_name
    if args.max_cubes and csv_name == "p4_screened_results.csv":
        csv_name = "p4_screened_smoke_results.csv"
    csv = p4.results_path(csv_name, root=out_root)
    assert args.overwrite or not os.path.exists(csv), (
        f"{csv} already exists. Published/reviewed artefacts are not replaced "
        "implicitly; pass --overwrite only for an intentional rerun."
    )

    if args.download:
        from scripts.scale_p4 import download
        download(args.tile, args.split, args.n, cube_dir)
    assert os.path.isdir(cube_dir), f"no cubes at {cube_dir}"
    paths = sorted(glob.glob(os.path.join(cube_dir, "*.nc")))
    if args.max_cubes:
        stride = max(1, len(paths) // args.max_cubes)
        paths = paths[::stride][:args.max_cubes]
        print(f"[p4-screened] *** SMOKE TEST: {len(paths)} strided cubes; "
              "these numbers are not results ***")
    assert paths, f"no .nc cubes at {cube_dir}"
    print(f"[p4-screened] {len(paths)} cubes at {cube_dir}")

    t0 = time.time()
    samples = [load_cube(path, verbose=False) for path in paths]
    manifest = build_manifest(samples, verbose=False)
    assert_strata_present(manifest)
    join = assert_weather_join(manifest, cube_dir, verbose=False)
    assert max(join["max_abs_diff"].values()) == 0.0, join
    del samples
    print(f"[p4-screened] manifest {manifest.shape}, "
          f"{manifest.cube_id.nunique()} cubes, built in {time.time() - t0:.0f}s")

    n_jobs = args.n_jobs or max(1, (os.cpu_count() or 2) - 1)
    t0 = time.time()
    df, data, info = p4.run_p4(
        manifest, cube_dir, k=args.k, n_jobs=n_jobs, verbose=False,
        plausibility_screen=True,
    )
    print(f"[p4-screened] run_p4: {df.shape[0]} rows x {df.shape[1]} cols "
          f"in {(time.time() - t0) / 60:.1f} min on {n_jobs} workers")

    # Match Tier-1: retain evidence from an expensive run before assertions;
    # the artefact is accepted only after both checks below.
    df.to_csv(csv, index=False)
    print(f"[p4-screened] wrote {csv}; invariants run next")

    p4.assert_results_complete(df)
    p4.assert_stage_b_ran_or_deferred(df, info)
    p4.assert_plausibility_screen_declared(df, required=True)
    back = pd.read_csv(csv)
    assert back.shape == df.shape, (back.shape, df.shape)
    p4.assert_results_complete(back)
    p4.assert_stage_b_ran_or_deferred(back, info)
    p4.assert_plausibility_screen_declared(back, required=True)
    print("[p4-screened] all invariants pass in memory and on the CSV")

    print(f"[p4-screened] screen geometry: "
          f"{int(back.n_implausible_frames.iloc[0])} implausible frames; "
          + ", ".join(
              f"{target} -{int(back.loc[back.target == target, 'n_rows_dropped_implausible'].iloc[0])}"
              for target in p4.TARGETS))

    old_csv = args.old_csv or os.path.join(out_root, "p4_scaled_results.csv")
    old = pd.read_csv(old_csv) if os.path.exists(old_csv) else None
    if old is None:
        print(f"[p4-screened] no unscreened comparison CSV at {old_csv}")
    compare_headlines(df, old)


if __name__ == "__main__":
    main()
