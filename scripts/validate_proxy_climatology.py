"""Does P4's Stage A PROXY climatology track the real leave-year-out one?

THE QUESTION THIS ANSWERS, AND WHY IT IS WORTH A SEPARATE SCRIPT
----------------------------------------------------------------
Tile 32UNU -- the subset the whole project is framed on -- has NO seasonal-split
coverage in GreenEarthNet. Verified against the bucket: the seasonal split
covers 15 tiles and 32UNU is not among them, nor is any Bavaria-area tile
(32TPT, 33TUN). So the real leave-target-year-out climatology, and therefore
H1, is **not computable on 32UNU at all** -- not "pending a download", not ever,
on this dataset.

That leaves Stage A's within-season day-of-year proxy as the only ceiling
32UNU can support, and raises the obvious question a reviewer will ask first:
**is the proxy any good?** This script answers it where both ARE computable --
on a seasonal-covered tile -- by running Stage A and Stage B over the same
cubes and comparing.

It is a VALIDATION EXPERIMENT, not a phase. It writes under
``data/validation_<tile>/`` and touches no phase directory.

WHAT IS AND IS NOT COMPARABLE HERE, STATED UP FRONT
----------------------------------------------------
Stage A and Stage B do not differ only in their climatology. They also differ
in FOLD STRUCTURE, necessarily: Stage A uses ``cube`` (year is not a grouping
axis when there is one year per cube), Stage B uses ``crossed`` (cube AND year
held out jointly, the only mode that agrees with a leave-year-out climatology).
So a difference between them is a difference between two PROTOCOLS, not between
two climatologies with everything else held fixed. Both are reported with their
fold-clustered intervals and the reader is told which is which; neither number
is presented as the other's error bar.

The targets differ too, and this is the more interesting half. Stage A's is
``spatial_mean(NDVI) - curve(day_of_year)`` -- a curve fitted to the SPATIAL
MEAN. Stage B's is ``spatial_mean(NDVI_pixel - climatology_pixel[doy])`` -- a
PER-PIXEL climatology, then averaged. Those are different quantities even
before any model is fitted, which is precisely why "does the proxy track the
real thing" is an empirical question rather than an algebraic one.

E-OBS COMPLETENESS
------------------
Outside 32UNU the weather stack can have real gaps, and they are not random:
on 30TVN ``eobs_fg`` is absent from 20 of 25 cubes and ``eobs_qq`` is missing
the same trailing 13 days in every one. Nothing is filled -- the run uses
``weather_finite6``, the fully-finite intersection, and says so in the CSV. The
consequence is that these numbers are NOT directly comparable with 32UNU's
8-variable ones, and this script prints that warning rather than leaving the
comparison to be made by eye.

    python -m scripts.validate_proxy_climatology --tile 30TVN --n 60
"""

from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np

# Tiles GreenEarthNet's seasonal split actually covers, read from the bucket on
# 2026-08-10. No Bavaria-area tile is among them.
SEASONAL_TILES = ("29SQC", "29TPF", "29UMV", "30TVK", "30TVN", "31TCF", "31TFK",
                  "31UCS", "31UEQ", "31UGQ", "31UGU", "32VNM", "32VPN",
                  "33TWN", "33VXK")

# Variables dropped for tiles with incomplete E-OBS. Measured on 30TVN.
INCOMPLETE_VARS = ("eobs_fg", "eobs_qq")
FEATURE_SET = "weather_finite6"


def download(tile: str, n: int, out_dir: str, verbose: bool = True) -> list:
    """Non-overlapping seasonal cubes for one tile. Skips what is already there."""
    from data.download_greenearthnet import (list_cubes, s3fs_client,
                                             select_non_overlapping)

    assert tile in SEASONAL_TILES, (
        f"{tile} is not in the seasonal split ({len(SEASONAL_TILES)} tiles: "
        f"{SEASONAL_TILES}). Stage B needs multi-year cubes; a tile without "
        "seasonal coverage cannot provide them."
    )
    os.makedirs(out_dir, exist_ok=True)
    s3 = s3fs_client()
    keys = list_cubes(tile=tile, split="seasonal", s3=s3)
    chosen = select_non_overlapping(keys, n=n)
    if verbose:
        print(f"[dl] {len(keys)} cubes in {tile}/seasonal, "
              f"{len(chosen)} non-overlapping selected (asked for {n})")

    t0, paths, fetched = time.time(), [], 0
    for key in chosen:
        path = os.path.join(out_dir, os.path.basename(key))
        if not os.path.exists(path):
            tmp = path + ".partial"
            s3.download(key, tmp)
            os.replace(tmp, path)
            fetched += 1
        paths.append(path)
    mb = sum(os.path.getsize(p) for p in paths) / 1e6
    if verbose:
        print(f"[dl] {len(paths)} cubes on disk ({fetched} newly fetched), "
              f"{mb:.0f} MB, {time.time() - t0:.0f}s -> {out_dir}")
    return paths


def build(cube_dir: str, verbose: bool = True):
    """Manifest with the incomplete E-OBS variables dropped, join verified."""
    from data.loader import load_cube
    from encoders.manifest import assert_weather_join, build_manifest

    paths = sorted(glob.glob(os.path.join(cube_dir, "*.nc")))
    assert paths, f"no cubes under {cube_dir}"
    samples = [load_cube(p, verbose=False) for p in paths]
    manifest = build_manifest(samples, verbose=False)

    present = [v for v in INCOMPLETE_VARS if v in manifest.columns]
    if present:
        manifest = manifest.drop(columns=present)
        if verbose:
            print(f"[build] dropped {present}: incomplete on this tile. NOTHING "
                  "IS FILLED, and these numbers are therefore NOT directly "
                  "comparable with 32UNU's 8-variable ones.")
    # Runs BEFORE anything is fitted: weather is this probe's entire input.
    assert_weather_join(manifest, cube_dir, verbose=verbose)

    import pandas as pd
    row_years = sorted(set(pd.to_datetime(manifest.timestamp).dt.year.tolist()))
    col_years = sorted(manifest.year.unique().tolist())
    assert col_years == row_years, (
        f"the manifest's year column {col_years} disagrees with the per-row "
        f"calendar years {row_years}; year-aware modes would refuse this"
    )
    if verbose:
        print(f"[build] manifest {manifest.shape}, "
              f"{manifest.cube_id.nunique()} cubes, years {row_years}")
    return manifest


def minimum_detectable_effect(per_fold, alpha: float = 0.05,
                              power: float = 0.80) -> dict:
    """How large an effect THIS design could have detected, and at what n.

    The number that turns "the interval includes zero" into something
    actionable. Uses the realised per-fold spread as the noise estimate, so it
    describes the design that was actually run rather than an assumed one.
    """
    from scipy.stats import norm

    v = np.asarray(per_fold, dtype=float)
    v = v[np.isfinite(v)]
    k = v.size
    if k < 2:
        return {"n_folds": k, "sd": float("nan"), "mde": float("nan")}
    sd = float(v.std(ddof=1))
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    mde = z * sd / np.sqrt(k)
    # Folds needed for the OBSERVED effect to clear the same bar.
    obs = abs(float(v.mean()))
    need = int(np.ceil((z * sd / obs) ** 2)) if obs > 0 else -1
    return {"n_folds": k, "sd": sd, "mde": float(mde),
            "observed": float(v.mean()), "folds_needed_for_observed": need}


def run(tile: str, n: int, k: int, out_root: str, verbose: bool = True):
    import pandas as pd

    from probes import p4_ceiling as p4

    cube_dir = os.path.join(out_root, "raw")
    download(tile, n, cube_dir, verbose=verbose)
    manifest = build(cube_dir, verbose=verbose)

    fs = (FEATURE_SET,)
    data = p4.build_p4_data(manifest, cube_dir, feature_sets=fs, verbose=False)

    t0 = time.time()
    df_a = p4.run_stage_a(data, targets=("cube_mean",), fold_modes=("cube",),
                          estimators=("linear",), feature_sets=fs, k=k,
                          verbose=False)
    print(f"[run] Stage A (PROXY climatology, cube folds): {len(df_a)} rows, "
          f"{time.time() - t0:.0f}s")

    t0 = time.time()
    df_b, info = p4.run_stage_b(manifest, cube_dir, targets=("cube_mean",),
                                estimators=("linear",), feature_sets=fs, k=k,
                                verbose=False)
    assert df_b is not None, (
        "Stage B did not run on a seasonal tile. Detection said "
        f"{info}; it should have found multi-year cubes."
    )
    print(f"[run] Stage B (REAL leave-year-out, crossed folds): {len(df_b)} "
          f"rows, {time.time() - t0:.0f}s")

    df = p4.add_margins(pd.concat([df_a, df_b], ignore_index=True), verbose=False)
    p4.assert_stage_b_ran_or_deferred(df, info)

    os.makedirs(out_root, exist_ok=True)
    csv = os.path.join(out_root, "p4_validation_results.csv")
    df.to_csv(csv, index=False)
    print(f"[run] wrote {csv} ({len(df)} rows x {df.shape[1]} cols)")
    return df, manifest


def report(df, manifest, tile: str) -> None:
    print("\n" + "=" * 78)
    print(f"PROXY vs REAL leave-year-out climatology -- tile {tile}, "
          f"{manifest.cube_id.nunique()} cubes")
    print("=" * 78)
    print("NOT a controlled comparison: Stage A uses `cube` folds and a "
          "day-of-year curve\nfitted to the SPATIAL MEAN; Stage B uses "
          "`crossed` folds and a PER-PIXEL\nclimatology. Protocols differ, "
          "necessarily -- see this module's docstring.\n")

    for stage, label in (("A", "PROXY (Stage A, cube folds)"),
                         ("B", "REAL  (Stage B, crossed folds)")):
        sub = df[df.stage == stage]
        w = sub[sub.model_kind == "weather"].iloc[0]
        c = sub[sub.model_kind == "observation"].iloc[0]
        print(f"{label}")
        print(f"  weather      r2_vs_clim {w.r2_vs_climatology_mean:+.3f}  "
              f"[{w.r2_vs_climatology_ci_lo:+.3f}, "
              f"{w.r2_vs_climatology_ci_hi:+.3f}]   "
              f"margin {w.margin_over_control:+.3f}")
        print(f"  observation  r2_vs_clim {c.r2_vs_climatology_mean:+.3f}   "
              f"(effective n {int(w.effective_n)} CUBES, "
              f"{int(w.n_folds)} folds)")
        mde = minimum_detectable_effect(
            [float(x) for x in w.per_fold_r2_vs_climatology.split(";")])
        print(f"  power        per-fold sd {mde['sd']:.3f} -> this design could "
              f"detect |effect| >= {mde['mde']:.3f} at 80% power")
        if mde["folds_needed_for_observed"] > 0:
            print(f"               the observed {mde['observed']:+.3f} would "
                  f"need ~{mde['folds_needed_for_observed']} folds to clear "
                  "that bar")
        print()

    a = df[(df.stage == "A") & (df.model_kind == "weather")].iloc[0]
    b = df[(df.stage == "B") & (df.model_kind == "weather")].iloc[0]
    gap = a.r2_vs_climatology_mean - b.r2_vs_climatology_mean
    overlap = (a.r2_vs_climatology_ci_lo <= b.r2_vs_climatology_ci_hi
               and b.r2_vs_climatology_ci_lo <= a.r2_vs_climatology_ci_hi)
    print(f"proxy - real = {gap:+.3f}; fold-clustered intervals "
          f"{'OVERLAP' if overlap else 'DO NOT OVERLAP'}")
    if overlap:
        print("=> the two are not distinguishable at this sample size. That is "
              "NOT evidence\n   the proxy is sound -- it is the absence of "
              "evidence either way, and the\n   per-fold sd above says how much "
              "more data would settle it.")
    else:
        print("=> the proxy and the real climatology give materially DIFFERENT "
              "answers.\n   Stage A's number cannot stand in for H1.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", default="30TVN", choices=SEASONAL_TILES)
    ap.add_argument("--n", type=int, default=60, help="cubes to attempt")
    ap.add_argument("--k", type=int, default=5, help="outer folds")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_root = args.out or os.path.join("data", f"validation_{args.tile}")
    df, manifest = run(args.tile, args.n, args.k, out_root)
    report(df, manifest, args.tile)


if __name__ == "__main__":
    main()
