"""Download <=20 Sentinel-2 minicubes over the Bavarian / Alpine foreland.

Tile centre (lon, lat) = (11.55, 48.15), Alpine foreland south of Munich
(Starnberg / Wolfratshausen: mixed grassland, arable, forest, some wetland).

    python -m data.download_minicubes --out data/raw --n 20

Cube geometry: 128 x 128 px at 20 m = 2.56 km. Centres are laid out on a grid
with 1.5x that spacing so NO TWO CUBES OVERLAP. Overlapping cubes would put
literally the same pixels on both sides of a train/test split later.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import threading
import time
import traceback

import numpy as np

# Grid tile centre, Alpine foreland south of Munich.
LON0, LAT0 = 11.55, 48.15
TIME_INTERVAL = "2019-01-01/2021-12-31"
BANDS = ["B02", "B03", "B04", "B8A"]
RESOLUTION = 20  # metres
XY_SHAPE = (128, 128)
SPACING_FACTOR = 1.5  # centre-to-centre spacing in units of cube width


def _month_count(time_interval: str) -> int:
    """Number of monthly STAC queries minicuber will issue for this window.

    Mirrors Minicuber.monthly_intervals exactly, so the progress line and the
    ETA extrapolation match what actually happens.
    """
    import pandas as pd

    start = pd.Timestamp(time_interval[:10])
    end = pd.Timestamp(time_interval[-10:])
    n = 0
    monthstart = start
    monthend = monthstart + pd.offsets.MonthEnd()
    while monthend < end - pd.Timedelta("15 days"):
        if (monthend - monthstart) < pd.Timedelta("15 days"):
            monthend = monthend + pd.Timedelta("1 days") + pd.offsets.MonthEnd()
        if monthend > end - pd.Timedelta("15 days"):
            break
        n += 1
        monthstart = monthend + pd.Timedelta("1 days")
        monthend = monthstart + pd.offsets.MonthEnd()
    return n + 1


def cube_centres(n: int, n_cols: int = 5, grid_n: int = 20):
    """First `n` non-overlapping (lon, lat) centres of the FULL `grid_n` grid.

    The layout is pinned by `grid_n`, never by `n`. If the number of rows were
    derived from `n`, then `--n 1` would centre a 1x1 grid on (LON0, LAT0) and
    produce a cube that is NOT a member of the final 20, and that overlaps two
    of them by 0.64 km, i.e. shares identical pixels with cubes that a later
    train/test split would separate. The preflight cube must be cube 00 of the
    real grid so the full run simply skips it.
    """
    width_m = XY_SHAPE[0] * RESOLUTION
    dlat = SPACING_FACTOR * width_m / 111_320.0
    dlon = SPACING_FACTOR * width_m / (111_320.0 * math.cos(math.radians(LAT0)))
    n_rows = int(math.ceil(grid_n / n_cols))
    centres = []
    for r in range(n_rows):
        for c in range(n_cols):
            if len(centres) >= grid_n:
                break
            lon = LON0 + (c - (n_cols - 1) / 2) * dlon
            lat = LAT0 + (r - (n_rows - 1) / 2) * dlat
            centres.append((round(lon, 5), round(lat, 5)))
    centres = centres[:n]
    sep_km = SPACING_FACTOR * width_m / 1000.0
    print(f"[grid] {len(centres)} of {grid_n} centres ({n_cols} cols x {n_rows} rows) | "
          f"cube {width_m / 1000:.2f} km, centre spacing {sep_km:.2f} km "
          f"(gap {sep_km - width_m / 1000:.2f} km, no overlap)")
    print(f"[grid] lon {min(c[0] for c in centres):.4f}..{max(c[0] for c in centres):.4f} | "
          f"lat {min(c[1] for c in centres):.4f}..{max(c[1] for c in centres):.4f}")
    return centres


def specs_for(lon: float, lat: float, time_interval: str = TIME_INTERVAL) -> dict:
    return {
        "lon_lat": (lon, lat),
        "xy_shape": XY_SHAPE,
        "resolution": RESOLUTION,
        "time_interval": time_interval,
        "providers": [
            {
                "name": "s2",
                "kwargs": {
                    "bands": BANDS,
                    "best_orbit_filter": False,   # keep every acquisition ...
                    "five_daily_filter": False,   # ... and the native irregular cadence
                    "brdf_correction": True,
                    "cloud_mask": True,           # -> s2_mask: 0 clear, 1 cloud, 2 shadow, 3 snow, 4 other
                    # The cloud mask is a mobilenet-v2 U-Net trained on cloudSEN12 at
                    # 10 m. Our data is 20 m, so upsample 2x to match its training
                    # scale. This is 4x the compute, and it is the runtime bottleneck.
                    "cloud_mask_rescale_factor": 2,
                    # Removes S2 BOA_ADD_OFFSET (baseline >= 04.00). Leaving this
                    # False makes NDVI confidently wrong. Default is True; explicit.
                    "correct_processing_baseline": True,
                    "s2_avail_var": True,
                    "aws_bucket": "planetary_computer",
                },
            }
        ],
    }


def _load_minicube(specs):
    # Must run before minicuber reaches stackstac.stack. See data/stackstac_compat.py.
    from data.stackstac_compat import apply as _apply_stackstac_compat

    _apply_stackstac_compat()
    try:
        import earthnet_minicuber as emc
        loader = getattr(emc, "load_minicube", None)
        if loader is None:
            from earthnet_minicuber.minicuber import load_minicube as loader  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "earthnet-minicuber is not installed. pip install earthnet-minicuber"
        ) from e
    return loader(specs, compute=True)


class _Heartbeat:
    """Print an elapsed-time line every `every` seconds while a cube downloads.

    minicuber prints one line per month (36 per cube) and can spend minutes
    inside a single stackstac read, so silence is ambiguous: still working, or
    hung? This makes it unambiguous.
    """

    def __init__(self, label: str, every: float = 30.0):
        self.label, self.every = label, every
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        t0 = time.time()
        while not self._stop.wait(self.every):
            print(f"          [heartbeat] {self.label}: "
                  f"{(time.time() - t0) / 60:.1f} min elapsed, still working",
                  flush=True)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()


def _is_readable(path: str) -> bool:
    """A cube truncated by a Colab disconnect must never be trusted by --skip."""
    try:
        import xarray as xr

        with xr.open_dataset(path) as ds:
            return "time" in ds.sizes and ds.sizes["time"] > 0
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--n", type=int, default=20, help="number of minicubes (<=20)")
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--keep-going", action="store_true",
                    help="continue past a failing cube instead of stopping at the "
                         "first traceback (default: stop, so the error is visible)")
    ap.add_argument("--time-interval", default=TIME_INTERVAL,
                    help="override the time window, e.g. 2020-06-01/2020-08-01. "
                         "Use a short window ONLY for timing probes, and send the "
                         "output to a scratch dir, never to data/raw.")
    args = ap.parse_args()
    assert 1 <= args.n <= 20, "Munich-first: keep this <= 20 cubes"

    n_months = _month_count(args.time_interval)
    if args.time_interval != TIME_INTERVAL:
        print(f"[warn] time_interval overridden to {args.time_interval} "
              f"({n_months} months). These cubes are NOT the project dataset.")

    os.makedirs(args.out, exist_ok=True)

    # A Ctrl-C leaves a .partial behind (KeyboardInterrupt is not an Exception).
    # Sweep them so a later run never inherits a half-written file.
    for stale in glob.glob(os.path.join(args.out, "*.partial")):
        print(f"[clean] removing stale {os.path.basename(stale)}")
        os.remove(stale)

    centres = cube_centres(args.n, args.cols)
    print(f"[plan] {len(centres)} cube(s) x {n_months} monthly queries each")

    ok, failed = [], []
    t_batch = time.time()
    for i, (lon, lat) in enumerate(centres):
        path = os.path.join(args.out, f"mc_{i:02d}_lon{lon:.4f}_lat{lat:.4f}.nc")
        tmp = path + ".partial"

        if os.path.exists(path) and not args.overwrite:
            if _is_readable(path):
                print(f"[{i + 1:02d}/{len(centres)}] exists, skipping {os.path.basename(path)}")
                ok.append(path)
                continue
            print(f"[{i + 1:02d}/{len(centres)}] {os.path.basename(path)} is corrupt "
                  "(truncated download?) -- redoing")
            os.remove(path)

        print(f"[{i + 1:02d}/{len(centres)}] downloading lon={lon} lat={lat} ...", flush=True)
        t0 = time.time()
        try:
            with _Heartbeat(f"cube {i + 1}/{len(centres)}"):
                cube = _load_minicube(specs_for(lon, lat, args.time_interval))
            dims = {str(k): int(v) for k, v in cube.sizes.items()}
            print(f"          cube dims {dims}")
            print(f"          vars {sorted(map(str, cube.data_vars))}")
            assert "time" in cube.sizes and cube.sizes["time"] > 0, "empty time axis"
            for b in BANDS:
                assert f"s2_{b}" in cube.data_vars, (
                    f"expected variable s2_{b}; got {sorted(map(str, cube.data_vars))}"
                )

            if "s2_mask" in cube:
                m = cube["s2_mask"].values
                print(f"          s2_mask shape {m.shape} dtype {m.dtype} | "
                      f"clear(=0) fraction {float(np.mean(m == 0)):.3f}")
            b04 = cube["s2_B04"].values
            print(f"          s2_B04 min {np.nanmin(b04):.4f} max {np.nanmax(b04):.4f} "
                  "(expect ~[0, 1]; negatives => BOA offset still present)")

            # Write to .partial then rename: an interrupted run can never leave a
            # half-written cube that the skip-if-exists branch would trust.
            enc = {v: {"zlib": True, "complevel": 4} for v in cube.data_vars}
            cube.to_netcdf(tmp, encoding=enc)
            os.replace(tmp, path)

            dt = time.time() - t0
            remaining = (len(centres) - i - 1) * dt
            print(f"          -> {os.path.basename(path)} "
                  f"({os.path.getsize(path) / 1e6:.1f} MB, {dt / 60:.1f} min) "
                  f"| ETA for the rest ~{remaining / 60:.0f} min")
            ok.append(path)
        except Exception:
            print(f"          FAILED lon={lon} lat={lat} after {(time.time() - t0) / 60:.1f} min")
            traceback.print_exc()
            failed.append((lon, lat))
            if not args.keep_going:
                print("\n" + "=" * 70)
                print("STOPPING at the first failure so the traceback above is the "
                      "last thing you see.")
                print("Diagnose it with:  python -m data.diagnose")
                print("Or push past this one centre with:  --keep-going")
                print("=" * 70)
                sys.exit(1)
        finally:
            # Runs on Ctrl-C too, which `except Exception` does not catch.
            if os.path.exists(tmp):
                os.remove(tmp)

    total_mb = sum(os.path.getsize(p) for p in ok) / 1e6 if ok else 0.0
    print(f"\n[done] {len(ok)}/{len(centres)} cubes in {args.out} "
          f"({total_mb:.0f} MB, {(time.time() - t_batch) / 60:.1f} min)")

    # Exit code must reflect reality: a run that produced nothing exited 0 in the
    # first version of this script, which let the notebook sail on to an empty
    # data/raw and fail three cells later with a misleading error.
    if not ok:
        print("[done] NO CUBES WERE PRODUCED. Run: python -m data.diagnose")
        sys.exit(1)
    if failed:
        print(f"[done] {len(failed)} failed centres: {failed}")
        print("[done] re-run the SAME command -- finished cubes are skipped.")
        sys.exit(1)


if __name__ == "__main__":
    main()
