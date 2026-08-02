"""Find out why the data step failed, in under a minute, with the real traceback.

    python -m data.diagnose

Four escalating checks against the GreenEarthNet download path, stopping at the
first hard failure and printing the full traceback instead of swallowing it:

    1. imports    is the stack actually installed?
    2. s3         can we reach the MPG object store and list the tile?
    3. one cube   does a real cube download and open?
    4. loader     do ndvi() and the loader agree on it?

Exit code is non-zero on failure, so sh() in the notebook raises.

For the older live-extraction path (earthnet-minicuber, Planetary Computer) see
data/download_minicubes.py. It works but measured 14.7 hours for 20 cubes.
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback

TILE = "32UNU"
SPLIT = "train"


def _hdr(n: int, title: str) -> None:
    print(f"\n{'=' * 70}\n[{n}] {title}\n{'=' * 70}", flush=True)


def check_imports() -> bool:
    _hdr(1, "IMPORTS")
    print(f"python {sys.version.split()[0]}")
    pkgs = ["numpy", "xarray", "netCDF4", "pandas", "zarr", "s3fs", "torch", "earthnet"]
    ok = True
    for p in pkgs:
        try:
            m = __import__(p)
            print(f"  [ok]   {p:<12} {getattr(m, '__version__', '?')}")
        except Exception as e:
            ok = False
            print(f"  [FAIL] {p:<12} {type(e).__name__}: {e}")
    if not ok:
        print("\n>>> A package above is missing. Re-run the install cell WITHOUT -q:")
        print("      !pip install earthnet s3fs xarray zarr netCDF4")
    return ok


def check_s3() -> bool:
    _hdr(2, "OBJECT STORE (s3.bgc-jena.mpg.de, anonymous)")
    try:
        from data.download_greenearthnet import BUCKET, list_cubes

        keys = list_cubes(TILE, SPLIT)
        print(f"  {BUCKET}/{SPLIT}/{TILE}")
        print(f"  cubes in tile: {len(keys)}")
        print(f"  first: {os.path.basename(keys[0])}")
        return True
    except Exception:
        traceback.print_exc()
        print("\n>>> Could not list the store. Network, or the bucket layout changed.")
        return False


def check_one_cube() -> bool:
    _hdr(3, "ONE REAL CUBE")
    try:
        import xarray as xr

        from data.download_greenearthnet import list_cubes, s3fs_client

        s3 = s3fs_client()
        key = list_cubes(TILE, SPLIT, s3)[0]
        tmp = os.path.join(tempfile.gettempdir(), os.path.basename(key))
        s3.download(key, tmp)
        print(f"  downloaded {os.path.basename(key)} ({os.path.getsize(tmp) / 1e6:.1f} MB)")
        with xr.open_dataset(tmp) as ds:
            print(f"  dims {dict(ds.sizes)}")
            need = ["s2_B02", "s2_B03", "s2_B04", "s2_B8A", "s2_mask"]
            missing = [v for v in need if v not in ds.variables]
            assert not missing, f"missing {missing}"
            print(f"  all of {need} present")
        globals()["_CUBE"] = tmp
        return True
    except Exception:
        traceback.print_exc()
        print("\n>>> The download or the file itself is broken.")
        return False


def check_loader() -> bool:
    _hdr(4, "LOADER AND CANONICAL NDVI")
    try:
        import numpy as np

        from data.loader import cube_ndvi, load_cube

        s = load_cube(globals()["_CUBE"])
        nd = cube_ndvi(s)
        print(f"  values {s.values.shape} | mask {s.mask.shape} | ndvi {nd.shape}")
        dt = np.diff(s.timestamps) / np.timedelta64(1, "D")
        print(f"  dt/days min {dt.min():.0f} median {np.median(dt):.0f} max {dt.max():.0f}")
        assert dt.min() != dt.max(), "time grid is perfectly regular, gaps were filled"
        assert np.isnan(nd[~s.mask]).all(), "masked pixels leaked into NDVI"
        v = nd[np.isfinite(nd)]
        print(f"  valid NDVI fraction {v.size / nd.size:.3f} | median {np.median(v):.3f}")
        assert np.median(v) > 0.0, "negative median NDVI, B04 and B8A may be swapped"
        print("\n>>> Pipeline works. The batch download should work too.")
        return True
    except Exception:
        traceback.print_exc()
        print("\n>>> THIS IS THE TRACEBACK THAT MATTERS.")
        return False


def main() -> None:
    steps = [("imports", check_imports), ("s3", check_s3),
             ("cube", check_one_cube), ("loader", check_loader)]
    for name, fn in steps:
        if not fn():
            print(f"\n{'=' * 70}\nDIAGNOSIS: first failure at step '{name}'.\n{'=' * 70}")
            sys.exit(1)
    print(f"\n{'=' * 70}\nAll four checks passed.\n{'=' * 70}")


if __name__ == "__main__":
    main()
