"""Find out why the download failed, in ~90 seconds, with the real traceback.

    python -m data.diagnose

Runs four escalating checks and STOPS at the first hard failure, printing the
full traceback instead of swallowing it:

    1. imports        - is the stack actually installed?
    2. STAC query     - can we reach Planetary Computer and see scenes?
    3. cloud mask     - can we fetch the U-Net checkpoint and run it?
    4. one month cube - the whole pipeline on the smallest possible request

Exit code is non-zero on failure, so `run()` in the notebook raises.
"""

from __future__ import annotations

import sys
import traceback

LON, LAT = 11.55, 48.15
BBOX = [LON - 0.02, LAT - 0.02, LON + 0.02, LAT + 0.02]
ONE_MONTH = "2020-07-01/2020-08-01"  # peak growing season: scenes must exist

_FAILED: list = []


def _hdr(n: int, title: str) -> None:
    print(f"\n{'=' * 70}\n[{n}] {title}\n{'=' * 70}", flush=True)


def check_imports() -> bool:
    _hdr(1, "IMPORTS")
    print(f"python {sys.version.split()[0]}")
    pkgs = ["numpy", "xarray", "rasterio", "zarr", "dask", "netCDF4", "pandas",
            "torch", "segmentation_models_pytorch", "pystac_client",
            "planetary_computer", "stackstac", "rioxarray", "pyproj", "shapely",
            "sen2nbar", "earthnet_minicuber", "earthnet"]
    ok = True
    for p in pkgs:
        try:
            m = __import__(p)
            print(f"  [ok]   {p:<32} {getattr(m, '__version__', '?')}")
        except Exception as e:
            ok = False
            print(f"  [FAIL] {p:<32} {type(e).__name__}: {e}")
    if not ok:
        print("\n>>> A package above is missing or broken. This is the cause.")
        print(">>> Re-run the install cell WITHOUT -q and read the pip output:")
        print("      !pip install earthnet earthnet-minicuber sen2nbar rasterio "
              "xarray zarr netCDF4 dask")
        print(">>> NOTE: sen2nbar is imported by earthnet-minicuber but missing "
              "from its install_requires, so pip will not pull it for you.")
    return ok


def check_stac() -> bool:
    _hdr(2, "STAC QUERY (Microsoft Planetary Computer)")
    try:
        import planetary_computer as pc
        import pystac_client

        cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
        search = cat.search(bbox=BBOX, collections=["sentinel-2-l2a"], datetime=ONE_MONTH)
        items = pc.sign(search)
        feats = items.to_dict()["features"]
        print(f"  bbox {BBOX}")
        print(f"  window {ONE_MONTH}")
        print(f"  scenes found: {len(feats)}")
        if not feats:
            print("\n>>> Zero scenes. Either the bbox/date is wrong or the catalog changed.")
            return False
        p = feats[0]["properties"]
        print(f"  first scene: {p.get('datetime')} epsg={p.get('proj:epsg')} "
              f"cloud={p.get('eo:cloud_cover')}")
        print(f"  assets: {sorted(feats[0]['assets'])[:8]} ...")
        return True
    except Exception:
        traceback.print_exc()
        print("\n>>> Could not query Planetary Computer. Network/firewall, or a "
              "pystac-client / planetary-computer version mismatch.")
        return False


def check_cloudmask() -> bool:
    _hdr(3, "CLOUD-MASK CHECKPOINT (nextcloud.bgc-jena.mpg.de)")
    try:
        import numpy as np
        import torch
        import xarray as xr
        from earthnet_minicuber.provider.s2.cloudmask import CloudMask

        cm = CloudMask(bands=["B02", "B03", "B04", "B8A"])
        print(f"  checkpoint loaded, ckpt_bands={cm.ckpt_bands}")
        print(f"  model training mode: {cm.model.training} (must be False)")

        stack = xr.DataArray(
            np.random.default_rng(0).uniform(0, 3000, (1, 4, 64, 64)).astype("float32"),
            dims=("time", "band", "y", "x"),
            coords={"time": [np.datetime64("2020-07-01")],
                    "band": ["B02", "B03", "B04", "B8A"],
                    "y": np.arange(64), "x": np.arange(64)},
        )
        out = cm(stack)
        m = out.sel(band="mask").values
        print(f"  ran on dummy (1, 4, 64, 64) -> mask {m.shape} "
              f"codes {sorted(np.unique(m).tolist())}")
        print(f"  torch device: cpu (expected - the mask is not GPU-accelerated)")
        return True
    except Exception:
        traceback.print_exc()
        print("\n>>> Cloud mask failed. Common causes: nextcloud.bgc-jena.mpg.de "
              "unreachable, or a segmentation-models-pytorch / torch mismatch.")
        return False


def check_one_month_cube() -> bool:
    _hdr(4, "ONE-MONTH MINICUBE (the real pipeline, smallest request)")
    try:
        from data.download_minicubes import _load_minicube, specs_for

        specs = specs_for(LON, LAT)
        specs["time_interval"] = ONE_MONTH
        specs["xy_shape"] = (64, 64)
        print(f"  specs: {specs['lon_lat']} {specs['xy_shape']} @ {specs['resolution']} m, "
              f"{specs['time_interval']}")
        cube = _load_minicube(specs)
        print(f"  dims {dict(cube.sizes)}")
        print(f"  vars {sorted(map(str, cube.data_vars))}")
        assert cube.sizes.get("time", 0) > 0, "empty time axis"
        import numpy as np

        b04 = cube["s2_B04"].values
        print(f"  s2_B04 min {np.nanmin(b04):.4f} max {np.nanmax(b04):.4f}")
        if "s2_mask" in cube:
            m = cube["s2_mask"].values
            print(f"  s2_mask clear(=0) fraction {float(np.mean(m == 0)):.3f}")
        print("\n>>> Pipeline works. The batch download should work too.")
        return True
    except Exception:
        traceback.print_exc()
        print("\n>>> THIS IS THE TRACEBACK THAT THE BATCH DOWNLOAD WAS SWALLOWING.")
        return False


def main() -> None:
    steps = [("imports", check_imports), ("stac", check_stac),
             ("cloudmask", check_cloudmask), ("cube", check_one_month_cube)]
    for name, fn in steps:
        if not fn():
            print(f"\n{'=' * 70}\nDIAGNOSIS: first failure at step '{name}'.\n"
                  f"{'=' * 70}")
            sys.exit(1)
    print(f"\n{'=' * 70}\nAll four checks passed.\n{'=' * 70}")


if __name__ == "__main__":
    main()
