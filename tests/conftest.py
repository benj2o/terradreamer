"""Shared fixtures. Adds the repo root to sys.path so `import data.*` works
without installing the project."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_synthetic_cube(path, T_acquired=8, H=8, W=8, seed=0):
    """Write a NetCDF cube that mimics earthnet-minicuber output.

    Deliberately nasty: irregular cadence, one duplicate timestamp, and two
    timesteps with no acquisition at all (all-NaN reflectance).
    """
    import pandas as pd
    import xarray as xr

    rng = np.random.default_rng(seed)
    gaps = [5, 5, 10, 5, 15, 5, 20][: T_acquired - 1]
    days = np.cumsum([0] + gaps)
    times = [np.datetime64("2019-04-01") + np.timedelta64(int(d), "D") for d in days]
    times = times + [times[3]] + [times[-1] + np.timedelta64(5, "D")] * 1  # dup + 1 extra
    times = times + [times[-1] + np.timedelta64(7, "D")]
    T = len(times)

    bands = ("B02", "B03", "B04", "B8A")
    data = {}
    for b in bands:
        arr = rng.uniform(0.02, 0.45, size=(T, H, W)).astype(np.float32)
        if b == "B8A":
            arr += 0.25  # keep NDVI positive-ish, like vegetation
        arr[-2:] = np.nan  # last two timesteps: no acquisition
        data[f"s2_{b}"] = (("time", "lat", "lon"), arr)

    codes = rng.choice([0, 0, 0, 1, 2, 4], size=(T, H, W)).astype(np.uint8)
    data["s2_mask"] = (("time", "lat", "lon"), codes)

    ds = xr.Dataset(
        data,
        coords={
            "time": pd.to_datetime(times),
            "lat": np.linspace(48.15, 48.13, H),
            "lon": np.linspace(11.55, 11.57, W),
        },
    )
    ds.to_netcdf(path)
    return path


@pytest.fixture
def synthetic_cube(tmp_path):
    return make_synthetic_cube(str(tmp_path / "mc_00_synthetic.nc"))
