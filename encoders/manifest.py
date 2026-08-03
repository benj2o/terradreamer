"""The (cube, frame) manifest that Phase 1.3 indexes and strata-splits on.

One row per RETAINED frame, carrying everything a probe needs to define a
horizon, group a split, or stratify a replication.

TWO FIELDS THAT ARE NOT OPTIONAL
--------------------------------

``original_axis_index`` -- horizons MUST be defined in DAYS on the original
regular time axis, never in retained frames. After cloudy frames are dropped,
a gap of 5 retained frames spans 25 days in clear weather and 40+ in cloudy
weather; cloud correlates with precipitation, and precipitation is a weather
feature. A frame-defined horizon therefore leaks weather into the horizon
itself and degrades persistence differently than it degrades the probe, which
contaminates exactly the comparison the paper rests on. This column, plus
``timestamp``, is what makes a day-defined horizon possible.

``landcover_stratum`` -- the plan makes per-stratum replication the condition
for BELIEVING the headline result, so without it the headline cannot be
believed. Source, established by lookup rather than assumption: GreenEarthNet
minicubes ship **`esawc_lc`** (ESA WorldCover 10 m) as an in-cube variable
alongside `cop_dem`, `nasa_dem`, `alos_dem`, `geom_cls` and the E-OBS climate
stack. No external join is needed. If a cube ever lacks the layer, the row
records the stratum as "ABSENT:<reason>" rather than silently null, and
``assert_strata_present`` fails loudly.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import xarray as xr

from data.loader import CubeSample
from encoders.base import GRID
from encoders.frames import MIN_CLEAR_FRACTION, select_clear_frames

__all__ = [
    "ESA_WORLDCOVER_CLASSES",
    "LANDCOVER_VAR",
    "DEM_VAR",
    "cube_landcover",
    "cube_grid_landcover",
    "cube_grid_elevation",
    "cube_weather",
    "manifest_rows",
    "build_manifest",
    "assert_strata_present",
]

LANDCOVER_VAR = "esawc_lc"
DEM_VAR = "cop_dem"

# ESA WorldCover 10 m v100/v200 class codes.
ESA_WORLDCOVER_CLASSES = {
    10: "tree_cover", 20: "shrubland", 30: "grassland", 40: "cropland",
    50: "built_up", 60: "bare_sparse", 70: "snow_ice", 80: "water",
    90: "herbaceous_wetland", 95: "mangrove", 100: "moss_lichen",
}


def cube_landcover(path: str) -> tuple:
    """(dominant class name, fractions dict) for one cube, from `esawc_lc`.

    Returns ("ABSENT:<reason>", {}) if the layer is missing, never None: a
    silently null stratum would let the replication condition be skipped.
    """
    with xr.open_dataset(path) as ds:
        if LANDCOVER_VAR not in ds.variables:
            return f"ABSENT:no {LANDCOVER_VAR} variable in cube", {}
        lc = np.asarray(ds[LANDCOVER_VAR].values)

    lc = lc[np.isfinite(lc)] if np.issubdtype(lc.dtype, np.floating) else lc.ravel()
    if lc.size == 0:
        return f"ABSENT:{LANDCOVER_VAR} present but empty", {}

    codes, counts = np.unique(lc.astype(int), return_counts=True)
    frac = {ESA_WORLDCOVER_CLASSES.get(int(c), f"unknown_{int(c)}"): float(n / counts.sum())
            for c, n in zip(codes, counts)}
    dominant = max(frac, key=frac.get)
    return dominant, frac


def cube_grid_landcover(path: str, grid: int = GRID) -> tuple:
    """Per-GRID-CELL dominant WorldCover class: (names (16,), purity (16,)).

    A 128 x 128 cube at 20 m is 2.56 km across; a grid cell is 640 m. At that
    footprint the Alpine foreland is genuinely mixed cropland / grassland /
    forest, so a per-cube dominant class is a coarse label over a heterogeneous
    scene. Per-cell strata give stratum contrast WITHIN one weather realisation
    -- the same cube, the same sky, different land cover -- which is a much
    stronger replication argument than comparing whole cubes that also differ
    in weather. It also lets P2/P3 filter grid cells by stratum with no
    re-encode.

    Cells are row-major, matching encoders.base.pool_to_grid and
    encoders.frames.grid_clear_fraction.

    CAVEAT, recorded because it is a methods footnote a reviewer will want:
    ESA WorldCover is a STATIC ~2020/21 product and these cubes are 2018. Land
    cover is stable over three years for forest/grassland/built-up but crop
    rotation can move a cropland cell between years. It is a stratification
    label, never a target, so the exposure is small -- but it is not zero.
    """
    with xr.open_dataset(path) as ds:
        if LANDCOVER_VAR not in ds.variables:
            return (np.array([f"ABSENT:no {LANDCOVER_VAR}"] * (grid * grid)),
                    np.full(grid * grid, np.nan))
        lc = np.asarray(ds[LANDCOVER_VAR].values)
    assert lc.ndim == 2, f"{LANDCOVER_VAR} must be (lat, lon), got {lc.shape}"
    H, W = lc.shape
    assert H % grid == 0 and W % grid == 0, f"{H}x{W} not divisible by {grid}"
    ch, cw = H // grid, W // grid

    names, purity = [], []
    for gy in range(grid):
        for gx in range(grid):
            block = lc[gy * ch:(gy + 1) * ch, gx * cw:(gx + 1) * cw].ravel()
            block = block[np.isfinite(block)] if np.issubdtype(block.dtype, np.floating) else block
            if block.size == 0:
                names.append("ABSENT:empty cell"); purity.append(np.nan); continue
            codes, counts = np.unique(block.astype(int), return_counts=True)
            j = int(counts.argmax())
            names.append(ESA_WORLDCOVER_CLASSES.get(int(codes[j]), f"unknown_{int(codes[j])}"))
            purity.append(float(counts[j] / counts.sum()))
    return np.array(names), np.array(purity)


def cube_grid_elevation(path: str, grid: int = GRID) -> np.ndarray:
    """Mean Copernicus DEM elevation per grid cell, (16,) metres, or NaNs.

    Elevation drives phenology timing in the Alpine foreland, so it is both a
    stratification axis and the control variable a reviewer will ask for when
    green-up dates differ between cubes.
    """
    with xr.open_dataset(path) as ds:
        if DEM_VAR not in ds.variables:
            return np.full(grid * grid, np.nan)
        dem = np.asarray(ds[DEM_VAR].values, dtype=float)
    H, W = dem.shape
    ch, cw = H // grid, W // grid
    return np.array([np.nanmean(dem[gy * ch:(gy + 1) * ch, gx * cw:(gx + 1) * cw])
                     for gy in range(grid) for gx in range(grid)])


def cube_weather(path: str) -> dict:
    """The in-cube E-OBS daily series, {var: (T_original,)}.

    P4's ENTIRE input, and it needs no external table: GreenEarthNet ships
    these per cube on the original daily axis, already aligned with
    ``original_axis_index``. Confirmed present and fully finite on tile 32UNU.
    Note it is EIGHT variables, not nine: tg, tn, tx (mean/min/max temperature),
    rr (precipitation), pp (pressure), fg (wind speed), hu (humidity),
    qq (shortwave radiation).
    """
    out = {}
    with xr.open_dataset(path) as ds:
        for v in ds.data_vars:
            v = str(v)
            if v.startswith("eobs_"):
                a = np.asarray(ds[v].values, dtype=float)
                assert a.ndim == 1, f"{v} expected (time,), got {a.shape}"
                out[v] = a
    return out


def _pixel_bbox(cube_id: str) -> tuple:
    """(row0, row1, col0, col1) parsed from the GreenEarthNet cube id.

    Ids look like 32UNU_2018-03-09_2018-08-05_1081_1209_3641_3769_16_96_56_136,
    where fields 3..6 are the tile-pixel window. Used as the spatial grouping
    key by probes/cv.py, so it must come from the id, not from coordinates.
    """
    stem = cube_id[:-3] if cube_id.endswith(".nc") else cube_id
    p = stem.split("_")
    assert len(p) >= 7, f"cannot parse a pixel bbox from {cube_id!r}"
    return tuple(int(x) for x in p[3:7])


def manifest_rows(
    sample: CubeSample,
    min_clear: float = MIN_CLEAR_FRACTION,
    landcover: tuple | None = None,
) -> list:
    """One dict per RETAINED frame of one cube."""
    sel = select_clear_frames(sample.values, sample.timestamps, sample.mask,
                             min_clear=min_clear, verbose=False)
    cube_id = os.path.basename(sample.path)
    stem = cube_id[:-3] if cube_id.endswith(".nc") else cube_id
    tile = stem.split("_")[0]
    year = int(stem.split("_")[1][:4])
    bbox = _pixel_bbox(cube_id)
    dominant, frac = landcover if landcover is not None else cube_landcover(sample.path)
    cell_lc, cell_purity = cube_grid_landcover(sample.path)
    cell_elev = cube_grid_elevation(sample.path)
    weather = cube_weather(sample.path)

    rows = []
    for i in range(sel.values.shape[0]):
        ts = sel.timestamps[i]
        rows.append({
            "cube_id": cube_id,
            "tile": tile,
            "year": year,
            "timestamp": ts,
            # Horizons are defined in DAYS on this axis, never in retained frames.
            "original_axis_index": int(sel.kept_idx[i]),
            "day_of_year": int(np.datetime64(ts, "D").astype("datetime64[D]").astype(object).timetuple().tm_yday),
            "pixel_bbox": bbox,
            "clear_frac": float(sel.clear_frac[i]),
            "landcover_stratum": dominant,
            "landcover_dominant_frac": float(frac.get(dominant, np.nan)) if frac else np.nan,
            # Per-CELL strata: 16 labels aligned with emb_grid / grid_clear_frac.
            # Constant down a cube's rows (WorldCover is static) but carried per
            # row so a probe can explode to (frame, cell) with one .explode().
            "grid_landcover": tuple(cell_lc.tolist()),
            "grid_landcover_purity": tuple(np.round(cell_purity, 4).tolist()),
            "grid_elevation_m": tuple(np.round(cell_elev, 1).tolist()),
            # E-OBS on the ORIGINAL daily axis, indexed by original_axis_index.
            **{v: float(a[sel.kept_idx[i]]) if sel.kept_idx[i] < a.size else np.nan
               for v, a in weather.items()},
        })
    return rows


def build_manifest(samples: Sequence[CubeSample], verbose: bool = True):
    """Manifest over many cubes as a pandas DataFrame."""
    import pandas as pd

    rows = []
    for s in samples:
        rows.extend(manifest_rows(s))
    df = pd.DataFrame(rows)
    assert len(df), "empty manifest"
    if verbose:
        print(f"[manifest] {len(df)} (cube, frame) rows over {df.cube_id.nunique()} cubes")
        print(f"[manifest] columns: {list(df.columns)}")
        print(f"[manifest] landcover strata (per cube): "
              f"{df.groupby('landcover_stratum').cube_id.nunique().to_dict()}")
        if "grid_landcover" in df.columns:
            from collections import Counter
            per_cube = df.drop_duplicates("cube_id")
            cells = Counter(c for row in per_cube.grid_landcover for c in row)
            print(f"[manifest] landcover strata (per grid cell, {sum(cells.values())} "
                  f"cells over {len(per_cube)} cubes): {dict(cells.most_common())}")
            mixed = sum(1 for row in per_cube.grid_landcover if len(set(row)) > 1)
            print(f"[manifest] cubes whose cells are NOT all one class: "
                  f"{mixed}/{len(per_cube)} -- this is the within-cube stratum "
                  "contrast the per-cube label was hiding")
        eobs = sorted(c for c in df.columns if c.startswith("eobs_"))
        print(f"[manifest] in-cube E-OBS joined on original_axis_index ({len(eobs)}): "
              f"{eobs}")
        span = (df.original_axis_index.max() - df.original_axis_index.min())
        print(f"[manifest] original_axis_index spans 0..{df.original_axis_index.max()} "
              f"(range {span}); horizons are defined in DAYS on this axis")
    return df


def assert_strata_present(df) -> None:
    """Every row must carry a stratum, or record why not. Never silently null."""
    assert "landcover_stratum" in df.columns, "manifest has no landcover_stratum"
    null = df.landcover_stratum.isna().sum()
    assert null == 0, f"{null} manifest rows have a null landcover_stratum"
    absent = df[df.landcover_stratum.astype(str).str.startswith("ABSENT:")]
    if len(absent):
        reasons = sorted(set(absent.landcover_stratum))
        raise AssertionError(
            f"{len(absent)} rows across {absent.cube_id.nunique()} cube(s) have no "
            f"land cover: {reasons}. Per-stratum replication is the condition for "
            "believing the headline result, so this cannot be left unresolved."
        )
