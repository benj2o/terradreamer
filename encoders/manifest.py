"""The (cube, frame) manifest that Phase 1.3 indexes and strata-splits on.

One row per RETAINED frame, carrying everything a probe needs to define a
horizon, group a split, or stratify a replication.

THERE ARE TWO TIME AXES, AND THEY ARE NOT THE SAME AXIS
-------------------------------------------------------
A GreenEarthNet minicube is stored on a ~150-step DAILY grid of which only
about 29 steps carry a Sentinel-2 acquisition. ``data.loader.load_cube`` drops
the empty steps, so everything downstream of it lives on the ACQUISITION axis.
Both axes are needed, and conflating them is silent:

``original_axis_index`` -- position on the ACQUISITION axis, i.e. exactly
``encoders.frames.select_clear_frames``' ``kept_idx``. It is the embedding join
key: ``(cube_id, original_axis_index) == (cube, kept_idx)``, asserted in
``probes.cv.join_embeddings``. Never re-point it at anything else.

``daily_axis_index`` -- position on the cube's ORIGINAL DAILY axis, derived
from the frame's TIMESTAMP via ``cube_daily_axis``. This is the axis the
in-cube E-OBS series lives on, and the axis a windowed weather feature must be
computed over.

Until 2026-08-10 the E-OBS columns were indexed with ``original_axis_index``
into a daily-axis array, so every row carried the weather of a day it was not
acquired on: over the 20-cube subset the offset was 4 to 122 days (median 53)
and **0 of 264 rows were correct** (mean-temperature MAE 6.26 K, max 22 K).
Fixed by joining on the timestamp; ``assert_weather_join`` re-derives the join
independently so it cannot regress.

HORIZONS ARE DEFINED IN DAYS, NEVER IN RETAINED FRAMES. After cloudy frames
are dropped, a gap of 5 retained frames spans 25 days in clear weather and 40+
in cloudy weather; cloud correlates with precipitation, and precipitation is a
weather feature. A frame-defined horizon therefore leaks weather into the
horizon itself and degrades persistence differently than it degrades the probe,
which contaminates exactly the comparison the paper rests on. ``timestamp`` and
``daily_axis_index`` are what make a day-defined horizon possible.

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
    "cube_daily_axis",
    "frame_daily_positions",
    "manifest_rows",
    "build_manifest",
    "assert_strata_present",
    "assert_weather_join",
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


def cube_daily_axis(path: str) -> np.ndarray:
    """(T_daily,) datetime64[ns]: the cube's ORIGINAL DAILY time axis.

    The axis ``cube_weather``'s series are indexed by, read from the same file
    so the two cannot come from different loads. ``data.loader.load_cube``
    deliberately drops the ~121 of 150 steps that carry no acquisition, so this
    axis is NOT the one ``original_axis_index`` counts on -- see the module
    docstring. A windowed weather feature must be computed over THIS axis:
    "the last 30 days" is 30 rows here and about 6 retained frames there, and
    which of those it is decides whether the window is weather-dependent.
    """
    with xr.open_dataset(path) as ds:
        t = np.asarray(ds["time"].values, dtype="datetime64[ns]")
    assert t.ndim == 1 and t.size, f"{path}: no usable time axis, got {t.shape}"
    assert (np.diff(t) > np.timedelta64(0, "ns")).all(), (
        f"{path}: the stored time axis is not strictly increasing, so "
        "searchsorted cannot be used to place a frame on it"
    )
    return t


def frame_daily_positions(daily_axis, frame_timestamps) -> np.ndarray:
    """(T_kept,) int positions of retained frames on the DAILY axis.

    Located by TIMESTAMP, and the located day is asserted to equal the frame's
    own timestamp -- an exact match, never a nearest neighbour. A frame whose
    day is not on the axis means the two came from different files, which is
    the one failure this join can have and it is not silent.
    """
    daily_axis = np.asarray(daily_axis, dtype="datetime64[ns]")
    ts = np.asarray(frame_timestamps, dtype="datetime64[ns]")
    assert ts.ndim == 1, f"expected (T_kept,) timestamps, got {ts.shape}"
    pos = np.searchsorted(daily_axis, ts)
    assert (pos >= 0).all() and (pos < daily_axis.size).all(), (
        f"{int(((pos < 0) | (pos >= daily_axis.size)).sum())} retained frame(s) "
        f"fall outside the cube's {daily_axis.size}-day axis "
        f"({daily_axis[0]} .. {daily_axis[-1]})"
    )
    bad = np.flatnonzero(daily_axis[pos] != ts)
    assert not bad.size, (
        f"{bad.size} retained frame(s) have no exact day on the cube's daily "
        f"axis, e.g. {ts[bad[0]]} nearest {daily_axis[pos[bad[0]]]}. The frames "
        "and the axis came from different files; nothing here rounds to the "
        "nearest day, because a weather value attached to the wrong day is a "
        "confident wrong number."
    )
    assert pos.shape == ts.shape
    return pos.astype(int)


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
    bbox = _pixel_bbox(cube_id)
    dominant, frac = landcover if landcover is not None else cube_landcover(sample.path)
    cell_lc, cell_purity = cube_grid_landcover(sample.path)
    cell_elev = cube_grid_elevation(sample.path)
    weather = cube_weather(sample.path)

    # E-OBS lives on the DAILY axis; sel.kept_idx counts ACQUISITIONS. Locate
    # each retained frame on the daily axis by its timestamp -- see the module
    # docstring for what indexing one axis with the other cost.
    daily_axis = cube_daily_axis(sample.path)
    for v, a in weather.items():
        assert a.shape == (daily_axis.size,), (
            f"{cube_id}: {v} has shape {a.shape} but the cube's daily axis is "
            f"{daily_axis.size} long; the E-OBS series and the time axis "
            "disagree and no index into them is trustworthy"
        )
    day_pos = frame_daily_positions(daily_axis, sel.timestamps)

    rows = []
    for i in range(sel.values.shape[0]):
        ts = sel.timestamps[i]
        rows.append({
            "cube_id": cube_id,
            "tile": tile,
            # Per-ROW calendar year, from THIS frame's own timestamp -- never
            # the cube id's window-start year. On a single-year cube the two
            # coincide and the distinction is invisible; on a seasonal cube
            # (one file spanning 2017-2020) they diverge on most rows, and
            # probes.cv's year-aware modes (year, crossed) refuse a manifest
            # where they disagree -- see probes.cv._row_years. Until
            # 2026-08-10 this column WAS the window-start year, which is
            # exactly the "lying year column" that guard exists to catch; it
            # had never been exercised against a real seasonal cube before.
            "year": int(np.datetime64(ts, "Y").astype(int) + 1970),
            "timestamp": ts,
            # The EMBEDDING join key: == encoders.frames' kept_idx. Counts
            # ACQUISITIONS, not days. Never re-point it.
            "original_axis_index": int(sel.kept_idx[i]),
            # The DAILY-axis position of the same frame, from its timestamp.
            # Horizons and weather windows are defined in DAYS on this axis.
            "daily_axis_index": int(day_pos[i]),
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
            # E-OBS on the ORIGINAL DAILY axis, indexed by daily_axis_index --
            # i.e. the weather of the day this frame was actually acquired on.
            # A windowed feature needs the whole series: import cube_weather and
            # cube_daily_axis, do not try to reconstruct a window from these
            # per-frame values, which are sampled only on acquisition days.
            **{v: float(a[day_pos[i]]) for v, a in weather.items()},
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
        print(f"[manifest] in-cube E-OBS joined on daily_axis_index -- the day the "
              f"frame was ACQUIRED ({len(eobs)}): {eobs}")
        off = (df.daily_axis_index - df.original_axis_index).to_numpy()
        print(f"[manifest] original_axis_index (acquisition axis) spans "
              f"0..{df.original_axis_index.max()}; daily_axis_index (DAILY axis) "
              f"spans 0..{df.daily_axis_index.max()}")
        print(f"[manifest] the two axes differ by {off.min()}..{off.max()} steps "
              f"(median {int(np.median(off))}) on {int((off != 0).sum())}/{len(df)} "
              "rows -- weather and horizons use daily_axis_index, the embedding "
              "join uses original_axis_index")
    return df


def assert_weather_join(df, cube_dir: str = None, verbose: bool = True) -> dict:
    """Re-derive the E-OBS join from the CUBES and refuse any disagreement.

    WHY THIS IS A SEPARATE FUNCTION AND NOT A COMMENT. The join it checks was
    wrong for a week and nothing noticed: the values were finite, in range, the
    right dtype and the right shape, and they were simply another day's weather.
    No assertion on the manifest alone can catch that, because the manifest is
    internally consistent either way. The only test is to go back to the file,
    look up the day by TIMESTAMP, and compare -- which is what this does, from
    the cube path each row already carries.

    Any probe whose input is the weather should call this before it fits
    anything. It reads coordinates and 1-D series only, no reflectance.
    """
    import pandas as pd   # noqa: F401  (df is a DataFrame; kept explicit)

    eobs = sorted(c for c in df.columns if c.startswith("eobs_"))
    assert eobs, "manifest carries no eobs_* columns; nothing to check"
    assert "daily_axis_index" in df.columns, (
        "manifest has no daily_axis_index, so it predates the timestamp-based "
        "E-OBS join (2026-08-10) and its weather columns are indexed by the "
        "ACQUISITION axis -- i.e. they hold another day's weather. Rebuild it "
        "with encoders.manifest.build_manifest."
    )
    cube_dir = cube_dir or os.path.join("data", "raw")
    n_checked, worst = 0, {}
    for cube_id, sub in df.groupby("cube_id"):
        path = os.path.join(cube_dir, str(cube_id))
        assert os.path.exists(path), (
            f"cannot verify the weather join for {cube_id!r}: no cube at {path}. "
            "Pass cube_dir=<the resolved data/raw> rather than skipping the check."
        )
        daily_axis = cube_daily_axis(path)
        weather = cube_weather(path)
        pos = frame_daily_positions(daily_axis, sub["timestamp"].to_numpy())
        stored = sub["daily_axis_index"].to_numpy().astype(int)
        assert (pos == stored).all(), (
            f"{cube_id}: daily_axis_index disagrees with the position derived "
            f"from the timestamp on {int((pos != stored).sum())}/{len(sub)} rows "
            f"(e.g. stored {stored[pos != stored][0]} vs derived "
            f"{pos[pos != stored][0]}). The manifest was built against a "
            "different cube file."
        )
        for v in eobs:
            got = sub[v].to_numpy().astype(float)
            want = weather[v][pos]
            d = float(np.abs(got - want).max())
            worst[v] = max(worst.get(v, 0.0), d)
            assert d <= 1e-9, (
                f"{cube_id}: {v} in the manifest differs from the in-cube series "
                f"at the frame's own day by up to {d:.4g}. This is the "
                "wrong-axis bug: the column is indexed by the acquisition axis "
                "instead of the daily axis. Rebuild the manifest."
            )
        n_checked += len(sub)
    out = {"n_rows": n_checked, "n_cubes": int(df.cube_id.nunique()),
           "variables": tuple(eobs), "max_abs_diff": worst}
    if verbose:
        print(f"[manifest] weather join VERIFIED against the cubes: "
              f"{n_checked} rows x {len(eobs)} E-OBS variables over "
              f"{out['n_cubes']} cubes, max abs difference "
              f"{max(worst.values()):.3g} (tolerance 1e-9)")
    return out


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
