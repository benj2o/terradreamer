"""Minicube loader: yields (values, timestamps, mask) triplets per cube.

Design rules baked in here:

* The time axis is NOT assumed regular. Nominal Sentinel-2 cadence over
  Bavaria is ~5-daily (two orbits merged), but cloud gaps, orbit edges and
  the L2A availability record make the realised series irregular. Timesteps
  with no acquisition at all are dropped; the returned ``timestamps`` array
  is the ground truth and must be carried everywhere alongside the values.
* Cloud codes are converted to a boolean valid-mask exactly once, here, in
  ``valid_mask_from_codes``. Nothing downstream ever sees the integer codes.
* Every array's shape is printed and asserted.
* NDVI is never computed here -- it is imported from data.ndvi.
"""

from __future__ import annotations

import glob
import os
from typing import Iterator, NamedTuple, Sequence

import numpy as np
import xarray as xr

from data.ndvi import ndvi

__all__ = [
    "S2_BANDS",
    "DLMASK_CLEAR_CODES",
    "S2_MASK_CLEAR_CODES",
    "GEN_SCL_ALLOWED",
    "SCL_CLEAR_CODES",
    "greenearthnet_valid_mask",
    "CubeSample",
    "valid_mask_from_codes",
    "load_cube",
    "iter_cubes",
    "assert_no_overlap",
    "cube_ndvi",
    "describe_cube",
]

# Channel order. Fixed project-wide: index 2 is B04 (red), index 3 is B8A (NIR).
S2_BANDS: tuple = ("B02", "B03", "B04", "B8A")

# `s2_dlmask`, the CloudSEN12-trained MobileNetV2/UNet mask that is
# GreenEarthNet's stated contribution (Benson et al., CVPR 2024). From the
# variable's own attrs: 0 clear sky, 1 thick cloud, 2 thin cloud, 3 cloud shadow.
# THIS IS THE PROJECT'S MASK.
DLMASK_CLEAR_CODES: tuple = (0,)

# `s2_mask`, attrs: "sen2flux Cloud Mask". 0 free sky, 1 cloud, 2 cloud shadows,
# 3 snow, 4 masked by SCL. Sen2Cor lineage, the EarthNet2021 mask GreenEarthNet
# supersedes. Fallback only, and loudly.
S2_MASK_CLEAR_CODES: tuple = (0,)

# GreenEarthNet's published clear-sky definition is a CONJUNCTION, not the
# dlmask alone: model_pixelwise/climatology.py keeps pixels where
# (s2_dlmask < 1) AND s2_SCL is in this allow-list. Dropping the SCL half would
# leave numbers that are not comparable with published baselines, which is the
# whole reason for preferring s2_dlmask. SCL 3 (shadow), 8/9/10 (cloud) and
# 11 (snow) are excluded, which is also how snow gets masked: s2_dlmask has no
# snow class of its own.
GEN_SCL_ALLOWED: tuple = (1, 2, 4, 5, 6, 7)

# Raw Sentinel-2 L2A scene classification, if that is all a cube carries.
#   4 = vegetation, 5 = bare soil, 6 = water.  (3 shadow, 8/9/10 cloud, 11 snow)
SCL_CLEAR_CODES: tuple = (4, 5, 6)

_BAND_NAME_CANDIDATES = ("{b}", "s2_{b}", "S2_{b}", "{b}_20m", "sen2{b}")
# Preference order matters: s2_dlmask first.
_MASK_NAME_CANDIDATES = ("s2_dlmask", "s2_mask", "mask", "cloudmask_en", "s2_cloudmask")
_SCL_NAME_CANDIDATES = ("s2_SCL", "SCL", "scl")


class CubeSample(NamedTuple):
    """One minicube. ``values`` and ``mask`` share (T, H, W); values adds bands."""

    values: np.ndarray  # (T, C, H, W) float32, band order == S2_BANDS
    timestamps: np.ndarray  # (T,) datetime64[ns], strictly increasing, irregular
    mask: np.ndarray  # (T, H, W) bool, True == VALID/clear
    path: str
    bands: tuple


def valid_mask_from_codes(codes, clear_codes: Sequence[int] = S2_MASK_CLEAR_CODES) -> np.ndarray:
    """Integer cloud codes -> boolean valid-mask (True == clear/usable).

    This is the ONLY place in the project where mask polarity is decided.
    NaN / non-finite codes are treated as invalid.
    """
    codes = np.asarray(codes)
    assert codes.ndim >= 2, f"expected at least (H, W), got shape {codes.shape}"
    finite = np.isfinite(codes) if np.issubdtype(codes.dtype, np.floating) else np.ones(codes.shape, dtype=bool)
    valid = np.zeros(codes.shape, dtype=bool)
    for c in clear_codes:
        valid |= codes == c
    valid &= finite
    assert valid.dtype == np.bool_ and valid.shape == codes.shape
    return valid


def greenearthnet_valid_mask(dlmask, scl=None) -> np.ndarray:
    """GreenEarthNet's published clear-sky definition, as a boolean valid-mask.

    (s2_dlmask == 0) AND (s2_SCL in GEN_SCL_ALLOWED), after
    vitusbenson/greenearthnet model_pixelwise/climatology.py, which keeps
    pixels where `(s2_dlmask < 1)` and `s2_SCL` is in `[1, 2, 4, 5, 6, 7]`.

    Polarity is still decided only in valid_mask_from_codes; this composes two
    of its results. Passing scl=None gives the dlmask alone, which is NOT the
    published definition and is only for ablation.
    """
    valid = valid_mask_from_codes(dlmask, DLMASK_CLEAR_CODES)
    if scl is not None:
        valid &= valid_mask_from_codes(scl, GEN_SCL_ALLOWED)
    assert valid.dtype == np.bool_
    return valid


def _resolve(ds: xr.Dataset, candidates: Sequence[str], band: str = "") -> str:
    for pattern in candidates:
        name = pattern.format(b=band)
        if name in ds.variables:
            return name
    raise KeyError(
        f"could not find variable for {band or candidates!r} in cube; "
        f"available variables: {sorted(map(str, ds.variables))}"
    )


def _spatial_dims(da: xr.DataArray) -> tuple:
    for y, x in (("lat", "lon"), ("y", "x"), ("latitude", "longitude")):
        if y in da.dims and x in da.dims:
            return y, x
    raise KeyError(f"no recognised spatial dims in {da.dims}")


def load_cube(
    path: str,
    bands: Sequence[str] = S2_BANDS,
    drop_empty_timesteps: bool = True,
    verbose: bool = True,
    scl_conjunction: bool = True,
) -> CubeSample:
    """Load one minicube from disk into (values, timestamps, mask).

    The mask is `s2_dlmask` AND the `s2_SCL` allow-list, which is
    GreenEarthNet's published clear-sky definition. Set `scl_conjunction=False`
    to ablate the SCL half; the result is then not comparable with published
    baselines.
    """
    engine = "zarr" if path.endswith(".zarr") else None
    ds = xr.open_dataset(path, engine=engine) if engine is None else xr.open_zarr(path)

    band_vars = [_resolve(ds, _BAND_NAME_CANDIDATES, b) for b in bands]
    try:
        mask_var = _resolve(ds, _MASK_NAME_CANDIDATES)
        clear_codes = DLMASK_CLEAR_CODES if mask_var == "s2_dlmask" else S2_MASK_CLEAR_CODES
    except KeyError:
        mask_var = _resolve(ds, _SCL_NAME_CANDIDATES)
        clear_codes = SCL_CLEAR_CODES
        print(f"[loader] WARNING: no cloud mask in {os.path.basename(path)}, "
              f"falling back to {mask_var} with clear codes {clear_codes}")

    if mask_var != "s2_dlmask":
        print(f"[loader] " + "!" * 62)
        print(f"[loader] ! WARNING: using {mask_var!r}, NOT s2_dlmask.")
        print(f"[loader] ! s2_mask is the sen2flux/Sen2Cor-lineage mask that")
        print(f"[loader] ! GreenEarthNet supersedes (Benson et al., CVPR 2024).")
        print(f"[loader] ! Numbers from this cube are NOT comparable with")
        print(f"[loader] ! published benchmark results.")
        print(f"[loader] " + "!" * 62)

    ydim, xdim = _spatial_dims(ds[band_vars[0]])

    # --- time axis: sort, de-duplicate, never assume a regular grid ----------
    ds = ds.sortby("time")
    t_raw = ds["time"].values.astype("datetime64[ns]")
    _, keep_idx = np.unique(t_raw, return_index=True)
    if keep_idx.size != t_raw.size:
        print(f"[loader] dropped {t_raw.size - keep_idx.size} duplicate timestamps")
        ds = ds.isel(time=np.sort(keep_idx))

    stack = [ds[v].transpose("time", ydim, xdim).values.astype(np.float32) for v in band_vars]
    values = np.stack(stack, axis=1)  # (T, C, H, W)
    codes = ds[mask_var].transpose("time", ydim, xdim).values
    timestamps = ds["time"].values.astype("datetime64[ns]")

    scl = None
    if mask_var == "s2_dlmask" and scl_conjunction:
        try:
            scl_var = _resolve(ds, _SCL_NAME_CANDIDATES)
            scl = ds[scl_var].transpose("time", ydim, xdim).values
        except KeyError:
            print(f"[loader] WARNING: {os.path.basename(path)} has s2_dlmask but no "
                  "s2_SCL, so the published conjunction cannot be applied.")
    ds.close()

    if mask_var == "s2_dlmask":
        mask = greenearthnet_valid_mask(codes, scl)
    else:
        mask = valid_mask_from_codes(codes, clear_codes)

    assert values.ndim == 4, f"values must be (T, C, H, W), got {values.shape}"
    assert values.shape[1] == len(bands), f"band axis {values.shape[1]} != {len(bands)}"
    assert mask.shape == (values.shape[0], values.shape[2], values.shape[3]), (
        f"mask {mask.shape} incompatible with values {values.shape}"
    )
    assert timestamps.shape[0] == values.shape[0], (
        f"{timestamps.shape[0]} timestamps vs {values.shape[0]} timesteps"
    )

    if drop_empty_timesteps:
        acquired = np.isfinite(values).any(axis=(1, 2, 3))  # (T,)
        n_drop = int((~acquired).sum())
        if n_drop:
            print(f"[loader] dropping {n_drop}/{acquired.size} timesteps with no acquisition")
        values, mask, timestamps = values[acquired], mask[acquired], timestamps[acquired]

    assert timestamps.size > 0, f"{path}: no usable timesteps left"
    assert np.all(np.diff(timestamps) > np.timedelta64(0, "ns")), (
        "timestamps must be strictly increasing after sort+dedupe"
    )

    sample = CubeSample(values, timestamps, mask, path, tuple(bands))
    if verbose:
        print(
            f"[loader] {os.path.basename(path)} | values {sample.values.shape} "
            f"{sample.values.dtype} | timestamps {sample.timestamps.shape} "
            f"{sample.timestamps.dtype} | mask {sample.mask.shape} {sample.mask.dtype}"
        )
    return sample


def iter_cubes(
    root: str,
    limit: int = 20,
    pattern: str = "*.nc",
    bands: Sequence[str] = S2_BANDS,
    verbose: bool = True,
) -> Iterator[CubeSample]:
    """Yield (values, timestamps, mask) triplets for up to ``limit`` cubes."""
    paths = sorted(glob.glob(os.path.join(root, pattern)))
    if not paths:
        paths = sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))
    assert paths, (
        f"no cubes matching {pattern!r} under {root}.\n"
        "The download step produced nothing. This cell is not the bug, it is "
        "the messenger. Run the diagnostic and read the FIRST failure:\n"
        "    python -m data.diagnose"
    )
    print(f"[loader] found {len(paths)} cube(s) under {root}, using {min(limit, len(paths))}")
    for p in paths[:limit]:
        yield load_cube(p, bands=bands, verbose=verbose)


def assert_no_overlap(root: str, pattern: str = "*.nc", verbose: bool = True) -> list:
    """Fail if any two cubes on disk share pixels.

    Overlapping cubes put identical pixels on both sides of whatever split
    probes/cv.py draws later, which is leakage that no amount of careful CV can
    undo. Reads only the coordinate arrays, not the data.
    """
    paths = sorted(glob.glob(os.path.join(root, pattern)))
    if not paths:
        paths = sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))
    assert paths, f"no cubes matching {pattern!r} under {root}"

    cubes = []
    for p in paths:
        with xr.open_dataset(p) as ds:
            ydim, xdim = ("lat", "lon") if "lat" in ds.coords else ("y", "x")
            cubes.append((os.path.basename(p), ds[xdim].values, ds[ydim].values))

    # Exact test, not bounding boxes. Cubes are gridded in UTM but carry lat/lon
    # coordinates, so the grid is slightly skewed relative to lat/lon and two
    # merely ADJACENT cubes have lat/lon bounding boxes that overlap by a pixel
    # or two while sharing no pixel at all. Comparing coordinate values settles
    # it: a shared pixel means a shared coordinate on both axes.
    overlaps = []
    for i in range(len(cubes)):
        for j in range(i + 1, len(cubes)):
            ni, xi, yi = cubes[i]
            nj, xj, yj = cubes[j]
            nx = np.intersect1d(xi, xj).size
            ny = np.intersect1d(yi, yj).size
            if nx and ny:
                overlaps.append((ni, nj, nx * ny))

    if verbose:
        print(f"[loader] overlap check: {len(cubes)} cubes, "
              f"{len(overlaps)} sharing pixels")
    assert not overlaps, (
        f"{len(overlaps)} overlapping cube pair(s). Identical pixels would end "
        f"up on both sides of a split:\n" +
        "\n".join(f"  {a}  <->  {b}  ({n} shared pixels)" for a, b, n in overlaps[:10]) +
        "\nDelete the stray cube(s) and re-run. A cube downloaded with an older "
        "--n is the usual cause."
    )
    return paths


def cube_ndvi(sample: CubeSample) -> np.ndarray:
    """(T, H, W) masked NDVI for a cube, via the canonical data.ndvi.ndvi."""
    b04_i = sample.bands.index("B04")
    b8a_i = sample.bands.index("B8A")
    out = ndvi(sample.values[:, b8a_i], sample.values[:, b04_i], sample.mask)
    assert out.shape == sample.mask.shape, f"{out.shape} != {sample.mask.shape}"
    return out


def describe_cube(sample: CubeSample) -> dict:
    """Print and return the per-cube diagnostics that gate this phase."""
    T, C, H, W = sample.values.shape
    name = os.path.basename(sample.path)

    valid_frac = float(sample.mask.mean())
    per_t_valid = sample.mask.reshape(T, -1).mean(axis=1)
    n_usable_steps = int((per_t_valid > 0.5).sum())

    dt_days = np.diff(sample.timestamps) / np.timedelta64(1, "D")
    refl = sample.values[np.isfinite(sample.values)]
    nd = cube_ndvi(sample)
    nd_valid = nd[np.isfinite(nd)]

    print(f"[cube] {name}")
    print(f"       values {sample.values.shape} | mask {sample.mask.shape} | timestamps {sample.timestamps.shape}")
    print(f"       time   {sample.timestamps[0]} -> {sample.timestamps[-1]}  (T={T})")
    print(f"       dt/days min={dt_days.min():.1f} median={np.median(dt_days):.1f} "
          f"max={dt_days.max():.1f} mean={dt_days.mean():.2f}  (nominal 5.0, irregular by design)")
    print(f"       CLOUD-MASK: valid fraction {valid_frac:.3f} "
          f"(masked {1 - valid_frac:.3f}) | timesteps >50% clear: {n_usable_steps}/{T}")
    print(f"       reflectance min={refl.min():.4f} max={refl.max():.4f} "
          f"(expect ~[0, 1]; negatives => BOA offset not removed)")
    print(f"       NDVI valid {nd_valid.size}/{nd.size} ({nd_valid.size / nd.size:.3f}) "
          f"min={nd_valid.min():.3f} median={np.median(nd_valid):.3f} max={nd_valid.max():.3f}")

    return {
        "name": name,
        "T": T, "C": C, "H": H, "W": W,
        "t_start": sample.timestamps[0], "t_end": sample.timestamps[-1],
        "valid_fraction": valid_frac,
        "masked_fraction": 1.0 - valid_frac,
        "n_steps_gt50pct_clear": n_usable_steps,
        "dt_days_median": float(np.median(dt_days)),
        "dt_days_max": float(dt_days.max()),
        "refl_min": float(refl.min()), "refl_max": float(refl.max()),
        "ndvi_valid_fraction": float(nd_valid.size / nd.size),
        "ndvi_median": float(np.median(nd_valid)),
    }
