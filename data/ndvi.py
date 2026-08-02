"""THE single definition of the project's target variable.

This module contains EXACTLY ONE function. Do not copy it, do not inline the
formula, do not write `(nir - red) / (nir + red)` anywhere else in this repo.
Import it:

    from data.ndvi import ndvi

Every NDVI number that appears in a figure, a table, or a probe must have come
through this function.
"""

from __future__ import annotations

import os

import numpy as np

# Denominator floor. Below this the ratio is numerically meaningless and we
# return NaN rather than +-inf or a huge value that would poison downstream means.
_DEN_EPS = 1e-12

# Set NDVI_VERBOSE=1 to have every call print its array shapes (the project
# convention is that shapes are printed and asserted; this is the opt-in switch
# so per-timestep calls do not spam the log).
_VERBOSE = os.environ.get("NDVI_VERBOSE", "0") == "1"

__all__ = ["ndvi"]


def ndvi(b8a, b04, mask):
    """Cloud-masked NDVI = (B8A - B04) / (B8A + B04).

    The mask is applied PER-PIXEL, BEFORE any aggregation: masked pixels come
    back as NaN and are therefore invisible to every downstream mean, quantile,
    or model input. There is no code path that returns unmasked values.

    Parameters
    ----------
    b8a : array_like
        Sentinel-2 band B8A (NIR narrow, 20 m). Any shape, e.g. (H, W),
        (T, H, W) or (T, 1, H, W). Reflectance units are arbitrary but must be
        IDENTICAL to b04 and must be offset-free (see note below).
    b04 : array_like
        Sentinel-2 band B04 (red, 10 m resampled to 20 m). Same shape as b8a.
    mask : array_like of bool
        REQUIRED. Same shape as b8a. ``True == VALID / clear`` observation,
        ``False == cloud, shadow, snow, no-data, out-of-swath``. Integer cloud
        codes (Sentinel-2 SCL, earthnet-minicuber ``s2_mask``) are rejected --
        convert them first with ``data.loader.valid_mask_from_codes``.

    Returns
    -------
    np.ndarray, float32, same shape as the inputs
        NDVI in [-1, 1] for physically valid reflectances; NaN wherever the
        mask is False, wherever an input is non-finite, or wherever
        |B8A + B04| < 1e-12.

    Notes
    -----
    NDVI is invariant to a multiplicative scale (0-1 floats vs 0-10000 int16)
    but NOT to an additive offset. Sentinel-2 processing baseline >= 04.00
    carries BOA_ADD_OFFSET = -1000; if that offset has not been removed
    upstream, this function returns confidently wrong numbers. The loader
    prints reflectance min/max so the offset is visible.

    The output is deliberately NOT clipped to [-1, 1]: clipping would hide
    exactly that offset bug.
    """
    b8a = np.asarray(b8a)
    b04 = np.asarray(b04)
    mask = np.asarray(mask)

    if _VERBOSE:
        print(f"[ndvi] b8a {b8a.shape} {b8a.dtype} | b04 {b04.shape} {b04.dtype} "
              f"| mask {mask.shape} {mask.dtype}")

    assert b8a.shape == b04.shape, (
        f"band shape mismatch: b8a {b8a.shape} != b04 {b04.shape}"
    )
    assert mask.shape == b8a.shape, (
        f"mask shape {mask.shape} must equal band shape {b8a.shape} "
        "(no broadcasting: an accidentally broadcast mask silently un-masks pixels)"
    )
    assert mask.dtype == np.bool_, (
        f"mask must be a boolean array with True == VALID/clear; got dtype "
        f"{mask.dtype}. An integer cloud-code array (SCL / s2_mask) is truthy "
        "almost everywhere and would silently disable masking -- convert it "
        "with data.loader.valid_mask_from_codes(codes) first."
    )
    assert b8a.ndim >= 2, f"expected at least 2 spatial dims, got shape {b8a.shape}"

    nir = b8a.astype(np.float64, copy=False)
    red = b04.astype(np.float64, copy=False)

    den = nir + red
    num = nir - red

    # Pre-filled with NaN, then written ONLY where the observation is usable.
    # Masked pixels are never computed, so no unmasked value can survive.
    out = np.full(den.shape, np.nan, dtype=np.float64)
    usable = mask & np.isfinite(den) & np.isfinite(num) & (np.abs(den) > _DEN_EPS)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(num, den, out=out, where=usable)

    out = out.astype(np.float32)
    assert out.shape == b8a.shape, f"output shape {out.shape} != input shape {b8a.shape}"
    return out
