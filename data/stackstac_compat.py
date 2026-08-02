"""Make earthnet-minicuber 0.1.3 work with stackstac >= 0.5.

THE BUG
-------
`earthnet_minicuber/provider/s2/sentinel2.py` calls::

    stackstac.stack(items_s2, ..., dtype="float32", ...)

with neither `rescale` nor `fill_value`. stackstac 0.5.0 added three guards
that all ask the same malformed question -- they test the *type* of a value
rather than the value itself::

    prepare.py:167   if rescale and not np.can_cast(type(asset_scale), dtype)
    prepare.py:176   if rescale and not np.can_cast(type(asset_offset), dtype)
    to_dask.py:39    if not np.can_cast(type(fill_value), dtype)

With `dtype="float32"` and the defaults `asset_scale=1`, `fill_value=np.nan`:

    np.can_cast(int,   np.float32) -> False     (int   -> int64   -> unsafe)
    np.can_cast(float, np.float32) -> False     (float -> float64 -> unsafe)

so *every* float32 stack raises. Both are False on NumPy 1.26 and 2.x, so
this is a stackstac bug against a pre-0.5 caller, not a NumPy-version problem.
Downgrading NumPy does not help.

THE SHIM
--------
Two injections, neither of which changes a single number:

    rescale=False                 skips guards 1 and 2. Safe because the
                                     rescale would be the identity (below).
    fill_value=np.float32("nan")  passes guard 3, because
                                     np.can_cast(np.float32, np.float32) is
                                     True. Still NaN, just carrying float32
                                     instead of Python float.

WHY rescale=False IS SAFE HERE
------------------------------
Microsoft Planetary Computer's `sentinel-2-l2a` assets carry **no**
`raster:bands` field, so stackstac falls back to `scale=1, offset=0` and the
"rescaling" it refuses to perform is the identity y = 1*x + 0. Verified against
the live STAC API::

    B02/B04/B8A/SCL raster:bands = None   (scene 2020-07-30, baseline 02.12)

minicuber does its own scaling afterwards anyway (`/10000` for reflectance
bands, `/65535` for AOT/WVP), and removes the BOA offset itself via
`correct_processing_baseline`. So this shim changes no number.

The guard below re-checks that assumption on every call rather than trusting
it: if PC ever publishes a non-identity scale/offset, we raise instead of
silently dropping a real rescale.
"""

from __future__ import annotations

__all__ = ["apply"]

_NOTE = ("[stackstac_compat] forcing rescale=False (identity scaling on this "
         "collection) and fill_value=float32(nan), to work around "
         "stackstac>=0.5 + earthnet-minicuber 0.1.3")


def _asset_raster_bands(item, name):
    """raster:bands for one asset, for either a pystac Item or a plain dict."""
    try:
        assets = item.assets if hasattr(item, "assets") else item["assets"]
        asset = assets[name]
    except (AttributeError, KeyError, TypeError):
        return None
    if hasattr(asset, "extra_fields"):
        return asset.extra_fields.get("raster:bands")
    if isinstance(asset, dict):
        return asset.get("raster:bands")
    return None


def _assert_identity_scaling(items, assets, max_items: int = 5) -> None:
    """Raise if any asset declares a scale/offset that rescale=False would drop."""
    if not assets:
        return
    try:
        sample = list(items)[:max_items]
    except TypeError:
        return
    for item in sample:
        for name in assets:
            rb = _asset_raster_bands(item, name)
            if not rb:
                continue
            scale = rb[0].get("scale", 1)
            offset = rb[0].get("offset", 0)
            if scale != 1 or offset != 0:
                raise RuntimeError(
                    f"asset {name!r} declares scale={scale} offset={offset}, so "
                    "forcing rescale=False would silently return unscaled data. "
                    "The stackstac_compat shim is no longer safe for this "
                    "collection -- fix the scaling explicitly before continuing."
                )


def apply() -> None:
    """Idempotently patch `stackstac.stack` for pre-0.5 callers.

    Call before any earthnet-minicuber download. minicuber looks up
    `stackstac.stack` at call time, so patching the module attribute is enough.
    Explicit `rescale=` / `fill_value=` from a caller are always respected.
    """
    import numpy as np
    import stackstac

    if getattr(stackstac.stack, "_ccai_patched", False):
        return

    _orig = stackstac.stack

    def stack(items, *args, **kwargs):
        if "rescale" not in kwargs:
            _assert_identity_scaling(items, kwargs.get("assets"))
            kwargs["rescale"] = False
        if "fill_value" not in kwargs:
            dtype = np.dtype(kwargs.get("dtype", "float64"))
            # Same NaN, carried in the output dtype so stackstac's
            # can_cast(type(fill_value), dtype) guard passes.
            kwargs["fill_value"] = dtype.type("nan") if dtype.kind == "f" else np.nan
        return _orig(items, *args, **kwargs)

    stack._ccai_patched = True
    stack.__doc__ = _orig.__doc__
    stackstac.stack = stack
    print(_NOTE)
