"""Frame selection and input-integrity checks shared by every Tier A encoder.

Design rules baked in here:

* A frame survives iff its clear-fraction EXCEEDS 0.5 (strictly greater, the
  same comparison as ``data.loader.describe_cube``'s ``per_t_valid > 0.5``).
  A frame with ZERO valid pixels has clear-fraction 0.0 and is dropped by the
  same rule -- there is no separate code path for it, so there is no separate
  code path to get wrong.
* Surviving frames are fed to encoders UNMODIFIED. Nothing here inpaints,
  fills, clips or rescales; per-model radiometric preprocessing lives inside
  each wrapper and is printed there.
* The exact clear-fraction of every retained frame is returned alongside the
  frames, so later probes can filter more strictly WITHOUT re-encoding.
* Reflectance among VALID pixels must stay <= 1.2. Above that, bright cloud is
  leaking through the mask into encoder input and into NDVI, which voids every
  downstream number. The global max (masked pixels included) may legitimately
  reach ~2 (Phase 1.1 measured 1.98 on bright cloud); that is harmless only
  because those pixels are masked, which is exactly what this assertion checks.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

__all__ = [
    "MIN_CLEAR_FRACTION",
    "MAX_VALID_REFLECTANCE",
    "SelectedFrames",
    "clear_fraction",
    "select_clear_frames",
    "assert_valid_reflectance",
]

# The Phase 1.2 frame-selection threshold. Strictly greater-than.
MIN_CLEAR_FRACTION = 0.5

# Fail loudly if any VALID pixel exceeds this. Vegetation/soil/water BOA
# reflectance sits well below 1; only cloud (or an un-removed BOA offset,
# which the loader prints) gets close.
MAX_VALID_REFLECTANCE = 1.2


class SelectedFrames(NamedTuple):
    """The frames that survive the clear-fraction rule, plus their bookkeeping."""

    values: np.ndarray      # (T_kept, C, H, W) float32, UNMODIFIED reflectance
    timestamps: np.ndarray  # (T_kept,) datetime64[ns], strictly increasing
    mask: np.ndarray        # (T_kept, H, W) bool, True == VALID/clear
    clear_frac: np.ndarray  # (T_kept,) float64, exact clear-fraction per kept frame
    kept_idx: np.ndarray    # (T_kept,) int, index into the ORIGINAL cube time axis


def clear_fraction(mask) -> np.ndarray:
    """(T, H, W) bool valid-mask -> (T,) fraction of valid pixels per frame.

    A frame with zero valid pixels gets exactly 0.0 -- a plain mean over a
    boolean mask cannot produce NaN, so this is safe on all-masked frames.
    """
    mask = np.asarray(mask)
    assert mask.ndim == 3, f"mask must be (T, H, W), got {mask.shape}"
    assert mask.dtype == np.bool_, (
        f"mask must be bool with True == VALID (got dtype {mask.dtype}); "
        "convert integer codes with data.loader.valid_mask_from_codes first"
    )
    out = mask.mean(axis=(1, 2))
    assert out.shape == (mask.shape[0],)
    assert np.isfinite(out).all()
    return out


def select_clear_frames(
    values,
    timestamps,
    mask,
    min_clear: float = MIN_CLEAR_FRACTION,
    verbose: bool = True,
) -> SelectedFrames:
    """Drop frames whose clear-fraction is not strictly above ``min_clear``.

    Returns the surviving frames UNMODIFIED, with their exact clear-fractions
    and their indices into the original time axis. May legitimately return
    zero frames for a fully clouded cube; the encoder-side empty-batch
    assertion (encoders.base.FrozenEncoder.encode) is what keeps such a cube
    from reaching a model, and encoders.pipeline.encode_cube refuses it with
    the cube's name attached.
    """
    values = np.asarray(values)
    mask = np.asarray(mask)
    timestamps = np.asarray(timestamps)

    assert values.ndim == 4, f"values must be (T, C, H, W), got {values.shape}"
    T, C, H, W = values.shape
    assert mask.shape == (T, H, W), f"mask {mask.shape} incompatible with values {values.shape}"
    assert timestamps.shape == (T,), f"{timestamps.shape[0] if timestamps.ndim else 0} timestamps vs {T} frames"

    cf = clear_fraction(mask)
    keep = cf > min_clear
    kept_idx = np.flatnonzero(keep)
    n_zero = int((cf == 0.0).sum())

    sel = SelectedFrames(
        values=values[keep],
        timestamps=timestamps[keep],
        mask=mask[keep],
        clear_frac=cf[keep],
        kept_idx=kept_idx,
    )

    T_kept = sel.values.shape[0]
    assert sel.values.shape == (T_kept, C, H, W)
    assert sel.mask.shape == (T_kept, H, W)
    assert sel.timestamps.shape == sel.clear_frac.shape == sel.kept_idx.shape == (T_kept,)
    assert T_kept == int(keep.sum())
    if T_kept:
        assert sel.clear_frac.min() > min_clear
        assert np.all(np.diff(sel.timestamps) > np.timedelta64(0, "ns")), (
            "timestamps no longer strictly increasing after frame selection"
        )

    if verbose:
        print(f"[frames] clear-fraction rule (> {min_clear}): kept {T_kept}/{T} frames "
              f"(dropped {T - T_kept}, of which {n_zero} had ZERO valid pixels)")
        if T_kept:
            print(f"[frames] kept values {sel.values.shape} | mask {sel.mask.shape} | "
                  f"clear_frac {sel.clear_frac.shape} "
                  f"min={sel.clear_frac.min():.3f} median={np.median(sel.clear_frac):.3f} "
                  f"max={sel.clear_frac.max():.3f}")
        else:
            print("[frames] WARNING: nothing survived; this cube must be skipped, "
                  "the encoders will refuse an empty batch")
    return sel


def assert_valid_reflectance(
    values,
    mask,
    max_ok: float = MAX_VALID_REFLECTANCE,
    verbose: bool = True,
) -> float:
    """Max reflectance among VALID pixels only; fail loudly above ``max_ok``.

    Phase 1.1 measured a global max of 1.98 -- physically possible for bright
    cloud, harmless only if such pixels are masked out. So the global max is
    printed for context but the assertion runs on valid pixels alone: if it
    trips, bright cloud is leaking THROUGH the mask into the encoder input and
    into NDVI, and the mask is failing at exactly the pixels it exists for.
    """
    values = np.asarray(values)
    mask = np.asarray(mask)
    assert values.ndim == 4, f"values must be (T, C, H, W), got {values.shape}"
    assert mask.shape == (values.shape[0], values.shape[2], values.shape[3]), (
        f"mask {mask.shape} incompatible with values {values.shape}"
    )
    assert mask.dtype == np.bool_, f"mask must be bool, got {mask.dtype}"

    finite = np.isfinite(values)
    valid = mask[:, None, :, :] & finite
    assert valid.shape == values.shape
    assert valid.any(), "cube has no valid finite pixel at all; nothing to check"

    vmax = float(values[valid].max())
    global_max = float(values[finite].max()) if finite.any() else float("nan")
    if verbose:
        print(f"[frames] reflectance max: valid-pixels={vmax:.4f} (limit {max_ok}) | "
              f"all finite pixels={global_max:.4f} (bright cloud, fine if masked)")
    assert vmax <= max_ok, (
        f"valid-pixel reflectance max {vmax:.4f} exceeds {max_ok}: bright cloud is "
        "leaking THROUGH the mask into the encoder input and into NDVI. The mask "
        "is failing at exactly the pixels it exists for. Check that the "
        "s2_dlmask + SCL conjunction was applied (data.loader.greenearthnet_valid_mask) "
        "before trusting any embedding or NDVI number from this cube."
    )
    return vmax
