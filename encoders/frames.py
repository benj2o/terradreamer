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

TWO DATA PROPERTIES OF GREENEARTHNET THAT THIS MODULE HANDLES EXPLICITLY,
both measured on all 20 cubes of tile 32UNU (see log.md, 2026-08-03):

1. A pixel can be BOTH mask-valid AND non-finite. 113 such pixels exist in
   the 264 retained frames (6.8e-6 of all pixels, in 6 frames). The published
   clear-sky conjunction is computed from the mask bands and does not consult
   the reflectance bands, so it can mark a no-data pixel "clear". A pixel with
   no reflectance value is not a usable observation, so ``finite_valid_mask``
   demotes it. This is exactly the convention ``data.ndvi.ndvi`` already
   applies internally (``usable = mask & isfinite(...)``); it is applied here
   too so the encoder path and the target path agree. Nothing is ever filled
   or invented: pixels only ever leave the valid set.

2. A handful of mask-valid pixels are physically implausible (reflectance
   > 1.2): 44 of 17,340,401 valid pixels, isolated singletons and 2-4 px
   clusters in 99.7-100% clear frames. That is NOT the systemic mask failure
   the check exists to catch, so ``assert_valid_reflectance`` fails on the
   PREVALENCE of such pixels, not on the single extreme value. See the module
   constant ``MAX_IMPLAUSIBLE_FRACTION`` for the margin.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

__all__ = [
    "MIN_CLEAR_FRACTION",
    "MAX_VALID_REFLECTANCE",
    "MAX_IMPLAUSIBLE_FRACTION",
    "SelectedFrames",
    "ReflectanceReport",
    "clear_fraction",
    "finite_valid_mask",
    "select_clear_frames",
    "assert_valid_reflectance",
]

# The Phase 1.2 frame-selection threshold. Strictly greater-than.
MIN_CLEAR_FRACTION = 0.5

# Reflectance above this is not physically plausible for a surface pixel.
# Vegetation, soil, water and snow all sit well below it; only cloud (or an
# un-removed BOA offset, which the loader prints) gets close.
MAX_VALID_REFLECTANCE = 1.2

# Tolerated PREVALENCE of implausible valid pixels, as a fraction of valid
# pixels. The check exists to catch a systemic mask failure -- cloud passing
# as clear over real area. Measured on tile 32UNU: the worst single cube is
# 1.9e-5 and the 20-cube aggregate is 2.5e-6, all isolated pixels. The
# smallest systemic failure worth the name -- one fully-clouded frame of a
# 13-frame cube passing as clear -- would be ~7.7e-2, about 770x this
# tolerance. The decision boundary is therefore well separated in both
# directions: speckle passes, a real leak cannot.
MAX_IMPLAUSIBLE_FRACTION = 1e-4


class SelectedFrames(NamedTuple):
    """The frames that survive the clear-fraction rule, plus their bookkeeping."""

    values: np.ndarray      # (T_kept, C, H, W) float32, UNMODIFIED reflectance
    timestamps: np.ndarray  # (T_kept,) datetime64[ns], strictly increasing
    mask: np.ndarray        # (T_kept, H, W) bool, True == VALID and finite
    clear_frac: np.ndarray  # (T_kept,) float64, exact clear-fraction per kept frame
    kept_idx: np.ndarray    # (T_kept,) int, index into the ORIGINAL cube time axis


class ReflectanceReport(NamedTuple):
    """What ``assert_valid_reflectance`` measured, so callers can print it."""

    valid_max: float        # max reflectance among valid pixels
    global_max: float       # max over all finite pixels, masked ones included
    n_implausible: int      # valid pixels above MAX_VALID_REFLECTANCE
    n_valid: int            # valid pixels considered
    fraction: float         # n_implausible / n_valid


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


def finite_valid_mask(values, mask, verbose: bool = True) -> tuple:
    """``mask`` AND "every band of this pixel is finite". Returns (mask, n_demoted).

    The published clear-sky conjunction (data.loader.greenearthnet_valid_mask)
    reads s2_dlmask and s2_SCL; it never looks at the reflectance bands, so it
    can mark a no-data pixel clear. A pixel with no reflectance value is not a
    usable observation of anything.

    This only ever REMOVES pixels from the valid set. It does not fill, it does
    not inpaint, and it does not touch the reflectance array. It is the same
    rule data.ndvi.ndvi already applies internally, applied here so the encoder
    input and the target agree about which pixels exist.
    """
    values = np.asarray(values)
    mask = np.asarray(mask)
    assert values.ndim == 4, f"values must be (T, C, H, W), got {values.shape}"
    assert mask.shape == (values.shape[0], values.shape[2], values.shape[3]), (
        f"mask {mask.shape} incompatible with values {values.shape}"
    )
    assert mask.dtype == np.bool_, f"mask must be bool, got {mask.dtype}"

    finite_all = np.isfinite(values).all(axis=1)  # (T, H, W)
    out = mask & finite_all
    n_demoted = int((mask & ~finite_all).sum())
    assert out.shape == mask.shape and out.dtype == np.bool_
    assert out.sum() <= mask.sum(), "the finite rule may only remove valid pixels"

    if verbose and n_demoted:
        print(f"[frames] {n_demoted} pixel(s) were mask-valid but non-finite "
              f"({n_demoted / max(int(mask.sum()), 1):.2e} of valid pixels) and are now "
              "masked. Nothing was filled: a pixel with no reflectance value is "
              "not an observation (same rule as data.ndvi.ndvi).")
    return out, n_demoted


def select_clear_frames(
    values,
    timestamps,
    mask,
    min_clear: float = MIN_CLEAR_FRACTION,
    verbose: bool = True,
) -> SelectedFrames:
    """Drop frames whose clear-fraction is not strictly above ``min_clear``.

    The mask is first made self-consistent with the reflectance (see
    ``finite_valid_mask``), then the clear-fraction is computed from it, so a
    no-data pixel can never be counted as evidence that a frame is clear.

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
    assert timestamps.shape == (T,), (
        f"{timestamps.shape[0] if timestamps.ndim else 0} timestamps vs {T} frames"
    )

    mask, _n_demoted = finite_valid_mask(values, mask, verbose=verbose)

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
        # Every kept frame must carry at least one usable pixel per band, or the
        # per-band statistics downstream would be reductions over nothing.
        assert np.isfinite(sel.values).any(axis=(2, 3)).all(), (
            "a retained frame has a band with no finite pixel at all"
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
    max_fraction: float = MAX_IMPLAUSIBLE_FRACTION,
    verbose: bool = True,
) -> ReflectanceReport:
    """Fail if physically implausible reflectance is PREVALENT among valid pixels.

    Phase 1.1 measured a global max of 1.98 -- physically possible for bright
    cloud, harmless only if such pixels are masked out. So the global max is
    reported for context while the check runs on valid pixels alone.

    What the check asserts, and why it is a prevalence and not a maximum: the
    failure this exists to catch is cloud passing as clear over real area,
    which would put a large, contiguous population of implausible pixels into
    the encoder input and into NDVI. A maximum over ~17 million pixels is the
    most outlier-sensitive statistic available, and on tile 32UNU it trips on
    44 isolated pixels (2.5e-6) that sit in 99.7-100% clear frames and move no
    downstream number by more than 4e-4 relative. Those are bright specular
    targets and sub-pixel anomalies, not a broken mask. The fraction separates
    the two cases by roughly three orders of magnitude.
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

    n_valid = int(valid.sum())
    over = valid & (values > max_ok)
    n_over = int(over.sum())
    frac = n_over / n_valid
    vmax = float(values[valid].max())
    global_max = float(values[finite].max()) if finite.any() else float("nan")

    rep = ReflectanceReport(valid_max=vmax, global_max=global_max, n_implausible=n_over,
                            n_valid=n_valid, fraction=frac)

    if verbose:
        print(f"[frames] reflectance: valid max={vmax:.4f} | all-finite max={global_max:.4f} "
              f"(bright cloud, fine if masked)")
        print(f"[frames] implausible valid pixels (> {max_ok}): {n_over}/{n_valid} "
              f"= {frac:.2e}  (tolerance {max_fraction:.0e})")

    assert frac <= max_fraction, (
        f"implausible valid pixels (> {max_ok}) are {frac:.2e} of valid pixels "
        f"({n_over}/{n_valid}), above the {max_fraction:.0e} tolerance. At this "
        "prevalence it is not isolated bright targets: bright cloud is leaking "
        "THROUGH the mask into the encoder input and into NDVI, and the mask is "
        "failing at exactly the pixels it exists for. Check that the s2_dlmask + "
        "SCL conjunction was applied (data.loader.greenearthnet_valid_mask) "
        "before trusting any embedding or NDVI number from this cube."
    )
    return rep
