"""Unit test for the canonical NDVI definition.

WRITTEN BEFORE data/ndvi.py EXISTS. Hand-computed by pencil, see below.

Toy 2x2, two bands (reflectance units, float32):

    B8A = [[0.40, 0.30],        B04 = [[0.10, 0.10],
           [0.50, 0.10]]               [0.25, 0.30]]

    valid mask (True == clear/usable):
          [[ True,  True],
           [ True, False]]     <- pixel (1,1) is cloud-masked

Hand-computed NDVI = (B8A - B04) / (B8A + B04):

    (0,0): (0.40 - 0.10) / (0.40 + 0.10) = 0.30 / 0.50 =  0.6
    (0,1): (0.30 - 0.10) / (0.30 + 0.10) = 0.20 / 0.40 =  0.5
    (1,0): (0.50 - 0.25) / (0.50 + 0.25) = 0.25 / 0.75 =  1/3 = 0.33333333...
    (1,1): (0.10 - 0.30) / (0.10 + 0.30) = -0.20 / 0.40 = -0.5  -> MUST BE NaN

Pixel (1,1) is deliberately given reflectances that produce a perfectly
finite NDVI (-0.5). If masking is applied late, in the wrong polarity, or
not at all, -0.5 leaks through and the test catches it.
"""

import numpy as np
import pytest

from data.ndvi import ndvi

B8A = np.array([[0.40, 0.30],
                [0.50, 0.10]], dtype=np.float32)
B04 = np.array([[0.10, 0.10],
                [0.25, 0.30]], dtype=np.float32)
VALID = np.array([[True, True],
                  [True, False]], dtype=bool)

EXPECTED_VALID = np.array([0.6, 0.5, 1.0 / 3.0], dtype=np.float64)


def test_shapes_are_preserved():
    out = ndvi(B8A, B04, VALID)
    print(f"[test] B8A {B8A.shape} B04 {B04.shape} mask {VALID.shape} -> ndvi {out.shape}")
    assert out.shape == (2, 2), f"expected (2, 2), got {out.shape}"
    assert np.issubdtype(out.dtype, np.floating), f"NDVI must be float, got {out.dtype}"


def test_valid_pixels_match_hand_computation():
    out = ndvi(B8A, B04, VALID)
    got = np.array([out[0, 0], out[0, 1], out[1, 0]], dtype=np.float64)
    print(f"[test] hand-computed {EXPECTED_VALID} vs ndvi() {got}")
    assert np.allclose(got, EXPECTED_VALID, rtol=1e-6, atol=1e-6), (
        f"valid NDVI mismatch: expected {EXPECTED_VALID}, got {got}"
    )


def test_masked_pixel_is_nan_and_does_not_leak():
    out = ndvi(B8A, B04, VALID)
    print(f"[test] masked pixel (1,1) -> {out[1, 1]!r} (unmasked value would be -0.5)")
    assert np.isnan(out[1, 1]), f"masked pixel must be NaN, got {out[1, 1]}"
    assert not np.isclose(np.nan_to_num(out[1, 1], nan=0.0), -0.5), (
        "unmasked NDVI leaked through the cloud mask"
    )
    assert np.isnan(out).sum() == 1, (
        f"exactly one NaN expected, got {int(np.isnan(out).sum())}"
    )


def test_mask_is_required_positionally():
    """ndvi() must never be callable without a mask."""
    with pytest.raises(TypeError):
        ndvi(B8A, B04)  # type: ignore[call-arg]


def test_integer_cloud_code_mask_is_rejected():
    """A raw SCL / s2_mask integer array is truthy almost everywhere and would
    silently disable masking. It must raise, not be coerced."""
    codes = np.array([[0, 0],
                      [0, 4]], dtype=np.uint8)
    with pytest.raises(AssertionError):
        ndvi(B8A, B04, codes)


def test_shape_mismatch_raises():
    with pytest.raises(AssertionError):
        ndvi(B8A, B04[:, :1], VALID)
    with pytest.raises(AssertionError):
        ndvi(B8A, B04, VALID[:1, :])


def test_zero_denominator_is_nan_not_inf():
    b8a = np.array([[0.0, 0.2]], dtype=np.float32)
    b04 = np.array([[0.0, 0.1]], dtype=np.float32)
    m = np.array([[True, True]], dtype=bool)
    out = ndvi(b8a, b04, m)
    print(f"[test] zero-denominator -> {out}")
    assert np.isnan(out[0, 0]), f"0/0 must be NaN, got {out[0, 0]}"
    assert np.isfinite(out[0, 1])


def test_nan_reflectance_propagates():
    b8a = np.array([[np.nan, 0.4]], dtype=np.float32)
    b04 = np.array([[0.1, 0.1]], dtype=np.float32)
    m = np.array([[True, True]], dtype=bool)
    out = ndvi(b8a, b04, m)
    assert np.isnan(out[0, 0])
    assert np.isclose(out[0, 1], 0.6, rtol=1e-6)


def test_time_dimension_broadcasts():
    """(T, H, W) stacks must work, with a per-timestep mask."""
    t = 3
    b8a = np.repeat(B8A[None], t, axis=0)
    b04 = np.repeat(B04[None], t, axis=0)
    m = np.repeat(VALID[None], t, axis=0)
    m[1] = False  # whole timestep clouded out
    out = ndvi(b8a, b04, m)
    print(f"[test] stacked input {b8a.shape} -> ndvi {out.shape}")
    assert out.shape == (t, 2, 2), f"expected {(t, 2, 2)}, got {out.shape}"
    assert np.isnan(out[1]).all(), "fully-clouded timestep must be all-NaN"
    assert np.isclose(out[0, 0, 0], 0.6, rtol=1e-6)


def test_scale_invariance_to_reflectance_units():
    """NDVI is a ratio: 0-1 floats and 0-10000 ints must agree.
    (This holds only if there is NO additive offset -- see loader harmonisation.)"""
    out_f = ndvi(B8A, B04, VALID)
    out_i = ndvi((B8A * 10000).astype(np.int16), (B04 * 10000).astype(np.int16), VALID)
    valid = ~np.isnan(out_f)
    print(f"[test] float NDVI {out_f[valid]} vs int16-scaled NDVI {out_i[valid]}")
    assert np.allclose(out_f[valid], out_i[valid], rtol=1e-4, atol=1e-4)
