"""Phase 1.2 encoder tests.

Everything that can run without pretrained weights runs always: frame
selection, the valid-reflectance assertion, the frozen/batched/asserted
machinery (exercised through a tiny dummy network), the raw-feature baseline,
and the cube pipeline. The three real wrappers download hundreds of MB of
weights, so they run only when PHASE1_2_WEIGHTS=1 is set; the Phase 1.2
notebook exercises them for real on all 20 cubes either way.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from data.loader import CubeSample, S2_BANDS
from data.ndvi import ndvi
from encoders import build_encoder
from encoders.base import FrozenEncoder
from encoders.frames import (
    assert_valid_reflectance,
    clear_fraction,
    select_clear_frames,
)
from encoders.pipeline import encode_cube, load_encoded, save_encoded
from encoders.raw_features import (
    NDVI_FEATURE_SLICE,
    RAW_FEATURE_NAMES,
    RawFeatureBaseline,
)


# --------------------------------------------------------------------- helpers
class _DummyEncoder(FrozenEncoder):
    """Smallest possible network wrapper: exists to test the FrozenEncoder
    machinery (freezing, batching, shape guards), not to embed anything."""

    name = "dummy"
    embed_dim = 6

    def _build(self):
        torch.manual_seed(0)
        return torch.nn.Linear(len(S2_BANDS), self.embed_dim)

    def _preprocessing_lines(self):
        return ["global average pool over H, W", "fixed-seed linear map C -> 6"]

    def _encode_batch(self, frames, mask):
        x = frames.mean(dim=(2, 3)).to(self.device)
        return self._model(x)


def _synthetic(T=6, H=12, W=12, seed=0):
    """Frames whose clear-fractions step 0, 1/(T-1), ..., 1 exactly."""
    rng = np.random.default_rng(seed)
    values = rng.uniform(0.02, 0.45, (T, len(S2_BANDS), H, W)).astype(np.float32)
    values[:, S2_BANDS.index("B8A")] += 0.25  # vegetation-like: NDVI > 0
    mask = np.zeros((T, H, W), dtype=bool)
    for t in range(T):
        n = round(t / (T - 1) * H * W)
        mask[t].flat[:n] = True
    ts = (np.datetime64("2018-04-01") + np.arange(T) * np.timedelta64(5, "D"))
    return values, ts.astype("datetime64[ns]"), mask


def _sample(T=6, H=12, W=12, seed=0):
    values, ts, mask = _synthetic(T, H, W, seed)
    return CubeSample(values, ts, mask, path="/nowhere/synthetic_cube.nc", bands=S2_BANDS)


@pytest.fixture(scope="module")
def dummy():
    return _DummyEncoder(device="cpu", verbose=False)


@pytest.fixture(scope="module")
def raw():
    return RawFeatureBaseline(device="cpu", verbose=False)


# ------------------------------------------------------------- frame selection
def test_clear_fraction_matches_independent_computation():
    rng = np.random.default_rng(1)
    mask = rng.random((7, 9, 11)) > 0.5
    got = clear_fraction(mask)
    independent = np.array([np.count_nonzero(mask[t]) / mask[t].size for t in range(7)])
    print(f"[test] clear_fraction {got.shape}: {np.round(got, 3)}")
    np.testing.assert_allclose(got, independent)


def test_select_keeps_only_frames_strictly_above_half():
    values, ts, mask = _synthetic(T=6)  # fractions 0, .2, .4, .6, .8, 1
    sel = select_clear_frames(values, ts, mask)
    assert sel.values.shape[0] == 3, "0.6, 0.8 and 1.0 pass; 0.4 and below do not"


def test_select_records_exact_clear_fraction_and_indices():
    values, ts, mask = _synthetic(T=6)
    sel = select_clear_frames(values, ts, mask)
    cf_all = clear_fraction(mask)
    np.testing.assert_array_equal(sel.kept_idx, np.flatnonzero(cf_all > 0.5))
    np.testing.assert_allclose(sel.clear_frac, cf_all[sel.kept_idx])
    np.testing.assert_array_equal(sel.timestamps, ts[sel.kept_idx])
    np.testing.assert_array_equal(sel.values, values[sel.kept_idx])


def test_select_boundary_is_strict():
    """A frame at exactly 0.5 clear is DROPPED: the rule is 'exceeds', not 'meets'."""
    values, ts, mask = _synthetic(T=3, H=10, W=10)
    mask[:] = False
    mask[0].flat[:50] = True   # exactly 0.50
    mask[1].flat[:51] = True   # 0.51
    mask[2].flat[:100] = True  # 1.00
    sel = select_clear_frames(values, ts, mask)
    assert sel.kept_idx.tolist() == [1, 2]
    assert sel.clear_frac.tolist() == [0.51, 1.0]


def test_all_masked_frame_is_dropped_by_the_same_rule():
    values, ts, mask = _synthetic(T=6)
    assert clear_fraction(mask)[0] == 0.0, "frame 0 must have ZERO valid pixels"
    sel = select_clear_frames(values, ts, mask)
    assert 0 not in sel.kept_idx


def test_fully_clouded_cube_selects_nothing_and_encoder_refuses_it(dummy):
    values, ts, mask = _synthetic(T=4)
    mask[:] = False
    sel = select_clear_frames(values, ts, mask)
    assert sel.values.shape[0] == 0
    with pytest.raises(AssertionError, match="EMPTY BATCH"):
        dummy.encode(torch.from_numpy(sel.values), verbose=False)


def test_encode_cube_refuses_cube_with_no_clear_frame_by_name(dummy):
    """Every frame 25% clear: passes the reflectance check (there ARE valid
    pixels), fails selection, and the refusal names the cube."""
    s = _sample()
    mask = np.zeros_like(s.mask)
    mask[:, ::2, ::2] = True  # 25% valid everywhere, no frame exceeds 0.5
    s = s._replace(mask=mask)
    with pytest.raises(AssertionError, match="synthetic_cube.nc"):
        encode_cube(s, dummy, verbose=False)


def test_encode_cube_refuses_fully_masked_cube(dummy):
    """Zero valid pixels anywhere: the reflectance check itself refuses it,
    before any frame can reach an encoder."""
    s = _sample()
    s = s._replace(mask=np.zeros_like(s.mask))
    with pytest.raises(AssertionError, match="no valid finite pixel"):
        encode_cube(s, dummy, verbose=False)


# --------------------------------------------------- valid-reflectance assertion
def test_bright_cloud_under_the_mask_is_harmless():
    values, ts, mask = _synthetic(T=3)
    values[1, 0][~mask[1]] = 1.98  # Phase 1.1's global max, on MASKED pixels
    vmax = assert_valid_reflectance(values, mask)
    print(f"[test] valid-pixel max {vmax:.3f} with 1.98 hidden under the mask")
    assert vmax <= 1.2


def test_bright_cloud_leaking_through_the_mask_fails_loudly():
    values, ts, mask = _synthetic(T=3)
    t, (y, x) = 2, np.argwhere(mask[2])[0]
    values[t, 0, y, x] = 1.98  # the same pixel, now marked VALID
    with pytest.raises(AssertionError, match="leaking THROUGH the mask"):
        assert_valid_reflectance(values, mask, verbose=False)


# ------------------------------------------------- FrozenEncoder machinery
def test_dummy_returns_T_kept_by_D(dummy):
    values, ts, mask = _synthetic()
    sel = select_clear_frames(values, ts, mask, verbose=False)
    z = dummy.encode(torch.from_numpy(sel.values), verbose=False)
    print(f"[test] dummy D={dummy.embed_dim}, embeddings {tuple(z.shape)}")
    assert z.shape == (sel.values.shape[0], dummy.embed_dim)


def test_batching_is_invariant_and_order_preserving(dummy):
    frames = torch.rand(13, len(S2_BANDS), 8, 8) * 0.4
    full = dummy.encode(frames, batch_size=13, verbose=False)
    for bs in (1, 4, 5):
        chunked = dummy.encode(frames, batch_size=bs, verbose=False)
        torch.testing.assert_close(chunked, full)


def test_encoder_is_frozen_and_output_carries_no_grad(dummy):
    assert not dummy._model.training
    assert all(not p.requires_grad for p in dummy._model.parameters())
    z = dummy.encode(torch.rand(2, len(S2_BANDS), 8, 8), verbose=False)
    assert not z.requires_grad


def test_a_stray_train_call_fails_loudly(dummy):
    dummy._model.train()
    try:
        with pytest.raises(AssertionError, match="train"):
            dummy.encode(torch.rand(2, len(S2_BANDS), 8, 8), verbose=False)
    finally:
        dummy._model.eval()


@pytest.mark.parametrize(
    "bad, why",
    [
        (torch.rand(8, 8), "rank 2"),
        (torch.rand(5, 8, 8), "rank 3, no band axis"),
        (torch.rand(5, 8, 8, len(S2_BANDS)), "channels-last"),
        (torch.rand(0, len(S2_BANDS), 8, 8), "empty batch"),
        (torch.randint(0, 255, (5, len(S2_BANDS), 8, 8)), "integer dtype"),
    ],
)
def test_wrong_shaped_input_fails_loudly_not_silently(dummy, bad, why):
    with pytest.raises(AssertionError):
        dummy.encode(bad, verbose=False)
    print(f"[test] dummy refused {why}: {tuple(bad.shape)} {bad.dtype}")


def test_nonfinite_input_fails_loudly(dummy):
    frames = torch.rand(3, len(S2_BANDS), 8, 8)
    frames[1, 2, 4, 4] = float("nan")
    with pytest.raises(AssertionError, match="non-finite"):
        dummy.encode(frames, verbose=False)


# ------------------------------------------------------- raw-feature baseline
def test_raw_features_shape_and_D(raw):
    values, ts, mask = _synthetic()
    sel = select_clear_frames(values, ts, mask, verbose=False)
    z = raw.encode(torch.from_numpy(sel.values), mask=torch.from_numpy(sel.mask),
                   verbose=False)
    print(f"[test] raw_features D={raw.embed_dim}, embeddings {tuple(z.shape)}")
    assert z.shape == (sel.values.shape[0], raw.embed_dim)
    assert raw.embed_dim == len(RAW_FEATURE_NAMES) == 35


def test_raw_features_match_canonical_ndvi_and_plain_band_stats(raw):
    values, ts, mask = _synthetic()
    sel = select_clear_frames(values, ts, mask, verbose=False)
    z = raw.encode(torch.from_numpy(sel.values), mask=torch.from_numpy(sel.mask),
                   verbose=False).numpy()

    names = list(RAW_FEATURE_NAMES)
    b8a = sel.values[:, S2_BANDS.index("B8A")]
    b04 = sel.values[:, S2_BANDS.index("B04")]
    nd = ndvi(b8a, b04, sel.mask)  # THE canonical definition, imported not copied
    T_kept = sel.values.shape[0]
    for t in range(T_kept):
        np.testing.assert_allclose(
            z[t, names.index("NDVI_p50")], np.nanmedian(nd[t]), rtol=1e-5)
        np.testing.assert_allclose(
            z[t, names.index("B02_mean")], sel.values[t, S2_BANDS.index("B02")].mean(),
            rtol=1e-5)


def test_raw_features_masked_pixels_cannot_leak_into_ndvi_stats(raw):
    """Corrupt every masked pixel; the NDVI half of the vector must not move.
    (The band half MUST move: band stats run on the unmodified frame, clouds
    included, exactly the input the network encoders see.)"""
    values, ts, mask = _synthetic()
    sel = select_clear_frames(values, ts, mask, verbose=False)

    corrupted = sel.values.copy()
    corrupted[np.broadcast_to(~sel.mask[:, None], corrupted.shape)] = 0.77

    z0 = raw.encode(torch.from_numpy(sel.values), mask=torch.from_numpy(sel.mask),
                    verbose=False).numpy()
    z1 = raw.encode(torch.from_numpy(corrupted), mask=torch.from_numpy(sel.mask),
                    verbose=False).numpy()

    np.testing.assert_array_equal(z0[:, NDVI_FEATURE_SLICE], z1[:, NDVI_FEATURE_SLICE])
    assert not np.allclose(z0[:, : NDVI_FEATURE_SLICE.start],
                           z1[:, : NDVI_FEATURE_SLICE.start]), (
        "band stats ignored the corruption; they are no longer computed on the "
        "unmodified frame the networks see"
    )


def test_raw_features_require_the_mask(raw):
    values, ts, mask = _synthetic()
    sel = select_clear_frames(values, ts, mask, verbose=False)
    with pytest.raises(AssertionError, match="REQUIRES the cloud mask"):
        raw.encode(torch.from_numpy(sel.values), verbose=False)


# ------------------------------------------------------------------ pipeline
def test_encode_cube_retained_count_matches_independent_count(dummy):
    s = _sample(T=9, seed=3)
    T, C, H, W = s.values.shape
    # independent route: count per frame, no encoders.frames involved
    independent = sum(
        1 for t in range(T) if np.count_nonzero(s.mask[t]) / (H * W) > 0.5
    )
    ec = encode_cube(s, dummy, verbose=False)
    print(f"[test] retained {ec.embeddings.shape[0]} == independent {independent}")
    assert ec.embeddings.shape == (independent, dummy.embed_dim)
    assert ec.clear_frac.shape == ec.kept_idx.shape == (independent,)
    np.testing.assert_allclose(
        ec.clear_frac,
        [np.count_nonzero(s.mask[i]) / (H * W) for i in ec.kept_idx],
    )


def test_encoded_cube_roundtrip(tmp_path, dummy):
    ec = encode_cube(_sample(), dummy, verbose=False)
    path = save_encoded(str(tmp_path), ec, verbose=False)
    back = load_encoded(path)
    np.testing.assert_array_equal(back.embeddings, ec.embeddings)
    np.testing.assert_array_equal(back.timestamps, ec.timestamps)
    np.testing.assert_array_equal(back.clear_frac, ec.clear_frac)
    np.testing.assert_array_equal(back.kept_idx, ec.kept_idx)
    assert (back.encoder, back.cube) == (ec.encoder, ec.cube)


# ------------------------------------------- real wrappers (weights required)
_HEAVY = pytest.mark.skipif(
    os.environ.get("PHASE1_2_WEIGHTS") != "1",
    reason="downloads pretrained weights; set PHASE1_2_WEIGHTS=1 to run "
    "(the Phase 1.2 notebook exercises these on all 20 real cubes)",
)


@_HEAVY
@pytest.mark.parametrize("name", ["imagenet_vit_b16", "dinov2_vitb14", "satlas_s2_swinb_rgb"])
def test_real_wrapper_loads_frozen_and_emits_asserted_shape(name):
    import sys

    if name == "dinov2_vitb14" and sys.version_info < (3, 10):
        pytest.skip(
            "facebookresearch/dinov2 hub code evaluates PEP 604 unions "
            "(float | None) at import and needs Python >= 3.10; Colab has it, "
            "this interpreter does not"
        )
    enc = build_encoder(name, device="cpu")
    frames = torch.rand(3, len(S2_BANDS), 128, 128) * 0.4
    z = enc.encode(frames, batch_size=2)
    print(f"[test] {name} D={enc.embed_dim}, embeddings {tuple(z.shape)}")
    assert z.shape == (3, enc.embed_dim)
    with pytest.raises(AssertionError):
        enc.encode(torch.rand(3, 128, 128), verbose=False)
