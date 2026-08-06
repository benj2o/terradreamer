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
    finite_valid_mask,
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
    grid_dim = 6
    FEATURE_RECIPE = "test double: mean-pool then a fixed-seed linear map C -> 6"

    def _build(self):
        torch.manual_seed(0)
        return torch.nn.Linear(len(S2_BANDS), self.embed_dim)

    def _preprocessing_lines(self):
        return ["global average pool over H, W", "fixed-seed linear map C -> 6"]

    def _encode_batch(self, frames, mask):
        x = frames.mean(dim=(2, 3)).to(self.device)
        return self._model(x)

    def _features_batch(self, frames, mask):
        from encoders.base import pool_to_grid
        f = torch.nan_to_num(frames, nan=0.0)
        pooled = self._model(f.mean(dim=(2, 3)).to(self.device))
        # (B, C, H, W) -> per-cell linear map, so grid.mean(1) == pooled exactly
        cells = pool_to_grid(f.to(self.device))          # (B, 16, C)
        return {"pooled": pooled, "grid": self._model(cells)}


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


# ------------------------------------------- mask / reflectance self-consistency
def test_finite_valid_mask_demotes_valid_but_nodata_pixels():
    """GreenEarthNet's conjunction reads only the mask bands, so it can call a
    no-data pixel clear. 113 such pixels exist across tile 32UNU."""
    values, ts, mask = _synthetic(T=3)
    y, x = np.argwhere(mask[2])[0]
    values[2, 1, y, x] = np.nan          # one band missing at a "clear" pixel
    assert mask[2, y, x], "fixture precondition: the pixel starts out valid"

    corrected, n = finite_valid_mask(values, mask)
    print(f"[test] demoted {n} mask-valid but non-finite pixel(s)")
    assert n == 1
    assert not corrected[2, y, x], "a pixel with no reflectance is not an observation"
    assert corrected.sum() == mask.sum() - 1, "only that pixel may change"
    assert (corrected <= mask).all(), "the rule may only remove valid pixels"


def test_selection_uses_the_corrected_mask():
    values, ts, mask = _synthetic(T=6)
    y, x = np.argwhere(mask[5])[0]
    values[5, 0, y, x] = np.nan
    sel = select_clear_frames(values, ts, mask, verbose=False)
    i = int(np.flatnonzero(sel.kept_idx == 5)[0])
    assert not sel.mask[i, y, x]
    # clear-fraction must reflect the correction, not the raw mask
    np.testing.assert_allclose(sel.clear_frac[i], sel.mask[i].mean())


def test_nodata_pixel_does_not_poison_raw_baseline_band_stats(raw):
    """Plain np.mean over a frame holding one NaN returns NaN for the whole
    frame; NaN-aware reductions are what keep the baseline finite."""
    values, ts, mask = _synthetic(T=6)
    values[5, 0, 3, 3] = np.nan
    sel = select_clear_frames(values, ts, mask, verbose=False)
    z = raw.encode(torch.from_numpy(np.ascontiguousarray(sel.values)),
                   mask=torch.from_numpy(sel.mask), verbose=False)
    assert torch.isfinite(z).all(), "a single no-data pixel blanked the embedding"


# --------------------------------------------------- valid-reflectance assertion
def test_bright_cloud_under_the_mask_is_harmless():
    values, ts, mask = _synthetic(T=3)
    values[1, 0][~mask[1]] = 1.98  # Phase 1.1's global max, on MASKED pixels
    rep = assert_valid_reflectance(values, mask)
    print(f"[test] valid max {rep.valid_max:.3f}, global max {rep.global_max:.3f} "
          "with 1.98 hidden under the mask")
    assert rep.valid_max <= 1.2 and rep.n_implausible == 0
    assert rep.global_max > 1.2, "the masked bright pixel should still be reported"


def test_isolated_bright_pixel_is_tolerated_and_counted():
    """The tile-32UNU case: a few specular pixels in a nearly-clear frame.
    Measured prevalence there is 2.5e-6; this must not halt the phase, but it
    must be counted rather than ignored."""
    values, ts, mask = _synthetic(T=3, H=128, W=128)
    t, (y, x) = 2, np.argwhere(mask[2])[0]
    values[t, 0, y, x] = 1.78
    rep = assert_valid_reflectance(values, mask, verbose=False)
    print(f"[test] tolerated {rep.n_implausible} px at {rep.fraction:.2e}")
    assert rep.n_implausible == 1
    assert rep.fraction < 1e-4


def test_systemic_cloud_leak_fails_loudly():
    """A whole frame of cloud passing as clear -- the failure the check exists
    for. Prevalence is ~1e-1, three orders of magnitude above the tolerance."""
    values, ts, mask = _synthetic(T=3, H=64, W=64)
    values[2][:, mask[2]] = 1.5  # every valid pixel of frame 2 is bright cloud
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


def test_entirely_nonfinite_frame_fails_loudly(dummy):
    frames = torch.rand(3, len(S2_BANDS), 8, 8)
    frames[1] = float("nan")  # a frame with nothing to look at
    with pytest.raises(AssertionError, match="entirely"):
        dummy.encode(frames, verbose=False)


def test_scattered_nodata_pixels_are_substituted_not_rejected():
    """GreenEarthNet marks some pixels no-data inside otherwise clear frames.
    A network wrapper must survive them and report the substitution."""

    class _Net(_DummyEncoder):
        name = "sanitising_dummy"

        def _encode_batch(self, frames, mask):
            x = self._sanitise(frames)
            return self._model(x.mean(dim=(2, 3)).to(self.device))

    enc = _Net(device="cpu", verbose=False)
    frames = torch.rand(3, len(S2_BANDS), 8, 8)
    frames[1, 2, 4, 4] = float("nan")
    z = enc.encode(frames, verbose=False)
    assert z.shape == (3, enc.embed_dim)
    assert torch.isfinite(z).all()
    assert enc._n_sanitised == 1, "the substitution must be counted, not silent"


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
@pytest.mark.parametrize("name", ["imagenet_vit_b16", "dinov2_vitb14", "satlas_s2_swinb_rgb", "satlas_s2_swinb_mi_rgb"])
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


# ------------------------------------------------ Phase 1.2b: grid + schema
def test_bundle_emits_pooled_and_grid(dummy):
    values, ts, mask = _synthetic(T=6, H=16, W=16)
    sel = select_clear_frames(values, ts, mask, verbose=False)
    b = dummy.encode_bundle(torch.from_numpy(sel.values),
                            mask=torch.from_numpy(sel.mask), verbose=False)
    T = sel.values.shape[0]
    assert b["pooled"].shape == (T, dummy.embed_dim)
    assert b["grid"].shape == (T, 16, dummy.grid_dim)
    print(f"[test] pooled {tuple(b['pooled'].shape)} grid {tuple(b['grid'].shape)}")


def test_grid_pools_back_to_pooled_when_divisible(dummy):
    """The classic patch-token bug is a wrong reshape/permute. Where the source
    lattice divides evenly into the grid, the cell mean must reproduce the
    pooled vector; the dummy's map is linear so this is exact."""
    values, ts, mask = _synthetic(T=4, H=16, W=16)
    sel = select_clear_frames(values, ts, mask, verbose=False)
    b = dummy.encode_bundle(torch.from_numpy(sel.values),
                            mask=torch.from_numpy(sel.mask), verbose=False)
    torch.testing.assert_close(b["grid"].mean(dim=1), b["pooled"], atol=1e-5, rtol=1e-4)


def test_uneven_lattice_explains_its_own_mismatch():
    """ViT-B/16 gives a 14x14 patch lattice, which does NOT divide into 4x4, so
    adaptive pooling uses uneven bins and the cell mean is a WEIGHTED patch
    mean. That is geometry, not a reshape bug -- pinned here so nobody
    'fixes' it."""
    from torch.nn.functional import adaptive_avg_pool2d
    for side, divisible in ((16, True), (14, False)):
        x = torch.rand(2, 8, side, side)
        d = float((adaptive_avg_pool2d(x, (4, 4)).flatten(2).mean(2)
                   - x.flatten(2).mean(2)).abs().max())
        print(f"[test] {side}x{side} -> 4x4 divisible={divisible} diff={d:.2e}")
        assert (d < 1e-6) == divisible


def test_grid_clear_fraction_bounds_and_consistency():
    from encoders.frames import grid_clear_fraction
    _v, _t, mask = _synthetic(T=6, H=16, W=16)
    g = grid_clear_fraction(mask)
    assert g.shape == (6, 16)
    assert (g >= 0).all() and (g <= 1).all()
    np.testing.assert_allclose(g.mean(axis=1), clear_fraction(mask), atol=1e-6)


def test_grid_clear_fraction_is_finer_than_the_frame_scalar():
    """A frame at 0.5 clear can hold cells at 0.0 and cells at 1.0 -- which is
    the whole reason probes filter cells, not just frames."""
    from encoders.frames import grid_clear_fraction
    mask = np.zeros((1, 16, 16), dtype=bool)
    mask[0, :8] = True                      # top half clear
    g = grid_clear_fraction(mask)
    assert g.min() == 0.0 and g.max() == 1.0
    np.testing.assert_allclose(g.mean(), 0.5)


def test_encoded_cube_roundtrips_grid_as_float16(tmp_path, dummy):
    ec = encode_cube(_sample(H=16, W=16), dummy, verbose=False)
    assert ec.grid.dtype == np.float16, "grid must be stored fp16"
    assert ec.grid_clear_frac.shape == (ec.embeddings.shape[0], 16)
    back = load_encoded(save_encoded(str(tmp_path), ec, verbose=False))
    np.testing.assert_array_equal(back.grid, ec.grid)
    np.testing.assert_array_equal(back.grid_clear_frac, ec.grid_clear_frac)


def test_cached_mask_roundtrips_and_reproduces_the_scalars(tmp_path, dummy):
    """Common-masking is probe-side, but it is impossible unless the per-pixel
    mask is cached. Verify the cache is faithful."""
    from encoders.pipeline import cube_masks, load_masks, save_masks
    s = _sample(T=9, H=16, W=16, seed=3)
    cm = cube_masks(s, verbose=False)
    back = load_masks(save_masks(str(tmp_path), cm, verbose=False))
    np.testing.assert_array_equal(back.mask, cm.mask)
    np.testing.assert_array_equal(back.kept_idx, cm.kept_idx)
    ec = encode_cube(s, dummy, verbose=False)
    np.testing.assert_allclose(back.mask.mean(axis=(1, 2)), ec.clear_frac, atol=1e-12)


# ------------------------------------------- Phase 1.2c: strata, weather, MI
def _cube_dirs():
    """The two places the shared cubes can sit, most specific first.

    Anchored on THIS FILE, not the working directory. pytest is run from the
    repo root locally but from a phase checkout on Drive, and a cwd-relative
    glob silently skipped four real-cube tests there -- a weaker gate in the
    one environment the phases actually run in.
    """
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return [
        os.path.join(repo, "data", "raw"),                  # dev clone
        os.path.join(os.path.dirname(repo), "data", "raw"),  # phase checkout:
    ]                                        # cubes are SHARED at the project
                                             # root, one level above the phase


def _real_cube():
    import glob
    for d in _cube_dirs():
        p = sorted(glob.glob(os.path.join(d, "*.nc")))
        if p:
            return p[0]
    pytest.skip("no cubes in " + " or ".join(_cube_dirs())
                + "; run data.download_greenearthnet")


def test_grid_landcover_aligns_with_the_embedding_grid():
    from encoders.manifest import cube_grid_landcover
    names, purity = cube_grid_landcover(_real_cube())
    assert names.shape == purity.shape == (16,), "one label per emb_grid cell"
    assert not any(n.startswith("ABSENT:") for n in names)
    assert ((purity > 0) & (purity <= 1)).all()
    print(f"[test] per-cell strata: {sorted(set(names.tolist()))}")


def test_grid_landcover_is_finer_than_the_per_cube_label():
    """A 640 m cell is homogeneous far more often than a 2.56 km cube, which
    is the whole point: within-cube stratum contrast under ONE weather
    realisation."""
    from encoders.manifest import cube_grid_landcover, cube_landcover
    names, purity = cube_grid_landcover(_real_cube())
    dominant, _ = cube_landcover(_real_cube())
    assert dominant in set(names.tolist())
    cell_pure = float(np.mean(purity))
    print(f"[test] mean per-cell purity {cell_pure:.3f}, cube label {dominant!r}")
    assert cell_pure > 0.5


def test_in_cube_eobs_is_present_and_finite():
    """P4's entire input ships in-cube; no external weather table is needed."""
    from encoders.manifest import cube_weather
    w = cube_weather(_real_cube())
    assert len(w) == 8, f"expected 8 E-OBS variables, got {sorted(w)}"
    for k, a in w.items():
        assert np.isfinite(a).all(), f"{k} has gaps"
    print(f"[test] E-OBS in-cube: {sorted(w)}")


def test_elevation_is_cached_per_cell():
    from encoders.manifest import cube_grid_elevation
    e = cube_grid_elevation(_real_cube())
    assert e.shape == (16,) and np.isfinite(e).all()
    print(f"[test] cell elevation {e.min():.0f}-{e.max():.0f} m")


@_HEAVY
def test_multi_image_encoder_is_batch_invariant_and_uses_context():
    """The MI window crosses batch boundaries via a context buffer, so
    batching MUST NOT change a single number -- and MI must actually differ
    from SI, or it is not a positive control at all."""
    enc = build_encoder("satlas_s2_swinb_mi_rgb", device="cpu", verbose=False)
    frames = torch.rand(11, len(S2_BANDS), 128, 128) * 0.4
    full = enc.encode_bundle(frames, batch_size=11, verbose=False)["pooled"]
    for bs in (1, 4):
        got = enc.encode_bundle(frames, batch_size=bs, verbose=False)["pooled"]
        torch.testing.assert_close(got, full, atol=0, rtol=0)
    si = build_encoder("satlas_s2_swinb_rgb", device="cpu", verbose=False)
    sif = si.encode_bundle(frames, batch_size=11, verbose=False)["pooled"]
    assert float((sif - full).abs().max()) > 1e-3, "MI ignores its temporal context"


def test_window_span_days_is_cached_and_varies_with_cloud():
    """The MI lookback is a variable number of DAYS and the variation is
    weather-correlated: a cloudier stretch drops more frames, so the same 8
    retained frames reach further back. Probes must be able to control for it,
    which means it has to be cached at encode time."""
    from encoders.pipeline import window_span_days

    # 8 frames, then a cloud gap, then 8 more: irregular by construction.
    t = np.array([0, 5, 10, 15, 20, 25, 30, 35, 75, 80], dtype="timedelta64[D]")
    ts = (np.datetime64("2018-04-01") + t).astype("datetime64[ns]")

    single = window_span_days(ts, window_len=1)
    assert (single == 0).all(), "a single-image lookback is one frame, span 0"

    multi = window_span_days(ts, window_len=8)
    assert multi.shape == ts.shape
    assert (multi >= 0).all() and np.isfinite(multi).all()
    assert multi[0] == 0                      # nothing to look back on yet
    assert multi[7] == 35                     # 8 clear frames, 5 d apart
    assert multi[9] == 70                     # same 8 frames, spanning a cloud gap
    print(f"[test] window_span_days: {multi.tolist()}")
    assert multi.max() > multi[7], (
        "span must widen across a cloud gap, or the confound is not being measured"
    )


def test_encoded_cube_roundtrips_window_span_days(tmp_path, dummy):
    ec = encode_cube(_sample(H=16, W=16), dummy, verbose=False)
    assert ec.window_span_days is not None
    assert (ec.window_span_days == 0).all(), "dummy is single-image"
    back = load_encoded(save_encoded(str(tmp_path), ec, verbose=False))
    np.testing.assert_array_equal(back.window_span_days, ec.window_span_days)


def test_stale_cache_schema_is_refused_loudly(tmp_path, dummy):
    """A cache from an older schema lacks newer keys, which np.load reports as
    simply absent -- so a probe would read window_span_days, find nothing, and
    silently drop the covariate. Encoder dimensionality cannot catch this: the
    MI encoder and window_span_days landed in different commits."""
    import numpy as np
    from encoders.pipeline import SCHEMA_VERSION

    ec = encode_cube(_sample(H=16, W=16), dummy, verbose=False)
    path = save_encoded(str(tmp_path), ec, verbose=False)
    assert load_encoded(path).window_span_days is not None

    # Rewrite it as the previous schema: drop the stamp and the newest field.
    with np.load(path) as z:
        payload = {k: z[k] for k in z.files
                   if k not in ("schema_version", "window_span_days")}
    np.savez_compressed(path, **payload)
    with pytest.raises(AssertionError, match="schema"):
        load_encoded(path)
    print(f"[test] refused a v0 cache against v{SCHEMA_VERSION}")


def test_phase_dirs_isolate_and_reset(tmp_path):
    """Re-running one phase must not touch another, and must never touch the
    shared cube directory."""
    from data.paths import phase_dir, reset_phase

    root = str(tmp_path)
    a = phase_dir("phase1_2", "embeddings", root=root)
    b = phase_dir("phase1_3", "folds", root=root)
    for d, n in ((a, 3), (b, 2)):
        for i in range(n):
            open(os.path.join(d, f"f{i}.npz"), "w").write("x")

    removed = reset_phase("phase1_2", root=root, verbose=False)
    assert removed == 3
    # The phase root survives (empty); phase_dir recreates kinds on demand.
    assert os.listdir(os.path.join(root, "phase1_2")) == [], "phase1_2 not cleared"
    assert os.listdir(phase_dir("phase1_2", "embeddings", root=root)) == []
    assert len(os.listdir(b)) == 2, "reset_phase leaked into another phase"
    print(f"[test] reset removed {removed} file(s), left phase1_3 intact")


def test_reset_phase_refuses_to_delete_the_shared_cubes(tmp_path):
    """data/raw is shared across phases; re-downloading 20 cubes to re-run a
    probe is waste, not hygiene."""
    from data.paths import RAW_DIR, reset_phase

    with pytest.raises(AssertionError, match="shared cube directory"):
        reset_phase(os.path.basename(RAW_DIR), root=os.path.dirname(RAW_DIR),
                    verbose=False)


# ------------------------------------------- the cube lookup itself
# These four real-cube tests silently SKIPPED on Colab for a whole phase,
# because the lookup was cwd-relative and the phase checkout does not contain
# the cubes -- they are SHARED at the project root, one level up. A weaker
# gate in the one environment the phases actually run in is worth pinning.

def test_cube_lookup_covers_both_layouts():
    """Dev clone AND phase checkout, most specific first."""
    dirs = _cube_dirs()
    assert len(dirs) == 2
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert dirs[0] == os.path.join(repo, "data", "raw")
    assert dirs[1] == os.path.join(os.path.dirname(repo), "data", "raw")


def test_cube_lookup_is_anchored_on_the_file_not_the_working_directory(monkeypatch,
                                                                      tmp_path):
    """pytest runs from the repo root locally and from a phase checkout on
    Drive. The answer must not depend on which."""
    before = _cube_dirs()
    monkeypatch.chdir(tmp_path)
    assert _cube_dirs() == before
