"""P2, dynamics in deltas: the assertions the spec names, plus the machinery
they rest on.

Most of it runs on a SYNTHETIC manifest with known cube membership, a known
day grid and a known embedding-to-NDVI relation, so each claim is checked
against an independent computation rather than against the code that made it.
The tests that need real cubes or the Phase 1.2 cache skip when those are
absent.

The four that matter most are the ones nothing downstream could catch:

* ``test_common_masking_uses_the_intersection_not_either_frames_own_mask`` --
  differencing two per-frame means, each over its own valid set, compares two
  different pieces of ground and calls the difference a change. Same shape,
  same dtype, plausible magnitude.
* ``test_gap_days_is_the_day_axis_and_the_axes_are_materially_different`` --
  the wrong column is finite, integer, in range, monotone within a cube and
  correlated with the right one. A test that compares the right column to
  itself would pass without measuring anything, so this one asserts the two
  candidates DISAGREE before believing either.
* the poison PAIRS on ``fit_gap_control`` and ``select_ridge_alpha`` -- test
  poison must not move the fit, TRAIN poison must. The first alone passes on a
  function that ignores its input entirely.
* ``test_ridge_path_agrees_with_sklearn`` -- the tuner's speed comes from
  re-solving one factorisation per penalty. If that is not the same estimator,
  every number in the table is from a model nobody specified.
"""

from __future__ import annotations

import glob
import inspect
import os

import numpy as np
import pandas as pd
import pytest

from encoders.base import GRID, GRID_CELLS
from probes import cv
from probes import p2_deltas as p2
from probes.p1_appearance import FeatureBlock

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(REPO, "data", "raw")
EMB = os.path.join(REPO, "data", "phase1_2", "embeddings")
MASKS = os.path.join(REPO, "data", "phase1_2", "masks")

needs_cubes = pytest.mark.skipif(
    not glob.glob(os.path.join(RAW, "*.nc")),
    reason="no cubes in data/raw; the real-data checks are skipped")
needs_cache = pytest.mark.skipif(
    not glob.glob(os.path.join(EMB, "*.npz")) or not glob.glob(
        os.path.join(MASKS, "*.npz")),
    reason="no Phase 1.2 embedding/mask cache; the real-data checks are skipped")


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

def _manifest(n_cubes=8, frames=8, start="2018-04-01", stagger_days=9,
              day_step=10, acq_step=2, clear=0.8):
    """One tile, staggered windows -- the shape of the real subset.

    ``day_step`` and ``acq_step`` are deliberately DIFFERENT, and their ratio
    (5 here, as on the real data) is what makes the axis test able to tell the
    two columns apart. A fixture with day_step == acq_step would let a probe
    that read the wrong column pass every test in this file.
    """
    rows = []
    for c in range(n_cubes):
        t0 = np.datetime64(start) + np.timedelta64(stagger_days * c, "D")
        for f in range(frames):
            ts = t0 + np.timedelta64(day_step * f, "D")
            doy = int((ts - ts.astype("datetime64[Y]")).astype(int)) + 1
            rows.append({
                "cube_id": f"32UNU_2018_cube{c}.nc",
                "tile": "32UNU",
                "year": 2018,
                "timestamp": ts,
                "original_axis_index": acq_step * f,
                "daily_axis_index": day_step * f,
                "day_of_year": doy,
                "pixel_bbox": (100 * c, 100 * c + 128, 200 * c, 200 * c + 128),
                "clear_frac": clear,
                "landcover_stratum": "cropland",
                "grid_landcover": tuple(["cropland"] * GRID_CELLS),
            })
    return pd.DataFrame(rows)


def _arrays(m, D=6, Dg=6, seed=0):
    """A fake encoder cache whose pooled vector encodes the frame's NDVI."""
    rng = np.random.default_rng(seed)
    n = len(m)
    pooled = rng.normal(0, 1, (n, D)).astype(np.float32)
    grid = rng.normal(0, 1, (n, GRID_CELLS, Dg)).astype(np.float32)
    return {"encoder": "fake",
            "pooled": pooled, "grid": grid,
            "grid_clear_frac": np.full((n, GRID_CELLS), 0.8, dtype=np.float32),
            "clear_frac": np.full(n, 0.8),
            "window_span_days": np.linspace(0, 100, n)}


@pytest.fixture
def manifest():
    return _manifest()


@pytest.fixture
def pairs(manifest):
    return p2.pair_index(manifest, verbose=False)


# ---------------------------------------------------------------------------
# THE assertion: the gap comes from the DAY axis, and the axes disagree
# ---------------------------------------------------------------------------

def test_gap_days_is_the_day_axis_and_the_axes_are_materially_different(pairs):
    """gap_days must be the daily axis, and the two candidates must DISAGREE.

    The second half is what makes the first half mean anything. On this fixture
    a consecutive pair spans 10 days and 2 acquisition steps, so a probe that
    read ``original_axis_index`` would report gaps five times too short --
    finite, integer, in range, and wrong.
    """
    assert (pairs.gap_days == 10).all(), pairs.gap_days[:5]
    assert (pairs.gap_acq_steps == 2).all(), pairs.gap_acq_steps[:5]
    assert (pairs.gap_frame_steps == 1).all()

    info = p2.assert_gap_axes_disagree(pairs, verbose=False)
    assert info["n_axes_equal"] == 0
    assert info["ratio_days_per_acq_step"] == pytest.approx(5.0)
    # The three readings are pairwise different, which is the only condition
    # under which this check can distinguish the right column from the wrong.
    assert info["median_gap_days"] != info["median_gap_acq_steps"]
    assert info["median_gap_days"] != info["median_gap_frame_steps"]


def test_the_axis_check_fails_when_the_gap_came_from_the_join_key(pairs):
    """The decisive companion: feed it the WRONG column and it must refuse.

    A guard that only ever passes is a hypothesis. This builds exactly the bug
    -- gap_days computed from ``original_axis_index`` -- and asserts the check
    catches it and names the cause.
    """
    from dataclasses import replace

    wrong = replace(pairs, gap_days=pairs.gap_acq_steps.astype(float))
    with pytest.raises(AssertionError, match="original_axis_index"):
        p2.assert_gap_axes_disagree(wrong, verbose=False)


def test_the_axis_check_refuses_a_gap_that_is_the_count_of_retained_frames(pairs):
    """"Frames between t and t+1" is 1 for every consecutive pair, so it is not
    a gap. Asserting it would be caught by the same check."""
    from dataclasses import replace

    counted = replace(pairs, gap_days=pairs.gap_frame_steps.astype(float))
    with pytest.raises(AssertionError):
        p2.assert_gap_axes_disagree(counted, verbose=False)


def test_pair_index_orders_by_the_acquisition_axis_not_by_row_position():
    """Pairs follow original_axis_index, so a shuffled manifest is harmless."""
    m = _manifest(n_cubes=3, frames=5)
    a = p2.pair_index(m, verbose=False)
    shuffled = m.sample(frac=1.0, random_state=0).reset_index(drop=True)
    b = p2.pair_index(shuffled, verbose=False)
    # Same pairs, identified by (cube, acquisition index) rather than by row.
    def key(m_, p_):
        oai = m_["original_axis_index"].to_numpy()
        return sorted(zip(p_.cube_id.tolist(), oai[p_.row_a].tolist(),
                          oai[p_.row_b].tolist()))
    assert key(m, a) == key(shuffled, b)
    np.testing.assert_array_equal(np.sort(a.gap_days), np.sort(b.gap_days))


def test_pair_index_never_pairs_across_cubes(pairs, manifest):
    cid = manifest["cube_id"].to_numpy()
    assert (cid[pairs.row_a] == cid[pairs.row_b]).all()
    assert (pairs.cube_id == cid[pairs.row_a]).all()
    assert pairs.n_pairs == len(manifest) - manifest.cube_id.nunique()


def test_pair_index_cross_checks_the_gap_against_the_timestamps():
    """daily_axis_index and the timestamps are two columns; they must agree."""
    m = _manifest(n_cubes=2, frames=4)
    # The LAST frame of cube 0, so the column stays strictly increasing and the
    # only thing wrong with it is that it disagrees with the timestamp. A
    # non-monotone value would trip a different, earlier assertion and this
    # test would pass without exercising the cross-check.
    m.loc[3, "daily_axis_index"] = 999
    with pytest.raises(AssertionError, match="timestamps"):
        p2.pair_index(m, verbose=False)


# ---------------------------------------------------------------------------
# THE assertion: common-masking uses the INTERSECTION
# ---------------------------------------------------------------------------

def _pair_with_disjointish_masks(H=8, W=8):
    """Two frames whose independent masks differ, with a KNOWN intersection.

    Frame a is valid on the left half, frame b on the top half; the
    intersection is the top-left quadrant. The NDVI values are arranged so the
    three possible answers -- intersection, a's own mask, b's own mask -- are
    all different numbers, which is what lets the test tell them apart.
    """
    a = np.zeros((H, W)); b = np.zeros((H, W))
    ma = np.zeros((H, W), bool); mb = np.zeros((H, W), bool)
    ma[:, : W // 2] = True          # left half
    mb[: H // 2, :] = True          # top half
    # top-left = the intersection; the other quadrants carry very different
    # values so any mask mix-up moves the mean a long way.
    a[: H // 2, : W // 2] = 0.10; b[: H // 2, : W // 2] = 0.30   # +0.20
    a[H // 2:, : W // 2] = 0.90; b[H // 2:, : W // 2] = 0.90     # a-only, no change
    a[: H // 2, W // 2:] = -0.50; b[: H // 2, W // 2:] = -0.50   # b-only, no change
    a[H // 2:, W // 2:] = 5.0; b[H // 2:, W // 2:] = -5.0        # in neither
    return a, b, ma, mb


def test_common_masking_uses_the_intersection_not_either_frames_own_mask():
    """The synthetic pair the spec asks for: two DIFFERENT independent masks.

    The intersection is the top-left quadrant, where the change is exactly
    +0.20. Using either frame's own mask would average the +0.20 quadrant
    together with a quadrant that did not change, giving +0.10 -- the same
    shape, the same dtype, a plausible magnitude, and wrong.
    """
    a, b, ma, mb = _pair_with_disjointish_masks()
    r = p2.common_masked_delta(a, b, ma, mb, grid=2)

    assert r["n_common_px"] == 16, r["n_common_px"]      # 4x4 of an 8x8
    assert r["n_px_a"] == 32 and r["n_px_b"] == 32
    assert r["d_cube_mean"] == pytest.approx(0.20)

    # The three wrong answers, computed independently, and all different.
    own_a = float(b[ma].mean() - a[ma].mean())
    own_b = float(b[mb].mean() - a[mb].mean())
    union = ma | mb
    own_u = float(b[union].mean() - a[union].mean())
    assert own_a == pytest.approx(0.10)
    assert own_b == pytest.approx(0.10)
    for wrong, name in ((own_a, "frame a's own mask"), (own_b, "frame b's own"),
                        (own_u, "the union")):
        assert r["d_cube_mean"] != pytest.approx(wrong), (
            f"the common-masked delta equals what {name} would give, so this "
            "test cannot tell the intersection from it")


def test_common_masking_reports_cells_with_no_shared_pixel():
    """A cell outside the intersection is reported invalid, never filled."""
    a, b, ma, mb = _pair_with_disjointish_masks()
    r = p2.common_masked_delta(a, b, ma, mb, grid=2)
    # grid=2 over 8x8: cell 0 is the top-left quadrant == the intersection.
    assert r["cell_valid"].tolist() == [True, False, False, False]
    assert r["n_common_px_cell"].tolist() == [16, 0, 0, 0]
    assert r["d_cell_mean"][0] == pytest.approx(0.20)
    assert not np.isfinite(r["d_cell_mean"][1:]).any()


def test_common_masking_collapses_loudly_rather_than_silently():
    """Zero shared pixels is a reported state, not a dropped row."""
    a, b, ma, mb = _pair_with_disjointish_masks()
    mb = ~ma                                    # disjoint by construction
    r = p2.common_masked_delta(a, b, ma, mb, grid=2)
    assert r["n_common_px"] == 0 and r["frac_common_px"] == 0.0
    assert not np.isfinite(r["d_cube_mean"])
    assert not r["cell_valid"].any()


def test_common_masking_refuses_an_integer_cloud_code_mask():
    a, b, ma, mb = _pair_with_disjointish_masks()
    with pytest.raises(AssertionError, match="boolean"):
        p2.common_masked_delta(a, b, ma.astype(np.uint8), mb, grid=2)


def test_common_masking_agrees_with_per_frame_means_when_the_masks_agree():
    """The one case where the distinction does not matter -- which is exactly
    why it cannot be spot-checked into existence."""
    rng = np.random.default_rng(3)
    a = rng.normal(0.3, 0.1, (8, 8)); b = rng.normal(0.4, 0.1, (8, 8))
    m = rng.random((8, 8)) > 0.3
    r = p2.common_masked_delta(a, b, m, m.copy(), grid=2)
    assert r["d_cube_mean"] == pytest.approx(float(b[m].mean() - a[m].mean()))


# ---------------------------------------------------------------------------
# THE assertion: leakage is prevented by the signature, with TWO poison tests
# ---------------------------------------------------------------------------

def test_the_gap_control_signature_takes_a_training_index_set():
    """The fit-inside-fold structure must be in the SIGNATURE, not in a habit."""
    sig = inspect.signature(p2.fit_gap_control)
    params = list(sig.parameters)
    assert params[:3] == ["gap_days", "values", "train_idx"], params
    assert sig.parameters["train_idx"].default is inspect.Parameter.empty, (
        "train_idx has a default, so the control can be fitted on everything "
        "by omitting it -- the leak this signature exists to prevent")
    assert sig.parameters["train_idx"].kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD), "train_idx must be positional"


def test_the_tuner_signature_takes_a_training_index_set():
    sig = inspect.signature(p2.select_ridge_alpha)
    params = list(sig.parameters)
    assert params[:3] == ["block", "manifest", "train_rows"], params
    assert sig.parameters["train_rows"].default is inspect.Parameter.empty
    assert not any(p in params for p in ("test_rows", "test", "test_idx")), (
        f"the tuner takes a test argument: {params}. There must be nothing to "
        "leak.")


def _gap_fixture(seed=0):
    rng = np.random.default_rng(seed)
    gap = np.repeat([5.0, 10.0, 15.0, 20.0, 25.0], 20)
    y = 0.001 * gap + 0.00002 * gap ** 2 + rng.normal(0, 0.005, gap.size)
    train = np.arange(0, 60)
    test = np.arange(60, gap.size)
    return gap, y, train, test


def test_the_gap_control_is_numerically_independent_of_the_held_out_rows():
    """Poison ONLY the test rows and refit: the coefficients must not move."""
    gap, y, train, test = _gap_fixture()
    before = p2.fit_gap_control(gap, y, train)
    poisoned = y.copy()
    poisoned[test] += 10.0                      # not subtle
    after = p2.fit_gap_control(gap, poisoned, train)
    np.testing.assert_array_equal(
        before.coef, after.coef,
        err_msg="the gap control moved when only HELD-OUT rows changed. It is "
                "fitted on the test fold, so its value is not a control.")
    assert before.train_r2 == after.train_r2
    assert before.n_train_rows == after.n_train_rows


def test_the_gap_control_does_move_when_a_training_row_changes():
    """The decisive companion: a test that can only pass would prove nothing."""
    gap, y, train, _ = _gap_fixture()
    before = p2.fit_gap_control(gap, y, train)
    poisoned = y.copy()
    poisoned[train[0]] += 10.0
    after = p2.fit_gap_control(gap, poisoned, train)
    assert not np.allclose(before.coef, after.coef), (
        "poisoning a TRAINING row left the control unchanged, so the "
        "held-out-poison test above proves nothing -- it would pass on a "
        "function that ignored its input entirely")


def test_the_gap_control_refuses_a_fold_that_cannot_support_its_degree():
    gap, y, _, _ = _gap_fixture()
    with pytest.raises(AssertionError, match="distinct gap lengths"):
        p2.fit_gap_control(gap, y, np.array([0, 1]), degree=3)


def _tuner_fixture(seed=0):
    m = _manifest(n_cubes=6, frames=6)
    rng = np.random.default_rng(seed)
    n = len(m)
    X = rng.normal(0, 1, (n, 4))
    y = X @ np.array([1.0, -0.5, 0.25, 0.0]) + rng.normal(0, 0.1, n)
    block = FeatureBlock(X=X, row_idx=np.arange(n), y=y, name="synthetic")
    train, test = next(iter(cv.folds(m, "cube", k=3, verbose=False)))
    return m, block, train, test


def test_the_tuner_is_numerically_independent_of_the_held_out_rows():
    """Poison ONLY the test fold: the selected penalty and every inner score
    must be bit-identical."""
    m, block, train, test = _tuner_fixture()
    before = p2.select_ridge_alpha(block, m, train, "r2")
    poisoned_y = block.y.copy()
    poisoned_y[test] += 100.0
    poisoned_X = block.X.copy()
    poisoned_X[test] *= 50.0
    bad = FeatureBlock(X=poisoned_X, row_idx=block.row_idx, y=poisoned_y,
                       name=block.name)
    after = p2.select_ridge_alpha(bad, m, train, "r2")
    assert before["param"] == after["param"]
    np.testing.assert_array_equal(before["inner_scores"], after["inner_scores"])


def test_the_tuner_does_move_when_the_training_fold_changes():
    """The decisive companion for the tuner."""
    m, block, train, _ = _tuner_fixture()
    before = p2.select_ridge_alpha(block, m, train, "r2")
    poisoned_y = block.y.copy()
    poisoned_y[train] += 1000.0 * np.arange(train.size)
    bad = FeatureBlock(X=block.X, row_idx=block.row_idx, y=poisoned_y,
                       name=block.name)
    after = p2.select_ridge_alpha(bad, m, train, "r2")
    assert not np.allclose(before["inner_scores"], after["inner_scores"]), (
        "poisoning the TRAINING fold left the inner scores unchanged, so the "
        "held-out-poison test above proves nothing")


def test_the_tuner_never_puts_a_training_cube_on_both_sides_of_an_inner_fold():
    m, block, train, _ = _tuner_fixture()
    sub = m.iloc[train]
    for itr, ite in cv.folds(sub, "cube", k=3, verbose=False):
        assert not (set(sub.cube_id.to_numpy()[itr])
                    & set(sub.cube_id.to_numpy()[ite]))


def test_the_tuner_breaks_ties_toward_stronger_regularisation():
    """A stated rule, not numpy's scan order: of two models the inner data
    cannot separate, the more regularised one wins."""
    m = _manifest(n_cubes=4, frames=6)
    n = len(m)
    # Pure noise: every penalty scores about the same, so the tie rule decides.
    rng = np.random.default_rng(1)
    block = FeatureBlock(X=np.zeros((n, 3)), row_idx=np.arange(n),
                         y=rng.normal(0, 1, n), name="tie")
    train = np.arange(n)
    out = p2.select_ridge_alpha(block, m, train, "r2")
    assert out["param"] == max(p2.RIDGE_ALPHA_GRID), (
        f"an exact tie selected alpha={out['param']}, not the strongest "
        f"penalty {max(p2.RIDGE_ALPHA_GRID)}")


# ---------------------------------------------------------------------------
# The ridge path IS ridge
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,D", [(120, 20), (30, 200)])
def test_ridge_path_agrees_with_sklearn(n, D):
    """Both regimes: tall (primal Gram) and wide (dual). Same estimator.

    The tuner re-solves one factorisation per penalty instead of refitting.
    That is only legitimate if it is arithmetically the same model, so it is
    checked against sklearn rather than argued for in a comment.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (n, D))
    y = X[:, 0] * 2 - X[:, 1] + rng.normal(0, 0.1, n)
    Xte = rng.normal(0, 1, (17, D))
    alphas = np.array([0.1, 1.0, 10.0, 1000.0])

    got = p2.ridge_path(X, y, Xte, alphas)
    sc = StandardScaler().fit(X)
    for i, a in enumerate(alphas):
        want = Ridge(alpha=a).fit(sc.transform(X), y).predict(sc.transform(Xte))
        np.testing.assert_allclose(got[i], want, rtol=0, atol=1e-8)


def test_ridge_path_standardises_on_train_only():
    """A test row with an extreme scale must not move the training standardiser."""
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (50, 5))
    y = rng.normal(0, 1, 50)
    te = rng.normal(0, 1, (5, 5))
    a = p2.ridge_path(X, y, te, np.array([1.0]))
    b = p2.ridge_path(X, y, np.vstack([te, te * 1e6]), np.array([1.0]))
    np.testing.assert_allclose(a[0], b[0][:5], rtol=0, atol=1e-9)


def test_ridge_path_survives_a_zero_variance_column():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (40, 4))
    X[:, 2] = 7.0
    out = p2.ridge_path(X, rng.normal(0, 1, 40), X[:5], np.array([1.0]))
    assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# Blocks, folds and the pair/manifest index that must never be confused
# ---------------------------------------------------------------------------

def test_a_pair_never_straddles_a_fold(manifest, pairs):
    for mode in ("cube", "spatial_block"):
        for tr, te in p2.outer_folds(manifest, mode, k=3, verbose=False):
            a_tr = np.isin(pairs.row_a, tr)
            b_tr = np.isin(pairs.row_b, tr)
            assert (a_tr == b_tr).all(), (
                f"{mode}: a pair has its two frames on opposite sides")


def test_the_delta_block_row_index_addresses_pairs_not_manifest_rows(manifest, pairs):
    """Part B blocks index PAIRS. Confusing the two changes no shape."""
    arrays = _arrays(manifest)
    b = p2.delta_block(arrays, pairs, "pooled", "linear")
    assert b.n_rows == pairs.n_pairs
    assert b.row_idx.max() == pairs.n_pairs - 1
    assert pairs.n_pairs != len(manifest), (
        "the fixture must have a different number of pairs and manifest rows, "
        "or this confusion would be invisible")
    mapped = p2._manifest_row_of(b, pairs.row_a)
    np.testing.assert_array_equal(mapped, pairs.row_a)


def test_the_delta_block_is_the_difference_of_the_two_frames(manifest, pairs):
    arrays = _arrays(manifest)
    b = p2.delta_block(arrays, pairs, "pooled", "linear")
    want = (arrays["pooled"][pairs.row_b].astype(np.float64)
            - arrays["pooled"][pairs.row_a].astype(np.float64))
    np.testing.assert_allclose(b.X, want, rtol=0, atol=1e-12)


def test_the_norm_readout_is_one_nonnegative_column(manifest, pairs):
    arrays = _arrays(manifest)
    b = p2.delta_block(arrays, pairs, "pooled", "norm")
    assert b.D == 1 and (b.X >= 0).all()
    lin = p2.delta_block(arrays, pairs, "pooled", "linear")
    np.testing.assert_allclose(b.X[:, 0], np.linalg.norm(lin.X, axis=1),
                               rtol=0, atol=1e-12)


def test_the_grid_delta_block_puts_sixteen_rows_on_every_pair(manifest, pairs):
    arrays = _arrays(manifest)
    b = p2.delta_block(arrays, pairs, "grid_cell", "linear")
    assert b.n_rows == pairs.n_pairs * GRID_CELLS
    assert np.bincount(b.row_idx).tolist() == [GRID_CELLS] * pairs.n_pairs


def test_the_reconstruction_block_addresses_manifest_rows(manifest):
    arrays = _arrays(manifest)
    b = p2.reconstruction_block(arrays, "grid_cell")
    assert b.n_rows == len(manifest) * GRID_CELLS
    np.testing.assert_array_equal(p2._manifest_row_of(b, None), b.row_idx)


def test_the_band_matched_view_is_only_defined_for_the_baseline(manifest):
    arrays = _arrays(manifest, D=6, Dg=6)
    with pytest.raises(AssertionError, match="raw_features"):
        p2.reconstruction_block(arrays, "grid_cell", "raw_rgb_only")


def test_the_band_matched_columns_exclude_every_ndvi_column():
    """The band-matched baseline exists to NOT see NDVI. If a rename ever let
    an NDVI column into it, the fair floor would quietly stop being fair."""
    from encoders.raw_features import RAW_FEATURE_NAMES

    cols = p2.RAW_BAND_COLUMNS[p2.BAND_MATCHED_BASELINE]
    names = [RAW_FEATURE_NAMES[i] for i in cols]
    assert names and not any(n.startswith("NDVI") for n in names), names
    assert not any(n.startswith("B8A") for n in names), names
    assert any(n.startswith("NDVI") for n in RAW_FEATURE_NAMES), (
        "raw_features no longer carries NDVI columns, so the two-verdict "
        "structure in p2_deltas' docstring is describing a baseline that no "
        "longer exists")


# ---------------------------------------------------------------------------
# The table: verdicts, controls, effective n, the MI flag
# ---------------------------------------------------------------------------

def _fake_table():
    """A minimal well-formed results table, built the way run_p2 builds one."""
    rows = []
    score = {"raw_features": 0.44, "imagenet_vit_b16": 0.52,
             "dinov2_vitb14": 0.43, "satlas_s2_swinb_rgb": 0.55,
             "satlas_s2_swinb_mi_rgb": 0.08}
    for part in p2.PARTS:
        deltas = p2.DELTA_TARGETS if part == "B_delta" else ("",)
        readouts = p2.READOUTS if part == "B_delta" else ("ridge",)
        for agg, level in p2.PART_A_COMBOS:
            for dt in deltas:
                for mode in p2.FOLD_MODES:
                    for ro in readouts:
                        for enc, fs in p2._encoder_views(p2.ENCODER_ORDER):
                            s = score[enc] + (0.01 if fs != "embedding" else 0)
                            rows.append(_row(part, agg, dt, level, mode, enc,
                                             fs, "delta" if part == "B_delta"
                                             else "reconstruction", ro, s))
                        kind = "gap_only" if part == "B_delta" else "retention"
                        rows.append(_row(part, agg, dt, level, mode, "none",
                                         "gap_days" if part == "B_delta"
                                         else "retention", kind, ro,
                                         0.2 if part == "B_delta" else -0.13))
    return pd.DataFrame(rows)


# Per-ENCODER fold scatter. A fixture where every fold carries the identical
# score makes every paired difference a zero-variance constant, so the paired
# interval has zero width and everything looks "separable" -- an artefact of
# the fixture, not a property of the code. Real per-fold scatter is what makes
# the separability tests below mean anything.
_FOLD_JITTER = {
    "raw_features": (-0.05, 0.02, 0.00, 0.04, -0.01),
    "imagenet_vit_b16": (0.03, -0.04, 0.06, -0.02, -0.03),
    "dinov2_vitb14": (0.07, -0.06, -0.02, 0.05, -0.04),
    "satlas_s2_swinb_rgb": (-0.02, 0.05, -0.05, 0.01, 0.01),
    "satlas_s2_swinb_mi_rgb": (0.01, -0.01, 0.02, -0.02, 0.00),
    "none": (0.0, 0.0, 0.0, 0.0, 0.0),
}


def _row(part, agg, dt, level, mode, enc, fs, kind, ro, score):
    base = p2._base_row(part, agg, dt, level, mode, enc, fs, kind, ro,
                        D=8, n_manifest=264, n_cubes=20, n_pairs=244)
    folds = [score + j for j in _FOLD_JITTER[enc]]
    base.update({
        "score_mean": score, "score_std": 0.05, "score_min": score - 0.1,
        "score_max": score + 0.1, "score_ci_lo": score - 0.08,
        "score_ci_hi": score + 0.08,
        "per_fold_score": ";".join(f"{v:.4f}" for v in folds),
        "spearman_mean": score, "rmse_mean": 0.1, "n_folds": 5,
        "n_folds_nan": 0, "effective_n": 20,
        "effective_n_per_fold": "4;4;4;4;4", "n_rows_test_total": 264,
        "selected_per_fold": "1;1;1;1;1", "n_at_grid_edge": 0,
        "n_common_px_median": 14000.0,
    })
    return base


def test_a_k2_verdict_is_recorded_for_every_encoder():
    """The spec's gate: five encoders, five verdicts, each derived from a
    recorded numeric comparison against raw_features on cube folds."""
    df = p2.add_k2_verdicts(p2.add_margins(_fake_table(), verbose=False),
                            verbose=False)
    p2.assert_k2_verdict_recorded(df)
    for enc in p2.ENCODER_ORDER:
        v = df[df.encoder == enc].k2_verdict.unique()
        assert v.size == 1 and v[0] in ("passed", "audited: lossy", "baseline")
    assert (df[df.encoder == "satlas_s2_swinb_mi_rgb"].k2_verdict
            == "audited: lossy").all()
    assert (df[df.encoder == "raw_features"].k2_verdict == "baseline").all()


def test_a_k2_verdict_that_contradicts_its_own_margin_is_refused():
    """The verdict must be DERIVED from the comparison, not asserted beside it."""
    df = p2.add_k2_verdicts(p2.add_margins(_fake_table(), verbose=False),
                            verbose=False)
    df.loc[df.encoder == "dinov2_vitb14", "k2_verdict"] = "passed"
    df.loc[df.encoder == "dinov2_vitb14", "k2_margin_over_raw"] = -0.3
    with pytest.raises(AssertionError, match="disagrees with"):
        p2.assert_k2_verdict_recorded(df)


def test_the_k2_verdict_carries_a_paired_separability_interval():
    """A binary verdict at 20 cubes is not self-interpreting.

    The paired per-fold difference is what the verdict is ABOUT, and its
    interval is what says whether "audited: lossy" means "did not beat the
    baseline" or "is measurably worse than it". P3 may only drop an encoder on
    the second.
    """
    df = p2.add_k2_verdicts(p2.add_margins(_fake_table(), verbose=False),
                            verbose=False)
    for col in ("k2_margin_ci_lo", "k2_margin_ci_hi", "k2_separable"):
        assert col in df.columns
    enc = df[df.encoder != "none"]
    assert (enc.k2_margin_ci_lo <= enc.k2_margin_ci_hi).all()
    assert (df[df.encoder == "raw_features"].k2_separable == False).all()  # noqa: E712

    # dinov2 sits 0.01 below the baseline against ~0.05 of fold scatter, so the
    # paired interval spans zero: "did not beat it", NOT "measurably worse".
    d = df[df.encoder == "dinov2_vitb14"].iloc[0]
    assert d.k2_verdict == "audited: lossy"
    assert not d.k2_separable
    assert d.k2_margin_ci_lo < 0 < d.k2_margin_ci_hi


def test_a_separable_difference_is_detected_when_one_exists():
    """The decisive companion: an encoder far below the baseline on EVERY fold
    must come out separable, or the column is measuring nothing."""
    df = p2.add_k2_verdicts(p2.add_margins(_fake_table(), verbose=False),
                            verbose=False)
    mi = df[df.encoder == "satlas_s2_swinb_mi_rgb"].iloc[0]
    assert mi.k2_verdict == "audited: lossy"
    assert mi.k2_separable, (
        "an encoder 0.36 below the baseline against 0.05 of fold scatter was "
        "not flagged separable; the paired interval is not being computed")
    assert mi.k2_margin_ci_hi < 0


def test_p3_exclusion_needs_both_a_lossy_verdict_and_separability():
    """The two columns are not redundant: the exclusion rule is their AND."""
    df = p2.add_k2_verdicts(p2.add_margins(_fake_table(), verbose=False),
                            verbose=False)
    per_enc = df.groupby("encoder")[["k2_verdict_band_matched",
                                     "k2_separable"]].first()
    lossy = set(per_enc[per_enc.k2_verdict_band_matched == "audited: lossy"].index)
    firm = {e for e in lossy if per_enc.loc[e, "k2_separable"]}
    assert lossy - firm, (
        "the fixture must contain at least one encoder that is lossy but NOT "
        "separable, or this distinction is untested")
    assert firm, "and at least one that is both"


def test_the_k2_verdict_comes_from_the_cube_clustered_primary_config():
    df = p2.add_k2_verdicts(p2.add_margins(_fake_table(), verbose=False),
                            verbose=False)
    assert (df.k2_reference_config == "cube_mean/grid_cell/cube").all()
    assert p2.K2_PRIMARY["fold_mode"] == "cube", (
        "the K2 verdict must come from cube-clustered folds")


def test_the_gap_control_is_identical_across_every_filtered_view():
    df = p2.add_margins(_fake_table(), verbose=False)
    p2.assert_control_identical_across_views(df)
    # And digit-for-digit, from every direction a reader might filter.
    b = df[df.part == "B_delta"]
    for col in ("encoder", "feature_level", "readout", "feature_set"):
        for v in b[col].unique():
            sub = b[b[col] == v]
            g = sub.groupby(["target", "fold_mode"])["control_score"].nunique()
            assert (g == 1).all(), (
                f"filtering to {col}={v!r} leaves more than one control value")


def test_a_control_that_differs_between_views_is_refused():
    """The decisive companion: corrupt one copy and the assertion must fire."""
    df = p2.add_margins(_fake_table(), verbose=False)
    i = df.index[(df.model_kind == "gap_only") & (df.readout == "norm")][0]
    df.loc[i, "score_mean"] += 1e-9              # digit-for-digit, not eyeball
    with pytest.raises(AssertionError, match="differs between views"):
        p2.assert_control_identical_across_views(df)


def test_a_control_score_that_disagrees_with_the_controls_own_row_is_refused():
    df = p2.add_margins(_fake_table(), verbose=False)
    df["control_score"] = df.control_score + 1e-9
    with pytest.raises(AssertionError):
        p2.assert_control_identical_across_views(df)


def test_the_degenerate_control_is_present_for_every_fold_mode():
    df = _fake_table()
    p2.assert_degenerate_control_present(df)
    assert set(df[df.model_kind == "gap_only"].fold_mode) == set(p2.FOLD_MODES)
    assert set(df[df.model_kind == "retention"].fold_mode) == set(p2.FOLD_MODES)


def test_a_missing_fold_mode_control_is_refused():
    df = _fake_table()
    df = df[~((df.model_kind == "gap_only") & (df.fold_mode == "loco"))]
    with pytest.raises(AssertionError, match="gap-length-alone control"):
        p2.assert_degenerate_control_present(df)


def test_mi_rows_are_flagged_and_excluded_from_the_ranking():
    df = p2.add_k2_verdicts(p2.add_margins(_fake_table(), verbose=False),
                            verbose=False)
    rank = p2.structural_hypothesis(df, verbose=False)
    p2.assert_mi_flagged_and_excluded(df, rank)
    assert not df[df.encoder == p2.MI_ENCODER].si_comparable.any()
    assert p2.MI_ENCODER not in rank["order"]
    assert p2.MI_ENCODER in rank["excluded"]
    # Reported, though: excluded from the RANKING is not excluded from the table.
    assert (df.encoder == p2.MI_ENCODER).sum() > 0


def test_an_mi_row_claiming_si_comparability_is_refused():
    df = p2.add_k2_verdicts(p2.add_margins(_fake_table(), verbose=False),
                            verbose=False)
    rank = p2.structural_hypothesis(df, verbose=False)
    df.loc[df.encoder == p2.MI_ENCODER, "si_comparable"] = True
    with pytest.raises(AssertionError, match="si_comparable=True"):
        p2.assert_mi_flagged_and_excluded(df, rank)


def test_the_structural_ranking_refuses_to_contain_the_multi_image_encoder():
    df = p2.add_k2_verdicts(p2.add_margins(_fake_table(), verbose=False),
                            verbose=False)
    df.loc[df.encoder == p2.MI_ENCODER, "si_comparable"] = True
    with pytest.raises(AssertionError, match="sliding-window increment"):
        p2.structural_hypothesis(df, verbose=False)


def test_the_structural_hypothesis_is_reported_with_its_caveat():
    df = p2.add_k2_verdicts(p2.add_margins(_fake_table(), verbose=False),
                            verbose=False)
    rank = p2.structural_hypothesis(df, verbose=False)
    assert "NOT POWERED" in rank["caveat"]
    assert rank["n_si_points"] == len(rank["order"]) >= 2
    # The fixture scores every fold mode identically, so the order is stable
    # and the hunch gets a real True/False.
    assert rank["order_stable_across_fold_modes"]
    assert isinstance(rank["supported"], bool)
    assert set(rank["order_by_fold_mode"]) == set(p2.FOLD_MODES)


def test_an_ordering_that_flips_between_fold_modes_is_not_determinable():
    """The decisive companion. On the real subset DINOv2 and Satlas SI swap
    places between `cube` and `loco`, so there is no ranking to read the hunch
    off -- and reporting False there would be as wrong as reporting True."""
    df = _fake_table()
    p = p2.STRUCTURAL_PRIMARY
    m = ((df.part == "B_delta") & (df.fold_mode == "loco")
         & (df.delta_target == p["delta_target"])
         & (df.aggregation == p["aggregation"])
         & (df.feature_level == p["feature_level"])
         & (df.readout == p["readout"])
         & (df.feature_set == "embedding"))
    # Push DINOv2 above Satlas under loco only.
    df.loc[m & (df.encoder == "dinov2_vitb14"), "score_mean"] = 0.99
    df = p2.add_k2_verdicts(p2.add_margins(df, verbose=False), verbose=False)
    rank = p2.structural_hypothesis(df, verbose=False)
    assert not rank["order_stable_across_fold_modes"]
    assert rank["supported"] is None, (
        f"an unstable ordering reported supported={rank['supported']!r}; it "
        "must be None -- neither confirmed nor refuted")
    assert len(set(v for v in rank["verdict_by_fold_mode"].values()
                   if v is not None)) == 2


def test_effective_n_counts_cubes_and_not_rows():
    df = _fake_table()
    p2.assert_effective_n_counts_cubes(df)
    assert (df.effective_n == 20).all()
    assert (df.effective_n < df.n_rows_test_total).all()


def test_an_effective_n_that_counts_rows_is_refused():
    df = _fake_table()
    df["effective_n"] = df["n_rows_test_total"]
    with pytest.raises(AssertionError):
        p2.assert_effective_n_counts_cubes(df)


def test_the_results_table_is_complete():
    p2.assert_results_complete(_fake_table())


def test_a_table_missing_the_band_matched_baseline_is_refused():
    df = _fake_table()
    df = df[df.feature_set != p2.BAND_MATCHED_BASELINE]
    with pytest.raises(AssertionError, match="band-matched"):
        p2.assert_results_complete(df)


def test_every_delta_row_declares_the_day_axis_as_its_gap_source():
    df = _fake_table()
    b = df[df.part == "B_delta"]
    assert (b.gap_source == "daily_axis_index").all()
    df.loc[df.part == "B_delta", "gap_source"] = "original_axis_index"
    with pytest.raises(AssertionError, match="daily_axis_index"):
        p2.assert_results_complete(df)


def test_margins_are_relative_to_the_matching_control():
    df = p2.add_margins(_fake_table(), verbose=False)
    r = df[(df.part == "B_delta") & (df.encoder == "dinov2_vitb14")
           & (df.feature_set == "embedding")].iloc[0]
    assert r.margin_over_control == pytest.approx(r.score_mean - 0.2)
    assert r.control_kind == "gap_only"
    a = df[(df.part == "A_reconstruction") & (df.encoder == "dinov2_vitb14")
           & (df.feature_set == "embedding")].iloc[0]
    assert a.margin_over_control == pytest.approx(a.score_mean - (-0.13))
    assert a.control_kind == "retention"


def test_the_band_matched_margin_is_against_the_rgb_only_baseline():
    df = p2.add_margins(_fake_table(), verbose=False)
    r = df[(df.part == "B_delta") & (df.encoder == "dinov2_vitb14")
           & (df.feature_set == "embedding")].iloc[0]
    assert r.margin_over_band_matched == pytest.approx(r.score_mean - 0.45)
    assert r.margin_over_raw == pytest.approx(r.score_mean - 0.44)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def test_spearman_returns_nan_rather_than_zero_on_a_constant_side():
    assert not np.isfinite(p2.spearman(np.ones(20), np.arange(20.0)))
    assert p2.spearman(np.arange(20.0), np.arange(20.0)) == pytest.approx(1.0)
    assert p2.spearman(np.arange(20.0), -np.arange(20.0)) == pytest.approx(-1.0)


def _same_summary(a, b):
    """Summaries equal field by field, treating NaN as equal to NaN.

    A plain `a == b` fails on any NaN because NaN != NaN, which would report a
    difference where there is none -- `n_common_px_median` is legitimately NaN
    for every Part A row.
    """
    assert set(a) == set(b), (set(a) ^ set(b))
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, float) and isinstance(y, float):
            assert (x == y) or (np.isnan(x) and np.isnan(y)), (k, x, y)
        else:
            assert x == y, (k, x, y)
    return True


def test_n_jobs_changes_wall_clock_and_never_a_number():
    """Fold-level parallelism must be bit-identical to serial.

    The ridge path is an exact linear solve, the standardiser is exact, and the
    fold generator has no RNG, so there is no legitimate source of difference.
    This is the test that lets the scaled run use 8 workers and still be quoted
    beside the serial 20-cube numbers.
    """
    m = _manifest(n_cubes=6, frames=6)
    arrays = _arrays(m)
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, len(m))
    block = p2.reconstruction_block(arrays, "grid_cell").with_labels(y)

    serial = p2.evaluate(block, m, "cube", "A_reconstruction", k=3, n_jobs=1)
    par = p2.evaluate(block, m, "cube", "A_reconstruction", k=3, n_jobs=2)
    assert len(serial) == len(par)
    for a, b in zip(serial, par):
        assert a.score == b.score, (a.score, b.score)
        assert a.selected == b.selected
        assert a.n_test == b.n_test and a.effective_n == b.effective_n
    assert _same_summary(p2.summarise(serial), p2.summarise(par))


def test_the_run_log_is_not_thinner_under_parallelism(tmp_path):
    """A loky worker re-imports the module, so its log handle is None and its
    lines vanish. The parallel run is the one the scale-up uses, so a log that
    silently loses its per-fold detail there is a log that documents only the
    configuration nobody runs."""
    m = _manifest(n_cubes=6, frames=6)
    arrays = _arrays(m)
    rng = np.random.default_rng(0)
    block = p2.reconstruction_block(arrays, "grid_cell").with_labels(
        rng.normal(0, 1, len(m)))

    def n_fold_lines(n_jobs, name):
        path = str(tmp_path / name)
        p2.open_run_log(path, verbose=False)
        try:
            p2.evaluate(block, m, "cube", "A_reconstruction", k=3, n_jobs=n_jobs)
        finally:
            p2.close_run_log()
        with open(path) as fh:
            return sum(1 for ln in fh if "A fold" in ln)

    serial, par = n_fold_lines(1, "s.log"), n_fold_lines(2, "p.log")
    assert serial == 3, serial
    assert par == serial, (
        f"the parallel run logged {par} fold lines against serial's {serial}")


def test_n_jobs_is_bit_identical_on_the_delta_probe_too():
    """Part B routes its fold selection through pairs.row_a; the parallel path
    must take the same route, not a re-derived one."""
    m = _manifest(n_cubes=6, frames=6)
    pairs = p2.pair_index(m, verbose=False)
    arrays = _arrays(m)
    rng = np.random.default_rng(1)
    y = np.sign(rng.normal(0, 1, pairs.n_pairs))
    block = p2.delta_block(arrays, pairs, "pooled", "linear").with_labels(y)

    kw = dict(pairs=pairs, readout="linear", k=3)
    serial = p2.evaluate(block, m, "cube", "B_delta", n_jobs=1, **kw)
    par = p2.evaluate(block, m, "cube", "B_delta", n_jobs=2, **kw)
    assert _same_summary(p2.summarise(serial), p2.summarise(par))


def test_summarise_reports_a_cube_clustered_interval_and_never_a_bare_mean():
    res = [p2.FoldResult(fold=i, n_train=100, n_test=20, n_train_cubes=16,
                         n_test_cubes=4, effective_n=4, score=0.3 + 0.05 * i,
                         spearman=0.3, rmse=0.1, selected=1.0,
                         at_grid_edge=False, n_common_px_median=1.0, log="")
           for i in range(5)]
    s = p2.summarise(res)
    assert s["effective_n"] == 20 and s["effective_n_per_fold"] == "4;4;4;4;4"
    assert s["score_ci_lo"] < s["score_mean"] < s["score_ci_hi"]
    assert len(s["per_fold_score"].split(";")) == 5


# ---------------------------------------------------------------------------
# Real data: the guards must be exercised against data of the shape they were
# written for, not only against fixtures.
# ---------------------------------------------------------------------------

@needs_cubes
def test_the_real_subset_disagrees_about_the_gap_on_every_pair():
    """The live check, on the real manifest. A guard that has never fired on
    real data is a hypothesis, not a control."""
    from data.loader import load_cube
    from encoders.manifest import build_manifest

    m = build_manifest([load_cube(p, verbose=False)
                        for p in sorted(glob.glob(os.path.join(RAW, "*.nc")))],
                       verbose=False)
    pairs = p2.pair_index(m, verbose=False)
    info = p2.assert_gap_axes_disagree(pairs, verbose=False)
    assert info["n_axes_equal"] == 0
    assert info["ratio_days_per_acq_step"] >= 2.0
    # The measured lattice: 5 days per acquisition step on Sentinel-2.
    assert info["median_gap_days"] == 10.0
    assert info["median_gap_acq_steps"] == 2.0
    assert all(g % 5 == 0 for g in info["distinct_gap_days"])


@needs_cubes
@needs_cache
def test_the_cached_masks_match_the_canonical_ndvis_finiteness():
    """The intersection is only the set it claims to be if the cached mask and
    the NDVI agree about what is valid."""
    from data.loader import load_cube
    from encoders.pipeline import load_masks

    path = sorted(glob.glob(os.path.join(RAW, "*.nc")))[0]
    cube = os.path.basename(path)
    mp = os.path.join(MASKS, f"{os.path.splitext(cube)[0]}__masks.npz")
    out = p2.cube_common_masked_deltas(load_cube(path, verbose=False),
                                       load_masks(mp), verbose=False)
    assert out["d_cube_mean"].ndim == 1 and out["d_cube_mean"].size >= 1
    assert out["d_cell_mean"].shape == (out["d_cube_mean"].size, GRID_CELLS)
    assert (out["n_common_px"] > 0).all(), (
        "a pair on the real subset has no shared pixel; that is a finding and "
        "belongs in the log, not in a silent drop")


@needs_cubes
@needs_cache
def test_a_stale_mask_cache_is_refused_rather_than_used():
    """The mask cache and the frame selection must describe the same frames."""
    from data.loader import load_cube
    from encoders.pipeline import load_masks

    path = sorted(glob.glob(os.path.join(RAW, "*.nc")))[0]
    cube = os.path.basename(path)
    mp = os.path.join(MASKS, f"{os.path.splitext(cube)[0]}__masks.npz")
    cm = load_masks(mp)
    stale = cm._replace(kept_idx=np.asarray(cm.kept_idx) + 1)
    with pytest.raises(AssertionError, match="kept_idx"):
        p2.cube_common_masked_deltas(load_cube(path, verbose=False), stale,
                                     verbose=False)
