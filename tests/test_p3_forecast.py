"""P3, the forecastability probe: the assertions the spec names, plus the
machinery they rest on.

Most of it runs on a SYNTHETIC manifest with known cube membership, a known day
grid and a known embedding-to-NDVI relation, so each claim is checked against an
INDEPENDENT computation rather than against the code that produced it. The tests
that need real cubes or the scaled cache skip when those are absent.

The five that matter most are the ones nothing downstream could catch:

* ``test_horizon_axis_is_the_day_axis_checked_against_the_live_gap_check`` --
  ``original_axis_index`` is finite, integer, in range, monotone within a cube
  and correlated with ``daily_axis_index``, so a horizon read off it is wrong by
  a factor of five and completely silent. This asserts the two candidates
  DISAGREE, and ties P3's own rows to the ratio
  ``p2_deltas.assert_gap_axes_disagree`` measured on the same manifest -- not to
  a constant typed into a test.
* ``test_mi_context_refuses_a_three_frame_stack`` -- the multi-image encoder's
  embedding at t already pools 8 preceding frames. A 3-stack would double-count
  a lookback the encoder already contains, would change no shape, and would give
  a plausible number. The refusal is exercised, and its message is asserted to
  name the reason.
* ``test_persistence_error_is_the_common_masked_delta_not_two_own_means`` --
  the disjoint-mask case. Two per-frame means over two different valid sets are
  two different pieces of ground; the difference has the same shape, the same
  dtype and a plausible magnitude, so only an independent computation finds it.
* the poison PAIRS on ``fit_readout``, ``p2_deltas.fit_gap_control`` (P3's
  horizon control) and ``p4_ceiling.doy_climatology_within_fold`` (P3's
  climatology baseline) -- test-fold poison must not move the fit, TRAINING-fold
  poison must. The first alone passes on a function that ignores its input.
* ``test_control_score_is_on_every_row_and_identical_across_views`` -- a control
  a reader can filter away is not a control.
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
from probes import p3_forecast as p3
from probes import p4_ceiling as p4

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCALED = os.path.join(REPO, "data", "scaled_32UNU")
SCALED_RAW = os.path.join(SCALED, "raw")
SCALED_EMB = os.path.join(SCALED, "embeddings")
SCALED_MASKS = os.path.join(SCALED, "masks")

needs_scaled = pytest.mark.skipif(
    not glob.glob(os.path.join(SCALED_RAW, "*.nc"))
    or not glob.glob(os.path.join(SCALED_EMB, "*.npz"))
    or not glob.glob(os.path.join(SCALED_MASKS, "*.npz")),
    reason="no data/scaled_32UNU cache; the real-data checks are skipped")


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

def _manifest(n_cubes=10, frames=26, start="2018-04-01", stagger_days=9,
              day_step=5, acq_step=1, clear=0.8):
    """One tile, staggered windows -- the shape of the real subset.

    ``day_step`` and ``acq_step`` are deliberately DIFFERENT and their ratio is
    5, as on the real data. A fixture where the two axes coincided would let a
    probe that read the wrong column pass every test in this file, which is the
    lesson P2 wrote down and this file inherits.
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
    """A fake encoder cache. Shapes only; the values are noise."""
    rng = np.random.default_rng(seed)
    n = len(m)
    return {"encoder": "fake",
            "pooled": rng.normal(0, 1, (n, D)).astype(np.float32),
            "grid": rng.normal(0, 1, (n, GRID_CELLS, Dg)).astype(np.float32),
            "grid_clear_frac": np.full((n, GRID_CELLS), 0.8, dtype=np.float32),
            "clear_frac": np.full(n, 0.8),
            "window_span_days": np.linspace(0, 100, n)}


@pytest.fixture
def manifest():
    return _manifest()


@pytest.fixture
def rows(manifest):
    return p3.horizon_index(manifest, 25, verbose=False)


# ---------------------------------------------------------------------------
# Horizons: the day axis, the drop policy, and the context rule
# ---------------------------------------------------------------------------

def test_horizon_axis_is_the_day_axis_checked_against_the_live_gap_check(manifest):
    """THE test this probe most needs, and it must be able to fail.

    The check is not "delta_days came from the column named daily_axis_index" --
    that is a docstring claim. It is: the two candidate columns DISAGREE on
    every row, by the ratio ``p2_deltas.assert_gap_axes_disagree`` measured
    independently on the consecutive pairs of the same manifest.
    """
    live = p2.assert_gap_axes_disagree(p2.pair_index(manifest, verbose=False),
                                       verbose=False)
    assert live["ratio_days_per_acq_step"] >= 2.0, (
        "the fixture's two axes are not materially different, so this test "
        "cannot distinguish the right column from the wrong one"
    )
    for H in p3.HORIZONS[:3]:
        r = p3.horizon_index(manifest, H, verbose=False)
        out = p3.assert_horizon_axis(r, manifest, live, verbose=False)
        assert out["n_axes_equal"] == 0
        assert out["median_delta_days"] == H
        # the acquisition-axis reading would have been about 5x shorter
        assert out["median_delta_acq_steps"] < 0.5 * H
        np.testing.assert_allclose(out["ratio_days_per_acq_step"],
                                   live["ratio_days_per_acq_step"], rtol=0.5)


def test_horizon_axis_check_rejects_a_row_set_built_off_the_join_key(manifest):
    """A guard only ever run on correct input is a hypothesis.

    Rebuild the rows with ``delta_days`` taken from ``original_axis_index`` --
    the exact mistake -- and assert the check refuses it.
    """
    live = p2.assert_gap_axes_disagree(p2.pair_index(manifest, verbose=False),
                                       verbose=False)
    good = p3.horizon_index(manifest, 25, verbose=False)
    from dataclasses import replace
    bad = replace(good, delta_days=good.delta_acq_steps.astype(float))
    with pytest.raises(AssertionError, match="original_axis_index"):
        p3.assert_horizon_axis(bad, manifest, live, verbose=False)


def test_horizon_index_runs_the_live_gap_check_before_any_horizon(manifest):
    """The imported check is at the TOP of horizon_index, not in its docstring.

    Proved by handing it a manifest whose two axes COINCIDE. On such a manifest
    the day-axis and the join key are the same column, no later assertion could
    tell them apart, and ``p2_deltas.assert_gap_axes_disagree`` is the only
    thing standing between that and a table of horizons five times too short.
    """
    coincident = _manifest(day_step=1, acq_step=1)
    with pytest.raises(AssertionError, match="gap_days == gap_acq_steps"):
        p3.horizon_index(coincident, 5, verbose=False)


def test_target_is_the_nearest_retained_frame_inside_the_same_cube(manifest):
    r = p3.horizon_index(manifest, 25, verbose=False)
    cube = manifest["cube_id"].to_numpy()
    dai = manifest["daily_axis_index"].to_numpy().astype(float)
    # never wraps across cubes -- t, the context and the target are one cube
    for j in range(r.n_rows):
        touched = np.concatenate([r.context_rows[j], [r.row_target[j]]])
        assert len(set(cube[touched].tolist())) == 1
        assert cube[r.row_t[j]] == r.cube_id[j]
    # the target is FORWARD in time and within tolerance of t + Delta
    assert (dai[r.row_target] > dai[r.row_t]).all()
    assert (np.abs(dai[r.row_target] - dai[r.row_t] - 25) <= r.tolerance_days).all()


def test_rows_without_three_prior_frames_are_dropped_not_padded(manifest):
    r = p3.horizon_index(manifest, 5, verbose=False)
    n_cubes = manifest["cube_id"].nunique()
    # the first CONTEXT_FRAMES-1 frames of every cube can never be a t
    first_two = []
    for c in sorted(manifest["cube_id"].unique()):
        pos = np.flatnonzero((manifest["cube_id"] == c).to_numpy())
        pos = pos[np.argsort(manifest["original_axis_index"].to_numpy()[pos])]
        first_two += pos[:p3.CONTEXT_FRAMES - 1].tolist()
    assert not np.isin(r.row_t, first_two).any(), (
        "a row whose context would have needed padding survived. A zero "
        "embedding is not a neutral input and would appear only at the start "
        "of a cube, i.e. only early in the season."
    )
    assert len(first_two) == n_cubes * (p3.CONTEXT_FRAMES - 1)
    # the context is oldest-first and ends at t
    assert (r.context_rows[:, -1] == r.row_t).all()
    assert (np.diff(r.context_rows, axis=1) > 0).all()


def test_the_row_set_is_identical_for_every_encoder(manifest):
    """The k>=3 rule defines the ROW SET once, before any encoder is chosen.

    If MI's single-frame context let it keep rows the others drop, the encoders
    would be scored on different data and the table's n would depend on which
    column a reader filtered to.
    """
    r = p3.horizon_index(manifest, 25, verbose=False)
    a = _arrays(manifest)
    Xsi, _ = p3.context_block(a, r, "dinov2_vitb14", "pooled")
    Xmi, _ = p3.context_block(a, r, p3.MI_ENCODER, "pooled")
    assert Xsi.shape[0] == Xmi.shape[0] == r.n_rows
    assert Xsi.shape[1] == p3.CONTEXT_FRAMES * a["pooled"].shape[1]
    assert Xmi.shape[1] == p3.MI_CONTEXT_FRAMES * a["pooled"].shape[1]


def test_retention_is_printed_and_shrinks_toward_the_long_horizon(manifest, capsys):
    rows = {H: p3.horizon_index(manifest, H, verbose=False)
            for H in (5, 25, 50, 100)}
    table = p3.print_horizon_retention(rows, manifest, verbose=True)
    out = capsys.readouterr().out
    assert "ROWS RETAINED PER HORIZON" in out
    for H in (5, 25, 50, 100):
        assert f"{H:>6d}" in out or f" {H} " in out
    p3.assert_retention_shrinks(table, verbose=True)
    assert table.sort_values("delta_days").n_rows.iloc[-1] < \
        table.sort_values("delta_days").n_rows.iloc[0]


def test_assert_retention_shrinks_refuses_a_table_that_does_not(manifest):
    """The assertion has to be able to fail, or it is decoration.

    A retention table that does NOT fall at the long horizon means the target
    search wrapped, extrapolated, or accepted a frame nowhere near t+Delta.
    """
    fake = pd.DataFrame({"delta_days": [5, 100], "n_rows": [100, 100],
                         "n_cubes": [10, 10], "cube_day_span_median": [135.0] * 2})
    with pytest.raises(AssertionError, match="wrapped across cubes"):
        p3.assert_retention_shrinks(fake, verbose=False)
    # ...and it says so, rather than asserting, where the boundary cannot bite
    short = fake.assign(delta_days=[5, 25])
    p3.assert_retention_shrinks(short, verbose=False)


def test_tolerance_does_no_work_on_a_five_day_lattice(manifest):
    """+/-3 accepts exact matches only here, and the run measures that."""
    t = p3.horizon_tolerance_sensitivity(manifest, horizons=(25, 50),
                                         tolerances=(2, 3), verbose=False)
    a = t[t.tolerance_days == 2].set_index("delta_days").n_rows
    b = t[t.tolerance_days == 3].set_index("delta_days").n_rows
    pd.testing.assert_series_equal(a, b, check_names=False)
    assert (t.n_off_nominal == 0).all(), (
        "a +/-3 day tolerance moved a horizon off its nominal value on a 5-day "
        "lattice; then the tolerance is doing work and must be reported"
    )


# ---------------------------------------------------------------------------
# The multi-image encoder's context
# ---------------------------------------------------------------------------

def test_mi_context_refuses_a_three_frame_stack(manifest, rows):
    """Feed the MI path a k=3 stack and watch it be REFUSED, loudly.

    Not "ignored": a silently-ignored argument is indistinguishable from one
    that was honoured, and the resulting row would claim a 3-frame context
    while carrying a 1-frame one.
    """
    a = _arrays(manifest)
    with pytest.raises(AssertionError) as e:
        p3.context_block(a, rows, p3.MI_ENCODER, "pooled", n_frames=3)
    msg = str(e.value)
    assert "REFUSED" in msg
    assert "double-count" in msg and "8 preceding" in msg, (
        "the refusal does not say WHY; a message that only says 'invalid' "
        "teaches the next caller nothing"
    )
    # ...and the single-image encoders are unaffected by the same call
    X, _ = p3.context_block(a, rows, "dinov2_vitb14", "pooled", n_frames=3)
    assert X.shape == (rows.n_rows, 3 * a["pooled"].shape[1])


def test_context_frames_for_is_the_single_source_of_the_k(manifest):
    assert p3.context_frames_for(p3.MI_ENCODER) == p3.MI_CONTEXT_FRAMES == 1
    for enc in p3.ENCODER_ORDER:
        if enc != p3.MI_ENCODER:
            assert p3.context_frames_for(enc) == p3.CONTEXT_FRAMES == 3


def test_mi_context_is_the_embedding_at_t_and_nothing_else(manifest, rows):
    a = _arrays(manifest)
    X, _ = p3.context_block(a, rows, p3.MI_ENCODER, "pooled")
    np.testing.assert_array_equal(X, a["pooled"][rows.row_t].astype(np.float64))


def test_cell_context_is_built_for_the_TARGET_VIEWS_rows_not_all_16(manifest, rows):
    """A cell with no surviving common pixel is not an observation.

    It is absent from the target, so it must be absent from the design. Building
    the full N x 16 block and hoping it lined up is the mismatch that killed the
    first full run: 8288 feature rows against 8149 target rows at Delta = 5 d.
    """
    a = _arrays(manifest)
    N = rows.n_rows
    # drop one cell of the first three forecast rows, as a dead cell would be
    keep = np.ones((N, GRID_CELLS), dtype=bool)
    keep[0, 3] = keep[1, 7] = keep[2, 0] = False
    row_of = np.repeat(np.arange(N), GRID_CELLS)[keep.reshape(-1)]
    cell_idx = np.tile(np.arange(GRID_CELLS), N)[keep.reshape(-1)]
    X, _ = p3.context_block(a, rows, "dinov2_vitb14", "grid_cell",
                            row_of=row_of, cell_idx=cell_idx)
    assert X.shape[0] == row_of.size == N * GRID_CELLS - 3

    # every feature row holds ITS cell of ITS three context frames, oldest first
    D = a["grid"].shape[-1]
    g = a["grid"].astype(np.float64)
    for i in (0, 5, X.shape[0] - 1):
        r, c = row_of[i], cell_idx[i]
        for j in range(p3.CONTEXT_FRAMES):
            np.testing.assert_array_equal(
                X[i, j * D:(j + 1) * D], g[rows.context_rows[r, j], c])

    # and a grid_cell context without cell_idx is REFUSED, not guessed
    with pytest.raises(AssertionError, match="cell_idx"):
        p3.context_block(a, rows, "dinov2_vitb14", "grid_cell", row_of=row_of)


def test_context_is_oldest_first_and_ends_at_t(manifest, rows):
    a = _arrays(manifest)
    D = a["pooled"].shape[1]
    X, names = p3.context_block(a, rows, "dinov2_vitb14", "pooled")
    np.testing.assert_array_equal(X[:, -D:],
                                  a["pooled"][rows.row_t].astype(np.float64))
    np.testing.assert_array_equal(
        X[:, :D], a["pooled"][rows.context_rows[:, 0]].astype(np.float64))
    assert names[0].endswith("_c0") and "t-2" in names[0] and "t-0" in names[-1]


# ---------------------------------------------------------------------------
# Common-masked levels, and the persistence baseline they define
# ---------------------------------------------------------------------------

def _pair(H=8, W=8, seed=0):
    """Two frames with DISJOINT-ish masks and different NDVI on each side."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.2, 0.5, (H, W))
    b = rng.uniform(0.5, 0.9, (H, W))
    ma = np.zeros((H, W), dtype=bool)
    mb = np.zeros((H, W), dtype=bool)
    ma[:, :6] = True                 # left six columns valid in a
    mb[:, 2:] = True                 # right six columns valid in b
    a[~ma] = np.nan
    b[~mb] = np.nan
    return a, b, ma, mb


def test_persistence_error_is_the_common_masked_delta_not_two_own_means():
    """THE differencing trap, at P3's level.

    Persistence predicts NDVI(t) and is scored against NDVI(t+Delta). If either
    is that frame's OWN mean, the residual compares two different pieces of
    ground. It has the same shape, the same dtype and a plausible magnitude, so
    only an independent computation finds it.
    """
    a, b, ma, mb = _pair()
    lev = p3.common_masked_levels(a, b, ma, mb, grid=2)
    pers, y = lev["cube_mean"]
    residual = y - pers

    both = ma & mb
    np.testing.assert_allclose(pers, np.nanmean(np.where(both, a, np.nan)))
    np.testing.assert_allclose(y, np.nanmean(np.where(both, b, np.nan)))

    # the residual IS p2's common-masked change, digit for digit
    d = p2.common_masked_delta(a, b, ma, mb, grid=2)
    np.testing.assert_allclose(residual, d["d_cube_mean"], rtol=0, atol=1e-12)

    # ...and it is NOT the difference of the two frames' own means
    naive = np.nanmean(b[mb]) - np.nanmean(a[ma])
    assert abs(naive - residual) > 1e-6, (
        "the fixture's masks are not disjoint enough for the two answers to "
        "differ, so this test cannot detect the trap it exists for"
    )


def test_common_masked_levels_are_pinned_to_p2s_delta_at_every_aggregation():
    a, b, ma, mb = _pair(seed=3)
    lev = p3.common_masked_levels(a, b, ma, mb, grid=2)
    d = p2.common_masked_delta(a, b, ma, mb, grid=2)
    for agg, key in (("cube_mean", "d_cube_mean"), ("cube_p90", "d_cube_p90"),
                     ("cell_mean", "d_cell_mean")):
        lo, hi = lev[agg]
        got = np.asarray(hi) - np.asarray(lo)
        want = np.asarray(d[key])
        ok = np.isfinite(want)
        np.testing.assert_allclose(np.asarray(got)[ok], want[ok],
                                   rtol=0, atol=1e-12)


def test_common_masked_levels_refuse_a_non_boolean_mask():
    """An integer cloud-code array is truthy almost everywhere, which would
    silently disable common-masking. p2 owns the refusal; P3 inherits it."""
    a, b, ma, mb = _pair()
    with pytest.raises(AssertionError, match="boolean"):
        p3.common_masked_levels(a, b, ma.astype(np.uint8), mb, grid=2)


def test_common_masked_levels_report_a_pair_with_no_shared_pixel():
    a = np.full((4, 4), 0.5)
    b = np.full((4, 4), 0.8)
    ma = np.zeros((4, 4), dtype=bool); ma[:, :2] = True
    mb = np.zeros((4, 4), dtype=bool); mb[:, 2:] = True
    a[~ma] = np.nan
    b[~mb] = np.nan
    lev = p3.common_masked_levels(a, b, ma, mb, grid=2)
    assert lev["n_common_px"] == 0
    assert np.isnan(lev["cube_mean"]).all()      # reported, never imputed


# ---------------------------------------------------------------------------
# The fitted read-out and the two fitted controls: TWO poison tests each
# ---------------------------------------------------------------------------

def _design(n=60, D=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, D))
    y = X[:, 0] * 2.0 - X[:, 1] + rng.normal(0, 0.1, n)
    tr = np.arange(0, n - 20)
    te = np.arange(n - 20, n)
    return X, y, tr, te


def test_fit_readout_takes_train_pos_positionally():
    """Leakage is prevented by the SIGNATURE, not by discipline. If train_pos
    had a default, the read-out could be fitted on everything by omitting it."""
    sig = inspect.signature(p3.fit_readout)
    p = sig.parameters["train_pos"]
    assert p.default is inspect.Parameter.empty
    assert p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD


@pytest.mark.parametrize("estimator", ["linear", "hgb"])
def test_fit_readout_poison_pair(estimator):
    """Test-fold poison must not move the fit; TRAINING poison must.

    The first alone passes trivially on a function that ignores its input
    entirely, which is why both halves are here.
    """
    X, y, tr, te = _design()
    base, _, _ = p3.fit_readout(X, y, tr, te, estimator)

    y_test_poison = y.copy()
    y_test_poison[te] += 1000.0
    got, _, _ = p3.fit_readout(X, y_test_poison, tr, te, estimator)
    np.testing.assert_array_equal(base, got)

    y_train_poison = y.copy()
    y_train_poison[tr[:5]] += 1000.0
    moved, _, _ = p3.fit_readout(X, y_train_poison, tr, te, estimator)
    assert not np.allclose(base, moved), (
        "poisoning a TRAINING row did not move the fit, so the held-out poison "
        "test above proved nothing"
    )


def test_fit_readout_uses_the_alpha_it_is_given_and_only_for_the_ridge():
    """A selected penalty must actually reach the estimator, and nowhere else.

    An override that were silently ignored would produce a ``nested_cv`` row
    identical to its ``fixed_alpha_D`` twin -- a column claiming a rule the fit
    never followed, which is worse than not having the rule.
    """
    from sklearn.linear_model import Ridge
    X, y, tr, te = _design()
    a, _, _ = p3.fit_readout(X, y, tr, te, "linear", alpha=1e-3)
    b, _, _ = p3.fit_readout(X, y, tr, te, "linear", alpha=1e5)
    assert not np.allclose(a, b), "the alpha override did not reach the ridge"
    assert isinstance(p4.make_estimator("linear", 4, alpha=7.0), Ridge)
    assert p4.make_estimator("linear", 4, alpha=7.0).alpha == 7.0
    assert p4.make_estimator("linear", 4).alpha == 4.0     # the alpha = D rule
    with pytest.raises(AssertionError, match="RIDGE only"):
        p4.make_estimator("hgb", 4, alpha=7.0)
    with pytest.raises(AssertionError, match="alpha must be"):
        p4.make_estimator("linear", 4, alpha=0.0)


def test_select_alpha_poison_pair():
    """The penalty is chosen on the TRAINING fold only -- both directions.

    ``p2_deltas.select_ridge_alpha`` already carries this pair; it is re-run
    here through P3's own wrapper, with P3's row-to-manifest mapping, because a
    guard is only a control in the phase that actually runs it -- and because
    the mapping (a feature row's fold is decided by the manifest row of ``t``)
    is P3's and not P2's.
    """
    m = _manifest(n_cubes=9, frames=12)
    rng = np.random.default_rng(0)
    n = len(m)
    X = rng.normal(size=(n, 5))
    y = X[:, 0] * 3.0 + rng.normal(scale=0.2, size=n)
    row_of = np.arange(n)
    folds = list(cv.folds(m, "cube", k=3, verbose=False))
    tr_rows, te_rows = folds[0]
    tr = np.flatnonzero(np.isin(row_of, tr_rows))

    base = p3.select_alpha(X, y, tr, m, tr_rows, row_of, name="poison")["param"]

    y_test_poison = y.copy()
    y_test_poison[np.isin(row_of, te_rows)] += 1000.0
    got = p3.select_alpha(X, y_test_poison, tr, m, tr_rows, row_of)["param"]
    assert got == base, (
        "poisoning the HELD-OUT rows changed the selected penalty, so the "
        "selection read rows this fold is not allowed to read"
    )

    y_train_poison = y.copy()
    y_train_poison[tr[:20]] += 1000.0
    moved = p3.select_alpha(X, y_train_poison, tr, m, tr_rows, row_of)["param"]
    assert moved != base, (
        "poisoning a TRAINING row did not move the selected penalty, so the "
        "held-out poison test above proved nothing"
    )


def test_select_alpha_refuses_a_tuning_index_wider_than_the_fit():
    """The rows the tuner reads must be the rows the fold's fit reads."""
    m = _manifest(n_cubes=9, frames=12)
    n = len(m)
    X = np.random.default_rng(1).normal(size=(n, 4))
    y = X[:, 0] + 0.1
    row_of = np.arange(n)
    tr_rows, _ = list(cv.folds(m, "cube", k=3, verbose=False))[0]
    tr = np.flatnonzero(np.isin(row_of, tr_rows))
    with pytest.raises(AssertionError, match="same set of rows"):
        p3.select_alpha(X, y, tr[:-5], m, tr_rows, row_of, name="narrow")
    # ...and narrower, too: a tuner that read FEWER rows than the fit would
    # choose a penalty for a model that is not the one being fitted.
    wide = np.concatenate([tr, np.setdiff1d(np.arange(n), tr)[:5]])
    with pytest.raises(AssertionError, match="same set of rows"):
        p3.select_alpha(X, y, wide, m, tr_rows, row_of, name="wide")


def test_horizon_control_poison_pair():
    """P3's horizon control is ``p2_deltas.fit_gap_control``, imported.

    Re-exercised here at P3's level and with P3's horizons, because a guard is
    only a control in the phase that actually runs it.
    """
    delta = np.repeat(np.array(p3.HORIZONS, dtype=float), 25)
    rng = np.random.default_rng(0)
    y = 0.7 - 0.001 * delta + rng.normal(0, 0.02, delta.size)
    tr = np.arange(0, 70)
    te = np.arange(70, delta.size)

    base = p2.fit_gap_control(delta, y, tr, degree=p3.HORIZON_CONTROL_DEGREE)
    y_test = y.copy(); y_test[te] += 5.0
    got = p2.fit_gap_control(delta, y_test, tr, degree=p3.HORIZON_CONTROL_DEGREE)
    np.testing.assert_array_equal(base.coef, got.coef)

    y_train = y.copy(); y_train[tr[:5]] += 5.0
    moved = p2.fit_gap_control(delta, y_train, tr,
                               degree=p3.HORIZON_CONTROL_DEGREE)
    assert not np.allclose(base.coef, moved.coef)


def test_horizon_control_refuses_a_single_horizon():
    """Within ONE horizon Delta is a constant, the degree-2 design is rank 1 and
    the control has nothing to say. The pooled fit is FORCED, not preferred."""
    delta = np.full(60, 25.0)
    y = np.linspace(0.5, 0.9, 60)
    with pytest.raises(AssertionError, match="distinct gap lengths"):
        p2.fit_gap_control(delta, y, np.arange(40),
                           degree=p3.HORIZON_CONTROL_DEGREE)


def test_climatology_baseline_poison_pair():
    """P3's climatology baseline is ``p4_ceiling.doy_climatology_within_fold``,
    imported exactly as it exists. Both halves, at P3's level."""
    doy = np.tile(np.arange(100, 300, 5, dtype=float), 3)
    rng = np.random.default_rng(1)
    y = 0.7 + 0.1 * np.sin(2 * np.pi * doy / 365.25) + rng.normal(0, 0.01, doy.size)
    tr = np.arange(0, 80)
    te = np.arange(80, doy.size)

    base = p4.doy_climatology_within_fold(doy, y, tr)
    y_test = y.copy(); y_test[te] += 9.0
    got = p4.doy_climatology_within_fold(doy, y_test, tr)
    np.testing.assert_array_equal(base.coef, got.coef)

    y_train = y.copy(); y_train[tr[:5]] += 9.0
    moved = p4.doy_climatology_within_fold(doy, y_train, tr)
    assert not np.allclose(base.coef, moved.coef)


def test_climatology_identifiability_is_checked_before_anything_is_fitted(manifest):
    rows = {H: p3.horizon_index(manifest, H, verbose=False)
            for H in (5, 25, 50)}
    counts = p3.assert_climatology_identifiable(manifest, rows,
                                                fold_modes=("cube",), k=3,
                                                verbose=False)
    assert min(counts.values()) >= 2 * p4.CLIMATOLOGY_HARMONICS + 1
    # and it refuses when the season is too thin to identify the curve
    thin = _manifest(n_cubes=4, frames=6, stagger_days=0)
    thin_rows = {5: p3.horizon_index(thin, 5, verbose=False)}
    with pytest.raises(AssertionError, match="distinct target days of year"):
        p3.assert_climatology_identifiable(thin, thin_rows, fold_modes=("cube",),
                                           k=2, verbose=False)


# ---------------------------------------------------------------------------
# Folds: a forecast row can never straddle one
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["cube", "loco", "spatial_block"])
def test_no_forecast_row_straddles_a_fold(manifest, mode):
    r = p3.horizon_index(manifest, 25, verbose=False)
    touched = r.manifest_rows()
    for i, (tr, te) in enumerate(p4.outer_folds(manifest, mode, k=3)):
        in_tr = np.isin(touched, tr)
        in_te = np.isin(touched, te)
        assert (in_tr.all(axis=1) | (~in_tr.any(axis=1))).all()
        assert (in_te.all(axis=1) | (~in_te.any(axis=1))).all()
        a = p3._side_rows(r, tr, i, "train")
        b = p3._side_rows(r, te, i, "test")
        assert not np.intersect1d(a, b).size
        assert a.size + b.size == r.n_rows


def test_side_rows_refuses_a_split_that_cuts_inside_a_cube(manifest):
    """The assertion is what would catch a future mode -- ``temporal`` -- that
    splits inside a cube and leaks a row's own future frame into training."""
    r = p3.horizon_index(manifest, 25, verbose=False)
    cut = np.flatnonzero((manifest["cube_id"] == manifest["cube_id"].iloc[0])
                         .to_numpy())[:4]
    with pytest.raises(AssertionError, match="straddling"):
        p3._side_rows(r, cut, 0, "train")


def test_folds_come_from_probes_cv_and_the_refusals_stand(manifest):
    for mode in ("year", "tile", "crossed"):
        with pytest.raises(Exception):
            list(cv.folds(manifest, mode, k=3, verbose=False))


# ---------------------------------------------------------------------------
# The table: controls, baselines, labels
# ---------------------------------------------------------------------------

_VIEWS = (("forecast", "dinov2_vitb14", "embedding"),
          ("forecast", "dinov2_vitb14_cir", "embedding"),
          ("raw_rgb_only_weather", "raw_features", "raw_rgb_only"),
          ("raw_features_weather", "raw_features", "embedding"),
          ("weather_only", "none", "weather"))
_CONTROL_VIEWS = (("observation", "none", "observation"),
                  ("permutation", "none", "weather"))


def _row(H, kind, enc, fs, est, rule, base, score):
    return {"delta_days": H, "aggregation": "cube_mean", "fold_mode": "cube",
            "estimator": est, "alpha_rule": rule, "feature_base": base,
            "model_kind": kind, "encoder": enc,
            "encoder_base": p3.base_encoder(enc) if enc != "none" else "none",
            "band_composite": p3.band_composite(enc), "feature_set": fs,
            "is_control": kind in p3.CONTROL_KINDS,
            "is_baseline": kind in p3.BASELINE_KINDS,
            "si_comparable": p3.base_encoder(enc) != p3.MI_ENCODER,
            "plausibility_screen": True,
            "plausibility_screen_def": p3.PLAUSIBILITY_SCREEN_LABEL,
            "n_rows_dropped_implausible": 4,
            "r2_mean": score, "r2_std": 0.01, "r2_ci_lo": score - 0.1,
            "r2_ci_hi": score + 0.1, "per_fold_r2": "0.1000;0.2000",
            "effective_n": 10, "r2_pooled": score,
            "r2_pooled_ci_lo": score - 0.1, "r2_pooled_ci_hi": score + 0.1,
            "n_rows_pooled": 40, "skill_vs_persistence": 0.2,
            "alpha_per_fold": "1;1", "alpha_median": 1.0,
            "n_folds_alpha_at_grid_edge": 0}


def _table():
    """A minimal results table with the structure add_margins expects.

    It spans every axis the keys now run over -- both penalty rules on the
    ridge, both feature bases on the model rows, and a ``_cir`` view beside its
    RGB twin -- because a fixture that covered only one value of a key would let
    a key mismatch through.
    """
    rows = []
    for H in (5, 25):
        for est in ("linear", "hgb", "mlp"):
            for rule in p3._alpha_rules_for(est):
                for kind, enc, fs in _VIEWS:
                    for base in p3.FEATURE_BASES:
                        rows.append(_row(
                            H, kind, enc, fs, est, rule, base,
                            0.1 + 0.01 * H + 0.001 * len(kind)
                            + 0.0001 * len(est) + 0.002 * len(rule)
                            + 0.003 * len(base) + 0.0005 * len(enc)))
                for kind, enc, fs in _CONTROL_VIEWS:
                    rows.append(_row(
                        H, kind, enc, fs, est, rule, p3.FEATURE_BASE_NONE,
                        0.1 + 0.01 * H + 0.001 * len(kind)
                        + 0.0001 * len(est) + 0.002 * len(rule)))
        for kind in ("persistence", "climatology_proxy", "horizon_only"):
            rows.append(_row(H, kind, "none", "none", "none",
                             p3.ALPHA_RULE_NA, p3.FEATURE_BASE_NONE, 0.05))
    return pd.DataFrame(rows)


def _payloads(df, seed=0):
    """Held-out (fold, position, y, prediction) arrays for a synthetic table.

    Every row is scored on the SAME (fold, position) grid -- which is what the
    real run guarantees and what makes a paired difference possible -- and each
    row's predictions are noised by its own r2_pooled so no two rows are
    identical.
    """
    rng = np.random.default_rng(seed)
    fold = np.repeat(np.arange(4), 10)
    pos = np.tile(np.arange(10), 4)
    y = rng.normal(size=fold.size)
    out = []
    for s in df.r2_pooled.to_numpy(float):
        out.append({"fold": fold.astype(np.int64), "pos": pos.astype(np.int64),
                    "y": y.copy(),
                    "pred": y + rng.normal(scale=1.0 - min(0.9, abs(s)),
                                           size=y.size)})
    return out


def test_control_score_is_on_every_row_and_identical_across_views():
    """A control a reader can filter away is not a control.

    Both halves are checked: every row carries the value, AND the value equals
    the control's own row. A uniformly wrong copy would satisfy the first alone.
    """
    df = p3.add_margins(_table(), verbose=False)
    assert df.control_score.notna().all()
    for (H, est, rule), sub in df.groupby(["delta_days", "control_estimator",
                                           "control_alpha_rule"]):
        assert sub.control_score.nunique() == 1, (
            f"control_score differs between views at Delta={H}, {est}, {rule}"
        )
        own = df[(df.model_kind == "observation") & (df.delta_days == H)
                 & (df.estimator == est)
                 & (df.alpha_rule == rule)].r2_pooled.iloc[0]
        assert sub.control_score.iloc[0] == own
    p3.assert_control_identical_across_views(df)


def test_assert_control_identical_across_views_catches_a_disagreeing_copy():
    df = p3.add_margins(_table(), verbose=False)
    df.loc[df.index[0], "control_score"] += 0.5
    with pytest.raises(AssertionError, match="control_score"):
        p3.assert_control_identical_across_views(df)


def test_margins_are_taken_against_the_same_estimator():
    """Comparing an MLP to a ridge control would compare two learners, not two
    feature sets -- and so would comparing a tuned ridge to a fixed-alpha one."""
    df = p3.add_margins(_table(), verbose=False)
    for est in ("linear", "hgb", "mlp"):
        for rule in p3._alpha_rules_for(est):
            sub = df[(df.estimator == est) & (df.alpha_rule == rule)
                     & (df.delta_days == 5)]
            ctrl = df[(df.model_kind == "observation") & (df.estimator == est)
                      & (df.alpha_rule == rule)
                      & (df.delta_days == 5)].r2_pooled.iloc[0]
            np.testing.assert_allclose(sub.margin_over_control,
                                       sub.r2_pooled - ctrl)


def test_the_penalty_rule_is_part_of_the_margin_key():
    """The two ridge rules are two LEARNERS. A margin across them would be a
    comparison of penalties dressed up as a comparison of feature sets."""
    df = p3.add_margins(_table(), verbose=False)
    fixed = df[(df.estimator == "linear") & (df.alpha_rule == p3.ALPHA_RULE_FIXED)]
    tuned = df[(df.estimator == "linear") & (df.alpha_rule == p3.ALPHA_RULE_TUNED)]
    assert len(fixed) == len(tuned) and len(fixed)
    assert fixed.control_score.iloc[0] != tuned.control_score.iloc[0]
    assert fixed.band_matched_score.iloc[0] != tuned.band_matched_score.iloc[0]


def test_the_shared_base_is_part_of_the_band_matched_key():
    """A row with the base must be measured against the baseline WITH the base."""
    df = p3.add_margins(_table(), verbose=False)
    for base in p3.FEATURE_BASES:
        sub = df[(df.feature_base == base) & (df.estimator == "linear")
                 & (df.alpha_rule == p3.ALPHA_RULE_FIXED)
                 & (df.delta_days == 5)]
        band = sub[(sub.encoder == p3.BASELINE_ENCODER)
                   & (sub.feature_set == p3.BAND_MATCHED_BASELINE)]
        assert len(band) == 1
        np.testing.assert_allclose(sub.margin_over_band_matched,
                                   sub.r2_pooled - float(band.r2_pooled.iloc[0]))


def test_alpha_rules_and_shared_base_are_complete_and_controls_take_no_base():
    df = p3.add_margins(_table(), verbose=False)
    p3.assert_alpha_rules_present(df)
    p3.assert_shared_base_present(df)
    p3.assert_plausibility_screen_declared(df)
    half = df[~((df.estimator == "linear")
                & (df.alpha_rule == p3.ALPHA_RULE_TUNED))]
    with pytest.raises(AssertionError, match="alpha_rule|both"):
        p3.assert_alpha_rules_present(half)
    nobase = df[df.feature_base == p3.FEATURE_BASE_NONE]
    with pytest.raises(AssertionError, match="feature_base"):
        p3.assert_shared_base_present(nobase)
    poisoned = df.copy()
    poisoned.loc[poisoned.is_control, "feature_base"] = p3.FEATURE_BASE_SHARED
    with pytest.raises(AssertionError, match="control row"):
        p3.assert_shared_base_present(poisoned)


def test_the_screen_must_be_declared_and_cannot_be_mixed():
    df = p3.add_margins(_table(), verbose=False)
    mixed = df.copy()
    mixed.loc[mixed.index[0], "plausibility_screen"] = False
    with pytest.raises(AssertionError, match="mixes screened"):
        p3.assert_plausibility_screen_declared(mixed)
    off = df.copy()
    off["plausibility_screen"] = False
    p3.assert_plausibility_screen_declared(off, required=False)
    with pytest.raises(AssertionError, match="plausibility_screen=False"):
        p3.assert_plausibility_screen_declared(off, required=True)


def test_controls_and_baselines_are_all_present():
    df = p3.add_margins(_table(), verbose=False)
    p3.assert_controls_present(df)
    p3.assert_baselines_present(df)
    dropped = df[df.model_kind != "persistence"]
    with pytest.raises(AssertionError, match="persistence"):
        p3.assert_baselines_present(dropped)
    dropped = df[df.model_kind != "horizon_only"]
    with pytest.raises(AssertionError, match="horizon_only"):
        p3.assert_controls_present(dropped)


def _paired_table(seed=0):
    """The synthetic table with its scores DERIVED from its payloads.

    In the real run ``r2_pooled`` and the paired difference come from the same
    held-out predictions, so they agree by construction and
    ``assert_separability_is_paired`` can check that they do. A fixture whose
    scores were invented independently of its payloads would fail that check for
    a reason that has nothing to do with the code under test.
    """
    df = _table()
    pl = _payloads(df, seed=seed)
    r2, lo, hi = [], [], []
    for p in pl:
        folds = [_fake_fold(i, p["fold"], p) for i in np.unique(p["fold"])]
        v, a, b = p3._pooled_with_fold_jackknife(folds, p3._r2_pooled)
        r2.append(v), lo.append(a), hi.append(b)
    df["r2_pooled"] = r2
    df["r2_pooled_ci_lo"] = lo
    df["r2_pooled_ci_hi"] = hi
    df["n_rows_pooled"] = [int(p["y"].size) for p in pl]
    df = p3.add_paired_separability(df, pl, verbose=False)
    return p3.add_margins(df, verbose=False)


def test_the_paired_difference_is_the_margin_it_is_the_interval_for():
    """An interval on a neighbouring quantity is worse than no interval."""
    df = _paired_table()
    p3.assert_separability_is_paired(df)
    for key, margin in p3.REFERENCE_MARGIN.items():
        d = df[f"paired_diff_vs_{key}"].to_numpy(float)
        m = df[margin].to_numpy(float)
        ok = np.isfinite(d) & np.isfinite(m)
        assert ok.any(), f"no row carries both paired_diff_vs_{key} and {margin}"
        np.testing.assert_allclose(d[ok], m[ok], rtol=0, atol=1e-9)


def test_a_marginal_ci_comparison_cannot_come_back():
    """THE guard. A verdict rebuilt from the two marginal intervals is refused.

    Both naive combinations are tried, because both are what an edit reaches
    for: the sum of the half-widths, and their root-sum-square.
    """
    df = _paired_table()
    ref = p3._reference_index(df)[p3.PRIMARY_REFERENCE]
    ha = ((df.r2_pooled_ci_hi - df.r2_pooled_ci_lo) / 2).to_numpy(float)
    hb = np.where(ref >= 0, ha[np.maximum(ref, 0)], np.nan)
    ref_name = p3.PRIMARY_REFERENCE
    for naive in (ha + hb, np.sqrt(ha ** 2 + hb ** 2)):
        bad = df.copy()
        lo, hi = bad.paired_diff - naive, bad.paired_diff + naive
        sep = (lo > 0) | (hi < 0)
        for suffix in ("", f"_vs_{ref_name}"):
            bad[f"paired_ci_lo{suffix}"] = lo
            bad[f"paired_ci_hi{suffix}"] = hi
            bad[f"separable{suffix}"] = sep
        with pytest.raises(AssertionError, match="marginal"):
            p3.assert_separability_is_paired(bad)


def test_separable_is_exactly_the_paired_interval_excluding_zero():
    df = _paired_table()
    ok = np.isfinite(df.paired_ci_lo) & np.isfinite(df.paired_ci_hi)
    want = (df.paired_ci_lo > 0) | (df.paired_ci_hi < 0)
    assert (df.separable[ok] == want[ok]).all()
    i = int(np.flatnonzero(ok.to_numpy())[0])
    flipped = not bool(want.to_numpy()[i])
    bad = df.copy()
    bad.loc[bad.index[i], "separable"] = flipped
    bad.loc[bad.index[i], f"separable_vs_{p3.PRIMARY_REFERENCE}"] = flipped
    with pytest.raises(AssertionError, match="excludes zero"):
        p3.assert_separability_is_paired(bad)
    # and the unsuffixed copy must BE the primary reference's, not resemble it
    drifted = df.copy()
    drifted.loc[drifted.index[i], "separable"] = flipped
    with pytest.raises(AssertionError, match="unsuffixed"):
        p3.assert_separability_is_paired(drifted)


def test_paired_difference_is_antisymmetric_and_zero_against_itself():
    a, b = _payloads(_table(), seed=3)[:2]
    same = p3.paired_difference(a, a)
    assert abs(same["diff"]) < 1e-12 and not same["separable"]
    fwd, rev = p3.paired_difference(a, b), p3.paired_difference(b, a)
    np.testing.assert_allclose(fwd["diff"], -rev["diff"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(fwd["ci_lo"], -rev["ci_hi"], rtol=0, atol=1e-12)
    assert fwd["n_folds"] == rev["n_folds"] == 4


def test_paired_difference_equals_the_difference_of_the_two_pooled_r2():
    """It is the MARGIN, computed once. The interval is what is new, not the
    point estimate."""
    a, b = _payloads(_table(), seed=7)[:2]

    def r2(p):
        y = p["y"]
        return 1.0 - ((y - p["pred"]) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    np.testing.assert_allclose(p3.paired_difference(a, b)["diff"],
                               r2(a) - r2(b), rtol=0, atol=1e-12)


def test_paired_difference_refuses_rows_scored_on_different_targets():
    a, b = _payloads(_table(), seed=11)[:2]
    b = dict(b, y=b["y"] + 0.5)
    with pytest.raises(AssertionError, match="TARGET"):
        p3.paired_difference(a, b)


def test_the_paired_interval_is_tighter_than_the_marginal_one():
    """The reason pairing is not a formality: the shared fold effect cancels.

    Not asserted as a universal law -- it is asserted where the two rows are
    strongly correlated across folds, which is the regime every comparison in
    this table lives in, because both rows are fitted on the same folds.
    """
    rng = np.random.default_rng(0)
    fold = np.repeat(np.arange(6), 20)
    shared = rng.normal(scale=1.0, size=6)[fold]     # a big per-fold effect
    y = rng.normal(size=fold.size) + shared
    a = {"fold": fold, "pos": np.arange(fold.size), "y": y,
         "pred": y - shared - 0.30 * rng.normal(size=y.size)}
    b = {"fold": fold, "pos": np.arange(fold.size), "y": y,
         "pred": y - shared - 0.35 * rng.normal(size=y.size)}
    pr = p3.paired_difference(a, b)
    half_paired = (pr["ci_hi"] - pr["ci_lo"]) / 2

    def marginal_half(p):
        r = p3._pooled_with_fold_jackknife(
            [_fake_fold(i, fold, p) for i in range(6)], p3._r2_pooled)
        return (r[2] - r[1]) / 2

    naive = marginal_half(a) + marginal_half(b)
    assert half_paired < naive, (half_paired, naive)


def _fake_fold(i, fold, p):
    m = fold == i
    return p3.P3FoldResult(
        fold=i, n_train=0, n_test=int(m.sum()), n_train_cubes=0,
        n_test_cubes=1, effective_n=1, r2=np.nan, rmse=0.0, mae=0.0, sse=0.0,
        sse_persistence=0.0, sse_climatology=np.nan, sst=0.0,
        same_cube_after_permutation=np.nan, converged=True, log="",
        test_pos=np.flatnonzero(m), test_y=p["y"][m], test_pred=p["pred"][m],
        test_persistence=p["y"][m])


def test_a_cir_row_is_paired_against_its_own_rgb_twin():
    df = _paired_table()
    cir = df[df.encoder == "dinov2_vitb14_cir"]
    assert len(cir) and cir.has_rgb_twin.all()
    assert not df[df.encoder == "dinov2_vitb14"].has_rgb_twin.any()
    p3.assert_cir_twins_present(df, encoders=("dinov2_vitb14_cir",))
    # the twin is matched on the LEARNER and the BASE, not just the name
    r = cir.iloc[0]
    twin = df[(df.encoder == "dinov2_vitb14")
              & (df.delta_days == r.delta_days)
              & (df.estimator == r.estimator)
              & (df.alpha_rule == r.alpha_rule)
              & (df.feature_base == r.feature_base)]
    assert len(twin) == 1
    np.testing.assert_allclose(r.margin_over_rgb_twin,
                               r.r2_pooled - float(twin.r2_pooled.iloc[0]))
    orphan = df[df.encoder != "dinov2_vitb14"]
    with pytest.raises(AssertionError, match="RGB twin"):
        p3.assert_cir_twins_present(orphan, encoders=("dinov2_vitb14_cir",))


def test_add_paired_separability_refuses_a_payload_count_mismatch():
    df = _table()
    with pytest.raises(AssertionError, match="fold payloads"):
        p3.add_paired_separability(df, _payloads(df)[:-1], verbose=False)


def test_the_five_baselines_and_three_controls_are_named_and_disjoint():
    assert len(p3.BASELINE_KINDS) == 5
    assert len(p3.CONTROL_KINDS) == 3
    assert not set(p3.BASELINE_KINDS) & set(p3.CONTROL_KINDS)
    assert set(p3.MODEL_KINDS) == {"forecast"} | set(p3.BASELINE_KINDS) \
        | set(p3.CONTROL_KINDS)


def test_climatology_rows_are_labelled_proxy_not_stage_b():
    df = p3.add_margins(_table(), verbose=False)
    df["climatology_def"] = np.where(df.model_kind == "climatology_proxy",
                                     p3.PROXY_CLIMATOLOGY_LABEL, "")
    df["is_proxy_climatology"] = df.model_kind == "climatology_proxy"
    p3.assert_climatology_rows_labelled(df)
    assert "NOT STAGE B" in p3.PROXY_CLIMATOLOGY_LABEL
    assert "not H1" in p3.PROXY_CLIMATOLOGY_LABEL
    df.loc[df.model_kind == "climatology_proxy", "climatology_def"] = ""
    with pytest.raises(AssertionError, match="proxy label"):
        p3.assert_climatology_rows_labelled(df)


def test_every_table_invariant_survives_a_CSV_ROUND_TRIP(tmp_path):
    """The CSV is what anyone else reads, so the assertions must hold ON IT.

    This is not hypothetical. ``climatology_def`` is an empty string on every
    non-climatology row, an empty string written to CSV reads back as NaN, and a
    naive ``== ""`` therefore passed on the in-memory table and FAILED on the
    artefact that table was written to -- the worst direction for an assertion
    to fail in.
    """
    df = p3.add_margins(_table(), verbose=False)
    df["climatology_def"] = np.where(df.model_kind == "climatology_proxy",
                                     p3.PROXY_CLIMATOLOGY_LABEL, "")
    df["is_proxy_climatology"] = df.model_kind == "climatology_proxy"
    path = tmp_path / "p3.csv"
    df.to_csv(path, index=False)
    back = pd.read_csv(path)
    assert back.shape == df.shape
    p3.assert_climatology_rows_labelled(back)
    p3.assert_control_identical_across_views(back)
    p3.assert_controls_present(back)
    p3.assert_baselines_present(back)


def test_mi_rows_are_flagged_and_report_one_context_frame():
    df = _table()
    df = pd.concat([df, pd.DataFrame([{
        "delta_days": 5, "aggregation": "cube_mean", "fold_mode": "cube",
        "estimator": "linear", "model_kind": "forecast",
        "encoder": p3.MI_ENCODER, "feature_set": "embedding",
        "is_control": False, "is_baseline": False, "r2_mean": 0.3,
        "r2_std": 0.01, "r2_ci_lo": 0.2, "r2_ci_hi": 0.4,
        "per_fold_r2": "0.3000;0.3000", "effective_n": 10, "r2_pooled": 0.3,
        "r2_pooled_ci_lo": 0.2, "r2_pooled_ci_hi": 0.4,
        "skill_vs_persistence": 0.1}])], ignore_index=True)
    df["si_comparable"] = df.encoder != p3.MI_ENCODER
    df["context_frames"] = np.where(df.encoder == p3.MI_ENCODER,
                                    p3.MI_CONTEXT_FRAMES,
                                    np.where(df.encoder == "none", 0,
                                             p3.CONTEXT_FRAMES))
    df.loc[df.encoder == "none", "context_frames"] = p3.CONTEXT_FRAMES
    p3.assert_mi_flagged_and_single_frame(df)
    df.loc[df.encoder == p3.MI_ENCODER, "context_frames"] = 3
    with pytest.raises(AssertionError, match="context frames"):
        p3.assert_mi_flagged_and_single_frame(df)


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------

# The cheapest roster that still exercises every axis: the not-a-network
# baseline (which carries the band-matched view too), and the MULTI-IMAGE
# encoder with its colour-infrared twin. The MI pair is chosen over DINOv2's
# deliberately -- it is k=1 context at D=1024 rather than k=3 at D=3840, so it
# is a fraction of the cost, and it is the one pair that also exercises the
# si_comparable flag and the single-frame refusal on BOTH composites.
_E2E_ENCODERS = ("raw_features", "satlas_s2_swinb_mi_rgb",
                 "satlas_s2_swinb_mi_rgb_cir")


@pytest.fixture(scope="module")
def real_manifest():
    from data.loader import load_cube
    from encoders.manifest import build_manifest
    # STRIDED, not the first 14: the cube filenames sort by window-start date,
    # so the first 14 share a narrow slice of the season and the proxy
    # climatology is not identifiable on them. A stride spans the year.
    paths = sorted(glob.glob(os.path.join(SCALED_RAW, "*.nc")))[::8][:14]
    return build_manifest([load_cube(p, verbose=False) for p in paths],
                          verbose=False)


@needs_scaled
def test_real_horizons_use_the_day_axis_and_the_axes_disagree(real_manifest):
    live = p2.assert_gap_axes_disagree(
        p2.pair_index(real_manifest, verbose=False), verbose=False)
    assert live["n_axes_equal"] == 0
    for H in p3.HORIZONS:
        r = p3.horizon_index(real_manifest, H, verbose=False)
        if not r.n_rows:
            continue
        assert (r.delta_days == H).all()
        assert (r.delta_acq_steps < r.delta_days).all()


@needs_scaled
def test_real_weather_window_is_exactly_t_to_t_plus_delta(real_manifest):
    """``days_available == Delta + 1`` is what PROVES the window starts at t;
    a clipped window would silently cover a different period."""
    r = p3.horizon_index(real_manifest, 25, verbose=False)
    src = p3.horizon_weather(real_manifest, SCALED_RAW, r, verbose=False)
    assert src.values.shape == (r.n_rows, len(src.names))
    assert src.permutable and src.frame_level
    assert all(n.endswith("_horizon") for n in src.names)
    # an independent re-derivation of one row's window mean temperature
    from encoders.manifest import cube_daily_axis, cube_weather, frame_daily_positions
    j = 0
    path = os.path.join(SCALED_RAW, str(r.cube_id[j]))
    axis, w = cube_daily_axis(path), cube_weather(path)
    ts = np.asarray(real_manifest["timestamp"].to_numpy(), dtype="datetime64[ns]")
    lo = frame_daily_positions(axis, ts[[r.row_t[j]]])[0]
    hi = frame_daily_positions(axis, ts[[r.row_target[j]]])[0]
    assert hi - lo == 25
    want = float(w["eobs_tg"][lo:hi + 1].mean())
    col = src.names.index("tg_mean_horizon")
    np.testing.assert_allclose(src.values[j, col], want, rtol=0, atol=1e-9)


@needs_scaled
def test_real_persistence_residual_is_p2s_common_masked_delta(real_manifest):
    rows = {25: p3.horizon_index(real_manifest, 25, verbose=False)}
    tgt = p3.build_forecast_targets(rows, real_manifest, SCALED_RAW,
                                    mask_dir=SCALED_MASKS, verbose=False)
    y = tgt[25]["cube_mean"]["y"]
    pers = tgt[25]["cube_mean"]["persistence"]
    assert y.shape == pers.shape == (rows[25].n_rows,)
    assert np.isfinite(y).all() and np.isfinite(pers).all()
    # the residual is a CHANGE, so its median magnitude is small while the
    # levels themselves are around 0.7 -- if persistence were scored against
    # the target frame's own mean this relation would not hold
    assert np.median(np.abs(y - pers)) < 0.15 < np.median(y)


@needs_scaled
def test_real_plausibility_screen_drops_rows_that_touch_a_cloud_frame(real_manifest):
    """The screen is APPLIED here, and it removes rows rather than frames.

    A frame cannot be removed from the manifest: the embedding join contract is
    ``(cube_id, original_axis_index) == (cube, kept_idx)`` against a cache built
    over every retained frame, so a shortened manifest would fail the join. The
    row is what goes.
    """
    from data.loader import load_cube

    order = sorted(real_manifest["cube_id"].unique().tolist())
    per_cube = {"__order__": order}
    for c in order:
        per_cube[c] = p4.cube_frame_targets(
            load_cube(os.path.join(SCALED_RAW, str(c)), verbose=False),
            verbose=False)
    ok = p3.manifest_frame_plausible(real_manifest, per_cube, verbose=False)
    assert ok.shape == (len(real_manifest),) and ok.dtype == bool

    off = p3.horizon_index(real_manifest, 25, verbose=False)
    on = p3.horizon_index(real_manifest, 25, frame_plausible=ok, verbose=False)
    assert not off.plausibility_screen and on.plausibility_screen
    assert on.n_rows + on.n_dropped_implausible == off.n_rows
    assert on.n_rows <= off.n_rows
    # every frame of every surviving row is plausible -- context, t and target
    assert ok[on.manifest_rows()].all()
    if int((~ok).sum()):
        assert on.n_rows < off.n_rows, (
            "an implausible frame exists in this manifest but no row was "
            "dropped; the screen reached the row set but did nothing"
        )


@needs_scaled
def test_real_twin_caches_are_not_the_same_arrays(real_manifest):
    """The one way this phase could produce a persuasive zero.

    Reading the colour-infrared views out of the RGB directory does not raise --
    the files simply are not there under that name -- but a mis-wired default
    that pointed both at one cache would give every twin difference of exactly
    zero, which looks like a finding.
    """
    from probes.p1_appearance import load_encoder_arrays
    a = load_encoder_arrays(real_manifest, "dinov2_vitb14_cir",
                            emb_dir=p3.encoder_embeddings_dir(
                                "dinov2_vitb14_cir", SCALED_EMB), verbose=False)
    b = load_encoder_arrays(real_manifest, "dinov2_vitb14", emb_dir=SCALED_EMB,
                            verbose=False)
    p3._assert_twins_are_distinct({"dinov2_vitb14_cir": a,
                                   "dinov2_vitb14": b}, verbose=False)
    with pytest.raises(AssertionError, match="IDENTICAL"):
        p3._assert_twins_are_distinct({"dinov2_vitb14_cir": b,
                                       "dinov2_vitb14": b}, verbose=False)


@needs_scaled
def test_real_run_end_to_end_is_shaped_and_labelled(real_manifest, tmp_path):
    """One aggregation, three horizons -- the whole path, cheaply.

    ``log_path`` is a tmp file ON PURPOSE. The default is
    ``data/phase1_8/logs/p3_run.log``, and a test that writes there truncates
    the phase's own run log -- which is the archived evidence for the published
    table, and which a notebook may be writing at the same moment.
    """
    encs = _E2E_ENCODERS
    df, data = p3.run_p3(real_manifest, SCALED_RAW, horizons=(5, 25, 50),
                         aggregations=("cube_mean",), encoders=encs,
                         emb_dir=SCALED_EMB, mask_dir=SCALED_MASKS, k=3,
                         log_path=str(tmp_path / "p3_test_run.log"),
                         n_jobs=1, verbose=False)
    p3.close_run_log()
    assert len(df)
    df = p3.add_margins(df, verbose=False)
    p3.assert_controls_present(df)
    p3.assert_baselines_present(df)
    p3.assert_control_identical_across_views(df)
    p3.assert_climatology_rows_labelled(df)
    p3.assert_mi_flagged_and_single_frame(df)
    p3.assert_effective_n_counts_cubes(df)
    p3.assert_alpha_rules_present(df)
    p3.assert_shared_base_present(df)
    p3.assert_plausibility_screen_declared(df)
    p3.assert_cir_twins_present(df, encoders=("satlas_s2_swinb_mi_rgb_cir",))
    p3.assert_separability_is_paired(df)
    assert (df.horizon_source == "daily_axis_index").all()
    assert df.common_masked.all()
    assert (df.effective_n <= df.n_cubes).all()
    # persistence must be exactly itself
    pers = df[df.model_kind == "persistence"]
    np.testing.assert_allclose(pers.skill_vs_persistence, 0.0, atol=1e-12)
    # the shared base ADDS exactly one column, and it is NDVI(t)
    for enc in encs:
        r = df[(df.encoder == enc) & (df.feature_set == "embedding")
               & (df.estimator == "linear")
               & (df.alpha_rule == p3.ALPHA_RULE_FIXED)
               & (df.delta_days == 25)]
        wide = float(r[r.feature_base == p3.FEATURE_BASE_SHARED].D.iloc[0])
        narrow = float(r[r.feature_base == p3.FEATURE_BASE_NONE].D.iloc[0])
        assert wide == narrow + 1, (enc, wide, narrow)
    # the tuned rule really tuned: at least one row moved off alpha = D
    tuned = df[(df.estimator == "linear")
               & (df.alpha_rule == p3.ALPHA_RULE_TUNED)]
    assert (tuned.alpha_median.to_numpy() != tuned.D.to_numpy()).any(), (
        "every nested-CV row selected exactly alpha = D, which would mean the "
        "selection never ran"
    )


@needs_scaled
def test_real_run_survives_a_CSV_ROUND_TRIP_with_the_new_columns(real_manifest,
                                                                tmp_path):
    """The CSV is what anyone else reads, so the new invariants must hold ON IT."""
    df, _ = p3.run_p3(real_manifest, SCALED_RAW, horizons=(5, 25, 50),
                      aggregations=("cube_mean",),
                      encoders=_E2E_ENCODERS,
                      emb_dir=SCALED_EMB, mask_dir=SCALED_MASKS, k=3,
                      log_path=str(tmp_path / "p3_roundtrip.log"),
                      n_jobs=1, verbose=False)
    p3.close_run_log()
    df = p3.add_margins(df, verbose=False)
    path = tmp_path / "p3.csv"
    df.to_csv(path, index=False)
    back = pd.read_csv(path)
    assert back.shape == df.shape
    p3.assert_separability_is_paired(back)
    p3.assert_alpha_rules_present(back)
    p3.assert_shared_base_present(back)
    p3.assert_plausibility_screen_declared(back)
    p3.assert_cir_twins_present(back, encoders=("satlas_s2_swinb_mi_rgb_cir",))
    p3.assert_control_identical_across_views(back)
    p3.assert_climatology_rows_labelled(back)
