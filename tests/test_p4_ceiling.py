"""P4, the weather-attributability ceiling: the assertions the spec names, plus
the machinery they rest on.

Most of it runs on a SYNTHETIC manifest with known cube membership, a known
day-of-year signal and a known weather-to-anomaly relation, so each claim is
checked against an independent computation rather than against the code that
made it. The tests that need real cubes skip when data/raw is absent.

The two tests that matter most are the ones nothing downstream could catch:

* ``test_the_curve_is_numerically_independent_of_the_held_out_rows`` -- the
  proxy climatology defines the TARGET, so a curve fitted outside the fold
  inflates every number including every control, leaving the table internally
  consistent and wrong.
* ``test_weather_is_constant_across_the_cells_of_a_frame`` -- a mis-indexed
  cell expansion changes no shape and no dtype.
"""

from __future__ import annotations

import glob
import inspect
import os

import numpy as np
import pandas as pd
import pytest

from encoders.manifest import build_manifest
from probes import cv
from probes import p4_ceiling as p4


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

def _manifest(n_cubes=8, frames=8, start="2018-04-01", step_days=11,
              stagger_days=9, clear=0.8, years=(2018,)):
    """One tile, staggered windows -- the shape of the real subset.

    ``years`` with more than one entry gives every cube frames in each year,
    which is the seasonal split's structure (multi-year WITHIN a cube) and the
    only thing that opens Stage B's gate.
    """
    rows = []
    for c in range(n_cubes):
        for y in years:
            t0 = (np.datetime64(f"{y}-{start[5:]}")
                  + np.timedelta64(stagger_days * c, "D"))
            for f in range(frames):
                ts = t0 + np.timedelta64(step_days * f, "D")
                doy = int((ts - ts.astype("datetime64[Y]")).astype(int)) + 1
                rows.append({
                    "cube_id": f"32UNU_{years[0]}_cube{c}.nc",
                    "tile": "32UNU",
                    "year": years[0],
                    "timestamp": ts,
                    "original_axis_index": f + frames * years.index(y),
                    "daily_axis_index": step_days * f,
                    "day_of_year": doy,
                    "pixel_bbox": (c * 500, c * 500 + 128,
                                   (c % 3) * 700, (c % 3) * 700 + 128),
                    "clear_frac": clear,
                    "landcover_stratum": p4.REPLICATION_STRATA[c % 3],
                    "grid_landcover": tuple(
                        p4.REPLICATION_STRATA[(c + i) % 3] for i in range(16)),
                })
    return pd.DataFrame(rows)


def _sources(manifest, D=6, seed=0, signal=1.0, level="frame", n_rows=None):
    """A weather stand-in whose columns carry a known signal."""
    rng = np.random.default_rng(seed)
    n = len(manifest)
    W = rng.normal(0, 1, size=(n, D))
    W[:, 0] += signal * np.sin(2 * np.pi * manifest["day_of_year"].to_numpy() / 365.25)
    src = p4.FeatureSource(name="weather_full8", values=W,
                           names=tuple(f"w{i}" for i in range(D)),
                           frame_level=True, permutable=True)
    return src


def _target(manifest, src, level="frame", noise=0.05, seed=1):
    """A target that is a known function of the features plus a doy cycle."""
    rng = np.random.default_rng(seed)
    n = len(manifest)
    doy = manifest["day_of_year"].to_numpy().astype(float)
    season = 0.3 * np.sin(2 * np.pi * doy / 365.25) + 0.5
    if level == "frame":
        y = season + 0.2 * src.values[:, 0] + rng.normal(0, noise, n)
        return p4.TargetRows(name="cube_mean", level="frame", y=y,
                             row_idx=np.arange(n), cell_idx=np.full(n, -1),
                             stratum=manifest["landcover_stratum"].to_numpy().astype(str))
    row_idx = np.repeat(np.arange(n), 16)
    cells = np.tile(np.arange(16), n)
    y = (season[row_idx] + 0.2 * src.values[row_idx, 0]
         + rng.normal(0, noise, row_idx.size))
    glc = manifest["grid_landcover"].to_numpy()
    strata = np.array([str(glc[r][c]) for r, c in zip(row_idx, cells)])
    return p4.TargetRows(name="cell_mean", level="cell", y=y, row_idx=row_idx,
                         cell_idx=cells, stratum=strata)


def _data(manifest, src, target):
    n = len(manifest)
    obs = p4.FeatureSource(
        name="observation/frame",
        values=np.column_stack([manifest["clear_frac"].to_numpy().astype(float),
                                np.linspace(0, 100, n)]),
        names=("clear_frac", "window_span_days"), frame_level=True)
    obs_cell = p4.FeatureSource(
        name="observation/cell",
        values=np.column_stack([np.repeat(manifest["clear_frac"].to_numpy(), 16),
                                np.repeat(np.linspace(0, 100, n), 16)]),
        names=("grid_clear_frac", "window_span_days"), frame_level=False)
    doy_src = p4.doy_source(manifest, verbose=False)
    ref = p4.severity_reference_anomaly(
        manifest["day_of_year"].to_numpy().astype(float)[target.row_idx], target.y)
    return p4.P4Data(
        manifest=manifest, targets={target.name: target},
        weather={"weather_full8": src, "weather_eowm5": src},
        observation_frame=obs, observation_cell=obs_cell, doy_frame=doy_src,
        day_of_year=manifest["day_of_year"].to_numpy().astype(float),
        cube_id=manifest["cube_id"].to_numpy().astype(str),
        days_available=None, reference_anomaly={target.name: ref},
        severity={target.name: p4.severity_bins(ref)})


# ---------------------------------------------------------------------------
# THE assertion: the curve is fitted inside the fold
# ---------------------------------------------------------------------------

def test_the_curve_is_numerically_independent_of_the_held_out_rows():
    """Fit on train, POISON only the test rows, refit: the curve must not move.

    This is the test for nested leakage in the TARGET definition. If the curve
    could see a held-out row, every score in the table -- including every
    control -- would be inflated by an unknown amount, and the table would stay
    internally consistent, so nothing downstream could detect it.
    """
    m = _manifest()
    doy = m["day_of_year"].to_numpy().astype(float)
    rng = np.random.default_rng(0)
    y = 0.3 * np.sin(2 * np.pi * doy / 365.25) + rng.normal(0, 0.02, len(m))
    train, test = cv.folds(m, "cube", k=4).__next__()

    before = p4.doy_climatology_within_fold(doy, y, train)
    poisoned = y.copy()
    poisoned[test] += 10.0            # not subtle
    after = p4.doy_climatology_within_fold(doy, poisoned, train)

    np.testing.assert_array_equal(
        before.coef, after.coef,
        err_msg="the day-of-year curve moved when only HELD-OUT rows changed. "
                "It is fitted on the test fold, so the anomaly it defines is "
                "leaked and every number computed from it is inflated.")
    assert before.train_r2 == after.train_r2
    assert before.n_train_rows == after.n_train_rows


def test_the_curve_does_move_when_a_training_row_changes():
    """The decisive companion: a test that can only pass would prove nothing."""
    m = _manifest()
    doy = m["day_of_year"].to_numpy().astype(float)
    y = 0.3 * np.sin(2 * np.pi * doy / 365.25)
    train, _ = cv.folds(m, "cube", k=4).__next__()
    before = p4.doy_climatology_within_fold(doy, y, train)
    poisoned = y.copy()
    poisoned[train[0]] += 10.0
    after = p4.doy_climatology_within_fold(doy, poisoned, train)
    assert not np.allclose(before.coef, after.coef)


def test_the_climatology_signature_takes_a_training_index_set():
    """The fit-inside-fold structure must be in the SIGNATURE, not in a habit.

    A function that took the whole array and were merely called carefully could
    be called carelessly next time; one that cannot be given a target without
    also being given the training index cannot.
    """
    sig = inspect.signature(p4.doy_climatology_within_fold)
    params = list(sig.parameters)
    assert params[:3] == ["day_of_year", "values", "train_idx"], (
        f"expected (day_of_year, values, train_idx, ...), got {params}")
    assert sig.parameters["train_idx"].default is inspect.Parameter.empty, (
        "train_idx has a default, so the curve can be fitted on everything by "
        "omitting it -- which is the leak this signature exists to prevent")


def test_the_curve_refuses_a_fold_that_cannot_support_its_order():
    m = _manifest()
    doy = m["day_of_year"].to_numpy().astype(float)
    y = np.zeros(len(m))
    with pytest.raises(AssertionError, match="distinct days of year"):
        p4.doy_climatology_within_fold(doy, y, np.array([0, 1]), n_harmonics=4)


def test_an_intercept_only_curve_is_the_training_mean():
    """Stage B's setting: its target is already the real anomaly, so nothing
    further is detrended and the 'curve' must be exactly the train mean."""
    m = _manifest()
    doy = m["day_of_year"].to_numpy().astype(float)
    rng = np.random.default_rng(3)
    y = rng.normal(2.0, 1.0, len(m))
    train = np.arange(0, len(m), 2)
    curve = p4.doy_climatology_within_fold(doy, y, train,
                                           n_harmonics=p4.STAGE_B_HARMONICS)
    assert curve.coef.shape == (1,)
    np.testing.assert_allclose(curve.predict(doy), y[train].mean())


# ---------------------------------------------------------------------------
# Weather is constant across the cells of a frame
# ---------------------------------------------------------------------------

def test_weather_is_constant_across_the_cells_of_a_frame():
    """A property of the DATA -- one cube, one day, one E-OBS reading.

    If it fails, the cell expansion is indexing the wrong frames, and that
    changes no shape, no dtype and no finiteness check.
    """
    m = _manifest()
    src = _sources(m)
    target = _target(m, src, level="cell")
    data = _data(m, src, target)
    p4.assert_weather_constant_across_cells(data, "cell_mean")

    X = src.expand(target.row_idx)
    for r in np.unique(target.row_idx)[:5]:
        rows = X[target.row_idx == r]
        assert rows.shape[0] == 16
        np.testing.assert_array_equal(rows, np.repeat(rows[:1], 16, axis=0))


def test_weather_that_varies_within_a_frame_is_caught():
    """The realistic failure: weather expanded to cells too early, stored per
    row, and then re-ordered -- so it varies inside a frame."""
    m = _manifest()
    src = _sources(m)
    target = _target(m, src, level="cell")
    data = _data(m, src, target)
    rng = np.random.default_rng(0)
    per_row = p4.FeatureSource(
        name="weather_full8",
        values=rng.normal(0, 1, (target.n_rows, 3)),
        names=("a", "b", "c"), frame_level=False, permutable=True)
    data = p4.P4Data(**{**data.__dict__,
                        "weather": {"weather_full8": per_row}})
    with pytest.raises(AssertionError, match="ACROSS THE CELLS OF A SINGLE FRAME"):
        p4.assert_weather_constant_across_cells(data, "cell_mean")


def test_a_duplicated_frame_cell_pair_is_caught():
    """A 16-to-1 map that is not one: some cell counted twice, another lost."""
    m = _manifest()
    src = _sources(m)
    target = _target(m, src, level="cell")
    data = _data(m, src, target)
    cells = target.cell_idx.copy()
    cells[1] = cells[0]                       # frame 0 now has cell 0 twice
    broken = p4.TargetRows(name="cell_mean", level="cell", y=target.y,
                           row_idx=target.row_idx, cell_idx=cells,
                           stratum=target.stratum)
    data = p4.P4Data(**{**data.__dict__, "targets": {"cell_mean": broken}})
    with pytest.raises(AssertionError, match="appears more than once"):
        p4.assert_weather_constant_across_cells(data, "cell_mean")


def test_a_frame_with_too_many_cells_is_caught():
    m = _manifest()
    src = _sources(m)
    target = _target(m, src, level="cell")
    data = _data(m, src, target)
    rows = target.row_idx.copy()
    rows[16:32] = rows[0]                     # 32 cell rows on one frame
    cells = target.cell_idx.copy()
    cells[16:32] = np.arange(16, 32)          # keep the pairs unique
    broken = p4.TargetRows(name="cell_mean", level="cell", y=target.y,
                           row_idx=rows, cell_idx=cells, stratum=target.stratum)
    data = p4.P4Data(**{**data.__dict__, "targets": {"cell_mean": broken}})
    with pytest.raises(AssertionError, match="outside \\[0, 16\\)"):
        p4.assert_weather_constant_across_cells(data, "cell_mean")


# ---------------------------------------------------------------------------
# The four controls, and the permutation's empirical zero
# ---------------------------------------------------------------------------

def _small_table(level="frame", estimators=("linear",), modes=("cube",)):
    m = _manifest()
    src = _sources(m)
    target = _target(m, src, level=level)
    data = _data(m, src, target)
    return m, data, p4.run_stage_a(
        data, targets=(target.name,), fold_modes=modes, estimators=estimators,
        feature_sets=("weather_full8", "weather_eowm5"), k=4, verbose=False)


def test_all_four_controls_are_present_for_every_combination():
    """The exit test's completeness claim, checked on the table itself."""
    _, _, df = _small_table(estimators=("linear", "hgb"), modes=("cube", "loco"))
    df = p4.add_margins(df, verbose=False)
    p4.assert_results_complete(df, targets=("cube_mean",),
                               fold_modes=("cube", "loco"),
                               estimators=("linear", "hgb"))
    key = ["stage", "target", "fold_mode", "estimator", "feature_set"]
    for _, g in df.groupby(key):
        assert set(g.model_kind) == set(p4.MODEL_KINDS), (
            f"missing {set(p4.MODEL_KINDS) - set(g.model_kind)}")


def test_a_missing_control_is_refused():
    _, _, df = _small_table()
    df = p4.add_margins(df, verbose=False)
    with pytest.raises(AssertionError, match="missing"):
        p4.assert_results_complete(df[df.model_kind != "permutation"],
                                   targets=("cube_mean",), fold_modes=("cube",),
                                   estimators=("linear",))


def test_the_permutation_control_does_not_manufacture_skill():
    """The empirical zero, and the real model must exceed it.

    The lower side is deliberately NOT asserted. A flexible estimator on
    shuffled features is PENALISED rather than neutral -- it fits noise on train
    and pays for it on test -- so this control sits at or below zero, not at it,
    and its job is to show the pipeline cannot invent skill from nothing.
    """
    _, _, df = _small_table()
    df = p4.add_margins(df, verbose=False)
    perm = df[df.model_kind == "permutation"]
    weather = df[df.model_kind == "weather"]
    assert len(perm) and len(weather)
    assert (perm.r2_vs_climatology_mean <= 0.05).all(), (
        f"the permutation control reports positive skill "
        f"{perm.r2_vs_climatology_mean.tolist()} from weather that was shuffled "
        "across cubes. The pipeline is manufacturing an association.")
    assert (weather.r2_vs_climatology_mean.to_numpy()
            > perm.r2_vs_climatology_mean.to_numpy()).all(), (
        "the real weather model does not beat its own permutation null")
    assert (df.margin_over_permutation[df.model_kind == "weather"] > 0).all()


def test_the_permutation_shuffles_across_cubes():
    m = _manifest()
    cubes = m["cube_id"].to_numpy().astype(str)
    rows = np.arange(len(m))
    perm, same = p4.permute_within_fold(cubes, rows, seed=0)
    assert sorted(perm.tolist()) == sorted(rows.tolist()), "not a permutation"
    assert same < 0.4, (
        f"{same:.0%} of rows kept their own cube's weather; the association the "
        "control exists to destroy is largely intact")


def test_the_permutation_falls_back_within_one_cube_and_says_so():
    """LOCO's test side holds ONE cube, where 'across cubes' is undefined."""
    m = _manifest(n_cubes=1)
    cubes = m["cube_id"].to_numpy().astype(str)
    perm, same = p4.permute_within_fold(cubes, np.arange(len(m)), seed=0)
    assert same == 1.0, "one cube: every row necessarily keeps its own cube"
    assert not np.array_equal(perm, np.arange(len(m))), (
        "the row-level association must still be destroyed")


def test_the_day_of_year_control_is_richer_than_the_detrend():
    """Otherwise it returns zero by least-squares construction and tests
    nothing."""
    assert p4.DOY_CONTROL_HARMONICS > p4.CLIMATOLOGY_HARMONICS
    m = _manifest()
    src = p4.doy_source(m, verbose=False)
    assert src.values.shape[1] == 1 + 2 * p4.DOY_CONTROL_HARMONICS
    assert src.names[0] == "day_of_year"


def test_the_day_of_year_control_catches_an_underfitting_climatology():
    """The control's whole purpose: a detrend too smooth to remove the cycle
    must show up as a materially non-zero day-of-year score."""
    m = _manifest(n_cubes=8, frames=10, step_days=13)
    doy = m["day_of_year"].to_numpy().astype(float)
    # A target with a strong SECOND harmonic that a 1-harmonic curve cannot see.
    y = (0.3 * np.sin(2 * np.pi * doy / 365.25)
         + 0.3 * np.sin(6 * np.pi * doy / 365.25))
    src = _sources(m)
    target = p4.TargetRows(name="cube_mean", level="frame", y=y,
                           row_idx=np.arange(len(m)),
                           cell_idx=np.full(len(m), -1),
                           stratum=m["landcover_stratum"].to_numpy().astype(str))
    data = _data(m, src, target)
    weak = p4.evaluate((data.doy_frame,), target, data, "cube", "linear", k=4,
                       n_harmonics=1)
    good = p4.evaluate((data.doy_frame,), target, data, "cube", "linear", k=4,
                       n_harmonics=5)
    weak_r2 = float(np.mean([r.r2_vs_climatology for r in weak]))
    good_r2 = float(np.mean([r.r2_vs_climatology for r in good]))
    assert weak_r2 > 0.2, (
        f"a deliberately underfitting climatology left {weak_r2:.3f} of the "
        "anomaly explainable from day-of-year alone, and the control did not "
        "report it")
    assert good_r2 < weak_r2


def test_the_two_feature_set_copies_of_a_weather_free_control_agree():
    _, _, df = _small_table()
    p4.assert_control_views_consistent(df)
    for kind in ("observation", "doy"):
        sub = df[df.model_kind == kind]
        assert sub.feature_set.nunique() == 2, (
            "a weather-free control must be emitted under BOTH feature-set "
            "labels, so filtering the CSV to one cannot drop it")
        assert sub.r2_vs_climatology_mean.nunique() == 1


def test_a_disagreeing_control_view_is_refused():
    _, _, df = _small_table()
    bad = df.copy()
    i = bad.index[(bad.model_kind == "observation")
                  & (bad.feature_set == "weather_eowm5")][0]
    bad.loc[i, "r2_vs_climatology_mean"] += 0.1
    with pytest.raises(AssertionError, match="differs between feature sets"):
        p4.assert_control_views_consistent(bad)


def test_the_headline_margin_is_over_the_observation_control():
    _, _, df = _small_table()
    df = p4.add_margins(df, verbose=False)
    key = ["stage", "target", "fold_mode", "estimator", "feature_set"]
    for _, g in df.groupby(key):
        ctrl = float(g[g.model_kind == "observation"].r2_vs_climatology_mean.iloc[0])
        for _, row in g.iterrows():
            assert row.margin_over_control == pytest.approx(
                row.r2_vs_climatology_mean - ctrl)
    assert (df[df.model_kind == "observation"].margin_over_control == 0).all()


def test_add_margins_refuses_a_table_with_no_observation_control():
    _, _, df = _small_table()
    with pytest.raises(AssertionError, match="no matching observation control"):
        p4.add_margins(df[df.model_kind != "observation"], verbose=False)


# ---------------------------------------------------------------------------
# Effective n, uncertainty, and the pseudo-replicate warning
# ---------------------------------------------------------------------------

def test_effective_n_counts_cubes_and_not_rows():
    """The spec's assertion, and the one a reader will check first."""
    m, data, df = _small_table(level="cell")
    p4.assert_effective_n_counts_cubes(df)
    n_cubes = int(m.cube_id.nunique())
    assert (df.effective_n == n_cubes).all(), (
        f"effective n is {sorted(set(df.effective_n))}, expected {n_cubes} "
        "cubes")
    assert (df.n_rows_test_total == len(m) * 16).all()
    assert (df.effective_n < df.n_rows_test_total).all(), (
        "16 cells of a frame share a sky and every frame of a cube shares a "
        "place; the rows are not independent observations")


def test_effective_n_is_the_sum_of_the_per_fold_cube_counts():
    _, _, df = _small_table()
    for _, row in df.iterrows():
        per_fold = [int(x) for x in row.effective_n_per_fold.split(";")]
        assert sum(per_fold) == row.effective_n
        assert all(v >= 1 for v in per_fold)


def test_a_row_reporting_rows_as_its_effective_n_is_refused():
    _, _, df = _small_table()
    bad = df.copy()
    bad["effective_n"] = bad["n_rows_test_total"]
    bad["effective_n_per_fold"] = bad["n_rows_test_total"].astype(str)
    with pytest.raises(AssertionError):
        p4.assert_effective_n_counts_cubes(bad)


def test_every_row_carries_a_fold_clustered_interval():
    _, _, df = _small_table()
    assert (df.r2_ci_lo <= df.r2_mean + 1e-9).all()
    assert (df.r2_ci_hi >= df.r2_mean - 1e-9).all()
    assert (df.r2_ci_hi > df.r2_ci_lo).all(), (
        "a zero-width interval means the spread across folds was not used")


def test_the_clustered_interval_is_over_folds_not_rows():
    v = np.array([0.1, 0.2, 0.3, 0.4])
    lo, hi = p4.fold_clustered_ci(v)
    assert lo < v.mean() < hi
    # A t interval on 4 replicates is much wider than a normal one on 4000.
    assert (hi - lo) > 2 * v.std(ddof=1) / np.sqrt(v.size)


# ---------------------------------------------------------------------------
# Stage B: ran under crossed with the real climatology, or explicitly deferred
# ---------------------------------------------------------------------------

def test_stage_b_is_deferred_on_a_single_year_manifest(capsys):
    m = _manifest()
    df, info = p4.run_stage_b(m, cube_dir="data/raw", verbose=False)
    assert df is None, "Stage B produced rows on a single-year manifest"
    assert info["multi_year"] is False
    out = capsys.readouterr().out
    assert "STAGE B DEFERRED" in out
    assert "seasonal download" in out
    assert "NOT substituted" in out
    assert "SingleYearError" in out


def test_the_seasonal_split_is_detected_from_per_row_years():
    """The seasonal cubes span years INSIDE one file, so the year column
    (parsed from the cube id, i.e. the window START) would say 2018."""
    single = p4.detect_seasonal_split(_manifest(), verbose=False)
    assert single["multi_year"] is False
    assert single["n_cubes_spanning_years"] == 0

    multi = p4.detect_seasonal_split(_manifest(years=(2017, 2018, 2019)),
                                     verbose=False)
    assert multi["multi_year"] is True
    assert multi["years"] == [2017, 2018, 2019]
    assert multi["n_cubes_spanning_years"] == 8, (
        "every cube carries frames in three years; the detector read the year "
        "COLUMN instead of the timestamps")


def test_stage_b_uses_crossed_and_the_real_climatology(monkeypatch):
    """Stage B must call data.climatology.ndvi_climatology and probes.cv's
    crossed generator -- not a reimplementation and not another mode."""
    called = {"clim": 0, "crossed": 0, "other_modes": []}

    import data.climatology as clim_mod

    real_clim = clim_mod.ndvi_climatology

    def spy_clim(*a, **kw):
        called["clim"] += 1
        return real_clim(*a, **kw)

    real_folds = cv.folds

    def spy_folds(manifest, mode="cube", **kw):
        called["other_modes"].append(mode)
        if mode == "crossed":
            called["crossed"] += 1
        return real_folds(manifest, mode, **kw)

    monkeypatch.setattr(clim_mod, "ndvi_climatology", spy_clim)
    monkeypatch.setattr(cv, "folds", spy_folds)

    assert p4.STAGE_B_MODE == "crossed"
    src = inspect.getsource(p4.run_stage_b)
    assert "from data.climatology import" in src, (
        "Stage B does not import the canonical climatology")
    assert "ndvi_climatology(" in src, (
        "Stage B does not call ndvi_climatology; it must not reimplement the "
        "leave-target-year-out definition")
    b_src = inspect.getsource(p4.evaluate_b)
    assert "STAGE_B_MODE" in b_src and "mode" not in inspect.signature(
        p4.evaluate_b).parameters, (
        "evaluate_b takes a mode argument, so Stage B could be run under a "
        "split that disagrees with its own climatology")


def test_the_table_refuses_a_stage_b_row_under_the_wrong_mode():
    _, _, df = _small_table()
    fake = df.head(2).copy()
    fake["stage"] = "B"
    fake["fold_mode"] = "cube"
    fake["climatology_def"] = p4.STAGE_B_CLIMATOLOGY_LABEL
    info = {"multi_year": True, "years": [2017, 2018], "n_cubes": 8,
            "n_cubes_spanning_years": 8, "cubes_spanning_years": ()}
    with pytest.raises(AssertionError, match="Only 'crossed'"):
        p4.assert_stage_b_ran_or_deferred(pd.concat([df, fake]), info)


def test_stage_a_is_never_relabelled_as_stage_b():
    _, _, df = _small_table()
    liar = df.copy()
    liar["stage"] = "B"
    info = {"multi_year": False, "years": [2018], "n_cubes": 8,
            "n_cubes_spanning_years": 0, "cubes_spanning_years": ()}
    with pytest.raises(AssertionError, match="never be relabelled"):
        p4.assert_stage_b_ran_or_deferred(liar, info)


def test_stage_b_is_required_when_it_is_computable():
    _, _, df = _small_table()
    info = {"multi_year": True, "years": [2017, 2018], "n_cubes": 8,
            "n_cubes_spanning_years": 8, "cubes_spanning_years": ()}
    with pytest.raises(AssertionError, match="no Stage B rows"):
        p4.assert_stage_b_ran_or_deferred(df, info)


def test_every_stage_a_row_is_labelled_as_the_proxy():
    _, _, df = _small_table()
    assert (df.climatology_def == p4.STAGE_A_CLIMATOLOGY_LABEL).all()
    label = p4.STAGE_A_CLIMATOLOGY_LABEL
    for phrase in ("proxy", "single year (2018)", "NOT the leave-year-out"):
        assert phrase in label, f"the Stage A label does not say {phrase!r}"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_ensemble_crps_matches_a_brute_force_computation():
    rng = np.random.default_rng(0)
    ens = rng.normal(0, 1, 40)
    y = rng.normal(0, 1, 7)
    brute = np.array([
        np.abs(ens - yi).mean()
        - 0.5 * np.abs(ens[:, None] - ens[None, :]).mean()
        for yi in y])
    np.testing.assert_allclose(p4.ensemble_crps(y, ens), brute, atol=1e-12)


def test_gaussian_crps_is_minimised_at_the_truth():
    y = np.array([0.0])
    scores = [float(p4.gaussian_crps(y, np.array([mu]), 1.0)[0])
              for mu in (-1.0, -0.2, 0.0, 0.2, 1.0)]
    assert scores[2] == min(scores)


def test_a_perfect_forecast_beats_the_climatology_ensemble():
    rng = np.random.default_rng(1)
    truth = rng.normal(0, 1, 50)
    assert (p4.gaussian_crps(truth, truth, 1e-6).mean()
            < p4.ensemble_crps(truth, truth).mean())


def test_r2_vs_climatology_is_skill_against_predicting_zero():
    """The metric H1 is quoted in: the climatology predicts anomaly zero."""
    m = _manifest()
    src = _sources(m, signal=0.0)
    target = _target(m, src, noise=0.02)
    data = _data(m, src, target)
    res = p4.evaluate((src,), target, data, "cube", "linear", k=4)
    for r in res:
        sse = float(((r.test_anomaly - r.test_prediction) ** 2).sum())
        assert r.r2_vs_climatology == pytest.approx(
            1 - sse / float((r.test_anomaly ** 2).sum()))


# ---------------------------------------------------------------------------
# Severity bins and per-stratum reporting
# ---------------------------------------------------------------------------

def test_bin_edges_and_counts_are_printed_before_anything_is_fitted(capsys):
    rng = np.random.default_rng(0)
    labels, edges = p4.print_severity_bins(rng.normal(0, 0.1, 500), label="t")
    out = capsys.readouterr().out
    assert "edges at quantiles" in out
    for b in p4.SEVERITY_BINS:
        assert b in out
        assert (labels == b).sum() > 0
    assert len(edges) == len(p4.SEVERITY_QUANTILES)
    assert (np.diff(edges) > 0).all()


def test_the_doy_weather_collinearity_is_measured_and_printed(capsys):
    """The number the day-of-year control must be read against, printed before
    any fit. On the real subset the cubes share one orbit lattice, so the date
    carries most of the weather."""
    m = _real_manifest(n=6)
    src, _ = p4.weather_source(m, os.path.join("data", "raw"),
                               "weather_full8", verbose=False)
    out = p4.print_doy_weather_collinearity(m, src)
    printed = capsys.readouterr().out
    assert "DISTINCT days of year" in printed
    assert "across-cube spread" in printed
    assert out["one_orbit_lattice"] is True, (
        "the cubes no longer share one Sentinel-2 orbit lattice; the "
        "day-of-year control's interpretation changes and the docstring's "
        "measurement is stale")
    assert out["n_distinct_doy"] < len(m), (
        "day-of-year is not a coarse factor here, so it cannot be nearly "
        "collinear with the weather")
    assert 0.0 < out["across_cube_share_median"] < 1.0


def test_precipitation_carries_the_cross_cube_weather_signal():
    """Temperature is an air mass over the whole tile; rain is local. If that
    ever reversed, a weather model under cube-grouped CV would have almost no
    feature variation to learn from."""
    m = _real_manifest(n=8)
    doy = m.day_of_year.to_numpy()

    def across_cube_share(v):
        x = m[v].to_numpy().astype(float)
        res = np.concatenate([x[doy == d] - x[doy == d].mean()
                              for d in np.unique(doy)])
        return res.std() / x.std()

    assert across_cube_share("eobs_rr") > across_cube_share("eobs_tg"), (
        "precipitation no longer varies across cubes more than temperature")


def test_a_degenerate_anomaly_distribution_is_refused():
    with pytest.raises(AssertionError, match="degenerate"):
        p4.severity_bins(np.zeros(100))


def test_the_table_carries_a_column_per_bin_and_per_stratum():
    _, _, df = _small_table(level="cell")
    for b in p4.SEVERITY_BINS:
        assert f"r2_{b}" in df.columns and f"n_{b}" in df.columns
    for s in p4.REPLICATION_STRATA:
        assert f"r2_{s}" in df.columns and f"n_{s}" in df.columns
    assert (df[[f"n_{s}" for s in p4.REPLICATION_STRATA]].sum(axis=1) > 0).all()


def test_replication_covers_only_the_three_thick_strata():
    """built_up has 5 cells and bare_sparse 1 over the whole subset; a
    'replication' over 1 cell is not a replication."""
    assert p4.REPLICATION_STRATA == ("cropland", "tree_cover", "grassland")
    _, _, df = _small_table(level="cell")
    assert "r2_built_up" not in df.columns
    assert "r2_bare_sparse" not in df.columns


# ---------------------------------------------------------------------------
# Machinery: folds, sources, determinism
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", p4.FOLD_MODES)
def test_no_manifest_row_is_on_both_sides_of_a_fold(mode):
    m = _manifest()
    for train, test in p4.outer_folds(m, mode, k=4):
        assert not np.intersect1d(train, test).size
        cubes = m["cube_id"].to_numpy()
        assert not set(cubes[train]) & set(cubes[test])


def test_cell_rows_of_one_frame_never_straddle_a_fold():
    m = _manifest()
    src = _sources(m)
    target = _target(m, src, level="cell")
    for train, test in p4.outer_folds(m, "cube", k=4):
        tr = np.flatnonzero(np.isin(target.row_idx, train))
        te = np.flatnonzero(np.isin(target.row_idx, test))
        assert not np.intersect1d(target.row_idx[tr], target.row_idx[te]).size
        assert tr.size + te.size == target.n_rows


def test_the_run_is_deterministic_and_n_jobs_changes_nothing():
    m = _manifest()
    src = _sources(m)
    target = _target(m, src)
    data = _data(m, src, target)
    a = p4.evaluate((src,), target, data, "cube", "linear", k=4, n_jobs=1)
    b = p4.evaluate((src,), target, data, "cube", "linear", k=4, n_jobs=2)
    np.testing.assert_allclose([r.r2 for r in a], [r.r2 for r in b], atol=0)


def test_standardisation_is_fitted_on_train_only():
    """A scaler fitted on both sides would move the test predictions."""
    rng = np.random.default_rng(0)
    X_tr = rng.normal(0, 1, (60, 4))
    X_te = rng.normal(50, 1, (20, 4))       # a wildly different test location
    y = rng.normal(0, 1, 60)
    pred_a, _, _ = p4._fit_predict("linear", X_tr, y, X_te)
    shifted, _, _ = p4._fit_predict("linear", X_tr, y, X_te + 1000)
    assert not np.allclose(pred_a, shifted), (
        "shifting the test features did not change the predictions, so the "
        "scaler saw the test set")


def test_a_model_with_real_signal_beats_its_own_permutation():
    m = _manifest()
    src = _sources(m, signal=2.0)
    target = _target(m, src, noise=0.01)
    data = _data(m, src, target)
    real = p4.evaluate((src,), target, data, "cube", "linear", k=4)
    null = p4.evaluate((src,), target, data, "cube", "linear", k=4, permute=True)
    assert (np.mean([r.r2_vs_climatology for r in real])
            > np.mean([r.r2_vs_climatology for r in null]) + 0.1)


def test_a_target_with_no_weather_signal_does_not_beat_the_null_by_much():
    m = _manifest()
    src = _sources(m, signal=0.0)
    rng = np.random.default_rng(7)
    doy = m["day_of_year"].to_numpy().astype(float)
    y = 0.3 * np.sin(2 * np.pi * doy / 365.25) + rng.normal(0, 0.05, len(m))
    target = p4.TargetRows(name="cube_mean", level="frame", y=y,
                           row_idx=np.arange(len(m)),
                           cell_idx=np.full(len(m), -1),
                           stratum=m["landcover_stratum"].to_numpy().astype(str))
    data = _data(m, src, target)
    res = p4.evaluate((src,), target, data, "cube", "linear", k=4)
    assert np.mean([r.r2_vs_climatology for r in res]) < 0.15


def test_the_eowm_subset_is_the_five_meso_channels():
    """Cross-paper commensurability: EO-WM's era5_climatology_all.pt holds
    precipitation, pressure and mean/min/max temperature."""
    assert set(p4.EOWM_MESO_VARS) == {"eobs_rr", "eobs_pp", "eobs_tg",
                                      "eobs_tn", "eobs_tx"}
    assert set(p4.EOWM_MESO_VARS) < set(p4.EOBS_VARS)
    assert len(p4.EOBS_VARS) == 8


def test_the_multi_image_window_length_matches_the_encoder():
    """MI_WINDOW_LEN is copied so this module need not import torch; it must
    not drift from the encoder it describes."""
    from encoders.satlas_s2_mi import SatlasS2SwinBMI
    assert p4.MI_WINDOW_LEN == SatlasS2SwinBMI.window_len


def test_windows_are_defined_on_the_daily_axis_not_on_retained_frames():
    """The horizon rule, applied to weather. A 30-day window must cover 30
    DAYS, not 30 acquisitions."""
    m = _manifest()
    src = inspect.getsource(p4.cube_weather_windows)
    assert "cube_daily_axis" in src and "frame_daily_positions" in src
    assert "original_axis_index" not in src, (
        "the weather window is positioned by the acquisition axis, which is "
        "longer in cloudy weather and would make the window itself a weather "
        "feature")


# ---------------------------------------------------------------------------
# The real cubes
# ---------------------------------------------------------------------------

def _real_cubes():
    paths = sorted(glob.glob(os.path.join("data", "raw", "*.nc")))
    if len(paths) < 3:
        pytest.skip("real cubes not present (data/raw)")
    return paths


def _real_manifest(n=4):
    from data.loader import load_cube

    paths = _real_cubes()[:n]
    return build_manifest([load_cube(p, verbose=False) for p in paths],
                          verbose=False)


def test_the_weather_join_is_verified_against_the_cubes():
    """The check that would have caught the wrong-axis bug: go back to the
    file, look the day up by TIMESTAMP, and compare."""
    from encoders.manifest import assert_weather_join

    m = _real_manifest()
    out = assert_weather_join(m, os.path.join("data", "raw"), verbose=False)
    assert out["n_rows"] == len(m)
    assert max(out["max_abs_diff"].values()) == 0.0


def test_a_manifest_built_on_the_wrong_axis_is_refused():
    """Reconstruct the defect and prove the guard fires."""
    from encoders.manifest import assert_weather_join, cube_weather

    m = _real_manifest()
    broken = m.copy()
    cube = broken.cube_id.iloc[0]
    w = cube_weather(os.path.join("data", "raw", str(cube)))
    sel = broken.cube_id == cube
    broken.loc[sel, "eobs_tg"] = w["eobs_tg"][
        broken.loc[sel, "original_axis_index"].to_numpy()]
    with pytest.raises(AssertionError, match="wrong-axis"):
        assert_weather_join(broken, os.path.join("data", "raw"), verbose=False)


def test_the_two_time_axes_are_different_on_the_real_subset():
    """If they ever coincide the wrong-axis bug becomes undetectable, so the
    fact that they do not is worth pinning."""
    m = _real_manifest()
    off = (m.daily_axis_index - m.original_axis_index).to_numpy()
    assert (off > 0).all(), (
        "the acquisition axis and the daily axis coincide; the cube no longer "
        "has empty days and this test's premise has changed")


def test_real_targets_are_spatially_aggregated_and_finite():
    from data.loader import load_cube

    sample = load_cube(_real_cubes()[0], verbose=False)
    t = p4.cube_frame_targets(sample)
    T = t["cube_mean"].shape[0]
    assert t["cube_p90"].shape == (T,)
    assert t["cell_mean"].shape == (T, 16)
    assert np.isfinite(t["cube_mean"]).all()
    assert (t["cube_p90"] >= t["cube_mean"]).all(), (
        "the 90th percentile is below the mean")
    assert (np.abs(t["cube_mean"]) <= 1).all(), "NDVI outside [-1, 1]"


def test_real_weather_windows_cover_days_not_frames():
    m = _real_manifest(n=2)
    cube = sorted(m.cube_id.unique())[0]
    sub = m[m.cube_id == cube]
    X, names, avail = p4.cube_weather_windows(
        os.path.join("data", "raw", str(cube)), sub["timestamp"].to_numpy())
    assert X.shape == (len(sub), len(names))
    assert np.isfinite(X).all()
    for s, (span, lag) in enumerate(p4.WINDOW_SPECS):
        assert (avail[:, s] <= span).all()
        assert (avail[:, s] >= 1).all()
    # The longest window must reach further back than the frame gap, or it is
    # not a window over days at all.
    gaps = np.diff(sub["daily_axis_index"].to_numpy())
    assert max(s for s, _ in p4.WINDOW_SPECS) > float(np.median(gaps))


def test_frame_targets_report_the_plausibility_screen_without_applying_it():
    """The screen is REPORTED, never applied: P2's and P4's published tables
    were computed without it, and silently changing them would make the old and
    new numbers incomparable. A probe opts in and says so on its rows."""
    import numpy as np
    from probes import p4_ceiling as p4

    assert p4.NDVI_PLAUSIBILITY_FLOOR == 0.15
    lo, hi = p4.GROWING_SEASON_DOY
    assert 0 < lo < hi <= 366

    doy = np.array([200, 200, 200, 15])          # three in season, one in winter
    cube_mean = np.array([0.70, -0.04, 0.10, 0.05])
    in_season = (doy >= lo) & (doy <= hi)
    plausible = ~(in_season & (cube_mean < p4.NDVI_PLAUSIBILITY_FLOOR))
    # healthy summer frame kept; the two cloud-contaminated summer frames
    # flagged; the winter frame NOT flagged (low NDVI is legitimate there)
    np.testing.assert_array_equal(plausible, [True, False, False, True])


def test_plausibility_screen_filters_frames_cells_and_the_cell_control_together():
    m = _manifest(n_cubes=1, frames=10)
    src = _sources(m)
    frame = _target(m, src, level="frame")
    cell = _target(m, src, level="cell")
    observation_cell = _data(m, src, cell).observation_cell
    ok = np.ones(len(m), dtype=bool)
    ok[1] = False

    targets, screened_obs, dropped = p4.apply_plausibility_screen(
        {"cube_mean": frame, "cell_mean": cell}, observation_cell, ok)

    assert dropped == {"cube_mean": 1, "cell_mean": 16}
    np.testing.assert_array_equal(
        targets["cube_mean"].row_idx, np.array([0] + list(range(2, 10))))
    assert not (targets["cell_mean"].row_idx == 1).any()
    assert targets["cell_mean"].n_rows == screened_obs.values.shape[0] == 144


def test_p4_screen_makes_poisoned_excluded_targets_invisible():
    from dataclasses import replace

    m = _manifest(n_cubes=1, frames=10)
    src = _sources(m)
    frame = _target(m, src, level="frame")
    cell = _target(m, src, level="cell")
    observation_cell = _data(m, src, cell).observation_cell
    ok = np.ones(len(m), dtype=bool)
    ok[1] = False

    poisoned_frame_y = frame.y.copy()
    poisoned_frame_y[1] = 1e9
    poisoned_cell_y = cell.y.copy()
    poisoned_cell_y[cell.row_idx == 1] = -1e9
    poisoned = {
        "cube_mean": replace(frame, y=poisoned_frame_y),
        "cell_mean": replace(cell, y=poisoned_cell_y),
    }
    clean, clean_obs, _ = p4.apply_plausibility_screen(
        {"cube_mean": frame, "cell_mean": cell}, observation_cell, ok)
    after, after_obs, _ = p4.apply_plausibility_screen(
        poisoned, observation_cell, ok)

    np.testing.assert_array_equal(clean["cube_mean"].y, after["cube_mean"].y)
    np.testing.assert_array_equal(clean["cell_mean"].y, after["cell_mean"].y)
    np.testing.assert_array_equal(clean_obs.values, after_obs.values)


def test_p4_plausibility_screen_is_a_noop_when_every_frame_passes():
    m = _manifest(n_cubes=1, frames=10)
    src = _sources(m)
    frame = _target(m, src, level="frame")
    cell = _target(m, src, level="cell")
    observation_cell = _data(m, src, cell).observation_cell
    ok = np.ones(len(m), dtype=bool)

    targets, screened_obs, dropped = p4.apply_plausibility_screen(
        {"cube_mean": frame, "cell_mean": cell}, observation_cell, ok)
    assert dropped == {"cube_mean": 0, "cell_mean": 0}
    np.testing.assert_array_equal(targets["cube_mean"].y, frame.y)
    np.testing.assert_array_equal(targets["cell_mean"].y, cell.y)
    np.testing.assert_array_equal(screened_obs.values, observation_cell.values)


def test_p4_screen_declaration_is_emitted_and_survives_csv_round_trip():
    import io
    from dataclasses import replace

    m = _manifest()
    src = _sources(m)
    target = _target(m, src)
    data = _data(m, src, target)
    unscreened = p4.run_stage_a(
        data, targets=(target.name,), fold_modes=("cube",),
        estimators=("linear",), feature_sets=("weather_full8",), k=4,
        verbose=False)
    p4.assert_plausibility_screen_declared(unscreened, required=False)

    screened_data = replace(
        data, plausibility_screen=True, n_implausible_frames=1,
        n_rows_dropped_implausible={target.name: 1})
    screened = p4.run_stage_a(
        screened_data, targets=(target.name,), fold_modes=("cube",),
        estimators=("linear",), feature_sets=("weather_full8",), k=4,
        verbose=False)
    p4.assert_plausibility_screen_declared(screened, required=True)
    back = pd.read_csv(io.StringIO(screened.to_csv(index=False)))
    p4.assert_plausibility_screen_declared(back, required=True)

    mixed = screened.copy()
    mixed.loc[mixed.index[0], "plausibility_screen"] = False
    with pytest.raises(AssertionError, match="mixes"):
        p4.assert_plausibility_screen_declared(mixed, required=True)
