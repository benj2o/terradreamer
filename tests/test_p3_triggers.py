"""P3 trigger metrics: the threshold refusal, the paired interval, the counts.

Everything here runs on SYNTHETIC prediction frames with known contingency
tables, because the module's input is a CSV and nothing about it needs cubes.
The three that matter most are the ones nothing downstream could catch:

* ``test_the_jackknife_is_paired_differences_own`` -- the intervals in this
  table claim to be the same estimator ``p3_forecast.paired_difference`` uses.
  A separately-written jackknife that happened to be close would be impossible
  to distinguish from the real thing by reading either one, so the generic form
  is required to reproduce ``paired_difference`` digit for digit on the same
  numbers.
* ``test_a_full_sample_threshold_is_refused`` -- a quantile fitted over
  everything produces rates in range, contingency tables that add up and a
  crossing rate near the nominal 10%. It is invisible in its own output, so it
  is the one thing the module refuses rather than reports.
* ``test_a_rate_over_too_few_events_is_not_reported`` -- two hits out of three
  events is not a 67% hit rate, and the counts have to survive while the rate
  does not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from probes import p3_forecast as p3
from probes import p3_triggers as tg
from probes import p4_ceiling as p4


# ---------------------------------------------------------------------------
# A synthetic predictions frame
# ---------------------------------------------------------------------------

def _frame(n_folds=5, per_fold=40, seed=0, model_kinds=("forecast",),
           full_sample_threshold=False, skill=0.7):
    """A predictions CSV with the real column set and known fold structure.

    ``skill`` blends the truth into the prediction, so the model is better than
    persistence by a controllable amount and the paired difference has a sign a
    test can assert rather than merely a value it can print.
    """
    rng = np.random.default_rng(seed)
    n = n_folds * per_fold
    # ONE truth, shared by every model kind, exactly as the real artefact has
    # it: the folds and the held-out observations are the same for every row of
    # a view, and only the PREDICTION differs. A fixture that redrew the truth
    # per model would let a bug that compared two models on two different sets
    # of rows pass every test in this file.
    fold = np.repeat(np.arange(n_folds), per_fold)
    doy = rng.uniform(120, 260, n)
    clim = 0.55 + 0.15 * np.sin(2 * np.pi * doy / 365.25)
    anom = rng.normal(0, 0.08, n)
    y = clim + anom
    pers = clim + 0.35 * anom + rng.normal(0, 0.09, n)
    model = clim + skill * anom + (1 - skill) * rng.normal(0, 0.08, n)

    rows = []
    for kind in model_kinds:
        pred = {"persistence": pers, "climatology_proxy": clim}.get(kind, model)
        g = pd.DataFrame({
            "cube_id": [f"cube{fold[i]}_{i % 4}.nc" for i in range(n)],
            "daily_axis_index": (np.arange(n) % per_fold) * 5,
            "delta_days": 25, "aggregation": "cube_mean",
            "fold_mode": "cube", "fold_index": fold, "model_kind": kind,
            "encoder": "dinov2_vitb14" if kind == "forecast" else "none",
            "feature_level": "pooled",
            "estimator": "linear" if kind == "forecast" else "none",
            "alpha_rule": (p3.ALPHA_RULE_TUNED if kind == "forecast"
                           else p3.ALPHA_RULE_NA),
            "y_true": y, "y_pred": pred.copy(), "y_persistence": pers,
            "y_climatology_proxy": clim,
            "feature_set": "embedding" if kind == "forecast" else "none",
            "feature_base": p3.FEATURE_BASE_NONE,
            "feature_row": np.arange(n),
            "day_of_year_target": doy,
            "n_threshold_train_rows": per_fold * (n_folds - 1),
        })
        a = (g.y_true - g.y_climatology_proxy).to_numpy()
        for lv in tg.TRIGGER_LEVELS:
            col = tg.THRESHOLD_COLUMN[lv]
            q = tg.THRESHOLD_QUANTILE[lv]
            if full_sample_threshold:
                g[col] = float(np.quantile(a, q))          # THE LEAK
            else:
                edge = np.empty(len(g))
                for f in range(n_folds):
                    m = (g.fold_index == f).to_numpy()
                    edge[m] = float(np.quantile(a[~m], q))  # train side only
                g[col] = edge
        rows.append(g)
    out = pd.concat(rows, ignore_index=True)
    return out[list(p3.PREDICTIONS_COLUMNS)]


@pytest.fixture
def frame():
    return _frame(model_kinds=("forecast", "persistence", "climatology_proxy"))


# ---------------------------------------------------------------------------
# The threshold, and the leak it is refused for
# ---------------------------------------------------------------------------

def test_the_threshold_levels_are_p4s_and_are_not_typed_here():
    """A second copy of "10th percentile" would drift away from p4's silently."""
    assert set(tg.TRIGGER_LEVELS) <= set(p4.SEVERITY_BINS)
    for lv in tg.TRIGGER_LEVELS:
        assert tg.THRESHOLD_QUANTILE[lv] == p4.SEVERITY_QUANTILES[
            p4.SEVERITY_BINS.index(lv)]
    assert tg.THRESHOLD_QUANTILE["extreme_low"] == 0.10
    assert tg.THRESHOLD_QUANTILE["low"] == 0.30


def test_a_train_fitted_threshold_is_accepted(frame):
    out = tg.assert_thresholds_are_train_fitted(frame, verbose=False)
    assert out["extreme_low"]["views_with_moving_edges"] == 1


def test_a_full_sample_threshold_is_refused():
    """THE LEAK, and it is invisible in its own output.

    The rates a full-sample threshold produces are all in range and its
    contingency tables all add up. What gives it away is that the line does not
    move between folds -- which is exactly what "fitted on everything" means.
    """
    bad = _frame(full_sample_threshold=True)
    with pytest.raises(AssertionError, match="identical on all"):
        tg.assert_thresholds_are_train_fitted(bad, verbose=False)


def test_a_full_sample_threshold_is_refused_even_with_one_fold():
    """The check that still bites when "it moves fold to fold" cannot.

    A view left with a single fold has no fold-to-fold movement to inspect, so
    the leak would walk straight through the direct check. Comparing the line
    against the full-sample quantile catches it from the other side.
    """
    bad = _frame(full_sample_threshold=True)
    bad["fold_index"] = 0
    bad["n_threshold_train_rows"] = 1
    with pytest.raises(AssertionError, match="EVERY fold"):
        tg.assert_thresholds_are_train_fitted(bad, verbose=False)


def test_a_threshold_that_varies_by_model_is_refused(frame):
    """One line per (view, fold). A line that moved with the model would mean
    it was derived from the predictions it is scoring."""
    bad = frame.copy()
    m = (bad.model_kind == "forecast").to_numpy()
    bad.loc[m, "anomaly_threshold_extreme_low"] -= 0.01
    with pytest.raises(AssertionError, match="more than one value inside"):
        tg.assert_thresholds_are_train_fitted(bad, verbose=False)


def test_a_threshold_fitted_on_more_rows_than_the_view_has_is_refused(frame):
    """The arithmetic check: train rows + held-out rows cannot exceed the view.

    This is the one that still fires when a view has a single fold, where
    "the line moves between folds" has nothing to say.
    """
    bad = frame.copy()
    bad["n_threshold_train_rows"] = int(bad.feature_row.nunique())
    with pytest.raises(AssertionError, match="reached into the test side"):
        tg.assert_thresholds_are_train_fitted(bad, verbose=False)


# ---------------------------------------------------------------------------
# The contingency table and its rates
# ---------------------------------------------------------------------------

def test_contingency_and_rates_against_a_hand_computation():
    obs = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
    fc = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0], dtype=bool)
    c = tg.contingency(obs, fc)
    assert list(c) == [2, 1, 2, 5]              # tp, fp, fn, tn
    r = tg.rates(c)
    assert r["hit_rate"] == pytest.approx(2 / 4)
    assert r["false_alarm_rate"] == pytest.approx(1 / 6)   # POFD, over NON-events
    assert r["far"] == pytest.approx(1 / 3)                # the OTHER convention
    assert r["csi"] == pytest.approx(2 / 5)
    assert r["pss"] == pytest.approx(2 / 4 - 1 / 6)


def test_the_two_false_alarm_conventions_are_not_the_same_number():
    """POFD and FAR differ by an order of magnitude when events are rare, which
    is the whole point of this table, so they carry different names."""
    obs = np.zeros(200, dtype=bool)
    obs[:10] = True
    fc = np.zeros(200, dtype=bool)
    fc[5:25] = True
    r = tg.rates(tg.contingency(obs, fc))
    assert r["false_alarm_rate"] == pytest.approx(15 / 190)
    assert r["far"] == pytest.approx(15 / 20)
    assert r["far"] > 5 * r["false_alarm_rate"]


# ---------------------------------------------------------------------------
# The interval
# ---------------------------------------------------------------------------

def test_the_jackknife_is_paired_differences_own():
    """Not "the same idea as" -- the same numbers, to 1e-12.

    ``p3_forecast.paired_difference`` is the delete-one-fold jackknife of an
    R-squared difference over per-fold sums. ``fold_jackknife`` is the same
    estimator with the statistic factored out. Feeding it the R-squared
    statistic must reproduce paired_difference exactly, or the trigger table's
    intervals are a second, unvalidated estimator wearing the same name.
    """
    rng = np.random.default_rng(7)
    K, per = 6, 30
    fold = np.repeat(np.arange(K), per)
    y = rng.normal(0.5, 0.1, K * per)
    pa = y + rng.normal(0, 0.05, K * per)
    pb = y + rng.normal(0, 0.09, K * per)
    pos = np.arange(K * per)
    a = {"fold": fold, "pos": pos, "y": y, "pred": pa}
    b = {"fold": fold, "pos": pos, "y": y, "pred": pb}
    want = p3.paired_difference(a, b)

    stats = np.zeros((K, 5))
    for i in range(K):
        m = fold == i
        stats[i] = [m.sum(), y[m].sum(), (y[m] ** 2).sum(),
                    ((y[m] - pa[m]) ** 2).sum(), ((y[m] - pb[m]) ** 2).sum()]

    def r2_diff(v):
        n, sy, sy2, sa, sb = v
        sst = sy2 - sy * sy / n
        return (sb - sa) / sst if sst > 0 else float("nan")

    got = tg.fold_jackknife(stats, r2_diff)
    assert got["value"] == pytest.approx(want["diff"], abs=1e-12)
    assert got["ci_lo"] == pytest.approx(want["ci_lo"], abs=1e-12)
    assert got["ci_hi"] == pytest.approx(want["ci_hi"], abs=1e-12)
    assert got["separable"] == want["separable"]
    assert got["n_folds"] == want["n_folds"] == K


def test_the_interval_is_not_a_marginal_one(frame):
    """The reason the whole table is paired.

    The model and persistence are scored on the SAME rows in the SAME folds, so
    most of the fold-to-fold movement is shared and cancels inside the
    difference. An interval on the difference is therefore much tighter than
    the two marginal intervals -- and it is the difference the table reports.
    """
    sub = frame[frame.model_kind == "forecast"]
    tr = tg.trigger_metrics(sub, levels=("low",), verbose=False)
    r = tr.iloc[0]
    paired = r.ci_hi_pss - r.ci_lo_pss

    fold = sub.fold_index.to_numpy()
    K = int(fold.max()) + 1
    stats = np.zeros((K, 8))
    tau = sub.anomaly_threshold_low.to_numpy()
    obs = (sub.y_true - sub.y_climatology_proxy).to_numpy() < tau
    fm = (sub.y_pred - sub.y_climatology_proxy).to_numpy() < tau
    fp_ = (sub.y_persistence - sub.y_climatology_proxy).to_numpy() < tau
    for i in range(K):
        m = fold == i
        stats[i, :4] = tg.contingency(obs[m], fm[m])
        stats[i, 4:] = tg.contingency(obs[m], fp_[m])
    a = tg.fold_jackknife(stats, lambda v: tg.rates(v[:4])["pss"])
    b = tg.fold_jackknife(stats, lambda v: tg.rates(v[4:])["pss"])
    marginal = max(a["ci_hi"] - a["ci_lo"], b["ci_hi"] - b["ci_lo"])
    assert paired < marginal, (paired, marginal)


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

def test_persistence_scored_against_itself_is_exactly_zero(frame):
    """The baseline column's own row. Not approximately zero -- the same
    forecast on both sides, so every difference and both interval ends are 0."""
    tr = tg.trigger_metrics(frame, verbose=False)
    ps = tr[tr.model_kind == "persistence"]
    assert len(ps) == len(tg.TRIGGER_LEVELS)
    for m in tg.TRIGGER_METRICS:
        np.testing.assert_allclose(ps[m], ps[f"{m}_persistence"], atol=1e-12)
        np.testing.assert_allclose(ps[f"diff_{m}"], 0.0, atol=1e-12)
        assert not ps[f"separable_{m}"].any()


def test_a_climatology_forecast_never_fires(frame):
    """Its prediction IS the climatology, so its anomaly is exactly zero and it
    cannot cross a negative line. Hit rate 0, false alarms 0, no skill --
    which is a property of the baseline and not a bug in the counting."""
    tr = tg.trigger_metrics(frame, verbose=False)
    cp = tr[tr.model_kind == "climatology_proxy"]
    assert len(cp) == len(tg.TRIGGER_LEVELS)
    assert (cp.tp == 0).all() and (cp.fp == 0).all()
    np.testing.assert_allclose(cp.hit_rate, 0.0, atol=1e-12)
    np.testing.assert_allclose(cp.pss, 0.0, atol=1e-12)
    assert (cp.fn > 0).all(), "the synthetic frame holds no events at all"


def test_a_rate_over_too_few_events_is_not_reported():
    """Counts survive; the RATE does not. p3_forecast.summarise's _MIN_BIN_ROWS
    rule, at the count an event-conditioned rate actually rests on."""
    f = _frame(n_folds=3, per_fold=8, seed=3)     # 24 rows -> ~2 extreme_low
    tr = tg.trigger_metrics(f, verbose=False)
    thin = tr[tr.n_events < tg.MIN_TRIGGER_EVENTS]
    assert len(thin), "the fixture is not thin enough to exercise the rule"
    for _, r in thin.iterrows():
        assert not r.enough_events
        assert np.isnan(r.hit_rate) and np.isnan(r.csi) and np.isnan(r.pss)
        assert np.isnan(r.diff_hit_rate) and not r.separable_hit_rate
        # the counts are still there, which is what "reported as such" means
        assert r.n_events == r.tp + r.fn
        assert r.n_rows == r.tp + r.fp + r.fn + r.tn


def test_every_cell_reports_its_n_and_the_thresholds_it_used(frame):
    tr = tg.trigger_metrics(frame, verbose=False)
    assert (tr.n_rows > 0).all() and (tr.n_folds > 0).all()
    assert (tr.n_events + tr.n_non_events == tr.n_rows).all()
    assert (tr.tp + tr.fp + tr.fn + tr.tn == tr.n_rows).all()
    assert (tr.tp_persistence + tr.fp_persistence + tr.fn_persistence
            + tr.tn_persistence == tr.n_rows).all()
    # the observed event is a property of the observations, so the model and
    # persistence must be scored against the SAME set of them
    assert (tr.tp + tr.fn == tr.tp_persistence + tr.fn_persistence).all()
    assert (tr.threshold_min <= tr.threshold_median).all()
    assert (tr.threshold_median <= tr.threshold_max).all()
    assert (tr.interval == tg.INTERVAL_LABEL).all()


def test_the_cell_key_keeps_two_learners_apart():
    """A (horizon, aggregation, mode, kind, encoder) cell is NOT one model.

    The same encoder appears under two penalty rules, and p3's margin key
    already says why those must not be averaged: they are two learners. Pooling
    them here would report a hit rate belonging to neither.
    """
    assert set(tg.REPORT_KEYS) < set(tg.CELL_KEYS)
    assert {"estimator", "alpha_rule", "feature_base"} <= set(tg.CELL_KEYS)
    # SAME seed, so the two share every held-out observation and every
    # threshold and differ only in the prediction -- which is what two penalty
    # rules on one view actually are.
    a = _frame(seed=1, skill=0.9)
    b = _frame(seed=1, skill=0.2)
    b["alpha_rule"] = p3.ALPHA_RULE_FIXED
    tr = tg.trigger_metrics(pd.concat([a, b], ignore_index=True), verbose=False)
    fc = tr[(tr.model_kind == "forecast") & (tr.level == "low")]
    assert len(fc) == 2, "the two penalty rules were pooled into one cell"
    assert fc.pss.nunique() == 2


def test_the_table_survives_a_CSV_ROUND_TRIP(frame, tmp_path):
    tr = tg.trigger_metrics(frame, verbose=False)
    p = tmp_path / "triggers.csv"
    tr.to_csv(p, index=False)
    back = pd.read_csv(p)
    assert back.shape == tr.shape
    for c in ("hit_rate", "pss", "diff_pss", "ci_lo_pss", "ci_hi_pss"):
        np.testing.assert_allclose(back[c].to_numpy(), tr[c].to_numpy(),
                                   rtol=0, atol=1e-12, equal_nan=True)


def test_load_predictions_refuses_a_file_without_the_columns(tmp_path, frame):
    p = tmp_path / "thin.csv"
    frame.drop(columns=["anomaly_threshold_low"]).to_csv(p, index=False)
    with pytest.raises(AssertionError, match="missing"):
        tg.load_predictions(str(p), verbose=False)
    with pytest.raises(AssertionError, match="emit_predictions=True"):
        tg.load_predictions(str(tmp_path / "nope.csv"), verbose=False)


def test_print_trigger_table_names_the_horizon_and_the_baseline(frame, capsys):
    """Lead time across the columns, and persistence beside every value."""
    tr = tg.trigger_metrics(frame, verbose=False)
    tg.print_trigger_table(tr, level="low", metric="pss",
                           feature_base=p3.FEATURE_BASE_NONE)
    out = capsys.readouterr().out
    assert "D=25d" in out
    assert "persistence" in out
    assert "30%" in out and "TRAIN-fold" in out
