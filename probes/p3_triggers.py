"""P3 trigger metrics: does the forecast FIRE when the anomaly crosses the line?

The R-squared table answers "how close is the prediction to the truth, on
average, in squared error". No operational decision is taken on that number. A
decision is taken on a CROSSING: the vegetation anomaly is about to fall below
some line, so irrigate / release water / issue the advisory. A model can hold a
respectable R-squared and never once fire when it matters, because most of the
variance in this subset is the seasonal march and the crossings are rare.

So this module reads the per-observation held-out predictions
(``p3_forecast.PREDICTIONS_COLUMNS``, written by a run with
``emit_predictions=True``) and reports, per horizon, the contingency table of

    "the anomaly at t+Delta is below the threshold"

against the same event as the model, and as PERSISTENCE, predicted it.

THE THRESHOLD IS NOT INVENTED HERE
-----------------------------------
It is ``p4_ceiling``'s severity rule, unchanged: the anomaly is the departure
from a day-of-year climatology curve, and the two lines are the SAME quantiles
of it that ``p4_ceiling.SEVERITY_QUANTILES`` already defines --

    extreme_low   the 10th percentile of the reference anomaly distribution
    low           the 30th percentile

-- so a "trigger" here is exactly p4's ``extreme_low`` severity bin boundary and
a reader does not have to hold two definitions of "bad" in their head.

ONE THING IS CHANGED, AND IT IS THE FIT
----------------------------------------
``p4_ceiling.severity_reference_anomaly`` fits its curve on ALL rows and takes
its quantiles over ALL rows, and is explicit that this is legitimate because
severity is a REPORTING axis: it labels held-out rows after the fact and never
enters a score.

A trigger threshold is not that. It is a number a forecast is scored against,
so a 10th percentile taken over the full sample has already seen the held-out
rows, and the crossing rate on the test side is then partly a property of the
test side. The fit therefore moves inside the fold --
``p3_forecast._trigger_reference``, on the fold's TRAINING rows and no others,
the same move ``doy_climatology_within_fold`` makes for the climatology itself.
The edges the fold actually used travel on every prediction row, and
``assert_thresholds_are_train_fitted`` refuses a file whose thresholds could
have come from the full sample.

EVERY NUMBER IS READ AGAINST PERSISTENCE, AND THE INTERVAL IS PAIRED
---------------------------------------------------------------------
The R-squared table does not report a bare score; it reports the paired
difference against a baseline with a fold-clustered interval, because two
marginal intervals over 115 cubes overlap for reasons that have nothing to do
with the models. The same applies with more force here: hit rate is a ratio
over a handful of events, and its marginal interval is enormous while the
model's and persistence's errors are almost perfectly correlated.

So each metric is reported three times -- the model's value, persistence's value
on the SAME held-out rows and the SAME threshold, and their difference with a
DELETE-ONE-FOLD jackknife interval. The jackknife is
``p3_forecast.paired_difference``'s, generalised from R-squared to any statistic
of additive per-fold counts, and ``tests/test_p3_triggers.py`` reproduces a
``paired_difference`` result through it digit for digit rather than asserting
the two are the same in prose.

WHAT IS REFUSED
----------------
A hit rate over fewer than ``MIN_TRIGGER_EVENTS`` events is NOT reported as a
number. Three events and two hits is not a 67% hit rate. The counts are always
written -- ``n_events``, ``tp``, ``fp``, ``fn``, ``tn`` -- so the reader sees
exactly what the cell held; only the RATE is withheld, which is the same rule
``p3_forecast.summarise`` applies to a severity bin under ``_MIN_BIN_ROWS``.

    .venv/bin/python -m probes.p3_triggers \\
        --predictions data/scaled_32UNU/p3_tier1_predictions.csv \\
        --out data/scaled_32UNU/p3_tier1_triggers.csv
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np

from probes import p3_forecast as p3
from probes import p4_ceiling as p4

__all__ = [
    "TRIGGER_LEVELS", "TRIGGER_METRICS", "MIN_TRIGGER_EVENTS",
    "REPORT_KEYS", "LEARNER_KEYS", "CELL_KEYS", "VIEW_KEYS",
    "THRESHOLD_COLUMN", "THRESHOLD_QUANTILE", "INTERVAL_LABEL",
    "load_predictions", "assert_thresholds_are_train_fitted",
    "contingency", "rates", "fold_jackknife", "trigger_metrics",
    "print_trigger_table", "triggers_path",
]

#: The two crossings, and the p4 quantile each one is. Read off
#: ``p4_ceiling.SEVERITY_QUANTILES``/``SEVERITY_BINS`` rather than typed, so a
#: change to p4's severity rule cannot leave this module quoting an old number.
TRIGGER_LEVELS = ("extreme_low", "low")
THRESHOLD_COLUMN = {lv: f"anomaly_threshold_{lv}" for lv in TRIGGER_LEVELS}
THRESHOLD_QUANTILE = {lv: float(p4.SEVERITY_QUANTILES[p4.SEVERITY_BINS.index(lv)])
                      for lv in TRIGGER_LEVELS}

#: Below this many OBSERVED events (or non-events, for the rates conditioned on
#: those) a rate is not a measurement. p3_forecast._MIN_BIN_ROWS is the same
#: rule for a severity bin's pooled R-squared; this is lower because an event
#: here is by construction a tenth of the rows.
MIN_TRIGGER_EVENTS = 10

#: The axes the spec reports on. Lead time first: it is the decision-relevant
#: one, and it is what ``print_trigger_table`` puts across the columns.
REPORT_KEYS = ("delta_days", "aggregation", "fold_mode", "model_kind",
               "encoder")
#: Carried alongside because a REPORT_KEYS cell is NOT one model. The same
#: encoder appears under two estimators, two penalty rules and two feature
#: bases, and ``p3_forecast._ESTIMATOR_KEY`` already records why those must not
#: be averaged together: ridge at alpha = D and ridge at a selected alpha are
#: two learners, and a cell pooling them would report the hit rate of neither.
LEARNER_KEYS = ("feature_set", "estimator", "alpha_rule", "feature_base")
CELL_KEYS = REPORT_KEYS + LEARNER_KEYS
#: What a threshold is a property of. Not the model: every row of one view and
#: fold is scored against one line.
VIEW_KEYS = ("delta_days", "aggregation", "fold_mode")

TRIGGER_METRICS = ("hit_rate", "false_alarm_rate", "csi", "pss")

INTERVAL_LABEL = ("paired per-fold difference against persistence on the SAME "
                  "held-out rows, delete-one-fold jackknife interval "
                  "(NOT two marginal CIs)")


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

def load_predictions(path: str, verbose: bool = True):
    """The per-observation CSV, checked for the columns this module needs."""
    import pandas as pd

    assert os.path.exists(path), (
        f"no predictions at {path}. They are written by a P3 run with "
        "emit_predictions=True; the published Tier-1 run did not write them, "
        "so this file has to be produced before any trigger metric exists."
    )
    df = pd.read_csv(path)
    missing = [c for c in p3.PREDICTIONS_COLUMNS if c not in df.columns]
    assert not missing, f"{path} is missing {missing}"
    assert len(df), f"{path} holds no rows"
    if verbose:
        print(f"[p3-trig] {path}: {len(df)} held-out predictions, "
              f"{df.groupby(list(CELL_KEYS)).ngroups} cells, "
              f"horizons {sorted(df.delta_days.unique())}, "
              f"fold modes {sorted(df.fold_mode.unique())}")
    return df


def assert_thresholds_are_train_fitted(df, verbose: bool = True) -> dict:
    """REFUSE a file whose thresholds could have been fitted on the test rows.

    A quantile fitted on the full sample is the failure this checks for, and it
    is invisible in the numbers it produces: every rate stays in range, the
    contingency tables stay consistent, and the crossing rate is simply a
    little too close to the nominal 10%. So it is checked four ways, and three
    of them would fail on a full-sample fit:

    1. ONE threshold per (view, fold), whatever the model. A threshold that
       varied by model would mean it came from the predictions.
    2. It MOVES across the folds of a view. A full-sample quantile is the same
       number on every fold, so this is the direct check.
    3. It is not the full-sample quantile ON EVERY FOLD. The same failure from
       the other side, and it still bites where (2) is vacuous -- a view left
       with one fold. One fold coinciding is allowed: a leave-one-fold-out
       quantile can land on the same order statistic by accident, and all of
       them landing there cannot.
    4. The arithmetic: the rows the threshold was fitted on, plus the rows the
       fold held out, cannot exceed the rows the view has. A fit on everything
       overflows this.
    """
    out = {}
    grp = list(VIEW_KEYS)
    for lv in TRIGGER_LEVELS:
        col = THRESHOLD_COLUMN[lv]
        assert np.isfinite(df[col].to_numpy()).all(), f"{col} has non-finite rows"

        # (1) one line per (view, fold), across every model row of the cell
        per_fold = df.groupby(grp + ["fold_index"])[col].nunique()
        assert (per_fold == 1).all(), (
            f"{col} takes more than one value inside a (view, fold): "
            f"{per_fold[per_fold != 1].to_dict()}. The threshold is a property "
            "of the fold, so every model row scored on that fold must have "
            "been scored against the same line."
        )

        # (2) and (3): it moves fold to fold, and it is not the full-sample one
        moved, pinned = 0, []
        for key, g in df.groupby(grp):
            k = g.fold_index.nunique()
            edges = g.groupby("fold_index")[col].first().to_numpy()
            if k >= 2:
                assert np.unique(edges).size > 1, (
                    f"{col} is identical on all {k} folds of {key}. A quantile "
                    "fitted on the full sample is the same number on every "
                    "fold; a quantile fitted inside each fold is not."
                )
                moved += 1
            anom = (g.y_true - g.y_climatology_proxy).to_numpy()
            full = float(np.quantile(anom[np.isfinite(anom)],
                                     THRESHOLD_QUANTILE[lv]))
            # EVERY fold, not any one of them. A leave-one-fold-out quantile
            # can land on the same order statistic as the full-sample one by
            # coincidence, and refusing a file for one such fold would be a
            # false alarm about a false-alarm table. All of them agreeing is
            # not a coincidence -- it is the definition of a full-sample fit.
            if bool((edges == full).all()):
                pinned.append((key, int(edges.size)))
        assert not pinned, (
            f"{col} equals the FULL-SAMPLE quantile on EVERY fold of {pinned}. "
            "That quantile has seen the held-out rows, so the crossing rate it "
            "defines is partly a property of the test side."
        )

        # (4) train rows + this fold's held-out rows <= the view's rows
        for key, g in df.groupby(grp):
            n_view = g.feature_row.nunique()
            for fold, gf in g.groupby("fold_index"):
                n_tr = int(gf.n_threshold_train_rows.iloc[0])
                n_te = int(gf.feature_row.nunique())
                assert n_tr > 0, (key, fold, n_tr)
                assert n_tr + n_te <= n_view, (
                    f"{key} fold {fold}: the threshold was fitted on {n_tr} "
                    f"rows and {n_te} rows were held out, but the view has "
                    f"only {n_view}. The fit reached into the test side."
                )
        out[lv] = {"views_with_moving_edges": moved,
                   "quantile": THRESHOLD_QUANTILE[lv]}
    if verbose:
        print("[p3-trig] thresholds are TRAIN-FITTED: one line per (view, "
              "fold), moving across folds in "
              f"{out[TRIGGER_LEVELS[0]]['views_with_moving_edges']} views, "
              "never the full-sample quantile, and never fitted on more rows "
              "than the view has outside the fold")
    return out


# ---------------------------------------------------------------------------
# The contingency table and its rates
# ---------------------------------------------------------------------------

def contingency(observed, forecast) -> np.ndarray:
    """(tp, fp, fn, tn) for two boolean arrays. Counts, never rates."""
    o = np.asarray(observed, dtype=bool)
    f = np.asarray(forecast, dtype=bool)
    assert o.shape == f.shape, (o.shape, f.shape)
    return np.array([int((o & f).sum()), int((~o & f).sum()),
                     int((o & ~f).sum()), int((~o & ~f).sum())], dtype=np.int64)


def rates(counts) -> dict:
    """The four metrics, from (tp, fp, fn, tn). NaN where undefined.

    ``false_alarm_rate`` is the PROBABILITY OF FALSE DETECTION, fp / (fp + tn) --
    the false alarms as a fraction of the non-events. That is the quantity the
    Peirce skill score subtracts, so naming it anything else would leave the
    skill score's own definition unstated. The other convention, fp / (tp + fp),
    is the false-alarm RATIO and is reported separately as ``far``: they differ
    by an order of magnitude when events are rare, which is exactly this case.
    """
    tp, fp, fn, tn = (float(x) for x in counts)
    pod = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    pofd = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
    far = fp / (tp + fp) if (tp + fp) > 0 else float("nan")
    return {"hit_rate": pod, "false_alarm_rate": pofd, "csi": csi,
            "far": far, "pss": pod - pofd}


def _support(counts) -> dict:
    """How many OBSERVATIONS each metric's denominator actually rests on.

    Not the row count. A hit rate over 900 rows and 3 events is a hit rate over
    3 events, and this is the number ``MIN_TRIGGER_EVENTS`` is applied to.
    """
    tp, fp, fn, tn = (int(x) for x in counts)
    events, non_events = tp + fn, fp + tn
    return {"hit_rate": events, "false_alarm_rate": non_events,
            "csi": events, "far": events,
            "pss": min(events, non_events)}


def fold_jackknife(per_fold, stat, alpha: float = 0.05) -> dict:
    """A statistic of additive per-fold counts, with a DELETE-ONE-FOLD interval.

    ``p3_forecast.paired_difference``'s estimator, generalised. There the
    statistic is R2(a) - R2(b) and the per-fold sufficient statistics are
    (n, sum y, sum y^2, SSE_a, SSE_b); here it is a difference of skill scores
    and they are two contingency tables. The arithmetic is unchanged and
    deliberately so -- folds hold disjoint sets of cubes, so deleting one
    deletes a CLUSTER, and the fold effect the model and the baseline share
    cancels inside the difference before its spread is taken.

    ``per_fold`` is (K, M) and ``stat`` maps a summed length-M vector to a
    scalar. Rebuilt from sums rather than by re-filtering the rows K times, for
    the reason ``paired_difference`` gives: at 115 leave-one-cube-out folds and
    two levels on every cell of a wide table, the naive form is the difference
    between seconds and an hour.
    """
    from scipy.stats import t as student_t

    per_fold = np.asarray(per_fold, dtype=np.float64)
    assert per_fold.ndim == 2, per_fold.shape
    K = per_fold.shape[0]
    total = per_fold.sum(axis=0)
    theta = float(stat(total))
    out = {"value": theta, "ci_lo": float("nan"), "ci_hi": float("nan"),
           "separable": False, "n_folds": int(K)}
    if K < 2 or not np.isfinite(theta):
        return out
    loo = np.array([stat(total - per_fold[i]) for i in range(K)], dtype=float)
    loo = loo[np.isfinite(loo)]
    if loo.size < 2:
        return out
    m = loo.mean()
    se = np.sqrt((loo.size - 1) / loo.size * ((loo - m) ** 2).sum())
    half = float(student_t.ppf(1 - alpha / 2, loo.size - 1) * se)
    lo, hi = theta - half, theta + half
    out.update(ci_lo=float(lo), ci_hi=float(hi),
               separable=bool(np.isfinite(lo) and np.isfinite(hi)
                              and (lo > 0 or hi < 0)))
    return out


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

def _cell_rows(key: dict, g, level: str, alpha: float) -> dict:
    """One (cell, level) row: counts, rates, persistence, paired differences."""
    tau = g[THRESHOLD_COLUMN[level]].to_numpy()
    anom_true = (g.y_true - g.y_climatology_proxy).to_numpy()
    anom_pred = (g.y_pred - g.y_climatology_proxy).to_numpy()
    anom_pers = (g.y_persistence - g.y_climatology_proxy).to_numpy()
    obs = anom_true < tau
    fc_model = anom_pred < tau
    fc_pers = anom_pers < tau

    fold = g.fold_index.to_numpy()
    uf, f_idx = np.unique(fold, return_inverse=True)
    per_fold = np.zeros((uf.size, 8), dtype=np.float64)
    for i in range(uf.size):
        m = f_idx == i
        per_fold[i, :4] = contingency(obs[m], fc_model[m])
        per_fold[i, 4:] = contingency(obs[m], fc_pers[m])
    total = per_fold.sum(axis=0)
    c_model, c_pers = total[:4], total[4:]

    r_model, r_pers = rates(c_model), rates(c_pers)
    sup = _support(c_model)
    row = dict(key)
    row["level"] = level
    row["threshold_quantile"] = THRESHOLD_QUANTILE[level]
    row["threshold_median"] = float(np.median(tau))
    row["threshold_min"] = float(tau.min())
    row["threshold_max"] = float(tau.max())
    row["n_threshold_train_rows_median"] = float(
        np.median(g.n_threshold_train_rows.to_numpy()))
    row["n_rows"] = int(len(g))
    row["n_folds"] = int(uf.size)
    row["n_cubes"] = int(g.cube_id.nunique())
    row["n_events"] = int(c_model[0] + c_model[2])
    row["n_non_events"] = int(c_model[1] + c_model[3])
    row["event_rate"] = float(row["n_events"] / row["n_rows"])
    for name, c in (("", c_model), ("_persistence", c_pers)):
        for j, f in enumerate(("tp", "fp", "fn", "tn")):
            row[f"{f}{name}"] = int(c[j])
    row["min_events"] = MIN_TRIGGER_EVENTS
    row["enough_events"] = bool(sup["hit_rate"] >= MIN_TRIGGER_EVENTS)
    row["interval"] = INTERVAL_LABEL

    for m in TRIGGER_METRICS + ("far",):
        ok = sup[m] >= MIN_TRIGGER_EVENTS
        row[f"n_support_{m}"] = int(sup[m])
        row[m] = r_model[m] if ok else float("nan")
        row[f"{m}_persistence"] = r_pers[m] if ok else float("nan")
    for m in TRIGGER_METRICS:
        if sup[m] < MIN_TRIGGER_EVENTS:
            row[f"diff_{m}"] = float("nan")
            row[f"ci_lo_{m}"] = float("nan")
            row[f"ci_hi_{m}"] = float("nan")
            row[f"separable_{m}"] = False
            continue

        def stat(v, _m=m):
            return rates(v[:4])[_m] - rates(v[4:])[_m]

        j = fold_jackknife(per_fold, stat, alpha=alpha)
        row[f"diff_{m}"] = j["value"]
        row[f"ci_lo_{m}"] = j["ci_lo"]
        row[f"ci_hi_{m}"] = j["ci_hi"]
        row[f"separable_{m}"] = bool(j["separable"])
    return row


def trigger_metrics(df, levels: Sequence[str] = TRIGGER_LEVELS,
                    alpha: float = 0.05, check_thresholds: bool = True,
                    verbose: bool = True):
    """One row per (cell, level). The whole table.

    ``check_thresholds`` runs the leak refusal first, because every number
    below is a statement about a threshold and a threshold fitted on the full
    sample makes all of them out-of-sample in name only.
    """
    import pandas as pd

    if check_thresholds:
        assert_thresholds_are_train_fitted(df, verbose=verbose)
    rows = []
    for key, g in df.groupby(list(CELL_KEYS), sort=True):
        k = dict(zip(CELL_KEYS, key))
        for lv in levels:
            rows.append(_cell_rows(k, g, lv, alpha))
    out = pd.DataFrame(rows)
    front = list(REPORT_KEYS) + ["level"] + list(LEARNER_KEYS)
    out = out[front + [c for c in out.columns if c not in front]]
    out = out.sort_values(front).reset_index(drop=True)
    if verbose:
        thin = int((~out.enough_events).sum())
        print(f"[p3-trig] {len(out)} (cell, level) rows over "
              f"{out.groupby(list(CELL_KEYS)).ngroups} cells. "
              f"{thin} have fewer than {MIN_TRIGGER_EVENTS} events and carry "
              "counts but no rate.")
    return out


# ---------------------------------------------------------------------------
# Reporting: lead time across the columns
# ---------------------------------------------------------------------------

def _fmt(x, w=6, p=3):
    return f"{x:+{w}.{p}f}" if np.isfinite(x) else " " * (w - 3) + "n/a"


def print_trigger_table(tr, level: str = "extreme_low",
                        aggregation: str = "cube_mean",
                        fold_mode: str = "cube",
                        estimator: str = "linear",
                        alpha_rule: str = p3.ALPHA_RULE_TUNED,
                        feature_base: str = p3.FEATURE_BASE_SHARED,
                        metric: str = "pss") -> None:
    """One learner, one level, HORIZON ACROSS THE COLUMNS.

    Lead time is the decision-relevant axis -- a trigger that fires five days
    out and one that fires a hundred days out are different products -- so it is
    the axis a reader scans, and the model's value sits beside persistence's on
    the same rows and the same threshold.
    """
    sub = tr[(tr.level == level) & (tr.aggregation == aggregation)
             & (tr.fold_mode == fold_mode)]
    horizons = sorted(sub.delta_days.unique())
    w, lw = 25, 48
    print("\n" + "=" * (lw + 2 + w * len(horizons)))
    print(f"TRIGGER: anomaly below the {level} line "
          f"({THRESHOLD_QUANTILE[level]:.0%} of the TRAIN-fold anomaly "
          f"distribution) | {metric}")
    print(f"  {aggregation} / {fold_mode} folds / {estimator} / {alpha_rule} / "
          f"base={feature_base}")
    print("  model value, (persistence), then the PAIRED difference. "
          "n = EVENTS, not rows.")
    print("=" * (lw + 2 + w * len(horizons)))
    print(f"  {'row':<{lw}}"
          + "".join(f"{('D=' + str(H) + 'd'):>{w}}" for H in horizons))
    # An unfitted baseline has no learner to match; a control takes no shared
    # base (p3.BASE_ROW_KINDS), so filtering it on one would drop it entirely
    # and the table would lose the rows it is supposed to be read against.
    takes_base = sub.model_kind.isin(p3.BASE_ROW_KINDS)
    sel = sub[(sub.estimator == "none")
              | ((sub.estimator == estimator) & (sub.alpha_rule == alpha_rule)
                 & (~takes_base | (sub.feature_base == feature_base)))]
    for name, g in sel.groupby(["model_kind", "encoder", "feature_set"]):
        cells = []
        for H in horizons:
            r = g[g.delta_days == H]
            if not len(r):
                cells.append("--".rjust(w))
            elif not np.isfinite(r.iloc[0][metric]):
                cells.append(f"n={int(r.iloc[0].n_events)} <min".rjust(w))
            else:
                r = r.iloc[0]
                star = "*" if r[f"separable_{metric}"] else " "
                cells.append(
                    (f"{_fmt(r[metric])} ({_fmt(r[metric + '_persistence'])})"
                     f" {_fmt(r['diff_' + metric], 6, 2)}{star}").rjust(w))
        print(f"  {'/'.join(str(x) for x in name):<{lw}}" + "".join(cells))
    n = sub.groupby("delta_days").n_events.max().to_dict()
    print(f"  {'EVENTS at this horizon':<{lw}}"
          + "".join(f"{n.get(H, 0):>{w}}" for H in horizons))
    print(f"  * = the paired interval excludes zero (separable from "
          f"persistence at {metric}); '<min' = fewer than "
          f"{MIN_TRIGGER_EVENTS} events, so no rate is reported")


def triggers_path(name: str = "p3_triggers.csv", root: str | None = None) -> str:
    """Where a trigger table is written. ``p3_forecast.results_path``'s rule."""
    return p3.results_path(name, root=root)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--predictions",
                    default=os.path.join("data", "scaled_32UNU",
                                         "p3_tier1_predictions.csv"),
                    help="the per-observation CSV (.csv or .csv.gz)")
    ap.add_argument("--out", default=None,
                    help="where the trigger table goes; default beside the "
                         "predictions, as p3_tier1_triggers.csv")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--no-threshold-check", action="store_true",
                    help="SKIP the train-fitted refusal. For inspecting a "
                         "broken file only; any table it produces is not a "
                         "result.")
    args = ap.parse_args()

    path = args.predictions
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        path += ".gz"
    df = load_predictions(path, verbose=True)
    tr = trigger_metrics(df, alpha=args.alpha,
                         check_thresholds=not args.no_threshold_check,
                         verbose=True)
    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(path)),
        os.path.basename(path).replace("_predictions.csv", "_triggers.csv")
        .replace(".gz", ""))
    tr.to_csv(out, index=False)
    print(f"[p3-trig] wrote {out} ({tr.shape[0]} rows x {tr.shape[1]} cols)")

    back = pd.read_csv(out)
    assert back.shape == tr.shape, (back.shape, tr.shape)
    for mode in sorted(df.fold_mode.unique()):
        for lv in TRIGGER_LEVELS:
            for m in ("hit_rate", "pss"):
                print_trigger_table(tr, level=lv, fold_mode=mode, metric=m)


if __name__ == "__main__":
    main()
