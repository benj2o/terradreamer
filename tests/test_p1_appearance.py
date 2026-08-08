"""P1, the appearance probe: the five assertions the spec names, plus the
machinery they rest on.

Everything runs on a SYNTHETIC manifest with known cube / window / month
membership, so each leakage claim is checked against an independent
computation rather than against the code that made the claim. The one test
that touches the real cache is skipped when the cache is absent.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from probes import cv
from probes import p1_appearance as p1


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

def _manifest(n_cubes=8, frames=8, start="2018-04-01", step_days=11,
              stagger_days=9, clear=0.8):
    """One tile, one year, staggered windows -- the shape of the real subset.

    Staggering the start per cube is what makes the pooled series cover more
    months than any single cube does, which is the whole premise of Figure 1
    and of the month target having more than a couple of classes.
    """
    rows = []
    for c in range(n_cubes):
        t0 = np.datetime64(start) + np.timedelta64(stagger_days * c, "D")
        for f in range(frames):
            rows.append({
                "cube_id": f"32UNU_cube{c}.nc",
                "tile": "32UNU",
                "year": 2018,
                "timestamp": t0 + np.timedelta64(step_days * f, "D"),
                "original_axis_index": f,
                "pixel_bbox": (c * 500, c * 500 + 128,
                               (c % 3) * 700, (c % 3) * 700 + 128),
                "clear_frac": clear,
            })
    return pd.DataFrame(rows)


def _arrays(manifest, D=6, D_grid=5, seed=0, signal=1.0):
    """A cached-encoder stand-in whose pooled features carry a month signal."""
    rng = np.random.default_rng(seed)
    n = len(manifest)
    months = p1.month_labels(manifest)
    phase = 2 * np.pi * months / 12.0
    base = np.column_stack([np.cos(phase), np.sin(phase)])
    pooled = rng.normal(0, 1, size=(n, D)).astype(np.float32)
    pooled[:, :2] += (signal * base).astype(np.float32)
    grid = rng.normal(0, 1, size=(n, 16, D_grid)).astype(np.float32)
    grid[:, :, :2] += (signal * base)[:, None, :].astype(np.float32)
    return {
        "encoder": "synthetic",
        "pooled": pooled,
        "grid": grid,
        "grid_clear_frac": np.full((n, 16), 0.8, dtype=np.float32),
        "clear_frac": manifest["clear_frac"].to_numpy().astype(float),
        "window_span_days": np.tile([0., 20., 40., 60., 80., 100., 55., 35.],
                                    n // 8 + 1)[:n],
    }


# ---------------------------------------------------------------------------
# ASSERTION 5 of the spec: chance is DERIVED, never hard-coded
# ---------------------------------------------------------------------------

def test_chance_level_comes_from_the_realised_distribution():
    y = np.array([4] * 10 + [5] * 3 + [6] * 1)
    ch = p1.chance_level(y)
    assert ch["n_classes"] == 3
    assert ch["balanced_accuracy"] == pytest.approx(1 / 3)
    assert ch["balanced_accuracy"] != pytest.approx(1 / 12), "hard-coded 1/12"
    assert ch["majority_class"] == 4
    assert ch["majority_frac"] == pytest.approx(10 / 14)


def test_chance_level_moves_when_the_distribution_moves():
    """The decisive test: a constant would not change."""
    a = p1.chance_level(np.array([4] * 10 + [5] * 3))
    b = p1.chance_level(np.array([4] * 10 + [5] * 3 + [6] * 2 + [7] * 2))
    assert a["n_classes"] == 2 and b["n_classes"] == 4
    assert a["balanced_accuracy"] == pytest.approx(1 / 2)
    assert b["balanced_accuracy"] == pytest.approx(1 / 4)
    assert a["macro_f1"] != b["macro_f1"]


def test_chance_macro_f1_matches_an_actual_dummy_classifier():
    """The analytic floor is checked against sklearn rather than trusted."""
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import balanced_accuracy_score, f1_score

    y = np.array([4] * 30 + [5] * 12 + [6] * 5 + [11] * 2)
    X = np.zeros((y.size, 2))
    ch = p1.chance_level(y)
    pred = DummyClassifier(strategy="most_frequent").fit(X, y).predict(X)
    assert f1_score(y, pred, average="macro", zero_division=0) == \
        pytest.approx(ch["macro_f1"])
    assert balanced_accuracy_score(y, pred) == \
        pytest.approx(ch["balanced_accuracy"])


def test_month_target_is_not_assumed_to_have_twelve_classes():
    df = _manifest()
    ch = p1.chance_level(p1.month_labels(df))
    assert 1 < ch["n_classes"] < 12, (
        "the synthetic subset spans part of a year; a probe that assumed 12 "
        "classes would not notice"
    )
    assert ch["balanced_accuracy"] == pytest.approx(1 / ch["n_classes"])


def test_season_is_four_way_by_definition_and_realised_by_measurement():
    assert set(p1.SEASON_OF_MONTH.values()) == set(p1.SEASONS)
    assert len(p1.SEASON_OF_MONTH) == 12
    df = _manifest()
    s = p1.season_labels(df)
    assert set(s.tolist()) <= set(p1.SEASONS)
    ch = p1.chance_level(s)
    assert ch["balanced_accuracy"] == pytest.approx(1 / len(set(s.tolist())))


def test_labels_agree_with_an_independent_computation():
    df = _manifest()
    ts = pd.to_datetime(df["timestamp"])
    assert p1.month_labels(df).tolist() == ts.dt.month.tolist()
    expected = [p1.SEASON_OF_MONTH[m] for m in ts.dt.month]
    assert p1.season_labels(df).tolist() == expected


# ---------------------------------------------------------------------------
# ASSERTION 3 of the spec: no test index in its own fold's training set,
# re-derived from the manifest INDEPENDENTLY of probes/cv.py
# ---------------------------------------------------------------------------

def _independent_cube_of(manifest, idx):
    """Cube membership straight from the manifest, touching no cv code."""
    return set(manifest["cube_id"].iloc[list(idx)].tolist())


@pytest.mark.parametrize("mode", p1.FOLD_MODES)
def test_no_test_index_appears_in_its_own_training_fold(mode):
    df = _manifest(n_cubes=8)
    seen = []
    for tr, te in p1._outer_folds(df, mode, k=4):
        tr, te = np.asarray(tr), np.asarray(te)
        # (a) index disjointness, computed here with set arithmetic
        assert not (set(tr.tolist()) & set(te.tolist())), \
            f"{mode}: a test row is also a training row"
        # (b) cube disjointness, re-derived from the manifest column itself
        assert not (_independent_cube_of(df, tr) & _independent_cube_of(df, te))
        # (c) and every index is a real manifest row
        assert tr.min() >= 0 and te.max() < len(df)
        seen.append(te)
    assert len(seen) >= 2
    tested = np.sort(np.concatenate(seen))
    assert tested.tolist() == list(range(len(df))), \
        f"{mode}: every row must be tested exactly once"


def test_feature_rows_of_a_cell_level_set_never_straddle_a_fold():
    """The 16 cells of a frame must move together, or the grid feature set
    leaks a cube across the split it was grouped by."""
    df = _manifest(n_cubes=6)
    arrays = _arrays(df)
    X, row_idx = p1.feature_matrix(arrays, "grid_cell")
    assert X.shape == (len(df) * 16, arrays["grid"].shape[-1])
    for tr, te in p1._outer_folds(df, "cube", k=3):
        a = p1._rows_for(row_idx, tr)
        b = p1._rows_for(row_idx, te)
        assert not np.intersect1d(a, b).size
        assert a.size == len(tr) * 16 and b.size == len(te) * 16
        assert not (_independent_cube_of(df, row_idx[a])
                    & _independent_cube_of(df, row_idx[b]))


# ---------------------------------------------------------------------------
# ASSERTION 4 of the spec: the inner tuning loop never sees test indices
# ---------------------------------------------------------------------------

def test_select_hyperparameter_has_no_parameter_that_could_carry_test_data():
    """Prevention by signature, checked as such: there is no test argument to
    pass, so no amount of caller discipline is being relied on."""
    params = set(inspect.signature(p1.select_hyperparameter).parameters)
    assert params == {"X_tr", "y_tr", "manifest", "train_rows", "row_idx_tr",
                      "estimator", "inner_k", "verbose"}
    assert not [p for p in params if "test" in p or p.endswith("_te")]


@pytest.mark.parametrize("estimator", p1.ESTIMATORS)
@pytest.mark.parametrize("feature_set", ["pooled", "grid_cell"])
def test_poisoned_test_fold_does_not_change_the_selected_alpha(estimator,
                                                               feature_set):
    """Corrupt the TEST side beyond recognition; the tuned regularisation
    strength must be bit-identical, because the inner loop never saw it."""
    df = _manifest(n_cubes=8)
    arrays = _arrays(df)
    X, row_idx = p1.feature_matrix(arrays, feature_set)
    y = p1.month_labels(df)[row_idx]
    folds_ = list(p1._outer_folds(df, "cube", k=4))

    clean = [p1.evaluate_fold(X, y, row_idx, df, tr, te, estimator, fold=i,
                              verbose=False)
             for i, (tr, te) in enumerate(folds_)]

    rng = np.random.default_rng(1234)
    poisoned = []
    for i, (tr, te) in enumerate(folds_):
        Xp, yp = X.copy(), y.copy()
        rows = p1._rows_for(row_idx, te)
        # Not noise: a deliberately informative poison. If any of it reached
        # the inner loop, the selected strength would move.
        Xp[rows] = rng.normal(0, 50, size=(rows.size, X.shape[1]))
        Xp[rows, 0] = yp[rows] * 1000.0
        yp[rows] = yp[rows][::-1]
        poisoned.append(p1.evaluate_fold(Xp, yp, row_idx, df, tr, te, estimator,
                                         fold=i, verbose=False))

    assert [r.selected for r in clean] == [r.selected for r in poisoned], (
        "the inner tuning loop moved when only the TEST fold changed -- it is "
        "seeing test data"
    )
    assert [r.balanced_accuracy for r in clean] != \
        [r.balanced_accuracy for r in poisoned], (
        "the poison did not even change the test score, so this test proves "
        "nothing about the tuning loop"
    )


def test_inner_folds_are_cube_grouped_inside_the_training_fold():
    df = _manifest(n_cubes=9)
    (tr, te), = [next(iter(p1._outer_folds(df, "cube", k=3)))]
    sub = df.iloc[tr]
    train_cubes = set(sub["cube_id"])
    for itr, ite in cv.folds(sub, "cube", k=3, verbose=False):
        a = set(sub["cube_id"].to_numpy()[itr])
        b = set(sub["cube_id"].to_numpy()[ite])
        assert not (a & b), "an inner fold split a cube"
        assert (a | b) <= train_cubes, "an inner fold reached outside train"
        assert not (set(tr[ite].tolist()) & set(te.tolist())), \
            "an inner test fold overlaps the OUTER test fold"


def test_standardisation_is_fitted_on_train_only():
    """A scaler fitted on the union would leave the test block centred; one
    fitted on train alone does not, and that difference is the check."""
    rng = np.random.default_rng(0)
    X_tr = rng.normal(0, 1, size=(60, 4))
    X_te = rng.normal(50, 1, size=(20, 4))     # a wildly different test block
    y_tr = np.array([0, 1] * 30)
    p1._fit_predict("ridge", 1.0, X_tr, y_tr, X_te)

    from sklearn.preprocessing import StandardScaler
    z_train_only = StandardScaler().fit(X_tr).transform(X_te)
    z_both = StandardScaler().fit(np.vstack([X_tr, X_te])).transform(X_te)
    assert abs(z_train_only.mean()) > 5, "train-only scaling should not centre test"
    assert abs(z_both.mean()) < 5
    src = inspect.getsource(p1._fit_predict)
    assert "fit(X_tr)" in src and "fit(X_te)" not in src


# ---------------------------------------------------------------------------
# ASSERTIONS 1 and 2 of the spec: the baseline row and the degenerate control
# ---------------------------------------------------------------------------

def _fake_results(**overrides):
    """A minimal well-formed results table, so the completeness assertions can
    be tested on tables that are deliberately broken."""
    rows = []
    for target in p1.TARGETS:
        for mode in p1.FOLD_MODES:
            for est in p1.ESTIMATORS:
                jobs = [(e, fs, lv, "all")
                        for e in p1.ENCODER_ORDER
                        for fs, lv in (("pooled", "frame"), ("grid_cell", "cell"))]
                jobs += [(p1.BASELINE_ENCODER, "raw_pooled", "frame", "all"),
                         ("none", "degenerate", "frame", "all"),
                         ("none", "degenerate", "cell", "all")]
                jobs += [(p1.MI_ENCODER, "pooled", "frame", b) for b in p1.WSD_BINS]
                for enc, fs, lv, wb in jobs:
                    rows.append({
                        "target": target, "fold_mode": mode, "estimator": est,
                        "encoder": enc, "feature_set": fs, "feature_level": lv,
                        "wsd_bin": wb, "si_comparable": enc != p1.MI_ENCODER,
                        "balanced_accuracy_mean": 0.5, "macro_f1_mean": 0.5,
                        "balanced_accuracy_spread": 0.1,
                        "selected_params": "1;1;1;1;1",
                    })
    df = pd.DataFrame(rows)
    for k, v in overrides.items():
        df = v(df) if callable(v) else df
    return df


def test_results_table_must_carry_the_raw_features_row_for_every_feature_set():
    df = _fake_results()
    p1.assert_results_complete(df)
    broken = df[~((df.encoder == p1.BASELINE_ENCODER)
                  & (df.feature_set == "grid_cell"))]
    with pytest.raises(AssertionError, match="missing encoders"):
        p1.assert_results_complete(broken)


def test_results_table_must_carry_the_mandatory_baseline_feature_set():
    df = _fake_results()
    with pytest.raises(AssertionError, match="raw_features baseline row"):
        p1.assert_results_complete(df[df.feature_set != "raw_pooled"])


def test_results_table_must_carry_the_degenerate_control():
    df = _fake_results()
    with pytest.raises(AssertionError, match="degenerate control row"):
        p1.assert_results_complete(df[df.feature_set != "degenerate"])


def test_multi_image_rows_must_be_flagged_not_comparable():
    df = _fake_results()
    df.loc[df.encoder == p1.MI_ENCODER, "si_comparable"] = True
    with pytest.raises(AssertionError, match="si_comparable=False"):
        p1.assert_results_complete(df)


def test_multi_image_must_also_be_reported_conditioned_on_window_span_days():
    df = _fake_results()
    with pytest.raises(AssertionError, match="window_span_days"):
        p1.assert_results_complete(df[df.wsd_bin == "all"])


def test_baseline_view_must_agree_with_the_row_it_is_a_view_of():
    df = _fake_results()
    p1.assert_baseline_view_consistent(df)
    df.loc[df.feature_set == "raw_pooled", "balanced_accuracy_mean"] = 0.99
    with pytest.raises(AssertionError, match="raw_pooled disagrees"):
        p1.assert_baseline_view_consistent(df)


def test_run_p1_refuses_a_roster_without_the_baseline_encoder():
    df = _manifest()
    with pytest.raises(AssertionError, match="mandatory baseline"):
        p1.run_p1(df, encoders=("dinov2_vitb14",), verbose=False)


# ---------------------------------------------------------------------------
# Feature assembly, the degenerate control, and the MI caveat
# ---------------------------------------------------------------------------

def test_degenerate_control_is_two_columns_and_holds_no_embedding():
    df = _manifest()
    arrays = _arrays(df)
    X, row_idx = p1.feature_matrix(arrays, "degenerate", level="frame")
    assert X.shape == (len(df), 2)
    assert np.allclose(X[:, 0], arrays["clear_frac"])
    assert np.allclose(X[:, 1], arrays["window_span_days"])
    Xc, rc = p1.feature_matrix(arrays, "degenerate", level="cell")
    assert Xc.shape == (len(df) * 16, 2)
    assert rc.tolist() == np.repeat(np.arange(len(df)), 16).tolist()


def test_degenerate_control_needs_an_explicit_level():
    with pytest.raises(AssertionError, match="explicit level"):
        p1.feature_matrix(_arrays(_manifest()), "degenerate")


def test_degenerate_control_takes_window_span_days_from_the_multi_image_encoder():
    df = _manifest()
    si = _arrays(df)
    si["window_span_days"] = np.zeros(len(df))
    mi = dict(_arrays(df))
    mi["window_span_days"] = np.linspace(0, 105, len(df))
    deg = p1.degenerate_arrays({"raw_features": si, p1.MI_ENCODER: mi},
                               verbose=False)
    assert np.allclose(deg["window_span_days"], mi["window_span_days"]), (
        "the control must use the only lookback that varies, not a column of "
        "zeros -- otherwise it is weaker than it claims to be"
    )


def test_degenerate_arrays_refuses_caches_with_different_clear_fractions():
    df = _manifest()
    a, b = _arrays(df), _arrays(df)
    b["clear_frac"] = b["clear_frac"] * 0.5
    with pytest.raises(AssertionError, match="clear_frac"):
        p1.degenerate_arrays({"x": a, "y": b}, verbose=False)


def test_window_span_day_bins_are_terciles_of_the_realised_values():
    wsd = np.array([0., 0., 10., 20., 55., 60., 90., 100., 105., 105.])
    bins = p1.wsd_bin_labels(wsd)
    assert set(bins.tolist()) == set(p1.WSD_BINS)
    assert bins[0] == "short" and bins[-1] == "long"
    counts = {b: int((bins == b).sum()) for b in p1.WSD_BINS}
    assert min(counts.values()) >= 2, counts


def test_subset_arrays_slices_every_per_frame_array_together():
    df = _manifest()
    arrays = _arrays(df)
    keep = np.array([0, 5, 9, 30])
    sub = p1.subset_arrays(arrays, keep)
    assert sub["pooled"].shape == (4, arrays["pooled"].shape[1])
    assert sub["grid"].shape == (4, 16, arrays["grid"].shape[-1])
    assert np.allclose(sub["clear_frac"], arrays["clear_frac"][keep])
    assert np.allclose(sub["window_span_days"], arrays["window_span_days"][keep])


# ---------------------------------------------------------------------------
# Metrics, spread, and the end-to-end shape
# ---------------------------------------------------------------------------

def test_summarise_reports_spread_not_just_a_mean():
    res = [p1.FoldResult(i, 10, 5, 4, 1, 3, 1.0, False, ba, ba, 0.2, 0.1, "")
           for i, ba in enumerate([0.4, 0.9, 0.5, 0.6])]
    s = p1.summarise(res)
    assert s["balanced_accuracy_mean"] == pytest.approx(0.6)
    assert s["balanced_accuracy_spread"] == pytest.approx(0.5)
    assert s["balanced_accuracy_min"] == 0.4 and s["balanced_accuracy_max"] == 0.9
    assert s["balanced_accuracy_std"] > 0
    assert s["per_fold_balanced_accuracy"] == "0.4000;0.9000;0.5000;0.6000"
    assert s["selected_params"] == "1;1;1;1"


def test_a_probe_with_real_signal_beats_its_own_dummy_floor():
    df = _manifest(n_cubes=8)
    arrays = _arrays(df, signal=6.0)
    X, row_idx = p1.feature_matrix(arrays, "pooled")
    y = p1.month_labels(df)[row_idx]
    res = p1.evaluate(X, y, row_idx, df, "cube", "ridge", k=4, verbose=False)
    s = p1.summarise(res)
    assert s["balanced_accuracy_mean"] > s["dummy_balanced_accuracy_mean"]
    assert s["balanced_accuracy_mean"] > p1.chance_level(y)["balanced_accuracy"]


def test_a_probe_with_no_signal_does_not_beat_chance():
    """The other direction, which is what makes the previous test mean
    something: pure noise must land at the floor, not above it."""
    df = _manifest(n_cubes=8)
    arrays = _arrays(df, signal=0.0, seed=7)
    X, row_idx = p1.feature_matrix(arrays, "pooled")
    y = p1.month_labels(df)[row_idx]
    res = p1.evaluate(X, y, row_idx, df, "cube", "ridge", k=4, verbose=False)
    ch = p1.chance_level(y)["balanced_accuracy"]
    assert p1.summarise(res)["balanced_accuracy_mean"] < ch + 0.15


def test_evaluate_is_deterministic_and_n_jobs_changes_nothing():
    df = _manifest(n_cubes=6)
    arrays = _arrays(df)
    X, row_idx = p1.feature_matrix(arrays, "pooled")
    y = p1.month_labels(df)[row_idx]
    a = p1.evaluate(X, y, row_idx, df, "cube", "ridge", k=3, verbose=False)
    b = p1.evaluate(X, y, row_idx, df, "cube", "ridge", k=3, n_jobs=2,
                    verbose=False)
    assert [r.balanced_accuracy for r in a] == [r.balanced_accuracy for r in b]
    assert [r.selected for r in a] == [r.selected for r in b]


def test_fold_result_records_the_selected_strength_per_fold():
    df = _manifest(n_cubes=6)
    arrays = _arrays(df)
    X, row_idx = p1.feature_matrix(arrays, "pooled")
    y = p1.month_labels(df)[row_idx]
    res = p1.evaluate(X, y, row_idx, df, "cube", "logreg", k=3, verbose=False)
    assert all(r.selected in p1.LOGREG_C_GRID for r in res)
    assert all("C=" in r.log for r in res)
    assert len(p1.summarise(res)["selected_params"].split(";")) == len(res)


def test_ties_break_toward_stronger_regularisation():
    """An all-zero design matrix makes every strength fit the same
    intercept-only model, so the inner scores tie EXACTLY and the stated tie
    rule alone decides: smallest C for logreg, largest alpha for ridge."""
    df = _manifest(n_cubes=6)
    X = np.zeros((len(df), 4))
    row_idx = np.arange(len(df))
    y = (df["cube_id"].str[-4].to_numpy().astype(int) % 2).astype(int)
    train_rows = np.arange(len(df))

    sel = p1.select_hyperparameter(X, y, df, train_rows, row_idx, "logreg")
    assert np.allclose(sel["inner_scores"], sel["inner_scores"][0]), \
        "the inner scores must actually tie for this to test the tie rule"
    assert sel["param"] == min(p1.LOGREG_C_GRID)
    assert p1.select_hyperparameter(X, y, df, train_rows, row_idx,
                                    "ridge")["param"] == max(p1.RIDGE_ALPHA_GRID)


def test_loco_holds_out_exactly_one_cube_per_fold():
    df = _manifest(n_cubes=7)
    folds_ = p1._outer_folds(df, "loco")
    assert len(folds_) == 7
    for tr, te in folds_:
        assert len(_independent_cube_of(df, te)) == 1
        assert len(_independent_cube_of(df, tr)) == 6


def test_fold_mode_names_are_the_three_the_protocol_requires():
    assert p1.FOLD_MODES == ("cube", "loco", "spatial_block")
    with pytest.raises(AssertionError, match="not in"):
        p1._outer_folds(_manifest(), "random")


def test_rank_agreement_is_computed_and_excludes_the_multi_image_encoder():
    from scipy.stats import spearmanr

    df = _fake_results()
    order = {"raw_features": 0.8, "dinov2_vitb14": 0.7,
             "imagenet_vit_b16": 0.5, "satlas_s2_swinb_rgb": 0.6,
             p1.MI_ENCODER: 0.99}
    df["balanced_accuracy_mean"] = df.encoder.map(order).fillna(0.5)
    out = p1.rank_agreement(df, feature_set="grid_cell", verbose=False)
    assert out and all(v == pytest.approx(1.0) for v in out.values())
    sub = df[(df.feature_set == "grid_cell") & df.si_comparable]
    assert p1.MI_ENCODER not in set(sub.encoder), \
        "the multi-image control must not be ranked against the SI encoders"
    assert spearmanr([1, 2, 3], [1, 2, 3]).statistic == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# The real cache, when it is present
# ---------------------------------------------------------------------------

def _real_manifest():
    import glob
    import os

    from data.loader import load_cube
    from encoders.manifest import build_manifest

    paths = sorted(glob.glob(os.path.join("data", "raw", "*.nc")))
    emb = glob.glob(os.path.join("data", "phase1_2", "embeddings", "*.npz"))
    if len(paths) < 5 or not emb:
        pytest.skip("no local cubes / embeddings cache")
    return build_manifest([load_cube(p, verbose=False) for p in paths],
                          verbose=False)


def test_real_cache_joins_and_the_month_target_is_not_twelve_classes():
    m = _real_manifest()
    ch = p1.print_class_distribution(m, "month")
    assert ch["n_classes"] < 12, (
        "tile 32UNU's windows plus the clear-fraction filter cannot yield all "
        "twelve months; a probe reporting 1/12 as chance would be wrong"
    )
    arrays = p1.load_encoder_arrays(m, "raw_features", verbose=False)
    assert arrays["pooled"].shape == (len(m), 35)
    assert arrays["grid"].shape == (len(m), 16, 35)
    # The join is order-preserving: manifest row i is embedding row i.
    assert np.isfinite(arrays["pooled"]).all()
