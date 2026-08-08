"""P1, the appearance sanity probe: how accessible are month and season from
ONE frame's embedding?

WHAT THIS PROBE IS, AND WHAT IT IS NOT
--------------------------------------
It is CALIBRATION, not a finding. Month is confounded with appearance --
greenness, snow, sun angle, haze -- so a high score establishes only that
appearance is trivially present in the representation. That is precisely what
makes P2 (change) and P3 (forecastability) interpretable: an encoder that
cannot recover the month from a single frame has no appearance signal to
build on, so a later failure would be uninformative.

The result worth reporting here is therefore the OPPOSITE of success: an EO
foundation model that FAILS P1 would suggest aggressive appearance-invariance
training, and that is a finding. Success is a floor being met.

WHAT IS DELIBERATELY NOT DONE
-----------------------------
* No encoder is fine-tuned, loaded, or even imported. This module reads the
  cached ``.npz`` written by Phase 1.2 and nothing else. Frozen by
  construction, because no network is present.
* No split is defined here. Every fold comes from ``probes.cv.folds`` /
  ``probes.cv.leave_one_cube_out``, including the INNER folds used to tune
  the regularisation strength -- the inner loop calls the same generator on
  the training sub-manifest. Any number produced outside ``probes/cv.py``
  does not exist.
* Chance level is never hard-coded. This subset is not 12 months: the cube
  windows and the clear-fraction filter leave a realised, imbalanced subset
  of months and only THREE of the four seasons. ``chance_level`` derives the
  floor from the labels it is handed, and ``print_class_distribution``
  prints the realisation before anything is fitted.

THE FOUR FEATURE SETS, AND WHY THE GRID ONE IS PRIMARY
------------------------------------------------------
    pooled      the encoder's pooled vector, one row per RETAINED frame.
                264 rows against D up to 3840 (DINOv2). p >> n.
    grid_cell   the 4x4 patch grid flattened per CELL: each of the 16 cells
                becomes its own row and inherits its frame's label, so
                264 -> 4224 rows at D_grid <= 768. PRIMARY. It is the only
                feature set on this subset where the design matrix is not
                wider than it is tall.
    raw_pooled  ``raw_features`` pooled, D=35. The MANDATORY baseline: a
                table of foundation-model scores without the not-a-network
                row underneath it is not a table. Emitted under its own
                feature-set name so that filtering the CSV to one feature set
                can never silently drop the baseline; it is the SAME fitted
                model as (encoder=raw_features, feature_set=pooled), and
                ``assert_baseline_view_consistent`` proves the two agree
                digit-for-digit rather than leaving the duplication on trust.
    degenerate  [clear_frac, window_span_days] alone. NO embedding. NOT
                OPTIONAL: if cloud retention by itself decodes month above
                chance, then part of every other row in the table is
                retention rather than representation, and the whole table has
                to be read through that. Reported at both levels, frame
                (clear_frac) and cell (grid_clear_frac), so the primary
                grid_cell rows have a matched control.

Effective sample size is a separate question from row count. Exploding to
4224 cells fixes the FIT (p < n); it does not create 4224 independent
observations, because 16 cells of one frame share a sky and 13 frames of one
cube share a place. The honest uncertainty is the fold-to-fold spread over 20
cubes, which is why every metric here is reported with its spread and never
as a bare mean.

THE BASELINE SEES A BAND THE NETWORKS DO NOT
--------------------------------------------
``raw_features`` is computed over all four bands the cubes carry, B02/B03/B04
and **B8A**, and includes seven statistics of canonical NDVI. All four network
encoders are RGB-only -- ``imagenet_vit_b16`` and ``dinov2_vitb14`` by
construction, and both Satlas wrappers because the RGB variant was chosen over
the multi-spectral one (which expects nine bands these cubes do not have; see
docs/DECISIONS.md). NIR is where the seasonal vegetation signal mostly lives,
so the baseline is not merely a simpler model of the same evidence here: it is
a model of MORE evidence. Any row in which ``raw_features`` beats a frozen
encoder is therefore a statement about the input, not only about the
representation, and must be read that way. It is stated here, at the top,
because it is the first thing the results table shows.

THE MULTI-IMAGE ENCODER IS NOT IN THE SAME COLUMN
-------------------------------------------------
``satlas_s2_swinb_mi_rgb`` aggregates 8 RETAINED frames. Retained frames are
irregularly spaced, so its lookback spans a variable number of DAYS (measured
on this subset: min 0, median 55, max 105). Its embedding at time t therefore
summarises up to three months of history, and a single "month" label is
ill-defined for it in a way it is not for a single-image encoder. Its score is
reported, flagged ``si_comparable=False``, and additionally broken out by
``window_span_days`` tercile. It is never ranked against the single-image
encoders in the same column.

PROTOCOL
--------
Primary split ``mode="cube"``, k=5. Robustness: leave-one-cube-out and
``mode="spatial_block"``. All three are reported; they are expected to agree
in ORDERING even where they disagree in level, and ``rank_agreement``
computes that agreement rather than asserting it by eye.

Nested CV for the regularisation strength: the inner split runs on the OUTER
TRAINING FOLD ONLY, via ``probes.cv.folds`` on the training sub-manifest, and
the selected value is printed per outer fold. ``select_hyperparameter`` takes
no test argument at all -- the leakage is prevented by the signature, not by
discipline, which is what makes it testable.

Standardisation is fitted on train and applied to test, per fold, never on the
full array.

Metrics: balanced accuracy and macro-F1 (the classes are imbalanced), against
a most-frequent-class dummy floor, each with the spread across folds.
"""

from __future__ import annotations

import os
import warnings
from typing import NamedTuple, Sequence

import numpy as np

from data.paths import phase_dir
from encoders.pipeline import (
    assert_embeddings_complete,
    audit_embeddings,
    load_encoded,
)
from probes import cv

__all__ = [
    "PHASE", "TARGETS", "FOLD_MODES", "ESTIMATORS", "FEATURE_SETS",
    "SEASONS", "SEASON_OF_MONTH", "ENCODER_ORDER", "MI_ENCODER",
    "LOGREG_C_GRID", "RIDGE_ALPHA_GRID", "WSD_BINS",
    "FoldResult",
    "month_labels", "season_labels", "target_labels",
    "class_distribution", "chance_level", "print_class_distribution",
    "embeddings_dir", "load_encoder_arrays", "feature_matrix",
    "subset_arrays", "degenerate_arrays", "wsd_bin_labels",
    "make_estimator", "param_grid", "param_name",
    "select_hyperparameter", "evaluate_fold", "evaluate",
    "summarise", "run_p1", "rank_agreement", "figure1",
    "assert_baseline_view_consistent", "assert_results_complete",
    "results_path", "figure_path",
]

# Phase 1.4 is P1. Artefacts are phase-scoped through data/paths.py; nothing
# here ever writes to a hand-typed path.
PHASE = "phase1_4"

TARGETS = ("month", "season")
FOLD_MODES = ("cube", "loco", "spatial_block")
ESTIMATORS = ("logreg", "ridge")
FEATURE_SETS = ("pooled", "grid_cell", "raw_pooled", "degenerate")

# The 4-way meteorological season. Defined over all twelve months on purpose:
# the map is the definition, the REALISATION is measured. On this subset only
# three of the four appear, and chance_level finds that out from the labels.
SEASONS = ("DJF", "MAM", "JJA", "SON")
SEASON_OF_MONTH = {12: "DJF", 1: "DJF", 2: "DJF",
                   3: "MAM", 4: "MAM", 5: "MAM",
                   6: "JJA", 7: "JJA", 8: "JJA",
                   9: "SON", 10: "SON", 11: "SON"}

# Report order: the not-a-network baseline first, the multi-image control last,
# because it is the one that does not belong in the same column as the rest.
ENCODER_ORDER = ("raw_features", "imagenet_vit_b16", "dinov2_vitb14",
                 "satlas_s2_swinb_rgb", "satlas_s2_swinb_mi_rgb")
BASELINE_ENCODER = "raw_features"
MI_ENCODER = "satlas_s2_swinb_mi_rgb"

# Regularisation grids. Centred on STRONG regularisation because every feature
# set here is at least mildly p >> n and the pooled ones are severely so; the
# weak end is kept only so a selected value at the boundary is visible as such.
LOGREG_C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
RIDGE_ALPHA_GRID = (1e-1, 1.0, 1e1, 1e2, 1e3, 1e4)

# The multi-image lookback strata. Terciles of window_span_days, computed from
# the realised values rather than fixed at nominal day counts -- the nominal
# 35-day span for 8 frames at a 5-day revisit is wrong by a factor of three
# here (measured median 55, max 105).
WSD_BINS = ("short", "medium", "long")

_INNER_K = 3          # inner folds for the nested tuning loop
_GRID_CELLS = 16      # encoders.base.GRID_CELLS; asserted against the arrays


# ---------------------------------------------------------------------------
# Targets, and the chance level DERIVED from them
# ---------------------------------------------------------------------------

def month_labels(manifest) -> np.ndarray:
    """(n,) int calendar month, 1..12, from the manifest timestamp."""
    ts = np.asarray(manifest["timestamp"].to_numpy(), dtype="datetime64[ns]")
    months = (ts.astype("datetime64[M]").astype(int) % 12 + 1).astype(int)
    assert months.shape == (len(manifest),), months.shape
    assert months.min() >= 1 and months.max() <= 12
    return months


def season_labels(manifest) -> np.ndarray:
    """(n,) str meteorological season, one of SEASONS."""
    months = month_labels(manifest)
    out = np.array([SEASON_OF_MONTH[int(m)] for m in months], dtype="<U3")
    assert out.shape == (len(manifest),), out.shape
    assert set(out.tolist()) <= set(SEASONS)
    return out


def target_labels(manifest, target: str) -> np.ndarray:
    assert target in TARGETS, f"target {target!r} not in {TARGETS}"
    return month_labels(manifest) if target == "month" else season_labels(manifest)


def class_distribution(y) -> dict:
    """{class: count} over the REALISED labels, in class order."""
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    return {c.item() if hasattr(c, "item") else c: int(n)
            for c, n in zip(classes, counts)}


def chance_level(y) -> dict:
    """The floor, DERIVED from the realised class distribution.

    Never 1/12, and on this subset never 1/4 either: the cube windows plus the
    clear-fraction filter decide which months and seasons exist at all, and
    that realisation is an empirical property of the data, not a constant.

    Returned, for a most-frequent-class predictor:

        n_classes            K, the number of classes actually present
        balanced_accuracy    1/K exactly -- a constant predictor scores 1 on
                             one class and 0 on the other K-1
        macro_f1             (2 p / (1 + p)) / K, where p is the majority
                             prevalence. Exact, and the empirical dummy is
                             asserted against it in the tests.
        majority_class, majority_frac
    """
    y = np.asarray(y)
    assert y.size, "chance level of an empty label array"
    dist = class_distribution(y)
    K = len(dist)
    majority = max(dist, key=dist.get)
    p = dist[majority] / y.size
    return {
        "n_classes": K,
        "counts": dist,
        "balanced_accuracy": 1.0 / K,
        "macro_f1": (2.0 * p / (1.0 + p)) / K,
        "majority_class": majority,
        "majority_frac": p,
    }


def print_class_distribution(manifest, target: str) -> dict:
    """Print the realisation, then return the chance level derived from it.

    Called before anything is fitted. The point is that the reader sees which
    classes exist on this subset -- and that the floor printed underneath was
    computed from that line, not looked up.
    """
    y = target_labels(manifest, target)
    ch = chance_level(y)
    n = y.size
    print(f"[p1] target {target!r}: {n} rows, {ch['n_classes']} REALISED classes")
    for c, k in ch["counts"].items():
        bar = "#" * int(round(40 * k / max(ch["counts"].values())))
        print(f"[p1]   {str(c):<5} {k:>4}  ({k / n:5.1%})  {bar}")
    if target == "month":
        missing = sorted(set(range(1, 13)) - set(ch["counts"]))
        print(f"[p1]   months NOT realised: {missing} -- chance is 1/"
              f"{ch['n_classes']}, NOT 1/12")
    else:
        missing = sorted(set(SEASONS) - set(ch["counts"]))
        print(f"[p1]   seasons NOT realised: {missing} -- chance is 1/"
              f"{ch['n_classes']}, NOT 1/4")
    print(f"[p1]   most-frequent-class floor: balanced accuracy "
          f"{ch['balanced_accuracy']:.4f} = 1/{ch['n_classes']}, "
          f"macro-F1 {ch['macro_f1']:.4f} "
          f"(majority {ch['majority_class']!r} at {ch['majority_frac']:.1%})")
    return ch


# ---------------------------------------------------------------------------
# Features, read from the Phase 1.2 cache and joined through probes.cv
# ---------------------------------------------------------------------------

def embeddings_dir() -> str:
    return phase_dir("phase1_2", "embeddings")


def load_encoder_arrays(manifest, encoder: str, emb_dir: str | None = None,
                        verbose: bool = True) -> dict:
    """Every cached array for ONE encoder, aligned to manifest row order.

    The join is ``probes.cv.join_embeddings`` per cube -- the contract
    (cube_id, original_axis_index) == (cube, kept_idx) is asserted there, not
    re-implemented here -- and the per-cube blocks are then permuted into
    manifest row order so that a fold's row positions index this array
    directly.

    Returns pooled (n, D), grid (n, 16, D_grid) float32, grid_clear_frac
    (n, 16), clear_frac (n,), window_span_days (n,).
    """
    emb_dir = emb_dir or embeddings_dir()
    cubes = sorted(manifest["cube_id"].unique().tolist())
    pooled, grid, gcf, cf, wsd, rows = [], [], [], [], [], []
    for cube in cubes:
        path = os.path.join(emb_dir, f"{os.path.splitext(cube)[0]}__{encoder}.npz")
        assert os.path.exists(path), (
            f"no cached embedding for ({cube!r}, {encoder!r}) at {path}. Run "
            "audit_embeddings/assert_embeddings_complete before this call -- a "
            "per-encoder comparison over a cache with holes is a comparison "
            "over DIFFERENT cubes."
        )
        out = cv.join_embeddings(manifest, load_encoded(path), verbose=False)
        rows.append(out["manifest_idx"])
        pooled.append(out["embeddings"])
        grid.append(np.asarray(out["grid"], dtype=np.float32))
        gcf.append(out["grid_clear_frac"])
        cf.append(out["clear_frac"])
        wsd.append(out["window_span_days"])

    rows = np.concatenate(rows)
    order = np.argsort(rows, kind="stable")
    assert rows[order].tolist() == list(range(len(manifest))), (
        f"{encoder}: the joined rows are not a permutation of the manifest's "
        f"{len(manifest)} rows ({rows.size} joined). A cube is missing or "
        "duplicated in the cache."
    )
    out = {
        "encoder": encoder,
        "pooled": np.concatenate(pooled)[order].astype(np.float32),
        "grid": np.concatenate(grid)[order],
        "grid_clear_frac": np.concatenate(gcf)[order].astype(np.float32),
        "clear_frac": np.concatenate(cf)[order].astype(np.float64),
        "window_span_days": np.concatenate(wsd)[order].astype(np.float64),
    }
    n = len(manifest)
    assert out["pooled"].shape[0] == n and out["pooled"].ndim == 2
    assert out["grid"].shape[:2] == (n, _GRID_CELLS), out["grid"].shape
    assert out["grid_clear_frac"].shape == (n, _GRID_CELLS)
    assert out["clear_frac"].shape == out["window_span_days"].shape == (n,)
    assert np.isfinite(out["pooled"]).all() and np.isfinite(out["grid"]).all()
    if verbose:
        print(f"[p1] {encoder:<24} pooled {out['pooled'].shape} | "
              f"grid {out['grid'].shape} | clear_frac {out['clear_frac'].shape} | "
              f"window_span_days {out['window_span_days'].shape} "
              f"(min {out['window_span_days'].min():.0f} "
              f"median {np.median(out['window_span_days']):.0f} "
              f"max {out['window_span_days'].max():.0f} d)")
    return out


def feature_matrix(arrays: dict, feature_set: str, level: str = "auto") -> tuple:
    """(X, row_idx) for one feature set. ``row_idx[i]`` is the MANIFEST row
    that X row i belongs to -- one-to-one for frame-level sets, 16-to-one for
    cell-level ones.

    Carrying row_idx rather than assuming alignment is what lets a fold defined
    on manifest rows select cell rows without a second, hand-rolled index.
    """
    n = arrays["pooled"].shape[0]
    frame_rows = np.arange(n)
    cell_rows = np.repeat(frame_rows, _GRID_CELLS)

    if feature_set in ("pooled", "raw_pooled"):
        X, row_idx = arrays["pooled"], frame_rows
    elif feature_set == "grid_cell":
        g = arrays["grid"]
        X, row_idx = g.reshape(n * _GRID_CELLS, g.shape[-1]), cell_rows
    elif feature_set == "degenerate":
        assert level in ("frame", "cell"), (
            f"the degenerate control needs an explicit level, got {level!r}"
        )
        if level == "frame":
            X = np.column_stack([arrays["clear_frac"],
                                 arrays["window_span_days"]])
            row_idx = frame_rows
        else:
            X = np.column_stack([arrays["grid_clear_frac"].reshape(-1),
                                 np.repeat(arrays["window_span_days"],
                                           _GRID_CELLS)])
            row_idx = cell_rows
    else:
        raise AssertionError(f"unknown feature set {feature_set!r}")

    X = np.ascontiguousarray(X, dtype=np.float64)
    assert X.ndim == 2 and X.shape[0] == row_idx.shape[0], (X.shape, row_idx.shape)
    assert np.isfinite(X).all(), f"{feature_set}: non-finite feature"
    return X, row_idx


def _rows_for(row_idx: np.ndarray, manifest_positions: np.ndarray) -> np.ndarray:
    """X-row positions belonging to a set of manifest rows."""
    keep = np.flatnonzero(np.isin(row_idx, manifest_positions))
    assert keep.size, "a fold side selected zero feature rows"
    return keep


# ---------------------------------------------------------------------------
# Estimators, the nested tuning loop, and one fold
# ---------------------------------------------------------------------------

def make_estimator(name: str, param: float):
    """One estimator at one regularisation strength. No RNG, no warm start.

    ``logreg`` is multinomial by construction: sklearn's lbfgs solver fits a
    single multinomial model whenever there are more than two classes, and the
    explicit ``multi_class`` argument that used to say so is deprecated, so
    passing it would emit a FutureWarning while changing nothing.

    Neither estimator is class-weighted. The metric IS balanced accuracy, so
    weighting would move the estimator toward the metric and inflate exactly
    the number this probe exists to establish a floor for. The unweighted fit
    is the conservative choice: if month is recoverable anyway, the conclusion
    survives the imbalance rather than being rescued from it.
    """
    from sklearn.linear_model import LogisticRegression, RidgeClassifier

    assert name in ESTIMATORS, f"estimator {name!r} not in {ESTIMATORS}"
    if name == "logreg":
        return LogisticRegression(C=float(param), max_iter=2000, tol=1e-4)
    return RidgeClassifier(alpha=float(param))


def param_grid(estimator: str) -> tuple:
    return LOGREG_C_GRID if estimator == "logreg" else RIDGE_ALPHA_GRID


def param_name(estimator: str) -> str:
    return "C" if estimator == "logreg" else "alpha"


def _score(y_true, y_pred, macro_f1: bool = False) -> float:
    """balanced accuracy (or macro-F1) over the classes PRESENT IN y_true.

    Scoring against the global class list instead would charge a fold zero F1
    for a month it was never given a chance to predict, which deflates the
    number without measuring anything.

    A cube-grouped test fold routinely omits a month that train contains, so
    the classifier predicts a class y_true does not hold. sklearn warns and
    then does the right thing -- it scores over y_true's classes, and the stray
    prediction still costs the true class its recall. At LOCO that is one
    warning per fold per grid point, thousands per run, which would bury the
    output this probe exists to print. Suppressed by exact message, here only,
    never globally.
    """
    from sklearn.metrics import balanced_accuracy_score, f1_score

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="y_pred contains classes not in y_true")
        if macro_f1:
            return float(f1_score(y_true, y_pred, labels=np.unique(y_true),
                                  average="macro", zero_division=0))
        return float(balanced_accuracy_score(y_true, y_pred))


def _fit_predict(estimator: str, param: float, X_tr, y_tr, X_te):
    """Standardise on TRAIN, fit on TRAIN, predict TEST. Never the reverse."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X_tr)
    Z_tr = scaler.transform(X_tr)
    Z_te = scaler.transform(X_te)
    # A constant-in-train column standardises to exactly 0 (sklearn sets its
    # scale to 1), which is harmless; a non-finite would not be.
    assert np.isfinite(Z_tr).all() and np.isfinite(Z_te).all()
    clf = make_estimator(estimator, param).fit(Z_tr, y_tr)
    return clf.predict(Z_te)


def select_hyperparameter(X_tr, y_tr, manifest, train_rows, row_idx_tr,
                          estimator: str, inner_k: int = _INNER_K,
                          verbose: bool = False) -> dict:
    """Tune the regularisation strength on the TRAINING fold only.

    THE SIGNATURE IS THE GUARANTEE. There is no test argument to leak: this
    function is handed the training design matrix, the training labels, and
    the manifest rows the training fold occupies, and nothing else. The
    tests exercise that by poisoning the test fold and asserting the selection
    is bit-identical, and by inspecting this signature for any parameter that
    could carry test data.

    ``row_idx_tr`` is the manifest row of each row OF X_tr -- the training
    slice of the full row index, not the full array. Handing in the full one
    would silently index past the end of X_tr for cell-level features, which
    is exactly the sort of off-by-a-subset that turns into a wrong number
    rather than an error when the arrays happen to be the same length.

    The inner split is ``probes.cv.folds`` on the TRAINING sub-manifest, mode
    "cube": cube grouping is the strictest rule any outer mode here obeys, so
    an inner fold can never put a training cube on both sides either.
    """
    grid = param_grid(estimator)
    sub = manifest.iloc[train_rows]
    n_cubes = int(sub["cube_id"].nunique())
    k = min(inner_k, n_cubes)
    assert k >= 2, (
        f"the inner tuning loop needs >= 2 training cubes, the outer training "
        f"fold has {n_cubes}. Reduce the outer k or use a mode that leaves "
        "more cubes in train."
    )
    inner = list(cv.folds(sub, "cube", k=k, verbose=False))
    assert inner, "the inner split produced no folds"
    row_idx_tr = np.asarray(row_idx_tr, dtype=int)
    assert row_idx_tr.shape[0] == X_tr.shape[0], (
        f"row_idx_tr {row_idx_tr.shape} does not describe X_tr {X_tr.shape}"
    )

    scores = np.zeros((len(grid), len(inner)))
    for j, (itr, ite) in enumerate(inner):
        # Inner positions index the SUB-manifest; map back to manifest rows,
        # then to rows OF X_tr. Both hops are explicit so neither can drift.
        tr_rows, te_rows = train_rows[itr], train_rows[ite]
        assert not np.intersect1d(tr_rows, te_rows).size
        a, b = _rows_for(row_idx_tr, tr_rows), _rows_for(row_idx_tr, te_rows)
        for i, p in enumerate(grid):
            pred = _fit_predict(estimator, p, X_tr[a], y_tr[a], X_tr[b])
            scores[i, j] = _score(y_tr[b], pred)

    mean = scores.mean(axis=1)
    # Ties break toward STRONGER regularisation. Both grids are ascending, and
    # stronger means SMALLER C but LARGER alpha, so logreg takes the first
    # argmax and ridge the last. A tie means the inner data cannot tell the two
    # apart, and of two indistinguishable models the more regularised one is
    # the one that will generalise -- and, just as importantly, the tie is
    # broken by a stated rule rather than by numpy's scan order.
    if estimator == "logreg":
        best = int(np.argmax(mean))                       # first maximum
    else:
        best = len(grid) - 1 - int(np.argmax(mean[::-1]))  # last maximum
    chosen = float(grid[best])
    if verbose:
        print(f"[p1]     inner {k}-fold on {n_cubes} training cubes: "
              f"{param_name(estimator)}={chosen:g} "
              f"(inner balanced acc {mean[best]:.3f})")
    return {"param": chosen, "inner_scores": mean, "grid": tuple(grid),
            "n_inner_folds": len(inner), "inner_best_score": float(mean[best]),
            "at_grid_edge": best in (0, len(grid) - 1)}


class FoldResult(NamedTuple):
    """One outer fold. ``selected`` is what the inner loop chose, printed per
    fold because a tuning loop nobody can see is a tuning loop nobody can
    check."""

    fold: int
    n_train: int
    n_test: int
    n_train_cubes: int
    n_test_cubes: int
    n_classes_test: int
    selected: float
    at_grid_edge: bool
    balanced_accuracy: float
    macro_f1: float
    dummy_balanced_accuracy: float
    dummy_macro_f1: float
    log: str


def evaluate_fold(X, y, row_idx, manifest, train_rows, test_rows,
                  estimator: str, fold: int = 0, inner_k: int = _INNER_K,
                  verbose: bool = True) -> FoldResult:
    """One outer fold, end to end: tune on train, fit on train, score on test.

    Public because it is the seam the leakage test needs: it is called twice
    with identical training arguments and DIFFERENT (poisoned) test arguments,
    and the selected regularisation strength must be identical.
    """
    from sklearn.dummy import DummyClassifier

    train_rows = np.asarray(train_rows, dtype=int)
    test_rows = np.asarray(test_rows, dtype=int)
    assert not np.intersect1d(train_rows, test_rows).size, (
        f"fold {fold}: manifest rows appear on both sides"
    )
    a, b = _rows_for(row_idx, train_rows), _rows_for(row_idx, test_rows)
    assert not np.intersect1d(a, b).size, f"fold {fold}: feature rows overlap"
    X_tr, y_tr, X_te, y_te = X[a], y[a], X[b], y[b]

    sel = select_hyperparameter(X_tr, y_tr, manifest, train_rows, row_idx[a],
                                estimator, inner_k=inner_k, verbose=False)
    pred = _fit_predict(estimator, sel["param"], X_tr, y_tr, X_te)

    # Metrics over the classes present in THIS test fold; see _score.
    labels = np.unique(y_te)
    ba, f1 = _score(y_te, pred), _score(y_te, pred, macro_f1=True)
    dpred = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr).predict(X_te)
    dba, df1 = _score(y_te, dpred), _score(y_te, dpred, macro_f1=True)

    cubes = manifest["cube_id"].to_numpy()
    edge = " [GRID EDGE]" if sel["at_grid_edge"] else ""
    log = (f"[p1]   fold {fold + 1}: train {a.size} rows / "
           f"{len(set(cubes[train_rows]))} cubes, test {b.size} rows / "
           f"{len(set(cubes[test_rows]))} cubes, {labels.size} classes in test | "
           f"{param_name(estimator)}={sel['param']:g}{edge} | "
           f"bal-acc {ba:.3f} (dummy {dba:.3f})  macro-F1 {f1:.3f} "
           f"(dummy {df1:.3f})")
    if verbose:
        print(log)
    return FoldResult(
        fold=fold, n_train=int(a.size), n_test=int(b.size),
        n_train_cubes=len(set(cubes[train_rows])),
        n_test_cubes=len(set(cubes[test_rows])),
        n_classes_test=int(labels.size),
        selected=float(sel["param"]), at_grid_edge=bool(sel["at_grid_edge"]),
        balanced_accuracy=float(ba), macro_f1=float(f1),
        dummy_balanced_accuracy=float(dba), dummy_macro_f1=float(df1),
        log=log,
    )


def _outer_folds(manifest, mode: str, k: int = 5, verbose: bool = False) -> list:
    """The outer folds for one mode, from probes.cv and nowhere else."""
    assert mode in FOLD_MODES, f"fold mode {mode!r} not in {FOLD_MODES}"
    if mode == "loco":
        return list(cv.leave_one_cube_out(manifest, verbose=verbose))
    return list(cv.folds(manifest, mode, k=k, verbose=verbose))


def evaluate(X, y, row_idx, manifest, mode: str, estimator: str, k: int = 5,
             inner_k: int = _INNER_K, n_jobs: int = 1,
             verbose: bool = True) -> list:
    """Every outer fold of one mode. Deterministic; ``n_jobs`` only changes
    wall-clock, never a number -- lbfgs, the ridge solve and the scaler are all
    exact, and the folds come from a generator with no RNG in it.

    Arrays are passed to the workers as ARGUMENTS rather than captured in a
    closure, so joblib memory-maps the large ones instead of pickling a 26 MB
    design matrix once per fold.
    """
    outer = _outer_folds(manifest, mode, k=k)
    call = dict(estimator=estimator, inner_k=inner_k, verbose=False)

    if n_jobs == 1:
        results = [evaluate_fold(X, y, row_idx, manifest, tr, te, fold=i, **call)
                   for i, (tr, te) in enumerate(outer)]
    else:
        from joblib import Parallel, delayed, parallel_config
        with parallel_config(backend="loky", inner_max_num_threads=1):
            results = Parallel(n_jobs=n_jobs)(
                delayed(evaluate_fold)(X, y, row_idx, manifest, tr, te,
                                       fold=i, **call)
                for i, (tr, te) in enumerate(outer))
    if verbose:
        for r in results:
            print(r.log)
    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def summarise(results: Sequence[FoldResult]) -> dict:
    """Mean AND spread across folds. A mean without the spread is not
    reportable at 20 cubes: fold-to-fold variation IS the uncertainty here,
    and it is routinely larger than the gap between two encoders."""
    assert results, "nothing to summarise"
    out = {"n_folds": len(results)}
    for field in ("balanced_accuracy", "macro_f1",
                  "dummy_balanced_accuracy", "dummy_macro_f1"):
        v = np.array([getattr(r, field) for r in results], dtype=float)
        out[f"{field}_mean"] = float(v.mean())
        out[f"{field}_std"] = float(v.std(ddof=1)) if v.size > 1 else 0.0
        out[f"{field}_min"] = float(v.min())
        out[f"{field}_max"] = float(v.max())
        out[f"{field}_spread"] = float(v.max() - v.min())
        out[f"per_fold_{field}"] = ";".join(f"{x:.4f}" for x in v)
    out["selected_params"] = ";".join(f"{r.selected:g}" for r in results)
    out["n_at_grid_edge"] = int(sum(r.at_grid_edge for r in results))
    out["n_rows_test_total"] = int(sum(r.n_test for r in results))
    return out


def _wsd_bin_edges(wsd: np.ndarray) -> np.ndarray:
    """Terciles of the REALISED lookback. Nominal day counts are wrong here."""
    return np.quantile(np.asarray(wsd, dtype=float), [1 / 3, 2 / 3])


def wsd_bin_labels(wsd: np.ndarray) -> np.ndarray:
    edges = _wsd_bin_edges(wsd)
    idx = np.digitize(np.asarray(wsd, dtype=float), edges, right=False)
    return np.array([WSD_BINS[min(int(i), 2)] for i in idx])


def subset_arrays(arrays: dict, rows: np.ndarray) -> dict:
    """Slice every per-frame array of one encoder to a subset of manifest rows.

    Used for the multi-image lookback strata. Re-joining a SUBSET manifest
    through ``cv.join_embeddings`` is not an option and must not be attempted:
    that function asserts the full contract original_axis_index == kept_idx over
    a whole cube, so handing it a manifest with some of a cube's frames removed
    fails by design. Slice the already-joined arrays instead.
    """
    rows = np.asarray(rows, dtype=int)
    out = {"encoder": arrays["encoder"]}
    for key in ("pooled", "grid", "grid_clear_frac", "clear_frac",
                "window_span_days"):
        out[key] = arrays[key][rows]
    assert out["pooled"].shape[0] == rows.size
    return out


def degenerate_arrays(arrays_by_encoder: dict, verbose: bool = True) -> dict:
    """The control's inputs: retention only, no embedding anywhere in it.

    ``clear_frac`` and ``grid_clear_frac`` come from the mask and are identical
    across encoders (asserted here rather than assumed -- if they ever diverge,
    the caches were written under different mask definitions).
    ``window_span_days`` is taken from the MULTI-IMAGE encoder, the only one
    where it is not identically zero, which makes this the strongest form of
    the control rather than the convenient one.
    """
    encs = list(arrays_by_encoder)
    ref = arrays_by_encoder[encs[0]]
    for e in encs[1:]:
        a = arrays_by_encoder[e]
        np.testing.assert_allclose(a["clear_frac"], ref["clear_frac"], atol=1e-9,
                                   err_msg=f"clear_frac differs between {encs[0]} "
                                           f"and {e}: different mask definitions")
        np.testing.assert_allclose(a["grid_clear_frac"], ref["grid_clear_frac"],
                                   atol=1e-6, err_msg="grid_clear_frac differs "
                                                      f"between {encs[0]} and {e}")
    src = MI_ENCODER if MI_ENCODER in arrays_by_encoder else encs[0]
    wsd = arrays_by_encoder[src]["window_span_days"]
    if verbose:
        print(f"[p1] degenerate control: clear_frac (identical across "
              f"{len(encs)} encoders) + window_span_days from {src} "
              f"(min {wsd.min():.0f} median {np.median(wsd):.0f} "
              f"max {wsd.max():.0f} d)")
    if src != MI_ENCODER:
        print(f"[p1] WARNING: {MI_ENCODER} absent, so window_span_days is "
              "identically 0 and the control degenerates to clear_frac alone")
    return {"encoder": f"none(clear_frac + window_span_days from {src})",
            "pooled": ref["pooled"], "grid": ref["grid"],
            "clear_frac": ref["clear_frac"],
            "grid_clear_frac": ref["grid_clear_frac"],
            "window_span_days": wsd}


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def _rows_dict(target, mode, estimator, encoder, feature_set, level, wsd_bin,
               X, manifest, results, chance, comparable, note):
    s = summarise(results)
    return {
        "target": target, "fold_mode": mode, "estimator": estimator,
        "encoder": encoder, "feature_set": feature_set, "feature_level": level,
        "wsd_bin": wsd_bin,
        "n_feature_rows": int(X.shape[0]), "n_manifest_rows": int(len(manifest)),
        "D": int(X.shape[1]),
        "n_classes_realised": int(chance["n_classes"]),
        "chance_balanced_accuracy": float(chance["balanced_accuracy"]),
        "chance_macro_f1": float(chance["macro_f1"]),
        "majority_class": str(chance["majority_class"]),
        "majority_frac": float(chance["majority_frac"]),
        "si_comparable": bool(comparable),
        "note": note,
        **s,
    }


def run_p1(manifest, encoders: Sequence[str] = ENCODER_ORDER,
           targets: Sequence[str] = TARGETS,
           fold_modes: Sequence[str] = FOLD_MODES,
           estimators: Sequence[str] = ESTIMATORS,
           emb_dir: str | None = None, k: int = 5, inner_k: int = _INNER_K,
           n_jobs: int = 1, verbose: bool = True):
    """The whole table. Returns a pandas DataFrame; writes nothing.

    Coverage per (target, fold_mode, estimator): every encoder at ``pooled``
    and ``grid_cell``, the ``raw_pooled`` baseline view, and the degenerate
    control at both levels. The multi-image encoder additionally gets one row
    per ``window_span_days`` tercile, and every MI row carries
    ``si_comparable=False``.
    """
    import pandas as pd

    emb_dir = emb_dir or embeddings_dir()
    encoders = tuple(encoders)
    assert BASELINE_ENCODER in encoders, (
        f"{BASELINE_ENCODER!r} is the mandatory baseline and is missing from "
        f"{encoders}. A table of foundation-model scores with no not-a-network "
        "row underneath it is not a table."
    )

    audit = audit_embeddings(emb_dir, cube_ids=set(manifest["cube_id"]),
                             verbose=verbose)
    assert_embeddings_complete(audit, set(manifest["cube_id"]), encoders)

    if verbose:
        print()
    arrays = {e: load_encoder_arrays(manifest, e, emb_dir, verbose=verbose)
              for e in encoders}
    arrays["__degenerate__"] = degenerate_arrays(arrays, verbose=verbose)
    wsd = arrays["__degenerate__"]["window_span_days"]
    bins = wsd_bin_labels(wsd)
    if verbose:
        edges = _wsd_bin_edges(wsd)
        print(f"[p1] window_span_days terciles at {edges[0]:.0f} / "
              f"{edges[1]:.0f} days -> "
              f"{ {b: int((bins == b).sum()) for b in WSD_BINS} } frames")

    rows = []
    for target in targets:
        if verbose:
            print("\n" + "=" * 78)
        y_frame = target_labels(manifest, target)
        chance = print_class_distribution(manifest, target) if verbose \
            else chance_level(y_frame)

        for mode in fold_modes:
            for estimator in estimators:
                head = f"{target} | {mode} | {estimator}"
                if verbose:
                    print(f"\n[p1] ---- {head} " + "-" * max(0, 60 - len(head)))

                jobs = []
                for enc in encoders:
                    jobs.append((enc, enc, "pooled", "frame"))
                    jobs.append((enc, enc, "grid_cell", "cell"))
                jobs.append((BASELINE_ENCODER, BASELINE_ENCODER,
                             "raw_pooled", "frame"))
                jobs.append(("__degenerate__", "none", "degenerate", "frame"))
                jobs.append(("__degenerate__", "none", "degenerate", "cell"))

                for src, enc_label, fs, level in jobs:
                    X, row_idx = feature_matrix(arrays[src], fs, level=level)
                    y = y_frame[row_idx]
                    res = evaluate(X, y, row_idx, manifest, mode, estimator,
                                   k=k, inner_k=inner_k, n_jobs=n_jobs,
                                   verbose=False)
                    comparable, note = src != MI_ENCODER, ""
                    if src == MI_ENCODER:
                        note = ("aggregates 8 retained frames over a variable "
                                "lookback (0-105 d); NOT directly comparable "
                                "to the single-image encoders")
                    elif fs == "degenerate":
                        note = ("retention only: "
                                + ("[clear_frac, window_span_days]" if level == "frame"
                                   else "[grid_clear_frac, window_span_days]")
                                + f", NO embedding; window_span_days from {MI_ENCODER}")
                    r = _rows_dict(target, mode, estimator, enc_label, fs, level,
                                   "all", X, manifest, res, chance, comparable,
                                   note)
                    rows.append(r)
                    if verbose:
                        print(f"[p1]   {enc_label:<24} {fs:<11} {level:<5} "
                              f"n={X.shape[0]:<5} D={X.shape[1]:<5} "
                              f"bal-acc {r['balanced_accuracy_mean']:.3f} "
                              f"+/-{r['balanced_accuracy_std']:.3f} "
                              f"(spread {r['balanced_accuracy_spread']:.3f}) "
                              f"macro-F1 {r['macro_f1_mean']:.3f} "
                              f"| dummy {r['dummy_balanced_accuracy_mean']:.3f} "
                              f"| {param_name(estimator)} "
                              f"[{r['selected_params']}]"
                              + ("  <- NOT SI-COMPARABLE" if not comparable else ""))

                # The MI encoder conditioned on its own lookback. Reported
                # BESIDE its unconditioned row, never instead of it: a single
                # "month" label means something different at a 0-day lookback
                # and at a 105-day one, and averaging over that difference is
                # what the caveat forbids.
                if MI_ENCODER in encoders:
                    for wbin in WSD_BINS:
                        keep = np.flatnonzero(bins == wbin)
                        sub = manifest.iloc[keep].reset_index(drop=True)
                        n_cubes = int(sub["cube_id"].nunique())
                        if n_cubes < 4:
                            if verbose:
                                print(f"[p1]   {MI_ENCODER} wsd={wbin}: SKIPPED, "
                                      f"only {n_cubes} cubes in this tercile -- "
                                      "a fold would hold out almost everything")
                            continue
                        a = subset_arrays(arrays[MI_ENCODER], keep)
                        X, row_idx = feature_matrix(a, "pooled")
                        ysub = target_labels(sub, target)[row_idx]
                        ch = chance_level(ysub)
                        res = evaluate(X, ysub, row_idx, sub, mode, estimator,
                                       k=min(k, n_cubes), inner_k=inner_k,
                                       n_jobs=n_jobs, verbose=False)
                        r = _rows_dict(target, mode, estimator, MI_ENCODER,
                                       "pooled", "frame", wbin, X, sub, res, ch,
                                       False,
                                       f"window_span_days tercile {wbin!r} = "
                                       f"{a['window_span_days'].min():.0f}-"
                                       f"{a['window_span_days'].max():.0f} days, "
                                       f"{n_cubes} cubes; chance recomputed on "
                                       "this tercile's own class distribution")
                        rows.append(r)
                        if verbose:
                            print(f"[p1]   {MI_ENCODER:<24} pooled      "
                                  f"wsd={wbin:<7} n={X.shape[0]:<5} "
                                  f"bal-acc {r['balanced_accuracy_mean']:.3f} "
                                  f"+/-{r['balanced_accuracy_std']:.3f} "
                                  f"(chance {ch['balanced_accuracy']:.3f} = 1/"
                                  f"{ch['n_classes']})")

    df = pd.DataFrame(rows)
    assert len(df), "run_p1 produced no rows"
    return df


# ---------------------------------------------------------------------------
# Table-level invariants
# ---------------------------------------------------------------------------

def assert_baseline_view_consistent(df) -> None:
    """``raw_pooled`` is a VIEW of (raw_features, pooled), and must agree.

    The duplication exists so that a reader filtering the CSV to a single
    feature set still has the not-a-network floor in front of them. Duplicated
    numbers that nobody checks are how two tables start disagreeing, so the
    identity is asserted rather than assumed.
    """
    key = ["target", "fold_mode", "estimator"]
    view = df[df.feature_set == "raw_pooled"]
    base = df[(df.feature_set == "pooled") & (df.encoder == BASELINE_ENCODER)]
    assert len(view) and len(view) == len(base), (
        f"raw_pooled has {len(view)} rows, (raw_features, pooled) has "
        f"{len(base)}; the baseline view is not one-to-one"
    )
    m = view.merge(base, on=key, suffixes=("_view", "_base"))
    assert len(m) == len(view), "baseline view does not join one-to-one"
    for col in ("balanced_accuracy_mean", "macro_f1_mean",
                "balanced_accuracy_spread", "selected_params"):
        a, b = m[f"{col}_view"].to_numpy(), m[f"{col}_base"].to_numpy()
        assert (a == b).all(), (
            f"raw_pooled disagrees with (raw_features, pooled) on {col}: "
            f"{a[:3]} vs {b[:3]}. They are the same fitted model; a "
            "disagreement means one of them was computed on different rows."
        )


def assert_results_complete(df, encoders: Sequence[str] = ENCODER_ORDER,
                            fold_modes: Sequence[str] = FOLD_MODES,
                            targets: Sequence[str] = TARGETS,
                            estimators: Sequence[str] = ESTIMATORS) -> None:
    """Every cell the exit test asks for is present, and so is the baseline.

    Checked here rather than only in the test suite, so a truncated run cannot
    be written to disk and read later as if it were the whole table.
    """
    for target in targets:
        for mode in fold_modes:
            for est in estimators:
                sub = df[(df.target == target) & (df.fold_mode == mode)
                         & (df.estimator == est)]
                assert len(sub), f"no rows for {target}/{mode}/{est}"
                for fs in ("pooled", "grid_cell"):
                    have = set(sub[(sub.feature_set == fs)
                                   & (sub.wsd_bin == "all")].encoder)
                    missing = set(encoders) - have
                    assert not missing, (
                        f"{target}/{mode}/{est}/{fs}: missing encoders "
                        f"{sorted(missing)}"
                    )
                    assert BASELINE_ENCODER in have, (
                        f"{target}/{mode}/{est}/{fs}: no {BASELINE_ENCODER} "
                        "row. Every feature set must carry the baseline."
                    )
                assert (sub.feature_set == "raw_pooled").any(), (
                    f"{target}/{mode}/{est}: the mandatory raw_features "
                    "baseline row is missing"
                )
                assert (sub.feature_set == "degenerate").any(), (
                    f"{target}/{mode}/{est}: the degenerate control row is "
                    "missing. It is not optional -- without it, part of every "
                    "other score here could be cloud retention."
                )
    mi = df[df.encoder == MI_ENCODER]
    assert len(mi) and not mi.si_comparable.any(), (
        f"{MI_ENCODER} rows must all carry si_comparable=False -- its embedding "
        "summarises up to three months of history, so a single 'month' label "
        "does not mean for it what it means for a single-image encoder"
    )
    assert df[df.encoder != MI_ENCODER].si_comparable.all(), (
        "a single-image encoder row is flagged si_comparable=False"
    )
    assert (mi.wsd_bin != "all").any(), (
        f"{MI_ENCODER} is reported unconditionally but never conditioned on "
        "window_span_days; its lookback spans 0-105 days and the caveat "
        "requires the breakdown"
    )
    assert_baseline_view_consistent(df)
    print(f"[p1] results table COMPLETE: {len(df)} rows, "
          f"{df.encoder.nunique()} encoder labels x {df.feature_set.nunique()} "
          f"feature sets x {df.fold_mode.nunique()} fold modes x "
          f"{df.target.nunique()} targets x {df.estimator.nunique()} estimators")


def rank_agreement(df, target: str = "month", estimator: str = "logreg",
                   feature_set: str = "grid_cell", verbose: bool = True) -> dict:
    """Spearman rank correlation of the encoder ORDERING between fold modes.

    The protocol requires the three modes to agree in ordering even where they
    disagree in level. This computes that agreement instead of asserting it by
    eye. The multi-image encoder is excluded: it is not in the same column.
    """
    from scipy.stats import spearmanr

    sub = df[(df.target == target) & (df.estimator == estimator)
             & (df.feature_set == feature_set) & (df.wsd_bin == "all")
             & (df.si_comparable)]
    piv = sub.pivot_table(index="encoder", columns="fold_mode",
                          values="balanced_accuracy_mean")
    modes = [m for m in FOLD_MODES if m in piv.columns]
    out = {}
    for i, a in enumerate(modes):
        for b in modes[i + 1:]:
            rho = float(spearmanr(piv[a], piv[b]).statistic)
            out[f"{a} vs {b}"] = rho
            if verbose:
                print(f"[p1] rank agreement ({target}, {estimator}, "
                      f"{feature_set}): {a} vs {b}  Spearman rho = {rho:+.3f}")
    if verbose:
        print(f"[p1] encoder order by {feature_set} balanced accuracy:")
        for m in modes:
            order = piv[m].sort_values(ascending=False)
            print(f"[p1]   {m:<14} " + " > ".join(
                f"{e}({v:.3f})" for e, v in order.items()))
    return out


# ---------------------------------------------------------------------------
# Figure 1, the latent clock
# ---------------------------------------------------------------------------

def figure1(manifest, arrays_by_encoder: dict, out_path: str | None = None,
            encoders: Sequence[str] = ENCODER_ORDER, verbose: bool = True):
    """PCA of the pooled embeddings, coloured by month, POOLED ACROSS CUBES.

    It cannot come from one cube: ~13 retained frames over a ~150-day window
    is a fragment of an annual cycle, so one cube draws an arc, not a loop.
    The tile's cube windows start between 2018-03-09 and 2018-07-07, so
    pooling across them is the only seasonal axis this subset offers.

    THIS FIGURE IS DESCRIPTIVE AND PRODUCES NO NUMBER. The PCA is fitted on
    all rows precisely because it is not an estimate of anything -- there is
    no train/test here to get wrong. Every reported score comes from
    ``run_p1`` through ``probes.cv``; nothing is read off these axes.

    Comparability across encoders: features are z-scored per encoder, then the
    PC scores are divided by sqrt(D). Plotted units are therefore a fraction
    of that representation's total standard deviation, which is the same unit
    for D=35 and D=3840, and all five panels are drawn on one shared, square,
    symmetric axis range.

    That range comes from the 99th percentile of the point radius, not from the
    maximum. ``raw_features`` has one frame roughly three times further out than
    anything else, and a limit set by it squeezes the other four panels into an
    indistinguishable blob -- which defeats the only reason the axes are shared.
    Points beyond the range are still drawn; how many fall outside is printed in
    the panel title, so a clipped point is disclosed rather than hidden.

    Colour is a SEQUENTIAL map over the realised months, not a cyclic one.
    Month is cyclic and a cyclic map would be the principled default, but this
    subset realises a contiguous 8-month ARC (April-November); cyclic maps put
    their darkest band across the middle of that arc and render June, July and
    August indistinguishable, losing exactly the structure the figure exists to
    show.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    encoders = [e for e in encoders if e in arrays_by_encoder]
    months = month_labels(manifest)
    realised = sorted(set(months.tolist()))
    windows = manifest.drop_duplicates("cube_id")["cube_id"].map(
        lambda c: "_".join(c.split("_")[1:3])).nunique()
    if verbose:
        print(f"[p1] Figure 1: {len(manifest)} frames pooled across "
              f"{manifest.cube_id.nunique()} cubes / {windows} distinct time "
              f"windows, {len(realised)} realised months {realised}")

    scores, evr = {}, {}
    for enc in encoders:
        X = arrays_by_encoder[enc]["pooled"].astype(np.float64)
        D = X.shape[1]
        Z = StandardScaler().fit_transform(X)
        p = PCA(n_components=2, svd_solver="full").fit(Z)
        s = p.transform(Z) / np.sqrt(D)   # fraction of total std, comparable
        assert s.shape == (len(manifest), 2), s.shape
        scores[enc], evr[enc] = s, p.explained_variance_ratio_
        if verbose:
            print(f"[p1]   {enc:<24} pooled {X.shape} -> PC scores {s.shape}, "
                  f"explained variance {evr[enc][0]:.1%} + {evr[enc][1]:.1%} "
                  f"= {evr[enc].sum():.1%}")

    radii = np.concatenate([np.hypot(s[:, 0], s[:, 1]) for s in scores.values()])
    lim = float(np.quantile(radii, 0.99)) * 1.25
    cmap = plt.get_cmap("turbo")
    lo, hi = min(realised), max(realised)
    colour = {m: cmap(0.06 + 0.88 * (m - lo) / max(1, hi - lo)) for m in realised}

    fig, axes = plt.subplots(1, len(encoders), figsize=(3.0 * len(encoders), 4.1),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, enc in zip(axes, encoders):
        s = scores[enc]
        for m in realised:
            sel = months == m
            ax.scatter(s[sel, 0], s[sel, 1], s=26, alpha=0.85,
                       color=colour[m], linewidths=0.3, edgecolors="white")
        outside = int((np.abs(s) > lim).any(axis=1).sum())
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.axhline(0, lw=0.5, color="0.85", zorder=0)
        ax.axvline(0, lw=0.5, color="0.85", zorder=0)
        head = enc + ("\nMULTI-IMAGE, variable lookback"
                      if enc == MI_ENCODER else
                      f"\nD={arrays_by_encoder[enc]['pooled'].shape[1]}")
        tail = f"PC1 {evr[enc][0]:.0%} / PC2 {evr[enc][1]:.0%}"
        if outside:
            tail += f"   ({outside} frame{'s' if outside > 1 else ''} off-axes)"
        ax.set_title(f"{head}\n{tail}", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("PC2  (fraction of total sd)", fontsize=8)
    # One shared x label rather than five: with the panels this close together,
    # per-axes labels collide with the legend below them.
    fig.supxlabel("PC1  (fraction of total sd)", fontsize=8, y=0.115)

    handles = [Line2D([], [], marker="o", ls="", markersize=6, color=colour[m],
                      label=f"{m:02d}") for m in realised]
    fig.legend(handles=handles, loc="lower center", ncol=len(realised),
               frameon=False, fontsize=8, title="month (2018)",
               title_fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Figure 1 - the latent clock: PCA of pooled frozen embeddings, "
                 "coloured by month\n"
                 f"tile 32UNU, {len(manifest)} retained frames pooled across "
                 f"{manifest.cube_id.nunique()} cubes / {windows} time windows. "
                 "Shared axes (99th-percentile radius).\n"
                 "Descriptive only: no score is read off this figure.",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0.17, 1, 0.88))

    out_path = out_path or figure_path()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        n_out = sum(int((np.abs(s) > lim).any(axis=1).sum())
                    for s in scores.values())
        print(f"[p1] Figure 1 -> {out_path} "
              f"({os.path.getsize(out_path) / 1e3:.0f} kB), "
              f"{len(encoders)} panels on shared limits +/-{lim:.3f} "
              f"(99th pct radius x1.25); {n_out} of "
              f"{len(manifest) * len(encoders)} points fall outside and are "
              "disclosed per panel")
    return out_path


# ---------------------------------------------------------------------------
# Artefact paths
# ---------------------------------------------------------------------------

def results_path(name: str = "p1_appearance_results.csv") -> str:
    return os.path.join(phase_dir(PHASE, "results"), name)


def figure_path(name: str = "figure1_latent_clock.png") -> str:
    return os.path.join(phase_dir(PHASE, "figures"), name)
