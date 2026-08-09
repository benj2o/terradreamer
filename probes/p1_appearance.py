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

THE BASELINE SEES A BAND THE NETWORKS DO NOT, SO THE TABLE CARRIES A
BAND-MATCHED BASELINE TOO
--------------------------------------------------------------------
``raw_features`` is computed over all four bands the cubes carry, B02/B03/B04
and **B8A**, and includes seven statistics of canonical NDVI. All four network
encoders are RGB-only -- ``imagenet_vit_b16`` and ``dinov2_vitb14`` by
construction, and both Satlas wrappers because the RGB variant was chosen over
the multi-spectral one (which expects nine bands these cubes do not have; see
docs/DECISIONS.md). NIR is where the seasonal vegetation signal mostly lives,
so ``raw_features`` is not a simpler model of the same evidence: it is a model
of MORE evidence, and comparing a network to it is not a like-for-like test of
the representation.

The fix costs nothing, because ``raw_features`` stores its per-band statistics
in a KNOWN COLUMN ORDER (``encoders.raw_features.RAW_FEATURE_NAMES``). The
band-matched baseline is a column slice of an array already on disk -- no
re-encode, no new weights:

    raw_rgb_only    B02/B03/B04 statistics only. THE FAIR COMPARISON: the same
                    bands the networks get, reduced by hand instead of by a
                    network.
    raw_nir_ndvi    B8A statistics plus canonical NDVI. What the networks are
                    denied.

The column indices are DERIVED from ``RAW_FEATURE_NAMES`` at import and
asserted to partition it, never hard-coded, so a change to the baseline's
feature order cannot silently repoint these slices at the wrong bands.

Read ``raw_rgb_only``, not ``raw_features``, when the question is whether a
frozen representation beats hand-crafted features. Read ``raw_features`` when
the question is what the best available summary of this cube achieves.

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
from dataclasses import dataclass, replace
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
    "embeddings_dir", "load_encoder_arrays", "FeatureBlock", "feature_matrix",
    "RGB_BANDS", "NIR_BANDS", "RAW_BAND_COLUMNS",
    "subset_arrays", "degenerate_arrays", "wsd_bin_labels",
    "assert_window_span_days_informative",
    "make_estimator", "param_grid", "param_name",
    "select_hyperparameter", "evaluate_fold", "evaluate",
    "summarise", "add_margin_over_control", "run_p1", "rank_agreement",
    "figure1",
    "assert_baseline_view_consistent", "assert_results_complete",
    "results_path", "figure_path",
]

# Phase 1.4 is P1. Artefacts are phase-scoped through data/paths.py; nothing
# here ever writes to a hand-typed path.
PHASE = "phase1_4"

TARGETS = ("month", "season")
FOLD_MODES = ("cube", "loco", "spatial_block")
# Both weightings of both estimators. See make_estimator for why the pair is
# reported rather than one of them chosen.
ESTIMATORS = ("logreg", "logreg_balanced", "ridge", "ridge_balanced")
FEATURE_SETS = ("pooled", "grid_cell", "raw_pooled", "degenerate",
                "raw_rgb_only", "raw_nir_ndvi")

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

# Regularisation grids. The strong end exists because the POOLED sets are
# severely p >> n (DINOv2 is D=3840 against 264 rows); the weak end exists
# because the PRIMARY grid_cell set is not p >> n at all (4224 rows against
# D <= 768) and genuinely wants less regularisation.
#
# The first Phase 1.4 run stopped at C=1 and selected there in 57% of folds --
# the regularisation was pinned by the grid rather than chosen by the data, so
# those scores were a lower bound of unknown tightness. Extending upward costs
# almost nothing: larger C separates sooner, so lbfgs converges in FEWER
# iterations (measured on dinov2 grid_cell: 67 iters at C=1, 30 at C=100).
# Both grids now bracket their interior modes on both sides.
LOGREG_C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2)
RIDGE_ALPHA_GRID = (1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5)

# The multi-image lookback strata. Terciles of window_span_days, computed from
# the realised values rather than fixed at nominal day counts -- the nominal
# 35-day span for 8 frames at a 5-day revisit is wrong by a factor of three
# here (measured median 55, max 105).
WSD_BINS = ("short", "medium", "long")

# The band-matched baselines, as COLUMN GROUPS of raw_features. Derived from
# the baseline's own feature names rather than hard-coded, and asserted to
# partition them, so reordering RAW_FEATURE_NAMES cannot silently repoint these
# at the wrong bands. See the module docstring for why they exist.
RGB_BANDS = ("B02", "B03", "B04")
NIR_BANDS = ("B8A", "NDVI")


def _raw_band_columns() -> dict:
    from encoders.raw_features import RAW_FEATURE_NAMES

    groups = {"raw_rgb_only": RGB_BANDS, "raw_nir_ndvi": NIR_BANDS}
    out = {k: np.array([i for i, n in enumerate(RAW_FEATURE_NAMES)
                        if n.split("_")[0] in bands], dtype=int)
           for k, bands in groups.items()}
    covered = np.concatenate(list(out.values()))
    assert sorted(covered.tolist()) == list(range(len(RAW_FEATURE_NAMES))), (
        f"the band groups {groups} do not partition RAW_FEATURE_NAMES "
        f"({len(RAW_FEATURE_NAMES)} features); a band was renamed or added and "
        "the band-matched baseline would silently be over the wrong columns"
    )
    for k, cols in out.items():
        assert cols.size, f"{k} selected no columns"
    return out


RAW_BAND_COLUMNS = _raw_band_columns()

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


@dataclass(frozen=True)
class FeatureBlock:
    """A design matrix, the manifest row of each of its rows, and its labels --
    bound together so they cannot be sliced apart.

    WHY THIS IS A TYPE AND NOT THREE VARIABLES. The cell-level feature set puts
    16 X rows on every manifest row, so ``X``, ``row_idx`` and ``y`` have a
    length that is not the manifest's and not each other's business to
    remember. Any code that slices one and forgets another produces an index
    that points at the wrong rows -- and for the FRAME-level sets, where all
    three happen to be 264 long, the mistake does not even change a shape. It
    returns a confident wrong number.

    That is not hypothetical: it happened once during Phase 1.4, when
    ``select_hyperparameter`` was handed a sliced ``X_tr`` next to the FULL
    ``row_idx``. It surfaced as an ``IndexError`` -- and the reconstruction says
    that was luck, not detection. Replaying the bad call over the real manifest:

        cube k=5          15/15 inner folds raise
        spatial_block     13/15 raise, 2 return a WRONG NUMBER
        LOCO              51/60 raise, 9 return a WRONG NUMBER

    **11 of 90 inner folds, 12%, would have silently selected a
    regularisation strength from the wrong rows** -- and they cluster in LOCO,
    where the training fold is largest and its row positions therefore most
    likely to fit inside the sliced array. The first probe run happened to
    exercise ``cube`` mode, where the bug is 100% fatal. Had it started with
    LOCO it would have produced a plausible table.

    An assertion at that call site would have fixed that call site; the hazard
    is a property of the PATTERN, and P2, P3 and P4 all consume cell-level rows
    and will write the pattern again.

    So the arrays travel as one value. ``take`` and ``select`` slice all three
    or none, ``with_labels`` is the only way labels get attached (via
    ``row_idx``, which is the one correct way to index them), and the
    length agreement is asserted on EVERY construction, including every slice.
    """

    X: np.ndarray            # (n_rows, D) float64
    row_idx: np.ndarray      # (n_rows,) manifest row position of each X row
    y: np.ndarray = None     # (n_rows,) labels, or None until with_labels
    name: str = ""

    def __post_init__(self):
        assert self.X.ndim == 2, f"{self.name}: X must be 2-D, got {self.X.shape}"
        assert self.row_idx.shape == (self.X.shape[0],), (
            f"{self.name}: row_idx {self.row_idx.shape} does not describe X "
            f"{self.X.shape}. These two must be sliced together, always."
        )
        assert self.y is None or self.y.shape == (self.X.shape[0],), (
            f"{self.name}: y {None if self.y is None else self.y.shape} does "
            f"not describe X {self.X.shape}"
        )

    @property
    def n_rows(self) -> int:
        return int(self.X.shape[0])

    @property
    def D(self) -> int:
        return int(self.X.shape[1])

    def with_labels(self, y_frame: np.ndarray) -> "FeatureBlock":
        """Attach per-MANIFEST-ROW labels, expanded through ``row_idx``.

        The only supported way to get labels onto a block. ``y_frame`` is one
        label per manifest row; the expansion to feature rows is this class's
        job precisely so no caller writes ``y_frame[row_idx]`` by hand against
        a row_idx that may not be the one belonging to this X.
        """
        y_frame = np.asarray(y_frame)
        assert y_frame.ndim == 1, y_frame.shape
        assert self.row_idx.max() < y_frame.shape[0], (
            f"{self.name}: row_idx reaches row {self.row_idx.max()} but only "
            f"{y_frame.shape[0]} labels were given"
        )
        return replace(self, y=y_frame[self.row_idx])

    def take(self, positions: np.ndarray) -> "FeatureBlock":
        """Slice X, row_idx and y together. There is no way to slice one."""
        positions = np.asarray(positions, dtype=int)
        return replace(self, X=self.X[positions], row_idx=self.row_idx[positions],
                       y=None if self.y is None else self.y[positions])

    def rows_for(self, manifest_rows) -> np.ndarray:
        """Positions of the feature rows belonging to a set of manifest rows."""
        keep = np.flatnonzero(np.isin(self.row_idx, np.asarray(manifest_rows)))
        assert keep.size, f"{self.name}: a fold side selected zero feature rows"
        return keep

    def select(self, manifest_rows) -> "FeatureBlock":
        """The sub-block belonging to a set of manifest rows."""
        return self.take(self.rows_for(manifest_rows))


def feature_matrix(arrays: dict, feature_set: str, level: str = "auto") -> FeatureBlock:
    """The FeatureBlock for one feature set.

    ``row_idx[i]`` is the MANIFEST row that X row i belongs to -- one-to-one for
    frame-level sets, 16-to-one for cell-level ones. Carrying it (bound to X, in
    a FeatureBlock) rather than assuming alignment is what lets a fold defined on
    manifest rows select cell rows without a second, hand-rolled index.
    """
    n = arrays["pooled"].shape[0]
    frame_rows = np.arange(n)
    cell_rows = np.repeat(frame_rows, _GRID_CELLS)

    if feature_set in ("pooled", "raw_pooled"):
        X, row_idx = arrays["pooled"], frame_rows
    elif feature_set == "grid_cell":
        g = arrays["grid"]
        X, row_idx = g.reshape(n * _GRID_CELLS, g.shape[-1]), cell_rows
    elif feature_set in RAW_BAND_COLUMNS:
        g = arrays["grid"]
        assert g.shape[-1] == 35, (
            f"{feature_set} is a column slice of the raw_features grid "
            f"(D=35), but this array has D={g.shape[-1]}. It is only defined "
            "for the raw_features encoder."
        )
        cols = RAW_BAND_COLUMNS[feature_set]
        X = g[:, :, cols].reshape(n * _GRID_CELLS, cols.size)
        row_idx = cell_rows
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
    assert np.isfinite(X).all(), f"{feature_set}: non-finite feature"
    label = feature_set if feature_set != "degenerate" else f"degenerate/{level}"
    return FeatureBlock(X=X, row_idx=row_idx, name=label)


# ---------------------------------------------------------------------------
# Estimators, the nested tuning loop, and one fold
# ---------------------------------------------------------------------------

def make_estimator(name: str, param: float):
    """One estimator at one regularisation strength. No RNG, no warm start.

    The ``logreg`` variants are multinomial by construction: sklearn's lbfgs
    solver fits a single multinomial model whenever there are more than two
    classes, and the explicit ``multi_class`` argument that used to say so is
    deprecated, so passing it would emit a FutureWarning while changing nothing.

    BOTH weightings are run, and the pair is the point. An UNWEIGHTED fit is the
    conservative reading: the metric is balanced accuracy, so weighting moves
    the estimator toward the metric, and a conclusion that survives without it
    survives the imbalance rather than being rescued from it. But unweighted is
    also the WRONG estimator for the metric in the ordinary sense -- November
    has 5 frames of 264, and a loss that never pays to predict it forfeits 1/8
    of balanced accuracy outright -- so reporting only the unweighted number
    understates every row by an unknown amount. Reporting both bounds it, and
    the control is weighted the same way, so the margin over the control (the
    number P1 is actually read on) is fair either way.
    """
    from sklearn.linear_model import LogisticRegression, RidgeClassifier

    assert name in ESTIMATORS, f"estimator {name!r} not in {ESTIMATORS}"
    weight = "balanced" if name.endswith("_balanced") else None
    if name.startswith("logreg"):
        return LogisticRegression(C=float(param), max_iter=2000, tol=1e-4,
                                  class_weight=weight)
    return RidgeClassifier(alpha=float(param), class_weight=weight)


def param_grid(estimator: str) -> tuple:
    return LOGREG_C_GRID if estimator.startswith("logreg") else RIDGE_ALPHA_GRID


def param_name(estimator: str) -> str:
    return "C" if estimator.startswith("logreg") else "alpha"


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


def select_hyperparameter(train: FeatureBlock, manifest, train_rows,
                          estimator: str, inner_k: int = _INNER_K,
                          verbose: bool = False) -> dict:
    """Tune the regularisation strength on the TRAINING fold only.

    THE SIGNATURE IS THE GUARANTEE. There is no test argument to leak: this
    function is handed the training block (design matrix, its manifest rows and
    its labels, bound together and already sliced to the training fold) and the
    manifest rows that fold occupies, and nothing else. The tests exercise that
    by poisoning the test fold and asserting the selection is bit-identical, and
    by inspecting this signature for any parameter that could carry test data.

    Taking a ``FeatureBlock`` rather than loose arrays is the fix for a real
    Phase 1.4 bug: this function was once handed a sliced ``X_tr`` beside the
    UNSLICED row index. See FeatureBlock for why an assertion here would not
    have been enough.

    The inner split is ``probes.cv.folds`` on the TRAINING sub-manifest, mode
    "cube": cube grouping is the strictest rule any outer mode here obeys, so
    an inner fold can never put a training cube on both sides either.
    """
    grid = param_grid(estimator)
    assert train.y is not None, "the training block carries no labels"
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

    scores = np.zeros((len(grid), len(inner)))
    for j, (itr, ite) in enumerate(inner):
        # Inner positions index the SUB-manifest; map back to manifest rows,
        # then let the block slice itself. Neither hop can drop an array.
        tr_rows, te_rows = train_rows[itr], train_rows[ite]
        assert not np.intersect1d(tr_rows, te_rows).size
        a, b = train.select(tr_rows), train.select(te_rows)
        for i, p in enumerate(grid):
            pred = _fit_predict(estimator, p, a.X, a.y, b.X)
            scores[i, j] = _score(b.y, pred)

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


def evaluate_fold(block: FeatureBlock, manifest, train_rows, test_rows,
                  estimator: str, fold: int = 0, inner_k: int = _INNER_K,
                  verbose: bool = True) -> FoldResult:
    """One outer fold, end to end: tune on train, fit on train, score on test.

    Public because it is the seam the leakage test needs: it is called twice
    with an identical training side and a DIFFERENT (poisoned) test side, and
    the selected regularisation strength must be identical.
    """
    from sklearn.dummy import DummyClassifier

    assert block.y is not None, "the block carries no labels; call with_labels"
    train_rows = np.asarray(train_rows, dtype=int)
    test_rows = np.asarray(test_rows, dtype=int)
    assert not np.intersect1d(train_rows, test_rows).size, (
        f"fold {fold}: manifest rows appear on both sides"
    )
    a_pos, b_pos = block.rows_for(train_rows), block.rows_for(test_rows)
    assert not np.intersect1d(a_pos, b_pos).size, f"fold {fold}: rows overlap"
    train, test = block.take(a_pos), block.take(b_pos)

    sel = select_hyperparameter(train, manifest, train_rows, estimator,
                                inner_k=inner_k, verbose=False)
    pred = _fit_predict(estimator, sel["param"], train.X, train.y, test.X)

    # Metrics over the classes present in THIS test fold; see _score.
    labels = np.unique(test.y)
    ba, f1 = _score(test.y, pred), _score(test.y, pred, macro_f1=True)
    dpred = (DummyClassifier(strategy="most_frequent")
             .fit(train.X, train.y).predict(test.X))
    dba, df1 = _score(test.y, dpred), _score(test.y, dpred, macro_f1=True)

    cubes = manifest["cube_id"].to_numpy()
    edge = " [GRID EDGE]" if sel["at_grid_edge"] else ""
    # The dummy is printed as a MARGIN, not as a second number to be eyeballed
    # against the global chance line. A fold holding one cube holds ~5 of the 8
    # months, so its implicit floor is ~1/5, not 1/8; bal-acc minus this fold's
    # own dummy is the only quantity that means the same thing in every fold.
    log = (f"[p1]   fold {fold + 1}: train {train.n_rows} rows / "
           f"{len(set(cubes[train_rows]))} cubes, test {test.n_rows} rows / "
           f"{len(set(cubes[test_rows]))} cubes, {labels.size} classes in test | "
           f"{param_name(estimator)}={sel['param']:g}{edge} | "
           f"bal-acc {ba:.3f} = dummy {dba:.3f} {ba - dba:+.3f} | "
           f"macro-F1 {f1:.3f} = dummy {df1:.3f} {f1 - df1:+.3f}")
    if verbose:
        print(log)
    return FoldResult(
        fold=fold, n_train=train.n_rows, n_test=test.n_rows,
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


def evaluate(block: FeatureBlock, manifest, mode: str, estimator: str, k: int = 5,
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
        results = [evaluate_fold(block, manifest, tr, te, fold=i, **call)
                   for i, (tr, te) in enumerate(outer)]
    else:
        from joblib import Parallel, delayed, parallel_config
        with parallel_config(backend="loky", inner_max_num_threads=1):
            results = Parallel(n_jobs=n_jobs)(
                delayed(evaluate_fold)(block, manifest, tr, te, fold=i, **call)
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
    """Tercile label per frame, REFUSING a degenerate cut.

    An all-zero (or near-constant) lookback gives identical quantiles, and
    ``np.digitize`` then drops every frame into one bin while the caller still
    prints three tercile rows. Nothing raises: zeros are finite, in range and
    the right shape. So the non-degeneracy is asserted here rather than left to
    be noticed in the bin counts.
    """
    wsd = np.asarray(wsd, dtype=float)
    edges = _wsd_bin_edges(wsd)
    assert edges[0] < edges[1], (
        f"window_span_days terciles are degenerate (both cuts at {edges[0]:.0f} "
        f"days over {wsd.size} frames, min {wsd.min():.0f} max {wsd.max():.0f}). "
        "The lookback covariate is constant, so conditioning on it measures "
        "nothing -- see assert_window_span_days_informative."
    )
    idx = np.digitize(wsd, edges, right=False)
    out = np.array([WSD_BINS[min(int(i), 2)] for i in idx])
    counts = {b: int((out == b).sum()) for b in WSD_BINS}
    assert min(counts.values()) > 0, f"an empty lookback tercile: {counts}"
    return out


def assert_window_span_days_informative(arrays_by_encoder: dict) -> None:
    """The multi-image lookback must actually vary, and the SI ones must not.

    WHY THIS GUARD EXISTS. ``window_span_days`` is identically 0 for a
    single-image encoder BY DESIGN -- that is the honest value, not a missing
    one -- so "all zeros" is a legitimate state for four of the five caches and
    no shape, dtype or finiteness check can tell a correct zero from a lost
    covariate. If the MI cache also came back zero (a pre-v3 file, a re-encode
    at ``window_len=1``, a join that dropped the field), two things would fail
    SILENTLY while still printing full output: the degenerate control would
    quietly become ``clear_frac`` alone, half the control it claims to be, and
    the MI lookback strata would collapse into one bin.

    The retention control is currently the most defensible result in this
    phase. It does not get to degrade quietly.
    """
    for enc, a in arrays_by_encoder.items():
        w = np.asarray(a["window_span_days"], dtype=float)
        if enc == MI_ENCODER:
            assert w.max() > 0 and np.unique(w).size >= 3, (
                f"{enc} is MULTI-IMAGE but its window_span_days is constant "
                f"(min {w.min():.0f} max {w.max():.0f}, "
                f"{np.unique(w).size} distinct value(s)). Expected a variable "
                "lookback (measured on this subset: 0-105 days). The covariate "
                "was not cached, or the cache predates schema v3 -- re-encode "
                "Phase 1.2 rather than reporting a control that is really "
                "clear_frac alone."
            )
        elif enc in ENCODER_ORDER:
            assert (w == 0).all(), (
                f"{enc} is single-image, so window_span_days must be exactly 0 "
                f"(got min {w.min():.0f} max {w.max():.0f}). A non-zero value "
                "means the roster and the cache disagree about window_len."
            )
    w = arrays_by_encoder[MI_ENCODER]["window_span_days"]
    print(f"[p1] window_span_days INFORMATIVE: {MI_ENCODER} varies over "
          f"{np.unique(w).size} distinct values, {w.min():.0f}-{w.max():.0f} d; "
          f"all single-image encoders exactly 0")


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
    wsd = np.asarray(arrays_by_encoder[src]["window_span_days"], dtype=float)
    assert src != MI_ENCODER or wsd.max() > 0, (
        "the degenerate control was asked for window_span_days from "
        f"{MI_ENCODER} and got a column of zeros. The control would silently "
        "become [clear_frac] alone -- half the control it reports -- while "
        "still printing a two-column D=2 row. See "
        "assert_window_span_days_informative."
    )
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

def add_margin_over_control(df, verbose: bool = True):
    """Attach the two margins P1 should actually be read on.

    ``margin_over_dummy`` -- balanced accuracy minus THIS row's own per-fold
    most-frequent dummy. A cube-grouped test fold holds a subset of the classes
    (one cube covers about 5 of the 8 months), so its implicit floor is not the
    global 1/K, and comparing a LOCO score against the flat chance line credits
    it with a margin the split never offered. Measured here: the dummy is 0.132
    under ``cube`` but 0.199 under ``loco``. The margin is the only quantity
    that means the same thing in every fold mode.

    ``margin_over_band_matched`` -- balanced accuracy minus ``raw_rgb_only``,
    the hand-crafted baseline over the SAME three bands every network encoder
    receives. This is the like-for-like representation-quality number; the
    margin over full ``raw_features`` is not, because that baseline additionally
    sees B8A and NDVI.

    ``margin_over_control`` -- balanced accuracy minus the DEGENERATE control
    for the same (target, fold_mode, estimator), matched at cell level. On this
    subset cloud retention is strongly seasonal, so distance from chance
    overstates what a representation contributed; distance from the control is
    the part that needed an image. Encoder and control are scored on the same
    folds with the same available classes, so this margin is also immune to the
    per-fold class-count effect above.

    Both are attached rather than left to the reader, because a column that has
    to be computed by hand is a column that gets read wrong.
    """
    key = ["target", "fold_mode", "estimator"]
    ctrl = df[(df.feature_set == "degenerate") & (df.feature_level == "cell")]
    assert len(ctrl) == len(ctrl.drop_duplicates(key)), "control rows not unique"
    lookup = ctrl.set_index(key)["balanced_accuracy_mean"]
    band = df[df.feature_set == "raw_rgb_only"]
    band_lookup = (band.set_index(key)["balanced_accuracy_mean"]
                   if len(band) else None)
    df = df.copy()
    # Recomputed here even though _rows_dict already set it (the per-row print
    # needs it before the table exists). Same arithmetic, so the two cannot
    # disagree, and this helper then works on any table with the base columns.
    df["margin_over_dummy"] = (df.balanced_accuracy_mean
                               - df.dummy_balanced_accuracy_mean)
    df["margin_over_control"] = (
        df.balanced_accuracy_mean
        - df.set_index(key).index.map(lookup).to_numpy()
    )
    assert df.margin_over_control.notna().all(), (
        "some row has no matching degenerate control; the control is not "
        "optional and every row must be comparable to it"
    )
    if band_lookup is not None:
        df["margin_over_band_matched"] = (
            df.balanced_accuracy_mean
            - df.set_index(key).index.map(band_lookup).to_numpy()
        )
        assert df.margin_over_band_matched.notna().all(), (
            "some row has no matching raw_rgb_only baseline"
        )
    if verbose:
        beaten = df[(df.feature_set == "grid_cell")
                    & (df.margin_over_control <= 0)]
        print(f"[p1] margins attached. {len(beaten)}/"
              f"{int((df.feature_set == 'grid_cell').sum())} grid_cell rows are "
              "AT OR BELOW the retention-only control")
    return df


def _rows_dict(target, mode, estimator, encoder, feature_set, level, wsd_bin,
               block, manifest, results, chance, comparable, note):
    s = summarise(results)
    X = block.X
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
        # Row-local, so the per-row print can use it; margin_over_control needs
        # the whole table and is attached by add_margin_over_control.
        "margin_over_dummy": float(s["balanced_accuracy_mean"]
                                   - s["dummy_balanced_accuracy_mean"]),
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
    assert_window_span_days_informative(arrays)
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
                # The BAND-MATCHED baseline and its complement. Same rows, same
                # folds, same estimator as the grid_cell networks -- the only
                # difference is which of the cube's four bands the features saw.
                jobs.append((BASELINE_ENCODER, BASELINE_ENCODER,
                             "raw_rgb_only", "cell"))
                jobs.append((BASELINE_ENCODER, BASELINE_ENCODER,
                             "raw_nir_ndvi", "cell"))
                jobs.append(("__degenerate__", "none", "degenerate", "frame"))
                jobs.append(("__degenerate__", "none", "degenerate", "cell"))

                for src, enc_label, fs, level in jobs:
                    block = feature_matrix(arrays[src], fs,
                                           level=level).with_labels(y_frame)
                    res = evaluate(block, manifest, mode, estimator,
                                   k=k, inner_k=inner_k, n_jobs=n_jobs,
                                   verbose=False)
                    comparable, note = src != MI_ENCODER, ""
                    if src == MI_ENCODER:
                        note = ("aggregates 8 retained frames over a variable "
                                "lookback (0-105 d); NOT directly comparable "
                                "to the single-image encoders")
                    elif fs == "raw_rgb_only":
                        note = ("BAND-MATCHED baseline: B02/B03/B04 statistics "
                                "only, the same bands every network encoder "
                                "sees. This is the like-for-like comparison")
                    elif fs == "raw_nir_ndvi":
                        note = ("B8A statistics + canonical NDVI: the evidence "
                                "the RGB-only network encoders are denied")
                    elif fs == "degenerate":
                        note = ("retention only: "
                                + ("[clear_frac, window_span_days]" if level == "frame"
                                   else "[grid_clear_frac, window_span_days]")
                                + f", NO embedding; window_span_days from {MI_ENCODER}")
                    r = _rows_dict(target, mode, estimator, enc_label, fs, level,
                                   "all", block, manifest, res, chance,
                                   comparable, note)
                    rows.append(r)
                    if verbose:
                        print(f"[p1]   {enc_label:<24} {fs:<11} {level:<5} "
                              f"n={block.n_rows:<5} D={block.D:<5} "
                              f"bal-acc {r['balanced_accuracy_mean']:.3f} "
                              f"+/-{r['balanced_accuracy_std']:.3f} "
                              f"(spread {r['balanced_accuracy_spread']:.3f}) "
                              f"| over dummy {r['margin_over_dummy']:+.3f} "
                              f"| macro-F1 {r['macro_f1_mean']:.3f} "
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
                        block = feature_matrix(a, "pooled").with_labels(
                            target_labels(sub, target))
                        ch = chance_level(block.y)
                        res = evaluate(block, sub, mode, estimator,
                                       k=min(k, n_cubes), inner_k=inner_k,
                                       n_jobs=n_jobs, verbose=False)
                        r = _rows_dict(target, mode, estimator, MI_ENCODER,
                                       "pooled", "frame", wbin, block, sub, res,
                                       ch, False,
                                       f"window_span_days tercile {wbin!r} = "
                                       f"{a['window_span_days'].min():.0f}-"
                                       f"{a['window_span_days'].max():.0f} days, "
                                       f"{n_cubes} cubes; chance recomputed on "
                                       "this tercile's own class distribution")
                        rows.append(r)
                        if verbose:
                            print(f"[p1]   {MI_ENCODER:<24} pooled      "
                                  f"wsd={wbin:<7} n={block.n_rows:<5} "
                                  f"bal-acc {r['balanced_accuracy_mean']:.3f} "
                                  f"+/-{r['balanced_accuracy_std']:.3f} "
                                  f"(chance {ch['balanced_accuracy']:.3f} = 1/"
                                  f"{ch['n_classes']})")

    df = pd.DataFrame(rows)
    assert len(df), "run_p1 produced no rows"
    return add_margin_over_control(df, verbose=verbose)


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
                assert (sub.feature_set == "raw_rgb_only").any(), (
                    f"{target}/{mode}/{est}: no band-matched baseline. Every "
                    "network encoder here is RGB-only, so a comparison against "
                    "full raw_features (which also sees B8A and NDVI) is not a "
                    "statement about the representation."
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
