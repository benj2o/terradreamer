"""P3, the forecastability probe: does a cheap read-out over FROZEN embeddings
plus weather forecast FUTURE NDVI, and does it beat persistence and climatology?

The question P1, P2 and P4 were calibration for. P1 established that appearance
is present, P2 that the SIGN of change is recoverable and its MAGNITUDE is not,
P4 that weather alone explains 0.09-0.13 of the post-(proxy)-climatology anomaly
at cell level. P3 asks the thing the paper is actually about: given the state at
time t and the weather over the next Delta days, can a linear or small non-linear
head over a frozen representation say where NDVI will be -- and does it beat the
two baselines any operational forecaster would already have.

WHAT THIS PROBE READS, AND WHAT IT REFUSES TO READ
---------------------------------------------------
The scaled cache (``data/scaled_32UNU/{raw,embeddings,masks}``, 115 cubes x 5
encoders), the cubes' in-cube E-OBS series, and the manifest. **No encoder is
imported, loaded or fine-tuned.** Frozen by construction, because no network is
present in this process at all. CPU throughout.

HORIZONS ARE DAYS, AND THE DAY AXIS IS NOT THE JOIN KEY
--------------------------------------------------------
``original_axis_index`` counts ACQUISITIONS (~14 per cube here) and is the
embedding join key. ``daily_axis_index`` counts DAYS (~150 per cube) and is the
axis a horizon lives on. They differ by about 5x -- one Sentinel-2 orbit lattice
-- so a horizon read off the join key would be five times too short, and the
error is silent: the wrong column is finite, integer, in range, monotone within
a cube, and correlated with the right one.

``horizon_index`` therefore calls ``p2_deltas.assert_gap_axes_disagree`` on the
REAL consecutive pairs before it computes a single horizon, and
``assert_horizon_axis`` then re-applies that measured disagreement to P3's own
(t, target) pairs: for every row the day-axis horizon must equal the timestamp
difference, and the acquisition-axis reading of the same horizon must be
materially shorter. A guard that has never fired on real data is a hypothesis.

TARGET-FRAME SELECTION, AND THE WINDOW-BOUNDARY DROP
-----------------------------------------------------
For each (cube, t, Delta_days) the target is the nearest RETAINED acquisition to
``t + Delta_days`` **inside the same cube's covered window**. If none falls
within ``TOLERANCE_DAYS`` of it, the row is DROPPED -- never wrapped to another
cube, never extrapolated, never filled.

The tolerance is +/-3 days and on this subset it does NO WORK, which is worth
stating rather than leaving as an unexamined parameter: every retained frame
satisfies ``doy % 5 == 2`` (one orbit lattice), so every realisable horizon is a
multiple of 5 days and the four nominal horizons are too. A +/-3 day tolerance
therefore accepts EXACT matches only, and +/-2 gives the identical row set.
``horizon_tolerance_sensitivity`` prints the row counts at several tolerances so
that this is a measurement and not an assumption; loosening to +/-5 would admit
the neighbouring lattice point and change the horizon by 5 days, which is 100%
of the shortest horizon.

The consequence is a real and reportable shrinkage at the long horizon:

    Delta =   5 d   518 rows over 115 cubes
    Delta =  25 d   489 rows over 114 cubes
    Delta =  50 d   450 rows over 115 cubes
    Delta = 100 d   196 rows over  94 cubes     <- the cubes run out of window

A 150-day cube whose retained frames span ~135 days simply cannot offer many
100-day-ahead pairs, and 21 cubes offer none at all. That is a statement about
what this benchmark can support at that horizon, so it is printed
(``print_horizon_retention``), asserted to be non-increasing toward 100 d
(``assert_retention_shrinks``), and carried on every row of the CSV as
``n_retained`` and ``n_cubes``. It is not hidden by pooling horizons.

CONTEXT: THREE FRAMES FOR THE SINGLE-IMAGE ENCODERS, ONE FOR THE MULTI-IMAGE ONE
---------------------------------------------------------------------------------
For ``raw_features``, ``imagenet_vit_b16``, ``dinov2_vitb14`` and
``satlas_s2_swinb_rgb`` the context is the k=3 most recent RETAINED frames at or
before t, concatenated OLDEST FIRST, so the last block is always t itself.

Rows with fewer than 3 prior retained frames are **DROPPED, not zero-padded.**
A zero embedding is not a neutral input to a linear or an MLP read-out: it is a
point far outside the data cloud after standardisation, and it would appear
exclusively at the START of every cube -- i.e. exclusively in early spring. A
padded model would therefore learn "spring looks like a zero vector", which is a
seasonal signal manufactured by the padding rule. The drop costs the first two
frames of each cube (230 of 1580) and is applied ONCE, to define the row set,
before any horizon is selected -- so every encoder is scored on identical rows.

``satlas_s2_swinb_mi_rgb`` gets its SINGLE embedding at t and nothing else. Its
embedding at t already max-pools up to 8 preceding retained frames (a lookback
of 0-105 days on this subset), so stacking three of them would double-count the
lookback: consecutive MI embeddings share up to 7 of their 8 source frames, and
a 3-stack would be three heavily overlapping summaries of one window presented
as three observations. ``context_block`` REFUSES a 3-frame stack for the MI
encoder with a loud message rather than accepting it silently, and every MI row
is flagged ``si_comparable=False`` -- the same convention P1 and P2 use.

TARGETS, SPATIALLY AGGREGATED, NEVER PIXEL-WISE, AND ALWAYS COMMON-MASKED
--------------------------------------------------------------------------
    cube_mean   PRIMARY. One row per (cube, t, Delta).
    cube_p90    Cube-90th-percentile. Secondary. Same rows.
    cell_mean   The 16 grid cells. The RESOLUTION axis, 16 rows per row.

Every target is the aggregate of NDVI at ``t + Delta`` over the pixels valid in
BOTH the frame at t and the target frame. That is not decoration: it is what
makes the persistence baseline exactly right (see below), and it is the same
intersection rule P2 established.

**Why the target is common-masked and not the target frame's own mean.** If the
target were frame-t+Delta's own mean and persistence predicted frame-t's own
mean, the persistence error would be a difference of two aggregates over two
DIFFERENT pixel sets -- P2's differencing trap, which agrees with the right
answer only when the masks agree, i.e. exactly when the distinction does not
matter. Under the common-masked definition the persistence residual **is**
``p2_deltas.common_masked_delta``'s change, by construction and not by
coincidence, and ``common_masked_levels`` asserts that identity for all three
aggregations on every pair. The delta arithmetic is imported, never re-derived;
this module only adds the two LEVELS on the same intersection and pins them to
it.

CONTEXT LEVEL FOLLOWS THE TARGET'S AGGREGATION
-----------------------------------------------
Cube-level targets get POOLED context; ``cell_mean`` gets GRID-CELL context, the
same 3 frames sliced to that cell. A pooled vector cannot address one of 16
cells, which is why P2's ``PART_A_COMBOS`` makes the same restriction. Stated
here because the spec says "pooled": pooled is what the two cube-level targets
-- the primary and the secondary -- use.

WEATHER IS THE HORIZON WINDOW, AND IT IS PERFECT-FORESIGHT
------------------------------------------------------------
``weather(t .. t+Delta)`` is built by ``p4_ceiling.cube_weather_windows``,
unmodified, evaluated at the TARGET frame with a window span of ``Delta + 1``
days and zero lag. A backward window of Delta+1 days ending at t+Delta is
exactly the forward window [t, t+Delta], so the P4 join -- daily axis, located
by TIMESTAMP, never by ``original_axis_index`` -- is reused rather than rebuilt.
``days_available == Delta + 1`` is asserted on every row, which is what proves
the window really starts at t.

This is the weather that ACTUALLY OCCURRED over the horizon, not a forecast of
it. That is deliberate and it is P4's framing: P4 measures how much of the
anomaly weather can explain at all, and P3 asks whether a frozen representation
plus that same weather can place NDVI in the future. A model here is therefore
an ATTRIBUTION-style upper bound on an operational forecast, never an
operational forecast, and no row should be read as one.

THE FIVE BASELINES, ALL MANDATORY, ALL IN THE SAME TABLE
---------------------------------------------------------
    persistence            NDVI(t+Delta) = NDVI(t), both common-masked, so its
                           residual IS p2's common-masked change. No fit, no
                           estimator, nothing to tune. THE baseline a forecast
                           has to beat.
    climatology_proxy      ``p4_ceiling.doy_climatology_within_fold`` at the
                           TARGET frame's day-of-year, fitted on training cubes
                           only. Every such row carries
                           ``climatology_def = PROXY_CLIMATOLOGY_LABEL`` and
                           ``is_proxy_climatology = True``: it is a within-
                           season, single-year, tile-level day-of-year curve and
                           **not** the leave-target-year-out definition. It is
                           NOT H1's number and must never be quoted as one --
                           Stage B remains open (HANDOFF_P3 section 5).
    weather_only           No embedding, no context, no current NDVI. P4's own
                           estimator family (linear, HGB), imported.
    raw_features_weather   The AUTOREGRESSIVE baseline. ``raw_features`` holds
                           ``NDVI_mean``..``NDVI_p90`` as columns, and here that
                           is legitimate and not the K2 leakage case: the target
                           is at t+Delta, not at t, so using current NDVI to
                           predict future NDVI is ordinary autoregression. It is
                           the strongest cheap baseline in the table and it is
                           included as such.
    raw_rgb_only_weather   BAND-MATCHED: the same three bands every network
                           receives, seven percentiles each, no NDVI column. On
                           P2's sign probe this row beat every network
                           (+0.695 against DINOv2's +0.606). It is mandatory
                           here for exactly that reason -- this phase must let
                           the data decide whether the embeddings add anything
                           over hand-crafted band statistics, and a table of
                           foundation-model scores without it is not a table.

THE THREE CONTROLS, NONE OF THEM OPTIONAL
-------------------------------------------
    horizon_only    ``Delta_days`` alone -> NDVI(t+Delta). No embedding, no
                    weather, no current NDVI. P1's degenerate control and P2's
                    gap-length control at this phase's level, and it is
                    ``p2_deltas.fit_gap_control`` IMPORTED -- a horizon in days
                    is precisely a gap in days. It is fitted POOLED ACROSS THE
                    FOUR HORIZONS and scored per horizon, and that is FORCED
                    rather than chosen: within one horizon Delta is a constant,
                    the degree-2 design is rank 1, and p2's own guard
                    ("the training fold covers only 1 distinct gap length")
                    refuses the fit. The pooled fit is the only one in which the
                    control has anything to say. If it scores above the proxy
                    climatology, systematic phenological drift is contaminating
                    the comparison.
    observation     [clear_frac(t), window_span_days(t), clear_frac(target)] ->
                    NDVI(t+Delta). P1's control carried forward and extended by
                    the one observation-process variable that is specific to a
                    forecast: how cloudy the frame we are scored ON happened to
                    be. ``margin_over_control`` is computed against this row for
                    EVERY model row, at the same estimator, and it is the number
                    to quote.
    permutation     Weather shuffled across cubes within the fold, refitted.
                    The EMPIRICAL ZERO for this pipeline, which with grouped
                    data and a flexible estimator is not exactly 0.0.

Controls are emitted under every label they are invariant to, and
``add_margins`` additionally writes ``control_score`` onto EVERY row, so
filtering the CSV to one encoder cannot leave a reader without the control.
``assert_control_identical_across_views`` asserts every copy agrees
digit-for-digit. ``fold_mode``, ``delta_days``, ``aggregation`` and ``estimator``
are part of the control's KEY, not labels it is invariant to: a control
evaluated on different held-out cubes, at a different horizon, against a
different target or with a different learner is a different number, and
reporting one value across them would be a false identity.

METRICS, AND WHAT "BEATING PERSISTENCE" MEANS
-----------------------------------------------
    r2_pooled              **THE number every margin is taken on.** R-squared
                           over every held-out prediction, each used exactly
                           once, against the pooled held-out mean. Its interval
                           is a DELETE-ONE-FOLD JACKKNIFE, which is the same
                           cube-level clustering the other phases apply to a
                           mean, extended to a statistic that is not one.
    r2_mean                the mean of the per-fold R-squareds, with
                           ``p4_ceiling.fold_clustered_ci``. The right number
                           under ``cube`` and ``spatial_block``. **It is NaN
                           under ``loco``** and the row says so through
                           ``n_folds_nan``: a held-out cube contributes two to
                           five forecast rows, and an R-squared against the mean
                           of three points is not a measurement. That is why the
                           margins are on ``r2_pooled`` -- a margin that went
                           missing for a whole fold mode would not be a margin.
    rmse                   in NDVI units, per fold, with the fold-clustered
                           interval. Defined at every fold size, so it is what
                           carries the LOCO uncertainty.
    skill_vs_persistence   1 - SSE(model)/SSE(persistence), pooled out of fold,
                           with its own fold jackknife. **This is the quantity
                           the goal statement names.** Positive means the model
                           beat persistence.
    medae_pooled           MEDIAN absolute error. On the table because three
                           cloud-contaminated frames carry most of the squared
                           error here (see ``print_target_outliers``), and a
                           median cannot be set by five rows.
    sse_share_top1pct      how much of a row's held-out squared error its worst
                           1% of rows carries. Read every R-squared against it.

Reported per horizon, per fold mode and per severity bin, each with its
interval and the effective n beside it.

SEVERITY BINS ARE A REPORTING AXIS AND THEY ARE WHERE THE RESULT LIVES
-----------------------------------------------------------------------
Bins are quantiles of the TARGET's anomaly relative to the proxy climatology,
computed once by ``p4_ceiling.severity_reference_anomaly`` (the one full-data
fit in that module, reporting only -- it is never a target, never a feature and
never enters a score) and printed with edges and counts before anything is
fitted. Stable periods flatter persistence: when NDVI is not moving, "predict
today's value" is nearly optimal and every model looks equal to it. The
extreme_low / extreme_high bins are where a forecast can actually differ from
persistence, so the headline is reported overall AND on those bins.

EFFECTIVE SAMPLE SIZE IS CUBES
--------------------------------
1580 frames, 1653 forecast rows and ~26 000 cell rows are not independent
observations: 16 cells share a sky, ~14 frames share a place, and at a 5-day
horizon consecutive rows of one cube share both endpoints' weather. Every
interval is clustered at the CUBE level, ``effective_n`` counts CUBES and sits
on every row, and at Delta=100 it is at most 94 because that is how many cubes
have a 100-day pair at all.

PROTOCOL
--------
Primary split ``mode="cube"``, k=5. Robustness: ``loco`` and ``spatial_block``,
both reported on the primary aggregation at every horizon. P3's own behaviour
under ``spatial_block`` is UNKNOWN going in -- P2's headline survived it and
P1's and P4's did not -- so it is measured, never inherited. The secondary
aggregations run under the primary mode, which is the same restriction P2 makes
for the same reason: they are a resolution axis, not a second experiment.

Nothing is tuned. Every hyperparameter comes from ``p4_ceiling.make_estimator``
at its fixed a-priori value (ridge alpha = D, applied to standardised features),
so there is no selection loop to prove clean -- strictly stronger than a guarded
one. Standardisation is fitted on train and applied to test, per fold. Every
fold comes from ``probes/cv.py``; any number produced outside it does not exist.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import NamedTuple, Sequence

import numpy as np

from data.paths import phase_dir
from encoders.base import GRID, GRID_CELLS
from encoders.frames import MIN_CLEAR_FRACTION, select_clear_frames
from encoders.pipeline import load_masks
from probes import cv
from probes import p2_deltas as p2
from probes import p4_ceiling as p4
from probes.p1_appearance import (
    BASELINE_ENCODER,
    ENCODER_ORDER,
    MI_ENCODER,
    RAW_BAND_COLUMNS,
    FeatureBlock,
    load_encoder_arrays,
)

__all__ = [
    "PHASE", "HORIZONS", "TOLERANCE_DAYS", "CONTEXT_FRAMES",
    "MI_CONTEXT_FRAMES", "AGGREGATIONS", "TARGET_LEVEL", "CONTEXT_LEVEL",
    "FOLD_MODES", "PRIMARY_AGGREGATION", "AGGREGATION_MODES", "ESTIMATORS",
    "FORECAST_ESTIMATORS", "WEATHER_ONLY_ESTIMATORS", "CONTROL_ESTIMATORS",
    "MODEL_KINDS", "BASELINE_KINDS", "CONTROL_KINDS", "FEATURE_SETS",
    "WEATHER_FEATURE_SET", "PROXY_CLIMATOLOGY_LABEL", "SEVERITY_BINS",
    "HORIZON_CONTROL_DEGREE", "ENCODER_ORDER", "MI_ENCODER", "BASELINE_ENCODER",
    "BAND_MATCHED_BASELINE",
    # logging (p2's, re-exported so there is ONE run-log implementation)
    "open_run_log", "close_run_log", "log", "run_log_path",
    # rows, and the axis guard
    "ForecastRows", "horizon_index", "assert_horizon_axis",
    "horizon_tolerance_sensitivity", "print_horizon_retention",
    "assert_retention_shrinks",
    # common-masked levels
    "common_masked_levels", "cube_ndvi_and_masks", "build_forecast_targets",
    "print_target_outliers", "NDVI_PLAUSIBILITY_FLOOR", "GROWING_SEASON_DOY",
    "assert_climatology_identifiable",
    "summarise_pixel_survival_by_horizon",
    # features
    "context_frames_for", "context_block", "horizon_weather",
    "observation_triple", "manifest_clear_fractions", "encoder_views",
    # the fitted read-out -- train_pos is REQUIRED and POSITIONAL
    "fit_readout",
    # evaluation
    "P3FoldResult", "evaluate_fold", "evaluate", "summarise",
    # the run and the table
    "P3Data", "build_p3_data", "run_p3", "add_margins",
    "print_headlines", "print_controls", "print_severity_table",
    "assert_results_complete", "assert_control_identical_across_views",
    "assert_controls_present", "assert_baselines_present",
    "assert_mi_flagged_and_single_frame", "assert_effective_n_counts_cubes",
    "assert_climatology_rows_labelled", "results_path",
]

# Phase 1.8 is P3. Artefacts are phase-scoped through data/paths.py; nothing
# here ever writes to a hand-typed path.
PHASE = "phase1_8"

# The horizons, in DAYS on daily_axis_index. Never in retained frames.
HORIZONS = (5, 25, 50, 100)

# How far from t + Delta the nearest retained acquisition may sit. On this
# subset every gap is a multiple of 5 days (one orbit lattice) and every horizon
# is too, so +/-3 accepts exact matches ONLY and +/-2 gives the identical row
# set -- see horizon_tolerance_sensitivity, which measures this rather than
# asserting it. It is stated as a tolerance because the rule is
# "nearest retained frame within tolerance, else drop", not "exact or drop".
TOLERANCE_DAYS = 3

CONTEXT_FRAMES = 3       # single-image encoders: the 3 most recent frames <= t
MI_CONTEXT_FRAMES = 1    # the multi-image encoder: its single embedding at t

AGGREGATIONS = ("cube_mean", "cube_p90", "cell_mean")
TARGET_LEVEL = {"cube_mean": "frame", "cube_p90": "frame", "cell_mean": "cell"}

# A pooled vector cannot address one of 16 cells, so the resolution axis takes
# grid-cell context. The two cube-level targets take pooled context, which is
# what the spec names.
CONTEXT_LEVEL = {"cube_mean": "pooled", "cube_p90": "pooled",
                 "cell_mean": "grid_cell"}

FOLD_MODES = ("cube", "loco", "spatial_block")
PRIMARY_AGGREGATION = "cube_mean"

# The primary target carries all three fold modes at every horizon; the
# secondary aggregations are a RESOLUTION axis and run under the primary mode.
# Same restriction, and the same reason, as P2's PART_A_COMBOS.
AGGREGATION_MODES = {"cube_mean": FOLD_MODES,
                     "cube_p90": ("cube",),
                     "cell_mean": ("cube",)}

# p4_ceiling's estimator names. "linear" is Ridge with alpha = D on standardised
# features -- the spec's ridge -- fixed a priori, not searched.
ESTIMATORS = p4.ESTIMATORS                      # ("linear", "hgb", "mlp")
FORECAST_ESTIMATORS = ("linear", "mlp")         # ridge + the small 2-layer MLP
WEATHER_ONLY_ESTIMATORS = ("linear", "hgb")     # P4's weather-only family
# Controls run under EVERY estimator that appears in the table, so
# margin_over_control compares a model to a control fitted by the same learner
# rather than to one fitted by a different one.
CONTROL_ESTIMATORS = ("linear", "hgb", "mlp")

BASELINE_KINDS = ("persistence", "climatology_proxy", "weather_only",
                  "raw_features_weather", "raw_rgb_only_weather")
CONTROL_KINDS = ("horizon_only", "observation", "permutation")
MODEL_KINDS = ("forecast",) + BASELINE_KINDS + CONTROL_KINDS

FEATURE_SETS = ("embedding", "raw_rgb_only", "weather", "observation",
                "horizon", "none")
BAND_MATCHED_BASELINE = "raw_rgb_only"

WEATHER_FEATURE_SET = "weather_full8"

# The horizon control's design is [1, d, d^2] -- P2's degree, imported, for the
# reason P2 gives: a straight line in the horizon is monotone, and a monotone
# map would make the fitted coefficients irrelevant to a rank statistic.
HORIZON_CONTROL_DEGREE = p2.GAP_CONTROL_DEGREE

SEVERITY_BINS = p4.SEVERITY_BINS

# The label that travels with every climatology row. P4's Stage A label, plus
# the sentence HANDOFF_P3 section 5 requires on anything derived from it.
PROXY_CLIMATOLOGY_LABEL = (p4.STAGE_A_CLIMATOLOGY_LABEL
                           + " -- PROXY, NOT STAGE B: this is not H1's number")

# The physical floor a common-masked cube-mean NDVI has to clear to be
# vegetation rather than cloud. Bare soil is about 0.15 and dense summer canopy
# about 0.85; three frames of 1580 sit BELOW ZERO in midsummer at clear
# fractions of 0.59-0.63, and they set this table's R-squared. Used for
# REPORTING only -- print_target_outliers measures them, nothing drops them.
NDVI_PLAUSIBILITY_FLOOR = 0.15
GROWING_SEASON_DOY = (120, 270)

_MIN_TEST_ROWS = 8       # below this a fold's R-squared is noise; recorded NaN
_MIN_BIN_ROWS = 20       # below this a severity bin's pooled score is not shown

# ---------------------------------------------------------------------------
# Console hygiene: ONE run-log implementation, and it is P2's.
# ---------------------------------------------------------------------------

log = p2.log
close_run_log = p2.close_run_log


def run_log_path(name: str = "p3_run.log") -> str:
    return os.path.join(phase_dir(PHASE, "logs"), name)


def open_run_log(path: str | None = None, verbose: bool = True) -> str:
    """Open the per-row log. Detail goes there, headlines to stdout."""
    return p2.open_run_log(path or run_log_path(), verbose=verbose)


# ---------------------------------------------------------------------------
# The rows: (cube, t, Delta) -> a target frame and k context frames
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ForecastRows:
    """One horizon's rows. Every index is a POSITION into the manifest's rows.

    ``context_rows`` is (N, CONTEXT_FRAMES), OLDEST FIRST, and its last column
    is ``row_t`` by construction -- asserted, because a context that did not end
    at t would be a lag the table does not declare.

    THREE candidate "horizons" are carried. Only ``delta_days`` is one. The
    other two exist so ``assert_horizon_axis`` can prove on the real data that
    the day axis was used, which is a check a docstring cannot make and a test
    comparing the right column to itself would pass without measuring.
    """

    delta_days_nominal: int
    tolerance_days: int
    row_t: np.ndarray             # (N,)
    row_target: np.ndarray        # (N,)
    context_rows: np.ndarray      # (N, K)
    cube_id: np.ndarray           # (N,) str
    delta_days: np.ndarray        # (N,) float -- THE horizon. daily_axis_index.
    delta_acq_steps: np.ndarray   # (N,) int -- NOT the horizon. The join key.
    delta_frame_steps: np.ndarray  # (N,) int -- NOT the horizon. Frames between.
    within_cube_index: np.ndarray  # (N,) int

    def __post_init__(self):
        N = self.row_t.shape[0]
        for name in ("row_target", "cube_id", "delta_days", "delta_acq_steps",
                     "delta_frame_steps", "within_cube_index"):
            got = getattr(self, name).shape
            assert got == (N,), f"ForecastRows.{name} is {got}, expected {(N,)}"
        assert self.context_rows.shape[0] == N and self.context_rows.ndim == 2, (
            f"ForecastRows.context_rows is {self.context_rows.shape}, expected "
            f"({N}, K)"
        )
        assert N, (
            f"no forecast rows survive at Delta = {self.delta_days_nominal} d. "
            "That is a statement about the benchmark, not a bug -- but it must "
            "be reported, not silently produce an empty table."
        )
        assert (self.context_rows[:, -1] == self.row_t).all(), (
            "the last context frame is not t. The context is the k most recent "
            "retained frames AT OR BEFORE t, so its newest member is t itself."
        )
        assert (self.row_target != self.row_t).all(), "a target frame is t"

    @property
    def n_rows(self) -> int:
        return int(self.row_t.shape[0])

    @property
    def n_cubes(self) -> int:
        return int(np.unique(self.cube_id).size)

    @property
    def context_frames(self) -> int:
        return int(self.context_rows.shape[1])

    def manifest_rows(self) -> np.ndarray:
        """(N, K+1): every manifest row a forecast row touches. All in one cube,
        so a cube-grouped fold keeps a row whole -- asserted per fold, never
        assumed."""
        return np.column_stack([self.context_rows, self.row_target])


def horizon_index(manifest, delta_days: int,
                  tolerance_days: int = TOLERANCE_DAYS,
                  context_frames: int = CONTEXT_FRAMES,
                  verbose: bool = True) -> ForecastRows:
    """Every (t, target) row of one horizon, with its k context frames.

    THE LIVE AXIS CHECK RUNS FIRST, before a single horizon is computed:
    ``p2_deltas.assert_gap_axes_disagree`` on the real consecutive pairs of this
    manifest. It is imported, not re-derived, and its measured ratio (days per
    acquisition step) is then applied to P3's own rows by
    ``assert_horizon_axis``.

    Frames are ordered by ``original_axis_index``, never by manifest row
    position: a row order is whatever ``build_manifest`` produced, and a
    neighbour taken by position would silently pair the last frame of one cube
    with the first of the next if the manifest were re-sorted.

    The target search is forward only (a forecast cannot look backwards) and
    stays inside the cube. A row with no retained frame within ``tolerance_days``
    of ``t + delta_days`` is DROPPED.
    """
    for col in ("cube_id", "original_axis_index", "daily_axis_index", "timestamp"):
        assert col in manifest.columns, (
            f"manifest has no {col!r}; build it with "
            "encoders.manifest.build_manifest (and rebuild it if it predates "
            "the 2026-08-10 daily_axis_index fix)"
        )
    assert int(delta_days) >= 1, f"delta_days must be >= 1, got {delta_days}"
    assert int(context_frames) >= 1, f"context_frames must be >= 1"
    assert float(tolerance_days) >= 0, "tolerance_days must be >= 0"

    # --- the imported live check, on the real data, BEFORE any horizon --------
    axis = p2.assert_gap_axes_disagree(p2.pair_index(manifest, verbose=False),
                                       verbose=False)
    # -------------------------------------------------------------------------

    K = int(context_frames)
    ts_all = np.asarray(manifest["timestamp"].to_numpy(), dtype="datetime64[ns]")
    row_t, row_tg, ctx, cubes = [], [], [], []
    d_day, d_acq, d_frame, widx = [], [], [], []
    n_no_context, n_no_target = 0, 0

    for cube in sorted(manifest["cube_id"].unique().tolist()):
        pos = np.flatnonzero((manifest["cube_id"] == cube).to_numpy())
        pos = pos[np.argsort(manifest["original_axis_index"].to_numpy()[pos],
                             kind="stable")]
        oai = manifest["original_axis_index"].to_numpy()[pos].astype(int)
        dai = manifest["daily_axis_index"].to_numpy()[pos].astype(float)
        assert (np.diff(oai) > 0).all() and (np.diff(dai) > 0).all(), (
            f"{cube}: the two time axes are not both strictly increasing in "
            "acquisition order; the manifest has duplicate or inconsistent rows"
        )
        kept = 0
        for i in range(pos.size):
            if i < K - 1:
                n_no_context += 1
                continue
            want = dai[i] + float(delta_days)
            fwd = np.arange(i + 1, pos.size)
            if fwd.size == 0:
                n_no_target += 1
                continue
            j = int(fwd[np.argmin(np.abs(dai[fwd] - want))])
            if abs(dai[j] - want) > float(tolerance_days):
                n_no_target += 1
                continue
            row_t.append(pos[i]); row_tg.append(pos[j])
            ctx.append(pos[i - K + 1:i + 1])
            cubes.append(cube)
            d_day.append(float(dai[j] - dai[i]))
            d_acq.append(int(oai[j] - oai[i]))
            d_frame.append(int(j - i))
            widx.append(kept)
            kept += 1

    rows = ForecastRows(
        delta_days_nominal=int(delta_days), tolerance_days=int(tolerance_days),
        row_t=np.asarray(row_t, dtype=int),
        row_target=np.asarray(row_tg, dtype=int),
        context_rows=np.asarray(ctx, dtype=int).reshape(len(row_t), K),
        cube_id=np.asarray(cubes, dtype=object),
        delta_days=np.asarray(d_day, dtype=float),
        delta_acq_steps=np.asarray(d_acq, dtype=int),
        delta_frame_steps=np.asarray(d_frame, dtype=int),
        within_cube_index=np.asarray(widx, dtype=int))

    assert_horizon_axis(rows, manifest, axis, ts_all=ts_all, verbose=verbose)
    if verbose:
        print(f"[p3] Delta = {delta_days:>3} d (+/-{tolerance_days}): "
              f"{rows.n_rows} rows over {rows.n_cubes} cubes "
              f"| context {rows.context_frames} frames "
              f"| dropped {n_no_context} for <{K} prior frames, "
              f"{n_no_target} for no retained frame within tolerance "
              f"(window boundary) | realised horizon min "
              f"{rows.delta_days.min():.0f} median "
              f"{np.median(rows.delta_days):.0f} max {rows.delta_days.max():.0f} d")
    return rows


def assert_horizon_axis(rows: ForecastRows, manifest, axis: dict,
                        ts_all=None, verbose: bool = True) -> dict:
    """P3's horizons came off the DAY axis, checked against P2's live result.

    Three things, and none of them is a docstring claim:

    1. Every row's ``delta_days`` equals the difference of the two TIMESTAMPS.
       The daily axis is a contiguous day grid, so these are one quantity
       computed two ways from independent columns.
    2. No row has ``delta_days == delta_acq_steps``. Equality would mean the
       horizon was read off ``original_axis_index`` -- the embedding join key,
       which counts ACQUISITIONS -- and every horizon in this phase would be
       about five times too short.
    3. The observed days-per-acquisition-step on P3's rows matches the ratio
       ``p2_deltas.assert_gap_axes_disagree`` measured on the consecutive pairs
       of the SAME manifest, to within a factor of two. That is what ties this
       assertion to the live check rather than to a constant someone typed.
    """
    if ts_all is None:
        ts_all = np.asarray(manifest["timestamp"].to_numpy(),
                            dtype="datetime64[ns]")
    from_ts = ((ts_all[rows.row_target] - ts_all[rows.row_t])
               / np.timedelta64(1, "D")).astype(float)
    np.testing.assert_array_equal(
        rows.delta_days, from_ts,
        err_msg="delta_days disagrees with the horizon computed from the two "
                "TIMESTAMPS, which are an independent column. The daily axis "
                "is a contiguous day grid, so these are one quantity computed "
                "two ways. The usual cause is that the horizon was taken from "
                "original_axis_index -- the embedding join key, which counts "
                "ACQUISITIONS -- instead of daily_axis_index; the other is a "
                "manifest whose time columns came from different loads, which "
                "encoders.manifest.build_manifest rebuilds.")

    same = int((rows.delta_days == rows.delta_acq_steps).sum())
    assert same == 0, (
        f"{same}/{rows.n_rows} rows have delta_days == delta_acq_steps at "
        f"Delta = {rows.delta_days_nominal} d. The two axes disagree on every "
        "pair of this manifest (about 5 days per acquisition step), so equality "
        "means the horizon was computed from original_axis_index -- the "
        "embedding join key -- and every horizon here is wrong by a factor of "
        "about five. Same category error as the 2026-08-10 weather-join bug."
    )
    ratio = float(np.median(rows.delta_days) / np.median(rows.delta_acq_steps))
    live = float(axis["ratio_days_per_acq_step"])
    assert 0.5 * live <= ratio <= 2.0 * live, (
        f"P3's rows show {ratio:.2f} days per acquisition step but "
        f"assert_gap_axes_disagree measured {live:.2f} on the consecutive pairs "
        "of the same manifest. The two must describe one orbit lattice; a "
        "disagreement means P3's rows were built off a different axis than the "
        "one the live check inspected."
    )
    out = {"delta_days_nominal": rows.delta_days_nominal,
           "n_rows": rows.n_rows, "n_cubes": rows.n_cubes,
           "median_delta_days": float(np.median(rows.delta_days)),
           "median_delta_acq_steps": float(np.median(rows.delta_acq_steps)),
           "median_delta_frame_steps": float(np.median(rows.delta_frame_steps)),
           "ratio_days_per_acq_step": ratio,
           "ratio_from_live_check": live, "n_axes_equal": same}
    if verbose:
        print(f"[p3]   HORIZON AXIS CHECK on {rows.n_rows} real rows: "
              f"delta_days median {out['median_delta_days']:.0f} vs "
              f"delta_acq_steps median {out['median_delta_acq_steps']:.0f} vs "
              f"frames-between {out['median_delta_frame_steps']:.0f} -- "
              f"{ratio:.1f} days per acquisition step against the live check's "
              f"{live:.1f}, {same}/{rows.n_rows} rows where the axes agree")
    return out


def horizon_tolerance_sensitivity(manifest, horizons: Sequence[int] = HORIZONS,
                                  tolerances: Sequence[int] = (2, 3, 5, 7),
                                  context_frames: int = CONTEXT_FRAMES,
                                  verbose: bool = True):
    """How much work the tolerance is doing. Measured, not assumed.

    A tolerance is a free parameter, and a free parameter nobody measured is a
    place a result can hide. On this subset the acquisition lattice is 5 days
    and every nominal horizon is a multiple of 5, so +/-2 and +/-3 select the
    SAME rows (exact matches only) while +/-5 admits the neighbouring lattice
    point and changes the realised horizon by 5 days -- which at Delta = 5 is
    the entire horizon. That is why the chosen value is 3 and not 5.
    """
    import pandas as pd

    recs = []
    for tol in tolerances:
        for H in horizons:
            r = horizon_index(manifest, H, tolerance_days=tol,
                              context_frames=context_frames, verbose=False)
            recs.append({"tolerance_days": tol, "delta_days": H,
                         "n_rows": r.n_rows, "n_cubes": r.n_cubes,
                         "realised_min": float(r.delta_days.min()),
                         "realised_max": float(r.delta_days.max()),
                         "n_off_nominal": int((r.delta_days != H).sum())})
    df = pd.DataFrame(recs)
    if verbose:
        print("[p3] TOLERANCE SENSITIVITY -- how many rows the tolerance buys, "
              "and at what cost in horizon accuracy")
        print(f"[p3]   {'tol':>4} {'Delta':>6} {'rows':>6} {'cubes':>6} "
              f"{'realised d':>12} {'off-nominal':>12}")
        for r in df.itertuples():
            print(f"[p3]   {r.tolerance_days:>4d} {r.delta_days:>6d} "
                  f"{r.n_rows:>6d} {r.n_cubes:>6d} "
                  f"{r.realised_min:>5.0f}-{r.realised_max:<6.0f} "
                  f"{r.n_off_nominal:>12d}")
        chosen = df[df.tolerance_days == TOLERANCE_DAYS]
        print(f"[p3]   at the chosen +/-{TOLERANCE_DAYS} d, "
              f"{int(chosen.n_off_nominal.sum())} of {int(chosen.n_rows.sum())} "
              "rows are off the nominal horizon: on a 5-day orbit lattice a "
              "+/-3 d tolerance accepts EXACT matches only, so it selects rows "
              "and never distorts a horizon.")
    return df


def print_horizon_retention(rows_by_horizon: dict, manifest,
                            verbose: bool = True):
    """n retained per horizon, and where the losses come from. Never hidden.

    The shrinkage toward Delta = 100 d is a property of the benchmark -- a
    150-day cube whose retained frames span ~135 days cannot offer many 100-day
    pairs -- and it belongs in the record beside every score computed at that
    horizon, not in a footnote.
    """
    import pandas as pd

    n_manifest = len(manifest)
    per_cube = manifest.groupby("cube_id").size()
    doy = manifest["day_of_year"].to_numpy().astype(int)
    recs = []
    for H in sorted(rows_by_horizon):
        r = rows_by_horizon[H]
        span = (manifest.groupby("cube_id")["daily_axis_index"]
                .agg(lambda s: s.max() - s.min()))
        recs.append({
            "delta_days": H, "n_rows": r.n_rows, "n_cubes": r.n_cubes,
            "n_cubes_with_none": int(per_cube.size - r.n_cubes),
            "distinct_target_doy": int(np.unique(doy[r.row_target]).size),
            "rows_per_cube_median": float(np.median(
                np.bincount(np.unique(r.cube_id, return_inverse=True)[1]))),
            "frac_of_eligible": float(r.n_rows / max(1, n_manifest
                                                     - (CONTEXT_FRAMES - 1)
                                                     * per_cube.size)),
            "cube_day_span_median": float(np.median(span.to_numpy())),
        })
    df = pd.DataFrame(recs)
    if verbose:
        elig = n_manifest - (CONTEXT_FRAMES - 1) * per_cube.size
        print("[p3] ROWS RETAINED PER HORIZON (drop policy: no retained frame "
              f"within +/-{TOLERANCE_DAYS} d of t+Delta inside the SAME cube -> "
              "drop the row; never wrap, never extrapolate)")
        print(f"[p3]   {n_manifest} retained frames, {per_cube.size} cubes; "
              f"{(CONTEXT_FRAMES - 1) * per_cube.size} rows are ineligible for "
              f"lack of {CONTEXT_FRAMES} prior frames, leaving {elig} eligible t")
        print(f"[p3]   {'Delta':>6} {'rows':>6} {'cubes':>6} "
              f"{'cubes w/ none':>14} {'% of eligible t':>16} "
              f"{'distinct target DOY':>21}")
        for r in df.itertuples():
            print(f"[p3]   {r.delta_days:>6d} {r.n_rows:>6d} {r.n_cubes:>6d} "
                  f"{r.n_cubes_with_none:>14d} {r.frac_of_eligible:>15.1%} "
                  f"{r.distinct_target_doy:>21d}")
        print(f"[p3]   median cube covers {df.cube_day_span_median.iloc[0]:.0f} "
              "days of retained frames, so the 100-day horizon is the one the "
              "window boundary bites: it is REPORTED here and carried on every "
              "row of the CSV as n_retained / n_cubes.")
        print("[p3]   distinct target DOY is printed because the proxy "
              f"climatology is a {p4.CLIMATOLOGY_HARMONICS}-harmonic curve with "
              f"{2 * p4.CLIMATOLOGY_HARMONICS + 1} parameters: a horizon whose "
              "target frames land on few distinct days gives it little to fit, "
              "and the 100-day row is the thin one.")
    return df


def assert_retention_shrinks(retention, min_window_frac: float = 0.5,
                             verbose: bool = True) -> None:
    """Retention must fall toward the longest horizon -- or the run must EXPLAIN
    why it need not, from the cube window, in the same breath.

    Not a style rule. If the count did NOT fall at 100 days over 135-day cubes,
    the row set would be including pairs the cubes cannot support -- a wrap
    across cubes, an extrapolation past the window, or a tolerance wide enough
    to reuse a much nearer frame -- and every one of those is silent in the
    score.

    The condition is checked WHERE IT BINDS. A horizon that is a small fraction
    of the cube's covered window loses nothing at the boundary and has no reason
    to shrink; requiring it to would be a false invariant that a future run with
    shorter horizons would have to work around. So the requirement applies when
    the longest horizon reaches at least ``min_window_frac`` of the median cube
    day-span (0.74 for 100 days over the 135-day median here), and otherwise the
    check states plainly that it did not apply and why. That is the "explained
    if it does not" half of the rule, and it is data, not prose.
    """
    df = retention.sort_values("delta_days")
    n = df.n_rows.to_numpy()
    longest, first = df.iloc[-1], df.iloc[0]
    span = float(df.cube_day_span_median.iloc[0])
    frac = float(longest.delta_days) / span if span > 0 else float("inf")
    binding = frac >= min_window_frac

    assert longest.n_cubes <= first.n_cubes, (
        f"{longest.delta_days} d reaches {longest.n_cubes} cubes against "
        f"{first.n_cubes} at {first.delta_days} d; a longer horizon cannot "
        "reach more cubes than a shorter one under the same drop policy."
    )
    if binding:
        assert longest.n_rows < first.n_rows, (
            f"the longest horizon ({longest.delta_days} d) retains "
            f"{longest.n_rows} rows, not fewer than the shortest "
            f"({first.delta_days} d, {first.n_rows}) -- and it reaches "
            f"{frac:.0%} of the median cube's {span:.0f}-day covered window, so "
            "the boundary MUST bite. No loss means the target search wrapped "
            "across cubes, extrapolated past the window, or accepted a frame "
            "nowhere near t+Delta."
        )
    if verbose:
        non_mono = int((np.diff(n) > 0).sum())
        if binding:
            drop = 1.0 - longest.n_rows / first.n_rows
            print(f"[p3]   retention SHRINKS as required: {first.n_rows} rows "
                  f"at {first.delta_days} d -> {longest.n_rows} at "
                  f"{longest.delta_days} d ({drop:.0%} lost to the window "
                  f"boundary), {longest.n_cubes}/{first.n_cubes} cubes still "
                  f"contribute. The longest horizon is {frac:.0%} of the "
                  f"median cube's {span:.0f}-day window, so the requirement "
                  "binds.")
        else:
            print(f"[p3]   retention was NOT required to shrink and this is "
                  f"why: the longest horizon ({longest.delta_days} d) is only "
                  f"{frac:.0%} of the median cube's {span:.0f}-day covered "
                  f"window, below the {min_window_frac:.0%} at which the "
                  f"boundary starts to bite. Measured: {first.n_rows} rows at "
                  f"{first.delta_days} d, {longest.n_rows} at "
                  f"{longest.delta_days} d.")
        print(f"[p3]   {non_mono} intermediate step(s) rise rather than fall -- "
              "retention is not required to be monotone in between, because a "
              "horizon lands on the acquisition lattice or it does not.")


# ---------------------------------------------------------------------------
# Common-masked LEVELS -- pinned to P2's common-masked DELTA
# ---------------------------------------------------------------------------

def assert_climatology_identifiable(manifest, rows_by_horizon: dict,
                                    fold_modes: Sequence[str] = FOLD_MODES,
                                    k: int = 5, verbose: bool = True) -> dict:
    """The proxy climatology can be fitted in the THINNEST training fold.

    ``p4_ceiling.doy_climatology_within_fold`` refuses to fit a
    ``CLIMATOLOGY_HARMONICS``-harmonic curve on fewer distinct days of year than
    it has parameters, and it is right to: such a curve interpolates its
    training days and the "climatology" becomes noise. That refusal would
    otherwise fire deep inside a fold, an hour into a run, so it is checked HERE
    against every (horizon, fold mode) the run will use -- and the worst-case
    count is PRINTED even when it passes, because 14 distinct days for 9
    parameters passes and is still thin, and a reader of the 100-day rows should
    know which of them is fitted on the least data in the table.

    The full-data reference fit is checked too. It defines the severity bins
    (``p4_ceiling.severity_reference_anomaly``, a reporting axis only), and it
    is over the rows of ONE horizon, so it is thinner than any fold's fit is
    wide.
    """
    doy = manifest["day_of_year"].to_numpy().astype(int)
    n_par = 2 * p4.CLIMATOLOGY_HARMONICS + 1
    out, worst = {}, None
    for H in sorted(rows_by_horizon):
        r = rows_by_horizon[H]
        touched = r.manifest_rows()
        out[("all_rows", H)] = int(np.unique(doy[r.row_target]).size)
        for mode in fold_modes:
            counts = []
            for tr, _ in p4.outer_folds(manifest, mode, k=k, verbose=False):
                sel = np.isin(touched, np.asarray(tr, dtype=int)).all(axis=1)
                if not sel.any():
                    continue
                counts.append(int(np.unique(doy[r.row_target[sel]]).size))
            out[(mode, H)] = min(counts) if counts else 0
    worst = min(out.items(), key=lambda kv: kv[1])
    (bad_mode, bad_H), bad_n = worst
    assert bad_n >= n_par, (
        f"the thinnest day-of-year fit ({bad_mode}, Delta = {bad_H} d) covers "
        f"only {bad_n} distinct target days of year, and the proxy climatology "
        f"has {n_par} parameters ({p4.CLIMATOLOGY_HARMONICS} harmonics). It "
        "would interpolate them exactly and the climatology baseline -- and the "
        "severity bins derived from the same curve -- would be noise. This is a "
        "SAMPLE-SIZE limit, not a bug: the acquisition lattice gives about 41 "
        "distinct days in total and a long horizon reaches only the late part "
        "of it. Run more cubes, drop the longest horizon, or lower "
        "CLIMATOLOGY_HARMONICS and record that on every row."
    )
    if verbose:
        print(f"[p3] proxy climatology is identifiable everywhere it is fitted: "
              f"the thinnest case is {bad_mode} at Delta = {bad_H} d with "
              f"{bad_n} distinct target days of year against {n_par} "
              f"parameters. {bad_n} for {n_par} is IDENTIFIED but THIN -- the "
              "longest-horizon climatology row is fitted on the least data in "
              "the table and should be read that way.")
    return out


def common_masked_levels(ndvi_a, ndvi_b, mask_a, mask_b, grid: int = GRID) -> dict:
    """The two frames' aggregates over the pixels valid in BOTH. Never one mask.

    ``p2_deltas.common_masked_delta`` owns the intersection and the change and
    is called here UNMODIFIED. What this adds is the two LEVELS on that same
    intersection, because a forecast needs a level and a delta probe needed only
    a difference. The addition is then PINNED: for all three aggregations

        level_b - level_a == the delta p2 returned

    is asserted to 1e-12 on every pair. That identity is what makes the
    persistence baseline exactly right rather than approximately right -- its
    residual is p2's common-masked change, by construction -- and it makes any
    future drift between the two impossible rather than merely unlikely.
    """
    d = p2.common_masked_delta(ndvi_a, ndvi_b, mask_a, mask_b, grid=grid)

    a = np.asarray(ndvi_a, dtype=np.float64)
    b = np.asarray(ndvi_b, dtype=np.float64)
    both = np.asarray(mask_a) & np.asarray(mask_b)     # the ONLY mask used
    H, W = a.shape
    ch, cw = H // grid, W // grid
    n_both = int(both.sum())

    if n_both == 0:
        nan16 = np.full(grid * grid, np.nan)
        out = {"cube_mean": (np.nan, np.nan), "cube_p90": (np.nan, np.nan),
               "cell_mean": (nan16, nan16)}
    else:
        va, vb = a[both], b[both]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            cells_a = np.nanmean(np.where(both, a, np.nan)
                                 .reshape(grid, ch, grid, cw), axis=(1, 3)).reshape(-1)
            cells_b = np.nanmean(np.where(both, b, np.nan)
                                 .reshape(grid, ch, grid, cw), axis=(1, 3)).reshape(-1)
        out = {"cube_mean": (float(va.mean()), float(vb.mean())),
               "cube_p90": (float(np.percentile(va, 90)),
                            float(np.percentile(vb, 90))),
               "cell_mean": (cells_a, cells_b)}

    # --- THE PIN: the levels must reproduce p2's delta, exactly ---------------
    for agg, key in (("cube_mean", "d_cube_mean"), ("cube_p90", "d_cube_p90"),
                     ("cell_mean", "d_cell_mean")):
        lo, hi = out[agg]
        got = np.asarray(hi, dtype=float) - np.asarray(lo, dtype=float)
        want = np.asarray(d[key], dtype=float)
        ok = np.isfinite(got) & np.isfinite(want)
        assert np.allclose(got[ok], want[ok], rtol=0, atol=1e-12), (
            f"the common-masked LEVELS do not reproduce "
            f"p2_deltas.common_masked_delta's {key}: max |diff| "
            f"{np.abs(got[ok] - want[ok]).max() if ok.any() else float('nan'):.3g}. "
            "These two must be one computation over one intersection; if they "
            "are not, the persistence baseline is not the common-masked change "
            "and P2's differencing trap is back."
        )
        assert np.array_equal(np.isfinite(got), np.isfinite(want)), (
            f"the levels and p2's {key} disagree about which rows are defined"
        )
    # -------------------------------------------------------------------------

    out["n_common_px"] = d["n_common_px"]
    out["frac_common_px"] = d["frac_common_px"]
    out["cell_valid"] = d["cell_valid"]
    out["n_common_px_cell"] = d["n_common_px_cell"]
    return out


def cube_ndvi_and_masks(sample, masks, min_clear: float = MIN_CLEAR_FRACTION):
    """(ndvi (T,H,W), mask (T,H,W) bool) for ONE cube, with P2's three checks.

    P3's pairs are not consecutive, so ``p2_deltas.cube_common_masked_deltas``
    -- which walks t -> t+1 -- cannot serve; what P3 needs is the validated
    stacks. The three consistency assertions below are the SAME three that
    function makes before it differences anything, and they are here for the
    same reason: if the cached mask belongs to a different frame selection, or
    disagrees with the canonical NDVI's finiteness, then "the pixels valid in
    both frames" is not the set every target in this phase is computed over, and
    nothing downstream could detect it. The intersection arithmetic itself is
    NOT duplicated -- it is ``common_masked_levels``, which calls p2.
    """
    from data.loader import CubeSample, cube_ndvi

    sel = select_clear_frames(sample.values, sample.timestamps, sample.mask,
                              min_clear=min_clear, verbose=False)
    name = os.path.basename(sample.path)
    assert np.array_equal(np.asarray(masks.kept_idx).astype(int),
                          np.asarray(sel.kept_idx).astype(int)), (
        f"{name}: the cached mask's kept_idx != this cube's frame selection. "
        "The mask cache was written under a different mask definition or "
        "min_clear; re-cache it."
    )
    assert (np.asarray(masks.timestamps) == np.asarray(sel.timestamps)).all(), (
        f"{name}: cached mask timestamps differ from the cube's retained frames"
    )
    kept = CubeSample(values=sel.values, timestamps=sel.timestamps,
                      mask=sel.mask, path=sample.path, bands=sample.bands)
    nd = cube_ndvi(kept)
    assert masks.mask.shape == nd.shape, (
        f"{name}: cached mask {masks.mask.shape} != NDVI {nd.shape}"
    )
    assert (masks.mask == np.isfinite(nd)).all(), (
        f"{name}: the cached per-pixel mask does not equal the finiteness of "
        "the canonical NDVI. One of them is stale."
    )
    return nd, np.asarray(masks.mask), np.asarray(sel.kept_idx).astype(int)


def build_forecast_targets(rows_by_horizon: dict, manifest, cube_dir: str,
                           mask_dir: str | None = None,
                           min_clear: float = MIN_CLEAR_FRACTION,
                           verbose: bool = True) -> dict:
    """Common-masked target and persistence levels for every row of every horizon.

    Each cube is loaded ONCE and every horizon's pairs in it are computed from
    the same NDVI stack, so the four horizons are guaranteed to be four views of
    one set of pixels rather than four independent reads that could diverge.
    """
    from data.loader import load_cube

    mask_dir = mask_dir or phase_dir("phase1_2", "masks")
    horizons = sorted(rows_by_horizon)
    # manifest row -> position within its cube's acquisition-ordered frames
    within = np.full(len(manifest), -1, dtype=int)
    for cube in sorted(manifest["cube_id"].unique().tolist()):
        pos = np.flatnonzero((manifest["cube_id"] == cube).to_numpy())
        pos = pos[np.argsort(manifest["original_axis_index"].to_numpy()[pos],
                             kind="stable")]
        within[pos] = np.arange(pos.size)
    assert (within >= 0).all(), "a manifest row belongs to no cube"

    out = {H: {"cube_mean": {}, "cube_p90": {}, "cell_mean": {},
               "n_common_px": np.full(rows_by_horizon[H].n_rows, np.nan),
               "frac_common_px": np.full(rows_by_horizon[H].n_rows, np.nan),
               "cell_valid": np.zeros((rows_by_horizon[H].n_rows, GRID_CELLS),
                                      dtype=bool)}
           for H in horizons}
    for H in horizons:
        N = rows_by_horizon[H].n_rows
        for agg in ("cube_mean", "cube_p90"):
            out[H][agg] = {"y": np.full(N, np.nan),
                           "persistence": np.full(N, np.nan)}
        out[H]["cell_mean"] = {"y": np.full((N, GRID_CELLS), np.nan),
                               "persistence": np.full((N, GRID_CELLS), np.nan)}

    for cube in sorted(manifest["cube_id"].unique().tolist()):
        mp = os.path.join(mask_dir, f"{os.path.splitext(cube)[0]}__masks.npz")
        assert os.path.exists(mp), (
            f"no cached per-pixel mask for {cube!r} at {mp}. Common-masking is "
            "mandatory here and cannot be approximated from clear_frac."
        )
        sample = load_cube(os.path.join(cube_dir, cube), verbose=False)
        nd, mk, _ = cube_ndvi_and_masks(sample, load_masks(mp), min_clear)
        for H in horizons:
            r = rows_by_horizon[H]
            sel = np.flatnonzero(r.cube_id == cube)
            for n in sel:
                ia, ib = within[r.row_t[n]], within[r.row_target[n]]
                lev = common_masked_levels(nd[ia], nd[ib], mk[ia], mk[ib])
                for agg in AGGREGATIONS:
                    lo, hi = lev[agg]
                    out[H][agg]["persistence"][n] = lo
                    out[H][agg]["y"][n] = hi
                out[H]["n_common_px"][n] = lev["n_common_px"]
                out[H]["frac_common_px"][n] = lev["frac_common_px"]
                out[H]["cell_valid"][n] = lev["cell_valid"]
                log(f"[p3] row {cube} H={H} t={ia}->{ib} | common px "
                    f"{lev['n_common_px']:>6} ({lev['frac_common_px']:.3f}) | "
                    f"NDVI(t) {lev['cube_mean'][0]:+.5f} -> "
                    f"NDVI(t+H) {lev['cube_mean'][1]:+.5f}")

    for H in horizons:
        N = rows_by_horizon[H].n_rows
        assert out[H]["cube_mean"]["y"].shape == (N,)
        assert out[H]["cell_mean"]["y"].shape == (N, GRID_CELLS)
        dead = int((out[H]["n_common_px"] == 0).sum())
        assert np.isfinite(out[H]["cube_mean"]["y"][out[H]["n_common_px"] > 0]).all()
        if verbose:
            print(f"[p3] targets Delta={H:>3} d: y {out[H]['cube_mean']['y'].shape} "
                  f"| cell {out[H]['cell_mean']['y'].shape} | {dead}/{N} rows "
                  f"with ZERO common pixels | "
                  f"{int((~out[H]['cell_valid']).sum())}/"
                  f"{out[H]['cell_valid'].size} cells with none")
    return out


def print_target_outliers(rows_by_horizon: dict, targets: dict, manifest,
                          floor: float = NDVI_PLAUSIBILITY_FLOOR,
                          season: tuple = GROWING_SEASON_DOY,
                          verbose: bool = True):
    """WHICH ROWS SET THIS TABLE'S R-SQUARED. Measured before anything is fitted.

    An R-squared is a sum of SQUARED errors, so a handful of rows with a huge
    error can decide it while the median row is predicted almost exactly. On
    this subset that is not hypothetical, and the cause is upstream of P3:

    **Three frames of 1580 carry a cube-mean NDVI below zero in midsummer**
    (days of year 177-202, clear fractions 0.59-0.63). Bare soil is about 0.15
    and dense summer canopy about 0.85; NDVI near zero in July over Allgau
    farmland is cloud, not vegetation. Both filters passed them: the
    clear-fraction rule kept the frames (they are 59-63% "clear") and the
    per-pixel mask marked the surviving pixels valid, so the contamination is
    inside the pixels both this phase and P2 and P4 treat as good data.

    Those three frames produce about five forecast rows whose persistence error
    is 0.66-0.86 NDVI, and **the worst 1% of rows carry roughly 71% of the
    persistence sum of squares at Delta = 5 d.** Any R-squared computed over
    them is a statement about three frames.

    Nothing is dropped here. The rows are REPORTED, the concentration is
    measured and carried onto every row of the CSV as ``sse_share_top1pct``, and
    ``medae_pooled`` -- a median, immune to five rows -- sits beside every mean.
    Dropping them would be a new filter that P2 and P4 do not apply, which would
    make P3's row set incomparable to theirs; it is recorded as an open item
    instead, because the right fix is a physical-plausibility screen at frame
    level in the shared target code, not a private one here.
    """
    import pandas as pd

    doy = manifest["day_of_year"].to_numpy().astype(int)
    cf = manifest["clear_frac"].to_numpy().astype(float)
    cube = manifest["cube_id"].to_numpy().astype(str)
    recs, frames = [], {}
    for H in sorted(rows_by_horizon):
        r = rows_by_horizon[H]
        y = np.asarray(targets[H]["cube_mean"]["y"], dtype=float)
        p = np.asarray(targets[H]["cube_mean"]["persistence"], dtype=float)
        d = y - p
        s = np.sort(d ** 2)[::-1]
        tot = float(s.sum())
        n1 = max(1, int(round(0.01 * s.size)))
        n5 = max(1, int(round(0.05 * s.size)))
        low = (y < floor) | (p < floor)
        recs.append({"delta_days": H, "n_rows": int(y.size),
                     "persistence_sse": tot,
                     "sse_share_top1pct": float(s[:n1].sum() / tot) if tot else np.nan,
                     "sse_share_top5pct": float(s[:n5].sum() / tot) if tot else np.nan,
                     "n_rows_below_floor": int(low.sum()),
                     "medae_persistence": float(np.median(np.abs(d))),
                     "rmse_persistence": float(np.sqrt((d ** 2).mean()))})
        for side, rr in (("t", r.row_t), ("target", r.row_target)):
            arr = p if side == "t" else y
            for i in np.flatnonzero(arr < floor):
                frames[int(rr[i])] = float(arr[i])
    df = pd.DataFrame(recs)
    lo, hi = season
    if verbose:
        print(f"[p3] WHICH ROWS SET THIS TABLE'S R-SQUARED (physical floor: "
              f"common-masked cube-mean NDVI < {floor} in the growing season, "
              f"DOY {lo}-{hi}, is cloud rather than vegetation -- bare soil is "
              "about 0.15)")
        print(f"[p3]   {'Delta':>6} {'rows':>6} {'<floor':>7} "
              f"{'medAE(pers)':>12} {'RMSE(pers)':>11} {'SSE from top 1%':>16} "
              f"{'top 5%':>8}")
        for r in df.itertuples():
            print(f"[p3]   {r.delta_days:>6d} {r.n_rows:>6d} "
                  f"{r.n_rows_below_floor:>7d} {r.medae_persistence:>12.4f} "
                  f"{r.rmse_persistence:>11.4f} {r.sse_share_top1pct:>15.1%} "
                  f"{r.sse_share_top5pct:>7.1%}")
        if frames:
            print(f"[p3]   {len(frames)} distinct FRAME(S) below the floor, and "
                  "they are the reason the concentration above is what it is:")
            for row, val in sorted(frames.items(), key=lambda kv: kv[1]):
                if lo <= doy[row] <= hi:
                    print(f"[p3]     {cube[row][:46]:<46} DOY {doy[row]:>3} "
                          f"clear_frac {cf[row]:.3f} -> common-masked NDVI "
                          f"{val:+.4f}   <- cloud that BOTH filters passed")
        print("[p3]   NOTHING IS DROPPED. The concentration travels on every "
              "row as sse_share_top1pct and a MEDIAN absolute error sits beside "
              "every mean, because a median cannot be set by five rows. The "
              "right fix is a plausibility screen in the SHARED frame-target "
              "code -- P2's and P4's numbers rest on the same three frames -- "
              "not a private filter here that would make P3's row set "
              "incomparable to theirs.")
    return df


def summarise_pixel_survival_by_horizon(rows_by_horizon: dict, targets: dict,
                                        verbose: bool = True):
    """Common-masked survival PER HORIZON. Measured, never modelled.

    P2 established that survival is not monotone in gap length -- a long gap
    exists BECAUSE the frames between it were cloudy, which leaves the two
    surviving endpoints unusually clear -- so it must not be predicted from the
    horizon. It is measured here at each of P3's four horizons for the same
    reason: a collapse at one of them would be a finding about what this
    benchmark supports at that horizon, not a reason to drop rows.
    """
    import pandas as pd

    recs = []
    for H in sorted(rows_by_horizon):
        f = targets[H]["frac_common_px"]
        n = targets[H]["n_common_px"]
        recs.append({"delta_days": H, "n_rows": int(f.size),
                     "px_min": float(np.nanmin(n)),
                     "px_median": float(np.nanmedian(n)),
                     "frac_min": float(np.nanmin(f)),
                     "frac_median": float(np.nanmedian(f)),
                     "n_zero_common": int((n == 0).sum()),
                     "cells_min": int(targets[H]["cell_valid"].sum(axis=1).min()),
                     "cells_median": float(np.median(
                         targets[H]["cell_valid"].sum(axis=1)))})
    df = pd.DataFrame(recs)
    if verbose:
        print("[p3] COMMON-MASKED PIXEL SURVIVAL BY HORIZON "
              f"(pixels valid in BOTH frames; {GRID_CELLS} grid cells)")
        print(f"[p3]   {'Delta':>6} {'rows':>6} {'px_min':>8} {'px_med':>8} "
              f"{'frac_min':>9} {'frac_med':>9} {'cells_min':>10} {'zero':>5}")
        for r in df.itertuples():
            print(f"[p3]   {r.delta_days:>6d} {r.n_rows:>6d} {r.px_min:>8.0f} "
                  f"{r.px_median:>8.0f} {r.frac_min:>9.3f} "
                  f"{r.frac_median:>9.3f} {r.cells_min:>10d} "
                  f"{r.n_zero_common:>5d}")
        worst = df.loc[df.frac_median.idxmin()]
        print(f"[p3]   {int(df.n_zero_common.sum())} of {int(df.n_rows.sum())} "
              f"rows have zero common pixels; the weakest horizon is "
              f"{worst.delta_days:.0f} d at median {worst.frac_median:.3f} "
              "surviving. Survival is measured per horizon, never modelled as a "
              "function of it -- P2 showed it is not monotone.")
    return df


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def context_frames_for(encoder: str) -> int:
    """3 for a single-image encoder, 1 for the multi-image one."""
    return MI_CONTEXT_FRAMES if encoder == MI_ENCODER else CONTEXT_FRAMES


def encoder_views(encoders: Sequence[str] = ENCODER_ORDER) -> list:
    """(encoder, feature_set, model_kind) triples.

    The band-matched view exists only for the not-a-network baseline, because it
    is a COLUMN SLICE of that baseline's own 35-column array -- no re-encode, and
    ``RAW_BAND_COLUMNS`` is derived from the baseline's feature names and
    asserted to partition them, so it cannot silently point at the wrong bands.
    The two ``raw_features`` views carry their own model_kind so the five
    mandatory baselines are checkable by name in the table.
    """
    views = [(e, "embedding",
              "raw_features_weather" if e == BASELINE_ENCODER else "forecast")
             for e in encoders]
    if BASELINE_ENCODER in encoders:
        views.append((BASELINE_ENCODER, BAND_MATCHED_BASELINE,
                      "raw_rgb_only_weather"))
    return views


def _columns_for(feature_set: str, D: int) -> np.ndarray:
    if feature_set == "embedding":
        return np.arange(D)
    cols = RAW_BAND_COLUMNS[feature_set]
    assert D == 35, (
        f"{feature_set} is a column slice of the raw_features array (D=35), "
        f"but this array has D={D}. It is defined for raw_features only."
    )
    return cols


def context_block(arrays: dict, rows: ForecastRows, encoder: str, level: str,
                  feature_set: str = "embedding",
                  n_frames: int | None = None,
                  row_of=None, cell_idx=None) -> tuple:
    """(X (n_feature_rows, k*D), names) -- the context at t, OLDEST FIRST.

    ``row_of`` / ``cell_idx`` come from the target view and are what make the
    design line up with the target. At cube level they are the identity and the
    16-cell axis does not exist. At CELL level they are NOT the full 16-to-1
    expansion: a cell with no surviving common pixel between t and t+Delta is
    not an observation, so it is absent from the target and must be absent from
    the design too. Building the full ``N * 16`` block and hoping it matched is
    exactly the mismatch ``FeatureBlock`` exists to prevent one level up -- and
    here it is not even silent, because the two lengths differ (8288 against
    8149 at Delta = 5 d on this subset).

    ``n_frames`` defaults to ``context_frames_for(encoder)`` and exists ONLY so
    a test can try to hand the multi-image encoder a 3-frame stack and watch it
    be refused. It is never passed by the run.

    THE MULTI-IMAGE REFUSAL IS LOUD AND IT IS HERE, not in a docstring. A
    ``satlas_s2_swinb_mi_rgb`` embedding at t already max-pools up to 8 preceding
    retained frames, so three consecutive MI embeddings share most of their
    source frames; stacking them would present one lookback window three times
    and count it three times. The single embedding at t IS the context.
    """
    assert level in ("pooled", "grid_cell"), f"level {level!r}"
    k = context_frames_for(encoder) if n_frames is None else int(n_frames)
    assert k >= 1, f"n_frames must be >= 1, got {k}"
    assert not (encoder == MI_ENCODER and k != MI_CONTEXT_FRAMES), (
        f"REFUSED: a {k}-frame context was requested for {MI_ENCODER}, which "
        f"takes {MI_CONTEXT_FRAMES}. Its embedding at t already aggregates up "
        "to 8 preceding RETAINED frames (a 0-105 day lookback on this subset), "
        "so stacking k=3 of them would double-count the lookback: consecutive "
        "MI embeddings share up to 7 of their 8 source frames, and the stack "
        "would be three overlapping summaries of one window presented as three "
        "observations. Use context_frames_for(encoder), which returns "
        f"{MI_CONTEXT_FRAMES} here and {CONTEXT_FRAMES} for every single-image "
        "encoder, and flag the row si_comparable=False."
    )
    assert k <= rows.context_frames, (
        f"{k} context frames requested but the row set carries only "
        f"{rows.context_frames}; rebuild the rows with context_frames={k}"
    )
    # The k MOST RECENT of the available context frames, still oldest first.
    ctx = rows.context_rows[:, rows.context_frames - k:]
    assert (ctx[:, -1] == rows.row_t).all(), "the newest context frame is not t"

    N = rows.n_rows
    row_of = np.arange(N) if row_of is None else np.asarray(row_of, dtype=int)
    assert row_of.ndim == 1 and row_of.size, "empty feature-row map"
    assert row_of.max() < N, (
        f"row_of reaches forecast row {row_of.max()} but only {N} exist"
    )
    if level == "pooled":
        assert cell_idx is None or (np.asarray(cell_idx) < 0).all(), (
            "a pooled context was asked for cell indices"
        )
        src = np.asarray(arrays["pooled"], dtype=np.float64)
        cols = _columns_for(feature_set, src.shape[1])
        blocks = [src[ctx[row_of, j]][:, cols] for j in range(k)]
        X = np.concatenate(blocks, axis=1)
    else:
        assert cell_idx is not None, (
            "a grid_cell context needs the target view's cell_idx: cells with "
            "no surviving common pixel are absent from the target and must be "
            "absent from the design"
        )
        cell_idx = np.asarray(cell_idx, dtype=int)
        assert cell_idx.shape == row_of.shape, (cell_idx.shape, row_of.shape)
        assert (cell_idx >= 0).all() and (cell_idx < GRID_CELLS).all()
        g = np.asarray(arrays["grid"], dtype=np.float64)
        assert g.shape[1] == GRID_CELLS, g.shape
        cols = _columns_for(feature_set, g.shape[-1])
        # each feature row sees the SAME grid cell of its k context frames
        blocks = [g[ctx[row_of, j], cell_idx][:, cols] for j in range(k)]
        X = np.concatenate(blocks, axis=1)
    assert X.shape == (row_of.size, k * cols.size), X.shape
    X = np.ascontiguousarray(X, dtype=np.float64)
    assert np.isfinite(X).all(), f"{encoder}/{feature_set}/{level}: non-finite"
    names = tuple(f"{feature_set}_t-{k - 1 - j}_c{c}"
                  for j in range(k) for c in range(cols.size))
    assert len(names) == X.shape[1]
    return X, names


def horizon_weather(manifest, cube_dir: str, rows: ForecastRows,
                    feature_set: str = WEATHER_FEATURE_SET,
                    verbose: bool = True) -> p4.FeatureSource:
    """Weather over [t, t+Delta], one vector per forecast row.

    ``p4_ceiling.cube_weather_windows`` is called UNMODIFIED. A backward window
    of ``Delta + 1`` days ending at the TARGET frame is exactly the forward
    window [t, t+Delta], so P4's join -- daily axis, located by TIMESTAMP, never
    by ``original_axis_index`` -- is reused rather than rebuilt. That the window
    really starts at t is not asserted by inspection: ``days_available`` is
    returned by that function and is asserted equal to ``Delta + 1`` on every
    row, which can only hold if the window's left edge landed on t.

    Rows are grouped by their REALISED horizon before the call, because the span
    is a per-call constant. On this subset every realised horizon equals its
    nominal one (see ``horizon_tolerance_sensitivity``), so there is one group;
    the loop exists so a looser tolerance could not silently attach the wrong
    window length to a row.
    """
    variables = p4.FEATURE_SET_VARS[feature_set]
    N = rows.n_rows
    ts = np.asarray(manifest["timestamp"].to_numpy(), dtype="datetime64[ns]")
    X = None
    names_canon = None
    filled = np.zeros(N, dtype=bool)

    for cube in sorted(np.unique(rows.cube_id).tolist()):
        in_cube = np.flatnonzero(rows.cube_id == cube)
        path = os.path.join(cube_dir, str(cube))
        for delta in np.unique(rows.delta_days[in_cube]):
            sel = in_cube[rows.delta_days[in_cube] == delta]
            span = int(round(float(delta))) + 1
            W, nm, avail = p4.cube_weather_windows(
                path, ts[rows.row_target[sel]], variables=variables,
                specs=((span, 0),), verbose=False)
            assert (avail[:, 0] == span).all(), (
                f"{cube}: the [t, t+{int(delta)}] window is "
                f"{sorted(set(avail[:, 0].tolist()))} days long, not {span}. A "
                "short window means the daily axis was clipped at the start of "
                "the cube -- i.e. the window did NOT begin at t -- and the "
                "weather attached to that row covers a different period than "
                "the horizon it is labelled with."
            )
            canon = tuple(n.replace(f"_{span}d", "_horizon") for n in nm)
            if names_canon is None:
                names_canon, X = canon, np.full((N, len(canon)), np.nan)
            assert canon == names_canon, (
                f"{cube}: weather feature names differ between window lengths "
                f"({canon[:2]} vs {names_canon[:2]})"
            )
            X[sel] = W
            filled[sel] = True

    assert filled.all(), f"{int((~filled).sum())} row(s) got no weather window"
    assert np.isfinite(X).all(), "a windowed weather feature is non-finite"
    src = p4.FeatureSource(name=f"{feature_set}/horizon", values=X,
                           names=names_canon, frame_level=True, permutable=True)
    if verbose:
        print(f"[p3]   weather Delta={rows.delta_days_nominal:>3} d: {X.shape} "
              f"= {len(names_canon)} (variable, aggregation) pairs over "
              f"{len(variables)} E-OBS variables, on ONE window spanning "
              f"[t, t+{rows.delta_days_nominal}d] of the DAILY axis. "
              "p4's aggregations carry the EXTREMES as well as the means "
              "(wettest day, wet-day fraction, min tn, max tx, max wind, max "
              "radiation), so a 100-day window is not reduced to its average. "
              "This is the weather that OCCURRED, not a forecast of it.")
    return src


def manifest_clear_fractions(manifest, per_cube: dict) -> tuple:
    """(clear_frac (n,), window_span_days (n,), grid_clear_frac (n, 16)).

    The first two come from ``p4_ceiling.observation_source``, which recomputes
    the lookback from the manifest timestamps through
    ``encoders.pipeline.window_span_days`` -- so no embedding is read to build a
    control. The per-cell fractions come from ``p4_ceiling.cube_frame_targets``
    and are put back into MANIFEST ROW ORDER here, once.
    """
    obs = p4.observation_source(manifest, per_cube, "frame", verbose=False)
    assert obs.names == ("clear_frac", "window_span_days"), obs.names
    v = np.asarray(obs.values, dtype=float)
    order = np.concatenate([np.flatnonzero((manifest["cube_id"] == c).to_numpy())
                            for c in per_cube["__order__"]])
    back = np.argsort(order, kind="stable")
    assert order[back].tolist() == list(range(len(manifest))), (
        "the per-cube clear fractions are not a permutation of the manifest rows"
    )
    gcf = np.concatenate([per_cube[c]["grid_clear_frac"]
                          for c in per_cube["__order__"]])[back]
    assert gcf.shape == (len(manifest), GRID_CELLS), gcf.shape
    return v[:, 0], v[:, 1], gcf


def observation_triple(rows: ForecastRows, clear_frac, window_span_days,
                       grid_clear_frac, level: str, row_of, cell_idx,
                       verbose: bool = False) -> p4.FeatureSource:
    """P1's degenerate control, extended by the one variable a forecast adds.

    [clear_frac(t), window_span_days(t), clear_frac(target)] -- no weather, no
    image, no current NDVI. The third column is what makes this a FORECAST's
    observation-process control rather than P1's: how cloudy the frame we are
    scored ON happened to be is a property of the observation process, it is
    strongly seasonal, and it reaches every model in the table through the
    common mask. If this control alone predicts NDVI(t+Delta), part of every
    other row is retention rather than representation.

    Built at FEATURE-ROW level (``row_of`` / ``cell_idx`` come from the target
    view), so the cell-level control is matched to the cell-level target -- the
    same matching P1 makes -- and cells with no surviving common pixel are
    absent here exactly as they are absent from the target.
    """
    row_of = np.asarray(row_of, dtype=int)
    cell_idx = np.asarray(cell_idx, dtype=int)
    t = rows.row_t[row_of]
    g = rows.row_target[row_of]
    if level == "pooled":
        assert (cell_idx < 0).all(), "a pooled view carries cell indices"
        X = np.column_stack([np.asarray(clear_frac)[t],
                             np.asarray(window_span_days)[t],
                             np.asarray(clear_frac)[g]])
        names = ("clear_frac_t", "window_span_days_t", "clear_frac_target")
    else:
        assert (cell_idx >= 0).all(), "a cell view carries no cell indices"
        gcf = np.asarray(grid_clear_frac, dtype=float)
        X = np.column_stack([gcf[t, cell_idx],
                             np.asarray(window_span_days)[t],
                             gcf[g, cell_idx]])
        names = ("grid_clear_frac_t", "window_span_days_t",
                 "grid_clear_frac_target")
    X = np.ascontiguousarray(X, dtype=np.float64)
    assert X.shape == (row_of.size, 3), X.shape
    assert np.isfinite(X).all(), "the observation control has a non-finite value"
    if verbose:
        print(f"[p3]   observation control ({level}): {X.shape} {list(names)} "
              "-- no weather, no image, no current NDVI")
    return p4.FeatureSource(name=f"observation/{level}", values=X, names=names,
                            frame_level=False)


# ---------------------------------------------------------------------------
# The fitted read-out. train_pos is REQUIRED and POSITIONAL.
# ---------------------------------------------------------------------------

def fit_readout(X, y, train_pos, test_pos, estimator: str) -> tuple:
    """Standardise on TRAIN, fit on TRAIN, predict TEST. Nothing is tuned.

    THE SIGNATURE IS THE GUARANTEE, exactly as
    ``p4_ceiling.doy_climatology_within_fold``, ``p2_deltas.fit_gap_control`` and
    ``p1_appearance.select_hyperparameter`` are. The FULL arrays go in and
    ``train_pos`` decides what is read, on the three lines marked below. There is
    no code path in which a row outside ``train_pos`` influences a coefficient --
    a property a test can check by poisoning the held-out rows and asserting the
    predictions are bit-identical, and by poisoning a TRAINING row and asserting
    they move. The first test alone passes on a function that ignores its input
    entirely, which is why there are two.

    The fit itself is ``p4_ceiling._fit_predict``: the same standardiser, the
    same estimators at the same fixed a-priori hyperparameters (ridge alpha = D
    on standardised features), the same convergence bookkeeping. There is no
    selection loop here, so there is nothing to prove clean about one.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    train_pos = np.asarray(train_pos, dtype=int)
    test_pos = np.asarray(test_pos, dtype=int)
    assert X.ndim == 2 and y.shape == (X.shape[0],), f"{X.shape} vs {y.shape}"
    assert train_pos.size and test_pos.size, "an empty fold side"
    assert not np.intersect1d(train_pos, test_pos).size, (
        "the same feature row is on both sides of the fold"
    )

    # --- THE ONLY READ OF THE ARRAYS -----------------------------------------
    X_tr, y_tr = X[train_pos], y[train_pos]
    X_te = X[test_pos]
    # -------------------------------------------------------------------------

    pred_te, pred_tr, converged = p4._fit_predict(estimator, X_tr, y_tr, X_te)
    assert pred_te.shape == (test_pos.size,), pred_te.shape
    return pred_te, pred_tr, converged


# ---------------------------------------------------------------------------
# One fold
# ---------------------------------------------------------------------------

class P3FoldResult(NamedTuple):
    """One outer fold. ``effective_n`` counts CUBES, never rows."""

    fold: int
    n_train: int
    n_test: int
    n_train_cubes: int
    n_test_cubes: int
    effective_n: int
    r2: float
    rmse: float
    mae: float
    sse: float
    sse_persistence: float
    sse_climatology: float
    sst: float
    same_cube_after_permutation: float
    converged: bool
    log: str
    # The held-out rows, kept so the severity breakdown is pooled from the SAME
    # fits the scores come from. A second pass would double the run and, worse,
    # would let the two disagree.
    test_pos: np.ndarray = None
    test_y: np.ndarray = None
    test_pred: np.ndarray = None
    test_persistence: np.ndarray = None


def _side_rows(rows: ForecastRows, manifest_rows, fold: int, side: str) -> np.ndarray:
    """Forecast rows every one of whose frames is on one side of the fold."""
    touched = rows.manifest_rows()
    inside = np.isin(touched, np.asarray(manifest_rows, dtype=int))
    whole = inside.all(axis=1)
    partial = inside.any(axis=1) & ~whole
    assert not partial.any(), (
        f"fold {fold}: {int(partial.sum())} forecast row(s) have their context, "
        f"t or target frame straddling the {side} side. Every frame of a row is "
        "in one cube and every fold mode in this repo is cube-grouped, so this "
        "cannot happen -- a mode that split inside a cube would leak the row's "
        "own future frame into training."
    )
    return np.flatnonzero(whole)


def evaluate_fold(sources, target: dict, rows: ForecastRows, manifest,
                  train_rows, test_rows, estimator: str, kind: str,
                  base_X=None, permute: bool = False, fold: int = 0,
                  climatology_harmonics: int = p4.CLIMATOLOGY_HARMONICS,
                  verbose: bool = False) -> P3FoldResult | None:
    """One outer fold, end to end. Returns None when a side holds no rows.

    A LOCO fold whose held-out cube contributes no row at this horizon is not a
    fold with a bad score; it is not a fold. At Delta = 100 d, 21 of 115 cubes
    have no pair at all, and counting them as zero-score folds would drag every
    interval toward zero for a reason that has nothing to do with forecasting.
    They are skipped and COUNTED (``n_folds_empty`` on the row).

    ``kind`` selects what is predicted, and only three of the eight kinds fit
    anything at all:

        persistence        pred = NDVI(t) on the common mask. No fit.
        climatology_proxy  a day-of-year curve fitted on the TRAINING rows via
                           p4_ceiling.doy_climatology_within_fold, evaluated at
                           the TARGET frame's day of year.
        everything else    fit_readout on the assembled design.
    """
    train_rows = np.asarray(train_rows, dtype=int)
    test_rows = np.asarray(test_rows, dtype=int)
    assert not np.intersect1d(train_rows, test_rows).size, (
        f"fold {fold}: manifest rows appear on both sides"
    )
    tr_rows = _side_rows(rows, train_rows, fold, "train")
    te_rows = _side_rows(rows, test_rows, fold, "test")
    if not tr_rows.size or not te_rows.size:
        return None

    y = np.asarray(target["y"], dtype=float)
    pers = np.asarray(target["persistence"], dtype=float)
    row_of = np.asarray(target["row_of"], dtype=int)   # feature row -> forecast row
    doy = np.asarray(target["day_of_year_target"], dtype=float)

    ok = np.isfinite(y) & np.isfinite(pers)
    tr = np.flatnonzero(np.isin(row_of, tr_rows) & ok)
    te = np.flatnonzero(np.isin(row_of, te_rows) & ok)
    if tr.size < 2 or te.size < 1:
        return None

    same_cube = float("nan")
    if kind == "persistence":
        pred, converged = pers[te], True
    elif kind == "climatology_proxy":
        curve = p4.doy_climatology_within_fold(
            doy, y, tr, n_harmonics=climatology_harmonics,
            label=f"{target['name']} fold {fold + 1}")
        pred, converged = curve.predict(doy[te]), True
    else:
        perm_map = None
        if permute:
            perm_map, same_cube = p4._perm_map(
                np.asarray(rows.cube_id, dtype=str), tr_rows, te_rows,
                rows.n_rows, seed=3000 + fold)
        X = base_X if (base_X is not None and not permute) \
            else p4.build_X(sources, row_of, perm_map)
        assert X.shape[0] == row_of.size, (X.shape, row_of.size)
        pred, _, converged = fit_readout(X, y, tr, te, estimator)

    resid = y[te] - pred
    sse = float((resid ** 2).sum())
    sst = float(((y[te] - y[te].mean()) ** 2).sum())
    sse_p = float(((y[te] - pers[te]) ** 2).sum())
    r2 = float(1.0 - sse / sst) if sst > 0 else float("nan")
    if te.size < _MIN_TEST_ROWS:
        r2 = float("nan")

    n_tr_c = int(np.unique(rows.cube_id[tr_rows]).size)
    n_te_c = int(np.unique(rows.cube_id[te_rows]).size)
    msg = (f"[p3]   fold {fold + 1}: {target['name']}/{kind}/{estimator} train "
           f"{tr.size} rows / {n_tr_c} cubes, test {te.size} rows / {n_te_c} "
           f"cubes (effective n = {n_te_c} CUBES, not {te.size} rows) | "
           f"R2 {r2:+.4f} | RMSE {np.sqrt(sse / te.size):.5f} | "
           f"SSE/SSE_pers {sse / sse_p if sse_p > 0 else float('nan'):.4f}")
    log(msg, echo=False)
    if verbose:
        print(msg)
    return P3FoldResult(
        fold=fold, n_train=int(tr.size), n_test=int(te.size),
        n_train_cubes=n_tr_c, n_test_cubes=n_te_c, effective_n=n_te_c,
        r2=r2, rmse=float(np.sqrt(sse / te.size)),
        mae=float(np.abs(resid).mean()), sse=sse, sse_persistence=sse_p,
        sse_climatology=float("nan"), sst=sst,
        same_cube_after_permutation=same_cube, converged=bool(converged),
        log=msg, test_pos=te, test_y=y[te], test_pred=np.asarray(pred, float),
        test_persistence=pers[te])


def evaluate(sources, target: dict, rows: ForecastRows, manifest, mode: str,
             estimator: str, kind: str, k: int = 5, permute: bool = False,
             base_X=None, n_jobs: int = 1, verbose: bool = False) -> list:
    """Every outer fold of one mode, from ``probes.cv`` and nowhere else.

    Deterministic: ``n_jobs`` changes wall-clock only. The ridge solve is exact,
    HGB carries ``early_stopping=False`` and a fixed seed, the MLP a fixed seed
    and no validation split, and the folds come from a generator with no RNG.

    ``base_X`` is the assembled design, built ONCE by the caller and reused
    across estimators. A DINOv2 context at k=3 is 11 520 columns; rebuilding it
    per estimator would cost a 150 MB allocation for nothing, and joblib
    memory-maps one array to the workers instead of pickling it per fold.
    """
    folds = p4.outer_folds(manifest, mode, k=k, verbose=False)
    if kind not in ("persistence", "climatology_proxy") and not permute \
            and base_X is None:
        base_X = p4.build_X(sources, np.asarray(target["row_of"], dtype=int))
    call = dict(estimator=estimator, kind=kind, base_X=base_X, permute=permute,
                verbose=False)
    if n_jobs == 1 or len(folds) < 2:
        out = [evaluate_fold(sources, target, rows, manifest, tr, te, fold=i,
                             **call) for i, (tr, te) in enumerate(folds)]
    else:
        from joblib import Parallel, delayed, parallel_config
        with parallel_config(backend="loky", inner_max_num_threads=1):
            out = Parallel(n_jobs=n_jobs)(
                delayed(evaluate_fold)(sources, target, rows, manifest, tr, te,
                                       fold=i, **call)
                for i, (tr, te) in enumerate(folds))
        # A loky worker re-imports this module, so its log handle is None and
        # the per-fold lines it wrote went nowhere. Every result carries its own
        # line precisely so the PARENT can write them, which keeps the run log
        # identical between serial and parallel rather than quietly thinner in
        # the mode the scaled run actually uses.
        for r in out:
            if r is not None:
                log(r.log)
    n_empty = int(sum(r is None for r in out))
    results = [r for r in out if r is not None]
    assert results, (
        f"{mode}: every fold was empty at Delta = {rows.delta_days_nominal} d"
    )
    if verbose:
        for r in results:
            print(r.log)
    return results, n_empty


def _pooled_with_fold_jackknife(results, fn, alpha: float = 0.05) -> tuple:
    """(value, lo, hi) for a POOLED statistic, clustered at the FOLD level.

    THE INTERVAL P3 NEEDS AND THE OTHER PHASES DID NOT. P2 and P4 average a
    per-fold score and take a t interval on the folds; that works because their
    folds each hold dozens of rows. Here a leave-one-cube-out fold holds TWO TO
    FIVE forecast rows, and an R-squared against the mean of three points is not
    a measurement -- ``r2`` is NaN for essentially every LOCO fold. Averaging
    what survives would report a number over an unrepresentative handful of
    folds.

    So the pooled statistic is computed over every held-out prediction exactly
    once, and its uncertainty comes from a DELETE-ONE-FOLD jackknife: recompute
    it with each fold's rows removed, and form the interval from the spread of
    those recomputations. Folds hold disjoint sets of cubes, so deleting one is
    deleting a cluster -- which is the same clustering ``fold_clustered_ci``
    applies to a mean, extended to a statistic that is not one.
    """
    from scipy.stats import t as student_t

    parts = [(r.test_y, r.test_pred, r.test_persistence) for r in results]
    y = np.concatenate([p[0] for p in parts])
    p = np.concatenate([p[1] for p in parts])
    q = np.concatenate([p[2] for p in parts])
    theta = fn(y, p, q)
    k = len(parts)
    if k < 2 or not np.isfinite(theta):
        return float(theta), float("nan"), float("nan")
    loo = []
    for i in range(k):
        keep = [j for j in range(k) if j != i]
        yy = np.concatenate([parts[j][0] for j in keep])
        pp = np.concatenate([parts[j][1] for j in keep])
        qq = np.concatenate([parts[j][2] for j in keep])
        loo.append(fn(yy, pp, qq))
    loo = np.asarray(loo, dtype=float)
    loo = loo[np.isfinite(loo)]
    if loo.size < 2:
        return float(theta), float("nan"), float("nan")
    m = loo.mean()
    se = np.sqrt((loo.size - 1) / loo.size * ((loo - m) ** 2).sum())
    half = student_t.ppf(1 - alpha / 2, loo.size - 1) * se
    return float(theta), float(theta - half), float(theta + half)


def _r2_pooled(y, p, q):
    sst = float(((y - y.mean()) ** 2).sum())
    return float(1.0 - ((y - p) ** 2).sum() / sst) if sst > 0 else float("nan")


def _skill_pooled(y, p, q):
    sse_p = float(((y - q) ** 2).sum())
    return float(1.0 - ((y - p) ** 2).sum() / sse_p) if sse_p > 0 else float("nan")


def summarise(results: Sequence[P3FoldResult], n_folds_empty: int = 0,
              severity=None) -> dict:
    """Mean, spread, a CUBE-CLUSTERED interval, and the pooled out-of-fold scores.

    Folds hold disjoint sets of cubes, so the per-fold values are the
    independent replicates -- not the rows, of which there are hundreds and
    which share skies, places and, at a 5-day horizon, both endpoints' weather.
    ``fold_clustered_ci`` is imported from ``p4_ceiling``.

    The POOLED scores exist because a per-fold R-squared is not always defined
    here: a LOCO fold can hold two rows, and two points have no meaningful
    variance to explain. The pooled quantities use every held-out prediction
    exactly once and are well defined wherever any row was predicted at all, so
    they are what the LOCO rows should be read on. ``skill_vs_persistence`` is
    the quantity the goal statement names.
    """
    assert results, "nothing to summarise"
    out = {"n_folds": len(results), "n_folds_empty": int(n_folds_empty)}
    for field in ("r2", "rmse", "mae"):
        v = np.array([getattr(r, field) for r in results], dtype=float)
        lo, hi = p4.fold_clustered_ci(v)
        ok = v[np.isfinite(v)]
        nan = float("nan")
        out[f"{field}_mean"] = float(ok.mean()) if ok.size else nan
        out[f"{field}_std"] = float(ok.std(ddof=1)) if ok.size > 1 else 0.0
        out[f"{field}_min"] = float(ok.min()) if ok.size else nan
        out[f"{field}_max"] = float(ok.max()) if ok.size else nan
        out[f"{field}_ci_lo"] = lo
        out[f"{field}_ci_hi"] = hi
        out[f"per_fold_{field}"] = ";".join(f"{x:.4f}" for x in v)
    out["n_folds_nan"] = int(sum(1 for r in results if not np.isfinite(r.r2)))
    out["n_train_median"] = int(np.median([r.n_train for r in results]))
    out["effective_n"] = int(sum(r.effective_n for r in results))
    out["effective_n_per_fold"] = ";".join(str(r.effective_n) for r in results)
    out["n_rows_test_total"] = int(sum(r.n_test for r in results))
    out["n_not_converged"] = int(sum(not r.converged for r in results))
    same = np.array([r.same_cube_after_permutation for r in results], dtype=float)
    out["same_cube_after_permutation_max"] = (
        float(np.nanmax(same)) if np.isfinite(same).any() else float("nan"))

    y = np.concatenate([r.test_y for r in results])
    p = np.concatenate([r.test_pred for r in results])
    q = np.concatenate([r.test_persistence for r in results])
    sse = float(((y - p) ** 2).sum())
    sse_p = float(((y - q) ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    (out["r2_pooled"], out["r2_pooled_ci_lo"],
     out["r2_pooled_ci_hi"]) = _pooled_with_fold_jackknife(results, _r2_pooled)
    (out["skill_vs_persistence"], out["skill_vs_persistence_ci_lo"],
     out["skill_vs_persistence_ci_hi"]) = _pooled_with_fold_jackknife(
        results, _skill_pooled)
    out["rmse_pooled"] = float(np.sqrt(sse / y.size))
    out["n_rows_pooled"] = int(y.size)
    # A MEDIAN sits beside every mean because three cloud-contaminated frames
    # (see print_target_outliers) carry most of the squared error on this
    # subset, and a median cannot be set by five rows.
    out["mae_pooled"] = float(np.abs(y - p).mean())
    out["medae_pooled"] = float(np.median(np.abs(y - p)))
    out["medae_persistence"] = float(np.median(np.abs(y - q)))
    out["skill_vs_persistence_medae"] = (
        float(1.0 - out["medae_pooled"] / out["medae_persistence"])
        if out["medae_persistence"] > 0 else float("nan"))
    # How concentrated this row's score is. sse_share_top1pct near 0.7 means
    # the R-squared above is a statement about a handful of held-out rows.
    sq = np.sort((y - p) ** 2)[::-1]
    n1 = max(1, int(round(0.01 * sq.size)))
    out["sse_share_top1pct"] = float(sq[:n1].sum() / sse) if sse > 0 else float("nan")

    if severity is not None:
        labels = np.asarray(severity)[np.concatenate([r.test_pos for r in results])]
        for b in SEVERITY_BINS:
            m = labels == b
            n = int(m.sum())
            out[f"n_{b}"] = n
            s_b = float(((y[m] - p[m]) ** 2).sum())
            sp_b = float(((y[m] - q[m]) ** 2).sum())
            st_b = float(((y[m] - y[m].mean()) ** 2).sum()) if n else 0.0
            enough = n >= _MIN_BIN_ROWS
            out[f"r2_{b}"] = (float(1.0 - s_b / st_b)
                             if enough and st_b > 0 else float("nan"))
            out[f"skill_vs_persistence_{b}"] = (float(1.0 - s_b / sp_b)
                                                if enough and sp_b > 0
                                                else float("nan"))
    return out


# ---------------------------------------------------------------------------
# The assembled inputs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class P3Data:
    """Everything the run fits on, assembled once from the cubes and the cache."""

    manifest: object
    rows: dict            # delta_days -> ForecastRows
    targets: dict         # delta_days -> {agg: {"y", "persistence"}, ...}
    weather: dict         # delta_days -> FeatureSource (per forecast row)
    observation: dict     # clear_frac / window_span_days / grid_clear_frac
    arrays: dict          # encoder -> cached embedding arrays
    doy_target: dict      # delta_days -> (N,) day of year of the TARGET frame
    severity: dict        # (delta_days, agg) -> (labels, edges) per feature row
    retention: object     # the per-horizon retention table
    survival: object      # the per-horizon pixel-survival table
    tolerance: object     # the tolerance sensitivity table
    outliers: object      # the per-horizon SSE-concentration table

    def target_view(self, delta_days: int, agg: str) -> dict:
        """One (horizon, aggregation) target, expanded to FEATURE ROWS.

        ``row_of`` maps every feature row back to its forecast row, which is
        what every fold selection and the permutation map are routed through.
        At cube level that is the identity; at cell level it is 16-to-1, and
        cells with no surviving common pixel are DROPPED, never filled.
        """
        rows = self.rows[delta_days]
        t = self.targets[delta_days][agg]
        N = rows.n_rows
        if TARGET_LEVEL[agg] == "frame":
            y = np.asarray(t["y"], dtype=float)
            pers = np.asarray(t["persistence"], dtype=float)
            row_of = np.arange(N)
            cell_idx = np.full(N, -1, dtype=int)
        else:
            valid = self.targets[delta_days]["cell_valid"].reshape(-1)
            keep = np.flatnonzero(valid)
            y = np.asarray(t["y"], dtype=float).reshape(-1)[keep]
            pers = np.asarray(t["persistence"], dtype=float).reshape(-1)[keep]
            row_of = np.repeat(np.arange(N), GRID_CELLS)[keep]
            cell_idx = np.tile(np.arange(GRID_CELLS), N)[keep]
        assert y.shape == pers.shape == row_of.shape == cell_idx.shape
        return {"name": f"{agg}@{delta_days}d", "aggregation": agg,
                "y": y, "persistence": pers, "row_of": row_of,
                "cell_idx": cell_idx,
                "day_of_year_target": self.doy_target[delta_days][row_of]}


def build_p3_data(manifest, cube_dir: str, encoders: Sequence[str] = ENCODER_ORDER,
                  horizons: Sequence[int] = HORIZONS,
                  emb_dir: str | None = None, mask_dir: str | None = None,
                  verbose: bool = True) -> P3Data:
    """Rows, targets, weather, controls and the reporting axes, once."""
    from data.loader import load_cube

    if verbose:
        print(f"[p3] manifest {manifest.shape} | "
              f"{manifest.cube_id.nunique()} cubes | "
              f"tiles {sorted(manifest.tile.unique())} | "
              f"years {sorted(manifest.year.unique())}")

    tol = horizon_tolerance_sensitivity(manifest, horizons=horizons,
                                        verbose=verbose)
    rows = {int(H): horizon_index(manifest, int(H), verbose=verbose)
            for H in horizons}
    retention = print_horizon_retention(rows, manifest, verbose=verbose)
    assert_retention_shrinks(retention, verbose=verbose)
    assert_climatology_identifiable(manifest, rows, verbose=verbose)

    targets = build_forecast_targets(rows, manifest, cube_dir,
                                     mask_dir=mask_dir, verbose=verbose)
    survival = summarise_pixel_survival_by_horizon(rows, targets, verbose=verbose)
    outliers = print_target_outliers(rows, targets, manifest, verbose=verbose)

    arrays = {}
    for enc in encoders:
        arrays[enc] = load_encoder_arrays(manifest, enc, emb_dir=emb_dir,
                                          verbose=False)
        if verbose:
            a = arrays[enc]
            print(f"[p3] {enc:<24} pooled {a['pooled'].shape} | "
                  f"grid {a['grid'].shape} | context "
                  f"{context_frames_for(enc)} frame(s) -> D_context "
                  f"{context_frames_for(enc) * a['pooled'].shape[1]} pooled")

    order = sorted(manifest["cube_id"].unique().tolist())
    per_cube = {"__order__": order}
    for cube in order:
        sample = load_cube(os.path.join(cube_dir, str(cube)), verbose=False)
        per_cube[cube] = p4.cube_frame_targets(sample, verbose=False)

    weather = {H: horizon_weather(manifest, cube_dir, rows[H], verbose=verbose)
               for H in rows}
    observation = dict(zip(("clear_frac", "window_span_days", "grid_clear_frac"),
                           manifest_clear_fractions(manifest, per_cube)))

    doy_all = manifest["day_of_year"].to_numpy().astype(float)
    doy_target = {H: doy_all[rows[H].row_target] for H in rows}

    data = P3Data(manifest=manifest, rows=rows, targets=targets,
                  weather=weather, observation=observation, arrays=arrays,
                  doy_target=doy_target, severity={}, retention=retention,
                  survival=survival, tolerance=tol, outliers=outliers)

    # Severity bins: quantiles of the TARGET's anomaly relative to the proxy
    # climatology, from p4's ONE full-data fit, which is a reporting axis only.
    sev = {}
    for H in rows:
        for agg in AGGREGATIONS:
            view = data.target_view(H, agg)
            anom = p4.severity_reference_anomaly(view["day_of_year_target"],
                                                 view["y"])
            if verbose and agg == PRIMARY_AGGREGATION:
                print(f"\n[p3] severity bins, Delta = {H} d, {agg}")
                labels, edges = p4.print_severity_bins(anom, f"{agg}@{H}d")
            else:
                labels, edges = p4.severity_bins(anom)
            sev[(H, agg)] = (labels, edges)
    object.__setattr__(data, "severity", sev)
    return data


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def _base_row(delta_days, aggregation, level, mode, encoder, feature_set,
              model_kind, estimator, D, rows: ForecastRows, n_feature_rows,
              context_frames) -> dict:
    return {
        "delta_days": int(delta_days),
        "tolerance_days": int(rows.tolerance_days),
        "aggregation": aggregation,
        "target_level": TARGET_LEVEL[aggregation],
        "target": f"{aggregation}_t+{int(delta_days)}d",
        "feature_level": level,
        "fold_mode": mode,
        "encoder": encoder,
        "feature_set": feature_set,
        "model_kind": model_kind,
        "estimator": estimator,
        "context_frames": int(context_frames),
        "is_control": model_kind in CONTROL_KINDS,
        "is_baseline": model_kind in BASELINE_KINDS,
        # The multi-image encoder's embedding at t already aggregates up to 8
        # preceding frames, so its context is one embedding and its lookback is
        # a variable number of DAYS. Flagged on every row it owns, so a filtered
        # view can never lose the caveat.
        "si_comparable": encoder != MI_ENCODER,
        "metric": "r2",
        "D": int(D),
        "n_retained": int(rows.n_rows),
        "n_cubes": int(rows.n_cubes),
        "n_feature_rows": int(n_feature_rows),
        "n_manifest_rows": 0,
        "horizon_source": "daily_axis_index",
        "common_masked": True,
        "weather_window": f"[t, t+{int(delta_days)}d] on the daily axis, "
                          "perfect-foresight",
        "climatology_def": (PROXY_CLIMATOLOGY_LABEL
                            if model_kind == "climatology_proxy" else ""),
        "is_proxy_climatology": model_kind == "climatology_proxy",
    }


def _horizon_control_folds(data: P3Data, agg: str, mode: str, k: int,
                           horizons: Sequence[int]) -> dict:
    """The horizon-alone control: Delta_days -> NDVI(t+Delta). No image.

    ``p2_deltas.fit_gap_control`` IMPORTED -- a horizon in days is exactly a gap
    in days, and that function already has the signature (``train_idx``
    positional), the degree-2 design and the two poisoning tests this control
    needs.

    IT IS FITTED POOLED ACROSS THE FOUR HORIZONS, and that is forced rather than
    chosen. Within one horizon Delta is a CONSTANT, the [1, d, d^2] design is
    rank 1, and p2's own guard refuses the fit with "the training fold covers
    only 1 distinct gap length". The pooled fit is the only one in which this
    control has anything to say; it is then SCORED separately on each horizon's
    held-out rows, which is what puts a per-horizon control row in front of a
    reader who filtered the CSV to one horizon.
    """
    assert len(set(int(H) for H in horizons)) >= HORIZON_CONTROL_DEGREE + 1, (
        f"the horizon control is a degree-{HORIZON_CONTROL_DEGREE} polynomial "
        f"in elapsed days, so it needs at least {HORIZON_CONTROL_DEGREE + 1} "
        f"DISTINCT horizons to be identified; this run has "
        f"{len(set(int(H) for H in horizons))} ({sorted(set(horizons))}). It is "
        "fitted POOLED across horizons precisely because within one horizon "
        "Delta is a constant and the design is rank 1 -- pooling is what gives "
        f"it anything to say. Run the full {len(HORIZONS)}-horizon set "
        f"{HORIZONS}, or lower HORIZON_CONTROL_DEGREE and record that on the "
        "row."
    )
    views = {H: data.target_view(H, agg) for H in horizons}
    delta = np.concatenate([np.full(views[H]["y"].size, float(H))
                            for H in horizons])
    y = np.concatenate([views[H]["y"] for H in horizons])
    which = np.concatenate([np.full(views[H]["y"].size, H) for H in horizons])
    pers = np.concatenate([views[H]["persistence"] for H in horizons])
    offset, rows_of = {}, []
    at = 0
    for H in horizons:
        offset[H] = at
        rows_of.append(views[H]["row_of"])
        at += views[H]["y"].size

    out = {H: ([], 0) for H in horizons}
    for i, (tr_rows, te_rows) in enumerate(p4.outer_folds(data.manifest, mode,
                                                          k=k, verbose=False)):
        tr_mask = np.zeros(y.size, dtype=bool)
        te_mask = np.zeros(y.size, dtype=bool)
        per_h = {}
        for H in horizons:
            r = data.rows[H]
            tr_r = _side_rows(r, tr_rows, i, "train")
            te_r = _side_rows(r, te_rows, i, "test")
            sl = slice(offset[H], offset[H] + views[H]["y"].size)
            tr_mask[sl] = np.isin(views[H]["row_of"], tr_r)
            te_mask[sl] = np.isin(views[H]["row_of"], te_r)
            per_h[H] = (tr_r, te_r)
        tr = np.flatnonzero(tr_mask & np.isfinite(y))
        te_all = np.flatnonzero(te_mask & np.isfinite(y))
        if tr.size < 3 or not te_all.size:
            for H in horizons:
                res, empty = out[H]
                out[H] = (res, empty + 1)
            continue
        ctrl = p2.fit_gap_control(delta, y, tr,
                                  degree=HORIZON_CONTROL_DEGREE,
                                  label=f"{agg} {mode} fold {i + 1} (POOLED "
                                        "over horizons)", verbose=True)
        for H in horizons:
            te = te_all[which[te_all] == H]
            res, empty = out[H]
            if not te.size:
                out[H] = (res, empty + 1)
                continue
            pred = ctrl.predict(delta[te])
            sse = float(((y[te] - pred) ** 2).sum())
            sst = float(((y[te] - y[te].mean()) ** 2).sum())
            sse_p = float(((y[te] - pers[te]) ** 2).sum())
            r2 = float(1.0 - sse / sst) if sst > 0 else float("nan")
            if te.size < _MIN_TEST_ROWS:
                r2 = float("nan")
            cubes = np.unique(data.rows[H].cube_id[per_h[H][1]]).size
            msg = (f"[p3]   fold {i + 1}: {agg}@{H}d/horizon_only train "
                   f"{tr.size} rows (POOLED over {len(horizons)} horizons), "
                   f"test {te.size} rows / {cubes} cubes | R2 {r2:+.4f}")
            log(msg)
            res.append(P3FoldResult(
                fold=i, n_train=int(tr.size), n_test=int(te.size),
                n_train_cubes=int(np.unique(
                    data.rows[H].cube_id[per_h[H][0]]).size),
                n_test_cubes=int(cubes), effective_n=int(cubes), r2=r2,
                rmse=float(np.sqrt(sse / te.size)),
                mae=float(np.abs(y[te] - pred).mean()), sse=sse,
                sse_persistence=sse_p, sse_climatology=float("nan"), sst=sst,
                same_cube_after_permutation=float("nan"), converged=True,
                log=msg, test_pos=te - offset[H], test_y=y[te],
                test_pred=pred, test_persistence=pers[te]))
            out[H] = (res, empty)
    return out


def run_p3(manifest, cube_dir: str, encoders: Sequence[str] = ENCODER_ORDER,
           horizons: Sequence[int] = HORIZONS,
           aggregations: Sequence[str] = AGGREGATIONS,
           k: int = 5, emb_dir: str | None = None, mask_dir: str | None = None,
           log_path: str | None = None, n_jobs: int = 1, data: P3Data = None,
           verbose: bool = True):
    """The whole table. Returns a DataFrame and writes nothing but the run log."""
    import pandas as pd

    open_run_log(log_path, verbose=verbose)
    try:
        if data is None:
            data = build_p3_data(manifest, cube_dir, encoders=encoders,
                                 horizons=horizons, emb_dir=emb_dir,
                                 mask_dir=mask_dir, verbose=verbose)
        rows = _run_p3_rows(data, encoders, horizons, aggregations, k, n_jobs,
                            verbose)
    finally:
        close_run_log()
    df = pd.DataFrame(rows)
    assert len(df), "run_p3 produced no rows"
    return df, data


def _run_p3_rows(data: P3Data, encoders, horizons, aggregations, k, n_jobs,
                 verbose) -> list:
    views = encoder_views(encoders)
    out = []
    horizons = [int(H) for H in horizons]

    for agg in aggregations:
        level = CONTEXT_LEVEL[agg]
        for mode in AGGREGATION_MODES[agg]:
            # The horizon control is ONE pooled fit per (aggregation, mode),
            # scored on each horizon's held-out rows.
            hc = _horizon_control_folds(data, agg, mode, k, horizons)
            for H in horizons:
                rows = data.rows[H]
                view = data.target_view(H, agg)
                sev = data.severity[(H, agg)][0]
                n_feat = view["y"].size
                w = data.weather[H]
                obs = observation_triple(
                    rows, data.observation["clear_frac"],
                    data.observation["window_span_days"],
                    data.observation["grid_clear_frac"], level,
                    view["row_of"], view["cell_idx"])
                if verbose:
                    head = f"{agg} | Delta={H}d | {level} | {mode}"
                    print(f"\n[p3] ---- {head} " + "-" * max(0, 46 - len(head)))
                    print(f"[p3]   {n_feat} feature rows over {rows.n_cubes} "
                          f"cubes ({rows.n_rows} forecast rows"
                          + (f" x up to {GRID_CELLS} cells)"
                             if TARGET_LEVEL[agg] == "cell" else ")"))

                def emit(kind, encoder, feature_set, estimator, sources,
                         D, ctx_frames, permute=False, base_X=None):
                    res, empty = evaluate(sources, view, rows, data.manifest,
                                          mode, estimator, kind, k=k,
                                          permute=permute, base_X=base_X,
                                          n_jobs=n_jobs)
                    row = {**_base_row(H, agg, level, mode, encoder,
                                       feature_set, kind, estimator, D, rows,
                                       n_feat, ctx_frames),
                           **summarise(res, empty, severity=sev)}
                    row["n_manifest_rows"] = int(len(data.manifest))
                    # p >> n is a MEASURED property of each row, not a caveat in
                    # prose: a k=3 DINOv2 context is 11 520 columns against a
                    # few hundred training rows, and an unregularised learner
                    # there is measuring capacity, not forecast skill.
                    row["d_over_n_train"] = (float(D) / row["n_train_median"]
                                             if row["n_train_median"] else
                                             float("nan"))
                    out.append(row)
                    if verbose:
                        print(f"[p3]   {kind:<21} {encoder:<24} "
                              f"{feature_set:<13} {estimator:<7} D={D:<6} "
                              f"R2 {row['r2_mean']:+.3f} "
                              f"[{row['r2_ci_lo']:+.3f}, {row['r2_ci_hi']:+.3f}]"
                              f" | pooled {row['r2_pooled']:+.3f} | vs pers "
                              f"{row['skill_vs_persistence']:+.3f} | n "
                              f"{row['effective_n']} CUBES"
                              + ("" if encoder != MI_ENCODER
                                 else "  [si_comparable=False]"))
                    return row

                # --- the five mandatory baselines -------------------------
                emit("persistence", "none", "none", "none", (), 0, 0)
                emit("climatology_proxy", "none", "none", "none", (), 0, 0)
                for est in WEATHER_ONLY_ESTIMATORS:
                    emit("weather_only", "none", "weather", est, (w,),
                         len(w.names), 0)

                # --- the encoder rows, and the two raw baselines -----------
                # The design is assembled ONCE per view and reused across
                # estimators: a DINOv2 k=3 context is 11 520 columns.
                for enc, fs, kind in views:
                    ctx_k = context_frames_for(enc)
                    Xc, names = context_block(
                        data.arrays[enc], rows, enc, level, fs,
                        row_of=view["row_of"],
                        cell_idx=(view["cell_idx"] if level == "grid_cell"
                                  else None))
                    assert Xc.shape[0] == n_feat, (Xc.shape, n_feat)
                    ctx = p4.FeatureSource(name=f"context/{enc}/{fs}",
                                           values=Xc, names=names,
                                           frame_level=False)
                    X = p4.build_X((ctx, w), view["row_of"])
                    for est in FORECAST_ESTIMATORS:
                        emit(kind, enc, fs, est, (ctx, w), X.shape[1], ctx_k,
                             base_X=X)
                    del Xc, ctx, X

                # --- the three controls, under every estimator -------------
                for est in CONTROL_ESTIMATORS:
                    emit("observation", "none", "observation", est, (obs,),
                         len(obs.names), 0)
                    emit("permutation", "none", "weather", est, (w,),
                         len(w.names), 0, permute=True)
                res, empty = hc[H]
                row = {**_base_row(H, agg, level, mode, "none", "horizon",
                                   "horizon_only", "none",
                                   HORIZON_CONTROL_DEGREE + 1, rows, n_feat, 0),
                       **summarise(res, empty, severity=sev)}
                row["n_manifest_rows"] = int(len(data.manifest))
                row["control_fit_scope"] = "pooled_over_horizons"
                row["d_over_n_train"] = ((HORIZON_CONTROL_DEGREE + 1)
                                         / row["n_train_median"]
                                         if row["n_train_median"] else float("nan"))
                out.append(row)
                if verbose:
                    print(f"[p3]   {'horizon_only':<21} {'none':<24} "
                          f"{'horizon':<13} {'none':<7} "
                          f"D={HORIZON_CONTROL_DEGREE + 1:<6} "
                          f"R2 {row['r2_mean']:+.3f} "
                          f"[{row['r2_ci_lo']:+.3f}, {row['r2_ci_hi']:+.3f}]"
                          "   <- CONTROL (elapsed days alone, no image, "
                          "fitted POOLED over horizons)")
    for r in out:
        r.setdefault("control_fit_scope", "within_horizon")
    return out


# ---------------------------------------------------------------------------
# Margins, and the assertions the exit test names
# ---------------------------------------------------------------------------

_CONTROL_KEY = ("delta_days", "aggregation", "fold_mode")


def add_margins(df, verbose: bool = True):
    """Attach the margins P3 is actually read on, and put the control on EVERY row.

    EVERY MARGIN IS TAKEN ON ``r2_pooled``, NOT ON ``r2_mean``, and that is
    forced by the data rather than preferred. A leave-one-cube-out fold here
    holds two to five forecast rows, so its own R-squared is NaN and
    ``r2_mean`` is NaN for essentially every LOCO row; a margin built on it
    would be missing for a whole fold mode. ``r2_pooled`` uses every held-out
    prediction exactly once, is defined wherever any row was predicted, and
    carries its own delete-one-FOLD jackknife interval. ``r2_mean`` and its
    fold-clustered interval stay on the table beside it and are the right
    number to quote under ``cube`` and ``spatial_block``, where a fold is large
    enough for an R-squared to mean something.

    ``margin_over_control`` -- ``r2_pooled`` minus the OBSERVATION control's, at
    the same (delta_days, aggregation, fold_mode) and THE SAME ESTIMATOR. This
    is the headline, exactly as P4 reports it. Comparing an MLP to a ridge
    control would be comparing two learners, not two feature sets, which is why
    the estimator is part of the key for this margin and not for the others.

    ``margin_over_persistence`` / ``margin_over_climatology`` -- against the two
    baselines the goal statement names. Those two rows are unfitted, so they
    have no estimator and the margin is taken against the single row per
    (delta_days, aggregation, fold_mode).

    ``margin_over_horizon_control`` -- against elapsed days alone.

    ``margin_over_band_matched`` -- minus ``raw_rgb_only`` at the same estimator:
    the hand-crafted baseline over the SAME three bands every network receives.
    THE like-for-like number. The margin over full ``raw_features`` is not, since
    that baseline additionally holds B8A and seven NDVI columns.

    ``control_score`` -- the observation control's own value, written onto EVERY
    row. That is what makes "filtering the CSV cannot drop the control" true
    rather than intended, and ``assert_control_identical_across_views`` asserts
    every copy is digit-for-digit identical.
    """
    key = list(_CONTROL_KEY)
    key_est = key + ["estimator"]
    df = df.copy()

    obs = df[df.model_kind == "observation"]
    assert len(obs) == len(obs.drop_duplicates(key_est)), (
        f"the observation control is not unique per {key_est}"
    )
    look = obs.set_index(key_est)["r2_pooled"]
    df["control_score"] = df.set_index(key_est).index.map(look).to_numpy()
    df["control_kind"] = "observation"
    # An unfitted baseline has no estimator, so it takes the LINEAR control --
    # named here rather than left to a silent NaN, and recorded on the row.
    fallback = obs[obs.estimator == "linear"].set_index(key)["r2_pooled"]
    miss = df.control_score.isna()
    df.loc[miss, "control_score"] = (df[miss].set_index(key).index
                                     .map(fallback).to_numpy())
    df["control_estimator"] = np.where(df.estimator.isin(CONTROL_ESTIMATORS),
                                       df.estimator, "linear")
    assert df.control_score.notna().all(), (
        "some row has no matching observation control. The controls are not "
        "optional and every row must be comparable to them."
    )
    df["margin_over_control"] = df.r2_pooled - df.control_score

    for kind, col in (("persistence", "margin_over_persistence"),
                      ("climatology_proxy", "margin_over_climatology"),
                      ("horizon_only", "margin_over_horizon_control")):
        ref = df[df.model_kind == kind]
        assert len(ref) == len(ref.drop_duplicates(key)), (
            f"the {kind} row is not unique per {key}"
        )
        r = ref.set_index(key)["r2_pooled"]
        df[f"{kind}_score"] = df.set_index(key).index.map(r).to_numpy()
        df[col] = df.r2_pooled - df[f"{kind}_score"]
        assert df[col].notna().all(), f"some row has no matching {kind} row"

    perm = df[df.model_kind == "permutation"].set_index(key_est)["r2_pooled"]
    df["permutation_score"] = df.set_index(key_est).index.map(perm).to_numpy()
    df["margin_over_permutation"] = df.r2_pooled - df.permutation_score

    band = df[(df.feature_set == BAND_MATCHED_BASELINE)
              & (df.encoder == BASELINE_ENCODER)]
    assert len(band), (
        f"no {BASELINE_ENCODER}/{BAND_MATCHED_BASELINE} rows to form "
        "margin_over_band_matched against; the band-matched baseline is what "
        "makes a network's score a statement about the representation rather "
        "than about which bands the baseline was given"
    )
    b = band.set_index(key_est)["r2_pooled"]
    df["margin_over_band_matched"] = (df.r2_pooled
                                      - df.set_index(key_est).index.map(b).to_numpy())
    if verbose:
        m = df[(~df.is_control) & (df.model_kind != "persistence")]
        beaten = m[m.skill_vs_persistence <= 0]
        print(f"[p3] margins attached. {len(beaten)}/{len(m)} non-persistence "
              "rows are AT OR BELOW persistence on pooled out-of-fold SSE -- "
              "i.e. predicting today's NDVI would have done at least as well.")
    return df


def print_headlines(df, aggregation: str = PRIMARY_AGGREGATION,
                    fold_mode: str = "cube") -> None:
    """The table a reader should see first: per horizon, best rows and baselines."""
    print("\n" + "=" * 92)
    print(f"P3 HEADLINES -- {aggregation} / {fold_mode} folds "
          "(R2 with a cube-clustered CI; skill is pooled out-of-fold vs "
          "persistence)")
    print("=" * 92)
    for H in sorted(df.delta_days.unique()):
        sub = df[(df.delta_days == H) & (df.aggregation == aggregation)
                 & (df.fold_mode == fold_mode)]
        if not len(sub):
            continue
        print(f"\n  Delta = {H} d   n = {int(sub.n_retained.iloc[0])} rows over "
              f"{int(sub.n_cubes.iloc[0])} cubes "
              f"(effective n = {int(sub.effective_n.iloc[0])} CUBES)")
        print(f"  {'model_kind':<21} {'encoder':<24} {'est':<7} "
              f"{'R2pooled':>9} {'jackknife CI':>18} {'R2/fold':>8} "
              f"{'vs pers':>9} {'vs ctrl':>9} {'medAE':>8} {'top1%':>6}")
        for r in sub.sort_values("r2_pooled", ascending=False).itertuples():
            tag = ("  <- CONTROL" if r.is_control
                   else ("  [si_comparable=False]" if not r.si_comparable
                         else ("  <- BASELINE" if r.is_baseline else "")))
            fold = (f"{r.r2_mean:>+8.3f}" if np.isfinite(r.r2_mean)
                    else f"{'n/a':>8}")
            print(f"  {r.model_kind:<21} {r.encoder:<24} {r.estimator:<7} "
                  f"{r.r2_pooled:>+9.3f} "
                  f"[{r.r2_pooled_ci_lo:>+7.3f},{r.r2_pooled_ci_hi:>+7.3f}] "
                  f"{fold} {r.skill_vs_persistence:>+9.3f} "
                  f"{r.margin_over_control:>+9.3f} {r.medae_pooled:>8.4f} "
                  f"{r.sse_share_top1pct:>5.0%}{tag}")
    print("\n  R2pooled uses every held-out prediction once and its interval is "
          "a delete-one-FOLD jackknife; R2/fold is the mean of the per-fold "
          "R-squareds and reads 'n/a' under loco, where a held-out cube "
          "contributes two to five rows and a per-fold R-squared is not a "
          "measurement. Margins are taken on R2pooled.")
    print("  medAE is the MEDIAN absolute error and top1% is the share of "
          "this row's held-out squared error carried by its worst 1% of rows. "
          "Read them together: an R-squared with top1% near 0.7 is a statement "
          "about a handful of rows (see print_target_outliers).")


def print_controls(df) -> None:
    """Every control, at every horizon and fold mode. Never filtered away."""
    print("\n" + "=" * 92)
    print("THE THREE CONTROLS (none optional). horizon_only is fitted POOLED "
          "over horizons -- see _horizon_control_folds")
    print("=" * 92)
    print(f"  {'Delta':>6} {'agg':<10} {'mode':<14} {'control':<14} {'est':<7} "
          f"{'R2':>8} {'CI':>18} {'n(cubes)':>9}")
    sub = df[df.is_control].sort_values(
        ["delta_days", "aggregation", "fold_mode", "model_kind", "estimator"])
    for r in sub.itertuples():
        print(f"  {r.delta_days:>6d} {r.aggregation:<10} {r.fold_mode:<14} "
              f"{r.model_kind:<14} {r.estimator:<7} {r.r2_pooled:>+9.3f} "
              f"[{r.r2_pooled_ci_lo:>+7.3f},{r.r2_pooled_ci_hi:>+7.3f}] "
              f"{r.effective_n:>9d}")


def print_severity_table(df, aggregation: str = PRIMARY_AGGREGATION,
                         fold_mode: str = "cube") -> None:
    """Where a forecast can actually differ from persistence."""
    print("\n" + "=" * 92)
    print(f"BY SEVERITY BIN -- {aggregation} / {fold_mode}, pooled out-of-fold "
          "skill vs persistence (stable periods flatter persistence)")
    print("=" * 92)
    cols = [f"skill_vs_persistence_{b}" for b in SEVERITY_BINS]
    for H in sorted(df.delta_days.unique()):
        sub = df[(df.delta_days == H) & (df.aggregation == aggregation)
                 & (df.fold_mode == fold_mode) & (~df.is_control)
                 & (df.model_kind != "persistence")]
        if not len(sub):
            continue
        best = sub.sort_values("skill_vs_persistence", ascending=False).head(4)
        print(f"\n  Delta = {H} d   bin counts: "
              + "  ".join(f"{b}={int(sub[f'n_{b}'].iloc[0])}"
                          for b in SEVERITY_BINS))
        print(f"  {'model_kind':<21} {'encoder':<24} {'est':<7} "
              + " ".join(f"{b[:9]:>10}" for b in SEVERITY_BINS))
        for r in best.itertuples():
            vals = " ".join(f"{getattr(r, c):>+10.3f}" for c in cols)
            print(f"  {r.model_kind:<21} {r.encoder:<24} {r.estimator:<7} {vals}")


def assert_results_complete(df, encoders: Sequence[str] = ENCODER_ORDER,
                            horizons: Sequence[int] = HORIZONS,
                            fold_modes: Sequence[str] = FOLD_MODES) -> None:
    """Every cell the exit test asks for is present."""
    for col in ("delta_days", "aggregation", "target_level", "fold_mode",
                "encoder", "feature_set", "model_kind", "estimator",
                "is_control", "is_baseline", "si_comparable", "metric",
                "r2_mean", "r2_ci_lo", "r2_ci_hi", "rmse_mean", "r2_pooled",
                "skill_vs_persistence", "effective_n", "effective_n_per_fold",
                "n_retained", "n_cubes", "n_rows_test_total", "horizon_source",
                "climatology_def", "context_frames", "control_fit_scope",
                "medae_pooled", "sse_share_top1pct", "d_over_n_train",
                "r2_pooled_ci_lo", "r2_pooled_ci_hi", "n_folds_nan",
                "skill_vs_persistence_ci_lo", "skill_vs_persistence_ci_hi"):
        assert col in df.columns, f"the results table has no {col!r} column"
    for b in SEVERITY_BINS:
        for pre in ("n_", "r2_", "skill_vs_persistence_"):
            assert f"{pre}{b}" in df.columns, f"no {pre}{b} column"
    for H in horizons:
        assert (df.delta_days == int(H)).any(), f"no rows at Delta = {H} d"
    primary = df[df.aggregation == PRIMARY_AGGREGATION]
    for mode in fold_modes:
        assert (primary.fold_mode == mode).any(), f"no {mode!r} rows"
        for H in horizons:
            assert ((primary.fold_mode == mode)
                    & (primary.delta_days == int(H))).any(), (
                f"no {mode!r} rows at Delta = {H} d")
    for agg in AGGREGATIONS:
        assert (df.aggregation == agg).any(), f"no {agg!r} rows"
    for enc in encoders:
        assert (df.encoder == enc).any(), f"no rows for {enc!r}"
    assert (df.horizon_source == "daily_axis_index").all(), (
        "a row does not declare daily_axis_index as its horizon source. The "
        "horizon must never come from original_axis_index, which counts "
        "ACQUISITIONS and is the embedding join key."
    )
    assert df.common_masked.all(), (
        "a row does not declare that its target was common-masked"
    )
    bad = df[~np.isfinite(df.r2_pooled)]
    assert bad.empty, (
        f"{len(bad)} row(s) have a non-finite POOLED R-squared, e.g.\n"
        f"{bad[['delta_days', 'aggregation', 'model_kind', 'fold_mode']].head(3)}"
    )
    # r2_mean is allowed to be NaN, and where it is, the row must SAY SO: a
    # leave-one-cube-out fold holds two to five forecast rows and an R-squared
    # against the mean of three points is not a measurement. n_folds_nan is the
    # count, and it is what makes the NaN a reported fact instead of a hole.
    nan_mean = df[~np.isfinite(df.r2_mean)]
    assert (nan_mean.n_folds_nan > 0).all(), (
        f"{int((nan_mean.n_folds_nan == 0).sum())} row(s) report a non-finite "
        "r2_mean while claiming every fold scored. The mean is NaN only "
        "because per-fold R-squareds were NaN, and the count of those has to "
        "be on the row."
    )


def assert_baselines_present(df) -> None:
    """All five mandatory baselines exist for every (horizon, aggregation, mode)."""
    have = df[df.is_baseline]
    combos = df[~df.is_control].groupby(list(_CONTROL_KEY)).size().index
    for kind in BASELINE_KINDS:
        got = set(map(tuple, have[have.model_kind == kind][list(_CONTROL_KEY)]
                      .to_numpy()))
        missing = [c for c in combos if tuple(c) not in got]
        assert not missing, (
            f"the {kind!r} baseline is missing for {len(missing)} "
            f"configuration(s), e.g. {missing[:3]}. All five baselines are "
            "mandatory and belong in the SAME table as the model rows."
        )
    assert ((df.encoder == BASELINE_ENCODER)
            & (df.feature_set == BAND_MATCHED_BASELINE)).any(), (
        f"no {BAND_MATCHED_BASELINE} rows. On P2's sign probe this row beat "
        "every network; without it a table of foundation-model scores has no "
        "like-for-like comparison in it."
    )


def assert_controls_present(df) -> None:
    """All three controls exist for every (horizon, aggregation, fold mode)."""
    combos = df[~df.is_control].groupby(list(_CONTROL_KEY)).size().index
    for kind in CONTROL_KINDS:
        sub = df[df.model_kind == kind]
        got = set(map(tuple, sub[list(_CONTROL_KEY)].to_numpy()))
        missing = [c for c in combos if tuple(c) not in got]
        assert not missing, (
            f"{len(missing)} configuration(s) have no {kind!r} control, e.g. "
            f"{missing[:3]}. The controls are not optional: without them a "
            "score cannot be distinguished from a calendar or from the "
            "observation process."
        )
    modes = set(df.fold_mode.unique())
    for kind in CONTROL_KINDS:
        got = set(df[df.model_kind == kind].fold_mode.unique())
        assert modes == got, (
            f"the {kind!r} control is missing for fold mode(s) "
            f"{sorted(modes - got)}"
        )


def assert_control_identical_across_views(df) -> None:
    """A control is digit-for-digit identical under every filter it is invariant to.

    Two things, and the second is the one that matters:

    1. The control ROWS agree across the labels they are emitted under. The
       observation and permutation controls hold no embedding, so they are
       invariant to encoder and feature set but NOT to estimator; the horizon
       control is invariant to estimator as well, since it is a fixed degree-2
       least-squares fit with no learner in it.
    2. ``control_score`` -- the copy ``add_margins`` writes onto EVERY row -- is
       identical across every filtered view AND equals the control's own row.
       A reader who filters to one encoder still has the control in front of
       them, and it is the same number the control's own row reports. Both
       halves are checked, because a uniformly wrong copy would satisfy the
       first alone.

    ``delta_days``, ``aggregation`` and ``fold_mode`` are part of the KEY, not
    labels the control is invariant to: a control evaluated at another horizon,
    against another target or on different held-out cubes is a different number,
    and reporting one value across them would be a false identity.
    """
    cols = ["r2_pooled", "r2_pooled_ci_lo", "r2_pooled_ci_hi", "per_fold_r2",
            "effective_n"]
    for kind in CONTROL_KINDS:
        key = list(_CONTROL_KEY) + (["estimator"] if kind != "horizon_only"
                                    else [])
        sub = df[df.model_kind == kind]
        if not len(sub):
            continue
        for col in cols:
            g = sub.groupby(key)[col].nunique(dropna=False)
            bad = g[g > 1]
            assert bad.empty, (
                f"the {kind} control's {col} differs between views for "
                f"{list(bad.index)[:3]}. It is invariant to everything outside "
                f"{key}, so every copy is the same fitted model; a "
                "disagreement means one copy was computed on different rows."
            )
    assert "control_score" in df.columns, (
        "control_score is not on the table. It must be on EVERY row, not only "
        "the control's own, or filtering the CSV can drop the control."
    )
    assert df.control_score.notna().all(), (
        f"{int(df.control_score.isna().sum())} row(s) carry no control_score"
    )
    key_est = list(_CONTROL_KEY) + ["control_estimator"]
    g = df.groupby(key_est)["control_score"].nunique(dropna=False)
    bad = g[g > 1]
    assert bad.empty, (
        f"control_score differs between filtered views of the table for "
        f"{list(bad.index)[:3]}: filtering by encoder or feature set changes "
        "the control a row is measured against, so the margins in those views "
        "are not comparable."
    )
    own = (df[df.model_kind == "observation"]
           .set_index(list(_CONTROL_KEY) + ["estimator"])["r2_pooled"])
    seen = df.groupby(key_est)["control_score"].first()
    np.testing.assert_array_equal(
        own.reindex(seen.index).to_numpy(), seen.to_numpy(),
        err_msg="the control_score carried on each row differs from the "
                "observation control's own row")


def assert_climatology_rows_labelled(df) -> None:
    """Every climatology row says PROXY, and no other row claims to be one.

    THE EMPTY LABEL IS COMPARED AS "UNLABELLED", NOT AS ``""``. An empty string
    written to CSV reads back as NaN, so a naive ``== ""`` passes on the
    in-memory table and FAILS on the artefact that table was written to -- which
    is the worst possible direction for an assertion to fail in, because the CSV
    is what anyone else reads. Checked here on the value's MEANING, and
    round-tripped through CSV in the tests.
    """
    clim = df[df.model_kind == "climatology_proxy"]
    assert len(clim), "no climatology baseline rows"
    assert (clim.climatology_def == PROXY_CLIMATOLOGY_LABEL).all(), (
        "a climatology row does not carry the proxy label. This is a "
        "within-season, single-year, tile-level day-of-year curve, NOT the "
        "leave-target-year-out definition, and it is NOT H1's number."
    )
    assert clim.is_proxy_climatology.astype(bool).all()
    other = df[df.model_kind != "climatology_proxy"]
    assert not other.is_proxy_climatology.astype(bool).any(), (
        "a non-climatology row is flagged is_proxy_climatology"
    )
    unlabelled = other.climatology_def.isna() | (other.climatology_def == "")
    assert unlabelled.all(), (
        f"{int((~unlabelled).sum())} non-climatology row(s) carry a climatology "
        f"definition, e.g. {other.loc[~unlabelled, 'climatology_def'].iloc[0]!r}"
    )


def assert_mi_flagged_and_single_frame(df) -> None:
    """Every multi-image row says si_comparable=False and context_frames=1."""
    mi = df[df.encoder == MI_ENCODER]
    assert len(mi), f"no rows for {MI_ENCODER}; its scores must be REPORTED"
    assert not mi.si_comparable.any(), (
        f"{int(mi.si_comparable.sum())} {MI_ENCODER} row(s) claim "
        "si_comparable=True. Its embedding at t already aggregates up to 8 "
        "preceding frames over a 0-105 day lookback."
    )
    assert (mi.context_frames == MI_CONTEXT_FRAMES).all(), (
        f"a {MI_ENCODER} row reports "
        f"{sorted(set(mi.context_frames.tolist()))} context frames; its context "
        f"is its SINGLE embedding at t ({MI_CONTEXT_FRAMES}), and a 3-stack "
        "would double-count a lookback it already contains."
    )
    si = df[(df.encoder != MI_ENCODER) & (df.encoder != "none")]
    assert si.si_comparable.all(), (
        "a single-image encoder is flagged si_comparable=False"
    )
    assert (si.context_frames == CONTEXT_FRAMES).all(), (
        f"a single-image encoder row does not report {CONTEXT_FRAMES} context "
        f"frames: {sorted(set(si.context_frames.tolist()))}"
    )


assert_effective_n_counts_cubes = p4.assert_effective_n_counts_cubes


def results_path(name: str = "p3_forecast_results.csv",
                 root: str | None = None) -> str:
    """Where a P3 table is written.

    ``root`` overrides the phase directory, so a run over the scaled cube set
    writes beside the cubes it was computed from -- the placement
    ``scripts/scale_p4.py`` and ``scripts/scale_p2.py`` already use -- and
    cannot overwrite a published artefact by forgetting an argument.
    """
    base = phase_dir(PHASE, "results") if root is None else root
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, name)
