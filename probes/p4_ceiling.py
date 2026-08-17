"""P4, the weather-attributability ceiling: what fraction of the
post-climatology NDVI anomaly is explainable from meteorological forcing alone?

That number is H1, and it is simultaneously P3's DENOMINATOR -- "how much was
there to get". A forecast probe that recovers 0.15 of anomaly variance is a
different result depending on whether weather alone reaches 0.20 or 0.60.

WHAT THIS PROBE READS, AND WHAT IT REFUSES TO READ
--------------------------------------------------
Cubes, and nothing else. No embedding is opened, no encoder is imported, no
weight is loaded -- the inputs are the in-cube E-OBS series (8 variables on the
cube's DAILY axis), the canonical NDVI target, and the manifest. CPU only.

``window_span_days`` -- half of P1's degenerate control -- is NOT read from the
Phase 1.2 cache. It is recomputed from the manifest timestamps through
``encoders.pipeline.window_span_days``, the same function the cache was written
with, so the control is available here without this phase depending on an
artefact of another one.

THE CONFOUND THIS BLOCK EXISTS TO MEASURE, NOT TO ASSUME AWAY
--------------------------------------------------------------
Cloudiness drives precipitation, which is a weather feature. It ALSO drives
which frames survive the clear-fraction filter, and which pixels the NDVI
spatial mean is taken over. So a weather-only model can score above zero by
exploiting the OBSERVATION PROCESS rather than the vegetation response, and on
this subset that is not hypothetical: P1 measured ``[clear_frac,
window_span_days]`` -- two numbers, no image -- decoding SEASON at 0.646-0.658
balanced accuracy, at or above every foundation model's cell-level score.

Therefore the headline of this probe is ``margin_over_control``, the weather
model minus the observation-process control, and never the raw R-squared. The
raw number is reported beside it; it is not the number to quote.

TWO STAGES, AND ONLY ONE OF THEM PRODUCES H1
---------------------------------------------
**Stage A** runs on the current 20 cubes (tile 32UNU, 2018 only). A
leave-target-year-out climatology is NOT COMPUTABLE there --
``data.climatology.ndvi_climatology`` raises ``SingleYearError`` by design and
that is correct -- so Stage A uses an explicitly named PROXY,
``doy_climatology_within_fold``: a tile-level day-of-year NDVI curve (low-order
harmonic) pooled across cubes. Every Stage A number carries the label

    "within-season proxy climatology, tile-level, single year (2018),
     NOT the leave-year-out definition"

in the ``climatology_def`` column of the CSV. Stage A exists to de-risk the
code and establish the controls. **It does not produce H1.**

**Stage B** runs if and only if multi-year cubes are present (the seasonal
split spans 2017-2020 inside a single cube). It uses the REAL leave-target-
year-out definition, imported from ``data.climatology``, with
``probes.cv`` mode ``crossed`` -- which holds cube AND year out jointly and so
agrees with the climatology's own leave-year-out structure. Any other mode
would pair a test year against a climatology built including that year. If the
seasonal cubes are absent, Stage B prints a deferral and exits cleanly. It is
never silently replaced by Stage A's number, and
``assert_stage_b_ran_or_deferred`` enforces that on the table.

THE CRITICAL CONSTRAINT ON THE PROXY: FIT INSIDE THE FOLD
----------------------------------------------------------
The day-of-year curve is fitted on TRAINING CUBES ONLY, inside each fold, and
applied to the held-out cubes. Fitting it once on all 20 cubes and then
cross-validating the residual leaks the test cubes into the TARGET DEFINITION.
That is nested leakage: it is invisible in the output, it inflates every number
including every control, and no downstream assertion can detect it because the
resulting table is internally consistent.

So the structure is in the signature, not in the discipline:

    doy_climatology_within_fold(day_of_year, values, train_idx, ...)

is handed the full arrays and the TRAINING INDEX SET, and reads the arrays
exactly once, through that index. The test poisons the held-out rows and
asserts the fitted coefficients are bit-identical.

A consequence worth stating plainly: **the target is fold-dependent.** The
anomaly is re-derived inside every fold, so "the anomaly" is not one array that
exists before cross-validation starts. Severity bins are the one exception and
they are a REPORTING axis only -- see ``severity_bins``.

TARGETS, SPATIALLY AGGREGATED, NEVER PIXEL-WISE
------------------------------------------------
    cube_mean   PRIMARY. Cube-mean NDVI anomaly, one row per (cube, frame).
    cube_p90    Cube-90th-percentile NDVI anomaly. Same rows.
    cell_mean   The 16 grid cells' mean NDVI anomaly, 16 rows per frame. This
                is what makes the per-stratum decomposition possible: same
                cube, same weather, different land cover.

**Weather is CONSTANT across the 16 cells of a frame.** Cell rows are therefore
PSEUDO-REPLICATES for the feature side -- they add target variance, not feature
variance. They fix the fit; they do not create 16 independent observations.
``assert_weather_constant_across_cells`` asserts the property (if it ever fails
the join is wrong), and all uncertainty is clustered at the CUBE level whatever
the row count.

EFFECTIVE SAMPLE SIZE
---------------------
20 cubes, one tile, one year. The effective n is ~20 independent weather
realisations -- NOT 264 frames and NOT 4224 cells. ``effective_n`` is on every
row of the CSV next to the R-squared, it counts CUBES, and every interval is a
fold-clustered CI over the per-fold values. An R-squared quoted here without
its effective n is not reportable.

WEATHER FEATURES ARE WINDOWED OVER THE DAILY AXIS, NEVER OVER RETAINED FRAMES
-----------------------------------------------------------------------------
"The last 30 days" is 30 steps on the cube's daily axis and about 6 retained
frames; a frame-defined window is longer in cloudy weather, which makes the
window itself a weather feature and contaminates exactly the comparison this
probe rests on. Windows are built from ``encoders.manifest.cube_weather`` and
``cube_daily_axis``, positioned by ``daily_axis_index``.

Aggregates are per-DAY RATES rather than raw sums. A frame early in a cube has
a truncated window (the earliest retained frame sits 4 days into the axis on
this subset), and a raw sum over a truncated window is biased low in a way that
correlates with position-in-cube, i.e. with time of year. A rate is noisier
there but unbiased. The truncation is printed, never imputed, and the number of
available days is deliberately NOT a feature -- it would hand the model a
season proxy.

Two feature sets are reported:

    weather_full8   all 8 E-OBS variables (tg/tn/tx/rr/pp/fg/hu/qq)
    weather_eowm5   the 5 matching EO-WM's meso channels (precipitation,
                    pressure, mean/min/max temperature), for cross-paper
                    commensurability. See docs/correspondence/.

ESTIMATORS, AND WHY NOTHING IS TUNED
-------------------------------------
Linear (ridge), HistGradientBoostingRegressor, and a small MLP -- increasing
capacity, all three reported. **No hyperparameter is selected from data.** P1
needed a nested tuning loop and a poisoning test to prove that loop never saw a
test fold; here there is no loop to prove anything about, which is strictly
stronger. The values are fixed a priori, printed by ``describe_estimators``,
and defensible from the shape of the problem: at D <= 64 against n >= 264 the
ridge path is flat, so the regularisation is not what decides these numbers.
Standardisation is still fitted on train and applied to test, per fold.

THE FOUR CONTROLS, NONE OF THEM OPTIONAL
-----------------------------------------
    observation               [clear_frac, window_span_days] -> anomaly, no
                              weather at all. P1's control carried forward, and
                              after P1's result it is THE floor that matters.
    doy                       day-of-year alone -> anomaly. If the detrend
                              worked this is ~0. A materially non-zero value
                              means the proxy climatology is UNDERFITTING and
                              the "anomaly" still carries seasonal cycle, which
                              weather would then trivially predict. Its basis is
                              deliberately RICHER than the detrend's (4
                              harmonics against 2, and the tree/MLP estimators
                              are non-linear in day-of-year), because a control
                              built from the detrend's own basis would return
                              zero by construction and test nothing.
    weather_plus_observation  both, to see whether weather adds anything beyond
                              the observation process and whether the two are
                              redundant.
    permutation               weather shuffled across cubes within the fold and
                              re-fitted. This is the EMPIRICAL ZERO for this
                              pipeline, which is not exactly 0.0 with grouped
                              data and a flexible estimator.

METRICS
-------
    r2                  sklearn r2_score: variance explained relative to the
                        TEST FOLD's own anomaly mean. The conventional number.
    r2_vs_climatology   1 - SSE(model)/SSE(0). The climatology predicts anomaly
                        zero, so this is skill against the climatology itself,
                        and it is the quantity the goal statement names --
                        "what fraction of post-climatology anomaly variance is
                        explainable". **This is H1's metric**, and the one
                        ``margin_over_control`` is computed on. It is also the
                        more stable of the two at 20 cubes, where a test fold's
                        own mean is itself a noisy estimate.
    crps / crps_skill   CRPS against a climatology-ENSEMBLE baseline: the model
                        as a Gaussian (mean = prediction, sd = its own TRAIN
                        residual sd), the baseline as the empirical ensemble of
                        TRAIN anomalies, which is the distribution the
                        climatology implies for a new observation.

Reported per severity bin, per stratum (``REPLICATION_STRATA`` only -- built_up
has 5 cells and bare_sparse 1, and a "replication" over 1 cell is not a
replication), with the fold-clustered CI and the effective n on every row.
"""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass
from typing import NamedTuple, Sequence

import numpy as np

from data.paths import phase_dir
from encoders.frames import MIN_CLEAR_FRACTION, grid_clear_fraction, select_clear_frames
from encoders.manifest import cube_daily_axis, cube_weather, frame_daily_positions
from encoders.pipeline import window_span_days
from probes import cv
from probes.cv import REPLICATION_STRATA
# THE design-matrix / row-index / label binding, reused rather than rewritten.
# It exists because cell-level rows put 16 X rows on every manifest row, and
# docs/DECISIONS.md 2026-08-09 records that P2, P3 and P4 would all write that
# pattern again. This is P4 not writing it again.
from probes.p1_appearance import FeatureBlock

__all__ = [
    "PHASE", "STAGES", "TARGETS", "TARGET_LEVEL", "FOLD_MODES", "STAGE_B_MODE",
    "ESTIMATORS", "FEATURE_SETS", "MODEL_KINDS", "CONTROL_KINDS",
    "EOBS_VARS", "EOWM_MESO_VARS", "WINDOW_SPECS", "AGGREGATIONS",
    "GDD_BASE_C", "CLIMATOLOGY_HARMONICS", "DOY_CONTROL_HARMONICS",
    "STAGE_B_HARMONICS", "DOY_PERIOD_DAYS", "MI_WINDOW_LEN",
    "SEVERITY_QUANTILES", "SEVERITY_BINS",
    "STAGE_A_CLIMATOLOGY_LABEL", "STAGE_B_CLIMATOLOGY_LABEL",
    "DoyCurve", "doy_design", "doy_climatology_within_fold",
    "severity_reference_anomaly", "severity_bins", "print_severity_bins",
    "print_doy_weather_collinearity",
    "TargetRows", "FeatureSource", "P4Data",
    "cube_frame_targets", "build_p4_data", "target_rows",
    "manifest_frame_plausible", "apply_plausibility_screen",
    "PLAUSIBILITY_SCREEN_LABEL",
    "cube_weather_windows", "weather_source", "observation_source", "doy_source",
    "build_X", "permute_within_fold", "assert_weather_constant_across_cells",
    "gaussian_crps", "ensemble_crps", "fold_clustered_ci",
    "make_estimator", "describe_estimators",
    "P4FoldResult", "evaluate_fold", "evaluate", "summarise",
    "outer_folds", "run_stage_a", "detect_seasonal_split", "run_stage_b",
    "run_p4", "add_margins",
    "assert_results_complete", "assert_control_views_consistent",
    "assert_stage_b_ran_or_deferred", "assert_effective_n_counts_cubes",
    "assert_plausibility_screen_declared",
    "results_path",
]

# Phase 1.5 is P4. Artefacts are phase-scoped through data/paths.py; nothing
# here ever writes to a hand-typed path.
PHASE = "phase1_5"

STAGES = ("A", "B")

# (target name -> the level its rows live at). cube_* are one row per frame;
# cell_mean is 16 rows per frame and is what the per-stratum decomposition
# needs.
TARGETS = ("cube_mean", "cube_p90", "cell_mean")
TARGET_LEVEL = {"cube_mean": "frame", "cube_p90": "frame", "cell_mean": "cell"}

FOLD_MODES = ("cube", "loco", "spatial_block")
# Stage B's mode, and the only one that agrees with a leave-target-year-out
# climatology. Not a default: it RAISES on a single-year manifest, correctly.
STAGE_B_MODE = "crossed"

ESTIMATORS = ("linear", "hgb", "mlp")
FEATURE_SETS = ("weather_full8", "weather_eowm5")

MODEL_KINDS = ("weather", "observation", "doy", "weather_plus_observation",
               "permutation")
CONTROL_KINDS = ("observation", "doy", "weather_plus_observation", "permutation")

# The in-cube E-OBS stack. EIGHT variables, not nine (docs/DECISIONS.md
# 2026-08-03): tg/tn/tx mean/min/max temperature, rr precipitation, pp
# pressure, fg wind speed, hu humidity, qq shortwave radiation.
EOBS_VARS = ("eobs_tg", "eobs_tn", "eobs_tx", "eobs_rr", "eobs_pp",
             "eobs_fg", "eobs_hu", "eobs_qq")

# EO-WM's five meso channels, from the authors' era5_climatology_all.pt
# ([100, 12, 5] per-tile per-month mean/std): precipitation, pressure,
# mean/min/max temperature. Our 8 are a superset, so this subset is reported
# alongside the full one for cross-paper commensurability. See
# docs/correspondence/2026-08-07-eowm-authors.md section (d).
EOWM_MESO_VARS = ("eobs_rr", "eobs_pp", "eobs_tg", "eobs_tn", "eobs_tx")

# Tiles outside the original subset can have INCOMPLETE E-OBS coverage, and it
# is not random: on 30TVN ``eobs_fg`` is absent from 20 of 25 cubes and
# ``eobs_qq`` is missing the same trailing 13 days in every one. Nothing is
# filled -- this is the fully-finite intersection, registered as a NAMED set so
# a cross-tile run declares which variables it used in the CSV rather than
# mutating the registry at runtime. It is not the default: on 32UNU all eight
# are present and ``weather_full8`` is the set to use.
WEATHER_FINITE6 = ("eobs_tg", "eobs_tn", "eobs_tx", "eobs_rr", "eobs_pp",
                   "eobs_hu")

FEATURE_SET_VARS = {"weather_full8": EOBS_VARS,
                    "weather_eowm5": EOWM_MESO_VARS,
                    "weather_finite6": WEATHER_FINITE6}

# (span_days, lag_days). The window for a frame at daily position p covers
# [p - lag - span + 1, p - lag] INCLUSIVE, clipped at the start of the cube.
# The lagged 30-day window is the antecedent-conditions term: soil moisture and
# green-up respond to weather that fell weeks before the observation, and no
# trailing window can express that.
#
# CHOSEN AGAINST THE MEASURED HISTORY, not by convention. A cube is a 150-day
# file and its earliest retained frame sits 4 days in; over the 264 rows the
# available history is min 4, median 64 days, and 47% of rows have fewer than
# 60. A 90-day window would be mostly clipping. The longest reach here is 60
# days (the lagged 30-day window), which is what this subset can carry.
WINDOW_SPECS = ((7, 0), (14, 0), (30, 0), (30, 30))

# Per-variable aggregations. All are per-DAY rates or extremes, never raw sums:
# see the module docstring on window truncation.
AGGREGATIONS = {
    "eobs_rr": ("rate", "max", "wetfrac"),   # mm/day, wettest day, wet-day fraction
    "eobs_tg": ("mean", "gdd_rate"),         # mean temperature, growing-degree-days/day
    "eobs_tn": ("mean", "min"),
    "eobs_tx": ("mean", "max"),
    "eobs_pp": ("mean", "sd"),               # pressure level and its variability
    "eobs_fg": ("mean", "max"),
    "eobs_hu": ("mean",),
    "eobs_qq": ("mean", "max"),
}

# Growing-degree-day base, degrees C. 5 C is the conventional base for
# temperate grassland and cereals, which is what this tile is.
# A cube-mean NDVI below this in the growing season is CLOUD, not vegetation.
# Bare soil sits near 0.15 and dense summer canopy near 0.85; three frames of
# 1580 on tile 32UNU carry a cube-mean NDVI BELOW ZERO at days of year 177-202,
# at clear fractions of 0.587-0.627. Both filters passed them: the clear-fraction
# rule kept the frames and the per-pixel mask marked the survivors valid, so the
# contamination sits inside the pixels every probe treats as good data. Those
# three frames carry 71% of P3's persistence sum of squares at a 5-day horizon.
#
# ``cube_frame_targets`` REPORTS the screen as a ``frame_plausible`` column and
# never applies it: P2's and P4's published tables were computed without it, and
# a filter that silently changed them would make the two incomparable. A probe
# opts in, and says so on every row.
NDVI_PLAUSIBILITY_FLOOR = 0.15
GROWING_SEASON_DOY = (120, 270)
PLAUSIBILITY_SCREEN_LABEL = (
    "p4_ceiling.cube_frame_targets frame_plausible: a common-masked cube-mean "
    f"NDVI below {NDVI_PLAUSIBILITY_FLOOR} inside days of year "
    f"{GROWING_SEASON_DOY[0]}-{GROWING_SEASON_DOY[1]} is cloud, not "
    "vegetation. APPLIED here: an implausible target frame is excluded, and "
    "cell-level targets inherit their frame's verdict."
)

GDD_BASE_C = 5.0
WET_DAY_MM = 1.0

# The proxy climatology's smoothness. CHOSEN BY THE DAY-OF-YEAR SANITY CONTROL,
# not by taste: 4 harmonics is the LOWEST order at which that control lands at
# zero for all three targets at once. Measured (cube k=5, linear, full8,
# r2_vs_climatology, DOY control at 6 harmonics):
#
#   H      cube_mean                cube_p90                 cell_mean
#          weather   DOY            weather   DOY            weather   DOY
#   2       +0.092  +0.040           +0.132  +0.106           +0.077  +0.010
#   3       +0.069  +0.009           +0.117  +0.055           +0.067  +0.002
#   4       +0.066  +0.007           +0.066  -0.007           +0.067  +0.001
#   5       +0.054  +0.000           +0.068  +0.001           +0.067  -0.001
#
# Read the H=2 row: the "anomaly" still carried enough seasonal cycle for
# day-of-year ALONE to explain 4% of it (10.6% on the 90th percentile), and the
# weather model's number was correspondingly inflated -- most of what looked
# like weather attributability at low order was leftover phenology. This is the
# control doing exactly the job it exists for.
#
# Note the direction: raising the order LOWERS the reported weather number
# everywhere. Selecting H against this control is therefore conservative -- it
# cannot be a way of talking the headline up -- which is what makes it a
# legitimate use of a held-out diagnostic. 4 harmonics resolves a ~91-day
# period, so the curve is still a seasonal climatology and not a smoother of
# individual frames (it absorbs 15-34% of raw variance, printed per fold).
CLIMATOLOGY_HARMONICS = 4
# The day-of-year CONTROL's basis is deliberately RICHER than the detrend's. A
# control built from the detrend's own basis would return zero by least-squares
# construction and would be testing arithmetic rather than the detrend.
# At 8 harmonics it turns positive again on the cube-level targets (+0.045,
# +0.048); that is 17 columns against 264 rows and is reported in log.md rather
# than used, because a control that overfits stops being a floor.
DOY_CONTROL_HARMONICS = 6
# Stage B detrends NOTHING extra: its target already IS the real leave-year-out
# anomaly, so the within-fold "curve" is the training mean (intercept only).
STAGE_B_HARMONICS = 0
DOY_PERIOD_DAYS = 365.25

# encoders.satlas_s2_mi.SatlasS2SwinBMI.window_len. Copied rather than imported
# so this module does not pull in torch; tests/test_p4_ceiling.py asserts the
# two agree, so it cannot drift.
MI_WINDOW_LEN = 8

# Severity is a REPORTING axis. Quantiles of the reference anomaly, so the
# extreme bins hold the 10% driest/wettest-looking frames of the subset.
SEVERITY_QUANTILES = (0.10, 0.30, 0.70, 0.90)
SEVERITY_BINS = ("extreme_low", "low", "near_normal", "high", "extreme_high")

STAGE_A_CLIMATOLOGY_LABEL = (
    "within-season proxy climatology, tile-level, single year (2018), "
    "NOT the leave-year-out definition"
)
STAGE_B_CLIMATOLOGY_LABEL = (
    "leave-target-year-out per-pixel climatology "
    "(data.climatology.ndvi_climatology, GreenEarthNet definition)"
)

_GRID = 4
_GRID_CELLS = _GRID * _GRID    # encoders.base.GRID_CELLS; asserted on the arrays

#: How often ``build_p4_data``'s per-cube loop reports progress. That loop is
#: silent otherwise and, on a 300-cube tile, runs for minutes before the first
#: fit -- long enough that a stall and normal operation look identical.
CUBE_HEARTBEAT_EVERY = 25


# ---------------------------------------------------------------------------
# The proxy climatology, and the constraint that makes it honest
# ---------------------------------------------------------------------------

def doy_design(day_of_year, n_harmonics: int = CLIMATOLOGY_HARMONICS,
               period_days: float = DOY_PERIOD_DAYS) -> np.ndarray:
    """(n, 2K+1) harmonic design matrix in day-of-year: [1, cos k, sin k].

    Periodic by construction, so the curve does not discontinue across 31
    December the way a polynomial in day-of-year would.
    """
    doy = np.asarray(day_of_year, dtype=float)
    assert doy.ndim == 1, f"day_of_year must be 1-D, got {doy.shape}"
    # 0 harmonics is the intercept alone. Stage B uses it on purpose: its target
    # is ALREADY the real leave-year-out anomaly, so a second harmonic detrend
    # would remove a seasonal cycle the climatology has removed once, and the
    # remaining "curve" is just the training mean.
    assert n_harmonics >= 0, f"n_harmonics must be >= 0, got {n_harmonics}"
    cols = [np.ones_like(doy)]
    for k in range(1, int(n_harmonics) + 1):
        w = 2.0 * np.pi * k * doy / float(period_days)
        cols += [np.cos(w), np.sin(w)]
    A = np.column_stack(cols)
    assert A.shape == (doy.size, 2 * int(n_harmonics) + 1), A.shape
    assert np.isfinite(A).all()
    return A


@dataclass(frozen=True)
class DoyCurve:
    """A fitted day-of-year NDVI curve, plus what it was fitted on.

    The provenance fields are not decoration: a curve is only a climatology
    with respect to a training set, and a curve that cannot say which one it
    came from cannot be checked for the leak this class exists to prevent.
    """

    coef: np.ndarray
    n_harmonics: int
    period_days: float
    n_train_rows: int
    doy_train_min: float
    doy_train_max: float
    train_r2: float
    label: str = ""

    def predict(self, day_of_year) -> np.ndarray:
        doy = np.asarray(day_of_year, dtype=float)
        out = doy_design(doy, self.n_harmonics, self.period_days) @ self.coef
        assert out.shape == doy.shape
        assert np.isfinite(out).all(), "the fitted curve produced a non-finite value"
        return out

    def n_extrapolated(self, day_of_year) -> int:
        """Rows outside the training day-of-year range. Reported, not refused:
        a held-out cube legitimately covers days no training cube reached."""
        doy = np.asarray(day_of_year, dtype=float)
        return int(((doy < self.doy_train_min) | (doy > self.doy_train_max)).sum())


def doy_climatology_within_fold(day_of_year, values, train_idx, *,
                                n_harmonics: int = CLIMATOLOGY_HARMONICS,
                                period_days: float = DOY_PERIOD_DAYS,
                                label: str = "", verbose: bool = False) -> DoyCurve:
    """The PROXY climatology: a tile-level day-of-year curve from TRAIN ONLY.

    THE SIGNATURE IS THE GUARANTEE, exactly as ``select_hyperparameter`` is in
    P1. This function is handed the full arrays and a TRAINING INDEX SET, and
    the arrays are read exactly once, through that index, on the two lines
    marked below. There is no code path in which a row outside ``train_idx``
    influences a coefficient -- which is a property a test can check by
    poisoning the held-out rows and asserting the coefficients do not move, and
    ``tests/test_p4_ceiling.py`` does exactly that.

    Why it matters more here than anywhere else in the project: this curve
    defines the TARGET. Fitting it on all 20 cubes and then cross-validating
    the residual leaks the test cubes into the target itself. That inflates
    every number in the table INCLUDING every control, so the margins do not
    reveal it either, and the table stays internally consistent while being
    wrong. It is the one error in this probe that nothing downstream can catch.

    This is a PROXY and not a climatology. It is within-season, single-year and
    tile-level; the real quantity is leave-target-year-out and per-pixel, it is
    not computable on this subset, and ``data.climatology.ndvi_climatology``
    correctly raises rather than guessing. Do not weaken that, and do not let a
    number produced here be labelled with the real definition:
    ``STAGE_A_CLIMATOLOGY_LABEL`` travels with every row that used it.
    """
    doy = np.asarray(day_of_year, dtype=float)
    y = np.asarray(values, dtype=float)
    train_idx = np.asarray(train_idx, dtype=int)
    assert doy.ndim == 1 and y.shape == doy.shape, (
        f"day_of_year {doy.shape} and values {y.shape} must be the same 1-D shape"
    )
    assert train_idx.ndim == 1 and train_idx.size, "empty training index set"
    assert train_idx.min() >= 0 and train_idx.max() < y.size, (
        f"train_idx reaches {train_idx.max()} but only {y.size} rows exist"
    )

    # --- THE ONLY READ OF THE ARRAYS. Everything below touches these two -----
    d_tr = doy[train_idx]
    y_tr = y[train_idx]
    # ------------------------------------------------------------------------

    finite = np.isfinite(y_tr) & np.isfinite(d_tr)
    assert finite.any(), "no finite training value to fit a climatology on"
    d_tr, y_tr = d_tr[finite], y_tr[finite]

    n_par = 2 * int(n_harmonics) + 1
    n_distinct = int(np.unique(d_tr).size)
    assert n_distinct >= n_par, (
        f"the training fold covers only {n_distinct} distinct days of year but "
        f"a {n_harmonics}-harmonic curve has {n_par} parameters. It would "
        "interpolate the training days exactly and the 'anomaly' would be "
        "noise. Use fewer harmonics or a fold mode that leaves more of the "
        "season in train."
    )
    A = doy_design(d_tr, n_harmonics, period_days)
    coef, *_ = np.linalg.lstsq(A, y_tr, rcond=None)
    assert coef.shape == (n_par,) and np.isfinite(coef).all(), (
        f"the climatology fit returned {coef}; the design matrix is singular"
    )
    resid = y_tr - A @ coef
    sst = float(((y_tr - y_tr.mean()) ** 2).sum())
    train_r2 = float(1.0 - (resid ** 2).sum() / sst) if sst > 0 else 0.0

    curve = DoyCurve(coef=coef, n_harmonics=int(n_harmonics),
                     period_days=float(period_days), n_train_rows=int(y_tr.size),
                     doy_train_min=float(d_tr.min()), doy_train_max=float(d_tr.max()),
                     train_r2=train_r2, label=label)
    if verbose:
        print(f"[p4]     doy climatology{' ' + label if label else ''}: fitted on "
              f"{curve.n_train_rows} TRAIN rows over days "
              f"{curve.doy_train_min:.0f}-{curve.doy_train_max:.0f}, "
              f"{n_harmonics} harmonics ({n_par} params), absorbs "
              f"{train_r2:.3f} of train variance")
    return curve


def severity_reference_anomaly(day_of_year, values,
                               n_harmonics: int = CLIMATOLOGY_HARMONICS,
                               period_days: float = DOY_PERIOD_DAYS) -> np.ndarray:
    """Anomaly from a curve fitted on ALL rows. A REPORTING quantity ONLY.

    THIS IS THE ONE FULL-DATA FIT IN THE MODULE AND IT IS DELIBERATE. Severity
    bins have to exist before any model is fitted -- the spec requires their
    edges and counts printed first -- and a bin definition that changed from
    fold to fold would not be a stratification, it would be 30 different ones.

    It is never a target, never a feature, and never enters a score. The
    anomalies that are actually modelled are re-derived inside every fold by
    ``doy_climatology_within_fold``. Assigning a row to a severity bin using
    all the data is a labelling choice about how to REPORT held-out
    performance; using all the data to define what "anomaly" means would be the
    leak, and that is the line this function stays on the right side of.
    """
    doy = np.asarray(day_of_year, dtype=float)
    y = np.asarray(values, dtype=float)
    curve = doy_climatology_within_fold(doy, y, np.arange(y.size),
                                        n_harmonics=n_harmonics,
                                        period_days=period_days,
                                        label="REFERENCE (full data, reporting only)")
    out = y - curve.predict(doy)
    assert out.shape == y.shape
    return out


def severity_bins(anomaly, quantiles: Sequence[float] = SEVERITY_QUANTILES) -> tuple:
    """(labels (n,), edges (len(q),)). Bins DERIVED from the distribution."""
    a = np.asarray(anomaly, dtype=float)
    finite = np.isfinite(a)
    assert finite.any(), "no finite anomaly to bin"
    edges = np.quantile(a[finite], list(quantiles))
    assert (np.diff(edges) > 0).all(), (
        f"severity bin edges are not strictly increasing ({edges}); the anomaly "
        "distribution is degenerate, so every row would land in one bin while "
        f"{len(SEVERITY_BINS)} bin rows were still printed"
    )
    idx = np.digitize(a, edges, right=False)
    labels = np.array([SEVERITY_BINS[min(int(i), len(SEVERITY_BINS) - 1)] for i in idx])
    assert labels.shape == a.shape
    return labels, edges


def print_doy_weather_collinearity(manifest, weather: FeatureSource,
                                   verbose: bool = True) -> dict:
    """How much of the weather is NOT determined by the calendar date.

    THIS IS THE NUMBER THE DAY-OF-YEAR CONTROL HAS TO BE READ AGAINST, and it
    is a property of the acquisition geometry rather than of anything this probe
    does. Measured on the 20-cube subset:

    * All 264 rows land on **36 distinct days of year** over a 235-day span, and
      every one satisfies ``doy % 5 == 2`` -- the cubes share ONE Sentinel-2
      orbit lattice. 261 of 264 rows share their date with another cube; one
      date is observed by all 20.
    * Within a date, the across-cube spread is 9-19% of the total for
      temperature, pressure and radiation (a 150 km tile is one air mass) but
      **77% for precipitation**, which is convective and local.

    So day-of-year is close to a 36-level categorical variable, and it carries a
    large part of the weather signal: for the windowed features the across-cube
    spread is a median 37% of the total, i.e. roughly 63% of a typical weather
    feature is recoverable from the date alone.

    The consequence is specific and it changes how one row of the table is read.
    A LINEAR day-of-year control can only express smooth seasonality, which is
    exactly what the proxy climatology is supposed to have removed -- so a value
    near zero there is the sanity check the spec asks for, and it is what we
    observe (|r2| <= 0.019 everywhere). A FLEXIBLE day-of-year control (the
    boosted tree, the MLP) can instead fit a per-date mean over 36 levels, which
    on this subset is most of a weather model expressed in a different basis. Its
    large values (up to +0.42) are therefore NOT evidence that the detrend
    failed; they are evidence that date and weather are substantially collinear
    here. Cite the linear row for the sanity check.
    """
    doy = manifest["day_of_year"].to_numpy().astype(int)
    uniq, counts = np.unique(doy, return_counts=True)
    lattice = np.unique(doy % 5)
    X = weather.values
    tot = X.std(axis=0)
    resid = np.concatenate([X[doy == d] - X[doy == d].mean(axis=0) for d in uniq])
    ratio = resid.std(axis=0) / np.maximum(tot, 1e-12)
    out = {"n_distinct_doy": int(uniq.size), "doy_span": int(doy.max() - doy.min()),
           "one_orbit_lattice": bool(lattice.size == 1),
           "rows_on_a_shared_date": int(counts[counts > 1].sum()),
           "across_cube_share_median": float(np.median(ratio)),
           "across_cube_share_min": float(ratio.min()),
           "across_cube_share_max": float(ratio.max())}
    if verbose:
        print(f"[p4] day-of-year vs weather, BEFORE anything is fitted:")
        print(f"[p4]   {len(manifest)} rows land on {uniq.size} DISTINCT days of "
              f"year over a {out['doy_span']}-day span; "
              f"{out['rows_on_a_shared_date']}/{len(manifest)} share their date "
              f"with another cube (max {counts.max()} cubes on one date)")
        if out["one_orbit_lattice"]:
            print(f"[p4]   every row satisfies doy % 5 == {lattice[0]}: the cubes "
                  "share ONE Sentinel-2 orbit lattice, so day-of-year is close "
                  f"to a {uniq.size}-level categorical variable")
        print(f"[p4]   across-cube spread of the windowed features, as a "
              f"fraction of their total spread: min {ratio.min():.2f} median "
              f"{np.median(ratio):.2f} max {ratio.max():.2f}")
        print(f"[p4]   => roughly {1 - np.median(ratio):.0%} of a typical weather "
              "feature is recoverable from the DATE alone. Read the LINEAR "
              "day-of-year control as the detrend sanity check; a flexible one "
              "fits a per-date mean, which here is most of a weather model in "
              "another basis.")
    return out


def print_severity_bins(anomaly, label: str = "") -> tuple:
    """Print the edges and the counts BEFORE anything is fitted, and return them."""
    labels, edges = severity_bins(anomaly)
    a = np.asarray(anomaly, dtype=float)
    print(f"[p4] severity bins{' for ' + label if label else ''} -- from the "
          f"REFERENCE anomaly distribution ({a.size} rows, sd {a.std():.4f}), "
          "a reporting axis only:")
    print(f"[p4]   edges at quantiles {list(SEVERITY_QUANTILES)} = "
          + ", ".join(f"{e:+.4f}" for e in edges))
    for b in SEVERITY_BINS:
        n = int((labels == b).sum())
        sub = a[labels == b]
        rng = f"[{sub.min():+.4f}, {sub.max():+.4f}]" if n else "empty"
        print(f"[p4]   {b:<14} {n:>5} rows ({n / a.size:5.1%})  {rng}")
    return labels, edges


# ---------------------------------------------------------------------------
# Targets: spatially aggregated NDVI, never pixel-wise
# ---------------------------------------------------------------------------


def _frame_plausible(cube_mean, timestamps) -> np.ndarray:
    """The one physical-plausibility definition shared by every P4 stage."""
    timestamps = np.asarray(timestamps, dtype="datetime64[D]")
    years = timestamps.astype("datetime64[Y]").astype(int) + 1970
    doy = (timestamps
           - np.asarray([f"{y}-01-01" for y in years],
                        dtype="datetime64[D]")).astype(int) + 1
    in_season = ((doy >= GROWING_SEASON_DOY[0])
                 & (doy <= GROWING_SEASON_DOY[1]))
    verdict = ~(in_season
                & (np.asarray(cube_mean) < NDVI_PLAUSIBILITY_FLOOR))
    assert verdict.shape == np.asarray(cube_mean).shape
    return verdict


def cube_frame_targets(sample, min_clear: float = MIN_CLEAR_FRACTION,
                       grid: int = _GRID, verbose: bool = False) -> dict:
    """NDVI aggregates per RETAINED frame of ONE cube. Never a pixel.

    NDVI comes through ``data.loader.cube_ndvi``, i.e. through the canonical
    ``data.ndvi.ndvi``, which returns NaN at masked and non-finite pixels. Every
    aggregate here is NaN-aware, so a cloudy pixel is invisible to the mean
    rather than counted as zero.

    Returns cube_mean (T_kept,), cube_p90 (T_kept,), cell_mean (T_kept, 16),
    cell_valid (T_kept, 16) bool and frame_plausible (T_kept,) bool. The
    verdict is reported, never applied here. Cells are ROW-MAJOR, the same order
    as ``encoders.frames.grid_clear_fraction`` and
    ``encoders.base.pool_to_grid``, so a cell row lines up with its land-cover
    label and its clear fraction.
    """
    from data.loader import CubeSample, cube_ndvi

    sel = select_clear_frames(sample.values, sample.timestamps, sample.mask,
                              min_clear=min_clear, verbose=False)
    kept = CubeSample(values=sel.values, timestamps=sel.timestamps, mask=sel.mask,
                      path=sample.path, bands=sample.bands)
    nd = cube_ndvi(kept)                      # (T_kept, H, W), NaN where masked
    T, H, W = nd.shape
    assert H % grid == 0 and W % grid == 0, f"{H}x{W} not divisible by {grid}"
    ch, cw = H // grid, W // grid

    with warnings.catch_warnings():
        # An all-masked GRID CELL is a real state (a frame at 0.6 clear can hold
        # a cell at 0.0); it is reported through cell_valid and dropped, never
        # filled. numpy's "Mean of empty slice" would otherwise be one warning
        # per cell per cube.
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        cube_mean = np.nanmean(nd.reshape(T, -1), axis=1)
        cube_p90 = np.nanpercentile(nd.reshape(T, -1), 90, axis=1)
        cells = np.nanmean(nd.reshape(T, grid, ch, grid, cw), axis=(2, 4))
    cell_mean = cells.reshape(T, grid * grid)
    cell_valid = np.isfinite(cell_mean)

    assert cube_mean.shape == cube_p90.shape == (T,)
    assert cell_mean.shape == (T, grid * grid)
    assert np.isfinite(cube_mean).all() and np.isfinite(cube_p90).all(), (
        "a retained frame has no valid pixel at all, which the clear-fraction "
        "rule should have removed"
    )
    # The physical-plausibility screen: REPORTED, never applied here. See
    # NDVI_PLAUSIBILITY_FLOOR for why it exists and why it is not a filter.
    frame_plausible = _frame_plausible(cube_mean, sel.timestamps)
    assert frame_plausible.shape == (T,), frame_plausible.shape

    gcf = grid_clear_fraction(sel.mask, grid=grid)
    assert gcf.shape == cell_mean.shape
    # The two disagree only where a cell is clear but every clear pixel is
    # non-finite; finite_valid_mask already demoted those, so this must hold.
    assert (cell_valid == (gcf > 0)).all(), (
        "a grid cell has clear pixels but no finite NDVI (or the reverse); the "
        "mask used for the target and the mask used for the clear fraction "
        "disagree"
    )
    if verbose:
        print(f"[p4] {os.path.basename(sample.path)}: NDVI {nd.shape} -> "
              f"cube_mean {cube_mean.shape} cube_p90 {cube_p90.shape} "
              f"cell_mean {cell_mean.shape}, "
              f"{int((~cell_valid).sum())}/{cell_valid.size} cells fully masked")
    return {"cube_mean": cube_mean, "cube_p90": cube_p90,
            "cell_mean": cell_mean, "cell_valid": cell_valid,
            "grid_clear_frac": gcf, "clear_frac": sel.clear_frac,
            "frame_plausible": frame_plausible,
            "timestamps": sel.timestamps, "kept_idx": sel.kept_idx}


# ---------------------------------------------------------------------------
# Weather, windowed over the DAILY axis
# ---------------------------------------------------------------------------

def _aggregate(window: np.ndarray, kind: str) -> float:
    """One aggregate of one variable over one window. Per-day where it sums."""
    n = window.size
    assert n, "empty weather window"
    if kind == "mean":
        return float(window.mean())
    if kind == "min":
        return float(window.min())
    if kind == "max":
        return float(window.max())
    if kind == "sd":
        return float(window.std())
    if kind == "rate":                       # sum / days available
        return float(window.sum() / n)
    if kind == "wetfrac":
        return float((window > WET_DAY_MM).mean())
    if kind == "gdd_rate":                   # growing-degree-days per day
        return float(np.maximum(window - GDD_BASE_C, 0.0).sum() / n)
    raise AssertionError(f"unknown aggregation {kind!r}")


def cube_weather_windows(path: str, frame_timestamps,
                         variables: Sequence[str] = EOBS_VARS,
                         specs: Sequence[tuple] = WINDOW_SPECS,
                         verbose: bool = False) -> tuple:
    """(X (T_kept, D), names (D,), days_available (T_kept, n_specs)) for ONE cube.

    Windows are on the cube's DAILY axis, positioned by the frame's TIMESTAMP
    through ``encoders.manifest.frame_daily_positions``. Never over retained
    frames: a frame-defined window is longer in cloudy weather, which would make
    the window itself a weather feature.

    ``days_available`` is RETURNED so truncation at the start of a cube can be
    reported -- and is deliberately NOT a feature, because it correlates with
    position-in-cube and therefore with time of year.
    """
    daily_axis = cube_daily_axis(path)
    weather = cube_weather(path)
    missing = [v for v in variables if v not in weather]
    assert not missing, (
        f"{os.path.basename(path)} has no {missing} in its E-OBS stack; it "
        f"carries {sorted(weather)}"
    )
    for v in variables:
        assert weather[v].shape == (daily_axis.size,), (
            f"{v} is {weather[v].shape} against a {daily_axis.size}-day axis"
        )
        assert np.isfinite(weather[v]).all(), f"{v} has gaps in {path}"
    pos = frame_daily_positions(daily_axis, frame_timestamps)

    T, cols, names = pos.size, [], []
    avail = np.zeros((T, len(specs)), dtype=int)
    for s, (span, lag) in enumerate(specs):
        assert span >= 1 and lag >= 0, f"bad window spec {(span, lag)}"
        # CLIPPED AT THE START OF THE CUBE, never wrapped and never filled.
        # There is no weather before the file begins, so a window that reaches
        # past it collapses toward the oldest day on record. That costs
        # precision -- the aggregate is a rate over fewer days -- and not scale,
        # which is the whole reason the aggregates are rates. The prevalence is
        # printed by weather_source; it is disclosed, not imputed.
        hi = np.maximum(pos - lag, 0)           # inclusive right edge
        lo = np.maximum(0, hi - span + 1)
        assert (lo <= hi).all() and (hi < daily_axis.size).all()
        avail[:, s] = hi - lo + 1
        for v in variables:
            series = weather[v]
            for kind in AGGREGATIONS[v]:
                col = np.array([_aggregate(series[lo[t]:hi[t] + 1], kind)
                                for t in range(T)], dtype=float)
                cols.append(col)
                names.append(f"{v[len('eobs_'):]}_{kind}_{span}d"
                             + (f"_lag{lag}" if lag else ""))
    X = np.column_stack(cols) if cols else np.zeros((T, 0))
    assert X.shape == (T, len(names)), (X.shape, len(names))
    assert np.isfinite(X).all(), "a windowed weather feature is non-finite"
    assert len(set(names)) == len(names), "duplicate weather feature name"
    if verbose:
        print(f"[p4] {os.path.basename(path)}: weather windows {X.shape}, "
              f"days available {avail.min()}-{avail.max()}")
    return X, tuple(names), avail


# ---------------------------------------------------------------------------
# The assembled inputs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TargetRows:
    """One target's rows: the RAW aggregate, its manifest row, and its stratum.

    ``y`` is the raw NDVI aggregate and NOT the anomaly, on purpose. The anomaly
    is fold-dependent -- it is the raw value minus a curve fitted on that fold's
    training cubes -- so an anomaly stored here would be an anomaly with respect
    to some other fold, or to all of them, which is the leak.
    """

    name: str
    level: str
    y: np.ndarray
    row_idx: np.ndarray
    cell_idx: np.ndarray
    stratum: np.ndarray

    def __post_init__(self):
        n = self.y.shape[0]
        assert self.level in ("frame", "cell"), self.level
        for f in ("row_idx", "cell_idx", "stratum"):
            assert getattr(self, f).shape == (n,), (
                f"{self.name}: {f} {getattr(self, f).shape} does not describe "
                f"y ({n},)"
            )
        assert np.isfinite(self.y).all(), f"{self.name}: non-finite target"

    @property
    def n_rows(self) -> int:
        return int(self.y.shape[0])


@dataclass(frozen=True)
class FeatureSource:
    """A block of columns, and whether it lives per FRAME or per feature row.

    ``frame_level`` is the whole point. Weather is one vector per (cube, frame)
    and is EXPANDED to the 16 cells of that frame, which is why cell rows are
    pseudo-replicates for the feature side. The per-cell clear fraction is not
    frame-level and is stored per row. ``permutable`` marks the blocks the
    permutation control is allowed to shuffle: the weather, and only the
    weather.
    """

    name: str
    values: np.ndarray
    names: tuple
    frame_level: bool
    permutable: bool = False

    def __post_init__(self):
        assert self.values.ndim == 2, f"{self.name}: {self.values.shape}"
        assert self.values.shape[1] == len(self.names), (
            f"{self.name}: {self.values.shape[1]} columns but "
            f"{len(self.names)} names"
        )
        assert np.isfinite(self.values).all(), f"{self.name}: non-finite feature"

    def expand(self, row_idx: np.ndarray, perm_map=None) -> np.ndarray:
        if not self.frame_level:
            assert self.values.shape[0] == row_idx.shape[0], (
                f"{self.name}: per-row source has {self.values.shape[0]} rows "
                f"against {row_idx.shape[0]} feature rows"
            )
            return self.values
        idx = row_idx if (perm_map is None or not self.permutable) \
            else np.asarray(perm_map, dtype=int)[row_idx]
        return self.values[idx]


@dataclass(frozen=True)
class P4Data:
    """Everything Stage A fits on, assembled once from the cubes."""

    manifest: object
    targets: dict                 # name -> TargetRows
    weather: dict                 # feature set -> FeatureSource (frame level)
    observation_frame: FeatureSource
    observation_cell: FeatureSource
    doy_frame: FeatureSource
    day_of_year: np.ndarray       # (n_manifest,)
    cube_id: np.ndarray           # (n_manifest,)
    days_available: np.ndarray    # (n_manifest, n_specs)
    reference_anomaly: dict       # target name -> (n_rows,) reporting anomaly
    severity: dict                # target name -> (labels, edges)
    plausibility_screen: bool = False
    n_implausible_frames: int = 0
    n_rows_dropped_implausible: dict | None = None

    def observation(self, level: str) -> FeatureSource:
        return self.observation_frame if level == "frame" else self.observation_cell


def target_rows(name: str, per_cube: dict, manifest, verbose: bool = False) -> TargetRows:
    """Assemble one target's rows across cubes, in MANIFEST ROW ORDER."""
    level = TARGET_LEVEL[name]
    n = len(manifest)
    strat_cube = manifest["landcover_stratum"].to_numpy().astype(str)
    grid_lc = manifest["grid_landcover"].to_numpy()
    order = np.concatenate([
        np.flatnonzero((manifest["cube_id"] == c).to_numpy())
        for c in per_cube["__order__"]
    ])
    back = np.argsort(order, kind="stable")
    assert order[back].tolist() == list(range(n)), (
        "per-cube target rows are not a permutation of the manifest"
    )

    if level == "frame":
        y = np.concatenate([per_cube[c][name]
                            for c in per_cube["__order__"]])[back]
        assert y.shape == (n,), f"{name}: {y.shape} against {n} manifest rows"
        rows = np.arange(n)
        cells = np.full(n, -1, dtype=int)
        strata = strat_cube
    else:
        vals = np.concatenate([per_cube[c][name]
                               for c in per_cube["__order__"]])[back]
        valid = np.concatenate([per_cube[c]["cell_valid"]
                                for c in per_cube["__order__"]])[back]
        assert vals.shape == valid.shape == (n, _GRID_CELLS), vals.shape
        rows_all = np.repeat(np.arange(n), _GRID_CELLS)
        cells_all = np.tile(np.arange(_GRID_CELLS), n)
        keep = np.flatnonzero(valid.reshape(-1))
        y = vals.reshape(-1)[keep]
        rows, cells = rows_all[keep], cells_all[keep]
        strata = np.array([str(grid_lc[r][c]) for r, c in zip(rows, cells)])
        if verbose:
            dropped = valid.size - keep.size
            print(f"[p4] {name}: {valid.size} cells -> {keep.size} rows "
                  f"({dropped} dropped: no valid pixel in the cell, so not an "
                  "observation -- nothing is filled)")
    return TargetRows(name=name, level=level, y=y, row_idx=rows,
                      cell_idx=cells, stratum=strata)


def manifest_frame_plausible(manifest, per_cube: dict,
                             verbose: bool = True) -> np.ndarray:
    """Return P4's per-frame verdict in manifest-row order.

    ``cube_frame_targets`` is the only place that defines physical
    plausibility. This helper only aligns those already-computed booleans. The
    alignment is asserted because attaching a verdict to the neighbouring
    frame would preserve every shape and silently screen the wrong target.
    """
    order, oai = [], manifest["original_axis_index"].to_numpy().astype(int)
    for cube in per_cube["__order__"]:
        pos = np.flatnonzero((manifest["cube_id"] == cube).to_numpy())
        verdict = np.asarray(per_cube[cube]["frame_plausible"], dtype=bool)
        assert pos.size == verdict.size, (
            f"{cube}: manifest has {pos.size} rows but frame_plausible has "
            f"{verdict.size}"
        )
        assert (np.diff(oai[pos]) > 0).all(), (
            f"{cube}: manifest rows are not ascending in original_axis_index; "
            "frame plausibility cannot be aligned by position"
        )
        order.append(pos)
    order = np.concatenate(order)
    back = np.argsort(order, kind="stable")
    assert order[back].tolist() == list(range(len(manifest))), (
        "per-cube plausibility rows are not a permutation of the manifest"
    )
    verdict = np.concatenate([
        np.asarray(per_cube[c]["frame_plausible"], dtype=bool)
        for c in per_cube["__order__"]
    ])[back]
    assert verdict.shape == (len(manifest),), verdict.shape
    if verbose:
        print(f"[p4] plausibility screen source: {int((~verdict).sum())}/"
              f"{verdict.size} retained frames fail")
    return verdict


def apply_plausibility_screen(targets: dict,
                              observation_cell: FeatureSource,
                              frame_plausible) -> tuple:
    """Filter target rows, plus the row-aligned cell observation control.

    Frame-level feature sources remain manifest-sized and are indexed by each
    surviving target's ``row_idx``. The cell observation source is different:
    it has one row per already-valid target cell, so it must receive the exact
    same keep mask as ``cell_mean`` or the control and target silently diverge.

    Returns ``(screened_targets, screened_observation_cell, dropped_by_target)``.
    """
    ok = np.asarray(frame_plausible, dtype=bool)
    assert ok.ndim == 1, ok.shape
    screened, dropped = {}, {}
    cell_keep = None
    for name, tr in targets.items():
        assert tr.row_idx.size == tr.n_rows
        assert (tr.row_idx >= 0).all() and (tr.row_idx < ok.size).all()
        keep = ok[tr.row_idx]
        assert keep.any(), f"{name}: plausibility screen removed every target row"
        screened[name] = TargetRows(
            name=tr.name,
            level=tr.level,
            y=tr.y[keep],
            row_idx=tr.row_idx[keep],
            cell_idx=tr.cell_idx[keep],
            stratum=tr.stratum[keep],
        )
        dropped[name] = int((~keep).sum())
        if name == "cell_mean":
            cell_keep = keep

    assert cell_keep is not None, "targets contain no cell_mean row set"
    assert observation_cell.frame_level is False
    assert observation_cell.values.shape[0] == cell_keep.size, (
        f"cell observation source has {observation_cell.values.shape[0]} rows "
        f"but cell_mean has {cell_keep.size}; they cannot share a screen mask"
    )
    screened_observation = FeatureSource(
        name=observation_cell.name,
        values=observation_cell.values[cell_keep],
        names=observation_cell.names,
        frame_level=False,
        permutable=observation_cell.permutable,
    )
    assert screened_observation.values.shape[0] == screened["cell_mean"].n_rows
    return screened, screened_observation, dropped


def weather_source(manifest, cube_dir: str, feature_set: str,
                   verbose: bool = True) -> tuple:
    """Frame-level windowed weather for every manifest row, in row order."""
    variables = FEATURE_SET_VARS[feature_set]
    order = sorted(manifest["cube_id"].unique().tolist())
    blocks, avail, rows, names = [], [], [], None
    for cube in order:
        pos = np.flatnonzero((manifest["cube_id"] == cube).to_numpy())
        sub = manifest.iloc[pos]
        X, nm, av = cube_weather_windows(os.path.join(cube_dir, str(cube)),
                                         sub["timestamp"].to_numpy(),
                                         variables=variables)
        names = nm if names is None else names
        assert nm == names, f"{cube}: weather feature names differ between cubes"
        blocks.append(X)
        avail.append(av)
        rows.append(pos)
    rows = np.concatenate(rows)
    order_back = np.argsort(rows, kind="stable")
    assert rows[order_back].tolist() == list(range(len(manifest))), (
        "the per-cube weather blocks are not a permutation of the manifest rows"
    )
    W = np.concatenate(blocks)[order_back]
    A = np.concatenate(avail)[order_back]
    assert W.shape == (len(manifest), len(names)), W.shape
    src = FeatureSource(name=feature_set, values=W, names=names,
                        frame_level=True, permutable=True)
    if verbose:
        print(f"[p4] {feature_set:<14} {W.shape} = {len(variables)} E-OBS "
              f"variables x {sum(len(AGGREGATIONS[v]) for v in variables)} "
              f"aggregations x {len(WINDOW_SPECS)} windows, over the DAILY axis")
        for s, (span, lag) in enumerate(WINDOW_SPECS):
            trunc = int((A[:, s] < span).sum())
            print(f"[p4]   window span={span:>2}d lag={lag:>2}d: days available "
                  f"min {A[:, s].min():>3} median {int(np.median(A[:, s])):>3} "
                  f"max {A[:, s].max():>3} | TRUNCATED on {trunc:>3}/"
                  f"{len(manifest)} rows ({trunc / len(manifest):5.1%})")
        print("[p4]   truncation is clipping at the start of the cube, never "
              "filling. Aggregates are per-day RATES so it costs precision and "
              "not scale, and days-available is deliberately NOT a feature: it "
              "correlates with position-in-cube, i.e. with time of year.")
    return src, A


def observation_source(manifest, per_cube: dict, level: str,
                       verbose: bool = True) -> FeatureSource:
    """P1's degenerate control, carried forward. No weather, no image.

    ``clear_frac`` (or ``grid_clear_frac`` at cell level) and
    ``window_span_days``. The lookback is RECOMPUTED here from the manifest
    timestamps via ``encoders.pipeline.window_span_days`` -- the same function
    the Phase 1.2 cache was written with -- so this phase reads no embedding.
    """
    order = per_cube["__order__"]
    n = len(manifest)
    wsd_blocks, rows = [], []
    for cube in order:
        pos = np.flatnonzero((manifest["cube_id"] == cube).to_numpy())
        sub = manifest.iloc[pos]
        ts = np.asarray(sub["timestamp"].to_numpy(), dtype="datetime64[ns]")
        assert (np.diff(ts) > np.timedelta64(0, "ns")).all(), (
            f"{cube}: manifest rows are not in time order, so a lookback "
            "computed over them would be wrong"
        )
        wsd_blocks.append(window_span_days(ts, MI_WINDOW_LEN))
        rows.append(pos)
    rows = np.concatenate(rows)
    back = np.argsort(rows, kind="stable")
    wsd = np.concatenate(wsd_blocks)[back].astype(float)
    assert wsd.shape == (n,)
    assert wsd.max() > 0 and np.unique(wsd).size >= 3, (
        f"window_span_days is constant (min {wsd.min():.0f} max {wsd.max():.0f}, "
        f"{np.unique(wsd).size} distinct values). The observation control would "
        "silently become clear_frac alone -- half the control it reports -- "
        "while still printing a two-column D=2 row."
    )
    if level == "frame":
        X = np.column_stack([manifest["clear_frac"].to_numpy().astype(float), wsd])
        names = ("clear_frac", "window_span_days")
        frame_level = True
    else:
        gcf = np.concatenate([per_cube[c]["grid_clear_frac"] for c in order])[back]
        valid = np.concatenate([per_cube[c]["cell_valid"] for c in order])[back]
        assert gcf.shape == valid.shape == (n, _GRID_CELLS)
        keep = np.flatnonzero(valid.reshape(-1))
        X = np.column_stack([gcf.reshape(-1)[keep],
                             np.repeat(wsd, _GRID_CELLS)[keep]])
        names = ("grid_clear_frac", "window_span_days")
        frame_level = False
    if verbose:
        print(f"[p4] observation control ({level}): {X.shape} "
              f"{list(names)} -- NO weather. window_span_days recomputed from "
              f"the manifest (min {wsd.min():.0f} median {np.median(wsd):.0f} "
              f"max {wsd.max():.0f} d), not read from the Phase 1.2 cache")
    return FeatureSource(name=f"observation/{level}", values=X, names=names,
                         frame_level=frame_level)


def doy_source(manifest, n_harmonics: int = DOY_CONTROL_HARMONICS,
               verbose: bool = True) -> FeatureSource:
    """The day-of-year sanity control's features: day-of-year and nothing else.

    RICHER than the detrend basis by construction (``DOY_CONTROL_HARMONICS``
    against ``CLIMATOLOGY_HARMONICS``), plus the raw day so a tree can split on
    it. A control drawn from the detrend's own basis would score exactly zero on
    train by least squares and would be testing arithmetic, not the detrend.
    """
    doy = manifest["day_of_year"].to_numpy().astype(float)
    A = doy_design(doy, n_harmonics)
    X = np.column_stack([doy, A[:, 1:]])           # drop the intercept column
    names = ("day_of_year",) + tuple(
        f"{fn}{k}" for k in range(1, n_harmonics + 1) for fn in ("cos", "sin"))
    assert X.shape == (len(manifest), 1 + 2 * n_harmonics)
    if verbose:
        print(f"[p4] day-of-year control: {X.shape} = day_of_year + "
              f"{n_harmonics} harmonics, deliberately RICHER than the "
              f"{CLIMATOLOGY_HARMONICS}-harmonic detrend so a near-zero score "
              "is informative rather than automatic")
    return FeatureSource(name="doy", values=X, names=names, frame_level=True)


def build_p4_data(manifest, cube_dir: str,
                  feature_sets: Sequence[str] = FEATURE_SETS,
                  verbose: bool = True,
                  plausibility_screen: bool = False) -> P4Data:
    """Assemble every target, feature block and reporting axis, once.

    ``feature_sets`` is a PARAMETER rather than a read of the module constant
    so a cross-tile run can select a different weather subset without mutating
    module state. A run that reassigned ``FEATURE_SETS`` at import time would
    change what every other caller in the same process built, which is the kind
    of action-at-a-distance this project's artefacts are otherwise free of.

    ``plausibility_screen`` is opt-in so the published unscreened table remains
    reproducible. It filters target rows, never manifest rows; the weather join
    and every frozen cache therefore keep their original indexing contract.
    """
    from data.loader import load_cube

    order = sorted(manifest["cube_id"].unique().tolist())
    per_cube = {"__order__": order}
    # This loop reads every cube off disk and is the longest stretch before any
    # fitting starts. At 20 cubes it is a few seconds and silence costs
    # nothing; at 300+ it is minutes, and a silent minutes-long stretch is
    # indistinguishable from a hang. The heartbeat is throttled rather than
    # per-cube so the log stays readable at every scale.
    t_cubes = time.time()
    for i, cube in enumerate(order, start=1):
        sample = load_cube(os.path.join(cube_dir, str(cube)), verbose=False)
        t = cube_frame_targets(sample)
        sub = manifest[manifest["cube_id"] == cube]
        assert t["cube_mean"].shape[0] == len(sub), (
            f"{cube}: {t['cube_mean'].shape[0]} retained frames from the cube "
            f"but {len(sub)} manifest rows -- the manifest and the cube on disk "
            "disagree about frame selection"
        )
        assert (np.asarray(t["timestamps"], dtype="datetime64[ns]")
                == np.asarray(sub["timestamp"].to_numpy(), dtype="datetime64[ns]")).all(), (
            f"{cube}: target timestamps do not match the manifest's"
        )
        per_cube[cube] = t
        if verbose and (i % CUBE_HEARTBEAT_EVERY == 0 or i == len(order)):
            dt = time.time() - t_cubes
            rate = i / dt if dt > 0 else float("nan")
            print(f"[p4]   targets {i}/{len(order)} cubes, {dt:.0f}s elapsed, "
                  f"{rate:.2f} cubes/s, ETA {(len(order) - i) / rate:.0f}s",
                  flush=True)
    if verbose:
        print(f"[p4] targets built from {len(order)} cubes via data.ndvi.ndvi "
              "(canonical), spatially aggregated -- never pixel-wise")

    targets = {name: target_rows(name, per_cube, manifest, verbose=verbose)
               for name in TARGETS}
    weather = {}
    days_avail = None
    for fs in feature_sets:
        src, A = weather_source(manifest, cube_dir, fs, verbose=verbose)
        weather[fs] = src
        days_avail = A if days_avail is None else days_avail

    obs_frame = observation_source(manifest, per_cube, "frame", verbose=verbose)
    obs_cell = observation_source(manifest, per_cube, "cell", verbose=verbose)
    doy_src = doy_source(manifest, verbose=verbose)

    frame_ok = manifest_frame_plausible(manifest, per_cube, verbose=verbose)
    n_implausible = int((~frame_ok).sum())
    dropped = {name: 0 for name in TARGETS}
    if plausibility_screen:
        targets, obs_cell, dropped = apply_plausibility_screen(
            targets, obs_cell, frame_ok)
        if verbose:
            print("[p4] plausibility screen APPLIED: "
                  + ", ".join(f"{name} -{dropped[name]} rows"
                              for name in TARGETS))
    elif verbose:
        print("[p4] plausibility screen OFF: frame verdicts were measured but "
              "no target row was removed")

    doy = manifest["day_of_year"].to_numpy().astype(float)
    ref, sev = {}, {}
    for name, tr in targets.items():
        ref[name] = severity_reference_anomaly(doy[tr.row_idx], tr.y)
        sev[name] = severity_bins(ref[name])

    data = P4Data(manifest=manifest, targets=targets, weather=weather,
                  observation_frame=obs_frame, observation_cell=obs_cell,
                  doy_frame=doy_src, day_of_year=doy,
                  cube_id=manifest["cube_id"].to_numpy().astype(str),
                  days_available=days_avail, reference_anomaly=ref, severity=sev,
                  plausibility_screen=bool(plausibility_screen),
                  n_implausible_frames=n_implausible,
                  n_rows_dropped_implausible=dropped)
    for name in TARGETS:
        assert_weather_constant_across_cells(data, name, verbose=verbose)
    return data


def assert_weather_constant_across_cells(data: P4Data, target: str,
                                         verbose: bool = False) -> None:
    """Weather must be identical across the 16 cells of any one frame.

    A PROPERTY OF THE DATA -- one cube, one day, one E-OBS reading -- and the
    reason all uncertainty here is clustered at the CUBE level: the 16 cells of
    a frame are pseudo-replicates on the feature side, adding target variance
    and no feature variance at all.

    WHAT IS ACTUALLY BEING TESTED, stated precisely, because it would be easy to
    write a version of this that passes vacuously. For a source declared
    ``frame_level`` the constancy is true BY CONSTRUCTION -- ``X = W[row_idx]``
    cannot vary within a group of equal ``row_idx``, so comparing X against
    itself proves nothing. The two things that can really go wrong, and that
    this does check:

      1. **A per-row array declared frame-level, or a frame-level array built
         per row.** The weather then varies within a frame, which is exactly the
         signature of weather that was expanded to cells too early and then
         re-ordered by a slice. Checked by measuring the within-frame spread of
         every source, whatever it declares itself to be.
      2. **A malformed row_idx / cell_idx pairing.** A frame carrying 17 cells,
         a duplicated (frame, cell) pair, or a cell index out of range means the
         expansion is not the 16-to-1 map it claims. Checked structurally, and
         cross-checked against the manifest, which is an independently built
         record of how many frames there are.

    Neither changes a shape, a dtype or a finiteness check, which is why they
    are assertions and not comments.
    """
    tr = data.targets[target]
    if tr.level != "cell":
        return
    n_manifest = len(data.manifest)

    # (2) structure, against the manifest rather than against itself.
    assert (tr.cell_idx >= 0).all() and (tr.cell_idx < _GRID_CELLS).all(), (
        f"{target}: cell index outside [0, {_GRID_CELLS}); the expansion is not "
        "over this grid"
    )
    assert (tr.row_idx >= 0).all() and (tr.row_idx < n_manifest).all(), (
        f"{target}: row_idx reaches outside the manifest's {n_manifest} rows"
    )
    pair = tr.row_idx.astype(np.int64) * _GRID_CELLS + tr.cell_idx
    assert np.unique(pair).size == pair.size, (
        f"{target}: a (frame, cell) pair appears more than once, so some cell "
        "is counted twice and some cell is missing. The expansion is not a "
        "16-to-1 map onto the manifest."
    )
    counts = np.bincount(tr.row_idx, minlength=n_manifest)
    assert counts.max() <= _GRID_CELLS, (
        f"{target}: a frame carries {counts.max()} cell rows, more than the "
        f"{_GRID_CELLS} the grid has"
    )

    # (1) within-frame spread of every source, measured not assumed.
    order = np.argsort(tr.row_idx, kind="stable")
    starts = np.flatnonzero(np.diff(tr.row_idx[order], prepend=-1))
    for label, src in (list(data.weather.items())
                       + [("doy", data.doy_frame)]):
        X = src.expand(tr.row_idx)[order]
        # Max minus min within each contiguous run of one row_idx.
        hi = np.maximum.reduceat(X, starts, axis=0)
        lo = np.minimum.reduceat(X, starts, axis=0)
        d = float(np.abs(hi - lo).max())
        assert d == 0.0, (
            f"{target}/{label}: the feature varies by up to {d:.4g} ACROSS THE "
            f"CELLS OF A SINGLE FRAME. Weather (and day-of-year) are one value "
            "per (cube, frame), so this means the cell rows are carrying "
            "another frame's values -- the join is wrong, and every number "
            "downstream would be a confident wrong number of the same shape."
        )
    if verbose:
        print(f"[p4] weather is CONSTANT across the cells of every frame "
              f"({tr.n_rows} {target} rows over {int((counts > 0).sum())} "
              f"frames, {counts.max()} cells max): cell rows are "
              "PSEUDO-REPLICATES on the feature side -- they add target "
              "variance and no feature variance, so uncertainty is clustered "
              "at the cube level whatever the row count")


# ---------------------------------------------------------------------------
# The permutation control's shuffle
# ---------------------------------------------------------------------------

def permute_within_fold(cube_id: np.ndarray, rows: np.ndarray, seed: int) -> tuple:
    """Map each manifest row on one fold side to another row's weather.

    Permutes at the MANIFEST-ROW level, not the feature-row level, so the 16
    cells of a frame keep sharing a sky -- the permutation is meant to destroy
    the cube-to-weather association, not the pseudo-replicate structure that
    makes cell rows what they are.

    A uniform permutation leaves a row on its own cube with probability
    sum_c (n_c/n)^2, about 5% at 20 cubes; the realised fraction is returned so
    the caller can report and assert it rather than hope. When a fold side holds
    ONE cube (LOCO's test side) "across cubes" is undefined and the shuffle is
    within that cube -- the row-level association is still destroyed, which is
    what the control measures.
    """
    rows = np.asarray(rows, dtype=int)
    rng = np.random.default_rng(int(seed))
    perm = rows[rng.permutation(rows.size)]
    same = float((cube_id[rows] == cube_id[perm]).mean())
    return perm, same


def _perm_map(cube_id, train_rows, test_rows, n_manifest, seed) -> tuple:
    """Identity everywhere except the two fold sides, each shuffled separately."""
    m = np.arange(n_manifest)
    p_tr, s_tr = permute_within_fold(cube_id, train_rows, seed)
    p_te, s_te = permute_within_fold(cube_id, test_rows, seed + 1)
    m[np.asarray(train_rows, dtype=int)] = p_tr
    m[np.asarray(test_rows, dtype=int)] = p_te
    return m, max(s_tr, s_te)


def build_X(sources: Sequence[FeatureSource], row_idx: np.ndarray,
            perm_map=None) -> np.ndarray:
    X = np.column_stack([s.expand(row_idx, perm_map) for s in sources])
    X = np.ascontiguousarray(X, dtype=np.float64)
    assert X.shape[0] == row_idx.shape[0]
    assert np.isfinite(X).all()
    return X


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def gaussian_crps(y, mu, sigma) -> np.ndarray:
    """CRPS of a Gaussian predictive distribution, closed form."""
    from scipy.stats import norm

    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = float(max(float(sigma), 1e-9))
    z = (y - mu) / sigma
    out = sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    assert out.shape == y.shape and np.isfinite(out).all()
    return out


def ensemble_crps(y, ensemble) -> np.ndarray:
    """CRPS of an EMPIRICAL ensemble forecast, exact.

    CRPS = E|X - y| - 0.5 E|X - X'|, both terms computed exactly from the sorted
    ensemble rather than sampled. Used with the TRAIN anomalies as the ensemble:
    that is the spread the climatology implies for a new observation, so this is
    the climatology-ensemble baseline the metric is scored against.
    """
    x = np.sort(np.asarray(ensemble, dtype=float))
    y = np.asarray(y, dtype=float)
    m = x.size
    assert m >= 2, "a climatology ensemble needs at least 2 members"
    cs = np.concatenate([[0.0], np.cumsum(x)])
    k = np.searchsorted(x, y, side="right")
    e1 = ((k * y - cs[k]) + ((cs[m] - cs[k]) - (m - k) * y)) / m
    i = np.arange(m, dtype=float)
    e2 = 2.0 * float(((2 * i - m + 1) * x).sum()) / (m * m)
    out = e1 - 0.5 * e2
    assert out.shape == y.shape and np.isfinite(out).all()
    return out


def fold_clustered_ci(values, alpha: float = 0.05) -> tuple:
    """(lo, hi) for the mean over FOLDS, which are the clusters.

    Folds hold disjoint sets of cubes, so the per-fold scores are the
    independent replicates here -- not the rows, of which there are 264 or 4224
    and which share skies and places. A t interval on 5 or 20 of them is wide,
    and that width IS the result at this sample size.
    """
    from scipy.stats import t as student_t

    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return (float("nan"), float("nan"))
    half = student_t.ppf(1 - alpha / 2, v.size - 1) * v.std(ddof=1) / np.sqrt(v.size)
    return (float(v.mean() - half), float(v.mean() + half))


# ---------------------------------------------------------------------------
# Estimators. Nothing is tuned, so nothing can leak through a tuning loop.
# ---------------------------------------------------------------------------

def make_estimator(name: str, n_features: int = 1, alpha: float | None = None):
    """One estimator at fixed, a-priori hyperparameters.

    NOTHING HERE IS SELECTED FROM DATA unless ``alpha`` is passed, and then the
    caller has selected it and must say so on the row -- see below. P1 needed a
    nested tuning loop and a poisoning test to prove that loop never saw a test
    fold; the default here has no loop, which is strictly stronger than a
    guarded one. Every default value is printed by ``describe_estimators``.

    The one hyperparameter that cannot be a constant is the ridge penalty,
    because the model kinds differ in width by a factor of 30 -- D=2 for the
    observation control against D=64 for weather and D=66 for the two together
    -- and a single alpha would regularise them by wildly different amounts,
    which would make ``margin_over_control`` a comparison between two differently
    penalised models rather than between two feature sets. So the rule is
    **alpha = D**, fixed a priori and applied to standardised features: the
    Bayesian reading is a prior whose TOTAL signal variance is held constant as
    the design widens, which is the scale-free choice. It is not fitted, not
    searched, and does not look at a target.

    Measured consequence, and the reason the rule is worth stating: under a flat
    alpha=1 the permutation control sits at -0.70 -- a flexible model on shuffled
    features is heavily penalised rather than neutral -- while under alpha=D it
    sits at -0.12, close to the zero a null should be near. The rule makes the
    empirical zero interpretable.

    THE ONE OVERRIDE, AND WHY IT IS A PARAMETER AND NOT A SECOND FUNCTION.
    ``alpha`` replaces the alpha = D rule for the ridge, and for the ridge only.
    It exists because alpha = D spans 79 to 11536 across P3's encoder views -- a
    146-fold range set by the WIDTH of the design rather than by the data -- and
    at the narrow end it over-penalises the 79-column band-matched baseline
    against which every network is scored. A caller that passes ``alpha`` has
    SELECTED it (``p2_deltas.select_ridge_alpha``, nested CV on the training fold
    only, two poisoning tests) and owes the table a column saying so; P3 writes
    ``alpha_rule`` on every row and keeps both rules side by side. Passing it for
    a non-ridge estimator is refused rather than ignored, because an ignored
    hyperparameter is a silently different model.

    All three estimators are deterministic: the ridge solve is exact, HGB has
    ``early_stopping=False`` with a fixed seed, and the MLP has a fixed seed and
    no validation split. Parallelism changes wall-clock only, never a number.
    """
    assert name in ESTIMATORS, f"estimator {name!r} not in {ESTIMATORS}"
    assert alpha is None or name == "linear", (
        f"alpha={alpha!r} was passed for estimator {name!r}. The penalty "
        "override applies to the RIDGE only; every other estimator's "
        "hyperparameters are fixed a priori, and silently ignoring the argument "
        "would fit a different model than the caller asked for."
    )
    if name == "linear":
        from sklearn.linear_model import Ridge
        if alpha is not None:
            assert np.isfinite(alpha) and alpha > 0, f"alpha must be > 0, got {alpha}"
            return Ridge(alpha=float(alpha))
        assert n_features >= 1, f"alpha = D needs D >= 1, got {n_features}"
        return Ridge(alpha=float(n_features))
    if name == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.1, min_samples_leaf=20,
            l2_regularization=1.0, early_stopping=False, random_state=0)
    from sklearn.neural_network import MLPRegressor
    return MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu",
                        alpha=1e-3, learning_rate_init=1e-3, max_iter=400,
                        early_stopping=False, random_state=0)


def describe_estimators(n_features: int = 64) -> None:
    print("[p4] estimators (increasing capacity, NOTHING tuned from data -- "
          "there is no selection loop to leak through):")
    for name in ESTIMATORS:
        print(f"[p4]   {name:<8} {make_estimator(name, n_features)}")
    print(f"[p4]   linear's alpha is D, the design width, fixed a priori and "
          f"shown here at D={n_features}; the observation control is D=2 and "
          "gets alpha=2, so margin_over_control compares two models penalised "
          "on the same scale rather than two arbitrary ones")


def _fit_predict(estimator: str, X_tr, y_tr, X_te, alpha: float | None = None) -> tuple:
    """Standardise on TRAIN, fit on TRAIN, predict TEST. Never the reverse.

    ``alpha`` overrides the ridge's alpha = D rule and nothing else; it is
    forwarded to ``make_estimator``, which refuses it for any other estimator.
    """
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X_tr)
    Z_tr, Z_te = scaler.transform(X_tr), scaler.transform(X_te)
    assert np.isfinite(Z_tr).all() and np.isfinite(Z_te).all()
    D = int(X_tr.shape[1])
    model = make_estimator(estimator, D, alpha=alpha)
    converged = True
    with warnings.catch_warnings():
        # Recorded per fold and reported per row rather than silenced: an MLP
        # that ran out of iterations is a weaker model than the one this probe
        # claims to be reporting, and the count belongs in the table.
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            model.fit(Z_tr, y_tr)
        except ConvergenceWarning:
            converged = False
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model = make_estimator(estimator, D, alpha=alpha).fit(Z_tr, y_tr)
    pred_tr = model.predict(Z_tr)
    pred_te = model.predict(Z_te)
    return pred_te, pred_tr, converged


# ---------------------------------------------------------------------------
# One fold
# ---------------------------------------------------------------------------

class P4FoldResult(NamedTuple):
    """One outer fold. ``effective_n`` counts CUBES, never rows."""

    fold: int
    n_train: int
    n_test: int
    n_train_cubes: int
    n_test_cubes: int
    effective_n: int
    r2: float
    r2_vs_climatology: float
    rmse: float
    mae: float
    crps: float
    crps_climatology: float
    crps_skill: float
    anomaly_sd_test: float
    curve_train_r2: float
    n_extrapolated: float
    same_cube_after_permutation: float
    converged: bool
    log: str
    # The held-out rows, kept so the per-severity and per-stratum breakdowns
    # are pooled from the SAME fits the scores come from. Recomputing them in a
    # second pass would double the run and, worse, would let the two disagree.
    test_pos: np.ndarray = None
    test_anomaly: np.ndarray = None
    test_prediction: np.ndarray = None


def evaluate_fold(sources, target: TargetRows, manifest, day_of_year,
                  cube_id, train_rows, test_rows, estimator: str,
                  base_X=None, permute: bool = False, fold: int = 0,
                  n_harmonics: int = CLIMATOLOGY_HARMONICS,
                  verbose: bool = False) -> P4FoldResult:
    """One outer fold, end to end, with the climatology fitted INSIDE it.

    Order of operations, and none of it is interchangeable:

      1. slice the fold on MANIFEST rows, expand to feature rows
      2. fit the day-of-year curve on the TRAIN feature rows only
      3. subtract it from BOTH sides to get the anomaly
      4. standardise on train, fit on train, predict test

    Step 2 before step 3 is what makes the anomaly a held-out quantity. Doing it
    once outside the loop would be the nested leak this module's docstring is
    mostly about.
    """
    train_rows = np.asarray(train_rows, dtype=int)
    test_rows = np.asarray(test_rows, dtype=int)
    assert not np.intersect1d(train_rows, test_rows).size, (
        f"fold {fold}: manifest rows appear on both sides"
    )
    row_idx = target.row_idx
    tr_pos = np.flatnonzero(np.isin(row_idx, train_rows))
    te_pos = np.flatnonzero(np.isin(row_idx, test_rows))
    assert tr_pos.size and te_pos.size, (
        f"fold {fold}: a side selected zero {target.name} feature rows"
    )

    perm_map, same_cube = None, float("nan")
    if permute:
        perm_map, same_cube = _perm_map(cube_id, train_rows, test_rows,
                                        len(manifest), seed=1000 + fold)
    X = base_X if (base_X is not None and not permute) \
        else build_X(sources, row_idx, perm_map)
    block = FeatureBlock(X=X, row_idx=row_idx, y=target.y, name=target.name)

    # --- the climatology, fitted on the TRAINING index set and nothing else ---
    doy = day_of_year[row_idx]
    curve = doy_climatology_within_fold(doy, block.y, tr_pos,
                                        n_harmonics=n_harmonics,
                                        label=f"{target.name} fold {fold + 1}")
    anomaly = block.y - curve.predict(doy)
    # -------------------------------------------------------------------------

    train, test = block.take(tr_pos), block.take(te_pos)
    a_tr, a_te = anomaly[tr_pos], anomaly[te_pos]

    pred, pred_tr, converged = _fit_predict(estimator, train.X, a_tr, test.X)
    resid = a_te - pred
    sse = float((resid ** 2).sum())
    sst_centred = float(((a_te - a_te.mean()) ** 2).sum())
    sst_zero = float((a_te ** 2).sum())
    r2 = float(1.0 - sse / sst_centred) if sst_centred > 0 else float("nan")
    r2_clim = float(1.0 - sse / sst_zero) if sst_zero > 0 else float("nan")

    sigma = float(np.std(a_tr - pred_tr, ddof=1)) if a_tr.size > 1 else 1e-9
    crps = float(gaussian_crps(a_te, pred, sigma).mean())
    crps_clim = float(ensemble_crps(a_te, a_tr).mean())
    crps_skill = float(1.0 - crps / crps_clim) if crps_clim > 0 else float("nan")

    n_tr_cubes = int(np.unique(cube_id[train_rows]).size)
    n_te_cubes = int(np.unique(cube_id[test_rows]).size)
    log = (f"[p4]   fold {fold + 1}: train {train.n_rows} rows / {n_tr_cubes} "
           f"cubes, test {test.n_rows} rows / {n_te_cubes} cubes "
           f"(effective n = {n_te_cubes} CUBES, not {test.n_rows} rows) | "
           f"R2 {r2:+.3f} vs-clim {r2_clim:+.3f} | CRPS {crps:.4f} vs "
           f"clim {crps_clim:.4f} (skill {crps_skill:+.3f})")
    if verbose:
        print(log)
    return P4FoldResult(
        fold=fold, n_train=train.n_rows, n_test=test.n_rows,
        n_train_cubes=n_tr_cubes, n_test_cubes=n_te_cubes,
        effective_n=n_te_cubes,
        r2=r2, r2_vs_climatology=r2_clim,
        rmse=float(np.sqrt((resid ** 2).mean())), mae=float(np.abs(resid).mean()),
        crps=crps, crps_climatology=crps_clim, crps_skill=crps_skill,
        anomaly_sd_test=float(a_te.std()), curve_train_r2=float(curve.train_r2),
        n_extrapolated=float(curve.n_extrapolated(doy[te_pos])),
        same_cube_after_permutation=float(same_cube), converged=bool(converged),
        log=log, test_pos=te_pos, test_anomaly=a_te, test_prediction=pred)


def _breakdown(results, key_values, keys, min_rows: int = 20) -> dict:
    """Pooled out-of-fold R-squared (vs climatology) within each key group.

    Pooled ACROSS folds, because a severity bin or a thin stratum routinely has
    too few rows inside one fold to estimate anything. Each contribution is
    still a held-out prediction from the fold that produced it, and the anomaly
    it is scored against is that fold's own within-fold anomaly -- so the pool
    is heterogeneous in the climatology by construction, which is the price of
    fitting the climatology inside the fold and is the right price to pay.
    """
    pos = np.concatenate([r.test_pos for r in results])
    a = np.concatenate([r.test_anomaly for r in results])
    p = np.concatenate([r.test_prediction for r in results])
    lab = key_values[pos]
    out = {}
    for k in keys:
        m = lab == k
        n = int(m.sum())
        out[f"n_{k}"] = n
        sse, sst = float(((a[m] - p[m]) ** 2).sum()), float((a[m] ** 2).sum())
        out[f"r2_{k}"] = (float(1.0 - sse / sst)
                          if n >= min_rows and sst > 0 else float("nan"))
    return out


def evaluate(sources, target: TargetRows, data: P4Data, mode: str,
             estimator: str, k: int = 5, permute: bool = False,
             n_harmonics: int = CLIMATOLOGY_HARMONICS, n_jobs: int = 1,
             verbose: bool = False) -> list:
    """Every outer fold of one mode. Deterministic; ``n_jobs`` only changes
    wall-clock -- the ridge solve is exact, HGB and the MLP carry fixed seeds
    with no validation split, and the folds come from a generator with no RNG.
    """
    outer = outer_folds(data.manifest, mode, k=k)
    base_X = None if permute else build_X(sources, target.row_idx)
    call = dict(estimator=estimator, base_X=base_X, permute=permute,
                n_harmonics=n_harmonics, verbose=False)

    if n_jobs == 1:
        results = [evaluate_fold(sources, target, data.manifest,
                                 data.day_of_year, data.cube_id, tr, te,
                                 fold=i, **call)
                   for i, (tr, te) in enumerate(outer)]
    else:
        from joblib import Parallel, delayed, parallel_config
        with parallel_config(backend="loky", inner_max_num_threads=1):
            results = Parallel(n_jobs=n_jobs)(
                delayed(evaluate_fold)(sources, target, data.manifest,
                                       data.day_of_year, data.cube_id, tr, te,
                                       fold=i, **call)
                for i, (tr, te) in enumerate(outer))
    if verbose:
        for r in results:
            print(r.log)
    return results


def outer_folds(manifest, mode: str, k: int = 5, verbose: bool = False) -> list:
    """The outer folds for one mode, from probes.cv and nowhere else."""
    assert mode in FOLD_MODES + (STAGE_B_MODE,), (
        f"fold mode {mode!r} not in {FOLD_MODES + (STAGE_B_MODE,)}"
    )
    if mode == "loco":
        return list(cv.leave_one_cube_out(manifest, verbose=verbose))
    return list(cv.folds(manifest, mode, k=k, verbose=verbose))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def summarise(results: Sequence[P4FoldResult]) -> dict:
    """Mean, spread and a FOLD-CLUSTERED CI. Never a bare mean.

    At 20 cubes the fold-to-fold spread is the uncertainty, and it is routinely
    larger than the gap between two estimators. ``effective_n`` is carried here
    so it can never be separated from the score it qualifies.
    """
    assert results, "nothing to summarise"
    out = {"n_folds": len(results)}
    for field in ("r2", "r2_vs_climatology", "rmse", "mae", "crps",
                  "crps_climatology", "crps_skill"):
        v = np.array([getattr(r, field) for r in results], dtype=float)
        lo, hi = fold_clustered_ci(v)
        out[f"{field}_mean"] = float(np.nanmean(v))
        out[f"{field}_std"] = float(np.nanstd(v, ddof=1)) if v.size > 1 else 0.0
        out[f"{field}_min"] = float(np.nanmin(v))
        out[f"{field}_max"] = float(np.nanmax(v))
        out[f"{field}_ci_lo"] = lo
        out[f"{field}_ci_hi"] = hi
        out[f"per_fold_{field}"] = ";".join(f"{x:.4f}" for x in v)
    out["effective_n"] = int(sum(r.effective_n for r in results))
    out["effective_n_per_fold"] = ";".join(str(r.effective_n) for r in results)
    out["n_rows_test_total"] = int(sum(r.n_test for r in results))
    out["curve_train_r2_mean"] = float(np.mean([r.curve_train_r2 for r in results]))
    out["n_extrapolated_total"] = int(sum(r.n_extrapolated for r in results))
    out["n_not_converged"] = int(sum(not r.converged for r in results))
    same = np.array([r.same_cube_after_permutation for r in results], dtype=float)
    out["same_cube_after_permutation_max"] = (float(np.nanmax(same))
                                              if np.isfinite(same).any() else float("nan"))
    out["anomaly_sd_test_mean"] = float(np.mean([r.anomaly_sd_test for r in results]))
    return out


# ---------------------------------------------------------------------------
# Stage A
# ---------------------------------------------------------------------------

def _sources_for(kind: str, data: P4Data, feature_set: str, level: str) -> tuple:
    obs = data.observation(level)
    if kind == "weather":
        return (data.weather[feature_set],), False
    if kind == "observation":
        return (obs,), False
    if kind == "doy":
        return (data.doy_frame,), False
    if kind == "weather_plus_observation":
        return (data.weather[feature_set], obs), False
    if kind == "permutation":
        return (data.weather[feature_set],), True
    raise AssertionError(f"unknown model kind {kind!r}")


_KIND_NOTE = {
    "weather": "weather only -- the quantity H1 is about",
    "observation": ("OBSERVATION-PROCESS CONTROL: [clear_frac or "
                    "grid_clear_frac, window_span_days], NO weather. P1's "
                    "degenerate control carried forward; this is the floor "
                    "margin_over_control is measured from. Identical across "
                    "feature sets by construction"),
    "doy": ("DAY-OF-YEAR SANITY CONTROL: day-of-year alone. ~0 means the "
            f"detrend worked; materially non-zero means the "
            f"{CLIMATOLOGY_HARMONICS}-harmonic proxy is UNDERFITTING and the "
            "anomaly still carries seasonal cycle. Identical across feature sets"),
    "weather_plus_observation": ("weather AND the observation process jointly: "
                                 "does weather add anything beyond retention, "
                                 "and are the two redundant"),
    "permutation": ("PERMUTATION CONTROL: weather shuffled across cubes within "
                    "each fold side. The EMPIRICAL ZERO of this pipeline, which "
                    "is not exactly 0.0 with grouped data and a flexible "
                    "estimator"),
}


def run_stage_a(data: P4Data, targets: Sequence[str] = TARGETS,
                fold_modes: Sequence[str] = FOLD_MODES,
                estimators: Sequence[str] = ESTIMATORS,
                feature_sets: Sequence[str] = FEATURE_SETS,
                k: int = 5, n_jobs: int = 1, verbose: bool = True):
    """Stage A: the PROXY climatology, on the 20 single-year cubes.

    Every row is labelled ``STAGE_A_CLIMATOLOGY_LABEL``. This stage de-risks the
    code and establishes the controls; **it does not produce H1**.
    """
    import pandas as pd

    rows = []
    for target in targets:
        tr = data.targets[target]
        level = tr.level
        sev_labels, sev_edges = data.severity[target]
        strata_all = tr.stratum
        if verbose:
            print("\n" + "=" * 78)
            print(f"[p4] TARGET {target} ({level} level): {tr.n_rows} rows over "
                  f"{len(data.manifest)} frames / "
                  f"{np.unique(data.cube_id).size} cubes")
        for mode in fold_modes:
            for estimator in estimators:
                head = f"{target} | {mode} | {estimator}"
                if verbose:
                    print(f"\n[p4] ---- {head} " + "-" * max(0, 58 - len(head)))
                for fs in feature_sets:
                    for kind in MODEL_KINDS:
                        srcs, permute = _sources_for(kind, data, fs, level)
                        res = evaluate(srcs, tr, data, mode, estimator, k=k,
                                       permute=permute, n_jobs=n_jobs)
                        s = summarise(res)
                        s.update(_breakdown(res, sev_labels, SEVERITY_BINS))
                        s.update(_breakdown(res, strata_all, REPLICATION_STRATA))
                        D = sum(len(x.names) for x in srcs)
                        row = {
                            "stage": "A", "target": target, "feature_level": level,
                            "fold_mode": mode, "estimator": estimator,
                            "feature_set": fs, "model_kind": kind,
                            "is_control": kind in CONTROL_KINDS,
                            "climatology_def": STAGE_A_CLIMATOLOGY_LABEL,
                            "climatology_harmonics": CLIMATOLOGY_HARMONICS,
                            "n_feature_rows": tr.n_rows,
                            "n_manifest_rows": int(len(data.manifest)),
                            "n_cubes": int(np.unique(data.cube_id).size),
                            "D": D,
                            "feature_names": ";".join(
                                n for x in srcs for n in x.names)[:2000],
                            "severity_edges": ";".join(f"{e:.5f}" for e in sev_edges),
                            "note": _KIND_NOTE[kind],
                            "plausibility_screen": bool(data.plausibility_screen),
                            "plausibility_screen_def": (
                                PLAUSIBILITY_SCREEN_LABEL
                                if data.plausibility_screen else "not_applied"),
                            "n_implausible_frames": int(
                                data.n_implausible_frames),
                            "n_rows_dropped_implausible": int(
                                (data.n_rows_dropped_implausible or {}).get(
                                    target, 0)),
                            **s,
                        }
                        rows.append(row)
                        if verbose:
                            print(f"[p4]   {fs:<14} {kind:<25} D={D:<4} "
                                  f"R2 {s['r2_mean']:+.3f} "
                                  f"[{s['r2_ci_lo']:+.3f}, {s['r2_ci_hi']:+.3f}] "
                                  f"| vs-clim {s['r2_vs_climatology_mean']:+.3f} "
                                  f"| CRPS skill {s['crps_skill_mean']:+.3f} "
                                  f"| effective n {s['effective_n']} CUBES "
                                  f"(rows {s['n_rows_test_total']})")
    df = pd.DataFrame(rows)
    assert len(df), "run_stage_a produced no rows"
    return df


# ---------------------------------------------------------------------------
# Stage B
# ---------------------------------------------------------------------------

def detect_seasonal_split(manifest, verbose: bool = True) -> dict:
    """Are multi-year cubes present? The gate Stage B opens on.

    The seasonal split's cubes span 2017-2020 INSIDE one file, so the test is
    on the per-ROW year derived from the timestamp, not on the ``year`` column
    (which is parsed from the cube id, i.e. the window START year, and would
    report 2018 for a cube covering four years).
    """
    ts = np.asarray(manifest["timestamp"].to_numpy(), dtype="datetime64[ns]")
    years = ts.astype("datetime64[Y]").astype(int) + 1970
    cubes = manifest["cube_id"].to_numpy().astype(str)
    per_cube = {c: np.unique(years[cubes == c]) for c in np.unique(cubes)}
    spanning = sorted(c for c, y in per_cube.items() if y.size > 1)
    out = {
        "multi_year": bool(np.unique(years).size > 1),
        "years": sorted(np.unique(years).tolist()),
        "n_cubes": int(np.unique(cubes).size),
        "n_cubes_spanning_years": len(spanning),
        "cubes_spanning_years": tuple(spanning[:5]),
    }
    if verbose:
        print(f"[p4] seasonal-split detection: years {out['years']} over "
              f"{out['n_cubes']} cubes; {out['n_cubes_spanning_years']} cube(s) "
              f"span more than one year -> multi_year={out['multi_year']}")
    return out


def _stage_b_deferral(info: dict) -> str:
    return (
        "\n" + "=" * 78 + "\n"
        "[p4] STAGE B DEFERRED -- pending the seasonal download.\n"
        f"[p4] The manifest spans {info['years']} over {info['n_cubes']} cubes "
        f"and {info['n_cubes_spanning_years']} cube(s) span more than one year.\n"
        "[p4] Stage B needs the REAL leave-target-year-out climatology\n"
        "[p4]   (data.climatology.ndvi_climatology, which raises "
        "SingleYearError here BY DESIGN)\n"
        "[p4] and probes.cv mode 'crossed', which holds cube AND year out "
        "jointly and\n"
        "[p4] therefore agrees with that climatology's own structure. Both "
        "correctly\n"
        "[p4] REFUSE a single-year manifest. Neither is weakened, and Stage A's\n"
        "[p4] within-season proxy number is NOT substituted for H1: Stage A "
        "de-risks\n"
        "[p4] the code and establishes the controls, and that is all it does.\n"
        "[p4] H1 is produced by Stage B, on the seasonal split, and nowhere "
        "else.\n" + "=" * 78
    )


def run_stage_b(manifest, cube_dir: str, data: P4Data = None,
                targets: Sequence[str] = TARGETS,
                estimators: Sequence[str] = ESTIMATORS,
                feature_sets: Sequence[str] = FEATURE_SETS,
                k: int = 5, n_jobs: int = 1, verbose: bool = True,
                plausibility_screen: bool = False):
    """Stage B: the REAL leave-target-year-out climatology. **H1's number.**

    Runs if and only if multi-year cubes are present. It imports
    ``data.climatology.ndvi_climatology`` -- the GreenEarthNet definition,
    adopted verbatim in behaviour -- and never reimplements it, and it uses
    ``probes.cv`` mode ``crossed`` and no other. Any other mode would pair a
    test year against a climatology built INCLUDING that year, which is the same
    nested leak as fitting the Stage A proxy outside the fold, one level up.

    Returns None after printing an explicit deferral when the seasonal cubes are
    absent. It never falls back to Stage A's proxy.
    """
    import pandas as pd

    from data.climatology import SingleYearError, ndvi_climatology

    info = detect_seasonal_split(manifest, verbose=verbose)
    if not info["multi_year"]:
        print(_stage_b_deferral(info))
        return None, info

    from data.loader import CubeSample, cube_ndvi, load_cube

    ts = np.asarray(manifest["timestamp"].to_numpy(), dtype="datetime64[ns]")
    row_year = ts.astype("datetime64[Y]").astype(int) + 1970
    order = sorted(manifest["cube_id"].unique().tolist())

    # Per-pixel leave-target-year-out anomaly, aggregated exactly as Stage A's
    # targets are. The climatology for a row in year Y of cube C pools C's
    # OTHER years, which is GreenEarthNet's protocol; crossed mode holds the
    # whole cube out of train, so nothing that built it is ever trained on.
    #
    # ONE RESIDUAL LEAK PATH, recorded rather than hidden. A TRAIN cube's rows
    # in year Y' are detrended by a climatology pooling that cube's other
    # years -- which can include the fold's HELD-OUT year Y. So test-year
    # information reaches the train TARGETS, via other cubes, at second order.
    # It is not the first-order leak (test rows come from held-out cubes, whose
    # own climatologies never touch train), and the climatology averages over
    # years so a single year moves it by ~1/(n_years-1). It is nonetheless not
    # zero, and the honest fix -- rebuilding each cube's climatology per fold,
    # excluding every held-out year rather than only the row's own -- costs a
    # climatology recomputation per (cube, fold) instead of per (cube, year).
    # Deferred, and quantified before H1 is quoted from this path.
    anom = {name: [] for name in TARGETS}
    valid_cells, frame_plausible_parts, rows = [], [], []
    for cube in order:
        pos = np.flatnonzero((manifest["cube_id"] == cube).to_numpy())
        sample = load_cube(os.path.join(cube_dir, str(cube)), verbose=False)
        sel = select_clear_frames(sample.values, sample.timestamps, sample.mask,
                                  verbose=False)
        kept = CubeSample(values=sel.values, timestamps=sel.timestamps,
                          mask=sel.mask, path=sample.path, bands=sample.bands)
        nd = cube_ndvi(kept)
        frame_plausible_parts.append(
            _frame_plausible(np.nanmean(nd.reshape(nd.shape[0], -1), axis=1),
                             sel.timestamps))
        full_nd = cube_ndvi(sample)
        years_here = np.unique(row_year[pos])
        dev = np.full(nd.shape, np.nan, dtype=np.float32)
        for y in years_here:
            try:
                clim = ndvi_climatology(full_nd, sample.timestamps, int(y))
            except SingleYearError:
                raise AssertionError(
                    f"{cube} carries frames from {years_here.tolist()} in the "
                    f"manifest but its own file holds only one year, so the "
                    f"leave-{y}-out climatology is not computable. Stage B "
                    "cannot run on this manifest; it must be deferred, not "
                    "substituted."
                )
            m = (sel.timestamps.astype("datetime64[Y]").astype(int) + 1970) == y
            doy = (np.asarray(sel.timestamps[m], dtype="datetime64[D]")
                   - np.asarray(sel.timestamps[m], dtype="datetime64[Y]")
                   ).astype(int)
            dev[m] = nd[m] - clim[doy]
        T, H, W = dev.shape
        ch, cw = H // _GRID, W // _GRID
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            anom["cube_mean"].append(np.nanmean(dev.reshape(T, -1), axis=1))
            anom["cube_p90"].append(np.nanpercentile(dev.reshape(T, -1), 90, axis=1))
            cells = np.nanmean(dev.reshape(T, _GRID, ch, _GRID, cw), axis=(2, 4))
        anom["cell_mean"].append(cells.reshape(T, _GRID_CELLS))
        valid_cells.append(np.isfinite(cells.reshape(T, _GRID_CELLS)))
        rows.append(pos)

    rows = np.concatenate(rows)
    back = np.argsort(rows, kind="stable")
    stacked = {n: np.concatenate(v)[back] for n, v in anom.items()}
    valid = np.concatenate(valid_cells)[back]
    frame_ok = np.concatenate(frame_plausible_parts)[back]
    assert frame_ok.shape == (len(manifest),), frame_ok.shape
    n_implausible = int((~frame_ok).sum())

    weather = {fs: weather_source(manifest, cube_dir, fs, verbose=verbose)[0]
               for fs in feature_sets}
    per_cube_obs = _observation_inputs_for_stage_b(manifest, valid, cube_dir)
    obs_frame = observation_source(manifest, per_cube_obs, "frame", verbose=False)
    obs_cell = observation_source(manifest, per_cube_obs, "cell", verbose=False)
    if plausibility_screen:
        cell_rows = np.repeat(np.arange(len(manifest)), _GRID_CELLS)[
            valid.reshape(-1)]
        cell_keep = frame_ok[cell_rows]
        assert obs_cell.values.shape[0] == cell_keep.size
        obs_cell = FeatureSource(
            name=obs_cell.name,
            values=obs_cell.values[cell_keep],
            names=obs_cell.names,
            frame_level=False,
            permutable=obs_cell.permutable,
        )
    doy_src = doy_source(manifest, verbose=False)
    doy = manifest["day_of_year"].to_numpy().astype(float)
    cube_id = manifest["cube_id"].to_numpy().astype(str)

    out = []
    for target in targets:
        level = TARGET_LEVEL[target]
        if level == "frame":
            y, row_idx = stacked[target], np.arange(len(manifest))
            cells = np.full(row_idx.size, -1, dtype=int)
            strata = manifest["landcover_stratum"].to_numpy().astype(str)
        else:
            keep = np.flatnonzero(valid.reshape(-1))
            y = stacked[target].reshape(-1)[keep]
            row_idx = np.repeat(np.arange(len(manifest)), _GRID_CELLS)[keep]
            cells = np.tile(np.arange(_GRID_CELLS), len(manifest))[keep]
            glc = manifest["grid_landcover"].to_numpy()
            strata = np.array([str(glc[r][c]) for r, c in zip(row_idx, cells)])
        finite_before_screen = np.isfinite(y)
        finite = finite_before_screen.copy()
        if plausibility_screen:
            finite &= frame_ok[row_idx]
        n_dropped = int((finite_before_screen & ~finite).sum())
        tr_rows = TargetRows(name=target, level=level, y=y[finite],
                             row_idx=row_idx[finite], cell_idx=cells[finite],
                             stratum=strata[finite])
        sev_labels, sev_edges = severity_bins(tr_rows.y)
        stage_b_data = P4Data(
            manifest=manifest, targets={target: tr_rows}, weather=weather,
            observation_frame=obs_frame, observation_cell=obs_cell,
            doy_frame=doy_src, day_of_year=doy, cube_id=cube_id,
            days_available=None, reference_anomaly={target: tr_rows.y},
            severity={target: (sev_labels, sev_edges)},
            plausibility_screen=bool(plausibility_screen),
            n_implausible_frames=n_implausible,
            n_rows_dropped_implausible={target: n_dropped})
        for estimator in estimators:
            for fs in feature_sets:
                for kind in MODEL_KINDS:
                    srcs, permute = _sources_for(kind, stage_b_data, fs, level)
                    # The target is ALREADY the real leave-year-out deviation,
                    # so the Stage A proxy must NOT be subtracted from it again.
                    # evaluate_b pins n_harmonics=STAGE_B_HARMONICS=0: the
                    # "curve" is then the training-fold mean and nothing else.
                    res = evaluate_b(srcs, tr_rows, stage_b_data, estimator,
                                     k=k, permute=permute, n_jobs=n_jobs)
                    s = summarise(res)
                    s.update(_breakdown(res, sev_labels, SEVERITY_BINS))
                    s.update(_breakdown(res, tr_rows.stratum, REPLICATION_STRATA))
                    out.append({
                        "stage": "B", "target": target, "feature_level": level,
                        "fold_mode": STAGE_B_MODE, "estimator": estimator,
                        "feature_set": fs, "model_kind": kind,
                        "is_control": kind in CONTROL_KINDS,
                        "climatology_def": STAGE_B_CLIMATOLOGY_LABEL,
                        "climatology_harmonics": 0,
                        "n_feature_rows": tr_rows.n_rows,
                        "n_manifest_rows": int(len(manifest)),
                        "n_cubes": int(np.unique(cube_id).size),
                        "D": sum(len(x.names) for x in srcs),
                        "feature_names": ";".join(
                            n for x in srcs for n in x.names)[:2000],
                        "severity_edges": ";".join(f"{e:.5f}" for e in sev_edges),
                        "note": _KIND_NOTE[kind] + " | H1",
                        "plausibility_screen": bool(plausibility_screen),
                        "plausibility_screen_def": (
                            PLAUSIBILITY_SCREEN_LABEL
                            if plausibility_screen else "not_applied"),
                        "n_implausible_frames": n_implausible,
                        "n_rows_dropped_implausible": n_dropped,
                        **s,
                    })
    df = pd.DataFrame(out)
    assert len(df), "Stage B ran but produced no rows"
    print(f"[p4] STAGE B RAN: {len(df)} rows, mode={STAGE_B_MODE!r}, "
          f"climatology={STAGE_B_CLIMATOLOGY_LABEL}")
    return df, info


def _observation_inputs_for_stage_b(manifest, valid, cube_dir) -> dict:
    """The per-cube dict ``observation_source`` expects, for Stage B."""
    from data.loader import load_cube

    order = sorted(manifest["cube_id"].unique().tolist())
    per_cube = {"__order__": order}
    for cube in order:
        sample = load_cube(os.path.join(cube_dir, str(cube)), verbose=False)
        sel = select_clear_frames(sample.values, sample.timestamps, sample.mask,
                                  verbose=False)
        pos = np.flatnonzero((manifest["cube_id"] == cube).to_numpy())
        per_cube[cube] = {"grid_clear_frac": grid_clear_fraction(sel.mask),
                          "cell_valid": valid[pos]}
    return per_cube


def evaluate_b(sources, target: TargetRows, data: P4Data, estimator: str,
               k: int = 5, permute: bool = False, n_jobs: int = 1) -> list:
    """Stage B's folds: mode ``crossed``, and no other mode is reachable here.

    The mode is not a parameter. ``crossed`` holds cube AND year out jointly and
    is the only mode that agrees with a leave-target-year-out climatology; every
    other one would score a test year against a climatology built including it.
    Hard-coding it here means Stage B cannot be run under the wrong split by
    passing an argument.
    """
    return evaluate(sources, target, data, STAGE_B_MODE, estimator, k=k,
                    permute=permute, n_harmonics=STAGE_B_HARMONICS,
                    n_jobs=n_jobs)


# ---------------------------------------------------------------------------
# Margins and table-level invariants
# ---------------------------------------------------------------------------

def add_margins(df, verbose: bool = True):
    """Attach the margins P4 is actually read on.

    ``margin_over_control`` -- ``r2_vs_climatology`` minus the OBSERVATION
    control's, for the same (stage, target, fold_mode, estimator, feature_set).
    **This is the headline.** On this subset cloud retention is strongly
    seasonal (P1: two numbers with no image decoded season at 0.65), so the raw
    R-squared overstates what the weather contributed; the margin is the part
    that needed a weather variable.

    ``margin_over_permutation`` -- minus the permutation control's, i.e. minus
    this pipeline's own empirical zero.

    ``margin_over_doy`` -- minus the day-of-year control's. If the proxy
    climatology underfits, the residual seasonal cycle is available to BOTH the
    weather model and the day-of-year control, so this margin is what survives
    that.
    """
    key = ["stage", "target", "fold_mode", "estimator", "feature_set"]
    df = df.copy()
    for kind, col in (("observation", "margin_over_control"),
                      ("permutation", "margin_over_permutation"),
                      ("doy", "margin_over_doy")):
        ctrl = df[df.model_kind == kind]
        assert len(ctrl) == len(ctrl.drop_duplicates(key)), (
            f"the {kind} control is not unique per {key}"
        )
        look = ctrl.set_index(key)["r2_vs_climatology_mean"]
        df[col] = (df.r2_vs_climatology_mean
                   - df.set_index(key).index.map(look).to_numpy())
        assert df[col].notna().all(), (
            f"some row has no matching {kind} control. The controls are not "
            "optional and every row must be comparable to all of them."
        )
    look_r2 = (df[df.model_kind == "observation"]
               .set_index(key)["r2_mean"])
    df["margin_over_control_r2"] = (df.r2_mean
                                    - df.set_index(key).index.map(look_r2).to_numpy())
    if verbose:
        w = df[df.model_kind == "weather"]
        beaten = w[w.margin_over_control <= 0]
        print(f"[p4] margins attached. {len(beaten)}/{len(w)} weather rows are "
              "AT OR BELOW the observation-process control -- i.e. weather adds "
              "nothing there that cloud retention did not already carry")
    return df


def assert_control_views_consistent(df) -> None:
    """The two feature-set copies of a weather-free control must agree exactly.

    ``observation`` and ``doy`` contain no weather, so their rows are the same
    fitted model under both feature-set labels. They are emitted twice so that
    filtering the CSV to one feature set still leaves all four controls in front
    of the reader -- and duplicated numbers nobody checks are how two tables
    start disagreeing, so the identity is asserted rather than trusted.
    """
    key = ["stage", "target", "fold_mode", "estimator", "model_kind"]
    for kind in ("observation", "doy"):
        sub = df[df.model_kind == kind]
        if not len(sub):
            continue
        g = sub.groupby(key)["r2_vs_climatology_mean"].nunique()
        bad = g[g > 1]
        assert bad.empty, (
            f"the {kind} control differs between feature sets for "
            f"{list(bad.index)[:3]}. It holds no weather, so the two rows are "
            "the same fitted model; a disagreement means one was computed on "
            "different rows."
        )


def assert_effective_n_counts_cubes(df) -> None:
    """Every row reports an effective n, and it counts CUBES, not rows."""
    assert "effective_n" in df.columns, "no effective_n column"
    assert df.effective_n.notna().all(), "a row has no effective n"
    assert (df.effective_n > 0).all(), "a row reports effective n <= 0"
    assert (df.effective_n <= df.n_cubes).all(), (
        "effective_n exceeds the number of cubes; it is counting something else"
    )
    bad = df[df.effective_n == df.n_rows_test_total]
    assert bad.empty or (bad.n_rows_test_total == bad.n_cubes).all(), (
        f"{len(bad)} row(s) report an effective n equal to the number of test "
        "ROWS. The effective sample size here is the number of independent "
        "weather realisations -- cubes -- not frames and not cells."
    )
    per_fold = df.effective_n_per_fold.str.split(";").apply(
        lambda v: sum(int(x) for x in v))
    assert (per_fold == df.effective_n).all(), (
        "effective_n is not the sum of the per-fold cube counts"
    )


def assert_stage_b_ran_or_deferred(df, info: dict) -> None:
    """Stage B ran under ``crossed`` with the real climatology, or was skipped.

    There is no third option, and in particular Stage A's proxy is never
    relabelled as H1. If multi-year cubes are present, Stage B rows must exist
    and must carry mode ``crossed`` and the real climatology's label.
    """
    has_b = "stage" in df.columns and (df.stage == "B").any()
    if info["multi_year"]:
        assert has_b, (
            f"the manifest spans {info['years']} -- multi-year cubes ARE "
            "present -- but the table has no Stage B rows. Stage B is not "
            "optional when it is computable; H1 comes from it."
        )
        b = df[df.stage == "B"]
        assert (b.fold_mode == STAGE_B_MODE).all(), (
            f"Stage B rows carry fold modes {sorted(set(b.fold_mode))}. Only "
            f"{STAGE_B_MODE!r} agrees with a leave-target-year-out climatology; "
            "any other mode pairs a test year against a climatology built "
            "including that year."
        )
        assert (b.climatology_def == STAGE_B_CLIMATOLOGY_LABEL).all(), (
            "a Stage B row is not labelled with the real leave-year-out "
            "climatology definition"
        )
    else:
        assert not has_b, (
            "the manifest is single-year, so the leave-target-year-out "
            "climatology is not computable -- yet Stage B rows exist. Stage A's "
            "within-season proxy must never be relabelled as Stage B."
        )
        assert (df.climatology_def == STAGE_A_CLIMATOLOGY_LABEL).all(), (
            "a row on a single-year manifest is not labelled as the proxy"
        )
    print(f"[p4] Stage B: {'RAN under crossed with the real climatology' if has_b else 'DEFERRED, explicitly, and NOT substituted'}")


def assert_plausibility_screen_declared(df, required: bool = True) -> None:
    """Every result row states whether the shared frame screen was applied."""
    for col in ("plausibility_screen", "plausibility_screen_def",
                "n_implausible_frames", "n_rows_dropped_implausible"):
        assert col in df.columns, f"the P4 table has no {col!r} column"
    values = df.plausibility_screen.astype(bool)
    assert values.nunique() == 1, (
        "the P4 table mixes screened and unscreened rows; their target row "
        "sets differ and the scores are not comparable"
    )
    actual = bool(values.iloc[0])
    assert actual == bool(required), (
        f"P4 declares plausibility_screen={actual}, expected {required}"
    )
    expected_label = PLAUSIBILITY_SCREEN_LABEL if required else "not_applied"
    assert (df.plausibility_screen_def.astype(str) == expected_label).all(), (
        "P4's plausibility_screen_def does not match the declared mode"
    )
    n_bad = df.n_implausible_frames.astype(int)
    assert n_bad.nunique() == 1 and (n_bad >= 0).all(), (
        "n_implausible_frames must be one non-negative run-level count"
    )
    dropped = df.n_rows_dropped_implausible.astype(int)
    assert (dropped >= 0).all(), "a P4 row reports a negative screen drop count"
    if required and int(n_bad.iloc[0]) > 0:
        assert (dropped > 0).any(), (
            "frames fail the screen but no P4 target row says it was dropped"
        )
    if not required:
        assert (dropped == 0).all(), (
            "an unscreened P4 table reports rows dropped by the screen"
        )


def assert_results_complete(df, targets: Sequence[str] = TARGETS,
                            fold_modes: Sequence[str] = FOLD_MODES,
                            estimators: Sequence[str] = ESTIMATORS,
                            feature_sets: Sequence[str] = FEATURE_SETS) -> None:
    """All four controls present for every cell the exit test asks for.

    Checked here rather than only in the test suite, so a truncated run cannot
    be written to disk and read later as if it were the whole table.
    """
    for stage in sorted(set(df.stage)):
        sub_stage = df[df.stage == stage]
        modes = fold_modes if stage == "A" else (STAGE_B_MODE,)
        for target in targets:
            for mode in modes:
                for est in estimators:
                    for fs in feature_sets:
                        sub = sub_stage[(sub_stage.target == target)
                                        & (sub_stage.fold_mode == mode)
                                        & (sub_stage.estimator == est)
                                        & (sub_stage.feature_set == fs)]
                        assert len(sub), (
                            f"stage {stage}: no rows for {target}/{mode}/{est}/{fs}"
                        )
                        have = set(sub.model_kind)
                        missing = set(MODEL_KINDS) - have
                        assert not missing, (
                            f"stage {stage} {target}/{mode}/{est}/{fs}: missing "
                            f"{sorted(missing)}. The four controls are not "
                            "optional -- without the observation control the "
                            "headline margin does not exist, and without the "
                            "permutation control there is no empirical zero to "
                            "read the raw R-squared against."
                        )
    for col in ("r2_mean", "r2_vs_climatology_mean", "crps_mean",
                "crps_climatology_mean", "effective_n", "r2_ci_lo", "r2_ci_hi"):
        assert col in df.columns, f"the results table has no {col} column"
    for b in SEVERITY_BINS:
        assert f"r2_{b}" in df.columns, f"no per-severity column for {b}"
    for s in REPLICATION_STRATA:
        assert f"r2_{s}" in df.columns, f"no per-stratum column for {s}"
    assert_control_views_consistent(df)
    assert_effective_n_counts_cubes(df)
    print(f"[p4] results table COMPLETE: {len(df)} rows, "
          f"{df.stage.nunique()} stage(s) x {df.target.nunique()} targets x "
          f"{df.fold_mode.nunique()} fold modes x {df.estimator.nunique()} "
          f"estimators x {df.feature_set.nunique()} feature sets x "
          f"{df.model_kind.nunique()} model kinds")


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def run_p4(manifest, cube_dir: str, targets: Sequence[str] = TARGETS,
           fold_modes: Sequence[str] = FOLD_MODES,
           estimators: Sequence[str] = ESTIMATORS,
           feature_sets: Sequence[str] = FEATURE_SETS,
           k: int = 5, n_jobs: int = 1, verbose: bool = True,
           plausibility_screen: bool = False):
    """Both stages. Returns (DataFrame, P4Data, stage-B info); writes nothing.

    The physical screen defaults off for backwards reproducibility. A screened
    rerun must opt in and every returned row records that choice.
    """
    import pandas as pd

    from encoders.manifest import assert_weather_join

    if verbose:
        print("[p4] verifying the E-OBS join against the cubes before anything "
              "is fitted -- weather is this probe's entire input")
    assert_weather_join(manifest, cube_dir, verbose=verbose)

    data = build_p4_data(manifest, cube_dir, feature_sets=feature_sets,
                         verbose=verbose,
                         plausibility_screen=plausibility_screen)
    if verbose:
        print()
        # The set this run ACTUALLY built, not the module default. A cross-tile
        # run on weather_finite6 never populates weather["weather_full8"], and
        # indexing the constant here would make verbose=True a KeyError on
        # exactly the runs that most need the output.
        print_doy_weather_collinearity(manifest, data.weather[feature_sets[0]])
        print()
        describe_estimators()
        print()
        for target in targets:
            print_severity_bins(data.reference_anomaly[target], label=target)
            print()

    df_a = run_stage_a(data, targets=targets, fold_modes=fold_modes,
                       estimators=estimators, feature_sets=feature_sets,
                       k=k, n_jobs=n_jobs, verbose=verbose)
    df_b, info = run_stage_b(manifest, cube_dir, data=data, targets=targets,
                             estimators=estimators, feature_sets=feature_sets,
                             k=k, n_jobs=n_jobs, verbose=verbose,
                             plausibility_screen=plausibility_screen)
    df = df_a if df_b is None else pd.concat([df_a, df_b], ignore_index=True)
    df = add_margins(df, verbose=verbose)
    assert_plausibility_screen_declared(
        df, required=bool(plausibility_screen))
    return df, data, info


def results_path(name: str = "p4_ceiling_results.csv",
                 root: str | None = None) -> str:
    """Phase-scoped by default; scaled reruns opt into their cache root."""
    base = phase_dir(PHASE, "results") if root is None else root
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, name)
