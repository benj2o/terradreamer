"""P2, dynamics in deltas: does a frozen encoder carry NDVI, and does the
CHANGE in its latents track the CHANGE on the ground?

Two questions, deliberately asked in this order.

**(a) Gate K2, the reconstruction floor.** Can a linear head read CURRENT NDVI
out of ``E(frame_t)`` at all? An encoder that cannot is not a representation of
this scene in any useful sense, and every later result about it -- change,
forecast -- would be uninterpretable, because a failure could always be blamed
on the missing floor rather than on the missing dynamics. The verdict is
recorded per encoder in the results table (``k2_verdict``), and an encoder that
fails is marked ``audited: lossy`` and EXCLUDED from P3. That is a result about
the encoder, not a failure of the probe.

**(b) The delta probe.** Do embedding CHANGES track real NDVI change, or does
the encoder discard it? Same frozen cache, differenced.

WHAT THIS PROBE READS, AND WHAT IT REFUSES TO READ
---------------------------------------------------
The Phase 1.2 ``.npz`` cache (embeddings and per-pixel masks), the cubes (for
NDVI, through ``data.ndvi.ndvi`` and nothing else), and the manifest. No
encoder is imported, no weight is loaded, no embedding is recomputed. Frozen by
construction, because no network is present. CPU only.

THE BASELINE CONTAINS THE TARGET, AND THAT IS WHY THERE ARE TWO K2 VERDICTS
----------------------------------------------------------------------------
``raw_features`` is the repo's mandatory not-a-network baseline, and P2's spec
names it as the K2 comparison. It is also, at the matched level, the identity
function on this target: ``encoders.raw_features.RAW_FEATURE_NAMES`` ends in
``NDVI_mean, NDVI_std, NDVI_p10 ... NDVI_p90``, computed over the same grid
cell, so a ridge on the ``raw_features`` grid recovers ``cell_mean`` NDVI at
**R2 = +1.000** and its pooled vector recovers ``cube_mean`` at **+0.999**
(measured, this subset). No frozen RGB encoder can beat that, and a verdict
that every network is "lossy" against a baseline holding the answer measures
the baseline, not the representation.

The degeneracy is a property of the LEVEL PAIRING, not of the baseline as
such. It bites where the feature level matches the target's: cell features
against a cell target, pooled features against a cube target. At the PRIMARY
K2 configuration -- ``cube_mean`` from ``grid_cell`` features -- predicting the
cube's mean from one cell's statistics is not the identity, and the baseline
scores an ordinary +0.440 (band-matched +0.417). So the primary verdict is a
real gate; the degeneracy shows up in the secondary views, where it must be
read as "the baseline was handed the answer" and not as an encoder result.

Both verdicts are therefore recorded, and they answer different questions:

    k2_verdict                vs ``raw_features``. Exactly as specified, and
                              at the primary configuration it is a fair
                              comparison. Read it with care in the pooled and
                              cell_mean views, where the baseline holds the
                              target.
    k2_verdict_band_matched   vs ``raw_rgb_only`` -- B02/B03/B04 statistics
                              only, the SAME three bands every network encoder
                              receives, and no NDVI column. Never degenerate at
                              any level. **This is the verdict that decides P3
                              inclusion**, and the one to quote.

This is the same asymmetry P1 found and fixed the same way: the band-matched
view is a column slice of an array already on disk (``RAW_BAND_COLUMNS``,
derived from the baseline's own feature names and asserted to partition them),
so it costs no re-encode and cannot silently point at the wrong bands.

THE TWO TIME AXES, AND THE ONE THAT DEFINES A GAP
--------------------------------------------------
A pair is two CONSECUTIVE RETAINED frames of one cube, ordered by
``original_axis_index``. The gap between them is measured in DAYS, on
``daily_axis_index``, and never on the acquisition axis. On this subset the
three candidate quantities are not close (measured over 244 pairs):

    gap_days        (daily_axis_index)      min 5   median 10   max 35
    gap_acq_steps   (original_axis_index)   min 1   median  2   max  6
    gap_frame_steps (array position)        always 1

They disagree on **244 of 244 pairs**, by a factor of about five at the median
-- one Sentinel-2 orbit lattice, 5 days per acquisition step. Using
``original_axis_index`` as a gap is the same category error that put P4's
weather features on the wrong day (0 of 264 rows correct, median 53 days off),
and it is silent: the wrong column is finite, integer, in range, monotone
within a cube, and correlated with the right one. ``assert_gap_axes_disagree``
therefore checks the three against each other ON THE REAL DATA at run time --
a guard that has never fired on real data is a hypothesis, not a control.

Why the gap has to be right: gap length is set by CLOUD (a long gap means the
frames between were dropped), cloud is precipitation, and precipitation is
weather. A gap measured in acquisitions launders that dependence.

COMMON-MASKING IS MANDATORY, AND IT IS NOT THE SAME AS MASKING EACH FRAME
--------------------------------------------------------------------------
The NDVI change of a pair is computed over pixels valid in BOTH frames, from
the per-pixel masks cached in Phase 1.2b. Differencing two per-frame means
instead -- each over its own valid set -- compares two different pieces of
ground and attributes the difference to time. The two agree only when the masks
agree, which is exactly when it does not matter.

The surviving pixel count is REPORTED PER PAIR (to the run log) and summarised
per gap length (to stdout), because a collapse at some gap length would be a
finding about what this benchmark can support, not a reason to drop pairs. On
this subset it does not collapse: 244/244 pairs survive, median 88.8% of
pixels, minimum 27.2%, and survival does NOT fall monotonically with gap --
0.895 at 5 days, 0.783 at 15, 0.985 at 30. Long gaps exist BECAUSE the frames
between were cloudy, which leaves the two surviving endpoints unusually clear.

TARGETS, SPATIALLY AGGREGATED, NEVER PIXEL-WISE
-------------------------------------------------
    cube_mean   PRIMARY. One row per frame (Part A) or per pair (Part B).
    cube_p90    Cube-90th-percentile. Secondary. Same rows.
    cell_mean   The 16 grid cells, 16 rows per frame/pair. The RESOLUTION axis.

Part B additionally splits each aggregation into the two targets the spec asks
for: ``sign`` and ``magnitude`` of the common-masked change. Every table states
its aggregation level in the ``aggregation`` column; no number here is
level-free.

THE FOUR CONTROLS, NONE OF THEM OPTIONAL
------------------------------------------
    gap_only        ``gap_days`` -> change magnitude/sign. NO embedding. This
                    is P2's analogue of P1's degenerate control, and it is not
                    optional for the same reason: P1 measured [clear_frac,
                    window_span_days] -- two numbers, no image -- decoding
                    season at 0.646-0.658, at or above most foundation models.
                    Measured here: gap alone reaches Spearman +0.209 against
                    ``cube_mean`` magnitude. Any delta score below that is a
                    calendar, not a representation. Reported as
                    ``margin_over_control``, exactly as P4 reports it.
    retention       [clear_frac, window_span_days] -> current NDVI. Part A's
                    version of the same control, because current NDVI is
                    strongly seasonal and cloud retention is strongly seasonal.
    raw_features    The raw-pixel-delta comparison row. Without it no encoder's
                    delta score has a scale: "Spearman 0.27" means nothing
                    until you know what differencing seven percentiles of three
                    bands achieves on the same pairs.
    raw_rgb_only    The BAND-MATCHED raw delta. The fair one -- see above.

Controls are emitted under EVERY filter label they are invariant to (every
encoder view, every feature level, every read-out) so that filtering the CSV
can never leave a table with no control in it, and
``assert_control_identical_across_views`` asserts the copies agree
digit-for-digit rather than leaving the duplication on trust.

``fold_mode`` is part of the control's KEY, not one of the labels it is
invariant to: a control evaluated on a different set of held-out pairs is a
different number, and reporting one value across modes would be a false
identity. ``assert_degenerate_control_present`` separately asserts the control
exists for every fold mode.

THE MULTI-IMAGE ENCODER IS NOT IN THE SAME COLUMN, AND HERE IT IS WORSE
------------------------------------------------------------------------
``satlas_s2_swinb_mi_rgb`` pools 8 RETAINED frames. Two CONSECUTIVE embeddings
therefore share up to 7 of their 8 source frames, so ``E(t+1) - E(t)`` is
mostly the difference between "frames 1-8" and "frames 2-9" -- a sliding-window
increment, not a state change over the gap. Its delta scores are reported and
flagged ``si_comparable=False``, exactly as P1 flagged its pooled score, and it
is EXCLUDED from the single-image ranking used for the structural hypothesis.
``assert_mi_flagged_and_excluded`` enforces both halves.

LEAKAGE IS PREVENTED BY THE SIGNATURE, NOT BY DISCIPLINE
----------------------------------------------------------
Every function here that FITS A CONTROL or SELECTS A HYPERPARAMETER takes the
training index set as a REQUIRED POSITIONAL argument -- ``fit_gap_control(
gap_days, values, train_idx, ...)`` and ``select_ridge_alpha(block, manifest,
train_rows, ...)`` -- matching ``p4_ceiling.doy_climatology_within_fold`` and
``p1_appearance.select_hyperparameter``. Neither can be fitted on everything by
omitting an argument, which is a property a test can check.

Each gets TWO poisoning tests, not one: poisoning the TEST fold must leave the
fit bit-identical, AND poisoning the TRAINING fold must change it. The first
alone passes trivially on a function that ignores its input entirely; the
second is what proves the first measured something.

EFFECTIVE SAMPLE SIZE IS CUBES
--------------------------------
264 frames, 244 pairs and 4224 cells are not independent observations: 16 cells
share a sky, ~13 frames share a place. ``effective_n`` counts CUBES, sits on
every row beside the score, and every interval is a cube-clustered (fold-level)
CI -- ``p4_ceiling.fold_clustered_ci``, imported, not re-derived. A correlation
quoted here without its effective n is not reportable.

PROTOCOL
--------
Primary split ``mode="cube"``, k=5. Robustness: leave-one-cube-out and
``spatial_block``, both reported. Nested CV for the ridge penalty: the inner
split runs on the OUTER TRAINING FOLD ONLY, via ``probes.cv.folds`` on the
training sub-manifest. Standardisation is fitted on train and applied to test,
per fold, never on the full array. Every fold comes from ``probes/cv.py``; any
number produced outside it does not exist.
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass
from typing import NamedTuple, Sequence

import numpy as np

from data.paths import phase_dir
from encoders.base import GRID, GRID_CELLS
from encoders.frames import MIN_CLEAR_FRACTION, select_clear_frames
from encoders.pipeline import load_masks
from probes import cv
from probes.p1_appearance import (
    BASELINE_ENCODER,
    ENCODER_ORDER,
    MI_ENCODER,
    RAW_BAND_COLUMNS,
    RIDGE_ALPHA_GRID,
    FeatureBlock,
    degenerate_arrays,
    load_encoder_arrays,
)
from probes.p4_ceiling import (
    GROWING_SEASON_DOY,
    NDVI_PLAUSIBILITY_FLOOR,
    assert_effective_n_counts_cubes,
    cube_frame_targets,
    fold_clustered_ci,
    outer_folds,
)

__all__ = [
    "PHASE", "PARTS", "AGGREGATIONS", "DELTA_TARGETS", "FOLD_MODES",
    "FEATURE_LEVELS", "FEATURE_SETS", "READOUTS", "MODEL_KINDS", "CONTROL_KINDS",
    "ENCODER_ORDER", "MI_ENCODER", "BASELINE_ENCODER", "BAND_MATCHED_BASELINE",
    "GAP_CONTROL_DEGREE", "TARGET_LEVEL", "PART_A_COMBOS", "PART_B_COMBOS",
    "K2_PRIMARY", "STRUCTURAL_PRIMARY",
    # logging
    "open_run_log", "close_run_log", "log", "run_log_path",
    # pairs and the axis guard
    "DeltaPairs", "pair_index", "filter_pairs_by_plausibility",
    "assert_gap_axes_disagree",
    # common-masked targets
    "common_masked_delta", "cube_common_masked_deltas", "build_pair_targets",
    "frame_targets", "screen_part_a_target", "summarise_pixel_survival",
    # features
    "encoder_arrays", "reconstruction_block", "delta_block", "retention_block",
    # the fitted control and the tuner -- both take train_idx positionally
    "GapControl", "gap_design", "fit_gap_control", "select_ridge_alpha",
    "ridge_path", "spearman",
    # evaluation
    "FoldResult", "evaluate_fold_a", "evaluate_fold_b", "evaluate", "summarise",
    # the run and the table
    "run_p2", "add_margins", "add_k2_verdicts", "structural_hypothesis",
    "print_controls", "print_k2_verdicts", "print_headlines",
    "assert_k2_verdict_recorded", "assert_control_identical_across_views",
    "assert_degenerate_control_present", "assert_mi_flagged_and_excluded",
    "assert_effective_n_counts_cubes", "assert_results_complete",
    "assert_plausibility_screen_declared", "P2_PLAUSIBILITY_SCREEN_LABEL",
    "results_path",
]

# Phase 1.6 is P2. Artefacts are phase-scoped through data/paths.py; nothing
# here ever writes to a hand-typed path.
PHASE = "phase1_6"

PARTS = ("A_reconstruction", "B_delta")

# The spatial aggregation of the target. Stated in every table, because a
# correlation without its aggregation level is not a number anyone can use.
AGGREGATIONS = ("cube_mean", "cube_p90", "cell_mean")
TARGET_LEVEL = {"cube_mean": "frame", "cube_p90": "frame", "cell_mean": "cell"}

# Part B's two targets, per aggregation.
DELTA_TARGETS = ("sign", "magnitude")

FOLD_MODES = ("cube", "loco", "spatial_block")

# The feature level. grid_cell is PRIMARY for the same reason it is in P1:
# 264 pooled rows against D up to 3840 is p >> n, and exploding to 16 cells per
# frame is the only level on this subset where the design matrix is not wider
# than it is tall. It fixes the FIT; it does not create 4224 observations.
FEATURE_LEVELS = ("grid_cell", "pooled")

# What the columns of X actually are.
FEATURE_SETS = ("embedding", "raw_rgb_only", "raw_nir_ndvi", "retention", "gap_days")

# Part B's two read-outs of the same difference vector.
READOUTS = ("norm", "linear")

MODEL_KINDS = ("reconstruction", "retention", "delta", "gap_only")
CONTROL_KINDS = ("retention", "gap_only")

# The band-matched baseline: the SAME three bands the networks receive, no NDVI
# column. See the module docstring for why the full baseline cannot serve.
BAND_MATCHED_BASELINE = "raw_rgb_only"

# The gap control's design is [1, gap, gap^2]. Degree 2 and not 1 because a
# straight line in gap is MONOTONE, and a monotone map has the same Spearman as
# gap itself -- which would make the control's fitted coefficients irrelevant to
# its reported value and the training-poison test vacuous. It is also the
# honest shape: NDVI change over a gap saturates, it does not grow linearly.
GAP_CONTROL_DEGREE = 2

# (aggregation, feature_level) combinations that are defined.
# A pooled vector cannot address one of 16 cells, so cell_mean is grid_cell only.
PART_A_COMBOS = (("cube_mean", "grid_cell"), ("cube_mean", "pooled"),
                 ("cube_p90", "grid_cell"), ("cube_p90", "pooled"),
                 ("cell_mean", "grid_cell"))
PART_B_COMBOS = PART_A_COMBOS

# The configuration the K2 verdict is read off: the primary target, the primary
# feature level, and cube-clustered folds. Named as a constant so the verdict
# cannot quietly be taken from whichever row happened to look best.
K2_PRIMARY = {"aggregation": "cube_mean", "feature_level": "grid_cell",
              "fold_mode": "cube"}

# The configuration the structural hypothesis is ranked on. Sign, not
# magnitude: magnitude is not separable from the gap control on this subset
# (see the log), so a ranking on it would be a ranking of noise.
STRUCTURAL_PRIMARY = {"aggregation": "cube_mean", "delta_target": "sign",
                      "feature_level": "pooled", "readout": "linear",
                      "fold_mode": "cube"}

P2_PLAUSIBILITY_SCREEN_LABEL = (
    "p4_ceiling.cube_frame_targets frame_plausible: a common-masked cube-mean "
    f"NDVI below {NDVI_PLAUSIBILITY_FLOOR} inside days of year "
    f"{GROWING_SEASON_DOY[0]}-{GROWING_SEASON_DOY[1]} is cloud, not "
    "vegetation. APPLIED here: Part A excludes that frame; Part B excludes a "
    "delta pair if either endpoint frame fails."
)

_INNER_K = 3          # inner folds for the nested tuning loop, as in P1
_MIN_TEST_ROWS = 8    # below this a fold's Spearman is noise; recorded as NaN


# ---------------------------------------------------------------------------
# Console hygiene: full per-pair detail to a file, headlines to stdout
# ---------------------------------------------------------------------------

_LOG_FH = None
_LOG_PATH = None


def run_log_path(name: str = "p2_run.log") -> str:
    return os.path.join(phase_dir(PHASE, "logs"), name)


def open_run_log(path: str | None = None, verbose: bool = True) -> str:
    """Open the per-pair log. Everything verbose goes here, not to stdout.

    244 pairs x 5 encoders x 3 fold modes is tens of thousands of lines. A
    console that prints all of it is a console nobody reads, and the shape
    assertions and control values -- the lines that would actually catch a
    wrong number -- scroll past. So the detail is written to
    ``data/phase1_6/logs/p2_run.log`` and stdout keeps the assertions, the K2
    verdicts, the four controls and the headline correlations.
    """
    global _LOG_FH, _LOG_PATH
    close_run_log()
    _LOG_PATH = path or run_log_path()
    os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
    # Line-buffered: a scaled run takes tens of minutes and writes tens of
    # thousands of lines, and a block-buffered log cannot be tailed to see
    # where it is. The cost is one write syscall per line, against a run whose
    # unit of work is a ridge solve.
    _LOG_FH = open(_LOG_PATH, "w", encoding="utf-8", buffering=1)
    if verbose:
        print(f"[p2] per-pair detail -> {_LOG_PATH} (stdout keeps shapes, "
              "K2 verdicts, the four controls and the headlines)")
    return _LOG_PATH


def close_run_log() -> None:
    global _LOG_FH
    if _LOG_FH is not None:
        _LOG_FH.flush()
        _LOG_FH.close()
        _LOG_FH = None


def log(msg: str, echo: bool = False) -> None:
    """Write one line to the run log; ``echo=True`` also puts it on stdout.

    With no log open the detail is dropped rather than printed: a probe
    imported into a test must not spray tens of thousands of lines at pytest.
    """
    if _LOG_FH is not None:
        _LOG_FH.write(msg + "\n")
    if echo:
        print(msg)
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# Pairs, and the axis that defines their gap
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeltaPairs:
    """Consecutive retained frames of one cube, with THREE candidate "gaps".

    Only ``gap_days`` is a gap. The other two are carried so that
    ``assert_gap_axes_disagree`` can prove, on the real data and at run time,
    that the correct column was used -- a check that cannot be made against a
    docstring, and that a test comparing the right column to itself would pass
    without measuring anything.

    ``row_a`` / ``row_b`` are POSITIONS into the manifest's rows, so a fold
    defined by ``probes.cv`` selects pairs directly.
    """

    row_a: np.ndarray            # (P,) manifest row position of frame t
    row_b: np.ndarray            # (P,) manifest row position of frame t+1
    cube_id: np.ndarray          # (P,) str
    gap_days: np.ndarray         # (P,) float -- THE gap. From daily_axis_index.
    gap_acq_steps: np.ndarray    # (P,) int   -- NOT the gap. Acquisition axis.
    gap_frame_steps: np.ndarray  # (P,) int   -- NOT the gap. Always 1.
    within_cube_index: np.ndarray  # (P,) int, position of the pair in its cube

    def __post_init__(self):
        P = self.row_a.shape[0]
        for name in ("row_b", "cube_id", "gap_days", "gap_acq_steps",
                     "gap_frame_steps", "within_cube_index"):
            got = getattr(self, name).shape
            assert got == (P,), f"DeltaPairs.{name} is {got}, expected {(P,)}"
        assert P, "no pairs"

    @property
    def n_pairs(self) -> int:
        return int(self.row_a.shape[0])

    @property
    def n_cubes(self) -> int:
        return int(np.unique(self.cube_id).size)


def pair_index(manifest, verbose: bool = True) -> DeltaPairs:
    """Every (t, t+1) pair of CONSECUTIVE RETAINED frames, per cube.

    Ordered by ``original_axis_index``, not by array position: the manifest's
    row order is whatever ``build_manifest`` produced, and a pair built from
    adjacent ROWS would silently pair the last frame of one cube with the first
    of the next if the manifest were ever re-sorted.

    The gap is read from ``daily_axis_index`` and cross-checked against the
    timestamps, which are an independent column. They must agree exactly: the
    daily axis is a contiguous day grid, so a disagreement means the manifest's
    two time columns came from different loads.
    """
    for col in ("cube_id", "original_axis_index", "daily_axis_index", "timestamp"):
        assert col in manifest.columns, (
            f"manifest has no {col!r}; build it with "
            "encoders.manifest.build_manifest (and rebuild it if it predates "
            "the 2026-08-10 daily_axis_index fix)"
        )
    row_a, row_b, cubes, g_day, g_acq, widx = [], [], [], [], [], []
    ts_all = np.asarray(manifest["timestamp"].to_numpy(), dtype="datetime64[ns]")

    for cube, sub in manifest.groupby("cube_id", sort=True):
        pos = np.flatnonzero((manifest["cube_id"] == cube).to_numpy())
        pos = pos[np.argsort(manifest["original_axis_index"].to_numpy()[pos],
                             kind="stable")]
        oai = manifest["original_axis_index"].to_numpy()[pos]
        dai = manifest["daily_axis_index"].to_numpy()[pos]
        assert (np.diff(oai) > 0).all(), (
            f"{cube}: original_axis_index is not strictly increasing after "
            "sorting -- the manifest has duplicate frames for this cube"
        )
        assert (np.diff(dai) > 0).all(), (
            f"{cube}: daily_axis_index is not strictly increasing in "
            "acquisition order; the two axes are inconsistent for this cube"
        )
        for i in range(pos.size - 1):
            row_a.append(pos[i]); row_b.append(pos[i + 1])
            cubes.append(cube)
            g_day.append(float(dai[i + 1] - dai[i]))
            g_acq.append(int(oai[i + 1] - oai[i]))
            widx.append(i)

    pairs = DeltaPairs(
        row_a=np.asarray(row_a, dtype=int), row_b=np.asarray(row_b, dtype=int),
        cube_id=np.asarray(cubes, dtype=object),
        gap_days=np.asarray(g_day, dtype=float),
        gap_acq_steps=np.asarray(g_acq, dtype=int),
        # The "number of retained frames between t and t+1" reading of a gap.
        # It is 1 for every consecutive pair by construction, which is exactly
        # why it is useless as a gap and worth carrying as a foil.
        gap_frame_steps=np.ones(len(row_a), dtype=int),
        within_cube_index=np.asarray(widx, dtype=int),
    )

    # The independent cross-check: days between the two timestamps.
    from_ts = ((ts_all[pairs.row_b] - ts_all[pairs.row_a])
               / np.timedelta64(1, "D")).astype(float)
    np.testing.assert_array_equal(
        pairs.gap_days, from_ts,
        err_msg="gap_days from daily_axis_index disagrees with the gap from "
                "the timestamps. The daily axis is a contiguous day grid, so "
                "these are the same quantity computed two ways; a "
                "disagreement means the manifest's time columns came from "
                "different loads. Rebuild it with build_manifest.")

    if verbose:
        print(f"[p2] pairs {pairs.row_a.shape} over {pairs.n_cubes} cubes "
              f"({len(manifest)} retained frames -> {pairs.n_pairs} consecutive "
              "pairs)")
        print(f"[p2]   gap_days (daily_axis_index): min {pairs.gap_days.min():.0f} "
              f"median {np.median(pairs.gap_days):.0f} max {pairs.gap_days.max():.0f} "
              f"| distinct {sorted(set(pairs.gap_days.tolist()))}")
    return pairs


def filter_pairs_by_plausibility(pairs: DeltaPairs,
                                 frame_plausible,
                                 verbose: bool = True) -> tuple:
    """Drop each delta pair that touches an implausible endpoint frame."""
    ok = np.asarray(frame_plausible, dtype=bool)
    assert ok.ndim == 1, ok.shape
    assert (pairs.row_a >= 0).all() and (pairs.row_b < ok.size).all(), (
        f"pair endpoints reach outside a plausibility mask of length {ok.size}"
    )
    keep = ok[pairs.row_a] & ok[pairs.row_b]
    assert keep.any(), "the plausibility screen removed every P2 pair"
    filtered = DeltaPairs(**{
        name: getattr(pairs, name)[keep]
        for name in ("row_a", "row_b", "cube_id", "gap_days",
                     "gap_acq_steps", "gap_frame_steps", "within_cube_index")
    })
    n_dropped = int((~keep).sum())
    if verbose:
        print(f"[p2] plausibility screen: {pairs.n_pairs} -> "
              f"{filtered.n_pairs} pairs ({n_dropped} touching an "
              "implausible endpoint dropped)")
    return filtered, n_dropped


def assert_gap_axes_disagree(pairs: DeltaPairs, min_ratio: float = 2.0,
                             verbose: bool = True) -> dict:
    """The LIVE check that the gap came from the day axis, not the join key.

    Three quantities could be called "the gap", and on this data they are not
    close: ``gap_days`` (daily axis) has median 10, ``gap_acq_steps``
    (``original_axis_index``, the EMBEDDING JOIN KEY) has median 2, and
    "retained frames between t and t+1" is 1 by construction. This asserts they
    are MATERIALLY different -- a test that cannot distinguish the correct
    column from the wrong one proves nothing, and a docstring warning proves
    less.

    ``min_ratio`` is the smallest median ratio that counts as material. The
    measured value on this subset is 5.0 (one Sentinel-2 orbit lattice: 5 days
    per acquisition step), so 2.0 is loose on purpose -- the assertion is
    guarding against the two columns being the SAME column, not against a
    particular revisit cadence.
    """
    d = np.asarray(pairs.gap_days, dtype=float)
    a = np.asarray(pairs.gap_acq_steps, dtype=float)
    f = np.asarray(pairs.gap_frame_steps, dtype=float)
    n_same = int((d == a).sum())
    med_d, med_a, med_f = float(np.median(d)), float(np.median(a)), float(np.median(f))
    ratio = med_d / med_a if med_a else float("inf")

    assert n_same == 0, (
        f"{n_same}/{d.size} pairs have gap_days == gap_acq_steps. On this "
        "subset the two axes disagree on EVERY pair (5 days per acquisition "
        "step). Equality means the gap was computed from "
        "original_axis_index -- the embedding join key, which counts "
        "ACQUISITIONS -- and every gap in this phase is wrong by about a "
        "factor of five. This is the same category error that put P4's "
        "weather features on the wrong day."
    )
    assert ratio >= min_ratio, (
        f"median gap_days {med_d} is only {ratio:.2f}x median gap_acq_steps "
        f"{med_a}; the two are not materially different, so this check cannot "
        "distinguish the right column from the wrong one and must not be "
        "treated as having passed."
    )
    assert (f == 1).all() and med_d > min_ratio * med_f, (
        "the 'number of retained frames between t and t+1' reading of a gap is "
        f"{med_f} for a consecutive pair and gap_days is {med_d}; if these "
        "were comparable the counting reading would be defensible, and it is "
        "not."
    )
    out = {"n_pairs": int(d.size), "median_gap_days": med_d,
           "median_gap_acq_steps": med_a, "median_gap_frame_steps": med_f,
           "ratio_days_per_acq_step": float(ratio), "n_axes_equal": n_same,
           "distinct_gap_days": sorted(set(d.tolist())),
           "distinct_gap_acq_steps": sorted(set(a.astype(int).tolist()))}
    if verbose:
        print(f"[p2] GAP AXIS CHECK on {out['n_pairs']} real pairs: "
              f"gap_days median {med_d:.0f} vs gap_acq_steps median {med_a:.0f} "
              f"vs frames-between {med_f:.0f} -- "
              f"{ratio:.1f} days per acquisition step, "
              f"{n_same}/{out['n_pairs']} pairs where the two axes agree")
    return out


# ---------------------------------------------------------------------------
# Common-masked NDVI change
# ---------------------------------------------------------------------------

def common_masked_delta(ndvi_a, ndvi_b, mask_a, mask_b, grid: int = GRID) -> dict:
    """NDVI change of ONE pair over pixels valid in BOTH frames.

    THE INTERSECTION, NOT EITHER FRAME'S OWN MASK. Differencing two per-frame
    aggregates -- each taken over its own valid set -- compares two different
    pieces of ground and calls the difference a change. The two answers agree
    only when the masks agree, i.e. exactly when the distinction does not
    matter, which is why the error is invisible in a spot check.

    Returns the change at all three aggregations, plus the surviving pixel
    counts that make the collapse (or, here, the non-collapse) visible.
    """
    a = np.asarray(ndvi_a, dtype=np.float64)
    b = np.asarray(ndvi_b, dtype=np.float64)
    ma = np.asarray(mask_a)
    mb = np.asarray(mask_b)
    assert a.shape == b.shape == ma.shape == mb.shape and a.ndim == 2, (
        f"shapes must all be one (H, W): ndvi {a.shape}/{b.shape} "
        f"mask {ma.shape}/{mb.shape}"
    )
    assert ma.dtype == np.bool_ and mb.dtype == np.bool_, (
        f"masks must be boolean with True == VALID, got {ma.dtype}/{mb.dtype}. "
        "An integer cloud-code array is truthy almost everywhere and would "
        "silently disable common-masking."
    )
    H, W = a.shape
    assert H % grid == 0 and W % grid == 0, f"{H}x{W} not divisible by {grid}"

    both = ma & mb                      # <-- the only mask used below
    assert np.isfinite(a[both]).all() and np.isfinite(b[both]).all(), (
        "a pixel valid in both frames has a non-finite NDVI; the mask and the "
        "NDVI disagree, so the intersection is not the set it claims to be"
    )
    n_both = int(both.sum())
    ch, cw = H // grid, W // grid
    cell_n = both.reshape(grid, ch, grid, cw).sum(axis=(1, 3)).reshape(-1)

    if n_both == 0:
        # Reported, never imputed. A pair with no shared pixel has no change.
        nan16 = np.full(grid * grid, np.nan)
        return {"n_common_px": 0, "frac_common_px": 0.0,
                "n_px_a": int(ma.sum()), "n_px_b": int(mb.sum()),
                "d_cube_mean": np.nan, "d_cube_p90": np.nan,
                "d_cell_mean": nan16, "n_common_px_cell": cell_n,
                "cell_valid": np.zeros(grid * grid, dtype=bool),
                "ndvi_a_mean": np.nan, "ndvi_b_mean": np.nan}

    va, vb = a[both], b[both]
    d_mean = float(vb.mean() - va.mean())
    d_p90 = float(np.percentile(vb, 90) - np.percentile(va, 90))

    with warnings.catch_warnings():
        # An all-masked CELL of the intersection is a real state; it is
        # reported through cell_valid and dropped, never filled.
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        d = np.where(both, b - a, np.nan)
        d_cells = np.nanmean(d.reshape(grid, ch, grid, cw), axis=(1, 3)).reshape(-1)
    cell_valid = cell_n > 0
    assert np.isfinite(d_cells[cell_valid]).all(), (
        "a cell with surviving common pixels produced a non-finite change"
    )

    return {"n_common_px": n_both, "frac_common_px": float(n_both / (H * W)),
            "n_px_a": int(ma.sum()), "n_px_b": int(mb.sum()),
            "d_cube_mean": d_mean, "d_cube_p90": d_p90,
            "d_cell_mean": d_cells, "n_common_px_cell": cell_n,
            "cell_valid": cell_valid,
            "ndvi_a_mean": float(va.mean()), "ndvi_b_mean": float(vb.mean())}


def cube_common_masked_deltas(sample, masks, min_clear: float = MIN_CLEAR_FRACTION,
                              grid: int = GRID, verbose: bool = False) -> dict:
    """Every consecutive pair of ONE cube, common-masked.

    The per-pixel masks come from the Phase 1.2b cache rather than being
    recomputed, and the cache's ``kept_idx`` and timestamps are asserted equal
    to this cube's frame selection -- if they ever diverge, the masks belong to
    a different frame set and every intersection below is over the wrong
    frames. NDVI comes through ``data.loader.cube_ndvi``, i.e. through the
    canonical ``data.ndvi.ndvi``, and nothing else.
    """
    from data.loader import CubeSample, cube_ndvi

    sel = select_clear_frames(sample.values, sample.timestamps, sample.mask,
                              min_clear=min_clear, verbose=False)
    name = os.path.basename(sample.path)
    assert np.array_equal(np.asarray(masks.kept_idx).astype(int),
                          np.asarray(sel.kept_idx).astype(int)), (
        f"{name}: the cached mask's kept_idx {np.asarray(masks.kept_idx).tolist()} "
        f"!= this cube's frame selection {np.asarray(sel.kept_idx).tolist()}. "
        "The mask cache was written under a different mask definition or "
        "min_clear; reset_phase('phase1_2') and re-cache."
    )
    assert (np.asarray(masks.timestamps) == np.asarray(sel.timestamps)).all(), (
        f"{name}: cached mask timestamps differ from the cube's retained frames"
    )
    kept = CubeSample(values=sel.values, timestamps=sel.timestamps, mask=sel.mask,
                      path=sample.path, bands=sample.bands)
    nd = cube_ndvi(kept)                     # (T_kept, H, W), NaN where masked
    T, H, W = nd.shape
    assert masks.mask.shape == nd.shape, (
        f"{name}: cached mask {masks.mask.shape} != NDVI {nd.shape}"
    )
    # The cache and the NDVI must agree about what is valid, or "the pixels
    # valid in both frames" is not the set this function computes.
    assert (masks.mask == np.isfinite(nd)).all(), (
        f"{name}: the cached per-pixel mask does not equal the finiteness of "
        "the canonical NDVI. One of them is stale; the intersection would be "
        "over the wrong pixels and nothing downstream could detect it."
    )
    assert T >= 2, f"{name}: {T} retained frame(s), no pair exists"

    out = {k: [] for k in ("n_common_px", "frac_common_px", "n_px_a", "n_px_b",
                           "d_cube_mean", "d_cube_p90", "ndvi_a_mean", "ndvi_b_mean")}
    d_cells, n_cells, valid_cells = [], [], []
    for i in range(T - 1):
        r = common_masked_delta(nd[i], nd[i + 1], masks.mask[i], masks.mask[i + 1],
                                grid=grid)
        for k in out:
            out[k].append(r[k])
        d_cells.append(r["d_cell_mean"])
        n_cells.append(r["n_common_px_cell"])
        valid_cells.append(r["cell_valid"])
        log(f"[p2] pair {name} {i}->{i+1} | common px {r['n_common_px']:>6} "
            f"({r['frac_common_px']:.3f}) of a {r['n_px_a']} / b {r['n_px_b']} | "
            f"d_cube_mean {r['d_cube_mean']:+.5f} d_cube_p90 {r['d_cube_p90']:+.5f} "
            f"| cells with common px {int(r['cell_valid'].sum())}/{grid*grid}")

    res = {k: np.asarray(v, dtype=float) for k, v in out.items()}
    res["d_cell_mean"] = np.asarray(d_cells, dtype=float)
    res["n_common_px_cell"] = np.asarray(n_cells, dtype=float)
    res["cell_valid"] = np.asarray(valid_cells, dtype=bool)
    res["timestamps"] = np.asarray(sel.timestamps)
    res["kept_idx"] = np.asarray(sel.kept_idx).astype(int)
    assert res["d_cube_mean"].shape == (T - 1,)
    assert res["d_cell_mean"].shape == (T - 1, grid * grid)
    if verbose:
        print(f"[p2] {name}: {T - 1} pairs | common px median "
              f"{np.median(res['n_common_px']):.0f} "
              f"({np.median(res['frac_common_px']):.3f})")
    return res


def build_pair_targets(pairs: DeltaPairs, manifest, cube_dir: str,
                       mask_dir: str | None = None,
                       min_clear: float = MIN_CLEAR_FRACTION,
                       verbose: bool = True) -> dict:
    """Common-masked change for every pair, aligned to ``pairs`` row order."""
    from data.loader import load_cube

    mask_dir = mask_dir or phase_dir("phase1_2", "masks")
    cubes = sorted(np.unique(pairs.cube_id).tolist())
    per_cube = {}
    for cube in cubes:
        mp = os.path.join(mask_dir, f"{os.path.splitext(cube)[0]}__masks.npz")
        assert os.path.exists(mp), (
            f"no cached per-pixel mask for {cube!r} at {mp}. Common-masking is "
            "mandatory and cannot be approximated from clear_frac; re-run "
            "Phase 1.2b's mask cache."
        )
        sample = load_cube(os.path.join(cube_dir, cube), verbose=False)
        per_cube[cube] = cube_common_masked_deltas(
            sample, load_masks(mp), min_clear=min_clear, verbose=False)

    P = pairs.n_pairs
    tgt = {"cube_mean": np.full(P, np.nan), "cube_p90": np.full(P, np.nan),
           "n_common_px": np.full(P, np.nan), "frac_common_px": np.full(P, np.nan)}
    cell = np.full((P, GRID_CELLS), np.nan)
    cell_n = np.full((P, GRID_CELLS), np.nan)
    cell_ok = np.zeros((P, GRID_CELLS), dtype=bool)
    for j in range(P):
        c = per_cube[pairs.cube_id[j]]
        i = int(pairs.within_cube_index[j])
        tgt["cube_mean"][j] = c["d_cube_mean"][i]
        tgt["cube_p90"][j] = c["d_cube_p90"][i]
        tgt["n_common_px"][j] = c["n_common_px"][i]
        tgt["frac_common_px"][j] = c["frac_common_px"][i]
        cell[j] = c["d_cell_mean"][i]
        cell_n[j] = c["n_common_px_cell"][i]
        cell_ok[j] = c["cell_valid"][i]
    tgt["cell_mean"] = cell
    tgt["n_common_px_cell"] = cell_n
    tgt["cell_valid"] = cell_ok

    assert tgt["cube_mean"].shape == (P,) and tgt["cell_mean"].shape == (P, GRID_CELLS)
    n_dead = int((tgt["n_common_px"] == 0).sum())
    assert np.isfinite(tgt["cube_mean"][tgt["n_common_px"] > 0]).all()
    if verbose:
        print(f"[p2] pair targets: cube_mean {tgt['cube_mean'].shape} | "
              f"cube_p90 {tgt['cube_p90'].shape} | cell_mean {tgt['cell_mean'].shape} "
              f"| {n_dead}/{P} pairs with ZERO common pixels | "
              f"{int((~cell_ok).sum())}/{cell_ok.size} cells with none")
    return tgt


def summarise_pixel_survival(pairs: DeltaPairs, targets: dict,
                             verbose: bool = True):
    """Common-masked pixel survival BY GAP LENGTH -- reported, never assumed.

    If the intersection collapses at some gap length, that is a finding about
    what this benchmark can support at that horizon, and it belongs in the
    record rather than in a silent drop. Returns a DataFrame, one row per
    distinct gap.
    """
    import pandas as pd

    df = pd.DataFrame({"gap_days": pairs.gap_days,
                       "n_common_px": targets["n_common_px"],
                       "frac_common_px": targets["frac_common_px"],
                       "n_cells": targets["cell_valid"].sum(axis=1)})
    g = (df.groupby("gap_days")
           .agg(n_pairs=("n_common_px", "size"),
                px_min=("n_common_px", "min"),
                px_median=("n_common_px", "median"),
                px_max=("n_common_px", "max"),
                frac_min=("frac_common_px", "min"),
                frac_median=("frac_common_px", "median"),
                cells_min=("n_cells", "min"),
                cells_median=("n_cells", "median"))
           .reset_index())
    g["n_pairs_zero_common"] = [
        int(((df.gap_days == d) & (df.n_common_px == 0)).sum()) for d in g.gap_days]
    if verbose:
        print("[p2] COMMON-MASKED PIXEL SURVIVAL BY GAP LENGTH "
              "(16384 px per frame, 16 cells)")
        print(f"[p2]   {'gap_d':>6} {'pairs':>6} {'px_min':>8} {'px_med':>8} "
              f"{'frac_min':>9} {'frac_med':>9} {'cells_min':>10} {'zero':>5}")
        for r in g.itertuples():
            print(f"[p2]   {r.gap_days:>6.0f} {r.n_pairs:>6d} {r.px_min:>8.0f} "
                  f"{r.px_median:>8.0f} {r.frac_min:>9.3f} {r.frac_median:>9.3f} "
                  f"{r.cells_min:>10.0f} {r.n_pairs_zero_common:>5d}")
        worst = g.loc[g.frac_median.idxmin()]
        print(f"[p2]   no collapse: {int(g.n_pairs_zero_common.sum())} pairs of "
              f"{int(g.n_pairs.sum())} have zero common pixels; the weakest gap "
              f"is {worst.gap_days:.0f} d at median {worst.frac_median:.3f} "
              "surviving. Survival is NOT monotone in gap -- a long gap exists "
              "because the frames between it were cloudy, which leaves the two "
              "surviving endpoints unusually clear.")
    return g


def frame_targets(manifest, cube_dir: str, min_clear: float = MIN_CLEAR_FRACTION,
                  verbose: bool = True) -> dict:
    """Part A's target: CURRENT NDVI per retained frame, aligned to the manifest.

    Delegates to ``p4_ceiling.cube_frame_targets`` -- the same aggregation, from
    the same canonical NDVI, so P2's floor and P4's ceiling are measured against
    the same target definition rather than two that happen to look alike. The
    returned ``frame_plausible`` verdict is aligned but not applied here.
    """
    from data.loader import load_cube

    cubes = sorted(manifest["cube_id"].unique().tolist())
    parts = {k: [] for k in ("cube_mean", "cube_p90", "cell_mean", "cell_valid",
                             "frame_plausible")}
    rows = []
    for cube in cubes:
        sample = load_cube(os.path.join(cube_dir, cube), verbose=False)
        t = cube_frame_targets(sample, min_clear=min_clear, verbose=False)
        pos = np.flatnonzero((manifest["cube_id"] == cube).to_numpy())
        order = np.argsort(manifest["original_axis_index"].to_numpy()[pos],
                           kind="stable")
        pos = pos[order]
        assert pos.size == t["cube_mean"].shape[0], (
            f"{cube}: manifest has {pos.size} retained rows but the cube yields "
            f"{t['cube_mean'].shape[0]} -- the manifest and the cube disagree "
            "about frame selection"
        )
        np.testing.assert_array_equal(
            manifest["original_axis_index"].to_numpy()[pos], t["kept_idx"],
            err_msg=f"{cube}: manifest original_axis_index != cube kept_idx")
        rows.append(pos)
        for k in parts:
            parts[k].append(t[k])

    rows = np.concatenate(rows)
    order = np.argsort(rows, kind="stable")
    assert rows[order].tolist() == list(range(len(manifest))), (
        "the per-cube targets are not a permutation of the manifest's rows"
    )
    out = {k: np.concatenate(v)[order] for k, v in parts.items()}
    n = len(manifest)
    assert out["cube_mean"].shape == out["cube_p90"].shape == (n,)
    assert out["cell_mean"].shape == out["cell_valid"].shape == (n, GRID_CELLS)
    assert out["frame_plausible"].shape == (n,)
    out["frame_plausible"] = np.asarray(out["frame_plausible"], dtype=bool)
    assert np.isfinite(out["cube_mean"]).all() and np.isfinite(out["cube_p90"]).all()
    if verbose:
        print(f"[p2] frame targets: cube_mean {out['cube_mean'].shape} | "
              f"cube_p90 {out['cube_p90'].shape} | cell_mean {out['cell_mean'].shape} "
              f"| {int((~out['cell_valid']).sum())}/{out['cell_valid'].size} cells "
              "fully masked (dropped, never filled) | "
              f"{int((~out['frame_plausible']).sum())}/{n} frames fail "
              "P4's shared plausibility verdict (reported, not yet applied)")
    return out


def screen_part_a_target(aggregation: str, y_rows, keep_rows,
                         frame_plausible) -> tuple:
    """Apply P2 Part A's frame verdict without shortening the manifest."""
    ok = np.asarray(frame_plausible, dtype=bool)
    y = np.asarray(y_rows, dtype=float).copy()
    assert y.shape[0] == ok.size, (
        f"{aggregation}: {y.shape[0]} target frames against {ok.size} verdicts"
    )
    if aggregation == "cell_mean":
        assert y.ndim == 2 and y.shape[1] == GRID_CELLS
        valid = np.asarray(keep_rows, dtype=bool).copy()
        assert valid.shape == y.shape
        valid &= ok[:, None]
        y[~ok, :] = np.nan
        return y, valid
    assert y.ndim == 1
    y[~ok] = np.nan
    return y, None


# ---------------------------------------------------------------------------
# Feature blocks
# ---------------------------------------------------------------------------

def encoder_arrays(manifest, encoders: Sequence[str] = ENCODER_ORDER,
                   emb_dir: str | None = None, verbose: bool = True) -> dict:
    """The cached arrays for every encoder, joined through ``probes.cv``.

    A thin loop over ``p1_appearance.load_encoder_arrays`` -- the join contract
    (cube_id, original_axis_index) == (cube, kept_idx) is asserted there, not
    re-implemented here.
    """
    out = {}
    for enc in encoders:
        out[enc] = load_encoder_arrays(manifest, enc, emb_dir=emb_dir,
                                       verbose=False)
        a = out[enc]
        if verbose:
            print(f"[p2] {enc:<24} pooled {a['pooled'].shape} | "
                  f"grid {a['grid'].shape} | clear_frac {a['clear_frac'].shape} | "
                  f"window_span_days {a['window_span_days'].shape} "
                  f"(max {a['window_span_days'].max():.0f} d)")
    return out


def _columns_for(feature_set: str, D: int) -> np.ndarray:
    if feature_set == "embedding":
        return np.arange(D)
    cols = RAW_BAND_COLUMNS[feature_set]
    assert D == 35, (
        f"{feature_set} is a column slice of the raw_features array (D=35), "
        f"but this array has D={D}. It is defined for raw_features only."
    )
    return cols


def reconstruction_block(arrays: dict, level: str,
                         feature_set: str = "embedding") -> FeatureBlock:
    """Part A's design matrix: E(frame_t) at ``level``.

    ``row_idx[i]`` is the MANIFEST row that X row i belongs to -- one-to-one at
    pooled level, 16-to-one at grid_cell level. It travels bound to X inside a
    ``FeatureBlock`` (imported from P1, where the reason it is a type and not
    three variables is documented at length) so a fold defined on manifest rows
    can select feature rows without a second, hand-rolled index.
    """
    assert level in FEATURE_LEVELS, f"level {level!r} not in {FEATURE_LEVELS}"
    n = arrays["pooled"].shape[0]
    if level == "pooled":
        X = arrays["pooled"]
        cols = _columns_for(feature_set, X.shape[1])
        X, row_idx = X[:, cols], np.arange(n)
    else:
        g = arrays["grid"]
        assert g.shape[1] == GRID_CELLS, g.shape
        cols = _columns_for(feature_set, g.shape[-1])
        X = g[:, :, cols].reshape(n * GRID_CELLS, cols.size)
        row_idx = np.repeat(np.arange(n), GRID_CELLS)
    X = np.ascontiguousarray(X, dtype=np.float64)
    assert np.isfinite(X).all(), f"{feature_set}/{level}: non-finite feature"
    return FeatureBlock(X=X, row_idx=row_idx,
                        name=f"{feature_set}/{level}")


def delta_block(arrays: dict, pairs: DeltaPairs, level: str, readout: str,
                feature_set: str = "embedding") -> FeatureBlock:
    """Part B's design matrix: a read-out of ``E(t+1) - E(t)`` at ``level``.

    ``norm``   ||E(t+1) - E(t)||, one column. A FIXED STATISTIC, not a fitted
               model: it uses no training data at all, so its per-fold score is
               the held-out rank correlation of the statistic itself. Sign-blind
               by construction -- a norm is non-negative -- which is why its
               score against the SIGN target is reported and expected near
               zero rather than quietly omitted.
    ``linear`` the raw difference vector, handed to a ridge read-out.

    ``row_idx`` here indexes PAIRS (and cells within pairs), not manifest rows;
    the caller maps a fold's manifest rows onto pairs through
    ``pairs.row_a``. Both frames of a pair are in the same cube and every fold
    mode in this repo is cube-grouped, so a pair can never straddle a fold --
    asserted per fold in ``evaluate_fold_b``, not assumed here.
    """
    assert level in FEATURE_LEVELS, f"level {level!r} not in {FEATURE_LEVELS}"
    assert readout in READOUTS, f"readout {readout!r} not in {READOUTS}"
    P = pairs.n_pairs
    if level == "pooled":
        src = arrays["pooled"].astype(np.float64)
        cols = _columns_for(feature_set, src.shape[1])
        dE = src[pairs.row_b][:, cols] - src[pairs.row_a][:, cols]
        row_idx = np.arange(P)
    else:
        g = arrays["grid"].astype(np.float64)
        cols = _columns_for(feature_set, g.shape[-1])
        dE = (g[pairs.row_b][:, :, cols] - g[pairs.row_a][:, :, cols])
        dE = dE.reshape(P * GRID_CELLS, cols.size)
        row_idx = np.repeat(np.arange(P), GRID_CELLS)
    assert dE.shape[0] == row_idx.size

    if readout == "norm":
        X = np.linalg.norm(dE, axis=1, keepdims=True)
    else:
        X = dE
    X = np.ascontiguousarray(X, dtype=np.float64)
    assert np.isfinite(X).all(), f"{feature_set}/{level}/{readout}: non-finite"
    return FeatureBlock(X=X, row_idx=row_idx,
                        name=f"delta:{feature_set}/{level}/{readout}")


def retention_block(arrays_by_encoder: dict, level: str,
                    verbose: bool = False) -> FeatureBlock:
    """Part A's control: [clear_frac, window_span_days]. NO embedding.

    Built from ``p1_appearance.degenerate_arrays``, which asserts clear_frac is
    identical across encoders and takes window_span_days from the MULTI-IMAGE
    encoder -- the only one where it is not identically zero, which makes this
    the strongest form of the control rather than the convenient one.
    """
    d = degenerate_arrays(arrays_by_encoder, verbose=verbose)
    n = d["clear_frac"].shape[0]
    if level == "pooled":
        X = np.column_stack([d["clear_frac"], d["window_span_days"]])
        row_idx = np.arange(n)
    else:
        X = np.column_stack([d["grid_clear_frac"].reshape(-1),
                             np.repeat(d["window_span_days"], GRID_CELLS)])
        row_idx = np.repeat(np.arange(n), GRID_CELLS)
    X = np.ascontiguousarray(X, dtype=np.float64)
    assert X.shape == (row_idx.size, 2), X.shape
    return FeatureBlock(X=X, row_idx=row_idx, name=f"retention/{level}")


# ---------------------------------------------------------------------------
# The gap-length-alone control. train_idx is REQUIRED and POSITIONAL.
# ---------------------------------------------------------------------------

def gap_design(gap_days, degree: int = GAP_CONTROL_DEGREE) -> np.ndarray:
    """(n, degree+1) polynomial design in gap length: [1, g, g^2, ...]."""
    g = np.asarray(gap_days, dtype=float)
    assert g.ndim == 1, f"gap_days must be 1-D, got {g.shape}"
    assert degree >= 1, f"degree must be >= 1, got {degree}"
    A = np.column_stack([g ** k for k in range(degree + 1)])
    assert A.shape == (g.size, degree + 1), A.shape
    assert np.isfinite(A).all()
    return A


@dataclass(frozen=True)
class GapControl:
    """A fitted gap-length-alone control, plus what it was fitted on.

    The provenance fields are not decoration: a control is only a control with
    respect to a training set, and one that cannot say which one it came from
    cannot be checked for the leak this class exists to prevent.
    """

    coef: np.ndarray
    degree: int
    n_train_rows: int
    gap_train_min: float
    gap_train_max: float
    train_r2: float
    label: str = ""

    def predict(self, gap_days) -> np.ndarray:
        g = np.asarray(gap_days, dtype=float)
        out = gap_design(g, self.degree) @ self.coef
        assert out.shape == g.shape and np.isfinite(out).all()
        return out


def fit_gap_control(gap_days, values, train_idx, *,
                    degree: int = GAP_CONTROL_DEGREE, label: str = "",
                    verbose: bool = False) -> GapControl:
    """P2's degenerate control: gap length -> NDVI change, fitted on TRAIN ONLY.

    THE SIGNATURE IS THE GUARANTEE, exactly as
    ``p4_ceiling.doy_climatology_within_fold`` and
    ``p1_appearance.select_hyperparameter`` are. This function is handed the
    full arrays and a TRAINING INDEX SET, and the arrays are read exactly once,
    through that index, on the two lines marked below. There is no code path in
    which a row outside ``train_idx`` influences a coefficient -- which is a
    property a test can check by poisoning the held-out rows and asserting the
    coefficients do not move, and by poisoning a TRAINING row and asserting
    they do.

    WHY THIS CONTROL IS NOT OPTIONAL. Gap length is a property of the
    OBSERVATION PROCESS, not of the vegetation: a long gap means the frames
    between were cloudy. It is also mechanically related to how much NDVI can
    change, because more days is more growing. So a delta probe can score well
    by having learned a calendar. P1's analogue -- [clear_frac,
    window_span_days], two numbers and no image -- decoded season at 0.646-0.658,
    at or above most foundation models, and P4 found it still absorbed a real
    share of the signal at 115 cubes. Every delta row in this table is reported
    against this control as ``margin_over_control``, and the raw Spearman is
    not the number to quote.
    """
    g = np.asarray(gap_days, dtype=float)
    y = np.asarray(values, dtype=float)
    train_idx = np.asarray(train_idx, dtype=int)
    assert g.ndim == 1 and y.shape == g.shape, (
        f"gap_days {g.shape} and values {y.shape} must be the same 1-D shape"
    )
    assert train_idx.ndim == 1 and train_idx.size, "empty training index set"
    assert train_idx.min() >= 0 and train_idx.max() < y.size, (
        f"train_idx reaches {train_idx.max()} but only {y.size} rows exist"
    )

    # --- THE ONLY READ OF THE ARRAYS. Everything below touches these two -----
    g_tr = g[train_idx]
    y_tr = y[train_idx]
    # ------------------------------------------------------------------------

    finite = np.isfinite(g_tr) & np.isfinite(y_tr)
    assert finite.any(), "no finite training row to fit the gap control on"
    g_tr, y_tr = g_tr[finite], y_tr[finite]
    n_par = int(degree) + 1
    n_distinct = int(np.unique(g_tr).size)
    assert n_distinct >= n_par, (
        f"the training fold covers only {n_distinct} distinct gap lengths but "
        f"a degree-{degree} control has {n_par} parameters. It would "
        "interpolate them exactly and the control would be noise; the gaps on "
        "this subset are a 5-day lattice, so a fold this thin means the fold "
        "mode is wrong, not the degree."
    )
    A = gap_design(g_tr, degree)
    coef, *_ = np.linalg.lstsq(A, y_tr, rcond=None)
    assert coef.shape == (n_par,) and np.isfinite(coef).all(), (
        f"the gap control fit returned {coef}; the design matrix is singular"
    )
    resid = y_tr - A @ coef
    sst = float(((y_tr - y_tr.mean()) ** 2).sum())
    train_r2 = float(1.0 - (resid ** 2).sum() / sst) if sst > 0 else 0.0
    ctrl = GapControl(coef=coef, degree=int(degree), n_train_rows=int(y_tr.size),
                      gap_train_min=float(g_tr.min()), gap_train_max=float(g_tr.max()),
                      train_r2=train_r2, label=label)
    if verbose:
        log(f"[p2]     gap control{' ' + label if label else ''}: fitted on "
            f"{ctrl.n_train_rows} TRAIN rows over gaps "
            f"{ctrl.gap_train_min:.0f}-{ctrl.gap_train_max:.0f} d, degree "
            f"{degree}, absorbs {train_r2:.3f} of train variance")
    return ctrl


# ---------------------------------------------------------------------------
# The ridge path, and the tuner. train_rows is REQUIRED and POSITIONAL.
# ---------------------------------------------------------------------------

def ridge_path(X_tr, y_tr, X_te, alphas) -> np.ndarray:
    """(len(alphas), n_test) ridge predictions, standardised on TRAIN.

    One factorisation, every penalty. The nested tuning loop fits the same
    training matrix at seven penalties inside every inner fold, and at LOCO
    that is 20 x 3 x 7 refits of a 4000 x 768 matrix per encoder per target.
    Forming the Gram matrix once and re-solving it per alpha is the same
    arithmetic at a fraction of the cost, and it is the same estimator:
    ``tests/test_p2_deltas.py`` asserts agreement with ``sklearn.linear_model.
    Ridge`` to 1e-8 on both the wide and the tall regime.

    Standardisation is fitted on TRAIN and applied to TEST, never the reverse.
    Zero-variance columns are given scale 1 rather than being dropped, so the
    column count is stable across folds and a coefficient vector means the same
    thing in each.
    """
    Xtr = np.asarray(X_tr, dtype=np.float64)
    Xte = np.asarray(X_te, dtype=np.float64)
    y = np.asarray(y_tr, dtype=np.float64)
    alphas = np.asarray(alphas, dtype=float)
    assert Xtr.ndim == 2 and Xte.ndim == 2 and Xtr.shape[1] == Xte.shape[1], (
        f"train {Xtr.shape} and test {Xte.shape} disagree on D"
    )
    assert y.shape == (Xtr.shape[0],), f"y {y.shape} vs X {Xtr.shape}"
    assert alphas.ndim == 1 and alphas.size and (alphas > 0).all()

    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    A = (Xtr - mu) / sd
    B = (Xte - mu) / sd
    y0 = y.mean()
    yc = y - y0
    n, D = A.shape

    out = np.empty((alphas.size, B.shape[0]), dtype=np.float64)
    if D <= n:
        G = A.T @ A                       # (D, D), formed once
        b = A.T @ yc
        for i, al in enumerate(alphas):
            coef = np.linalg.solve(G + al * np.eye(D), b)
            out[i] = B @ coef + y0
    else:
        # Wide regime (pooled DINOv2 is D=3840 against ~200 rows): the dual
        # form solves an n x n system instead of a D x D one and is exactly the
        # same estimator, since (A'A + aI)^-1 A' == A'(AA' + aI)^-1.
        K = A @ A.T                       # (n, n), formed once
        M = B @ A.T
        for i, al in enumerate(alphas):
            dual = np.linalg.solve(K + al * np.eye(n), yc)
            out[i] = M @ dual + y0
    assert out.shape == (alphas.size, Xte.shape[0])
    assert np.isfinite(out).all(), "the ridge path produced a non-finite prediction"
    return out


def spearman(a, b) -> float:
    """Spearman rank correlation, NaN when either side is constant.

    A constant side has no ranks to correlate, and scipy warns and returns NaN;
    that is the right answer and it is propagated rather than coerced to 0,
    which would read as "measured, and it was nothing".
    """
    from scipy.stats import spearmanr

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape and a.ndim == 1, f"{a.shape} vs {b.shape}"
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.unique(a[ok]).size < 2 or np.unique(b[ok]).size < 2:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(spearmanr(a[ok], b[ok]).statistic)


def _r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    sst = float(((y_true - y_true.mean()) ** 2).sum())
    if sst <= 0:
        return float("nan")
    return float(1.0 - ((y_true - y_pred) ** 2).sum() / sst)


def _score(metric: str, y_true, y_pred) -> float:
    assert metric in ("r2", "spearman"), metric
    return _r2(y_true, y_pred) if metric == "r2" else spearman(y_pred, y_true)


def select_ridge_alpha(block: FeatureBlock, manifest, train_rows, metric: str,
                       group_of_row=None, inner_k: int = _INNER_K,
                       alphas: Sequence[float] = RIDGE_ALPHA_GRID,
                       verbose: bool = False) -> dict:
    """Tune the ridge penalty on the TRAINING fold only.

    THE SIGNATURE IS THE GUARANTEE. ``train_rows`` is a REQUIRED POSITIONAL
    argument, so the penalty cannot be selected over everything by omitting it,
    and the block is sliced to the training rows INSIDE this function -- the
    full block goes in, the training index decides what is read. A test poisons
    the held-out rows and asserts the selection is bit-identical, and a
    companion poisons a TRAINING row and asserts it moves.

    Not ``p1_appearance.select_hyperparameter``: that one scores balanced
    accuracy over classes and P2's heads are regression (Part A) and rank
    (Part B). The grid, the inner generator, the tie rule and the
    fitted-on-train-only structure are the same, deliberately.

    The inner split is ``probes.cv.folds`` on the TRAINING sub-manifest, mode
    "cube": cube grouping is the strictest rule any outer mode here obeys, so
    an inner fold can never put a training cube on both sides either.

    ``group_of_row`` maps a block row index to the MANIFEST row that decides
    its fold (identity for Part A; ``pairs.row_a`` for Part B, where a block
    row is a pair). Passing it explicitly rather than inferring it is what
    keeps one tuner honest for both parts.
    """
    assert metric in ("r2", "spearman"), metric
    assert block.y is not None, "the block carries no labels"
    train_rows = np.asarray(train_rows, dtype=int)
    assert train_rows.ndim == 1 and train_rows.size, "empty training row set"
    grid = tuple(float(a) for a in alphas)

    manifest_row = (np.asarray(block.row_idx) if group_of_row is None
                    else np.asarray(group_of_row)[block.row_idx])
    assert manifest_row.shape == (block.n_rows,), manifest_row.shape

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

    scores = np.full((len(grid), len(inner)), np.nan)
    for j, (itr, ite) in enumerate(inner):
        tr_rows, te_rows = train_rows[itr], train_rows[ite]
        assert not np.intersect1d(tr_rows, te_rows).size
        a = np.flatnonzero(np.isin(manifest_row, tr_rows) & np.isfinite(block.y))
        b = np.flatnonzero(np.isin(manifest_row, te_rows) & np.isfinite(block.y))
        if a.size < 2 or b.size < 2:
            continue
        preds = ridge_path(block.X[a], block.y[a], block.X[b], grid)
        for i in range(len(grid)):
            scores[i, j] = _score(metric, block.y[b], preds[i])

    mean = np.nanmean(scores, axis=1)
    assert np.isfinite(mean).any(), (
        "every inner fold scored NaN; the inner split left no usable rows"
    )
    mean = np.where(np.isfinite(mean), mean, -np.inf)
    # Ties break toward STRONGER regularisation. The grid is ascending in
    # alpha and stronger means LARGER alpha, so take the LAST maximum. Of two
    # models the inner data cannot tell apart, the more regularised one is the
    # one that will generalise -- and the tie is broken by a stated rule
    # rather than by numpy's scan order.
    best = len(grid) - 1 - int(np.argmax(mean[::-1]))
    chosen = float(grid[best])
    if verbose:
        log(f"[p2]     inner {k}-fold on {n_cubes} training cubes: "
            f"alpha={chosen:g} (inner {metric} {mean[best]:+.4f})")
    return {"param": chosen, "inner_scores": mean, "grid": grid,
            "n_inner_folds": len(inner), "inner_best_score": float(mean[best]),
            "at_grid_edge": best in (0, len(grid) - 1)}


# ---------------------------------------------------------------------------
# One fold
# ---------------------------------------------------------------------------

class FoldResult(NamedTuple):
    """One outer fold. ``effective_n`` counts CUBES, never rows."""

    fold: int
    n_train: int
    n_test: int
    n_train_cubes: int
    n_test_cubes: int
    effective_n: int
    score: float           # r2 (Part A) or Spearman (Part B)
    spearman: float        # always present, so the two parts are comparable
    rmse: float
    selected: float
    at_grid_edge: bool
    n_common_px_median: float
    log: str


def _fold_cubes(manifest, rows) -> int:
    return int(np.unique(manifest["cube_id"].to_numpy()[rows]).size)


def _manifest_row_of(block: FeatureBlock, group_of_row) -> np.ndarray:
    """The MANIFEST row that decides each block row's fold.

    Identity for Part A, where ``row_idx`` already indexes the manifest;
    ``pairs.row_a`` for Part B, where ``row_idx`` indexes PAIRS. Every fold
    selection and the inner tuning split go through this one mapping, so the
    two parts cannot drift into indexing the manifest with pair positions --
    which is a mistake that changes no shape and returns a confident wrong
    number, exactly the hazard ``FeatureBlock`` exists to prevent one level up.
    """
    row_idx = np.asarray(block.row_idx)
    out = row_idx if group_of_row is None else np.asarray(group_of_row)[row_idx]
    assert out.shape == (block.n_rows,), (
        f"{block.name}: the manifest-row map is {out.shape}, expected "
        f"{(block.n_rows,)}"
    )
    return out


def _fit_and_score(block: FeatureBlock, manifest, train_rows, test_rows,
                   metric: str, group_of_row, fit: bool, fold: int,
                   inner_k: int, verbose: bool):
    """Tune on train, standardise on train, fit on train, predict test.

    ``train_rows`` / ``test_rows`` are always MANIFEST rows -- what
    ``probes.cv`` yields -- whichever part is calling.
    """
    manifest_row = _manifest_row_of(block, group_of_row)
    ok = np.isfinite(block.y)
    tr = np.flatnonzero(np.isin(manifest_row, train_rows) & ok)
    te = np.flatnonzero(np.isin(manifest_row, test_rows) & ok)
    assert tr.size and te.size, (
        f"fold {fold}: a side selected zero usable {block.name} rows "
        f"(train {tr.size}, test {te.size})"
    )
    if not fit:
        # The ``norm`` read-out is a FIXED STATISTIC, not a fitted model: it
        # uses no training data at all, so there is nothing to tune, nothing to
        # fit, and the held-out rank correlation of the statistic itself is the
        # score. Recorded with alpha = NaN rather than a made-up value.
        assert block.D == 1, (
            f"{block.name}: an unfitted read-out must be a single statistic, "
            f"got D={block.D}"
        )
        return tr, te, block.X[te, 0], block.y[te], float("nan"), False

    sel = select_ridge_alpha(block, manifest, train_rows, metric,
                             group_of_row=group_of_row, inner_k=inner_k,
                             verbose=verbose)
    pred = ridge_path(block.X[tr], block.y[tr], block.X[te],
                      np.array([sel["param"]]))[0]
    return tr, te, pred, block.y[te], sel["param"], sel["at_grid_edge"]


def evaluate_fold_a(block: FeatureBlock, manifest, train_rows, test_rows,
                    fold: int = 0, inner_k: int = _INNER_K,
                    verbose: bool = False) -> FoldResult:
    """Part A, one outer fold: tune on train, fit on train, score R2 on test."""
    train_rows = np.asarray(train_rows, dtype=int)
    test_rows = np.asarray(test_rows, dtype=int)
    assert not np.intersect1d(train_rows, test_rows).size, (
        f"fold {fold}: manifest rows appear on both sides"
    )
    tr, te, pred, y_te, alpha, edge = _fit_and_score(
        block, manifest, train_rows, test_rows, "r2", None, True, fold,
        inner_k, verbose)
    r2 = _r2(y_te, pred)
    if te.size < _MIN_TEST_ROWS:
        r2 = float("nan")
    n_tr_c = _fold_cubes(manifest, train_rows)
    n_te_c = _fold_cubes(manifest, test_rows)
    msg = (f"[p2]   A fold {fold + 1}: {block.name} train {tr.size} rows / "
           f"{n_tr_c} cubes, test {te.size} rows / {n_te_c} cubes "
           f"(effective n = {n_te_c} CUBES, not {te.size} rows) | "
           f"alpha {alpha:g} | R2 {r2:+.4f}")
    log(msg, echo=False)
    return FoldResult(fold=fold, n_train=int(tr.size), n_test=int(te.size),
                      n_train_cubes=n_tr_c, n_test_cubes=n_te_c,
                      effective_n=n_te_c, score=r2,
                      spearman=spearman(pred, y_te),
                      rmse=float(np.sqrt(((y_te - pred) ** 2).mean())),
                      selected=alpha, at_grid_edge=bool(edge),
                      n_common_px_median=float("nan"), log=msg)


def evaluate_fold_b(block: FeatureBlock, manifest, pairs: DeltaPairs,
                    train_rows, test_rows, readout: str,
                    n_common_px=None, fold: int = 0, inner_k: int = _INNER_K,
                    verbose: bool = False) -> FoldResult:
    """Part B, one outer fold: score Spearman on held-out PAIRS.

    A pair belongs to exactly one cube and every fold mode in ``probes.cv`` is
    cube-grouped, so a pair cannot straddle a fold. That is asserted here on
    every fold rather than argued once: the assertion is what would catch a
    future mode -- ``temporal``, say -- that splits inside a cube.
    """
    train_rows = np.asarray(train_rows, dtype=int)
    test_rows = np.asarray(test_rows, dtype=int)
    in_tr_a = np.isin(pairs.row_a, train_rows)
    in_tr_b = np.isin(pairs.row_b, train_rows)
    in_te_a = np.isin(pairs.row_a, test_rows)
    in_te_b = np.isin(pairs.row_b, test_rows)
    assert (in_tr_a == in_tr_b).all() and (in_te_a == in_te_b).all(), (
        f"fold {fold}: {int((in_tr_a != in_tr_b).sum())} pair(s) have their two "
        "frames on opposite sides of the split. Both frames of a pair are in "
        "one cube, so this cannot happen under a cube-grouped mode -- a mode "
        "that splits inside a cube would leak the pair's own second frame "
        "into training."
    )
    pair_tr = np.flatnonzero(in_tr_a)
    pair_te = np.flatnonzero(in_te_a)
    assert pair_tr.size and pair_te.size, (
        f"fold {fold}: a side selected zero pairs (train {pair_tr.size}, "
        f"test {pair_te.size})"
    )
    # ``block.row_idx`` indexes PAIRS here, so the fold selection and the inner
    # tuning split are routed through pairs.row_a -- the manifest row of each
    # pair's first frame. Handing pair positions to a function that indexes the
    # manifest would change no shape and return a confident wrong number.
    tr, te, pred, y_te, alpha, edge = _fit_and_score(
        block, manifest, train_rows, test_rows, "spearman",
        group_of_row=pairs.row_a, fit=(readout == "linear"), fold=fold,
        inner_k=inner_k, verbose=verbose)
    rho = spearman(pred, y_te)
    if te.size < _MIN_TEST_ROWS:
        rho = float("nan")
    n_tr_c = int(np.unique(pairs.cube_id[pair_tr]).size)
    n_te_c = int(np.unique(pairs.cube_id[pair_te]).size)
    px = (float(np.median(np.asarray(n_common_px)[pair_te]))
          if n_common_px is not None else float("nan"))
    msg = (f"[p2]   B fold {fold + 1}: {block.name} train {tr.size} rows / "
           f"{n_tr_c} cubes, test {te.size} rows / {n_te_c} cubes "
           f"(effective n = {n_te_c} CUBES, not {te.size} rows) | "
           f"alpha {alpha:g} | spearman {rho:+.4f} | common px median {px:.0f}")
    log(msg, echo=False)
    return FoldResult(fold=fold, n_train=int(tr.size), n_test=int(te.size),
                      n_train_cubes=n_tr_c, n_test_cubes=n_te_c,
                      effective_n=n_te_c, score=rho, spearman=rho,
                      rmse=float(np.sqrt(((y_te - pred) ** 2).mean())),
                      selected=alpha, at_grid_edge=bool(edge),
                      n_common_px_median=px, log=msg)


def evaluate(block: FeatureBlock, manifest, mode: str, part: str,
             pairs: DeltaPairs | None = None, readout: str = "linear",
             n_common_px=None, k: int = 5, inner_k: int = _INNER_K,
             n_jobs: int = 1, verbose: bool = False) -> list:
    """Every outer fold of one mode, from ``probes.cv`` and nowhere else.

    Deterministic; ``n_jobs`` only changes wall-clock, never a number -- the
    ridge path is an exact linear solve, the standardiser is exact, and the
    folds come from a generator with no RNG in it. Same guarantee, and the same
    implementation, as ``p1_appearance.evaluate``.

    It matters more here than there. Leave-one-cube-out is ``k = n_cubes``, so
    scaling the subset scales the FOLD COUNT as well as the row count: at 20
    cubes LOCO is 20 folds over 264 rows, at 115 it is 115 folds over 1580.
    That is 5.75x the folds each costing ~6x as much, and it is why the serial
    run of the scaled table takes hours while the 20-cube one took nine
    minutes.

    Arrays are passed to the workers as ARGUMENTS rather than captured in a
    closure, so joblib memory-maps the large ones instead of pickling a design
    matrix once per fold. ``inner_max_num_threads=1`` stops each worker's BLAS
    from also going wide, which on an 8-core box otherwise oversubscribes into
    being slower than serial.
    """
    assert part in PARTS, f"part {part!r} not in {PARTS}"
    folds = outer_folds(manifest, mode, k=k, verbose=False)

    if part == "A_reconstruction":
        fn, extra = evaluate_fold_a, dict(inner_k=inner_k, verbose=False)
        args = lambda tr, te: (block, manifest, tr, te)          # noqa: E731
    else:
        assert pairs is not None, "Part B needs the pair index"
        fn = evaluate_fold_b
        extra = dict(n_common_px=n_common_px, inner_k=inner_k, verbose=False)
        args = lambda tr, te: (block, manifest, pairs, tr, te, readout)  # noqa: E731

    if n_jobs == 1 or len(folds) < 2:
        out = [fn(*args(tr, te), fold=i, **extra)
               for i, (tr, te) in enumerate(folds)]
    else:
        from joblib import Parallel, delayed, parallel_config
        with parallel_config(backend="loky", inner_max_num_threads=1):
            out = Parallel(n_jobs=n_jobs)(
                delayed(fn)(*args(tr, te), fold=i, **extra)
                for i, (tr, te) in enumerate(folds))
        # A loky worker re-imports this module, so its ``_LOG_FH`` is None and
        # the per-fold lines it wrote went nowhere. Every FoldResult carries its
        # own line precisely so the PARENT can write them, which keeps the run
        # log identical between serial and parallel rather than quietly thinner
        # in the mode the scaled run actually uses.
        for r in out:
            log(r.log)
    assert out, f"{mode}: no folds"
    if verbose:
        for r in out:
            print(r.log)
    return out


def summarise(results: Sequence[FoldResult]) -> dict:
    """Mean, spread and a CUBE-CLUSTERED CI. Never a bare mean.

    Folds hold disjoint sets of cubes, so the per-fold scores are the
    independent replicates -- not the 244 pairs or the 4224 cells, which share
    skies and places. ``fold_clustered_ci`` is imported from
    ``p4_ceiling``; the interval is wide, and at 20 cubes that width IS the
    result. ``effective_n`` is carried here so it can never be separated from
    the score it qualifies.
    """
    assert results, "nothing to summarise"
    out = {"n_folds": len(results)}
    for field in ("score", "spearman", "rmse"):
        v = np.array([getattr(r, field) for r in results], dtype=float)
        lo, hi = fold_clustered_ci(v)
        ok = v[np.isfinite(v)]
        nan = float("nan")
        out[f"{field}_mean"] = float(ok.mean()) if ok.size else nan
        out[f"{field}_std"] = float(ok.std(ddof=1)) if ok.size > 1 else 0.0
        out[f"{field}_min"] = float(ok.min()) if ok.size else nan
        out[f"{field}_max"] = float(ok.max()) if ok.size else nan
        out[f"{field}_ci_lo"] = lo
        out[f"{field}_ci_hi"] = hi
        out[f"per_fold_{field}"] = ";".join(f"{x:.4f}" for x in v)
    out["n_folds_nan"] = int(sum(1 for r in results if not np.isfinite(r.score)))
    out["effective_n"] = int(sum(r.effective_n for r in results))
    out["effective_n_per_fold"] = ";".join(str(r.effective_n) for r in results)
    out["n_rows_test_total"] = int(sum(r.n_test for r in results))
    out["selected_per_fold"] = ";".join(f"{r.selected:g}" for r in results)
    out["n_at_grid_edge"] = int(sum(r.at_grid_edge for r in results))
    px = np.array([r.n_common_px_median for r in results], dtype=float)
    out["n_common_px_median"] = (float(np.median(px[np.isfinite(px)]))
                                 if np.isfinite(px).any() else float("nan"))
    return out


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def _encoder_views(encoders: Sequence[str]) -> list:
    """(encoder, feature_set) pairs. The band-matched views exist only for the
    not-a-network baseline, because they are column slices of ITS array."""
    views = [(e, "embedding") for e in encoders]
    if BASELINE_ENCODER in encoders:
        views += [(BASELINE_ENCODER, "raw_rgb_only"),
                  (BASELINE_ENCODER, "raw_nir_ndvi")]
    return views


def _base_row(part, aggregation, delta_target, level, mode, encoder,
              feature_set, model_kind, readout, D, n_manifest, n_cubes,
              n_pairs) -> dict:
    return {
        "part": part,
        "aggregation": aggregation,
        "target_level": TARGET_LEVEL[aggregation],
        "delta_target": delta_target,
        "target": (aggregation if part == "A_reconstruction"
                   else f"{aggregation}_{delta_target}"),
        "feature_level": level,
        "fold_mode": mode,
        "encoder": encoder,
        "feature_set": feature_set,
        "model_kind": model_kind,
        "readout": readout,
        "is_control": model_kind in CONTROL_KINDS,
        # The multi-image encoder's consecutive embeddings share up to 7 of 8
        # source frames, so its DELTA is a sliding-window increment and not a
        # state change. Flagged on every row it owns, in both parts, so a
        # filtered view can never lose the caveat.
        "si_comparable": encoder != MI_ENCODER,
        "metric": "r2" if part == "A_reconstruction" else "spearman",
        "D": int(D),
        "n_manifest_rows": int(n_manifest),
        "n_cubes": int(n_cubes),
        "n_pairs": int(n_pairs),
        "common_masked": part == "B_delta",
        "gap_source": "daily_axis_index" if part == "B_delta" else "",
    }


def run_p2(manifest, cube_dir: str, encoders: Sequence[str] = ENCODER_ORDER,
           fold_modes: Sequence[str] = FOLD_MODES, k: int = 5,
           inner_k: int = _INNER_K, emb_dir: str | None = None,
           mask_dir: str | None = None, log_path: str | None = None,
           n_jobs: int = 1, verbose: bool = True,
           plausibility_screen: bool = False):
    """The whole table: Part A (gate K2) and Part B (the delta probe).

    Returns a pandas DataFrame and writes nothing but the run log. Every fold
    comes from ``probes.cv``; every interval is cube-clustered; every control
    is emitted under every label it is invariant to.

    ``n_jobs`` parallelises OVER FOLDS and changes wall-clock only -- see
    ``evaluate``. At 20 cubes serial is fine (nine minutes); at 115 the fold
    count grows with the cube count under LOCO and serial is hours.

    ``plausibility_screen`` defaults off for published-table reproducibility.
    When enabled, Part A excludes bad target frames and Part B excludes pairs
    touching either bad endpoint; the manifest and embedding cache stay whole.
    """
    import pandas as pd

    open_run_log(log_path, verbose=verbose)
    try:
        rows = _run_p2_rows(manifest, cube_dir, encoders, fold_modes, k,
                            inner_k, emb_dir, mask_dir, n_jobs, verbose,
                            plausibility_screen)
    finally:
        close_run_log()
    df = pd.DataFrame(rows)
    assert len(df), "run_p2 produced no rows"
    assert_plausibility_screen_declared(
        df, required=bool(plausibility_screen))
    return df


def _run_p2_rows(manifest, cube_dir, encoders, fold_modes, k, inner_k,
                 emb_dir, mask_dir, n_jobs, verbose,
                 plausibility_screen) -> list:
    n = len(manifest)
    n_cubes = int(manifest["cube_id"].nunique())

    if verbose:
        print(f"[p2] manifest {manifest.shape} | {n_cubes} cubes | "
              f"tiles {sorted(manifest.tile.unique())} | "
              f"years {sorted(manifest.year.unique())}")

    # --- inputs ------------------------------------------------------------
    arrays = encoder_arrays(manifest, encoders, emb_dir=emb_dir, verbose=verbose)
    y_frame = frame_targets(manifest, cube_dir, verbose=verbose)
    frame_ok = np.asarray(y_frame["frame_plausible"], dtype=bool)
    n_implausible = int((~frame_ok).sum())
    pairs_all = pair_index(manifest, verbose=verbose)
    assert_gap_axes_disagree(pairs_all, verbose=verbose)
    if plausibility_screen:
        pairs, n_pairs_dropped = filter_pairs_by_plausibility(
            pairs_all, frame_ok, verbose=verbose)
    else:
        pairs, n_pairs_dropped = pairs_all, 0
        if verbose:
            print("[p2] plausibility screen OFF: verdicts measured, no frame "
                  "or pair removed")
    y_pair = build_pair_targets(pairs, manifest, cube_dir, mask_dir=mask_dir,
                                verbose=verbose)
    summarise_pixel_survival(pairs, y_pair, verbose=verbose)

    views = _encoder_views(encoders)
    rows = []
    screen_meta = {
        "plausibility_screen": bool(plausibility_screen),
        "plausibility_screen_def": (
            P2_PLAUSIBILITY_SCREEN_LABEL
            if plausibility_screen else "not_applied"),
        "n_implausible_frames": n_implausible,
        "n_pairs_dropped_implausible": n_pairs_dropped,
    }

    # --- PART A: the reconstruction floor ----------------------------------
    if verbose:
        print("\n[p2] " + "=" * 68)
        print("[p2] PART A -- reconstruction floor (gate K2): E(frame_t) -> NDVI(t)")
        print("[p2] " + "=" * 68)
    for agg, level in PART_A_COMBOS:
        # cell_mean is the only Part A target whose individual rows can be
        # invalid -- a frame at 0.6 clear can hold a fully-masked cell. Those
        # rows are dropped, never filled; the cube-level targets are finite on
        # every retained frame by the clear-fraction rule.
        y_rows = y_frame[agg]
        keep_rows = y_frame["cell_valid"] if agg == "cell_mean" else None
        if plausibility_screen:
            y_rows, keep_rows = screen_part_a_target(
                agg, y_rows, keep_rows, frame_ok)
        for mode in fold_modes:
            head = f"{agg} | {level} | {mode}"
            if verbose:
                print(f"\n[p2] ---- A {head} " + "-" * max(0, 50 - len(head)))
            for enc, fs in views:
                block = reconstruction_block(arrays[enc], level, fs)
                block = _attach(block, agg, level, y_rows, keep_rows)
                res = evaluate(block, manifest, mode, "A_reconstruction",
                               k=k, inner_k=inner_k, n_jobs=n_jobs)
                rows.append({**_base_row("A_reconstruction", agg, "", level,
                                         mode, enc, fs, "reconstruction",
                                         "ridge", block.D, n, n_cubes,
                                         pairs.n_pairs),
                             **screen_meta,
                             **summarise(res)})
                if verbose:
                    s = rows[-1]
                    print(f"[p2]   {enc:<24} {fs:<13} D={block.D:<5} "
                          f"R2 {s['score_mean']:+.3f} "
                          f"[{s['score_ci_lo']:+.3f}, {s['score_ci_hi']:+.3f}] "
                          f"| effective n {s['effective_n']} CUBES")
            # The retention-only control, emitted once per (agg, level, mode).
            ctrl = retention_block(arrays, level)
            ctrl = _attach(ctrl, agg, level, y_rows, keep_rows)
            res = evaluate(ctrl, manifest, mode, "A_reconstruction",
                           k=k, inner_k=inner_k, n_jobs=n_jobs)
            rows.append({**_base_row("A_reconstruction", agg, "", level, mode,
                                     "none", "retention", "retention", "ridge",
                                     ctrl.D, n, n_cubes, pairs.n_pairs),
                         **screen_meta,
                         **summarise(res)})
            if verbose:
                s = rows[-1]
                print(f"[p2]   {'none':<24} {'retention':<13} D={ctrl.D:<5} "
                      f"R2 {s['score_mean']:+.3f} "
                      f"[{s['score_ci_lo']:+.3f}, {s['score_ci_hi']:+.3f}] "
                      "   <- CONTROL (clear_frac + window_span_days, no image)")

    # --- PART B: the delta probe -------------------------------------------
    if verbose:
        print("\n[p2] " + "=" * 68)
        print("[p2] PART B -- delta probe: E(t+1)-E(t) -> common-masked NDVI change")
        print("[p2] " + "=" * 68)
    for agg, level in PART_B_COMBOS:
        raw = y_pair[agg]
        ok = y_pair["cell_valid"] if agg == "cell_mean" else None
        for dt in DELTA_TARGETS:
            y_p = np.abs(raw) if dt == "magnitude" else np.sign(raw)
            for mode in fold_modes:
                head = f"{agg} {dt} | {level} | {mode}"
                if verbose:
                    print(f"\n[p2] ---- B {head} " + "-" * max(0, 50 - len(head)))
                for readout in READOUTS:
                    for enc, fs in views:
                        block = delta_block(arrays[enc], pairs, level, readout, fs)
                        block = _attach_pair(block, agg, level, y_p, ok)
                        res = evaluate(block, manifest, mode, "B_delta",
                                       pairs=pairs, readout=readout,
                                       n_common_px=y_pair["n_common_px"],
                                       k=k, inner_k=inner_k, n_jobs=n_jobs)
                        rows.append({**_base_row("B_delta", agg, dt, level, mode,
                                                 enc, fs, "delta", readout,
                                                 block.D, n, n_cubes,
                                                 pairs.n_pairs),
                                     **screen_meta,
                                     **summarise(res)})
                        if verbose:
                            s = rows[-1]
                            flag = ("" if s["si_comparable"]
                                    else "  [si_comparable=False]")
                            print(f"[p2]   {readout:<7} {enc:<24} {fs:<13} "
                                  f"D={block.D:<5} rho {s['score_mean']:+.3f} "
                                  f"[{s['score_ci_lo']:+.3f}, {s['score_ci_hi']:+.3f}]"
                                  f"{flag}")
                    # The gap-length-alone control. It holds no embedding, so
                    # it is invariant to encoder / feature_level / readout and
                    # is emitted under each of those labels with the SAME
                    # numbers -- asserted by assert_control_identical_across_views.
                    res = _gap_control_folds(manifest, pairs, y_p, ok, agg,
                                             level, mode, k, y_pair["n_common_px"])
                    rows.append({**_base_row("B_delta", agg, dt, level, mode,
                                             "none", "gap_days", "gap_only",
                                             readout, GAP_CONTROL_DEGREE + 1,
                                             n, n_cubes, pairs.n_pairs),
                                 **screen_meta,
                                 **summarise(res)})
                    if verbose:
                        s = rows[-1]
                        print(f"[p2]   {readout:<7} {'none':<24} {'gap_days':<13} "
                              f"D={GAP_CONTROL_DEGREE + 1:<5} "
                              f"rho {s['score_mean']:+.3f} "
                              f"[{s['score_ci_lo']:+.3f}, {s['score_ci_hi']:+.3f}]"
                              "   <- CONTROL (gap length alone, no embedding)")
    return rows


def _attach(block: FeatureBlock, agg: str, level: str, y_rows, keep_rows):
    """Attach Part A labels to a block, NaN-ing rows the target cannot cover."""
    if agg == "cell_mean":
        assert level == "grid_cell", "cell_mean is defined at grid_cell level only"
        y = np.asarray(y_rows, dtype=float).reshape(-1)
        valid = np.asarray(keep_rows, dtype=bool).reshape(-1)
        y = np.where(valid, y, np.nan)
        assert y.shape == (block.n_rows,), (
            f"{block.name}: {y.shape} labels for {block.n_rows} feature rows"
        )
        return FeatureBlock(X=block.X, row_idx=block.row_idx, y=y, name=block.name)
    return block.with_labels(np.asarray(y_rows, dtype=float))


def _attach_pair(block: FeatureBlock, agg: str, level: str, y_pair, keep):
    """Attach Part B labels. ``block.row_idx`` indexes PAIRS, not manifest rows."""
    if agg == "cell_mean":
        assert level == "grid_cell", "cell_mean is defined at grid_cell level only"
        y = np.asarray(y_pair, dtype=float).reshape(-1)
        y = np.where(np.asarray(keep, dtype=bool).reshape(-1), y, np.nan)
        assert y.shape == (block.n_rows,), (
            f"{block.name}: {y.shape} labels for {block.n_rows} feature rows"
        )
        return FeatureBlock(X=block.X, row_idx=block.row_idx, y=y, name=block.name)
    return block.with_labels(np.asarray(y_pair, dtype=float))


def _gap_control_folds(manifest, pairs: DeltaPairs, y_p, cell_ok, agg, level,
                       mode, k, n_common_px) -> list:
    """The gap control, fitted INSIDE each fold, scored on held-out pairs."""
    if agg == "cell_mean":
        gap = np.repeat(pairs.gap_days, GRID_CELLS)
        y = np.where(np.asarray(cell_ok, dtype=bool).reshape(-1),
                     np.asarray(y_p, dtype=float).reshape(-1), np.nan)
        pair_of = np.repeat(np.arange(pairs.n_pairs), GRID_CELLS)
    else:
        gap = pairs.gap_days
        y = np.asarray(y_p, dtype=float)
        pair_of = np.arange(pairs.n_pairs)

    out = []
    for i, (tr_rows, te_rows) in enumerate(outer_folds(manifest, mode, k=k,
                                                       verbose=False)):
        in_tr = np.isin(pairs.row_a, tr_rows)[pair_of] & np.isfinite(y)
        in_te = np.isin(pairs.row_a, te_rows)[pair_of] & np.isfinite(y)
        tr, te = np.flatnonzero(in_tr), np.flatnonzero(in_te)
        assert tr.size and te.size, f"gap control fold {i}: an empty side"
        ctrl = fit_gap_control(gap, y, tr, label=f"{agg} {mode} fold {i + 1}",
                               verbose=True)
        pred = ctrl.predict(gap[te])
        rho = spearman(pred, y[te])
        if te.size < _MIN_TEST_ROWS:
            rho = float("nan")
        n_te_c = int(np.unique(pairs.cube_id[pair_of[te]]).size)
        px = float(np.median(np.asarray(n_common_px)[pair_of[te]]))
        msg = (f"[p2]   B fold {i + 1}: gap_only/{agg}/{mode} train {tr.size} "
               f"rows, test {te.size} rows / {n_te_c} cubes | spearman {rho:+.4f}")
        log(msg)
        out.append(FoldResult(
            fold=i, n_train=int(tr.size), n_test=int(te.size),
            n_train_cubes=int(np.unique(pairs.cube_id[pair_of[tr]]).size),
            n_test_cubes=n_te_c, effective_n=n_te_c, score=rho, spearman=rho,
            rmse=float(np.sqrt(((y[te] - pred) ** 2).mean())),
            selected=float("nan"), at_grid_edge=False,
            n_common_px_median=px, log=msg))
    return out


# ---------------------------------------------------------------------------
# Margins, verdicts, and the assertions the exit test names
# ---------------------------------------------------------------------------

def add_margins(df, verbose: bool = True):
    """Attach the margins P2 is actually read on.

    ``margin_over_control`` -- the score minus the matching CONTROL's, for the
    same (part, target, fold_mode, feature_level, readout): the gap-length-alone
    control in Part B, the retention-only control in Part A. **This is the
    headline**, reported exactly as P4 reports ``margin_over_control``. A delta
    score of +0.27 against a gap control of +0.21 is a margin of +0.06, and the
    margin is the part that needed an embedding.

    ``margin_over_raw`` -- minus the ``raw_features`` row of the same
    configuration: the raw-pixel comparison, without which no delta score has a
    scale.

    ``margin_over_band_matched`` -- minus ``raw_rgb_only``, the hand-crafted
    baseline over the SAME three bands every network receives. This is the
    like-for-like number; the margin over full ``raw_features`` is not, because
    that baseline additionally sees B8A and the NDVI columns.

    ``control_score`` -- the control's own value, written onto EVERY row. This
    is what makes "filtering the CSV cannot drop the control" true rather than
    merely intended: whatever a reader filters to -- one encoder, one fold
    mode, one feature level, one read-out -- the control travels on the row
    they are looking at, and ``assert_control_identical_across_views`` asserts
    every copy of it is digit-for-digit identical.
    """
    key = ["part", "target", "fold_mode", "feature_level", "readout"]
    df = df.copy()

    ctrl = df[df.is_control]
    assert len(ctrl) == len(ctrl.drop_duplicates(key)), (
        f"the control is not unique per {key}"
    )
    look = ctrl.set_index(key)["score_mean"]
    df["control_score"] = df.set_index(key).index.map(look).to_numpy()
    df["control_kind"] = df.set_index(key).index.map(
        ctrl.set_index(key)["model_kind"]).to_numpy()
    df["margin_over_control"] = df.score_mean - df.control_score
    assert df.margin_over_control.notna().all(), (
        "some row has no matching control. The controls are not optional and "
        "every row must be comparable to them."
    )
    for fs, col in ((("embedding", BASELINE_ENCODER), "margin_over_raw"),
                    ((BAND_MATCHED_BASELINE, BASELINE_ENCODER),
                     "margin_over_band_matched")):
        name, enc = fs
        ref = df[(df.feature_set == name) & (df.encoder == enc)]
        assert len(ref), f"no {enc}/{name} rows to form {col} against"
        assert len(ref) == len(ref.drop_duplicates(key)), (
            f"{enc}/{name} is not unique per {key}"
        )
        r = ref.set_index(key)["score_mean"]
        df[col] = df.score_mean - df.set_index(key).index.map(r).to_numpy()
        assert df[col].notna().all(), f"some row has no matching {enc}/{name} row"
    if verbose:
        b = df[(df.part == "B_delta") & (~df.is_control)
               & (df.feature_set == "embedding")]
        beaten = b[b.margin_over_control <= 0]
        print(f"[p2] margins attached. {len(beaten)}/{len(b)} delta rows are AT "
              "OR BELOW the gap-length-alone control -- i.e. the embedding "
              "change adds nothing there that the calendar did not already carry")
    return df


def add_k2_verdicts(df, verbose: bool = True):
    """Record a K2 verdict per encoder, on cube-clustered folds.

    TWO verdicts, for the reason set out at length in the module docstring:
    ``raw_features`` contains ``NDVI_mean``..``NDVI_p90`` and therefore IS the
    target at the matched level, so ``k2_verdict`` measures the baseline.
    ``k2_verdict_band_matched``, against the RGB-only slice of the same
    baseline, is the one that decides P3 inclusion.

    Both are read off ``K2_PRIMARY`` -- the primary aggregation, the primary
    feature level, cube-grouped folds -- named as a constant so the verdict
    cannot be taken from whichever row happened to look best, and both are
    written onto EVERY row of the table (Part B included) so a delta score can
    never be read without knowing whether its encoder cleared the floor.

    AND A THIRD COLUMN, because a binary verdict at this sample size is not
    self-interpreting. ``k2_separable`` is the PAIRED comparison: encoder minus
    baseline WITHIN each fold, then a cube-clustered interval on that
    difference. The two encoders are scored on identical folds, so pairing is
    both legitimate and far tighter than eyeballing two overlapping marginal
    intervals -- and it is the difference that the verdict is about. Where the
    paired interval spans zero, "audited: lossy" means "did not beat the
    baseline", not "is measurably worse", and P3 must not drop an encoder on
    it. Measured here: DINOv2 sits 0.006 below the baseline with marginal
    intervals 0.6 wide. Excluding it on that number would be excluding on
    noise, and the column is what makes that visible instead of arguable.
    """
    df = df.copy()
    sel = df[(df.part == "A_reconstruction")
             & (df.aggregation == K2_PRIMARY["aggregation"])
             & (df.feature_level == K2_PRIMARY["feature_level"])
             & (df.fold_mode == K2_PRIMARY["fold_mode"])]
    assert len(sel), f"no Part A rows at {K2_PRIMARY}; K2 cannot be decided"

    def _row_of(enc, fs):
        r = sel[(sel.encoder == enc) & (sel.feature_set == fs)]
        assert len(r) == 1, f"expected 1 row for ({enc}, {fs}), got {len(r)}"
        return r.iloc[0]

    def _score_of(enc, fs):
        return float(_row_of(enc, fs).score_mean)

    def _folds_of(enc, fs):
        return np.array([float(x) for x in
                         str(_row_of(enc, fs).per_fold_score).split(";")])

    raw = _score_of(BASELINE_ENCODER, "embedding")
    band = _score_of(BASELINE_ENCODER, BAND_MATCHED_BASELINE)
    raw_folds = _folds_of(BASELINE_ENCODER, "embedding")

    verdict, verdict_bm, m_raw, m_band = {}, {}, {}, {}
    ci_lo, ci_hi, separable = {}, {}, {}
    for enc in sel.encoder.unique():
        if enc == "none":
            continue
        s = _score_of(enc, "embedding")
        m_raw[enc] = s - raw
        m_band[enc] = s - band
        # The PAIRED difference, fold by fold, on folds both models shared.
        d = _folds_of(enc, "embedding") - raw_folds
        lo, hi = fold_clustered_ci(d)
        ci_lo[enc], ci_hi[enc] = lo, hi
        separable[enc] = bool(np.isfinite(lo) and np.isfinite(hi)
                              and (lo > 0 or hi < 0))
        if enc == BASELINE_ENCODER:
            verdict[enc] = verdict_bm[enc] = "baseline"
            separable[enc] = False
        else:
            verdict[enc] = "passed" if s > raw else "audited: lossy"
            verdict_bm[enc] = "passed" if s > band else "audited: lossy"

    df["k2_verdict"] = df.encoder.map(verdict).fillna("n/a (control)")
    df["k2_verdict_band_matched"] = df.encoder.map(verdict_bm).fillna("n/a (control)")
    df["k2_margin_over_raw"] = df.encoder.map(m_raw)
    df["k2_margin_over_band_matched"] = df.encoder.map(m_band)
    df["k2_margin_ci_lo"] = df.encoder.map(ci_lo)
    df["k2_margin_ci_hi"] = df.encoder.map(ci_hi)
    df["k2_separable"] = df.encoder.map(separable)
    df["k2_reference_config"] = (f"{K2_PRIMARY['aggregation']}/"
                                 f"{K2_PRIMARY['feature_level']}/"
                                 f"{K2_PRIMARY['fold_mode']}")
    df["k2_raw_features_score"] = raw
    df["k2_band_matched_score"] = band
    if verbose:
        print_k2_verdicts(df)
    return df


def print_k2_verdicts(df) -> None:
    """The K2 table, on stdout. Verdicts are a result, not a diagnostic."""
    sel = df[(df.part == "A_reconstruction")
             & (df.aggregation == K2_PRIMARY["aggregation"])
             & (df.feature_level == K2_PRIMARY["feature_level"])
             & (df.fold_mode == K2_PRIMARY["fold_mode"])
             & (df.feature_set == "embedding")].sort_values("score_mean",
                                                            ascending=False)
    raw = float(df.k2_raw_features_score.iloc[0])
    band = float(df.k2_band_matched_score.iloc[0])
    print("\n[p2] " + "=" * 68)
    print(f"[p2] GATE K2 -- reconstruction floor at {df.k2_reference_config.iloc[0]}, "
          "cube-clustered")
    print(f"[p2]   raw_features   R2 {raw:+.3f}  <- the SPECIFIED baseline. It "
          "holds NDVI_mean..NDVI_p90,")
    print("[p2]                              so at the matched level it IS the "
          "target. See the docstring.")
    print(f"[p2]   raw_rgb_only   R2 {band:+.3f}  <- BAND-MATCHED (B02/B03/B04 "
          "only). The fair floor,")
    print("[p2]                              and the verdict that decides P3 "
          "inclusion.")
    print(f"[p2]   {'encoder':<24} {'R2':>8} {'CI':>18}  {'vs raw':>8} "
          f"{'paired CI on vs-raw':>22}  {'vs band':>8}  verdict / verdict(band)")
    for r in sel.itertuples():
        sep = "SEPARABLE" if r.k2_separable else "not separable"
        print(f"[p2]   {r.encoder:<24} {r.score_mean:>+8.3f} "
              f"[{r.score_ci_lo:>+7.3f},{r.score_ci_hi:>+7.3f}] "
              f"{r.k2_margin_over_raw:>+8.3f} "
              f"[{r.k2_margin_ci_lo:>+7.3f},{r.k2_margin_ci_hi:>+7.3f}] "
              f"{r.k2_margin_over_band_matched:>+8.3f}  "
              f"{r.k2_verdict} / {r.k2_verdict_band_matched}  ({sep})"
              f"{'' if r.si_comparable else '  [si_comparable=False]'}")
    lossy = sorted(set(sel[sel.k2_verdict_band_matched == "audited: lossy"].encoder))
    firm = sorted(set(sel[(sel.k2_verdict_band_matched == "audited: lossy")
                          & sel.k2_separable].encoder))
    print(f"[p2]   audited: lossy (band-matched): {lossy if lossy else 'none'}")
    print(f"[p2]   EXCLUDED FROM P3 -- lossy AND separable from the baseline on "
          f"the paired\n[p2]   per-fold difference: {firm if firm else 'none'}")
    soft = sorted(set(lossy) - set(firm))
    if soft:
        print(f"[p2]   NOT excluded, verdict not separable at n=20 cubes: {soft}")
        print("[p2]   -- these did not BEAT the baseline; they are not measurably "
              "worse than it.\n[p2]      Dropping them from P3 would be dropping "
              "on noise. Re-measure at scale.")
    print("[p2] " + "=" * 68)


def print_controls(df) -> None:
    """The four control values, on stdout. None of them is optional."""
    print("\n[p2] THE FOUR CONTROLS "
          f"(primary view: {K2_PRIMARY['aggregation']}/"
          f"{K2_PRIMARY['feature_level']}/{K2_PRIMARY['fold_mode']})")
    a = df[(df.part == "A_reconstruction")
           & (df.aggregation == K2_PRIMARY["aggregation"])
           & (df.feature_level == K2_PRIMARY["feature_level"])
           & (df.fold_mode == K2_PRIMARY["fold_mode"])]
    for label, sub in (
            ("1 retention   [clear_frac, window_span_days] -> NDVI(t), R2",
             a[a.model_kind == "retention"]),
            ("2 raw_features  E(t) -> NDVI(t)  (holds NDVI columns), R2",
             a[(a.encoder == BASELINE_ENCODER) & (a.feature_set == "embedding")]),
            ("3 raw_rgb_only  E(t) -> NDVI(t)  (band-matched), R2",
             a[a.feature_set == BAND_MATCHED_BASELINE])):
        assert len(sub) == 1, f"{label}: expected 1 row, got {len(sub)}"
        r = sub.iloc[0]
        print(f"[p2]   {label:<62} {r.score_mean:+.3f} "
              f"[{r.score_ci_lo:+.3f}, {r.score_ci_hi:+.3f}]")
    b = df[(df.part == "B_delta") & (df.model_kind == "gap_only")
           & (df.aggregation == STRUCTURAL_PRIMARY["aggregation"])
           & (df.feature_level == STRUCTURAL_PRIMARY["feature_level"])
           & (df.readout == STRUCTURAL_PRIMARY["readout"])
           & (df.fold_mode == STRUCTURAL_PRIMARY["fold_mode"])]
    for dt in DELTA_TARGETS:
        r = b[b.delta_target == dt]
        assert len(r) == 1, f"gap_only/{dt}: expected 1 row, got {len(r)}"
        r = r.iloc[0]
        print(f"[p2]   {'4 gap_only    gap_days -> ' + dt + ', Spearman':<62} "
              f"{r.score_mean:+.3f} [{r.score_ci_lo:+.3f}, {r.score_ci_hi:+.3f}]")


def print_headlines(df) -> None:
    """The delta probe's headline correlations, with cube-clustered CIs."""
    p = STRUCTURAL_PRIMARY
    print(f"\n[p2] HEADLINE -- Part B delta probe, {p['aggregation']} / "
          f"{p['feature_level']} / {p['readout']} read-out / {p['fold_mode']} folds")
    for dt in DELTA_TARGETS:
        sub = df[(df.part == "B_delta") & (df.delta_target == dt)
                 & (df.aggregation == p["aggregation"])
                 & (df.feature_level == p["feature_level"])
                 & (df.readout == p["readout"])
                 & (df.fold_mode == p["fold_mode"])
                 & (df.feature_set.isin(("embedding", BAND_MATCHED_BASELINE))
                    | df.is_control)].sort_values("score_mean", ascending=False)
        print(f"[p2]   -- target: {dt} of the common-masked NDVI change")
        for r in sub.itertuples():
            tag = ("  <- CONTROL" if r.is_control else
                   ("  [si_comparable=False]" if not r.si_comparable else ""))
            name = (f"{r.encoder}/{r.feature_set}" if r.feature_set != "embedding"
                    else r.encoder)
            print(f"[p2]     {name:<34} rho {r.score_mean:+.3f} "
                  f"[{r.score_ci_lo:+.3f}, {r.score_ci_hi:+.3f}]  "
                  f"margin {r.margin_over_control:+.3f}  "
                  f"n={r.effective_n} CUBES{tag}")


def structural_hypothesis(df, verbose: bool = True) -> dict:
    """Does encoder ORDER on the delta probe match the training-objective hunch?

    research_plan_v3 Section 3/P2 suggests augmentation-invariance training
    (DINOv2) may discard state-change more than reconstruction-style training.
    This ranks the SI-comparable encoders on the primary delta configuration
    and REPORTS the ordering.

    It is not a powered test and must not be written up as one. There are two
    EO-relevant single-image points here -- ``satlas_s2_swinb_rgb`` and
    ``dinov2_vitb14`` -- plus ``imagenet_vit_b16`` and ``raw_features`` as
    anchors. Four points, one tile, one year, and the cube-clustered intervals
    overlap. The multi-image encoder is excluded because its consecutive
    embeddings share up to 7 of 8 source frames, which is a property of its
    input window and not of its training objective.

    AND THE ORDERING IS CHECKED UNDER EVERY FOLD MODE, because on this subset it
    is not stable. The two encoders the hunch is about swap places between
    ``cube`` and ``loco`` -- which is P1's finding recurring on the same pair
    (P1: 0.387 vs 0.386, a rank that moved with a scipy version). When the
    ordering flips, ``supported`` is reported as ``None``: neither confirmed nor
    refuted, because the ranking it would be read off does not exist at this
    sample size. Reporting ``False`` there would be as wrong as reporting
    ``True``.
    """
    p = STRUCTURAL_PRIMARY

    def _rank(mode):
        return df[(df.part == "B_delta")
                  & (df.delta_target == p["delta_target"])
                  & (df.aggregation == p["aggregation"])
                  & (df.feature_level == p["feature_level"])
                  & (df.readout == p["readout"])
                  & (df.fold_mode == mode)
                  & (df.feature_set == "embedding")
                  & (~df.is_control)
                  & (df.si_comparable)].sort_values("score_mean", ascending=False)

    sub = _rank(p["fold_mode"])
    assert len(sub) >= 2, "not enough SI-comparable rows to rank"
    assert MI_ENCODER not in set(sub.encoder), (
        f"{MI_ENCODER} is in the structural-hypothesis ranking. Its consecutive "
        "embeddings share up to 7 of 8 source frames, so its delta is a "
        "sliding-window increment, not a state change over the gap."
    )
    order = sub.encoder.tolist()
    pair = ("dinov2_vitb14", "satlas_s2_swinb_rgb")

    def _verdict(o):
        if not all(e in o for e in pair):
            return None
        # The hunch predicts DINOv2 BELOW Satlas, i.e. a larger rank index.
        return o.index(pair[0]) > o.index(pair[1])

    by_mode, verdicts = {}, {}
    for mode in sorted(set(df[df.part == "B_delta"].fold_mode)):
        o = _rank(mode).encoder.tolist()
        if not o:
            continue
        by_mode[mode] = o
        verdicts[mode] = _verdict(o)
    stable = len({v for v in verdicts.values() if v is not None}) <= 1
    supported = _verdict(order) if stable else None

    out = {"config": dict(p), "order": order,
           "scores": {r.encoder: float(r.score_mean) for r in sub.itertuples()},
           "ci": {r.encoder: (float(r.score_ci_lo), float(r.score_ci_hi))
                  for r in sub.itertuples()},
           "margins": {r.encoder: float(r.margin_over_control)
                       for r in sub.itertuples()},
           "hypothesis": ("augmentation-invariance training (dinov2_vitb14) "
                          "discards state-change MORE than reconstruction-style "
                          "training (satlas_s2_swinb_rgb)"),
           "supported": supported,
           "order_stable_across_fold_modes": bool(stable),
           "order_by_fold_mode": by_mode,
           "verdict_by_fold_mode": verdicts,
           "n_si_points": len(order),
           "excluded": [MI_ENCODER],
           "caveat": ("SUGGESTIVE, NOT POWERED: 2 EO-relevant single-image "
                      "points plus 2 anchors, one tile, one year, 20 cubes, "
                      "overlapping cube-clustered intervals.")}
    if verbose:
        print("\n[p2] STRUCTURAL HYPOTHESIS (report, do not assume)")
        print(f"[p2]   hunch: {out['hypothesis']}")
        print(f"[p2]   SI-comparable ranking on {p['aggregation']}/"
              f"{p['delta_target']}/{p['feature_level']}/{p['readout']}/"
              f"{p['fold_mode']}:")
        for i, e in enumerate(order):
            lo, hi = out["ci"][e]
            print(f"[p2]     {i + 1}. {e:<24} rho {out['scores'][e]:+.3f} "
                  f"[{lo:+.3f}, {hi:+.3f}]  margin over control "
                  f"{out['margins'][e]:+.3f}")
        print(f"[p2]   excluded from the ranking: {out['excluded']} "
              "(si_comparable=False)")
        print("[p2]   the same ranking under every fold mode:")
        for mode, o in by_mode.items():
            mark = {True: "hunch holds", False: "hunch fails",
                    None: "pair absent"}[verdicts[mode]]
            print(f"[p2]     {mode:<14} {' > '.join(o)}   ({mark})")
        if stable:
            print(f"[p2]   hunch SUPPORTED by the ordering: {out['supported']}")
        else:
            print("[p2]   hunch: NOT DETERMINABLE. The two encoders it is about "
                  "SWAP PLACES between")
            print("[p2]     fold modes, so there is no ordering to read it off. "
                  "P1 found the same")
            print("[p2]     thing about the same pair (0.387 vs 0.386, a rank "
                  "that moved with a")
            print("[p2]     scipy version). This is the sample size, not the "
                  "training objective.")
        print(f"[p2]   {out['caveat']}")
    return out


# ---------------------------------------------------------------------------
# The exit test's assertions
# ---------------------------------------------------------------------------

def assert_k2_verdict_recorded(df, encoders: Sequence[str] = ENCODER_ORDER) -> None:
    """Every encoder has a K2 verdict, and it came from a cube-clustered
    comparison against ``raw_features``."""
    for col in ("k2_verdict", "k2_verdict_band_matched", "k2_margin_over_raw",
                "k2_margin_over_band_matched", "k2_reference_config",
                "k2_margin_ci_lo", "k2_margin_ci_hi", "k2_separable"):
        assert col in df.columns, f"no {col} column; run add_k2_verdicts"
    allowed = {"passed", "audited: lossy", "baseline"}
    for enc in encoders:
        sub = df[df.encoder == enc]
        assert len(sub), f"no rows at all for encoder {enc!r}"
        v = set(sub.k2_verdict.unique())
        vb = set(sub.k2_verdict_band_matched.unique())
        assert len(v) == 1 and v <= allowed, (
            f"{enc}: k2_verdict is {v}, expected exactly one of {allowed}"
        )
        assert len(vb) == 1 and vb <= allowed, (
            f"{enc}: k2_verdict_band_matched is {vb}"
        )
        m = sub.k2_margin_over_raw.dropna().unique()
        assert m.size == 1 and np.isfinite(m[0]), (
            f"{enc}: k2_margin_over_raw is {m}; the verdict must be derived "
            "from a recorded numeric comparison, not asserted"
        )
        if enc == BASELINE_ENCODER:
            assert v == {"baseline"}, f"{enc} must be labelled the baseline"
        else:
            # The verdict must agree with the margin it claims to come from.
            expect = "passed" if m[0] > 0 else "audited: lossy"
            assert v == {expect}, (
                f"{enc}: k2_verdict {v} disagrees with k2_margin_over_raw "
                f"{m[0]:+.4f}; the verdict was not derived from the comparison"
            )
    cfg = set(df.k2_reference_config.unique())
    assert cfg == {f"{K2_PRIMARY['aggregation']}/{K2_PRIMARY['feature_level']}/"
                   f"{K2_PRIMARY['fold_mode']}"}, (
        f"the K2 verdict claims reference config {cfg}, not the cube-clustered "
        f"primary {K2_PRIMARY}"
    )


# What each control is invariant to, and therefore what its value must be
# identical across. The gap-length-alone control is a function of gap_days and
# the target alone, so it does not change with the encoder, the feature level
# or the read-out. The retention control DOES change with the feature level --
# clear_frac is per frame, grid_clear_frac is per cell, and P1 reports both on
# purpose -- so its key carries the level. Stating this as data rather than as
# prose is what lets one assertion check both without a special case.
_CONTROL_KEY = {
    "gap_only": ("part", "target", "fold_mode"),
    "retention": ("part", "target", "fold_mode", "feature_level"),
}


def assert_control_identical_across_views(df) -> None:
    """A control must be digit-for-digit identical under every filter it is
    invariant to.

    Two things are checked, and the second is the one that matters:

    1. The control ROWS themselves agree across the labels they are emitted
       under (read-out, feature set, encoder label, and for ``gap_only`` the
       feature level too). Duplicated numbers nobody checks are exactly how two
       views of one table start disagreeing.
    2. ``control_score`` -- the copy of the control that ``add_margins`` writes
       onto EVERY row -- is identical across every filtered view. This is the
       real guarantee: a reader who filters to one encoder still has the
       control's value in front of them, on the row, and it is the same number
       the control's own row reports.

    ``fold_mode`` is part of the KEY, not one of the invariant labels: a
    control evaluated on a different set of held-out cubes is a different
    number, and reporting one value across modes would be a false identity, not
    a consistency guarantee. ``assert_degenerate_control_present`` separately
    asserts the control exists for every fold mode.
    """
    cols = ["score_mean", "score_std", "score_ci_lo", "score_ci_hi",
            "per_fold_score", "effective_n"]
    for kind, key in _CONTROL_KEY.items():
        key = list(key)
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
                "disagreement means one copy was computed on different rows "
                "and the CSV was built inconsistently."
            )
        if "control_score" not in df.columns:
            continue
        rows = df[df.control_kind == kind]
        g = rows.groupby(key)["control_score"].nunique(dropna=False)
        bad = g[g > 1]
        assert bad.empty, (
            f"the {kind} control's value differs between filtered views of the "
            f"table for {list(bad.index)[:3]}: filtering by encoder, feature "
            "level or read-out changes the control a row is measured against. "
            "The margins in those views are not comparable."
        )
        # The copy on the rows must equal the control's own row, not merely be
        # self-consistent -- otherwise both could be uniformly wrong.
        own = sub.groupby(key)["score_mean"].first()
        seen = rows.groupby(key)["control_score"].first()
        np.testing.assert_array_equal(
            own.reindex(seen.index).to_numpy(), seen.to_numpy(),
            err_msg=f"the {kind} control_score carried on each row differs "
                    "from the control's own row")


def assert_degenerate_control_present(df) -> None:
    """The gap-length-alone control exists for EVERY fold mode, target, level
    and read-out that has a delta row; the retention control for every Part A
    configuration."""
    b = df[df.part == "B_delta"]
    if len(b):
        need = b[~b.is_control].groupby(
            ["target", "fold_mode", "feature_level", "readout"]).size().index
        have = set(map(tuple, b[b.model_kind == "gap_only"][
            ["target", "fold_mode", "feature_level", "readout"]].to_numpy()))
        missing = [t for t in need if tuple(t) not in have]
        assert not missing, (
            f"{len(missing)} delta configuration(s) have no gap-length-alone "
            f"control, e.g. {missing[:3]}. The control is P2's version of P1's "
            "degenerate control and it is not optional: without it a delta "
            "score cannot be distinguished from a calendar."
        )
        modes = set(b.fold_mode.unique())
        got = set(b[b.model_kind == "gap_only"].fold_mode.unique())
        assert modes == got, (
            f"the gap-length-alone control is missing for fold mode(s) "
            f"{sorted(modes - got)}"
        )
    a = df[df.part == "A_reconstruction"]
    if len(a):
        need = a[~a.is_control].groupby(
            ["target", "fold_mode", "feature_level"]).size().index
        have = set(map(tuple, a[a.model_kind == "retention"][
            ["target", "fold_mode", "feature_level"]].to_numpy()))
        missing = [t for t in need if tuple(t) not in have]
        assert not missing, (
            f"{len(missing)} Part A configuration(s) have no retention-only "
            f"control, e.g. {missing[:3]}"
        )
        modes = set(a.fold_mode.unique())
        got = set(a[a.model_kind == "retention"].fold_mode.unique())
        assert modes == got, (
            f"the retention control is missing for fold mode(s) "
            f"{sorted(modes - got)}"
        )


def assert_mi_flagged_and_excluded(df, ranking: dict) -> None:
    """Every multi-image row says ``si_comparable=False``, and the ranking
    does not contain it."""
    mi = df[df.encoder == MI_ENCODER]
    assert len(mi), f"no rows for {MI_ENCODER}; its scores must be REPORTED"
    assert not mi.si_comparable.any(), (
        f"{int(mi.si_comparable.sum())} {MI_ENCODER} row(s) claim "
        "si_comparable=True. Its consecutive embeddings share up to 7 of 8 "
        "source frames, so E(t+1)-E(t) is a sliding-window increment, not a "
        "state change over the gap."
    )
    others = df[(df.encoder != MI_ENCODER) & (df.encoder != "none")]
    assert others.si_comparable.all(), (
        "a single-image encoder is flagged si_comparable=False"
    )
    assert MI_ENCODER not in ranking["order"], (
        f"{MI_ENCODER} appears in the structural-hypothesis ranking at "
        f"position {ranking['order'].index(MI_ENCODER)}"
    )
    assert MI_ENCODER in ranking["excluded"], (
        f"{MI_ENCODER} is not recorded as excluded from the ranking"
    )


def assert_results_complete(df, encoders: Sequence[str] = ENCODER_ORDER,
                            fold_modes: Sequence[str] = FOLD_MODES) -> None:
    """Every cell the exit test asks for is present, and so are the baselines."""
    for col in ("part", "aggregation", "delta_target", "target", "feature_level",
                "fold_mode", "encoder", "feature_set", "model_kind", "readout",
                "is_control", "si_comparable", "metric", "score_mean",
                "score_ci_lo", "score_ci_hi", "effective_n", "n_cubes",
                "n_rows_test_total", "effective_n_per_fold", "gap_source"):
        assert col in df.columns, f"the results table has no {col!r} column"
    assert set(df.part.unique()) == set(PARTS), (
        f"the table has parts {sorted(df.part.unique())}, expected {sorted(PARTS)}"
    )
    for part in PARTS:
        sub = df[df.part == part]
        for enc in encoders:
            assert (sub.encoder == enc).any(), f"{part}: no rows for {enc!r}"
        for mode in fold_modes:
            assert (sub.fold_mode == mode).any(), f"{part}: no {mode!r} rows"
    assert ((df.encoder == BASELINE_ENCODER)
            & (df.feature_set == BAND_MATCHED_BASELINE)).any(), (
        f"no {BAND_MATCHED_BASELINE} rows; the band-matched baseline is what "
        "makes the K2 verdict a statement about the representation rather than "
        "about which bands the baseline was given"
    )
    b = df[df.part == "B_delta"]
    assert len(b), "the table has no delta rows"
    assert (b.gap_source == "daily_axis_index").all(), (
        "a delta row does not declare daily_axis_index as its gap source. The "
        "gap must never come from original_axis_index, which counts "
        "ACQUISITIONS and is the embedding join key."
    )
    assert b.common_masked.all(), (
        "a delta row does not declare that its target was common-masked"
    )
    for agg in AGGREGATIONS:
        assert (df.aggregation == agg).any(), f"no {agg!r} rows"
    for dt in DELTA_TARGETS:
        assert (b.delta_target == dt).any(), f"no {dt!r} delta rows"
    bad = df[~np.isfinite(df.score_mean)]
    assert bad.empty, (
        f"{len(bad)} row(s) have a non-finite score, e.g.\n"
        f"{bad[['part', 'target', 'encoder', 'fold_mode']].head(3)}"
    )


def assert_plausibility_screen_declared(df, required: bool = True) -> None:
    """Every P2 row declares one uniform frame/pair screening contract."""
    for col in ("plausibility_screen", "plausibility_screen_def",
                "n_implausible_frames", "n_pairs_dropped_implausible"):
        assert col in df.columns, f"the P2 table has no {col!r} column"
    values = df.plausibility_screen.astype(bool)
    assert values.nunique() == 1, (
        "the P2 table mixes screened and unscreened rows; their pair sets "
        "differ and the scores are not comparable"
    )
    actual = bool(values.iloc[0])
    assert actual == bool(required), (
        f"P2 declares plausibility_screen={actual}, expected {required}"
    )
    expected_label = (
        P2_PLAUSIBILITY_SCREEN_LABEL if required else "not_applied")
    assert (df.plausibility_screen_def.astype(str) == expected_label).all(), (
        "P2's plausibility_screen_def does not match the declared mode"
    )
    n_bad = df.n_implausible_frames.astype(int)
    n_pairs = df.n_pairs_dropped_implausible.astype(int)
    assert n_bad.nunique() == 1 and n_pairs.nunique() == 1, (
        "P2 screen counts must be run-level constants on every row"
    )
    assert (n_bad >= 0).all() and (n_pairs >= 0).all()
    if not required:
        assert (n_pairs == 0).all(), (
            "an unscreened P2 table reports pairs dropped by the screen"
        )
    if required and int(n_bad.iloc[0]) > 0:
        assert int(n_pairs.iloc[0]) > 0, (
            "frames fail the screen but no P2 pair reports being dropped"
        )


def results_path(name: str = "p2_deltas_results.csv", root: str | None = None) -> str:
    """Where a P2 table is written.

    ``root`` overrides the phase directory. A SCALED run's results belong beside
    the cubes they were computed from -- the placement ``scripts/scale_p4.py``
    already uses -- and not in the 20-cube phase folder, because the two tables
    answer the same question at different sample sizes and the published one
    must stay on disk to be compared against. The default is the phase
    directory, so the 20-cube call is unchanged and a scaled run cannot
    overwrite the published artefact by forgetting an argument.
    """
    base = phase_dir(PHASE, "results") if root is None else root
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, name)
