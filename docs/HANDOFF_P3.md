# Handoff to P3 — read this first

State of the repo as of **2026-08-11**, after Phase 1.6 (P2). One page. Every
claim here points at the file that owns it; nothing is restated in full.

- **why** a decision was made → [DECISIONS.md](DECISIONS.md)
- **what was measured** → [../log.md](../log.md)
- **how to run a phase** → [../RUNBOOK.md](../RUNBOOK.md)
- **evidence** (verbatim stdout / executed notebooks) → [runs/](runs/),
  `../notebooks/runs/`
- the previous handoff, still accurate except where noted → [HANDOFF_P2.md](HANDOFF_P2.md)

---

## 1. What P3 inherits, and what changed under it

| | state |
|---|---|
| `data/raw` | **20 cubes**, tile 32UNU, 2018. Unchanged. Still do not add cubes here (HANDOFF_P2 §4). |
| `encoders/manifest.py` | Unchanged since P4's two fixes. Still **rebuild any cached manifest**. |
| `probes/cv.py` | Unchanged. `cube`, `loco`, `spatial_block` run; `year`, `tile`, `crossed` correctly raise. |
| `probes/p2_deltas.py` | **New.** Gate K2, the delta probe, and the pair index P3 should reuse rather than rebuild. |
| `probes/p1_appearance.FeatureBlock` | Reuse it. P2 imports it, as P4 does. |
| tests | **0 failed, 5 skipped.** The count grows every phase; the invariant is 0 failed. |

**Reusable from P2, so P3 does not re-derive them:**

| you need | import |
|---|---|
| consecutive-frame pairs with a **correct** day gap | `p2.pair_index` → `DeltaPairs` |
| the run-time proof the gap is not the join key | `p2.assert_gap_axes_disagree` |
| common-masked NDVI change for one pair | `p2.common_masked_delta` |
| ...for a whole cube / the whole subset | `p2.cube_common_masked_deltas`, `p2.build_pair_targets` |
| current-NDVI targets aligned to the manifest | `p2.frame_targets` (wraps `p4.cube_frame_targets`) |
| a ridge path over a penalty grid, one factorisation | `p2.ridge_path` (pinned against sklearn to 1e-8) |
| a leakage-safe penalty tuner | `p2.select_ridge_alpha(block, manifest, train_rows, metric)` |

---

## 2. Findings that constrain what P3 may claim

- **K2 is passed, and the exclusion list is empty.** All four network encoders
  clear the band-matched floor at the primary configuration. `satlas_s2_swinb_mi_rgb`
  is the only `audited: lossy` verdict that is **separable** from the baseline
  on the paired per-fold difference — and it is the multi-image control, which
  was never in the single-image column anyway. **No single-image encoder is
  excluded from P3.**
- **Read `k2_verdict_band_matched`, not `k2_verdict`.** `raw_features` carries
  `NDVI_mean`…`NDVI_p90` as columns, so at the MATCHED level (cell features →
  cell target, pooled → cube target) it reconstructs the target at R² = +1.000
  and the verdict against it measures the baseline, not the encoder. The
  primary configuration (`cube_mean` from `grid_cell`) is not matched and is a
  fair comparison; the secondary views are not. See `probes/p2_deltas.py`'s
  docstring.
- **A verdict is not a rejection, and K2 is a floor check, not a ranking.**
  `k2_separable` is the paired per-fold interval on (encoder − baseline). Every
  single-image encoder's spans zero: Satlas SI's nominal +0.105 lead is
  `[-0.118, +0.327]`, DINOv2's −0.006 is `[-0.142, +0.131]`. Only the MI
  control is separable (`[-0.592, -0.134]`). So K2 establishes "nothing is
  catastrophically lossy" and **nothing more** — do not quote the +0.545 vs
  +0.440 spread as an encoder ranking, and do not drop an encoder on a
  non-separable verdict.
- **SIGN is recoverable; MAGNITUDE is not.** On the primary delta
  configuration, every encoder beats the gap-length control on the sign of the
  NDVI change by a wide margin (DINOv2 +0.54, Satlas SI +0.48, ImageNet +0.46
  against a control of −0.12). On **magnitude, the gap-length control at +0.209
  beats every encoder** — DINOv2's +0.174 is the best and is still below it.
  **P3 must not treat "the encoder knows how much it changed" as established.
  It is not.**
- **The gap-length control is P2's version of P1's degenerate control, and it
  bites.** Same lesson, third phase running: a target correlated with elapsed
  time carries a competitor that needs no image. P3's horizons are *defined* in
  days, so this control is if anything more dangerous there. Carry it.
- **The structural hunch is NOT DETERMINABLE, and the reason matters more than
  the hunch.** DINOv2 ranks above Satlas SI under `cube` folds (+0.536 vs
  +0.481) and *below* it under `loco` (+0.481 vs +0.531) and `spatial_block`
  (+0.473 vs +0.489); at cell level they are +0.4576 and +0.4596. **This is the
  same pair P1 could not rank** (0.387 vs 0.386, a rank that moved with a scipy
  version). Two probes, same answer: at 20 cubes of one tile these two encoders
  are not separable. `structural_hypothesis` returns `supported=None` and
  `order_stable_across_fold_modes=False`. Do not build on the ordering in
  either direction.
- **Common-masking does not collapse, and survival is not monotone in gap.**
  244/244 pairs survive; median 88.8%, minimum 27.2%. Survival is *worse* at 15
  days (0.783) than at 30 (0.985), because a long gap exists precisely because
  the frames between it were cloudy, which leaves the two surviving endpoints
  unusually clear. **Do not model pixel survival as a function of horizon.**
- **`spatial_block` still kills things**, as in P1 and P4. Reported, not dropped.
- **Effective n is CUBES.** Unchanged and non-negotiable. 264 frames, 244 pairs
  and 4224 cells are not independent observations.

---

## 3. Patterns worth copying from P2

- **Emit the control's value onto every row** (`control_score`), not only as its
  own row. That is what makes "filtering the CSV cannot drop the control" true
  rather than intended, and it makes the digit-for-digit assertion checkable
  from any direction a reader might filter.
- **Say what a control is invariant to, as data.** `_CONTROL_KEY` states that
  the gap control is invariant to encoder/level/read-out but NOT to fold mode.
  One assertion then covers two controls with different invariances and no
  special case.
- **Prove a guard can fail.** `assert_gap_axes_disagree` is exercised in the
  tests by feeding it the *wrong column* and asserting it refuses. A guard only
  ever run on correct input is a hypothesis.
- **Make the fixture able to expose the bug.** The synthetic manifest uses
  `day_step=10, acq_step=2` — a 5× disagreement, like the real data. A fixture
  where the two axes coincide would let a probe that read the wrong column pass
  every test in the file.
- **Two poison tests, always.** Held-out poison must not move the fit; training
  poison must. The first alone passes on a function that ignores its input.
- **A speed optimisation needs an equivalence test.** `ridge_path` re-solves one
  factorisation per penalty instead of refitting; it is pinned against
  `sklearn.linear_model.Ridge` to 1e-8 in both the tall and wide regimes.

---

## 4. Traps, with the reason each is a trap

- **`original_axis_index` is the embedding join key and counts ACQUISITIONS.
  `daily_axis_index` counts DAYS.** P3's horizons are in days. Using the join
  key would make every horizon ~5× too short — and it is silent, because the
  wrong column is finite, integer, in range, monotone within a cube, and
  correlated with the right one. `p2.assert_gap_axes_disagree` is the live
  check; call it, do not re-derive it.
- **Differencing two per-frame means is not a change.** Each mean is over its
  own valid pixel set, so the difference partly compares two different pieces of
  ground. The two answers agree only when the masks agree — i.e. exactly when
  the distinction does not matter — which is why a spot check will not find it.
- **`raw_features` contains the target.** Any probe whose target is an NDVI
  aggregate must report the band-matched baseline beside it or the comparison is
  meaningless. `RAW_BAND_COLUMNS` is a column slice; it costs nothing.
- **A pair straddling a fold leaks the second frame into training.** It cannot
  happen under a cube-grouped mode, which is why P2 asserts it per fold rather
  than arguing it once — the assertion is what would catch `temporal`, the mode
  P3 is explicitly allowed to use as a robustness variant.
- **Hard-coded test counts go stale every phase.** Assert 0 failed, not N passed.

---

## 5. Open items (not blocking P3)

| item | where |
|---|---|
| Stage B climatology's second-order leak path — **still unfixed**, must be quantified before any H1-style number is quoted | `probes/p4_ceiling.run_stage_b`, DECISIONS 2026-08-11 |
| P4's proxy climatology is still **unvalidated** | `scripts/validate_proxy_climatology.py` |
| `docs/specs/` has no `phase1_6` entry; `probes/p2_deltas.py`'s module docstring serves as the spec, as it does for P4 | — |
| The magnitude target is unresolved, not settled. It may need a longer baseline than 20 cubes of one tile can give | `data/phase1_6/results/` |

---

## 6. P2's result in one line

On tile 32UNU at 20 cubes, all five encoders clear the K2 reconstruction floor
(band-matched; only the multi-image control is separably below it), and frozen
embedding **deltas recover the SIGN of common-masked NDVI change well**
(DINOv2 ρ = +0.536 [+0.397, +0.674] against a gap-length control of −0.118) —
but **recover its MAGNITUDE no better than gap length alone**, which at +0.209
beats every encoder. P3 inherits a licensed set of encoders and one explicit
warning: direction is available in these representations, rate is not.
