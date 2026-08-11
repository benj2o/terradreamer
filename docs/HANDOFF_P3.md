# Handoff to P3 — read this first

State of the repo as of **2026-08-11**, after Phase 1.6 (P2) and its
115-cube re-run (Phase 1.7 cache + `scripts/scale_p2.py`). One page. Every
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
| `data/raw` | **20 cubes**, tile 32UNU, 2018. Unchanged. Still do not add cubes here (HANDOFF_P2 §4) — Phase 1.2's cache is keyed to exactly these. |
| `data/scaled_32UNU/raw` | **115 cubes**, same tile/year, 64 px non-overlap. The set P4's ceiling AND P2's scaled table were both measured on. |
| `data/scaled_32UNU/embeddings` | **575 `.npz`** = 115 × 5 encoders (Phase 1.7, Colab GPU). **This is what P3 should read.** Verified against the 20-cube cache: metadata bit-identical, network outputs within 1.5e-4. |
| `data/scaled_32UNU/masks` | **115 `.npz`** per-pixel masks. Required for common-masking; cannot be approximated from `clear_frac`. |
| mirror | The same three directories are on Drive at `My Drive/NeurIPS-CCAI-2026/data/scaled_32UNU/`. `data/scaled_*` is gitignored, so the repo carries the CODE and the Drive/local copies carry the 239 MB. |
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
| fold-level parallelism, bit-identical to serial | `p2.evaluate(..., n_jobs=N)` — LOCO is `k = n_cubes`, so at 115 the fold count scales too; serial is ~4× slower |
| proof that two caches are the same experiment | `encoders.pipeline.assert_caches_agree(old_dir, new_dir, cubes, encoders)` |
| a scaled runner to copy | `scripts/scale_p2.py` (mirrors `scale_p4.py`; validates the cache, then diffs every headline against the 20-cube table) |

---

## 2. Findings that constrain what P3 may claim

- **K2 is passed, the exclusion list is EMPTY, and at 115 cubes it is a
  ranking.** Three of four networks separably beat the hand-crafted baseline:
  satlas SI +0.713 `[+0.094, +0.156]` on the paired difference, imagenet +0.687
  `[+0.036, +0.162]`, dinov2 +0.649 `[+0.021, +0.101]`, against raw_features
  +0.588. **No encoder is excluded from P3** — `satlas_s2_swinb_mi_rgb`, the one
  excluded at 20 cubes, is no longer separable (−0.069 `[−0.192, +0.053]`) and
  passes band-matched. Quote the 115-cube numbers.
- **Read `k2_verdict_band_matched`, not `k2_verdict`.** `raw_features` carries
  `NDVI_mean`…`NDVI_p90` as columns, so at the MATCHED level (cell features →
  cell target, pooled → cube target) it reconstructs the target at R² = +1.000
  and the verdict against it measures the baseline, not the encoder. The
  primary configuration (`cube_mean` from `grid_cell`) is not matched and is a
  fair comparison; the secondary views are not. See `probes/p2_deltas.py`'s
  docstring.
- **`k2_separable` earned its keep in both directions.** At 20 cubes it stopped
  DINOv2 — the strongest encoder on the whole delta probe — being excluded on a
  −0.006 point estimate; at 115 DINOv2 is +0.061 and separably ABOVE the
  baseline. It also carried the reverse: the MI encoder WAS separably lossy at
  20 and is not at 115, so its exclusion was retracted. **Never drop an encoder
  on a non-separable verdict.**
- **SIGN is recoverable; MAGNITUDE is not — and at 115 cubes the reason is
  cleaner than it looked.** Sign margins are +0.44 to +0.72 and move by less
  than 0.05 across all three fold modes; `spatial_block` does not kill them.
  Magnitude: the 20-cube claim that "the gap-length control beats every
  encoder" was **wrong** — that control was itself a small-sample artefact and
  falls from +0.209 to **+0.063 `[−0.004, +0.130]`**. The corrected finding is
  worse for the encoders, not better: every absolute correlation is +0.06 to
  +0.12, margins are +0.00 to +0.07, and **three of four encoders flip sign
  across fold modes**. Only DINOv2 is consistently positive, by ≤0.065.
  **P3 must not treat "the encoder knows how much it changed" as established.**
- **On the delta SIGN probe the band-matched hand-crafted baseline BEATS every
  network.** `raw_rgb_only` reaches +0.695 against DINOv2's +0.606 — same three
  bands, no NDVI column, seven percentiles per band. So "sign is recoverable" is
  established; "frozen foundation models are the best way to recover it" is
  **not**. The networks clear `raw_rgb_only` on gate K2 and lose to it here.
  P3 must carry the band-matched row on every table for exactly this reason, and
  must not quote `raw_features` (+0.785) as if it were a model result — it holds
  `NDVI_mean` as a column.
- **The gap-length control is P2's version of P1's degenerate control, and it
  bites.** Same lesson, third phase running: a target correlated with elapsed
  time carries a competitor that needs no image. P3's horizons are *defined* in
  days, so this control is if anything more dangerous there. Carry it.
- **The structural hunch is now DETERMINABLE, and it is REFUTED.** At 20 cubes
  DINOv2 and Satlas SI swapped rank between fold modes and
  `structural_hypothesis` returned `supported=None`. At 115 the ordering is
  **identical under all three modes** — `raw_features > dinov2 > satlas_SI >
  imagenet` — so `order_stable_across_fold_modes=True, supported=False`.
  research_plan_v3 §3/P2 expected augmentation-invariance training to discard
  state-change MORE than reconstruction-style training; it does the opposite,
  consistently. Still 2 EO-relevant SI points, so this refutes the stated
  direction rather than establishing a mechanism.
- **Common-masking does not collapse, and survival is not monotone in gap.**
  At 115 cubes: **1465/1465 pairs survive**, median 83.5% of pixels, minimum
  21.3%, zero pairs with no shared pixel — the 20-cube behaviour replicates
  (244/244, median 88.8%). Survival is *worse* at 15 days than at 30, because a
  long gap exists precisely because the frames between it were cloudy, which
  leaves the two surviving endpoints unusually clear. **Do not model pixel
  survival as a function of horizon.**
- **`spatial_block` does NOT kill P2**, unlike P1 and P4. Sign margins under
  `spatial_block` are within 0.02 of the `cube` ones at 115 cubes. This is the
  one probe in the project whose headline survives the spatial stress test, and
  it is worth saying so rather than inheriting the general warning.
- **Effective n is CUBES.** Unchanged and non-negotiable. At 115 cubes that is
  1580 frames, 1465 pairs and 25 280 cells over **115** independent weather
  realisations — cluster on the last number, never the first three.

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
| The magnitude target is now **settled negative** at 115 cubes: no encoder recovers rate (ρ ≤ 0.12, margins sign-flipping). If P3 needs rate, it needs a different read-out or a change-trained encoder, not more cubes | `data/scaled_32UNU/p2_scaled_results.csv` |

---

## 6. P2's result in one line

On tile 32UNU at **115 cubes**, three of four frozen networks **separably beat**
a hand-crafted 35-feature baseline at recovering current NDVI (gate K2 passed,
no encoder excluded), and their **deltas recover the SIGN of common-masked NDVI
change robustly** (DINOv2 ρ = +0.606 [+0.546, +0.665], margin +0.54 over the
gap-length control, stable across all three fold modes) — but recover its
**MAGNITUDE not at all** (every ρ ≤ +0.12, margins ≤ +0.07, sign-flipping across
fold modes for three of four encoders). P3 inherits every encoder and one
warning: **direction is available in these representations, rate is not.**

*The 20-cube version of this line said the gap-length control beat every encoder
on magnitude and that the MI encoder was excluded. Both were small-sample
artefacts. See log.md.*
