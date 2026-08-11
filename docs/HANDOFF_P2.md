# Handoff to P2 — read this first

State of the repo as of **2026-08-11**, after Phase 1.5 (P4). One page. Every
claim here points at the file that owns it; nothing is restated in full.

- **why** a decision was made → [DECISIONS.md](DECISIONS.md)
- **what was measured** → [../log.md](../log.md)
- **how to run a phase** → [../RUNBOOK.md](../RUNBOOK.md)
- **evidence** (verbatim stdout / executed notebooks) → [runs/](runs/),
  `../notebooks/runs/`

---

## 1. What P2 inherits, and what changed under it

| | state |
|---|---|
| `data/raw` | **20 cubes**, tile 32UNU, 2018. Phase 1.2's 100 `.npz` are keyed to exactly these — do not add cubes here (see §4). |
| `encoders/manifest.py` | **Two bugs fixed since P1.** Rebuild any cached manifest. |
| `probes/cv.py` | Unchanged. `cube`, `loco`, `spatial_block` run; `year`, `tile`, `crossed` correctly raise on this subset. |
| `probes/p1_appearance.FeatureBlock` | Reuse it. It exists because P2/P3/P4 all consume cell-level rows; P4 imports it rather than rewriting it. |
| tests | **383 passed, 5 skipped.** The count grows each phase; the invariant is 0 failed / 5 skipped. |

**The two manifest fixes, because they change values P2 may read:**

1. **E-OBS was joined to the wrong day.** `original_axis_index` counts
   ACQUISITIONS (~29/cube); the weather series live on the DAILY axis (~150).
   Indexing one with the other put 0 of 264 rows on their own day (median 53
   days off; mean-temp MAE 6.26 K). Fixed: new `daily_axis_index` column,
   timestamp join, and `assert_weather_join()` which re-derives it from the
   cubes. **Call it before fitting anything on weather.**
2. **`year` was the filename's window-start**, not the frame's calendar year.
   Invisible on single-year 32UNU; wrong on 1666/2092 rows of a real seasonal
   cube. This had blocked `crossed` mode — and therefore P4 Stage B — on every
   tile since Phase 1.3.

**Two axes, one rule:** `original_axis_index` = the embedding join key.
`daily_axis_index` = horizons and weather windows. Never swap them.

---

## 2. Findings that constrain what P2 may claim

- **The degenerate control is a competitor, not a floor.** `[clear_frac,
  window_span_days]` — two numbers, no image — decoded season at 0.646–0.658 in
  P1, and in P4 it still absorbs a real share of the signal at 115 cubes (it did
  **not** wash out with more data). **Any P2 target correlated with time-of-year
  must carry a retention-only row.** Cheap: the covariates are already cached,
  and `window_span_days` is recomputable from the manifest via
  `encoders.pipeline.window_span_days` — no embedding read required.
- **Day-of-year is nearly a 36-level categorical here.** All rows satisfy
  `doy % 5 == 2` — one Sentinel-2 orbit lattice — and ~63% of a typical weather
  feature is recoverable from the date alone. Adding cubes adds rows per date,
  **not new dates**. A flexible learner given only day-of-year will therefore
  score well; that is collinearity, not signal. In P4 it accounted for **four
  fifths** of the best-looking headline (`cube_p90` HGB +0.320 against its own
  DOY control at +0.256).
- **Cell-level targets resist it**; frame-level ones do not. 16 cells share a
  date, so per-date memorisation buys much less. P4's defensible cell is
  `cell_mean`. Prefer cell-level targets.
- **`spatial_block` kills everything**, in P1 and P4 alike. Expect it to kill
  P2's numbers too, and report it rather than dropping it.
- **Capacity is sample-size dependent.** At 20 cubes "capacity hurts, only the
  linear model is positive"; at 115 cubes the boosted tree is strongest and the
  linear model is weak. Do not inherit either conclusion — re-measure at P2's n.
- **Effective n is CUBES.** 264 frames and 4195 cells are not independent
  observations: 16 cells share a sky, ~13 frames share a place. Cluster every
  interval at the cube level and print the effective n beside the score.

---

## 3. Patterns worth copying from P4

- **Prevent leakage with the signature, not with discipline.** P4's climatology
  takes a required `train_idx` positional, so it cannot be fitted on everything
  by omitting an argument. The test poisons held-out rows and asserts
  bit-identical coefficients — **plus a companion asserting it DOES move when a
  training row changes.** A test that can only pass proves nothing.
- **Let a control choose a hyperparameter, if the choice is one-directional.**
  P4's harmonic order was picked by its day-of-year sanity control; raising it
  only ever *lowers* the reported headline, which is what makes selecting
  against a held-out diagnostic legitimate there.
- **Emit controls under every filter label** and assert the copies agree
  digit-for-digit, so filtering the CSV can never silently drop a control.
- **No tuning beats guarded tuning.** P4 fixes all hyperparameters a priori;
  there is no selection loop, so there is nothing to prove clean.
- **Archive verbatim stdout** to `docs/runs/` (scripts) or `notebooks/runs/`
  (phases). Numbers living only in prose have no evidence behind them.

---

## 4. Traps, with the reason each is a trap

- **Do not add cubes to `data/raw`.** Phase 1.2's cache holds 20×5 `.npz`;
  a larger manifest makes `assert_embeddings_complete` fail for every phase that
  reads embeddings. P4 could scale only because it reads none and takes
  `cube_dir` as a parameter (`scripts/scale_p4.py` → `data/scaled_32UNU/`).
- **A guard that has never fired on real data is a hypothesis, not a control.**
  The `year` bug sat behind a correct refusal, a passing test, and a docstring
  naming the exact fix — for four phases — because no multi-year cube had ever
  been fed to them. Exercise P2's guards against real data of the shape they
  were written for.
- **A derived column cannot be validated against the table it lives in.** The
  wrong-day weather was finite, in range, right dtype and internally consistent.
  Only going back to the source caught it.
- **Hard-coded test counts go stale every phase.** Assert 0 failed, not N passed.

---

## 5. Open items (not blocking P2)

| item | where |
|---|---|
| Stage B climatology has a second-order leak path; **must be fixed and quantified before any H1-style number is quoted** | `probes/p4_ceiling.run_stage_b`, DECISIONS 2026-08-11 |
| P4's proxy climatology is **unvalidated** — the real one differs by 0.328 in point estimate and the intervals overlap only because one is enormous | `scripts/validate_proxy_climatology.py`, `runs/2026-08-10_*` |
| `docs/specs/` has no `phase1_5` entry; the module docstring of `probes/p4_ceiling.py` serves as the spec | — |
| Growing 32UNU beyond 115 cubes is unblocked (192 listed, 115 after the 64 px non-overlap rule) | `scripts/scale_p4.py` |

---

## 6. P4's result in one line

On tile 32UNU at 115 cubes, weather explains **0.09–0.13** of the within-season
post-climatology NDVI anomaly at grid-cell level (`cell_mean`, HGB, cube and
LOCO folds), clearing both the observation-process control and the day-of-year
control. It is a **proxy**-climatology ceiling, not H1 — every row says so in
its `climatology_def` column. This is P3's denominator, not a headline result.
