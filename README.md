# Probing frozen EO foundation-model representations for dynamics and forecastability

GreenEarthNet / EarthNet2021 minicubes. Tile 32UNU first (Allgäu / Upper Swabia):
everything is prototyped on a small Alpine-foreland subset before any scale-up.
Scale-up generalises by land-cover strata, not by city.

## Standing constraints

1. No pretrained model is ever fine-tuned. All pretrained encoders are frozen:
   `.eval()` and `torch.no_grad()`, always.
2. Every array's shape is printed and asserted.
3. Never re-implement a function that exists in this repo. Import it.
4. Two files are canonical and must never be duplicated or redefined:
   - [`data/ndvi.py`](data/ndvi.py), the single definition of the target.
   - `probes/cv.py` (Phase 1.3), the single definition of leakage-safe splits.
5. Any number produced outside `probes/cv.py` does not exist. Phase 1.1 numbers
   are diagnostics, not results.

## Layout

```
data/ndvi.py                 canonical NDVI. Exactly one function. Never copy it.
data/loader.py               (values, timestamps, mask) per cube. Mask polarity is
                             decided here, once, in valid_mask_from_codes.
data/download_greenearthnet.py  PRIMARY: 20 pre-processed cubes, tile 32UNU, ~15 s
data/download_minicubes.py   live Sentinel-2 extraction. Any location, 14.7 h/20 cubes
data/stackstac_compat.py     shim for stackstac >= 0.5 vs earthnet-minicuber 0.1.3
data/climatology.py          GreenEarthNet NDVI climatology. Raises on one year
data/diagnose.py             four escalating checks, stops at the first failure
encoders/                    Tier A frozen encoders, one interface:
                             encode_bundle(frames[T, C, H, W]) -> {"pooled": [T, D],
                             "grid": [T, 16, D_grid] fp16, named variants, ...}.
                             base.py holds the frozen/batched/asserted machinery,
                             the 4x4 grid pooling and window_len/window_span_days;
                             frames.py the clear-fraction rule, valid-reflectance
                             check and per-cell clear fraction. Five wrappers:
                               raw_features          D=35    not a network
                               imagenet_vit_b16      D=1536  concat(cls, patch-mean)
                               dinov2_vitb14         D=3840  published linear-probe recipe
                               satlas_s2_swinb_rgb   D=1024  single-image, EO-native
                               satlas_s2_swinb_mi_rgb D=1024 multi-image POSITIVE CONTROL,
                                                     window_len=8, caches window_span_days
                             pipeline.py cube -> saved .npz (pooled/grid/variants/
                             grid_clear_frac/window_span_days) + per-cube mask cache
encoders/manifest.py         (cube, frame) manifest. TWO TIME AXES, and they are
                             not the same axis: original_axis_index counts
                             ACQUISITIONS and is the embedding join key;
                             daily_axis_index is the position on the cube's
                             ORIGINAL DAILY axis, from the frame's timestamp, and
                             is what horizons and weather windows are defined on.
                             Also pixel_bbox, per-cube AND per-grid-cell
                             landcover_stratum (esawc_lc), per-cell elevation
                             (cop_dem), and in-cube E-OBS weather (8 vars) joined
                             on daily_axis_index. assert_weather_join re-derives
                             that join from the cubes and refuses a disagreement
data/paths.py                phase-scoped artefact dirs: phase_dir/describe_phase/
                             reset_phase. data/raw is shared and never reset
probes/p1_appearance.py      P1, the appearance sanity probe: month / season
                             from ONE frame's frozen embedding. Reads the Phase
                             1.2 .npz and nothing else -- no encoder is even
                             imported. SIX feature sets -- pooled, grid_cell
                             (PRIMARY), raw_pooled, raw_rgb_only (the
                             BAND-MATCHED baseline: the networks are RGB-only),
                             raw_nir_ndvi, and degenerate (retention only, no
                             image) -- x 4 estimators x 3 fold modes, nested CV
                             for the regularisation strength. FeatureBlock binds
                             X / row_idx / y so they cannot be sliced apart.
                             Chance is DERIVED from the realised class
                             distribution, never hard-coded; margin_over_control
                             and margin_over_band_matched are the columns to
                             read. Also figure1(), the latent clock
probes/p4_ceiling.py         P4, the weather-attributability ceiling: how much of
                             the post-climatology NDVI anomaly is explainable
                             from weather alone. Reads CUBES only -- no
                             embedding, no weights. TWO STAGES: Stage A uses an
                             explicitly named PROXY climatology
                             (doy_climatology_within_fold, fitted on TRAINING
                             cubes inside each fold -- the signature takes a
                             training index set, so the nested leak is prevented
                             by the type rather than by discipline) and does NOT
                             produce H1; Stage B uses the real leave-year-out
                             data/climatology.py under mode "crossed" and is
                             DEFERRED, explicitly, until the seasonal split.
                             3 targets x 3 fold modes x 3 estimators x 2 feature
                             sets x 5 model kinds. FOUR mandatory controls --
                             observation-process, day-of-year sanity, both
                             jointly, and a permutation null. The headline is
                             margin_over_control, never the raw R-squared
probes/p2_deltas.py          P2, dynamics in deltas. TWO PARTS. Part A is gate
                             K2, the reconstruction floor: can a ridge read
                             CURRENT NDVI out of E(frame_t) at all? The verdict
                             is recorded per encoder and a failing encoder is
                             marked "audited: lossy" and excluded from P3 --
                             but only if k2_separable, the PAIRED per-fold
                             interval, excludes zero. TWO verdicts, because
                             raw_features holds NDVI_mean..NDVI_p90 and IS the
                             target at the matched level; read
                             k2_verdict_band_matched. Part B is the delta probe:
                             ||E(t+1)-E(t)|| and a linear read-out of the raw
                             difference, against the SIGN and MAGNITUDE of
                             COMMON-MASKED NDVI change (pixels valid in BOTH
                             frames, from the 1.2b mask cache -- differencing
                             two per-frame means compares two different pieces
                             of ground). The gap is DAYS on daily_axis_index,
                             never original_axis_index, and
                             assert_gap_axes_disagree proves that on real pairs
                             at run time. FOUR controls -- gap-length-alone
                             (P2's degenerate control; it beat every encoder on
                             magnitude at 20 cubes and collapsed to +0.063 at
                             115 -- a control can be a small-sample artefact
                             too), retention, the raw-pixel delta, and the
                             band-matched raw delta -- with the control's value
                             carried on EVERY row so no filtered view can lose
                             it. The multi-image encoder shares up to 7 of 8
                             source frames between consecutive embeddings, so it
                             is reported, flagged si_comparable=False, and
                             excluded from the structural-hypothesis ranking
probes/cv.py                 THE split definition. Six modes over the manifest,
                             each yielding (train_idx, test_idx) row positions:
                               cube          DEFAULT, GroupKFold on cube_id
                               crossed       cube AND year held out jointly
                               year          raises on one year
                               tile          raises on one tile
                               spatial_block within-tile substitute for tile
                               temporal      P3 robustness variant only
                             Every mode refuses if one cube would land on both
                             sides, whatever mode asked. join_embeddings holds
                             the (cube_id, original_axis_index) == (cube,
                             kept_idx) contract and carries window_span_days
scripts/inventory.py         list every file and folder, classify each movable
                             unit by the phase that owns it. Only looks
scripts/restamp_cache.py     diagnose an embedding cache and re-stamp files
                             that are COMPLETE but predate the schema stamp
                             (window_span_days landed in a1a6a12, the stamp in
                             f4ed234, a later commit). Refuses anything missing
                             a field; invents nothing. Also reports Drive
                             duplicates. DRY RUN by default
scripts/organise_phases.py   file those units into the per-phase Drive layout.
                             Imports the classifier rather than re-deriving it.
                             DRY RUN by default; data/raw is never moved
notebooks/organise_drive.ipynb  the SAME job as the two scripts, but
                             self-contained for Colab: it imports nothing from
                             the project, because a tool that reorganises the
                             folder a checkout lives in cannot depend on that
                             checkout existing. Its inline copy of the
                             classifier is pinned against scripts/inventory.py
                             by tests, so the two cannot drift
tests/                     test_ndvi.py was written before data/ndvi.py existed
notebooks/phase1_1_data_toy_load.ipynb
notebooks/phase1_2_encoders.ipynb
notebooks/phase1_3_cv.ipynb
notebooks/phase1_4_p1_appearance.ipynb
notebooks/phase1_5_p4_ceiling.ipynb
notebooks/phase1_6_p2_deltas.ipynb
notebooks/phase1_7_scaled_encoding.ipynb   builds the 115-cube frozen-encoder
                               cache. Computes NO result; the only notebook in
                               the project that wants a GPU
notebooks/runs/                the executed exit-test runs, outputs kept, as
                               evidence. Never re-run in place; excluded from
                               the Colab bundle by make_zip.sh
RUNBOOK.md                 Colab walkthrough: folders, restarts, expected output
docs/HANDOFF_P3.md         ONE PAGE, read first: what the next phase inherits,
                           what changed under it, the findings that constrain
                           what it may claim, and the traps with reasons.
                           docs/HANDOFF_P2.md is the previous one, still
                           accurate except where P3's supersedes it
docs/DECISIONS.md          why the project is shaped this way. Append-only
docs/runs/                 verbatim stdout of script runs, kept as evidence --
                           the same role notebooks/runs/ plays for phases
log.md                     measurements and adopted definitions
```

## Conventions

| thing | convention |
|---|---|
| band order | `S2_BANDS = ("B02", "B03", "B04", "B8A")`, index 2 red, 3 NIR |
| `values` | `(T, C, H, W)` float32 |
| `mask` | `(T, H, W)` bool, True means VALID or clear |
| `timestamps` | `(T,)` `datetime64[ns]`, strictly increasing, irregular |
| masked NDVI | `NaN`. Never 0, never silently dropped |

## Run

```bash
pip install -r requirements.txt
python -m pytest tests -q
python -m data.diagnose
python -m data.download_greenearthnet --out data/raw --n 20 --tile 32UNU
```

## Data

Pre-processed GreenEarthNet minicubes, tile `32UNU` (Allgäu / Upper Swabia),
the closest Alpine-foreland tile the dataset contains. There are no Munich cubes
to find: `32UPU`, which holds Munich, is not in GreenEarthNet.

Each cube is 128 x 128 px at 20 m over a 150-day window in 2018, about 29
Sentinel-2 acquisitions after empty days are dropped. All cubes in the tile are
from 2018, so there is no interannual signal to probe. That bounds Phase 1.2 to
within-season dynamics, which is the EarthNet benchmark's own setup.

`NDVI_VERBOSE=1` makes `ndvi()` print the shapes of every call.

For Colab, follow [RUNBOOK.md](RUNBOOK.md).

## Phase status

- 1.1 data toy-load: `ndvi()` unit test green, loader and downloader in place.
- 1.2 frozen encoder embeddings: **DONE**, superseded by the 2026-08-04
  five-encoder run below. Originally four wrappers at D = 35 / 768 / 768 / 1024
  and 80 embedding files; 264 retained frames each; frame selection
  at clear-fraction > 0.5 with the exact fraction stored alongside every
  embedding; peak GPU 1.20 GB at T=290. No quality comparison happens in this
  phase. P2 and P3 are unaffected by the single-year subset; the ceiling claim
  narrows to "within-season". See the `probes` package docstring.
- 1.2b re-encode: feature extraction now follows each model's published probing
  recipe with named variants for the ablation; patch-grid embeddings
  `[T_kept, 16, D_grid]` alongside the pooled vector; per-pixel masks cached so
  common-masking stays possible; `(cube, frame)` manifest with land-cover
  strata. Clay v1.5 deferred (not pip-installable), Prithvi-EO-2.0 dropped
  permanently (needs SWIR bands these cubes lack).
- 1.2c strata and control: land cover per GRID CELL (19/20 cubes are not
  single-class, 5 strata over 320 cells) so replication compares strata within
  one weather realisation; in-cube E-OBS (8 vars) and cop_dem joined into the
  manifest, which collapses P4's dependency on any external weather table; and
  `satlas_s2_swinb_mi_rgb`, a multi-image POSITIVE CONTROL -- the only encoder
  that can represent change at all. Clay v1.5 still deferred, not dropped.
  Its lookback is a variable number of days (measured: median 55, max 105,
  min 0 -- three times the 35-day nominal span for 8 frames at the 5-day
  revisit) and correlates with cloud, so every embedding stores
  `window_span_days` as a covariate. See [log.md](log.md) for the full
  measurement and `docs/DECISIONS.md` for two effects recorded but
  deliberately deferred to Phase 1.3 (MI's max-pool aggregation is lossy for
  trajectory; the built_up/bare_sparse strata are too thin to replicate on).
- **Current roster is FIVE encoders**, D = 35 / 1536 / 3840 / 1024 / 1024, all
  verified against real weights on a Colab T4 (2026-08-04, archived under
  `notebooks/runs/`). 100 `.npz` = 20 cubes x 5 encoders, 264 frames each.
  `dinov2_vitb14` ran for the first time in that pass -- its extraction had
  only ever been exercised through the synthetic dummy before.
- **Artefacts are phase-scoped.** `data/<phase>/<kind>/` via `data/paths.py`:
  `phase_dir("phase1_2", "embeddings")`, `describe_phase(...)` to see what is
  on disk, `reset_phase("phase1_2")` to clear exactly one phase before a
  re-run. `data/raw/` is shared and never cleared. Do this rather than
  `rm -rf`: `zipfile.extractall` overwrites but never deletes, so a stale
  artefact from an older mask definition survives into the next run and the
  resumable cache reuses it silently.
- 1.3 `probes/cv.py`: **DONE**. Six fold generators over the manifest, one
  leakage rule enforced inside every one of them: frames of the same cube never
  land on both sides, whatever mode asked. The cube/year collision (extreme
  split confounds them; a seasonal cube spans years) is resolved by the
  `crossed` mode, which holds cube AND year out jointly and so agrees with the
  climatology's leave-target-year-out protocol. On this subset `cube`,
  `spatial_block` and `temporal` run while `year`, `tile` and `crossed`
  correctly RAISE -- the exit test asserts the refusals rather than working
  around them. 261 tests pass, 5 skipped.
- **One Drive SUBFOLDER per phase**, from 1.3 on:
  `NeurIPS-CCAI-2026/{phase1_1,phase1_2,phase1_3}/`, each its own checkout,
  with the cubes at `NeurIPS-CCAI-2026/data/raw` because `data/raw` is shared
  and not a phase. A phase writes only under its own subfolder (via
  `data/paths.py`) and READS earlier phases' artefacts in place, so deleting a
  phase subfolder is a complete undo that cannot touch another phase. See
  [RUNBOOK.md](RUNBOOK.md).
- **K1 fired (2026-08-07).** EO-WM's authors replied with configs and a
  climatology file, but their Earthformer baseline is self-trained (not an
  official checkpoint) and the core EO-WM code is unreleased, so we drop
  "match EO-WM's published rows" as a validation surface and evaluate under
  our own protocol. See
  [docs/correspondence/2026-08-07-eowm-authors.md](docs/correspondence/2026-08-07-eowm-authors.md).
- 1.4 P1 appearance probe: **DONE** (2026-08-09; canonical run local CPU,
  independently reproduced on Colab to +/-0.003; archived under
  `notebooks/runs/`). `probes/p1_appearance.py` + a 432-row results CSV +
  Figure 1 under `data/phase1_4/`. 323 tests pass, 5 skipped.
  - **Realised, not assumed:** 8 months (April-November), 3 seasons (no DJF),
    **15** distinct time windows (not the 16 recorded earlier). Chance is 1/8
    and 1/3, derived from the labels.
  - **Given the same bands, the frozen networks beat hand-crafted features.**
    Every network encoder here is RGB-only, so the table carries a BAND-MATCHED
    baseline (`raw_rgb_only`, a column slice of `raw_features` -- no re-encode).
    On month/cube the networks clear it by +0.02 to +0.06. Full `raw_features`
    leads only because it additionally sees B8A and NDVI, and neither half
    alone approaches the combination.
  - **No encoder FAILS P1.** All five clear chance by a wide margin on both
    targets under all three fold modes, so the surprise worth reporting -- an
    EO model trained to appearance-invariance -- did not occur. P2/P3 licensed.
  - **The degenerate control is a competitor, not a floor.**
    `[clear_frac, window_span_days]` alone, no image, is at or above 42 of 120
    grid-cell rows. Where it sits is the finding: month under `cube`/`loco` is
    clean (0/20 and 0/20 at or below), season is not (10/20, 9/20), and under
    `spatial_block` season collapses (17/20). **Cite P1 as month under cube or
    LOCO, and cite the margin over the control**, not the distance from chance.
  - **The positive control works.** `satlas_s2_swinb_mi_rgb`, the only encoder
    that can represent change, is the only one that beats the retention control
    on season (+0.092 cube, +0.041 loco) -- and the worst on month. Exactly the
    signature its 0-105 day lookback predicts; flagged `si_comparable=False`
    and reported conditioned on that lookback.
  - **Two encoders cannot be ranked here.** `dinov2_vitb14` and
    `satlas_s2_swinb_rgb` land 0.387/0.386 locally and 0.389/0.389 on Colab, so
    the fold-mode rank agreement itself changes with a scipy version. Recorded
    as the sample-size limit it is: 20 cubes, 1 tile, 1 year.
  - Grid-cell rows (264 -> 4224) are the PRIMARY feature set. Regularisation
    grid-edge selection fell 34.7% -> 21.3% after widening, and the residual is
    saturation, not truncation (C=10^4 changes 0.1% of predictions).
  - The local embedding cache is the full **100 files**: `dinov2_vitb14` needed
    Python >= 3.10 for its `torch.hub` code, not a GPU (20 cubes, CPU, 42 s).
- **Three findings from Phase 1.5 that constrain every later phase**, stated
  here because they change what the project can claim:
  - **`H1` as originally scoped is not obtainable on this benchmark.** Tile
    32UNU has **no seasonal-split coverage** (the split covers 15 tiles, no
    Bavaria-area tile among them), so the leave-target-year-out climatology is
    not computable there at all — not deferred. And on a tile that *does* have
    it, `probes.cv.crossed_folds` clamps the fold count to the **number of
    years**; GreenEarthNet has four, so the interval is irreducibly wide
    (measured: 3.5× the cubes narrowed it 11%). Any honest leave-year-out
    evaluation on this dataset inherits a 4-sample interannual denominator.
    We report the **within-season proxy ceiling** instead, and say why.
  - **The 20 original cubes are from the `train` split, not `extreme`.** The
    real extreme split is 32UMC/32UNC/32UPC/32UQC. Nothing measured changes —
    every property actually used was verified from the files — but
    `docs/specs/phase1_3_cv.md` reserved "the 20 extreme cubes" for P3 on a
    false premise, and the real extreme split is untouched.
  - **Two latent bugs in `encoders/manifest.py`, both invisible on this
    subset.** E-OBS was joined on the *acquisition* axis into *daily*-axis
    arrays (0 of 264 rows carried their own day's weather); and `year` came
    from the cube *filename's* window-start rather than each frame's timestamp,
    which had silently **blocked P4's Stage B on every tile since Phase 1.3**.
    Both fixed and pinned by regression tests verified to fail against the old
    code. **P2 inherits the fixed manifest** — see the handoff below.
- 1.5 P4 weather-attributability ceiling: **Stage A DONE, Stage B DEFERRED**
  (2026-08-10; local CPU, 3.0 min). `probes/p4_ceiling.py` + a 270-row results
  CSV under `data/phase1_5/`. 382 tests pass, 5 skipped.
  - **A manifest defect was found and fixed first.** The in-cube E-OBS columns
    were indexed with `original_axis_index` (which counts ACQUISITIONS, ~29 per
    cube) into arrays on the DAILY axis (~150 steps), so **0 of 264 rows carried
    the weather of their own day** — offset 4–122 steps, median 53;
    mean-temperature MAE 6.26 K. Nothing could have caught it from inside the
    manifest: the values were finite, in range and internally consistent, and
    simply belonged to another day. `daily_axis_index` and
    `assert_weather_join` fix and pin it; `original_axis_index` is unchanged, so
    Phase 1.2–1.4 are unaffected.
  - **Stage A does not produce H1, and says so on every row.** The 20 cubes are
    single-year, so a leave-target-year-out climatology is not computable;
    Stage A uses a within-season PROXY, labelled as such in the CSV's
    `climatology_def` column. `data/climatology.py` still raises
    `SingleYearError` and was not weakened. **Stage B is deferred, explicitly,
    and the code refuses to let Stage A be relabelled as it.**
  - **The ceiling is not measurable at this sample size.** The fold-clustered
    95% CI includes zero for all 54 weather rows. Primary cell
    (`cube_mean`/`cube`/linear/8 vars): weather **+0.066 [−0.159, +0.291]**
    against an observation-process control of **+0.028**, so the headline
    **`margin_over_control` = +0.038**. Effective n is **20 CUBES**, not 264
    frames and not 4195 cells, and it is on every row of the CSV.
  - **33 of 54 weather rows sit at or below the retention control** — P1's
    finding propagating exactly as predicted. Cite the margin, never the R².
  - **Capacity hurts.** The small MLP scores −2.6 to −17.9 and loses to its own
    permutation null; only the linear model is positive on the primary target.
  - **The day-of-year sanity control chose the climatology's smoothness.** At 2
    harmonics, day-of-year alone still explained 4.0% of the "anomaly" (10.6% of
    the 90th percentile) and the weather number was inflated to match; 4
    harmonics is the lowest order that zeroes it on all three targets, and
    raising the order only ever LOWERS the headline, so the selection is
    conservative.
  - **36 days of year, one orbit lattice.** All 264 rows satisfy `doy % 5 == 2`
    and 261 share a date with another cube, so ~63% of a typical weather feature
    is recoverable from the date alone — temperature/pressure/radiation vary
    9–19% across cubes within a date, **precipitation 77%**. That is why only
    the *linear* day-of-year control is a sanity check, and why precipitation
    does nearly all the cross-cube work here.
  - **The 5-variable EO-WM subset is worse than the full 8** (−0.044 vs +0.038
    on the primary cell). Reported for commensurability, not as our number.
- 1.5b P4 scaled to **115 cubes** (2026-08-11, 37 min; `scripts/scale_p4.py`,
  cubes under `data/scaled_32UNU/` so `data/raw` stays the 20-cube set Phase
  1.2's cache is keyed to). **17 of 54 weather rows now exclude zero, against
  0 of 54 at 20 cubes.**
  - **The first measured ceiling in this project.** `cell_mean` under HGB:
    **+0.130 [+0.063, +0.197]** (cube) and **+0.085 [+0.007, +0.162]** (LOCO),
    clearing the observation control by +0.137/+0.109 **and** the day-of-year
    control by +0.094/+0.086. So: weather explains ~0.09–0.13 of the
    within-season post-climatology anomaly at grid-cell level.
  - **Cite `cell_mean`, not `cube_p90`.** HGB's larger +0.320 on `cube_p90` is
    four fifths day-of-year (its own DOY control is +0.256) — the 36-date orbit
    lattice is unchanged by adding cubes, so a flexible learner still fits a
    per-date mean. 16 cells share a date, which is why the cell-level target
    resists it.
  - **The estimator ordering reversed, and that is a sample-size result.** At 20
    cubes "capacity hurts, only linear is positive"; at 115 HGB is strongest and
    linear is weak. Boosted trees needed the data. The earlier conclusion is
    superseded, not contradicted.
  - Unchanged: `spatial_block` kills everything; the MLP is still catastrophic
    (now confidently wrong, excluding zero on the wrong side in 8 rows); the
    permutation control stays negative (max −0.007); the retention confound is
    **not** a small-sample artefact (mean across cells flat at −0.087 → −0.092);
    Stage B still correctly deferred. The linear DOY control is clean at 0.011,
    so the H=4 harmonic order holds at 5.75× the data.
- 1.7 scaled embedding cache: **DONE** (2026-08-11, Colab GPU).
  `notebooks/phase1_7_scaled_encoding.ipynb` builds
  `data/scaled_32UNU/{embeddings,masks}/` — 115 cubes x 5 encoders, 1580
  retained frames, the same cube set Phase 1.5b measured P4's ceiling on. It
  computes **no result**; it removes the sample-size excuse.
  - **Why it is a separate phase.** P4 scaled trivially because it reads no
    embeddings. P1/P2/P3 read the Phase 1.2 cache, which exists for exactly 20
    cubes — so scaling them means re-running four real networks. This is the
    only GPU-bound work in the project; every probe stays CPU-only.
  - **`data/phase1_2/` is never written to.** It is keyed to 20 cubes and every
    published result must stay reproducible from it. The scaled cache is a
    second, independent directory beside the shared cubes.
  - **Step 10 is the load-bearing check.** The 20 original cubes are a strict
    subset of the 115, so re-encoding them tests reproduction against a
    published cache: frame selection, timestamps and clear fractions must be
    **bit-identical** (they come from the cube, not the network); embeddings
    must agree to a stated 1e-3 (float32 on different hardware — P1's Colab
    reproduction moved scores by ±0.003). Verified locally on a 3-cube subset
    with the four encoders that build on Python 3.9: **exact, 0.0 difference**.
  - **Python 3.9 silently loses one encoder.** `dinov2_vitb14`'s `torch.hub`
    code uses `X | None`; the other four build and the cache gets a hole. Step 3
    asserts >= 3.10 up front.
  - P1 and P2 need **no code change** to consume it — both already take
    `emb_dir` / `mask_dir`.
- **1.6b P2 re-run at 115 cubes** (2026-08-11, `scripts/scale_p2.py`, 100 min
  on 7 workers; 600-row CSV at `data/scaled_32UNU/p2_scaled_results.csv`).
  **Two of the 20-cube headline claims did not survive, and both were the
  confident ones.**
  - **Gate K2 became a RANKING.** Three of four networks now *separably* beat
    the hand-crafted baseline on the paired per-fold difference: satlas SI
    **+0.713** `[+0.094, +0.156]`, imagenet **+0.687** `[+0.036, +0.162]`,
    dinov2 **+0.649** `[+0.021, +0.101]`, against raw_features +0.588. Every
    interval collapsed — DINOv2's by 6×, ImageNet's by 11×.
  - **The P3 exclusion is RETRACTED.** `satlas_s2_swinb_mi_rgb` went from
    separably lossy (−0.363) to not separable (−0.069) and passes band-matched.
    **No encoder is excluded from P3.**
  - **The magnitude finding was wrong, and the control was the artefact.** The
    gap-length control fell from +0.209 to **+0.063 `[−0.004, +0.130]`**. So
    the corrected result is not "encoders win" but **"nobody recovers
    magnitude"** — every ρ is +0.06 to +0.12 and three of four encoders flip
    margin sign across fold modes.
  - **SIGN replicates and is the phase's real result.** Margins +0.44 to +0.72,
    moving by less than 0.05 across all three fold modes. **`spatial_block`
    does not kill it** — the only probe in the project where that is true.
  - **The structural hypothesis is now determinable and REFUTED.** Identical
    ordering under all three fold modes (`raw_features > dinov2 > satlas SI >
    imagenet`), so `order_stable=True, supported=False`.
  - **The cache was verified before any number was read from it.**
    `assert_caches_agree` on the 20 shared cubes: frame selection, timestamps
    and clear fractions **bit-identical**; `raw_features` (pure numpy) differed
    by exactly 0 and the four networks by 5e-5 to 1.5e-4 — GPU-vs-CPU float
    jitter and nothing else.
- 1.6 P2 dynamics in deltas: **DONE** (2026-08-11; local CPU, 8.6 min).
  `probes/p2_deltas.py` + a 600-row results CSV and a pixel-survival table under
  `data/phase1_6/`. 0 failed, 5 skipped. Full per-pair detail in
  `data/phase1_6/logs/p2_run.log`; stdout keeps shapes, verdicts, controls and
  headlines.
  - **Gate K2 is cleared, and NO single-image encoder is excluded from P3.** At
    the primary cell (`cube_mean` from `grid_cell`, cube folds): satlas SI
    **+0.545**, imagenet **+0.521**, raw_features **+0.440**, dinov2 **+0.435**,
    band-matched floor **+0.417**, retention control **−0.129**. The only
    `audited: lossy` verdict that is *separable* from the baseline on the paired
    per-fold difference is `satlas_s2_swinb_mi_rgb`, the multi-image control.
  - **The verdict is recorded twice, because `raw_features` contains the
    target.** It carries `NDVI_mean`..`NDVI_p90`, so at the MATCHED level it IS
    the answer (`cell_mean` from grid features **R² +1.000**; `cube_mean` from
    pooled **+0.9998**). The primary configuration is not matched and is a fair
    gate. **Read `k2_verdict_band_matched`**, against the RGB-only slice.
  - **A lossy verdict is not a rejection, and K2 is a floor check rather than a
    ranking.** `k2_separable` — the *paired* per-fold interval on (encoder −
    baseline) — spans zero for **every** single-image encoder: satlas SI's
    nominal +0.105 lead is `[−0.118, +0.327]`, DINOv2's −0.006 is
    `[−0.142, +0.131]`. Only the MI control separates (`[−0.592, −0.134]`). So
    the gate says "nothing is catastrophically lossy" and nothing more. Dropping
    DINOv2 — the strongest encoder on the whole delta probe — on a 0.006 point
    estimate would have been dropping on noise.
  - **The delta probe splits in two. SIGN yes, MAGNITUDE no.** Against the
    common-masked NDVI change (`cube_mean`/pooled/linear/cube): on **sign**,
    dinov2 **+0.536 [+0.397, +0.674]**, satlas SI +0.481, imagenet +0.458,
    against a gap-length control of **−0.118**. On **magnitude the
    gap-length-alone control at +0.209 beats every encoder**, dinov2's +0.174
    included. **Direction is in these representations; rate is not.**
  - **The gap is measured in DAYS and a run-time check proves it.** Three
    readings of "the gap" disagree on **244 of 244 pairs** — `gap_days` median
    10, `original_axis_index` (the embedding JOIN KEY) median 2, frames-between
    always 1 — a 5.0× factor, one Sentinel-2 orbit lattice.
    `assert_gap_axes_disagree` asserts the disagreement on real pairs before
    anything is fitted, and the tests feed it the *wrong* column and assert it
    refuses.
  - **Common-masking does not collapse, and survival is not monotone in gap.**
    244/244 pairs keep shared pixels (median 88.8% of 16384, min 27.2%), and
    survival is *worse* at 15 days (0.783) than at 30 (0.985) — a long gap
    exists because the frames between it were cloudy, leaving the endpoints
    unusually clear. Do not model availability as a function of horizon.
  - **The structural hunch is NOT DETERMINABLE.** DINOv2 ranks above satlas SI
    under `cube` and below it under `loco` and `spatial_block`; at cell level
    they are +0.4576 and +0.4596. **The same pair P1 could not rank.**
    `structural_hypothesis` returns `supported=None` rather than reading a
    verdict off whichever mode was looked at first.
  - **`spatial_block` does not kill this one.** Unlike P1 and P4, the sign
    margins survive it (satlas +0.480, dinov2 +0.464, imagenet +0.358).
- 1.6b P2 scaled to **115 cubes** (2026-08-11, 100 min; `scripts/scale_p2.py`
  against the Phase 1.7 cache). **Three of the four claims above changed, and
  the two that changed most were the confident ones.** K2 becomes a RANKING
  (satlas SI +0.713, imagenet +0.687, dinov2 +0.649, all *separably* above
  `raw_features` +0.588); **the MI exclusion is RETRACTED** (−0.069
  `[−0.192, +0.053]`, not separable) so **no encoder is excluded from P3**; the
  gap-length control on magnitude was itself a small-sample artefact and falls
  from +0.209 to **+0.063 `[−0.004, +0.130]`**, which makes the corrected
  finding *worse* for the encoders, not better (every |ρ| ≤ 0.12, 3 of 4 flip
  margin sign across fold modes); and the structural hypothesis becomes
  determinable and is **REFUTED** (`raw_features > dinov2 > satlas_SI >
  imagenet`, identical under all three modes). Only SIGN survived unchanged —
  dinov2 **+0.606 `[+0.546, +0.665]`**, margin +0.54, moving < 0.05 across
  modes. On sign the band-matched `raw_rgb_only` reaches **+0.695**, above every
  network. See [log.md](log.md) and `docs/DECISIONS.md`.
- 1.8 P3 forecastability probe: **DONE** (2026-08-12; local CPU, 41.6 min on 7
  workers). `probes/p3_forecast.py` + a **460-row × 99-column** results CSV under
  `data/phase1_8/` and beside the cubes at `data/scaled_32UNU/`. 496 tests pass,
  5 skipped. 4 horizons × 5 encoders × 3 fold modes × 5 baselines × 3 controls,
  3 aggregations, severity bins and a cube-clustered interval on every row.
  - **Forecasting works, and it clears every control.** `cube_mean`, ridge, cube
    folds, pooled out-of-fold R²: the best row reaches **+0.672 / +0.704 /
    +0.454 / +0.628** at Δ = 5 / 25 / 50 / 100 days, against an
    observation-process control that never exceeds **+0.056**, a horizon-alone
    control that is **negative at every horizon**, and a permutation null at
    −0.014 to −0.103. Best skill against persistence: **+0.605 / +0.549 /
    +0.459 / +0.573**. Effective n = **115 / 114 / 115 / 94 CUBES**.
  - **The hand-crafted rows are still not beaten.** Seven percentiles of the
    same three bands (`raw_rgb_only`, no NDVI column, no network) sit within
    **0.03** of the best frozen encoder at 5 and 25 days and are **ahead of it
    at 100 days**; encoders lead clearly only at 50 days (+0.11 to +0.15). Full
    `raw_features` + weather — legitimately autoregressive here, since the target
    is at t+Δ — **wins outright at 3 of the 4 horizons**. This is P2's
    band-matched result on a second target: *NDVI is forecastable from a frozen
    representation plus weather; frozen foundation models are not the best way
    to do it.*
  - **`spatial_block` does NOT kill it** — the second probe in the project to
    survive the strictest geography holdout. DINOv2 goes +0.554 → **+0.488
    `[+0.433, +0.543]`** at 100 days. Unknown going in; measured, not inherited.
  - **What `spatial_block` DOES kill is the proxy climatology** (−8.6, −1328,
    −2784 at Δ = 5, 50, 100). A tile-level day-of-year curve fitted on one
    geographic cluster does not transfer to another. That bounds the *baseline*,
    not the encoders, and it is another reason P4's proxy is still unvalidated.
  - **The extreme/dynamic subset reverses the reading.** At 5 and 25 days every
    network is NEGATIVE against persistence in `extreme_low` and `high`; at 50
    and 100 days they turn positive on `extreme_low` (+0.17 to +0.72) and go
    negative on `extreme_high` (to −3.14). **No row beats persistence on both
    tails at any horizon.** The overall number alone would have hidden that.
  - **The window boundary is a result.** Rows retained fall **518 → 489 → 450 →
    196** as Δ goes 5 → 100, and **21 of 115 cubes contribute no 100-day pair at
    all**: the median cube covers 135 days of retained frames. The ±3-day
    tolerance does no work — on a 5-day orbit lattice it accepts exact matches
    only, and ±2 selects the identical 1653 rows.
  - **Three cloud-contaminated frames set the short-horizon R².** Three frames of
    1580 carry a midsummer cube-mean NDVI *below zero* at clear fractions of
    0.59–0.63 — cloud that both the clear-fraction filter and the per-pixel mask
    passed. The worst 1% of rows carry **71%** of the persistence sum of squares
    at 5 days. Nothing is dropped: `sse_share_top1pct` and a median absolute
    error travel on every row, and the fix belongs in the *shared*
    `p4_ceiling.cube_frame_targets`, which P2 and P4 read too.
  - **The MLP is unusable at this width** (a k=3 DINOv2 context is 11 520 columns
    against 158–416 training rows; D/n up to **73**, pooled R² −324). Ridge, whose
    penalty is α = D by rule, is fine at the same width. Every row carries
    `d_over_n_train`, so p ≫ n is a measured property rather than a caveat.
  - **The multi-image encoder leads the networks at 3 of 4 horizons, and that is
    a lookback result**, not a representation result: its ONE embedding at t
    already pools up to 8 retained frames (0–105 days, median 55) while the
    single-image encoders get 3 frames spanning 10–20. Flagged
    `si_comparable=False` on every row; `context_block` REFUSES a 3-frame stack
    for it.
  - Every climatology row is labelled **proxy, not Stage B** in
    `climatology_def`. It is not H1's number and must never be quoted as one.
  - **Convergence page:** [docs/HANDOFF_CONVERGENCE.md](docs/HANDOFF_CONVERGENCE.md)
    — P1/P2/P4/P3 × (prior, measured result, verdict), and the recommendation.
- **1.9 colour-infrared re-encode:** **DONE** (2026-08-12, Colab GPU).
  `notebooks/phase1_9_cir_encoding.ipynb` builds
  `data/scaled_32UNU/embeddings_cir/` — 115 cubes × 4 networks, the SAME frozen
  weights and the SAME frame selection, fed (B8A, B04, B03) instead of
  (B04, B03, B02). It computes **no result**; it removes the band-access
  confound. Frame selection is bit-identical to the RGB cache and the embeddings
  are materially different (max relative difference 0.29–0.95 per encoder).
  `raw_features` has no `_cir` twin and asking for one is refused with the
  reason: it already reads all four bands.
- **Tier 1 P3 re-run:** **DONE** (2026-08-12, local CPU, 173.7 min on 7 workers;
  `scripts/rerun_p3_tier1.py`, **1540-row × 153-column** CSV at
  `data/scaled_32UNU/p3_tier1_results.csv`, verbatim stdout at
  `notebooks/runs/2026-08-12_p3_tier1_32UNU_115cubes.txt`). 522 tests pass,
  5 skipped, **0 failed**. **Four corrections at once, each a column, so the old
  and new tables stay comparable** — and three of the 2026-08-12 headline claims
  did not survive.
  - **NO ENCODER SEPARABLY BEATS THE BAND-MATCHED BASELINE, at any horizon.** On
    the *paired* per-fold difference with a fold-clustered interval — the test
    P2 used for K2, not two overlapping marginal CIs — all nine encoder views are
    **separably BELOW** `raw_rgb_only` at Δ = 5 d (−0.141 to −0.267), 4 of 9 are
    below at 25 d, and none is above at any horizon. The only row separably
    *above* it is `raw_features` (+0.017 `[+0.008, +0.027]` at 5 d), and that is
    not a network: it reads all four bands plus seven NDVI statistics. The old
    table could only say "within 0.03"; the paired test gives the sign.
  - **Two numbers and no image beat every frozen network at 5 and 25 days.**
    `[NDVI(t), weather]` — 17 columns — reaches **+0.773 / +0.693 / +0.522 /
    +0.406** pooled out-of-fold R², above all nine encoder views at the two
    shorter horizons.
  - **Three cloud frames were carrying the old result.** Applying
    `p4_ceiling.cube_frame_targets`' `frame_plausible` screen drops 8 forecast
    rows of 518 at Δ = 5 d and moves **persistence from +0.169 to +0.690**: its
    R² was low because three frames of 1580 sat in its residual, not because
    NDVI moves in five days. Every gap the old table reported shrinks, and the
    ones the old headline rested on shrink most. **Seven of nine encoder views
    are now NEGATIVE against persistence at 5 days.**
  - **The band-access confound is real, small, and does not rescue the claim.**
    Colour-infrared vs its own RGB twin, paired: **16 of 48 comparisons
    separable, 22 of 48 with `_cir` ahead at all**. The two single-image
    networks that gain from NIR gain **+0.05 to +0.14 at 5 and 25 days** and
    lose it by 50; the multi-image encoder is made worse by NIR beyond 5 days;
    DINOv2 is worse with NIR short and better long. Not once is it enough to
    reach the band-matched baseline. So of the two readings the confound left
    open — "hand-crafted beats learned" vs "NIR beats RGB" — the evidence
    favours the first: **NIR helps, and it is not the explanation.**
  - **The penalty rule was measuring design width.** α = D spans **79 → 11536**
    across these views, a 146× range set by the embedding dimension. Both rules
    are in the table under `alpha_rule` and neither is deleted. Nested CV on the
    training fold (`p2_deltas.select_ridge_alpha`, imported, two poisoning tests)
    is worth +0.03–0.04 to the wide network rows and +0.044/+0.029 to the narrow
    band-matched row — it helps both sides and closes nothing.
  - **The shared base removes an unearned advantage worth +0.711.** Adding
    `NDVI(t)` moves `weather_only` by **+0.711** at 5 d and `raw_features` by
    **−0.003**, because `raw_features` already held it. It is worth +0.000 to
    +0.013 to the deep encoders — their embeddings already carry current NDVI,
    as P2's gate K2 said they must. Controls take no base, ever, and an assertion
    refuses a table where one does.
  - **Thirteen table invariants, re-checked ON THE CSV.** New this phase:
    `assert_separability_is_paired` (a verdict rebuilt from marginal intervals is
    refused — its fourth check catches the exact half-width a marginal rule
    produces), `assert_alpha_rules_present`, `assert_shared_base_present`,
    `assert_plausibility_screen_declared`, `assert_cir_twins_present`.
  - Unchanged: `spatial_block` does not kill the encoder rows and still destroys
    the proxy climatology; the MLP is unusable at this width; the multi-image
    encoder leads the networks at 50 and 100 d and is still `si_comparable=False`;
    the window boundary still costs 62% of the rows by 100 days.
