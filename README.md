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
notebooks/runs/                the executed exit-test runs, outputs kept, as
                               evidence. Never re-run in place; excluded from
                               the Colab bundle by make_zip.sh
RUNBOOK.md                 Colab walkthrough: folders, restarts, expected output
docs/DECISIONS.md          why the project is shaped this way. Append-only
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
