# Phase 1.3 — probes/cv.py, the leakage-safe split definition

The prompt below is the spec handed to the agent executing this block. It is
kept in the repo so the reasoning survives the chat it was pasted into.

---

I am executing a geospatial AI research project: probing frozen EO
foundation-model representations for dynamics/forecastability on
GreenEarthNet/EarthNet2021 minicubes. Tile 32UNU (Allgäu/Upper Swabia)
first: everything is prototyped on a small Alpine-foreland subset before
any scale-up. Act as a senior ML research engineer.

Standing constraints (true for every block in this project):
- No pretrained model is ever fine-tuned. All pretrained encoders are
  frozen: eval() and torch.no_grad(), always.
- Every array's shape is printed and asserted.
- Never re-implement a function that already exists in the repo - import
  it. Two functions are canonical and must never be duplicated or
  redefined: data/ndvi.py (the single definition of the target) and
  probes/cv.py (the single definition of leakage-safe evaluation splits).
- Any number produced outside probes/cv.py does not exist.
- Artefacts are phase-scoped. Write through data/paths.py
  (phase_dir/describe_phase/reset_phase), never to a hand-typed path.
- Commit before running make_zip.sh. The bundle is built from
  `git ls-files`, so an uncommitted file is silently absent from it. A
  Colab test-collection count that differs from local is a stale-bundle
  signal, not noise.
- Runtime for this block: Colab CPU is sufficient, <=12 GB. No GPU needed.

REPO STATE. Clone https://github.com/benj2o/terradreamer and read, in
this order: README.md, docs/DECISIONS.md, log.md, RUNBOOK.md. Then read
probes/cv.py, encoders/manifest.py, encoders/pipeline.py and
data/paths.py. Summary of what exists and works, verified 2026-08-04:

- data/ndvi.py, data/loader.py, data/download_greenearthnet.py
- data/paths.py - phase-scoped artefact dirs. reset_phase("phase1_3")
  clears exactly one phase; data/raw is shared and never cleared.
- data/climatology.py - GreenEarthNet leave-target-year-out; raises
  SingleYearError on a single year
- encoders/ - FIVE frozen wrappers, all verified against real weights on
  a Colab T4:
      raw_features            D=35     not a network
      imagenet_vit_b16        D=1536   concat(cls_last, patch_mean_last)
      dinov2_vitb14           D=3840   DINOv2's published linear-probe recipe
      satlas_s2_swinb_rgb     D=1024   single-image, EO-native
      satlas_s2_swinb_mi_rgb  D=1024   MULTI-IMAGE positive control, window_len=8
- data/phase1_2/embeddings/ - 100 .npz = 20 cubes x 5 encoders. Keys:
  embeddings [T_kept, D] float32, grid [T_kept, 16, D_grid] float16,
  grid_clear_frac [T_kept, 16], clear_frac [T_kept], kept_idx [T_kept],
  timestamps [T_kept], window_span_days [T_kept], variant__* , encoder, cube
- data/phase1_2/masks/ - 20 .npz, per-cube per-pixel valid masks, cached
  so probe-side common-masking stays possible
- 264 retained frames per encoder; T_kept min 10, median 13, max 16
- 118 tests pass, 5 skipped (weight-downloading and MI batch-invariance
  tests, gated behind PHASE1_2_WEIGHTS=1)
- Conventions: values (T,C,H,W) float32; mask (T,H,W) bool True=VALID;
  timestamps (T,) datetime64[ns] strictly increasing, irregular;
  masked NDVI is NaN, never 0

THE MANIFEST ALREADY EXISTS. DO NOT REBUILD IT.

`encoders/manifest.py` provides `build_manifest(samples) -> DataFrame`,
one row per RETAINED (cube, frame). Import it. Columns, verified:

    cube_id, tile, year, timestamp, original_axis_index, day_of_year,
    pixel_bbox, clear_frac, landcover_stratum, landcover_dominant_frac,
    grid_landcover, grid_landcover_purity, grid_elevation_m,
    eobs_tg, eobs_fg, eobs_hu, eobs_pp, eobs_qq, eobs_rr, eobs_tn, eobs_tx

Three things about it that change how you write cv.py:

- It is built from the CUBES (data/raw/*.nc), not from data/phase1_2/
  embeddings/. Do not try to reconstruct it from the .npz files.
- `landcover_stratum` already exists and is derived from the in-cube
  `esawc_lc` (ESA WorldCover 10 m). There is NO external join and no
  lookup left to run. `grid_landcover` additionally gives 16 per-cell
  labels aligned with the embedding grid.
- `pixel_bbox` is a 4-tuple (row0, row1, col0, col1) parsed from the cube
  id. It is the spatial grouping key for spatial_block mode.

JOIN CONTRACT. cv.py yields integer indices into MANIFEST ROWS. A probe
joins a manifest row to an embedding row on
(cube_id, original_axis_index) == (cube, kept_idx). State this contract
in the module docstring and assert it holds wherever you touch both.

DATASET FACTS THAT DRIVE THIS BLOCK (audited, not assumed):
- Split structure:
    train    23,816 cubes / 85 tiles / 2017-2020 / single-year per cube
    iid       4,205 / 82 tiles / 2017-2020
    ood       4,202 / 15 tiles / 2017-2020
    seasonal  3,880 / 15 tiles / 2017-2020, MULTI-YEAR WITHIN ONE CUBE
              (~1,050 timesteps, years 2017/2018/2019/2020 in one file)
    extreme   3,972 /  4 tiles / 2018 ONLY  <- the current 20 cubes
- Planned use: prototype P1/P2/P4 on seasonal; keep the 20 extreme cubes
  as P3's severity/dynamic subset; scale up on train.
- Of 20,259 distinct train locations, 3,111 are revisited and 2,745 of
  those span different start years.

THE CENTRAL DESIGN PROBLEM THIS BLOCK MUST SOLVE:
On the extreme split, cube and year are perfectly confounded (all 2018),
so grouping by cube IS grouping by year. On the seasonal split, ONE cube
internally spans 2017-2020, so grouping by year means splitting WITHIN a
cube - which contradicts the rule that frames of the same cube must never
appear on both sides of a fold. The two rules collide.

Resolution to implement: a CROSSED holdout. A test fold contains only
(cube, year) pairs whose cube is unseen in train AND whose year is unseen
in train. This preserves both rules simultaneously and matches
GreenEarthNet's own leave-target-year-out protocol used in
data/climatology.py, so the CV is consistent with the climatology
baseline rather than fighting it.

Phase 1.3 - probes/cv.py

Goal: one leakage-safe split definition that every probe imports, so
evaluation honesty is defined in exactly one place and verified once.

Deliverable: probes/cv.py operating on the manifest above. Keep the
existing cube_years / assert_multi_year / year_groups / SingleYearError
and build around them; do not rewrite them.

Modes, each a generator yielding (train_idx, test_idx):
1. cube      - DEFAULT. GroupKFold with cube_id as the group. Handles
               spatial leakage at prototype scale.
2. crossed   - cube AND year jointly held out, per the design problem
               above. This is the mode P4 uses whenever the manifest
               contains more than one year. Must RAISE on a single-year
               manifest, naming the cube mode as the correct fallback.
3. year      - year as the group. Must RAISE with a clear, actionable
               message when the manifest contains a single year (the
               extreme split), directing the caller to the crossed or
               cube mode. Mirror the existing raise in
               data/climatology.py.
4. tile      - entire geographic tiles held out. Must RAISE on a
               single-tile manifest, same pattern.
5. spatial_block - prototype-scale substitute for tile mode: cluster
               cubes by pixel_bbox distance within a tile and hold out
               whole clusters. Document that this is a substitute and
               that true tile holdout is deferred to scale-up.
6. temporal  - train strictly before, test strictly after a cutoff
               timestamp. NOT a default; available as a robustness
               variant for P3 only, because it starves training data
               when cube grouping already holds.

WHAT IS USABLE ON THE CURRENT SUBSET, so nobody reads a correct refusal
as a bug: the 20 cubes are one tile (32UNU) and one year (2018).
Therefore `cube`, `spatial_block` and `temporal` run; `year`, `tile` and
`crossed` all correctly RAISE. That is the designed behaviour and the
exit test must assert it, not work around it.

Hard requirements:
- Every mode must REFUSE (raise) if frames of the same cube would land
  on both sides of a fold, regardless of which mode requested it. This
  check runs inside the splitter, not in the caller.
- A frame-level clear_frac filter, applied before splitting, defaulting
  to > 0.5 (matching what the embeddings were encoded under) and
  configurable upward. Probes must be able to apply a stricter cut using
  the stored clear_frac without re-encoding anything.
- k configurable, default 5. Also expose leave-one-cube-out, which is
  the honest choice at the 20-cube prototype scale.
- Carry `window_span_days` through to the caller wherever the manifest is
  joined to embeddings. The multi-image encoder's lookback is a variable
  number of days (measured: min 0, median 55, max 105) and correlates
  with cloud, so it is a covariate a probe must be able to condition on.
  cv.py must not silently drop it.
- Every mode prints, per fold: n_train rows, n_test rows, n_train cubes,
  n_test cubes, and the years present on each side.

If you add any stratified variant, note that the per-cell strata have a
thin tail - over 320 cells: cropland 127, tree_cover 99, grassland 88,
built_up 5, bare_sparse 1. Stratified logic must cover cropland /
tree_cover / grassland only, and report-but-exclude the other two.

Tests:
- Synthetic manifest with known cube / tile / year / bbox membership.
- Assert no cube_id appears on both sides of any fold, in EVERY mode.
- Assert no tile_id appears on both sides in tile and spatial_block modes.
- Assert crossed mode leaves both the cube and the year of every test row
  unseen in train.
- Assert year mode raises on a single-year manifest, tile mode raises on
  a single-tile manifest, and crossed mode raises on a single-year
  manifest - all three with readable, actionable messages.
- Assert the clear_frac filter removes exactly the rows it should,
  computed independently.
- Feed a manifest with duplicate (cube_id, timestamp) rows and confirm a
  loud failure rather than silent duplication across folds.
- Keep all 118 existing tests green.

Exit test: all runnable modes generate valid folds on a synthetic
manifest; the three raise paths fire; the same-cube check fires when
deliberately provoked; probes/cv.py imports cleanly against the REAL
manifest built from data/raw via encoders.manifest.build_manifest, and
the join contract to data/phase1_2/embeddings/ is asserted on at least
one real (cube, encoder) pair; notebooks/phase1_3_cv.ipynb runs clean end
to end.

Generate the code exactly to this spec. After the code, list the three
most likely silent failure modes and the print statement that would
expose each.
