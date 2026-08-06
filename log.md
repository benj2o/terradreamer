# Working log

Running record of measurements and adopted definitions. Reverse chronological.
Decisions and their rationale live in [docs/DECISIONS.md](docs/DECISIONS.md).

## 2026-08-05: the Drive cache is unstamped, not incomplete -- a verified re-stamp

**Observed.** Phase 1.3 Step 10 stopped on Colab with the schema guard firing:

```
AssertionError: Copy of 32UNU_2018-05-03_..._dinov2_vitb14.npz was written with
cache schema v0, but this code expects v3.
```

Two separate faults in one line, and the guard was right about both.

*The filename.* `Copy of ` is Google Drive's own prefix for a duplicated file.
Step 10 took `sorted(glob(...))[0]` per encoder, so a stray duplicate could be
the file that loads -- selection by alphabetical accident.

*The version.* `window_span_days` landed in **a1a6a12** and the schema stamp in
**f4ed234**, a LATER commit. Artefacts written between the two carry every
field and no stamp. So "v0" here means UNSTAMPED, which is not the same as
incomplete, and the two have very different costs: a re-stamp takes seconds,
a re-encode takes a GPU run.

**Changed.** Three things, none of which weakens the guard.

1. `encoders.pipeline.inspect_encoded` reads a file WITHOUT enforcing the
   version, so the two cases can be told apart. `migrate_to_current` re-stamps
   a file only if every key in the new `REQUIRED_KEYS` is present AND the full
   `assert_encoded` set passes; it writes atomically through a temp file and
   then re-opens the result through the real `load_encoded`. `assert_encoded`
   is the invariant block extracted from `load_encoded`, so a migrated file is
   held to exactly the load-time standard rather than a subset.

   **Nothing is invented.** A missing `window_span_days` is NOT recomputed from
   the timestamps sitting beside it, and a test asserts that specifically. The
   hazard the stamp exists to prevent was silence -- a missing key read as
   merely absent; requiring every key and re-running every assertion is the
   opposite of that.

2. `scripts/restamp_cache.py`, dry-run by default: reports current /
   unstamped-but-complete / genuinely-incomplete / Drive-duplicate counts, then
   migrates only the second class.

3. Phase 1.3 Step 10 now **diagnoses the whole cache before joining anything**.
   It recognises a file as ours only when the cube half of its name is a cube in
   the manifest -- exact, and needs no locale-specific matching for `Copy of` /
   `Kopie von` / `Copie de`. One stale file used to halt the run at an arbitrary
   point with the rest undiagnosed; now every file is classified first and the
   remedy is printed once.

   Step 10 does not migrate anything itself: Phase 1.3 must not write into
   Phase 1.2's folder.

**Verified** on a copy of the real cache, both paths executed:

```
dry run    7 files -> current 2, unstamped-complete 3, incomplete 1, duplicate 1
--apply    3 stamped v3 and re-verified; the incomplete one left alone
after      5 load through the real guard, 2 still refused (correct)
           the MI file keeps window_span_days max 85 d through the migration
```

and Step 10's stale path fired on a scratch cache holding 3 unstamped, 1
incomplete and 1 `Copy of` file, reporting all four before raising.

Test count 198 -> **223 passed, 5 skipped (228 collected)**; 25 new tests,
including one parametrized over every `REQUIRED_KEYS` field to prove each
absence is refused individually.

## 2026-08-05: a full-tree listing in both notebooks

`organise_drive.ipynb` Step 5 and `phase1_3_cv.ipynb` Step 12: every folder and
every file from the absolute project root, no depth limit, no summarising. The
existing cells could not answer "did it land where I think it did" -- they
print movable units and rollups, both collapsed.

The Phase 1.3 copy is wired to that notebook's variables rather than pasted:
it derives the PROJECT root from `REPO` (the parent when the checkout sits in
a `phase1_*` subfolder, `REPO` itself in the flat layout), then adds two
Phase 1.3-specific sections.

`PHASE 1.3 OUTPUTS` checks the five files Step 11 wrote are present.
`READ-ONLY INPUTS` prints where the cubes and the Phase 1.2 embeddings were
resolved from, tagging any that sit `[inside this checkout]`. In the nested
layout NEITHER should carry that tag, and its absence is the visible evidence
that this phase read another phase's artefacts without writing into its folder
-- the property the per-phase layout exists for, checked rather than asserted.

Both layout branches exercised, not reasoned about:

```
nested   OK data/raw 2 cubes / OK phase folders / OK nothing loose
         -> "Layout matches the target."
         read-only inputs carry NO [inside this checkout] tag
flat     NOTE data/ also holds the checkout modules -- expected, not stray
         -> points at organise_drive.ipynb rather than reporting a fault
```

The flat branch matters: a naive root check would have reported 17 "stray"
entries on a working flat checkout and sent someone chasing a fault that does
not exist.

No new classification logic, so the drift guard is untouched and the count
stays 198 passed, 5 skipped.

## 2026-08-05: the organiser had a bootstrap bug; a self-contained Colab notebook

**Observed.** `python -m scripts.inventory` does not work on Colab, and the
reason is structural rather than a path problem: the scripts live INSIDE
`phase1_3_repo.zip`, so running them requires a checkout already extracted into
the very folder you are trying to reorganise. A tool that tidies the folder a
checkout lives in cannot depend on that checkout existing. Shipping the
reorganiser inside the thing it reorganises was the mistake.

**Changed.** `notebooks/organise_drive.ipynb`: four cells, imports nothing from
the project, works on a Drive folder with no checkout in it. Mount, inventory
(with SHA-256), plan, verify. `APPLY = False` until you flip it.

That requires the classification rules to exist in two places, which the repo
otherwise forbids. The exception is narrow and the drift is guarded rather than
trusted: the notebook's rules sit between sentinel comments,
`tests/test_scripts_organise.py` extracts and execs that block, and compares
`classify` and `destination` against `scripts.inventory` over a table of 18
paths and 7 (path, checkout_phase) pairs, plus the `SKIP` and `SHARED`
constants. Verified the guard fails when the copies disagree, by breaking the
notebook on purpose and watching two tests go red.

**Verified end to end** on a simulated Drive tree, notebook executed rather
than reasoned about:

```
before  11 units at the root, data/ holding three kinds at once
after    3 units: data/raw (shared), phase1_2/, phase1_3/
         47 files before and after, SHA-256 multiset IDENTICAL
         a second plan is empty -- idempotent
```

Test count 171 -> **198 passed, 5 skipped (203 collected)**; the 27 new tests
are the drift guard. Every documented expectation updated in step.

## 2026-08-05: project-tree tooling, and the test count moves to 171

Two scripts, `scripts/inventory.py` and `scripts/organise_phases.py`, added to
reorganise a Drive checkout into the per-phase layout. Not science: nothing
here produces a number.

The pair is split so that the thing which LOOKS and the thing which MOVES are
different programs, and the mover imports its classification from the lister
rather than re-deriving it -- so the plan you approve in the listing is the
plan that runs.

Verified on a simulated Drive tree (131 files, 20.3 MB, 3 cubes + 120
artefacts + a loose checkout + two bundles):

```
before   11 units at the root, data/ holding three different kinds at once
after     3 units: data/raw (shared), phase1_2/, phase1_3/
          131 files before and after, sha256 multiset identical
          a second run plans 0 moves -- idempotent
```

Four refusals are pinned by tests rather than documented: `data/raw` is never
filed under a phase (mirroring `data.paths.reset_phase`), no destination may
escape the root, an existing destination is never overwritten, and a checkout
moves whole or not at all with its owning phase stated explicitly. The last one
matters because a checkout is **not** classifiable from the filesystem -- it
looks identical whichever bundle produced it -- so the script asks instead of
guessing from mtimes.

`apply_moves` re-checks every refusal immediately before touching disk rather
than trusting the plan it was handed; a hand-forged move of `data/raw` raises
there too.

### The test count moved to 171 passed, 5 skipped (176 collected)

Was 146/5 (151 collected). The 25 new tests are `tests/test_scripts_organise.py`.
It rose again to 198/5 the same day when the drift guard landed; see the entry
above.
Every documented expectation has been updated in step -- RUNBOOK (both phases),
README, and the Phase 1.3 notebook's Step 5 -- because a collection count that
disagrees with the docs is the project's stale-bundle signal, and a signal
nobody trusts is worse than no signal.

The 2026-08-05 exit-test run recorded below measured 146/5 and is left as
measured. It is dated evidence, not a live expectation.

## 2026-08-05: Phase 1.3 exit test PASSED, local CPU, 20 cubes

`notebooks/runs/phase1_3_cv_2026-08-05_localCPU.ipynb`. CPU only, no weights,
nothing re-encoded. These are split-structure numbers; the probes that turn
folds into results are P1-P4.

### Gates

```
Step 5   146 passed, 5 skipped      (118 pre-existing + 28 new fold tests)
Step 6   manifest (264, 21), 20 cubes, tile ['32UNU'], year [2018]
Step 7   cube k=5 / LOCO / spatial_block k=5 / temporal all yield folds
Step 8   year, tile, crossed all RAISE -- the designed behaviour
Step 9   same-cube gate fires on a seasonal manifest; duplicates refused
Step 10  join contract asserted on one real pair per encoder
Step 11  5 files, 0.20 MB under data/phase1_3/folds/
```

### Fold structure on the 20-cube subset

```
cube, k=5        test 52-53 rows / 4 cubes per fold; every row tested once
LOCO             20 folds, test sizes 10..16 (= T_kept per cube)
spatial_block    5 blocks by pixel_bbox centroid, sizes [9, 3, 4, 3, 1]
temporal         cutoff 2018-08-15: 17 cubes train / 3 test, 65 frames DROPPED
```

**31 folds re-checked independently** in the notebook (not by the code under
test): no `cube_id` on both sides of any fold, in any mode.

The spatial_block sizes are uneven because the 20 cubes are not uniformly
spread over the tile: 9 of them fall in one contiguous region (tile-pixel rows
1145-2937, columns 3705-5369, against a full extent of 889-5241 by 377-5369),
and one cube sits alone far enough from every other to be its own block. That
is a property of the round-robin selection over 16 time windows, not of the
clustering; complete linkage on the same centroids is deterministic and
reproduces the same blocks on every run (pinned by a test).

The temporal claim above was checked exhaustively rather than by sampling:
over **every** day in the retained-frame span, the number of cutoffs that
separate complete cubes with nothing straddling is **0**.

### What temporal mode costs, measured

The 16 time windows overlap so heavily that **no cutoff separates complete
cubes**: for every candidate date, at least one cube has retained frames on
both sides. Cube-atomic assignment (whole cube to its majority side,
wrong-side frames dropped) at 2018-08-15 keeps 199 of 264 frames, i.e.
**65 dropped, 24.6%**, and leaves 3 test cubes. That is the starvation the
spec predicted, quantified: it is why temporal is a P3 robustness variant and
never a default.

### The three refusals, verbatim

```
year     SingleYearError  -> "Use 'cube' mode ... or 'crossed' once the
                             manifest spans years. Do not fall back to a
                             random split"
tile     SingleTileError  -> "use 'spatial_block' as the prototype-scale
                             substitute ... DEFERRED TO SCALE-UP"
crossed  SingleYearError  -> "cube and year are perfectly confounded, so
                             grouping by cube already holds the year out"
```

### Join contract, real (cube, encoder) pairs

`(cube_id, original_axis_index) == (cube, kept_idx)`, asserted on kept_idx,
timestamps and clear_frac for every encoder present:

```
imagenet_vit_b16         (14, 1536)   window_span_days 0 exactly
raw_features             (14,   35)   window_span_days 0 exactly
satlas_s2_swinb_rgb      (14, 1024)   window_span_days 0 exactly
satlas_s2_swinb_mi_rgb   (14, 1024)   window_span_days min 0 median 38 max 85
```

The MI figures reproduce the 2026-08-03 measurement on cube 1 exactly. The
join REFUSES an embedding without `window_span_days` rather than returning one
silently missing the covariate.

**Four encoders, not five, in this run**: `dinov2_vitb14` cannot be encoded
locally (its hub code needs Python >= 3.10; the dev venv is 3.9.6). The Colab
run sees all five. This also forced a local `reset_phase("phase1_2")` and
re-encode, because the local `.npz` cache predated `SCHEMA_VERSION` and
`load_encoded` correctly refused it -- the v3 guard from `f4ed234` doing
exactly what it was added for.


## 2026-08-04: Phase 1.2 exit test PASSED on the full five-encoder roster

Colab T4, 20 cubes, `notebooks/runs/phase1_2_encoders_2026-08-04_T4_5enc.ipynb`.
The headline: **`dinov2_vitb14` ran against real weights for the first time.**
Its extraction was rewritten in `b84c484` to follow DINOv2's published
linear-probe protocol and had only ever been exercised through the synthetic
dummy encoder, because the real-weight tests need Python >= 3.10 and the local
venv is 3.9.6. It was the most structurally complex wrapper in the roster and
the only one never watched running. It works.

### D per encoder, all verified against real weights

| encoder | pooled D | grid | variants |
|---|---|---|---|
| `raw_features` | 35 | 4x4 x 35 | - |
| `imagenet_vit_b16` | 1536 | 4x4 x 768 | cls_last 768, patch_mean_last 768 |
| `dinov2_vitb14` | **3840** | 4x4 x 768 | cls_last 768, cls_last4_concat 3072, patch_mean_last 768 |
| `satlas_s2_swinb_rgb` | 1024 | 4x4 x 1024 | - (no CLS: Swin is hierarchical) |
| `satlas_s2_swinb_mi_rgb` | 1024 | 4x4 x 1024 | - (positive control) |

### Exit test

```
Step 4   107 passed, 9 skipped   (116 collected -- see the caveat below)
Step 6   five encoders built frozen; DINOv2's 4-block CLS concat live
Step 7   all five return [14, D] on cube 1, identical kept timestamps
Step 8   20/20 cubes pass, worst prevalence 1.89e-05 vs 1e-04
Step 10  peak GPU 1.56 GB at T=290, batch_size=16   (budget 12 GB)
Step 11  100 .npz = 20 cubes x 5 encoders, 264 frames per encoder, 0 cached
```

`T_kept` min 10, median 13, max 16, summing to 264 per encoder -- identical to
the local CPU run and to Phase 1.1's clear-frame counts. Three independent
routes to the same 264.

MI `window_span_days` on cube 1: min 0, median 38, max 85 days.

### Caveat: the run collected 116 tests, local collects 123

The 7-test gap is exactly `tests/test_paths.py`, which was UNTRACKED when the
bundle was built. `make_zip.sh` lists files with `git ls-files`, so an
untracked file is silently absent from every Colab bundle. It is committed now.
The rule this re-proves: **commit before `make_zip.sh`**, and treat a changed
collection count as a stale-bundle signal, not noise.

Two breaks in the notebook were found afterwards and fixed in `31c481f`: a
Python string literal split across raw newlines in Step 11 (SyntaxError before
any encoding), and a Step 2 import of the since-removed `migrate_legacy`.
Neither could have been caught by the 2026-08-04 run, because that run used
the older bundle in which neither line existed yet.

## 2026-08-04: Phase 1.2c exit test PASSED on Colab T4, all five encoders

The re-run that closed the one piece of never-executed code. Numbers are shape
and integrity checks, not results: no quality comparison happens before
`probes/cv.py`.

Environment: Tesla T4, torch 2.11.0+cu128, Python 3.12.

### D per encoder, all five verified against real weights

| encoder | D (pooled) | grid | variants |
|---|---|---|---|
| `raw_features` | 35 | 4x4 x 35 | - |
| `imagenet_vit_b16` | 1536 | 4x4 x 768 | cls_last 768, patch_mean_last 768 |
| `dinov2_vitb14` | **3840** | 4x4 x 768 | cls_last 768, cls_last4_concat 3072, patch_mean_last 768 |
| `satlas_s2_swinb_rgb` | 1024 | 4x4 x 1024 | - |
| `satlas_s2_swinb_mi_rgb` | 1024 | 4x4 x 1024 | - |

**`dinov2_vitb14` ran against real weights for the first time here.** Its
extraction was rewritten in `b84c484` to follow the published linear-probe
protocol and could not be executed locally (its hub code needs Python >= 3.10;
the dev venv is 3.9.6), so until this run the most structurally complex wrapper
in the roster was also the only one never watched running. It produced exactly
the declared shapes: `cls_last4_concat` 3072, `pooled` 3840, `grid` (14, 16, 768)
off a 16x16 patch lattice.

### Gates

```
Step 4   111 passed, 5 skipped        (was 95/3 before 1.2b/1.2c)
Step 7   T_kept = 14/29 on cube 1, identical across all five encoders
Step 8   20/20 cubes, worst prevalence 1.89e-05 vs 1e-04 tolerance
Step 10  peak GPU 1.56 GB at T=290, batch_size=16   (budget 12 GB)
Step 11  100 .npz, 264 frames per encoder, T_kept min 10 median 13
```

Peak GPU rose 1.20 -> 1.56 GB because the ViT probe default is now 1536-dim
rather than 768. `window_span_days` on cube 1: min 0, median 38, max 85 days.

### One thing this run did NOT do, and the guard added because of it

Step 11 reported **100 cached, 0 encoded**: the artefacts already existed from a
prior execution, so this pass re-asserted them rather than recomputing them.
Re-assertion is real verification -- every one of the 100 files passed
`load_encoded`, including the fp16 grid check and the
`grid_clear_frac.mean == clear_frac` identity -- but it left one question open.

Encoder dimensionality does **not** prove a cache is current. The multi-image
encoder landed in `d58e98e` and `window_span_days` in `a1a6a12`, a later
commit, so a cache can hold MI files at exactly the right D and still predate
the covariate. `np.load` reports a missing key as simply absent, and
`load_encoded` returned `None` for it and continued. A probe would then read
`window_span_days`, find nothing, and quietly drop the confound control.

Fixed by stamping `SCHEMA_VERSION` (now 3) into every `.npz` and refusing to
load anything older, with the reset command in the message. Version 1 was
Phase 1.2, 2 added the grid, 3 added `window_span_days`.

## 2026-08-03: Phase 1.2c re-encode, and the MI lookback measured

Local CPU run, all 20 cubes. Four of the five encoders: `dinov2_vitb14` runs on
Colab only, because the `facebookresearch/dinov2` hub code evaluates PEP 604
unions at import and needs Python >= 3.10 (this venv is 3.9.6).

```
build              52 s      3 network wrappers, weights already cached
re-encode         150 s      CPU, 4 encoders x 20 cubes
embeddings       27.3 MB     80 .npz  (pooled fp32 + grid fp16 + variants
                             + grid_clear_frac + window_span_days)
per-cube masks    0.074 MB   20 .npz
manifest          264 rows   strata green
```

### window_span_days: the MI lookback is NOT a fixed window

264 MI embeddings, calendar days spanned by each 8-retained-frame window:

```
min 0   p25 25   median 55   p75 80   max 105
```

If every frame sat at the 5-day Sentinel-2 revisit, 8 frames would span **35
days**. The median is **55** and the maximum is **105**, three times nominal.
The spread between quartiles alone is 25 to 80 days.

This is the measurement that justifies caching it. The lookback is not a
nuisance of a few days around a nominal value; it is a covariate varying over
a 105-day range, and it varies **with cloud**, because a cloudier stretch drops
more frames and the same 8 retained frames therefore reach further back. Any
P2/P3 comparison involving the MI encoder that does not condition on
`window_span_days` is comparing embeddings built from different amounts of
history, with the difference correlated with weather. `min = 0` is the first
frame of each cube, which has no history to look back on.

Single-image encoders store `window_span_days = 0` exactly, an honest constant
rather than a NaN.

### Projection to seasonal scale, five encoders (~290 frames per cube)

```
  200 cubes    7.5 GB    ~689 min CPU
  500 cubes   18.8 GB   ~1722 min CPU
 1000 cubes   37.6 GB   ~3443 min CPU
```

At 1000 cubes the store is ~38 GB, comfortably past a default Drive allowance.
The grid is already fp16, so the lever remains cube count, not precision. GPU
cuts the time roughly an order of magnitude; the disk figure is
device-independent.

## 2026-08-03: Phase 1.2b dimensionalities, sizes and timings

Local CPU run over all 20 cubes of tile 32UNU. Clay deferred (see DECISIONS),
so three encoders here, not five.

### Feature variants and dimensionality

| encoder | variant | dim |
|---|---|---|
| `raw_features` | pooled (35 whole-frame stats) | 35 |
| | grid, per cell | 35 (x16 = 560) |
| `imagenet_vit_b16` | cls_last | 768 |
| | patch_mean_last | 768 |
| | **pooled** = concat of the two | **1536** |
| | grid, per cell (14x14 -> 4x4) | 768 |
| `dinov2_vitb14` | cls_last | 768 |
| | cls_last4_concat | 3072 |
| | patch_mean_last | 768 |
| | **pooled** = concat(cls_last4_concat, patch_mean_last) | **3840** |
| | grid, per cell (16x16 -> 4x4) | 768 |
| `satlas_s2_swinb_rgb` | **pooled** = GAP of final stage | **1024** |
| | grid, per cell (4x4 map, pooling is identity) | 1024 |

Satlas has NO CLS token and therefore no `cls_*` variant; Swin is hierarchical.

### Grid round-trip (the classic patch-token reshape bug)

`grid.mean(cells)` versus the whole-frame patch mean:

```
16x16 -> 4x4  divisible      max abs diff 5.96e-08   (DINOv2)
 4x4  -> 4x4  identity       max abs diff 5.96e-08   (Satlas, exact)
14x14 -> 4x4  NOT divisible  max abs diff 8.50e-03   (ViT-B/16)
```

The ViT gap is geometry, not a bug: 14 does not divide by 4, so
`adaptive_avg_pool2d` uses uneven bins and the cell mean is a *weighted* patch
mean. Pinned by `test_uneven_lattice_explains_its_own_mismatch`.

### Sizes and timings, 20 cubes / 264 retained frames

```
re-encode          52 s        CPU, 3 encoders, 20 cubes
embeddings         18.5 MB     60 .npz (pooled fp32 + grid fp16 + variants)
per-cube masks      0.07 MB    20 .npz, bool (T_kept, 128, 128)
mask compression   ~146x       vs packed bits; clear-fraction is bimodal
manifest           264 rows    strata: cropland 8, grassland 6, tree_cover 6 cubes
```

### Projection to seasonal scale (~290 frames per cube)

```
  200 cubes    4.1 GB     ~192 min CPU
  500 cubes   10.1 GB     ~479 min CPU
 1000 cubes   20.3 GB     ~958 min CPU
```

**Size the seasonal prototype against Drive before running it.** At 1000 cubes
the grid store alone is ~20 GB, which exceeds a default Drive allowance; the
grid is already fp16, so the lever is cube count, not precision. GPU will cut
the time roughly an order of magnitude; the disk figure is device-independent.

## 2026-08-03: Phase 1.2 exit test PASSED, Colab T4, 20 cubes

`notebooks/phase1_2_encoders.ipynb` run clean end to end in a fresh runtime.
These are shape and integrity numbers, not results: **no quality comparison
happens in Phase 1.2, and any number produced outside `probes/cv.py` does not
exist.** The embeddings are inputs to Phase 1.3, nothing more.

Environment: Tesla T4 15.6 GB, torch 2.11.0+cu128, torchvision 0.26.0+cu128,
Python 3.12.

### Embedding dimension per encoder

| encoder | what | params | D |
|---|---|---|---|
| `raw_features` | not a network | 0 | **35** |
| `imagenet_vit_b16` | torchvision ViT-B/16 | 85.8M | **768** |
| `dinov2_vitb14` | torch.hub DINOv2 ViT-B/14 | 86.6M | **768** |
| `satlas_s2_swinb_rgb` | SatlasPretrain Swin-B | 87.9M | **1024** |

All three networks reported `eval()=True, requires_grad=False` on every
parameter at construction, re-asserted on every `encode` call.

### Frame retention, 20 cubes

```
T_kept       min 10  median 13  max 16      (of T = 28-30)
frames total 264 per encoder, 80 cube x encoder pairs
distribution 10:1  11:1  12:4  13:6  14:4  15:3  16:1
```

`T_kept` reproduces Phase 1.1's `frames >50% clear` counts (min 10, median 13,
max 16) exactly, so neither the finite-mask correction nor anything else in
Phase 1.2 changed a frame's keep/drop decision. All four encoders returned
identical kept timestamps on every cube.

### Valid-pixel reflectance, 20 cubes

All 20 pass. Worst prevalence **1.89e-05** against the 1e-4 tolerance; 12 of 20
cubes have zero implausible valid pixels. The all-finite maximum reaches
**1.9817**, confirming Phase 1.1's 1.98 figure and confirming it stays behind
the mask: the valid maximum on that same cube is 1.7839, from 5 isolated
pixels.

### Other gates

- Unit tests **95 passed, 3 skipped** (the 3 download weights inside pytest).
- Malformed input refused loudly by all four wrappers: rank-3, channels-last,
  empty batch, plus the baseline refusing to run without the mask.
- Memory does not scale with T: **peak GPU 1.20 GB at T=290, batch_size=16**,
  against the 12 GB T4 budget. The seasonal split (~290 frames) fits with an
  order of magnitude to spare.

### Cross-machine reproducibility

The per-cube valid maxima and implausible-pixel counts from the T4 run match a
local CPU run of the same code exactly (0.8235/0, 1.7839/5, 1.4655/4,
0.7446/0, 1.4637/1, ...), as do `T_kept` and the 264-frame total. DINOv2 ran on
Colab; it is skipped locally only because `facebookresearch/dinov2` hub code
evaluates PEP 604 unions at import and needs Python >= 3.10.

## 2026-08-03: two encoder-input data properties, tile 32UNU, 20 cubes

Both found by Phase 1.2's own assertions firing on real cubes. Measured over
all 20 cubes: 17,340,401 valid pixels, 264 retained frames.

### Implausible reflectance among VALID pixels

| quantity | value |
|---|---|
| valid pixels with reflectance > 1.2 | **44** (2.537e-06) |
| worst single cube | 15 px, 1.89e-05 |
| cubes affected | 8 of 20 |
| max valid-pixel reflectance | 1.7839 |
| per band (B02, B03, B04, B8A) | 7, 7, 12, 18 |

They are **isolated singletons and 2-4 px clusters in 99.7-100% clear frames**,
not contiguous regions. At the worst pixel `s2_dlmask=0`, `s2_SCL=5` (bare
soil), while the legacy `s2_mask=1` (cloud). Bright across all visible bands
(B02 1.78, B03 1.63, B04 1.55): the signature of a specular target such as a
greenhouse or metal roof, of which the Allgau has many, or a sub-pixel bright
fragment.

Downstream impact, measured rather than assumed:

- NDVI at those pixels stays in **[-0.19, 0.72]**; NDVI over all 20 cubes stays
  inside [-1, 1]. A pixel bright in both B04 and B8A gives NDVI near 0, not a
  wild value.
- Recomputing the raw-feature baseline with those pixels masked moves no
  feature by more than **3.9e-04 relative** (worst: `NDVI_p10`).

**Adopted.** The check asserts on **prevalence, not the maximum**. A maximum
over 17M pixels is the most outlier-sensitive statistic available; the failure
worth halting for is cloud passing as clear over real area, which for even one
fully-clouded frame of a 13-frame cube would be ~7.7e-02. Tolerance set at
**1e-4**, about 5x the worst observed cube and ~770x below the smallest
systemic leak worth the name. The threshold of 1.2 for "physically implausible"
is unchanged.

### Pixels that are both mask-valid and no-data

| quantity | value |
|---|---|
| pixels mask-valid AND non-finite | **113** |
| non-finite pixels in retained frames | 117 (6.762e-06) |
| retained frames affected | 6 of 264 (2.3%) |
| worst frame's non-finite fraction | 0.0014 |

GreenEarthNet's clear-sky conjunction is computed from `s2_dlmask` and `s2_SCL`
and never consults the reflectance bands, so it can mark a no-data pixel clear.
`data.ndvi.ndvi` already guarded against this internally (`usable = mask &
isfinite(...)`); the encoder path did not, and a single NaN returns an all-NaN
ViT embedding.

**Adopted.** `encoders.frames.finite_valid_mask` ANDs the mask with
"every band finite", so the encoder path and the target path agree about which
pixels exist. Pixels only ever leave the valid set; nothing is filled. Frame
selection is unchanged in effect: `T_kept` is min 10, median 13, max 16, which
reproduces Phase 1.1's `frames >50% clear` counts exactly, so no frame changed
its keep/drop decision.

For the three network wrappers, which have no concept of a mask and cannot
consume NaN, non-finite pixels are replaced by a printed sentinel
(`NONFINITE_FILL = 0.0`) inside each wrapper. 6.8e-06 of band-pixels, at most
0.14% of any one frame. The raw-feature baseline needs no sentinel: its
statistics are NaN-aware.

## 2026-08-02: mask switch, s2_mask to s2_dlmask

Tile 32UNU, 20 cubes, 9,486,336 finite pixels.

Clear-sky fraction over all finite pixels, three definitions:

| definition | clear fraction |
|---|---|
| `s2_mask == 0` (legacy sen2flux/Sen2Cor lineage) | 0.4263 |
| `s2_dlmask == 0` (variable switch alone) | 0.4815 |
| `s2_dlmask == 0` AND `s2_SCL in (1,2,4,5,6,7)` (**adopted**) | 0.4570 |

The SCL conjunction removes a further 5.1% of dlmask-clear pixels.

**Correction to the figures quoted when this switch was requested.** The
0.366 -> 0.509 shift was measured on a single cube, the first one listed. Over
all 20 the shift is **0.4263 -> 0.4570**, a relative gain of +7.2% in usable
pixels, not +39%. Effective sample size per probe therefore rises modestly.

Per-cube under the adopted definition:

```
valid fraction    min 0.372  median 0.455  max 0.555
frames >50% clear min 10     median 13     max 16     (of T ~= 29)
NDVI valid frac   median 0.455
NDVI median       min 0.592  median 0.747  max 0.822
```

Snow: `s2_dlmask` has no snow class (0 clear, 1 thick cloud, 2 thin cloud,
3 cloud shadow). Snow is masked through the SCL half instead, since 11 is not
in the allow-list. Under the legacy mask snow was only 0.00054 of finite pixels
in this tile and window, so the practical effect here is negligible, but it is
the reason the conjunction is not optional.

Evidence for the lineage claim is in the cubes themselves, not only the paper:

- `s2_dlmask.attrs["description"]`: "Deep Learning Cloud Mask, trained by Vitus
  Benson on cloudSEN12, leveraging code from Cesar Aybar."
- `s2_mask.attrs["description"]`: "sen2flux Cloud Mask", with class 4 being
  "masked by SCL".

## 2026-08-02: adopted climatology definition

Per-pixel NDVI climatology, adopted verbatim in behaviour from
`model_pixelwise/climatology.py` in vitusbenson/greenearthnet:

> For a target year, pool NDVI from **all other years**, keep pixels where
> `s2_dlmask == 0` and `s2_SCL` is in `(1,2,4,5,6,7)`, linearly interpolate
> remaining gaps along time, group by day-of-year and average across years,
> then smooth with a 30-day centred rolling mean over a wrap-padded year and
> drop the 30 padding days at each end.

Two notes on our implementation in `data/climatology.py`:

- NDVI comes from `data.ndvi.ndvi`, which returns NaN where
  `|B8A + B04| < 1e-12`. GreenEarthNet instead adds `1e-8` to the denominator.
  The two agree everywhere the denominator is not near zero; ours declines to
  guess where it is.
- **Not computable on the current subset.** Every cube in 32UNU is from 2018,
  and the definition requires at least one year other than the target year.
  `ndvi_climatology` raises `SingleYearError` rather than inventing a
  within-year substitute. It becomes available at scale-up.

## 2026-08-02: year-grouped CV deferred

`probes/cv.py` keeps a year-grouped mode, but it raises `SingleYearError` when
the manifest contains one year, which is the case for the whole 32UNU subset.
The year-leakage check is therefore **deferred to scale-up**. Spatial grouping
is unaffected and remains the operative split for Phase 1.3.

The error message explicitly warns against falling back to a random split,
which would put the same season on both sides and produce exactly the inflated
number the year grouping exists to prevent.

## 2026-08-02: probe scope under the single-year subset

Recorded in the `probes` package docstring so it is read where it binds:

- **P2 (5-day deltas): unaffected.** The step is the S2 revisit, well inside a
  150-day cube.
- **P3 (horizons to ~100 days): unaffected.** A 150-day window holds a 100-day
  horizon with context to spare.
- **Ceiling claim: narrowed to "within-season".** No interannual or multi-year
  seasonality on this subset. Conventional for the benchmark, but it has to be
  written down rather than assumed.

**PCA latent clock.** Not computable from one cube: ~29 frames over ~150 days is
a fragment of an annual cycle, so the "clock" would be a line rather than a
loop. Adopted instead: pool embeddings across cubes spanning the tile's 16
distinct time windows (starts 2018-03-09 through 2018-07-07, aggregate coverage
2018-03 to 2018-12) and colour by month or season. The spread across windows is
the only seasonal axis available.
