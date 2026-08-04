# Working log

Running record of measurements and adopted definitions. Reverse chronological.
Decisions and their rationale live in [docs/DECISIONS.md](docs/DECISIONS.md).

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
