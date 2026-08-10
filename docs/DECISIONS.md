# Decisions

Append-only. One entry per decision, newest last. Each entry records what was
assumed, what the data or the literature actually showed, what changed, and the
commit.

Commits are the record a debugger reads. This file is the record a reviewer or
supervisor reads. Neither replaces the other.

Measurements live in [../log.md](../log.md).

---

## 2026-08-02: Data source, live extraction to pre-processed GreenEarthNet

**Assumed.** That building minicubes from Sentinel-2 with `earthnet-minicuber`
was a reasonable way to get 20 cubes, and that self-derived cloud masks were
equivalent to the dataset's.

**Observed.** A timing probe over 2 of 36 months measured 2.5 minutes, putting
20 full cubes at **14.7 hours**. The bottleneck is CPU, not network: the
cloud-mask U-Net runs at 2x upsampling on Colab's 2 vCPUs and is never moved to
CUDA. A tile-filtered pull of 20 pre-processed cubes takes **11 seconds and
67 MB**, a factor of about 4800. Separately, only the dataset's published masks
give numbers comparable with published baselines.

**Changed.** `data/download_greenearthnet.py` fetches cubes anonymously from
`s3.bgc-jena.mpg.de`, filtered to one MGRS tile.
`earthnet.download(..., limit=N)` cannot do this: `limit` slices a lexicographic
listing of the whole split, so it returns cubes from whichever tile sorts first.
`data/download_minicubes.py` is kept for locations the dataset does not cover.

`data/ndvi.py` and `data/loader.py` needed no changes: the cubes sit on a daily
grid where about 29 days carry an acquisition, and the loader's existing
drop-empty-timesteps logic produces the irregular axis the spec asks for.

**Commit.** `8e6285d`

---

## 2026-08-02: Tile, 32UPU (Munich) is not in the dataset

**Assumed.** That the prototype subset would sit on the Munich tile at
(11.55, 48.15), per "Munich first".

**Observed.** Of the 12 MGRS tiles intersecting a Bavaria bounding box, only
`32TPT`, `32UNU` and `33TUN` exist in GreenEarthNet. **`32UPU`, which contains
Munich, is absent.** `32UNU` (9.00-10.49 E, 47.76-48.75 N, Allgäu / Upper
Swabia) holds 192 cubes, sits in the same latitude band as Munich, about 135 km
west, with the same grassland, arable and forest mix. `32TPT` is nearer in
kilometres but is high Alps, a different biome, so proximity would have been the
wrong criterion.

**Changed.** Subset is tile `32UNU`. Scale-up generalises by land-cover strata,
not by city, so this is a label change and not a change to the science. Wording
across code and docs now names 32UNU. Munich survives in three sentences, each
stating the tile is absent, so nobody goes looking for cubes that do not exist.

**Commit.** `6e9d2f7` (wording), source switch in `8e6285d`

---

## 2026-08-02: Temporal extent, single year accepted

**Assumed.** A 2019-2021 span, roughly 387 acquisitions per cube, enough for
interannual dynamics.

**Observed.** Every one of the 192 cubes in `32UNU` is from **2018**, across 16
distinct ~150-day windows. After empty days are dropped each cube holds about
**29 acquisitions over ~150 days**. There is no interannual signal in the subset
at all, and no cube selection can create one.

**Changed.** Accepted; the live-extraction path is not reopened. Three
consequences are now enforced in code rather than left to memory:

1. **Climatology is not computable.** Adopted the published definition verbatim
   in behaviour; it is cross-year by construction.
   `data.climatology.ndvi_climatology` raises `SingleYearError` rather than
   substituting a within-year mean, which is a different quantity and on a
   single-season cube approximates a smoothed copy of the signal it would
   baseline.
2. **Year-grouped CV is deferred to scale-up.** `probes.cv.year_groups` keeps
   the mode and raises on a single-year manifest, naming the tempting wrong
   fallback (a random split, which would put the same season on both sides).
   Spatial grouping is unaffected and is the operative split for Phase 1.3.
3. **The PCA latent clock pools across cubes.** ~29 frames over ~150 days is a
   fragment of an annual cycle; a clock from one cube would be a line, not a
   loop. Pool across the 16 time windows, colour by month.

**Scope.** P2 (5-day deltas) and P3 (horizons to ~100 days) fit inside single
cubes and are unaffected. Only the ceiling claim narrows to **within-season**,
which matches the benchmark's own design. Recorded in `probes/__init__.py`.

**Commit.** `ebfc5d0`

---

## 2026-08-02: Cloud mask, s2_mask to s2_dlmask with the SCL conjunction

**Assumed.** That `s2_mask` was the mask to use, and that masking more was the
safer default for a target variable.

**Observed.** Two things, one of which inverts the safety argument and one of
which changes the task.

`s2_mask` is the Sen2Cor-lineage mask GreenEarthNet supersedes (Benson et al.,
CVPR 2024). The cubes carry the evidence themselves:
`s2_mask.attrs["description"]` reads "sen2flux Cloud Mask" with class 4 being
"masked by SCL", while `s2_dlmask.attrs["description"]` reads "Deep Learning
Cloud Mask, trained by Vitus Benson on cloudSEN12". Conservatism is not the
right axis: only `s2_dlmask`-based numbers are comparable with published
results, which is the whole reason for using this dataset.

The fix is **not** a variable swap. `model_pixelwise/climatology.py` in
vitusbenson/greenearthnet defines clear sky as a **conjunction**, keeping pixels
where `(s2_dlmask < 1)` AND `s2_SCL` is in `[1, 2, 4, 5, 6, 7]`. Taking
`s2_dlmask` alone would have been a third definition, matching neither the
legacy mask nor the published one. The SCL half is also what masks snow:
`s2_dlmask` has no snow class (0 clear, 1 thick cloud, 2 thin cloud, 3 cloud
shadow), and SCL 11 is excluded from the allow-list.

Clear fraction over 20 cubes, 9,486,336 finite pixels:

| definition | clear fraction |
|---|---|
| `s2_mask == 0` | 0.4263 |
| `s2_dlmask == 0` | 0.4815 |
| `s2_dlmask == 0` AND SCL allow-list (**adopted**) | 0.4570 |

**Correction to the figures that motivated the switch.** The quoted
0.366 -> 0.509 shift was a single cube. Across all 20 the shift is
**0.4263 -> 0.4570**: effective sample size per probe rises about **7.2%
relative**, not 39%.

**Changed.** `_MASK_NAME_CANDIDATES` prefers `s2_dlmask`;
`greenearthnet_valid_mask` applies the conjunction; falling back to `s2_mask`
prints a six-line warning naming the comparability consequence.
`scl_conjunction=False` ablates the SCL half for comparison only. Mask polarity
is still decided in exactly one place, `valid_mask_from_codes`.

**Commit.** `d4d20e1`

---

## 2026-08-03: Phase 1.2, encoder input conventions

Three calls the spec left open, decided once here so no wrapper improvises.

**Assumed.** That the SatlasPretrain multi-spectral Sentinel-2 model was the
natural EO-native choice, and that all RGB models could share one radiometric
convention.

**Observed.** `Sentinel2_SwinB_SI_MS` expects 9 channels (TCI + B05, B06, B07,
B08, B11, B12); the cubes carry only (B02, B03, B04, B8A). Feeding zeros for
six missing bands is inventing data, and the spec forbids filling. The
single-image RGB variant `Sentinel2_SwinB_SI_RGB` is fully served by the bands
we have, but it is trained on TCI/255, i.e. approximately
`clip(2.5 * reflectance, 0, 1)` -- a clamp that would hide exactly the bright
pixels the >1.2 valid-reflectance assertion exists to catch if it were applied
globally.

**Changed.** Three conventions, each printed by the wrapper that owns it:

1. **Satlas variant is `Sentinel2_SwinB_SI_RGB`**, embedding = global average
   pool of the last Swin-B stage (`fpn=False`), D=1024. The TCI clamp is
   applied INSIDE this wrapper only, because it is that model's trained input
   distribution.
2. **ImageNet ViT-B/16 and DINOv2 ViT-B/14 get reflectance as-is** (RGB from
   B04/B03/B02, resize to 224, ImageNet mean/std). No TCI brightening, no
   clipping: clipping anywhere outside the Satlas wrapper would mask the
   mask-leak signal, and brightening is a free parameter we refuse to tune.
3. **The raw baseline splits its stat domains**: per-band stats over ALL
   pixels of the unmodified frame -- the identical input the networks see,
   clouds included -- while NDVI stats run on VALID pixels only via the
   canonical `data.ndvi.ndvi`. Matching input domains is what makes the
   baseline row meaningful; masked NDVI is simply the target's definition.

Frame rule recorded with them: keep frames with clear-fraction STRICTLY above
0.5 (same comparison as `describe_cube`), store the exact fraction with every
embedding so later probes filter without re-encoding.

**Commit.** `8bfca1e` (encoders), notebook and bundle in `5cb1bde`

---

## 2026-08-03: the >1.2 check asserts prevalence, not the maximum

**Assumed.** That any valid pixel above 1.2 reflectance means bright cloud is
leaking through the mask, so a maximum over valid pixels was the right
tripwire. Phase 1.2 shipped with `assert vmax <= 1.2`.

**Observed.** It fired on the second real cube, at 1.7839. Measuring the whole
population rather than the extreme value (numbers in [../log.md](../log.md)):
**44 valid pixels out of 17,340,401** exceed 1.2, i.e. 2.5e-06, spread over 8
of 20 cubes. They are isolated singletons and 2-4 px clusters sitting in
**99.7-100% clear** frames, not contiguous regions in cloudy ones. At the worst
pixel `s2_dlmask=0` and `s2_SCL=5` (bare soil) while the superseded `s2_mask`
says cloud; it is bright across all three visible bands, the signature of a
specular target rather than cloud. NDVI at those pixels stays within
[-0.19, 0.72], and masking them moves no raw-baseline feature by more than
3.9e-04 relative.

A maximum over 17 million pixels is the most outlier-sensitive statistic
available. Asserting on it does not test "the mask works"; it tests that no
single pixel anywhere is anomalous, which no real sensor product satisfies.

**Changed.** The threshold of 1.2 for "physically implausible for a surface
pixel" stands. What is asserted is now the **fraction** of valid pixels above
it, tolerance `MAX_IMPLAUSIBLE_FRACTION = 1e-4`. That is ~5x the worst observed
cube (1.9e-05) and ~770x below the smallest systemic leak worth the name (one
fully-clouded frame of a 13-frame cube passing as clear, ~7.7e-02), so the two
cases are separated by roughly three orders of magnitude in both directions.
The count, the fraction, the valid maximum and the global maximum are all
printed every time, so nothing is silently tolerated.

Rejected: raising the threshold to ~2.0, which would have kept a maximum-based
test while blinding it exactly in the 1.2-2.0 band where real cloud sits.

**Commit.** `a6dca0d`

---

## 2026-08-03: a pixel can be both mask-valid and no-data

**Assumed.** That within a frame surviving the clear-fraction rule, every
mask-valid pixel carries a reflectance value, so encoder input could be
asserted finite everywhere.

**Observed.** It cannot. **113 pixels across the 20 cubes are simultaneously
mask-valid and non-finite**, in 6 of 264 retained frames (at most 0.14% of any
one frame). GreenEarthNet's published clear-sky conjunction is computed from
`s2_dlmask` and `s2_SCL` and never consults the reflectance bands, so nothing
in it prevents marking a no-data pixel clear. `data.ndvi.ndvi` had already
guarded against this internally (`usable = mask & isfinite(...)`) — the target
path was safe and the encoder path was not.

This was the more serious of the two: one NaN propagates through every matmul
of a ViT and returns an all-NaN embedding, and it would have halted Step 11 on
every run.

**Changed.** Three consequences, each enforced in code:

1. `encoders.frames.finite_valid_mask` ANDs the mask with "every band of this
   pixel is finite", so the encoder path and the target path agree about which
   pixels exist. This is the existing `data.ndvi.ndvi` convention applied
   consistently, not a new policy, and it only ever REMOVES pixels from the
   valid set — nothing is filled or invented. Frame selection is unchanged in
   effect: `T_kept` min 10, median 13, max 16 reproduces Phase 1.1's clear-frame
   counts exactly, so no frame changed its keep/drop decision.
2. The three network wrappers substitute `NONFINITE_FILL = 0.0` at non-finite
   pixels, inside each wrapper, before the resize (so a NaN cannot smear over
   its neighbours) and before Satlas's clamp (which does not remove NaN). Each
   wrapper prints the substitution in its preprocessing list and `encode`
   reports the count. **This is a deliberate, narrow departure from the "feed
   frames UNMODIFIED, do not fill masked pixels" rule**: a convolution has no
   concept of a mask, so a dense finite tensor is a hard requirement of the
   model rather than a modelling choice. It is not inpainting — no value is
   estimated — and it touches 6.8e-06 of band-pixels. The alternative, dropping
   the 6 affected frames, was rejected: it costs 2.3% of retained frames and
   makes frame retention depend on sensor dropout, which is a bias.
3. The raw-feature baseline takes no sentinel at all. Its band statistics are
   NaN-aware, so no-data pixels are skipped rather than filled, and cloudy
   pixels (which have values) are still included. A plain `np.mean` over a
   frame holding one NaN returns NaN for the whole frame, which would have
   blanked 6 frames of the baseline row.

**Commit.** `a6dca0d`

---

## 2026-08-03: commit references repaired after two history rewrites

**Assumed.** That the commit hash recorded at the foot of each entry above
would stay valid. This file's header promises the commit as part of the record,
so a dead hash is a broken record, not a cosmetic problem.

**Observed.** Six of the seven hashes cited here did not resolve on `main`.
Two history rewrites had replaced them: an earlier cleanup (surviving locally
as `backup-pre-rewrite`) and the 2026-08-03 pass that stripped
`Co-Authored-By` trailers from the three Phase 1.2 commits (`backup-pre-attribution-strip`).
Both rewrites preserved trees exactly and changed only commit objects, but every
hash below them moved. Anyone cloning from GitHub would have found
`436095a`, `dc1cdf7`, `96a0054`, `b4a996e`, `73704ce` and `8ddf508` missing.

**Changed.** Each reference was remapped to the commit on `main` carrying a
**byte-identical tree**, verified pairwise rather than matched by subject line:

| cited before | now | subject |
|---|---|---|
| `436095a` | `8e6285d` | Phase 1.1: switch to pre-processed GreenEarthNet cubes |
| `dc1cdf7` | `d4d20e1` | fix(mask): prefer s2_dlmask, with the SCL conjunction |
| `96a0054` | `6e9d2f7` | docs(tile): name the subset 32UNU, not Munich |
| `b4a996e` | `ebfc5d0` | feat(single-year): climatology, year-grouped CV guard |
| `73704ce` | `8bfca1e` | feat(encoders): four Tier A frozen wrappers |
| `8ddf508` | `5cb1bde` | feat(colab): phase 1.2 notebook, bundle rename |

**Consequence for anyone reading this file.** Every hash here now resolves on
`main`. If the history is ever rewritten again, re-run the check that found
this: for each hash cited, `git merge-base --is-ancestor <hash> main`. A
rewrite is cheap; a decision record pointing at commits nobody can retrieve is
not.

**Commit.** `deec2b6`

---

## 2026-08-03: Phase 1.2b, feature extraction follows each model's own protocol

**Assumed.** That a single pooled vector per frame — final CLS for the ViTs,
global-average-pool for Swin — was a neutral choice not worth recording.

**Observed.** It is neither neutral nor recorded. Nothing in the repo said
WHICH layer or WHICH token each wrapper extracted. DINOv2's published
linear-probe protocol (Oquab et al., TMLR 2023) concatenates the class token of
the last FOUR blocks with the average-pooled patch tokens of the last block,
and reports final-CLS-only as underperforming it. Probing final-CLS-only and
then concluding "DINOv2 loses dynamics" would have measured our extraction, not
the representation — the single most attackable methodological choice in a
representation audit.

**Changed.** Each wrapper follows its own model's recipe, declares it in a
`FEATURE_RECIPE` string printed at build time, and emits named intermediate
variants so the extraction ablation is one line of work rather than a
re-encode:

| encoder | probe default (`pooled`) | D | variants |
|---|---|---|---|
| `imagenet_vit_b16` | concat(cls_last, patch_mean_last) | 1536 | cls_last 768, patch_mean_last 768 |
| `dinov2_vitb14` | concat(cls_last4_concat, patch_mean_last) | 3840 | cls_last 768, cls_last4_concat 3072, patch_mean_last 768 |
| `satlas_s2_swinb_rgb` | global-avg-pool of final stage | 1024 | none — see below |
| `raw_features` | 35 whole-frame statistics | 35 | none |

Swin is hierarchical and emits a feature-map pyramid with **no CLS token and no
CLS equivalent**. Its docstring says so explicitly, and it has no `cls_*`
variant to ablate against, unlike the two ViTs. That asymmetry is a property of
the architecture, not an oversight.

**Commit.** `b84c484`

---

## 2026-08-03: Phase 1.2b, patch-grid embeddings alongside the pooled vector

**Assumed.** One pooled vector per frame was enough to probe dynamics.

**Observed.** It puts every probe deep in the p >> n regime: ~264 rows per
encoder at prototype scale against D=768-3840, where ridge results track the
regularisation path rather than the representation. It also collapses the
target to one scalar per frame, making P2 and P3 one-dimensional problems where
encoders saturate and tie — i.e. the design would have manufactured the null
result it set out to test.

**Changed.** Every wrapper additionally emits patch tokens spatially pooled to
a 4x4 grid **in the same forward pass** (marginal cost ~zero): `emb_pooled
[T_kept, D]`, `emb_grid [T_kept, 16, D_grid]`. Pooled stays the P1
(month/appearance) feature; the grid is what P2/P3 consume. The raw baseline
recomputes the same 35 statistics independently per cell (35 x 16 = 560) so it
stays spatially comparable to the networks — otherwise the comparison would
confound representation quality with spatial resolution.

**Storage.** `emb_grid` is stored float16, `emb_pooled` float32. Probe inputs
are standardised, so fp16 is ample. Measured and projected sizes are in
[../log.md](../log.md); at 1000 seasonal cubes the grid store is ~20 GB, which
is a Drive-sizing constraint to plan for, not to discover mid-run.

**A geometry note that looks like a bug and is not.** ViT-B/16 gives a 14x14
patch lattice, which does not divide evenly into 4x4, so `adaptive_avg_pool2d`
uses uneven bins and the cell mean is a *weighted* patch mean — it does not
reproduce `patch_mean_last` exactly (max abs diff ~9e-2). Where the lattice
does divide evenly (DINOv2's 16x16, Satlas's 4x4 final map) the round-trip is
exact to 6e-8. Both cases are pinned by tests so nobody "fixes" the first.

**Commit.** `b84c484`

---

## 2026-08-03: Phase 1.2b, cache the per-pixel mask so common-masking stays possible

**Assumed.** A scalar `clear_frac` per frame was sufficient bookkeeping.

**Observed.** It is not, and the gap is unrecoverable after the fact. Cube-mean
NDVI at time t is a mean over the pixels valid at t; at t+delta it is a mean
over a DIFFERENT pixel set. Measured "NDVI change" therefore partly measures
which pixels happened to be visible. Clear-fraction on this tile swings
0.44-0.63, so the confound is live, and it is a standard remote-sensing
reviewer objection.

**Changed.** Two additions, neither of which implements the fix itself:

1. The per-pixel valid mask is cached **once per cube** (not per cube-encoder)
   in `data/masks/<cube>__masks.npz`, with the original-axis frame indices.
   These compress extremely well — clear-fraction is bimodal, so frames are
   near-fully clear or near-fully clouded — measured at **146x** versus packed
   bits, 0.07 MB for all 20 cubes.
2. `grid_clear_frac [T_kept, 16]` is stored per (cube, encoder): the valid
   fraction within each grid cell. P2/P3 filter cells on this, which is
   finer-grained than dropping whole frames — a frame at 0.5 clear can hold
   cells at 0.0 and cells at 1.0.

Common-masking itself is **deliberately not implemented here**: it is
probe-side logic. This block only makes it possible. The Phase 1.2 demotion
rule is unchanged and applies throughout — a pixel the mask calls valid but
which carries no finite reflectance is not an observation.

**Commit.** `b84c484`

---

## 2026-08-03: Prithvi-EO-2.0 dropped permanently; Clay v1.5 deferred, not dropped

**Assumed.** That the EO-native tier could be filled by whichever EO foundation
models were most cited.

**Observed.** Two different problems.

**Prithvi-EO-2.0** expects six HLS bands including SWIR B11/B12. These cubes
carry only B02/B03/B04/B8A. Every Prithvi number would rest on band-filling —
inventing two channels the sensor record does not contain here — and would
carry a caveat label that no reviewer should accept. **Dropped permanently.**
Recorded here so it is not revisited: the blocker is the band set of the
GreenEarthNet minicube, not the model, and it does not go away at scale-up on
this dataset.

**Clay v1.5** is the right second EO-native model — it takes B02/B03/B04/B8A
natively with no band pain, which is why it is the least-cost addition. But it
is **not pip-installable**: there is no `claymodel` distribution on PyPI, and
`made-with-clay/Clay` on HuggingFace ships only `v1.5/clay-v1.5.ckpt`, a raw
Lightning checkpoint that requires the `Clay-foundation/model` source tree to
instantiate. That is a materially different integration cost from the other
four wrappers, all of which are one `pip install` plus one constructor.

**Changed.** Clay is **deferred to 1.2c, not dropped**, and the roster claim is
scoped accordingly: until it lands there is exactly ONE EO-native foundation
model in the roster, so no claim of the form "EO foundation-model
representations lose dynamics" can be made — n=1 will not survive review. That
constraint is now explicit rather than implied. When Clay lands it must wire
lat/lon, timestamp, GSD and band wavelengths through explicitly and assert none
of them silently defaulted; the loader already carries all four.

**Commit.** `b84c484`

---

## 2026-08-03: Phase 1.2b, land cover comes from inside the cube

**Assumed.** That deriving a land-cover stratum would need an external join
against ESA WorldCover by bounding box, with the caveat that implies.

**Observed.** Ran the lookup before assuming, as the spec required. It is not
needed: GreenEarthNet minicubes already ship **`esawc_lc`** (ESA WorldCover
10 m) as an in-cube variable, alongside `cop_dem`, `nasa_dem`, `alos_dem`,
`geom_cls` and a nine-variable E-OBS climate stack. No external join, no
reprojection, no caveat.

**Changed.** `encoders/manifest.py` derives the dominant WorldCover class per
cube from `esawc_lc`. Over the 20-cube subset: **cropland 8, grassland 6,
tree_cover 6** — a usable three-way stratification for the per-stratum
replication that the plan makes the condition for believing the headline
result. A cube missing the layer records `ABSENT:<reason>` rather than a silent
null, and `assert_strata_present` fails loudly on it.

The manifest also exposes `original_axis_index`. Horizons must be defined in
DAYS on the original regular axis, never in retained frames: after cloudy
frames are dropped, a gap of 5 retained frames spans 25 days in clear weather
and 40+ in cloudy weather, cloud correlates with precipitation, and
precipitation is a weather feature — so a frame-defined horizon leaks weather
into the horizon and degrades persistence differently than it degrades the
probe, contaminating exactly the comparison the paper rests on.

**Commit.** `b84c484`

---

## 2026-08-03: Phase 1.2c, per-cell strata, in-cube weather, and a positive control

Three review points, all accepted after checking the cubes rather than the
assumptions.

**1. Land cover is now per GRID CELL, not just per cube.** A 128 x 128 cube at
20 m is 2.56 km across; a 4x4 grid cell is 640 m. Measured on the subset:
**19 of 20 cubes are NOT single-class**, and the per-cell view resolves 5
strata over 320 cells (cropland 127, tree_cover 99, grassland 88, built_up 5,
bare_sparse 1) where the per-cube label resolved only 3 over 20 units. So the
per-cube dominant class was a coarse label over a heterogeneous scene, exactly
as the review argued. Per-cell strata give stratum contrast WITHIN one weather
realisation -- same cube, same sky, different land cover -- which is a far
stronger replication argument than comparing whole cubes that also differ in
weather, and it lets P2/P3 filter grid cells by stratum with no re-encode.
Cells are row-major, aligned with `emb_grid` and `grid_clear_frac`.

**Methods caveat, recorded deliberately.** ESA WorldCover is a STATIC ~2020/21
product; these cubes are 2018. Stable over three years for forest, grassland
and built-up, but crop rotation can move a cropland cell between years. It is a
stratification label and never a target, so the exposure is small -- not zero,
and it belongs in the paper as a footnote.

**2. The in-cube E-OBS stack is P4's entire input.** Confirmed present, fully
finite, and on the original daily axis, so it joins on `original_axis_index`
with no interpolation. P4's dependency chain on a separate weather feature
table collapses to nothing. One correction to the review: it is **eight**
variables, not nine -- `tg`/`tn`/`tx` (mean/min/max temperature), `rr`
(precipitation), `pp` (pressure), `fg` (wind), `hu` (humidity), `qq` (radiation).
`cop_dem` is cached per cell alongside it: elevation spans **271-864 m** across
the subset, which is a real phenology-timing gradient in the Alpine foreland and
the control a reviewer will ask for when green-up dates differ between cubes.

**3. Satlas multi-image fills the positive-control gap without waiting on
Clay.** Every other wrapper is single-image and cannot, even in principle,
represent change -- so a null result from those alone is not distinguishable
from "we only ever showed them one frame". `satlaspretrain-models` already
ships `Sentinel2_SwinB_MI_RGB` (verified against the package's own model list),
so this costs no new dependency. Added as `satlas_s2_swinb_mi_rgb`, D=1024.

Input convention read out of the package rather than assumed:
`AggregationBackbone.groups == [[0..7]]`, so it takes **8 images stacked
channel-wise** (24 ch) and max-pools features across the group. For retained
frame t the window is the 8 most recent retained frames ending at t, with the
earliest repeated when fewer exist -- dropping the first 7 frames of every cube
would have cost ~7/13 of this subset. The window crosses batch boundaries via a
context buffer, so batching must not change a number: verified **exactly
bit-identical** at batch_size 1, 4 and 64. MI embeddings differ from SI by 65.2
max abs, confirming it actually uses the context rather than ignoring it.

The window spans IRREGULAR gaps between retained frames, so probes must keep
using `original_axis_index` for anything horizon-related and must not read this
embedding as a fixed-duration lookback.

**Roster now.** Two EO-native encoders (Satlas SI and MI) plus the positive
control property, so the n=1 objection is answered. Clay v1.5 stays deferred,
not dropped -- this is alongside Clay, not instead of it.

**Commit.** `d58e98e`

---

## 2026-08-03: MI lookback is weather-correlated, so cache window_span_days

**Assumed.** That warning in the multi-image wrapper's docstring -- "the window
spans IRREGULAR gaps, use original_axis_index for horizons" -- was enough to
protect probes from the variable lookback.

**Observed.** It is necessary but not sufficient, and the reason is the same
confound the horizon rule exists for, now living INSIDE the encoder. The MI
window is 8 RETAINED frames; retained frames are irregularly spaced; so each
embedding's effective lookback is a variable number of DAYS. The variation is
not random: a cloudier stretch drops more frames, so the same 8 retained frames
reach further back in time. Lookback is therefore weather-correlated, exactly
like a frame-defined horizon.

A docstring cannot fix this. **A probe cannot control for a quantity that was
never cached**, and the span is computable only at encode time, from the
retained-frame timestamps that exist nowhere in the saved artefacts once
encoding is done.

**Changed.** `encoders.pipeline.window_span_days` computes, and every `.npz`
now stores, `window_span_days [T_kept]`: the calendar days between the earliest
and the current frame of each embedding's input window. Probes pass it as a
covariate. `FrozenEncoder.window_len` declares the window (1 everywhere except
the MI encoder's 8), and single-image encoders get exactly 0.0 -- an honest
constant rather than a NaN, since their lookback is one frame by construction.

**Commit.** `a1a6a12`

---

## 2026-08-03: two probe-time issues recorded now, deliberately not acted on

Both are probe-side, so deferring them costs nothing and acting now would
prejudge Phase 1.3. Recorded here so they are not rediscovered as surprises.

**1. MI max-pooling is lossy in a specific direction.** The multi-image
backbone aggregates its 8 images by MAX-POOLING features across the group
(`AggregationBackbone`). Max-pool keeps extremes and discards trajectory: it
can represent "something bright appeared in this window" but not "the scene
greened monotonically". If MI later underperforms on P2's delta probe, the
result is therefore **ambiguous between two very different conclusions** --
"temporal context does not help" and "max-pool destroyed the change signal" --
and those have opposite implications for the paper's headline. This needs one
sentence in the methods either way, and if MI does underperform it needs an
aggregation ablation before any conclusion is drawn from it.

**2. The per-cell strata have a thin tail.** Over 320 cells: cropland 127,
tree_cover 99, grassland 88, but **built_up 5 and bare_sparse 1**. Per-stratum
replication claims must cover **cropland / tree_cover / grassland only**. The
other two are to be reported for completeness and **explicitly excluded** from
the replication argument -- a "replication" over 1 cell is not a replication,
and quietly including them would let a reviewer read more coverage into the
claim than the data supports.

**Commit.** `a1a6a12`

---

## 2026-08-04: per-phase artefact directories, and a cache schema version

**Assumed.** That `data/embeddings/` and `data/masks/` were adequate homes for
artefacts, and that a "cached" file was by definition a usable file.

**Observed.** Two problems, one structural and one that had already fired
silently.

*Structural.* All phases wrote into the same directories, so re-running one
phase meant either deleting artefacts another phase still depended on, or
hand-typing a path into `rm -rf`. There was no way to say "redo Phase 1.2,
leave Phase 1.3 alone".

*Silent.* The Phase 1.2c Colab re-run reported **100 cached, 0 encoded**. The
files were valid, but nothing in the pipeline could establish they were
CURRENT. Encoder dimensionality does not settle it: the multi-image encoder
landed in `d58e98e` and `window_span_days` in `a1a6a12`, a later commit, so a
cache can hold MI files at exactly the right D and still predate the covariate.
`np.load` reports a missing key as absent rather than failing, so
`load_encoded` returned `None` and continued. A probe would have read
`window_span_days`, found nothing, and silently dropped the control for a
confound this project spent a whole block establishing.

**Changed.**

1. `data/paths.py`. Artefacts live at `data/<phase>/<kind>` --
   `data/phase1_2/embeddings`, `data/phase1_2/masks`, `data/phase1_3/...`.
   `reset_phase(phase)` deletes exactly one phase and prints what it removed;
   it REFUSES to touch `data/raw`, which stays shared because the cubes are
   phase-independent and re-downloading them to re-run a probe is waste rather
   than hygiene. `migrate_legacy()` moves pre-phase directories once and
   idempotently, so an existing Drive checkout is not forced into a needless
   re-encode by a layout change.
2. `SCHEMA_VERSION` (now 3) is stamped into every `.npz`, and `load_encoded`
   REFUSES anything older, naming the reset command. v1 Phase 1.2, v2 added the
   grid, v3 added `window_span_days`. Bump it whenever a stored field is added,
   removed, or changes meaning.

The general lesson, worth stating once: **a resumable cache is a correctness
hazard unless it can prove its own version.** Every earlier guard in this
project checks the data; this one checks the artefact against the code that
claims to have produced it.

**Commit.** `f4ed234`

---

## 2026-08-05: Phase 1.3, the crossed holdout resolves the cube/year collision

**Assumed.** That "group by cube" and "group by year" were two independent
leakage rules that could each be switched on when its axis was available.

**Observed.** They collide, and the collision is a property of the dataset's
own split structure rather than of our code:

* On the **extreme** split (the current 20 cubes) every cube is 2018, so cube
  and year are perfectly confounded: grouping by cube ALREADY groups by year,
  and a year-grouped split has exactly one group.
* On the **seasonal** split one cube internally spans 2017-2020 (~1,050
  timesteps in a single file). Grouping by year there means splitting WITHIN a
  cube -- which is precisely what the spatial rule forbids.

So on one split the year rule is vacuous, and on the other it directly
contradicts the cube rule. Choosing either rule alone produces a confident,
leaky number on one of the two splits.

**Changed.** A **crossed** mode: a test fold contains only (cube, year) pairs
whose cube is unseen in train AND whose year is unseen in train. Rows matching
on one axis only belong to neither side of that fold and are dropped from it.
This satisfies both rules simultaneously, and it is the same protocol as
`data/climatology.py` (leave-target-year-out, adopted verbatim from
GreenEarthNet), so the CV is consistent with the climatology baseline instead
of fighting it. It is the mode P4 uses whenever the manifest spans more than
one year, and it RAISES on a single-year manifest naming `cube` as the correct
fallback -- because on that subset grouping by cube already holds the year out.

Two smaller consequences recorded with it:

1. **The same-cube check runs inside the splitter, not in the caller**, for
   every mode. `year` mode on a seasonal manifest therefore raises
   `LeakageError` naming `crossed`, rather than silently splitting a cube.
2. **Year-aware modes derive the year per ROW from the timestamp** and refuse a
   manifest whose `year` column (parsed from the cube id, i.e. the window START
   year) disagrees. On a seasonal cube those differ, and splitting on the wrong
   label is exactly the failure the mode exists to prevent.

**Commit.** `76dd0c1`

---

## 2026-08-05: Phase 1.3, three modes that are honest about what they are not

**Assumed.** That `tile` and `temporal` were ordinary modes to be offered
alongside `cube`, and that a spatial mode at prototype scale was a matter of
choosing a block size.

**Observed.** Each of the three is weaker than its name suggests on this
subset, in a different way, and saying so in a docstring is not enough -- the
code has to enforce it.

**`tile` RAISES on one tile.** The 20 cubes are all 32UNU. A tile-grouped split
would have exactly one group.

**`spatial_block` is a SUBSTITUTE, and is labelled one everywhere.** It
clusters cubes by `pixel_bbox` centroid distance (complete linkage,
deterministic, no RNG) and holds out whole clusters. Test cubes still share the
tile's weather and phenology with train, so it is strictly weaker than tile
holdout; true tile holdout is deferred to scale-up. On a MULTI-tile manifest
the mode collapses to tile holdout, because clustering within tiles when whole
tiles are available would be gratuitously weaker.

**`temporal` is cube-atomic, and pays for it in dropped frames.** The
same-cube rule binds here too, so a cube straddling the cutoff cannot
contribute frames to both sides. Measured on the real subset: the 16 time
windows overlap so heavily that NO cutoff separates complete cubes. Each cube
is therefore assigned WHOLE to the side holding the majority of its frames
(ties to train) and its wrong-side frames are DROPPED, with the count printed.
At cutoff 2018-08-15: 17 cubes to train, 3 to test, **65 of 264 frames
dropped**. That starvation is the real price of the mode, which is why it is a
P3 robustness variant and never a default.

**Changed.** All three behaviours are asserted by the exit test rather than
described. The notebook's Step 8 fails if a refusal does NOT fire.

**Commit.** `76dd0c1`

---

## 2026-08-05: one Drive folder per phase

**Assumed.** That phase-scoped directories inside one checkout
(`data/<phase>/<kind>`, `f4ed234`) were sufficient isolation, and that all
phases could share one Drive folder.

**Observed.** They are sufficient for artefacts and insufficient for the
CHECKOUT. Every phase extracting its bundle into the same folder means one
zip's `extractall` overwrites another phase's code in place, so "re-run Phase
1.2 as it was" is not reachable once Phase 1.3 has been uploaded, and clearing
a phase means trusting `reset_phase` to be the only thing that ever wrote
outside its own directory.

**Changed.** From Phase 1.3 on, each phase gets its own Drive folder
(`NeurIPS-CCAI-2026-phase1_3/`) holding its own checkout. Two rules keep it
coherent:

* **A phase writes only under its own folder**, through `data.paths.phase_dir`.
* **A phase READS earlier artefacts in place.** The Phase 1.3 bootstrap
  searches Drive one and two levels down for `data/raw/*.nc` and
  `data/phase1_2/embeddings/*.npz`, uses whichever it finds, and never writes
  there. Copying 70 MB of cubes per phase would be waste; writing into Phase
  1.2's folder would destroy the property this change exists for.

Deleting a phase folder is now a complete undo of that phase.
`reset_phase(phase)` remains the finer-grained version, and `data/raw` stays
shared and is cleared by neither.

**Commit.** `76dd0c1`

---

## 2026-08-06: a shared Drive folder is an untrusted input, and must be audited

**Assumed.** That a directory of `.npz` written by our own pipeline could be
read back by globbing it, and that `load_encoded`'s schema guard was enough
protection because it refuses anything stale.

**Observed.** A real Phase 1.3 run died at the join step, and the traceback
carried four separate faults, only one of which was the schema:

```
AssertionError: Copy of 32UNU_..._dinov2_vitb14.npz was written with cache
schema v0, but this code expects v3
```

1. **`EMB_IN` resolved to the wrong directory.** The bootstrap searched "this
   checkout first", and a stale copy of `data/phase1_2/embeddings` was sitting
   INSIDE the phase1_3 checkout. It won over the real Phase 1.2 folder. Taking
   the first hit is not a selection.
2. **Google Drive had renamed a duplicate to `Copy of <name>.npz`.** That still
   ends in the right `__<encoder>.npz` suffix, and `C` sorts before a digit, so
   `sorted(glob(...))[0]` picked the copy *every time*. The selection was
   deterministic and deterministically wrong.
3. **The copy was pre-schema**, so the run halted on it -- at an arbitrary
   point, with the other 99 files undiagnosed.
4. **Only one pair per encoder was ever checked**, so a cache with holes in it
   would have passed.

**Changed.** `encoders.pipeline.audit_embeddings` partitions a directory before
any probe reads from it: `current` / `unstamped` / `incomplete` / `foreign` /
`duplicates` / `unreadable`, each with its own remedy printed.
`assert_embeddings_complete` then requires every (cube, encoder) pair the
caller needs. **Every later phase should call both before touching an
embedding**; that is why this lives in the repo and not in a notebook cell.

Duplicate detection compares the filename against the one derived from the
file's OWN stored `cube` and `encoder`. That is exact and locale-independent --
Drive localises the prefix ("Kopie von", "Copie de", ...), so matching on the
prefix would work in English and fail silently in German.

Two smaller rules follow from the same incident:

* **A phase never prefers another phase's artefacts found inside its own
  checkout**, whatever the file counts say. The layout contract -- a phase reads
  its inputs in place and never owns a copy -- is now enforced in the resolver
  rather than described in the RUNBOOK. The block is sentinel-fenced and
  exercised by `tests/test_notebook_resolver.py` against a simulated Drive,
  including the exact tree that failed.
* **Coverage is asserted before the contract.** Asserting the join on one pair
  proves the contract; asserting it on all of them proves the cache is whole.
  A cube silently missing one encoder turns a per-encoder comparison into a
  comparison over DIFFERENT cubes, which no downstream assertion can detect --
  and that is precisely the comparison P1-P4 rest on.

**The general lesson, worth stating once.** Phase 1.2 established that a
resumable cache is a correctness hazard unless it can prove its own version.
This adds the other half: **a cache on shared storage cannot be trusted to
contain only what we put there.** Drive copies files, users copy folders, and
both produce artefacts that are byte-identical to something valid and yet
wrong to read. The remedy is not a stricter loader -- `load_encoded` was
already correct -- but a selection step that decides what to load.

**Commit.** `0116d4d`

---

## 2026-08-06: a flat legacy layout hid 100 good embeddings from every resolver

**Assumed.** That the resolver in `notebooks/phase1_3_cv.ipynb` (Step 2)
searching for `data/phase1_2/embeddings` at various Drive depths was enough to
find Phase 1.2's artefacts wherever a user's checkout happened to sit.

**Observed.** On a real Drive, `organise_drive.ipynb` Step 5 (whole-Drive scan,
`fdc8f18`) showed the actual, correct, current 100-file five-encoder embedding
set sitting at `NeurIPS-CCAI-2026/phase1_2/data/embeddings/` -- one folder
short of what any resolver looks for. `data.paths` phase-scoped artefact dirs
(`f4ed234`) moved the write path from `data/embeddings` to
`data/<phase>/embeddings`; this checkout's DATA predates that refactor even
though its CODE does not (it has `paths.py`, so it was extracted from a later
bundle after the embeddings were already written). The resolver has no
fallback for the flat layout and never should: with THREE flat-layout
candidates on the same Drive (the real one, plus an intentional 80-file
`Back-Up/` copy, plus the already-diagnosed stale `Copy of` duplicates one
folder over), guessing which is current would be exactly the "first hit wins"
mistake `0116d4d` exists to prevent.

**Changed.** `organise_drive.ipynb` Step 5 gained a legacy-layout detector:
finds any `.../data/embeddings` or `.../data/masks` whose PARENT is literally
named `data` -- i.e. no phase segment between them -- holding at least one
`.npz`, and prints the exact rename by walking up for the nearest
`phase1_N`-named ancestor. It only detects and prints; nothing moves a file,
matching every other tool in this project. Verified against a reconstruction
of the real tree: flags the two flat directories (the real 100 files and the
80-file backup) with correct rename targets, does not flag the already-nested
`Copy of` duplicates (a different, already-handled defect), and the flag
disappears once the suggested rename is performed.

The resolver itself is UNCHANGED and stays strict: nested path only, no flat
fallback. Silently accepting a second shape a directory could take is how the
first incident happened; this is a human-in-the-loop diagnostic, not a wider
search.

**Commit.** `142a7ff`

---

## 2026-08-07: K1 fired -- EO-WM's published rows are not a validation surface

K1's pre-registered condition: EO-WM eval unreproducible (no code reply AND
appendix rows not matchable within ~10%) -> drop their benchmark surface,
keep our own protocol.

**Fired**, for a better-documented reason than anticipated. The authors
replied helpfully and shared configs, a reconstruction eval script, and a
weather climatology file -- see
[docs/correspondence/2026-08-07-eowm-authors.md](correspondence/2026-08-07-eowm-authors.md).
But two things put the published rows out of reach within our budget:

- The Earthformer baseline is not an official checkpoint. It is a
  200-epoch self-trained model on `CuboidEarthNet2021` with meso auxiliary
  input -- reproducing it means reproducing that training run, not loading
  weights.
- The core EO-WM model and training code are unreleased pending
  acceptance.

Separately, Tab. 1 and Tab. 2 never contained persistence rows in the
first place -- "copy last clear frame" and "persistence (cloud-free mean)"
are Appendix A.1 reconstruction references, not forecasting baselines. So
even the part of K1's condition that assumed a persistence row to match
against does not apply as framed.

**Consequence.** We drop "match EO-WM's published rows" as a validation
surface and evaluate under our own protocol: our own persistence baseline
for P3, our own mask definition (GreenEarthNet's `s2_dlmask` + `s2_SCL`
allow-list, not EarthNet2021-era masks), citing the authors' configuration
for commensurability where it is directly comparable (5 of our 8 E-OBS
channels, their reconstruction-diagnostic conventions if H3 reaches
EarthNetScore). This decides the eval surface, not the project -- exactly
as pre-registered.

**Commit.** `d380d2a`

---

## 2026-08-08: `dinov2_vitb14` can be encoded locally after all -- it needed Python 3.10, not a GPU

**Assumed.** That `dinov2_vitb14` was a Colab-only encoder. Every local run
since Phase 1.2 has had FOUR encoders in its cache, and `RUNBOOK.md` recorded
that as a standing limitation of the dev machine.

**Observed.** The limitation is neither the GPU nor the weights. DINOv2's
`torch.hub` code annotates a signature with `float | None`, a PEP-604
expression Python evaluates at class-definition time, and the dev venv is
3.9.6:

```
TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'
  .../dinov2/layers/attention.py:58 in class Attention
```

Nothing about the model requires a GPU. All 20 cubes encode on CPU in **42
seconds**.

**Changed.** The 20 missing `.npz` were produced in a throwaway Python 3.10
virtualenv pinned to the SAME `torch==2.8.0` / `torchvision==0.23.0` as the dev
venv, driving `encoders.pipeline.encode_cube` and `save_encoded` unchanged --
no new machinery, no second code path, and every file re-read through
`load_encoded` before it was trusted. The local cache is now the full 20 x 5 =
100 files the README describes, so P1 evaluates the whole roster locally
instead of four fifths of it.

Two things this deliberately does NOT do. It does not change
`encoders/dinov2_vit.py`: pinning an older hub revision would make the local
encoder a different encoder from the one the archived Colab run used, to save
an interpreter. And it does not add the 3.10 environment to the repo -- it is a
one-off to fill a cache, not a supported second toolchain. Anyone reproducing
this needs Python >= 3.10 and no GPU.

**Commit.** `03f7cf2`

---

## 2026-08-08: Phase 1.4, P1: grid-cell rows are the PRIMARY feature set

**Assumed.** That the pooled embedding was the natural probe input, with the
patch grid as an optional extra -- it is the vector each encoder's own
linear-probe recipe produces.

**Observed.** On this subset the pooled vector is severely p >> n, and not
marginally so. DINOv2's pooled feature is **D = 3840 against 264 rows**: a
linear classifier can separate almost any labelling of 264 points in 3840
dimensions, so a high pooled score is not evidence that month is present, it is
evidence that the design matrix is wide. Regularisation tuned by nested CV
controls the damage but does not remove the ambiguity.

The cached `grid` array resolves it without re-encoding anything. Flattening
the 4x4 patch grid per CELL turns each frame into 16 rows that inherit its
label: **264 -> 4224 rows at D_grid <= 768**. That is the only feature set here
where the matrix is taller than it is wide, for every encoder.

**Changed.** `grid_cell` is the feature set the headline table is read from,
and `pooled` is reported beside it rather than instead of it. Two properties
make the explosion safe, and both are asserted:

* **Cube grouping survives it.** A cell row carries its frame's manifest row,
  so all 16 cells of a frame land on the same side of every fold. The folds
  still come from `probes/cv.py` unchanged -- the explosion happens in the
  feature matrix, never in the split.
* **The effective sample size does NOT go up.** 16 cells of one frame share a
  sky and 13 frames of one cube share a place; 4224 is a fix for the FIT, not
  4224 independent observations. This is why every metric is reported with its
  across-fold spread over 20 cubes and never as a bare mean, and why the
  degenerate control is run at cell level too rather than only at frame level.

**Commit.** `03f7cf2`

---

## 2026-08-08: Phase 1.4, P1: the multi-image encoder is REPORTED, never RANKED

**Assumed.** That `satlas_s2_swinb_mi_rgb` was a fifth column of the same
table -- it is the positive control, and a control that scores well is the
point of having it.

**Observed.** The target does not mean the same thing for it. MI aggregates 8
RETAINED frames, and retained frames are irregularly spaced, so its lookback is
a variable number of DAYS: measured on this subset **min 0, median 55, max
105** (`window_span_days`, cached at encode time in `a1a6a12` precisely so a
probe could condition on it). An MI embedding at time t therefore summarises up
to three months of history. Asking "which month is this?" of a vector that
averages August, September and October is not the question the single-image
encoders are being asked, and the two answers do not belong in one ranked
column.

The failure mode is specific and would be invisible: MI could score HIGHER than
a single-image encoder because the extra history disambiguates a cloudy frame,
or LOWER because the label is smeared across the window -- and either way the
number would be read as a statement about representation quality.

**Changed.** Three things, none of them "drop the row":

* Every MI row carries `si_comparable=False`, asserted by
  `assert_results_complete`, and the console output tags it
  `<- NOT SI-COMPARABLE`.
* `rank_agreement` excludes MI when it compares encoder orderings across fold
  modes, so it can never enter a ranking by accident.
* MI is additionally reported **conditioned on `window_span_days`**, in
  terciles computed from the realised values rather than at the nominal 35-day
  span (which is wrong by a factor of three here). Chance is recomputed on each
  tercile's own class distribution, because subsetting by lookback changes
  which months are present.

The Figure 1 panel keeps the same treatment: MI is drawn on the shared axes and
labelled `(MULTI-IMAGE, variable lookback)` rather than left to look like a
fifth single-image encoder.

**Commit.** `03f7cf2`

---

## 2026-08-08: Phase 1.4, P1: the baseline sees a band the networks do not

**Assumed.** That `raw_features` was a strictly simpler model of the same
evidence -- "per-frame summary statistics of exactly the input the network
encoders see", as its own docstring puts it -- so beating it would be a clean
statement about representation quality.

**Observed.** The docstring is true of the four SPECTRAL bands and false of the
band set. `raw_features` reduces all of `B02, B03, B04, B8A` and adds seven
statistics of canonical NDVI, which needs B8A. Every network encoder here is
RGB-only: `imagenet_vit_b16` and `dinov2_vitb14` by construction, and both
Satlas wrappers because the RGB variant was chosen over the multi-spectral one
(2026-08-03, above -- `..._SI_MS` expects nine bands these cubes do not carry,
and filling the gap would invent data). So `rgb_from_s2` drops B8A before any
network sees a pixel.

NIR is where the seasonal vegetation signal mostly lives. The baseline is
therefore not a weaker model of the same evidence in P1; it is a model of MORE
evidence.

**Changed.** Nothing in the code -- both choices remain right for their own
reasons. What changes is how the table is read, and that is now stated at the
top of `probes/p1_appearance.py`, in the log, and here: **a row where
`raw_features` beats a frozen encoder is a statement about the input as much as
about the representation.** The comparison that is clean is
network-vs-network, and network-vs-degenerate-control.

This also sets up the honest version of the missing experiment: an RGB-only
raw-feature baseline would isolate the representation effect from the band
effect. It is not run here -- it would mean a new encoder variant and a
re-encode, which is Phase 1.2 work, not P1's -- and it is recorded as the
first thing to add if a later phase wants to make a representation-quality
claim from an appearance probe.

**Commit.** `03f7cf2`

---

## 2026-08-08: Phase 1.4, P1: the degenerate control is a competitor, not a floor

**Assumed.** That `[clear_frac, window_span_days]` -- two numbers, no image --
would sit near chance, and that the control's job was to let us say so in one
line and move on. The P1 spec called it "not optional" as a matter of
discipline, on the expectation that it would be uninformative.

**Observed.** It is not uninformative. Measured on the primary feature set
(logreg, cell level, full numbers in [log.md](../log.md)):

```
season  spatial_block   0.681 +/-0.119   vs chance 0.333   BEATS all five encoders
season  loco            0.674 +/-0.233   vs chance 0.333   ties the best encoder
season  cube            0.627 +/-0.123   vs chance 0.333
month   spatial_block   0.328 +/-0.052   vs chance 0.125   within noise of the best
month   cube            0.282 +/-0.118   vs chance 0.125
```

**Cloud retention on this subset is strongly seasonal.** That is obvious in
hindsight -- Alpine-foreland cloud climatology *is* a season -- but the
magnitude is not: two numbers containing no image reach 0.68 balanced accuracy
on a 3-class season task and beat every frozen encoder under `spatial_block`.

**Changed.** Three things, in what P1 is allowed to claim rather than in the
code:

* **A P1 SEASON score is not evidence about representation quality on this
  subset.** Every encoder clears chance comfortably and none of them clearly
  clears the retention control, so the season column cannot separate "this
  embedding encodes season" from "this embedding encodes how cloudy it was".
  It is reported in full and read as a calibration line only.
* **MONTH under `cube` or `loco` is the only cell that separates.** There all
  four single-image encoders lead the control by +0.06 to +0.15. Everywhere
  else the margin collapses: on month under `spatial_block` only
  `raw_features` (+0.023) and `satlas_s2_swinb_rgb` (+0.011) stay above it,
  and on season no encoder clears it under any fold mode (best +0.027, worst
  -0.224). Eight classes make retention a weaker proxy than three do. Where P1
  is cited later, cite month under cube or LOCO, and cite it with the margin
  over the control rather than the margin over chance.
* **P2 and P3 inherit the control, not just the finding.** Any probe on this
  subset whose target correlates with time-of-year now has to carry a
  retention-only row, because the same confound is sitting underneath it. This
  is cheap -- the covariates are already cached -- and it is the difference
  between measuring a representation and measuring the weather that decided
  which frames survived.

**What this does NOT change.** The P1 verdict itself. No encoder fails P1:
every one clears chance by a wide margin on both targets under all three fold
modes, so the surprise the probe was watching for -- an EO model trained to
appearance-invariance -- did not occur, and P2/P3 are licensed. The control
narrows what a P1 pass means; it does not withdraw it.

**Commit.** `03f7cf2`

---

## 2026-08-08: Phase 1.4, P1: the logistic-regression C grid is too narrow, and it is left that way for this run

**Assumed.** That `C in (1e-4 .. 1)` was a generous range for a p >> n problem,
where the useful regularisation is strong and the weak end is there only so a
boundary selection is visible as one.

**Observed.** The boundary selection is not rare. Across all 1920 outer folds,
`C = 1` -- the maximum -- is chosen **550 times (57%)**, and 555/960 logreg
folds land on some grid edge against 111/960 for ridge, whose modes are
interior (alpha 100-1000). The `grid_cell` feature set is the reason: at 4224
rows against D <= 768 it is no longer p >> n, so the data genuinely wants less
regularisation than the grid can offer.

For those folds the regularisation was pinned by the grid rather than by the
data, and the reported score is a LOWER bound.

**Changed.** Nothing, for this run, deliberately. The direction of the bias is
known and one-sided -- extending the grid can only raise the affected scores --
so no conclusion in [log.md](../log.md) is reversed by it: every encoder clears
chance, and the control comparison that the P1 verdict rests on is unaffected
because the control's own scores would move the same way. Re-running to chase
it would cost 34 minutes to make some numbers slightly larger without changing
a claim.

What it DOES contaminate is the estimator comparison. "logreg beats ridge" is
partly an artefact of where the two grids end, so this run does not support
that statement and does not make it.

`at_grid_edge` is recorded per fold and `n_at_grid_edge` per row, so this was
visible in the output rather than inferred afterwards -- which is the reason
the field exists. **Extend `LOGREG_C_GRID` upward (10, 100) before P2 reuses
this machinery**, and note that P2's targets may be genuinely p >> n again, in
which case the low end still earns its place.

**Commit.** `03f7cf2`

---

## 2026-08-09: the design matrix, its manifest index and its labels become one type

**Assumed.** That `feature_matrix` returning `(X, row_idx)` was fine, and that
the one place they got separated -- `select_hyperparameter`, handed a sliced
`X_tr` beside the UNSLICED `row_idx` -- was a typo caught by an `IndexError`.

**Observed.** The `IndexError` was luck, not detection. Replaying the bad call
over the real manifest, per inner fold:

```
cube k=5        15/15 raise
spatial_block   13/15 raise,  2 return a WRONG NUMBER
LOCO            51/60 raise,  9 return a WRONG NUMBER
```

**11 of 90 inner folds -- 12% -- would have silently tuned the regularisation
strength on the wrong rows**, and they cluster in LOCO, where the training fold
is largest and its row positions are therefore most likely to fall inside the
sliced array. The first probe run happened to exercise `cube` mode first, where
the bug is 100% fatal. Had it started with LOCO it would have produced a
plausible, wrong table and nothing would have looked out of place.

The hazard is not that one call site was wrong. It is that the CELL-level
feature sets put 16 X rows on every manifest row, so `X`, `row_idx` and `y`
have three different lengths that no single function is responsible for keeping
in step -- and for the FRAME-level sets, where all three are 264 long, the same
mistake does not even change a shape.

**Changed.** `FeatureBlock`: a frozen dataclass holding `X`, `row_idx` and `y`,
whose length agreement is asserted on EVERY construction, including every
slice. `take` and `select` slice all three or none; `with_labels` is the only
way labels get attached, and it expands them through `row_idx` rather than by
position. `select_hyperparameter` and `evaluate_fold` take a block, so there is
no longer a pair of arguments that can disagree -- the leakage-relevant
signature check in the tests now also asserts the parameter is the bound type.

Chosen over adding an assertion at the call site because P2, P3 and P4 all
consume cell-level rows and will write this pattern again. An assertion
protects one site; a type protects the pattern. This is also why it is worth
doing BEFORE scale-up rather than after: the same slicing runs at twenty times
the volume there, and a wrong number costs a full re-run to discover.

**Commit.** `3256748`

---

## 2026-08-09: both class weightings are reported, because one of them was understating every row

**Assumed** (2026-08-08, above). That leaving both estimators unweighted was
the conservative choice: the metric is balanced accuracy, so weighting moves
the estimator toward the metric, and a conclusion that survives without it
survives the imbalance rather than being rescued from it.

**Observed.** Conservative, but also simply the wrong estimator for the metric,
and by an amount that is not small. November holds 5 frames of 264. An
unweighted multinomial loss has almost no incentive to ever predict it, and a
class never predicted contributes 0 recall to a metric that averages recall
over classes -- so it forfeits up to 1/8 of balanced accuracy outright, before
any question of what the representation contains. Measured directly
(month, cube k=5, grid_cell, C=1):

```
                     unweighted            class_weight="balanced"
raw_features         0.425 +/-0.068        0.455 +/-0.056
  recall on Oct+Nov  0.332                 0.528
dinov2_vitb14        0.392 +/-0.084        0.395 +/-0.084
```

Note what that says: weighting helps the BASELINE most, and barely moves the
network. It is not a cosmetic improvement to the encoders' numbers -- it widens
the gap the baseline already had. That is the honest reason to run it, and the
reason it was worth measuring before deciding.

**Changed.** `ESTIMATORS` becomes four: `logreg`, `logreg_balanced`, `ridge`,
`ridge_balanced`. Both weightings of both estimators are reported, and neither
is "the" answer. The unweighted rows are the conservative floor; the weighted
rows are the estimator that matches the metric. The DEGENERATE CONTROL is
weighted the same way in each pair, so `margin_over_control` -- the number P1
is actually read on -- is a like-for-like comparison under either.

Nothing is retracted from the earlier entry: the reasoning for wanting an
unweighted row stands, and that row is still there. What was wrong was
reporting ONLY it.

**Commit.** `3256748`

---

## 2026-08-09: the logreg C grid is extended, because "a known lower bound" is not a table

**Assumed** (2026-08-08, above). That leaving `C` capped at 1 was acceptable
for this run: the bias is one-sided and known, so no conclusion is reversed,
and re-running to chase it would cost 34 minutes to make some numbers slightly
larger.

**Observed.** Two things make that the wrong trade after all. The scale of it:
`C = 1` was selected in 550 of 960 logreg folds, so the majority of the
headline table was reporting a bound rather than a measurement -- "no
conclusion is reversed" is true and is not the same as "this table can be
published". And the cost was mis-estimated in the wrong direction: larger `C`
separates sooner, so lbfgs converges in FEWER iterations. Measured on dinov2
grid_cell, 4224 x 768: 67 iterations at `C=1`, 47 at `C=10`, **30 at `C=100`**.
Extending the grid upward makes the run faster per fit, not slower.

A grid that has to be defended in prose is a grid that should have been wider.

**Changed.** `LOGREG_C_GRID` gains `10` and `100`; `RIDGE_ALPHA_GRID` gains
`1e5`. Both grids now bracket their interior modes on both sides, so a
selection at an edge means the data wants an extreme value rather than that the
grid ran out. `n_at_grid_edge` stays in the CSV as the check that this is so.

**Commit.** `3256748`

---

## 2026-08-09: the band asymmetry, MEASURED -- the networks win the fair comparison

**Assumed** (2026-08-08, above). That `raw_features` beating every frozen
encoder was an input effect that could be stated but not separated, because
isolating it "would mean a new encoder variant and a re-encode, which is Phase
1.2 work, not P1's".

**Observed.** It costs a column slice. `raw_features` stores its per-band
statistics in a KNOWN ORDER (`encoders.raw_features.RAW_FEATURE_NAMES`), so the
band-matched baseline is 21 columns of an array already on disk. Measured, month
/ cube k=5 / logreg / grid_cell:

```
raw_ALL      (35: RGB + B8A + NDVI)   0.430    <- what we had been comparing to
dinov2_vitb14        (RGB-only net)   0.387
satlas_s2_swinb_rgb  (RGB-only net)   0.386
imagenet_vit_b16     (RGB-only net)   0.350
raw_NIR+NDVI (14)                     0.339
raw_RGB-ONLY (21)    BAND-MATCHED     0.328
DEGENERATE CONTROL   (2)              0.282
```

**The conclusion reverses.** Given the same three bands, every frozen network
beats hand-crafted percentiles (+0.02 to +0.06). `raw_features` led only because
it also sees B8A and NDVI -- and neither half alone (0.328, 0.339) approaches
the combination (0.430), so its advantage is the pair, not the extra band by
itself.

**Changed.** `raw_rgb_only` and `raw_nir_ndvi` are first-class feature sets,
`margin_over_band_matched` is a column, and `assert_results_complete` refuses a
table without the band-matched row. The column indices are DERIVED from
`RAW_FEATURE_NAMES` and asserted to partition it, so reordering the baseline's
features cannot silently repoint the slice; `feature_matrix` refuses the slice
on any encoder whose grid is not D=35.

**What survives from the earlier entry.** The mechanism, and the reading rule:
`raw_features` is still a model of more evidence, so it is still not the
comparison to make a representation claim against. What was wrong was calling
the separation expensive and deferring it. The correction cost ten minutes and
changed the headline, which is the argument for running the cheap control
BEFORE writing the interpretation, not after.

**One caveat that does not go away.** The advantage is not robust to the
strictest split: under `spatial_block`, `dinov2_vitb14` (-0.017) and
`imagenet_vit_b16` (-0.020) fall BELOW the band-matched baseline on month. The
more honest the geography holdout, the less of the representation advantage
survives. Reported, not smoothed.

**Commit.** `3256748`

---

## 2026-08-09: the C grid, closed out -- the residual edge is saturation, not truncation

**Assumed** (2026-08-09, above). That widening `C` to 100 would move the grid
edge from "the grid ran out" to "the data wants an extreme value", and that
`n_at_grid_edge` would then be small.

**Observed.** Half right. Edge selection fell from **34.7% to 21.3%** of 4320
outer folds, and `ridge` is now bracketed on both sides (25/1080 at 1e4, none at
1e5). But plain `logreg` still selects the TOP edge in 27.5% of folds
(297/1080), which is the same complaint one order of magnitude further out.

So the question was tested rather than argued. satlas grid_cell, cube fold 1:

```
C=100     test bal-acc 0.4492
C=1000    test bal-acc 0.4527    98.9% of predictions identical to C=100
C=10000   test bal-acc 0.4533    99.9% identical to C=1000
```

**The fit has saturated.** On a 4224 x 768 design matrix the data genuinely
wants essentially no regularisation, and past C=100 the decision rule stops
moving: extending the grid two more decades changes 0.1% of predictions and
0.004 of balanced accuracy, against a fold spread of +/-0.044.

**Changed.** Nothing further. The grid stays at 1e-4..1e2 and this measurement
is the reason the remaining 27.5% is reported rather than chased. Note the
asymmetry that makes it interpretable: `logreg_balanced` uses the whole grid
(85 folds at 1e-4, 189 at 1e2), so the concentration at the top is a property of
the unweighted objective on a near-separable problem, not of the grid.

**Commit.** `8ebee9c`

---

## 2026-08-09: two encoders cannot be ranked at this sample size, and we can prove it

**Assumed.** That the fold-to-fold spread was the honest uncertainty, and that
reporting it was sufficient caution around the encoder ordering.

**Observed.** It is not sufficient, because the ordering is unstable to
something smaller than the spread. The same commit, the same 100 `.npz`, run on
Python 3.9.6 locally and 3.12 on Colab -- different sklearn, scipy and BLAS:

```
month / cube / logreg / grid_cell     local     Colab
dinov2_vitb14                         0.387     0.389
satlas_s2_swinb_rgb                   0.386     0.389
```

Every score agrees to +/-0.003. But those two encoders are separated by 0.001
locally and 0.000 on Colab, so **the rank agreement statistic itself changes**:
month / logreg / cube-vs-spatial_block is Spearman +0.800 locally and +0.400 on
Colab.

**Changed.** Nothing in the code -- there is nothing to fix. This is recorded as
a claim boundary: on this subset P1 supports "every encoder clears the
band-matched baseline / the retention control by X" and does NOT support "encoder
A is better than encoder B". Any ordering in the tables is presentation, not a
result, and the cross-environment reproduction is kept in `notebooks/runs/` as
the evidence.

It is also the strongest available argument for scale-up, and a cheap one to
have made: it did not need a new experiment, only running the existing one
twice. 20 cubes / 1 tile / 1 year cannot separate two encoders. Three of six
fold modes (`year`, `tile`, `crossed`) additionally refuse to run at all here,
so the evaluation contribution cannot be demonstrated in the modes that matter
most. Both are fixed by the same download.

**Commit.** `8ebee9c`
