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
