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

---

## 2026-08-10: the manifest indexed the daily weather axis with the acquisition axis

**Assumed.** That `original_axis_index` was, as `encoders/manifest.py`'s own
docstring called it, an index into "the original regular time axis", and that
the in-cube E-OBS stack therefore "joins on `original_axis_index` with no
interpolation" (2026-08-03, above). Both sentences were written down and neither
was checked against a cube.

**Observed.** They describe two different axes. A GreenEarthNet minicube is
stored on a ~150-step **daily** grid of which about 29 steps carry a Sentinel-2
acquisition; `data.loader.load_cube` drops the empty steps by design, so
`select_clear_frames`' `kept_idx` -- and hence `original_axis_index` -- counts
**acquisitions**. The E-OBS series are on the **daily** grid. `manifest_rows`
indexed one with the other.

Over the 20-cube subset, **0 of 264 rows carried the weather of the day their
frame was acquired**: offset 4 to 122 steps, median 53. Mean-temperature error
MAE 6.26 K (max 22.0), precipitation MAE 2.52 mm (max 35.0), radiation MAE 89.1
(max 232). A frame acquired on 2018-04-22 carried 2018-03-17's temperature, 0.0 C
instead of 17.0 C.

**Nothing caught it and nothing could have.** The values were finite, in range,
the right shape and the right dtype, and the manifest was internally consistent
-- they were simply another day's weather. Every guard this project has added so
far checks the artefact against the code that wrote it, or the code against
itself. This is the first defect that required going back to the SOURCE and
re-deriving the join independently.

**Changed.**

1. `cube_daily_axis(path)` returns the cube's daily time axis, read from the same
   file as the E-OBS series so the two cannot come from different loads.
2. `frame_daily_positions(daily_axis, timestamps)` locates each retained frame on
   it by TIMESTAMP, with an **exact-match assertion**. Nothing rounds to the
   nearest day: a weather value attached to the wrong day is a confident wrong
   number, and a nearest-neighbour join would have hidden this bug rather than
   surfaced it.
3. The manifest gains a `daily_axis_index` column and joins E-OBS on it.
   `original_axis_index` is UNCHANGED -- it is the embedding join key asserted in
   `probes.cv.join_embeddings`, and re-pointing it would have broken Phase 1.2's
   contract to fix Phase 1.5's bug.
4. `assert_weather_join(df, cube_dir)` re-derives the entire join from the cubes
   and refuses any disagreement above 1e-9. **Any probe whose input is the
   weather calls it before fitting anything**; `run_p4` does, first thing.

**Scope.** P1-P1.4 are unaffected: no probe before P4 reads a weather column, and
`probes/cv.py` groups on cube/tile/timestamp. The module docstring now opens by
naming the two axes, because the failure was a naming collision as much as an
indexing one.

**The general lesson, worth stating once.** Phase 1.2 established that a
resumable cache is a correctness hazard unless it can prove its own version;
Phase 1.3's Drive incident added that a cache on shared storage cannot be trusted
to contain only what we put there. This adds a third: **a derived column cannot
be validated against the table it lives in.** Its plausibility there is exactly
what makes it dangerous.

**Commit.** `13d079d`

---

## 2026-08-10: Phase 1.5, P4: the proxy climatology is fitted INSIDE the fold, and its order is chosen by the sanity control

**Assumed.** That the within-season proxy climatology was bookkeeping -- a
day-of-year curve to subtract before modelling -- and that the interesting
choices in P4 were the features and the estimators.

**Observed.** It is the single most consequential object in the probe, for two
reasons that only became visible once it was measured.

*First, it defines the TARGET, so where it is fitted is a leakage question and
not a style question.* Fitting one curve on all 20 cubes and then
cross-validating the residual leaks the held-out cubes into the definition of
what is being predicted. That inflates every number **including every control**,
so the margins do not reveal it; the resulting table is internally consistent and
wrong. It is the one error in this probe that nothing downstream can catch.

*Second, its ORDER changes the headline by a factor of two.* Measured (cube k=5,
linear, 8 variables, `r2_vs_climatology`):

```
H    cube_mean            cube_p90             cell_mean
     weather   DOY        weather   DOY        weather   DOY
2     +0.092  +0.040       +0.132  +0.106       +0.077  +0.010
4     +0.066  +0.007       +0.066  -0.007       +0.067  +0.001
```

At two harmonics the "anomaly" still carried enough seasonal cycle for
day-of-year ALONE to explain 4.0% of it and 10.6% of the 90th percentile -- and
the weather model's number was inflated to match, because weather predicts
phenology trivially. **Most of what looked like weather attributability at low
order was leftover seasonal cycle.**

**Changed.**

1. `doy_climatology_within_fold(day_of_year, values, train_idx, ...)`. The
   training index set is a REQUIRED POSITIONAL argument, so the curve cannot be
   fitted on everything by omitting one; the arrays are read exactly once,
   through that index; and the test poisons the held-out rows by +10.0 and
   asserts the coefficients are bit-identical, with a companion test asserting
   the curve DOES move when a training row changes. A test that can only pass
   proves nothing.
2. **`CLIMATOLOGY_HARMONICS = 4`, chosen BY the day-of-year sanity control** --
   the lowest order at which that control lands at zero for all three targets at
   once. This is a legitimate use of a held-out diagnostic precisely because it
   is one-directional: raising the order LOWERS the reported weather number
   everywhere, so selecting against this control cannot talk the headline up.
3. `DOY_CONTROL_HARMONICS = 6`, deliberately RICHER than the detrend. A control
   built from the detrend's own basis returns zero by least-squares construction
   and tests arithmetic rather than the detrend.
4. Every Stage A row carries `climatology_def` = "within-season proxy
   climatology, tile-level, single year (2018), NOT the leave-year-out
   definition". `data/climatology.py` is untouched and still raises
   `SingleYearError`; the proxy lives in the probe, under its own name, and the
   two cannot be confused in the CSV.

**Commit.** `13d079d`

---

## 2026-08-10: Phase 1.5, P4: the headline is the margin over the observation control, not the R-squared

**Assumed.** That P4's deliverable was an R-squared -- "weather explains X of the
anomaly" -- and that controls were there to reassure a reviewer.

**Observed.** On this subset the raw number is not interpretable on its own, and
P1 had already shown why. Cloudiness drives precipitation, which is a weather
feature; it ALSO drives which frames survive the clear-fraction filter and which
pixels the NDVI spatial mean is taken over. P1 measured the consequence:
`[clear_frac, window_span_days]` -- two numbers, no image -- decoded SEASON at
0.646-0.658 balanced accuracy, at or above every foundation model.

Measured here, `cube_mean / cube / linear / weather_full8`:

```
weather (D=64)              +0.066   [-0.159, +0.291]
OBSERVATION CONTROL (D=2)   +0.028   [-0.073, +0.129]
margin                      +0.038
```

**Of the 6.6% weather appears to explain, 2.8 points were already available from
cloud retention alone**, with no weather variable involved. Across the table,
**33 of 54 weather rows sit at or below the observation control.**

**Changed.** `margin_over_control` is computed on `r2_vs_climatology` and is the
column the probe is read on; the raw R-squared is reported beside it and is not
the number to quote. `assert_results_complete` refuses a table missing any of the
four controls, for every (stage, target, fold mode, estimator, feature set) --
without the observation control the headline does not exist, and without the
permutation control there is no empirical zero to read the raw number against.

Two further findings recorded with it:

* **The permutation control is negative, not zero** (-0.07 linear, -0.14 hgb,
  -1.75 mlp on `cell_mean`). A flexible estimator on shuffled features is
  PENALISED rather than neutral -- it fits noise on train and pays on test -- so
  the empirical zero is a ceiling, not a centre. What the control establishes is
  that this pipeline never manufactures positive skill from a destroyed
  association.
* **The 5-variable EO-WM subset is worse than the full 8** (-0.044 against +0.038
  on the primary cell). Their meso channels are precipitation, pressure and
  mean/min/max temperature; dropping wind, humidity and radiation costs the whole
  margin. Reported for commensurability, not as our number.

**Commit.** `13d079d`

---

## 2026-08-10: Phase 1.5, P4: 20 cubes cannot measure a ceiling, and the CI says so

**Assumed.** That Stage A would produce a usable within-season ceiling, narrower
in scope than H1 but quantitative -- "weather explains X% within a season".

**Observed.** The fold-clustered 95% interval **includes zero for every one of
the 54 weather rows**, under every estimator and every fold mode. The effective
sample size is 20 cubes -- one tile, one year, 20 independent weather
realisations -- and 264 frames or 4195 cells do not change that: 16 cells of a
frame share a sky and 13 frames of a cube share a place, and weather is CONSTANT
across the 16 cells of a frame by construction (asserted, because a mis-indexed
cell expansion changes no shape).

Capacity makes it worse rather than better. The small MLP scores -2.6 to -17.9
and loses to its own permutation null in most cells; `hgb` is around zero except
on `cube_p90`. Only the linear model is positive on the primary target. With 211
training rows against 64 features and 20 independent realisations, capacity buys
overfitting.

Two structural limits were measured rather than assumed, and both are properties
of the acquisition geometry:

1. **36 distinct days of year, one orbit lattice.** All 264 rows satisfy
   `doy % 5 == 2`; 261 of 264 share their date with another cube; one date
   carries all 20. Day-of-year is close to a 36-level categorical variable.
2. **The date carries most of the weather.** Within a date, the across-cube
   spread is 9-19% of the total for pressure, radiation and temperature -- a
   150 km tile is one air mass -- but **77% for precipitation**, which is
   convective and local. Roughly 63% of a typical windowed feature is recoverable
   from the date alone.

Together these explain the one result that looks like a failure and is not: the
day-of-year control is ~0 under the linear estimator (worst |r2| 0.019, so the
detrend worked) but reaches +0.42 under the boosted tree. A flexible learner
given only day-of-year fits a per-date mean over 36 dates, which on this subset
is most of a weather model in a different basis. **Cite the linear row as the
sanity check.** Precipitation is doing nearly all of the cross-cube work a
weather model can do here.

**Changed.** Nothing in the code -- there is nothing to fix. Recorded as a claim
boundary: Stage A supports "the within-season weather-attributability ceiling is
not separable from zero at 20 cubes, and the margin over the observation-process
control is +0.04 with an interval spanning zero". It does NOT support a ceiling
value, and **it is not H1**. `print_doy_weather_collinearity` prints both limits
before anything is fitted, so the constraint is in the output rather than in this
file.

One stratum result is worth carrying forward. Under `cube` and `loco`, cropland
(+0.13) and tree cover (+0.10) replicate while **grassland is negative**
(-0.16, -0.08). Grassland NDVI in the Alpine foreland is dominated by MOWING,
which is a management decision and not a weather response -- so the one stratum
where a weather-only ceiling should be expected to fail is the one that fails.
That is a check on the probe passing, not a defect.

**Commit.** `13d079d`

---

## 2026-08-10: Phase 1.5, P4: Stage B is deferred, and the code refuses to let it be substituted

**Assumed.** That a probe blocked on missing data should compute what it can and
label the result.

**Observed.** Labelling is not enough when the two quantities have the same name.
Stage A's proxy and Stage B's real climatology both produce a column called
"NDVI anomaly", both feed the same estimators, and both write to the same CSV.
The failure mode is not that someone computes the wrong one -- it is that Stage
A's number gets quoted as H1 six weeks later by someone reading a table.

**Changed.** Stage B is written in full and gated on detection, not on intent:

* `detect_seasonal_split` reads the per-ROW year from the timestamp, never the
  `year` column (which is parsed from the cube id, i.e. the window START year,
  and reports 2018 for a cube spanning 2017-2020).
* If multi-year cubes are absent, `run_stage_b` prints an explicit deferral
  naming what is missing and exits cleanly. It never falls back.
* If they are present it uses `data.climatology.ndvi_climatology` -- imported,
  never reimplemented -- and `probes.cv` mode `crossed`. **`evaluate_b` takes no
  mode argument**, so Stage B cannot be run under a split that disagrees with its
  own climatology by passing one. Any other mode would pair a test year against a
  climatology built including that year: the same nested leak as fitting the
  Stage A proxy outside the fold, one level up.
* `assert_stage_b_ran_or_deferred` enforces the trichotomy on the TABLE: if
  multi-year cubes exist there must be Stage B rows, they must carry
  `fold_mode == "crossed"` and the real climatology's label; if they do not,
  there must be no Stage B rows at all and every row must be labelled as the
  proxy. Relabelling Stage A as Stage B raises.

**Commit.** `13d079d`

---

## 2026-08-10: the manifest's `year` was the cube id's window-start, not the frame's own year

**Assumed.** That `build_manifest`'s `year` column was the frame's calendar
year, and that `probes.cv`'s year-aware modes (`year`, `crossed`) were ready to
run the moment multi-year cubes existed. Phase 1.3 shipped `crossed` explicitly
as "P4's mode whenever the manifest spans more than one year".

**Observed.** `manifest_rows` parsed `year` from the cube FILENAME's
window-start field and wrote it to every row of that cube. On tile 32UNU the two
are indistinguishable -- every cube is a single 2018 window, so the id-derived
year and each frame's calendar year always agree -- and the defect was invisible
for four phases. On a real seasonal cube (one file spanning 2017-2020) they
disagree on most rows: measured on 30TVN, **1666 of 2092 rows**.

`probes.cv._row_years` caught it immediately and refused, exactly as designed,
naming the fix in its own error message ("Fix build_manifest to derive year per
row from the timestamp"). `tests/test_cv_folds.py` even had a test for the
condition -- `test_year_mode_rejects_a_lying_year_column`, whose docstring reads
"build_manifest CURRENTLY derives year from the cube id". The guard, the
diagnosis and the remedy were all written down in Phase 1.3 and simply never
executed, because nothing in the repo had ever handed them a multi-year cube.

**The consequence is larger than the bug.** `crossed` is the ONLY fold mode that
agrees with a leave-target-year-out climatology, so P4's Stage B -- the stage
that produces H1 -- could not have run on any tile, ever, until this was fixed.
Stage B was written, tested against synthetic multi-year manifests, and gated on
a detector that correctly reported `multi_year=True`; it had simply never met
real seasonal data. It ran for the first time on 2026-08-10.

**Changed.** `year` is derived per ROW from the frame's own timestamp.
`tests/test_encoders.py::test_manifest_year_is_per_row_not_the_cube_id_window_start`
builds a synthetic cube whose frames straddle 31 December while its filename
says 2018, and pins the fix; it was verified to FAIL against the previous code
and pass against the new one, rather than being written after the fact and
assumed to discriminate.

**The general lesson, and it is the third of its kind.** Phase 1.2: a resumable
cache is a correctness hazard unless it can prove its own version. Phase 1.3: a
cache on shared storage cannot be trusted to contain only what we put there.
Earlier today: a derived column cannot be validated against the table it lives
in. Now: **a guard that has never fired on real data is a hypothesis, not a
control.** Three separate mechanisms here -- the refusal, the test, and the
docstring naming the fix -- all correctly described a bug that stayed live for
four phases, because the input that would trigger them did not exist in the
repo. Exercising every guard against real data of the shape it was written for
is a task in its own right.

**Commit.** `13d079d`

---

## 2026-08-10: tile 32UNU has NO seasonal coverage, so H1 is not computable there at all

**Assumed.** That the seasonal split was a download away, and that Stage B was
blocked on effort rather than on availability. Every prior entry treats "pending
the seasonal download" as a scheduling matter.

**Observed.** It is not available for this tile and never will be. Read from the
bucket directly:

```
seasonal split covers 15 tiles:
  29SQC 29TPF 29UMV 30TVK 30TVN 31TCF 31TFK 31UCS
  31UEQ 31UGQ 31UGU 32VNM 32VPN 33TWN 33VXK
32UNU present: NO      32TPT: NO      33TUN: NO    (no Bavaria-area tile)
```

So the leave-target-year-out climatology -- and therefore **H1 as originally
scoped** -- cannot be computed on the tile this project is framed on. Not
deferred: unavailable.

A second correction fell out of the same listing. The 20 working cubes have been
called "the extreme split" since 2026-08-02, including in
`docs/specs/phase1_3_cv.md`, which reserved them as "P3's severity/dynamic
subset". They are not: the real `extreme` split covers **32UMC, 32UNC, 32UPC,
32UQC**, and 32UNU is not among them. The 20 cubes came from `--split train`
(the downloader's default), where 32UNU holds 192. Nothing measured is affected
-- every property the cubes are actually used for (single year, one tile, 264
retained frames) was verified directly from the files -- but the label was
wrong, and the plan built on it reserved a subset that was never these cubes.

**Changed.** Scope, in three parts:

1. **32UNU's ceiling is the within-season proxy, permanently.** It is reported
   as such, with the dataset limitation stated: GreenEarthNet provides no
   multi-year product for this tile, so the leave-year-out baseline used
   elsewhere in the literature is not computable here.
2. **A seasonal-covered tile is used as a VALIDATION SITE, not a second case
   study** -- `scripts/validate_proxy_climatology.py` runs Stage A and Stage B
   over the same cubes on 30TVN and compares. The question it answers is "is the
   proxy any good", which is what makes 32UNU's permanent proxy number
   defensible; it is deliberately not a second geography for the paper.
3. **Growing the 32UNU cube pool is now unblocked**, since the subset
   `phase1_3_cv.md` reserved for P3 was never these cubes and the real extreme
   split is untouched.

**Commit.** `13d079d`

---

## 2026-08-10: H1 is year-limited, so the paper reports the proxy ceiling and the leave-year-out check as structurally underpowered

**Assumed.** That Stage B's wide interval on the 25-cube pilot was a sample-size
problem of the ordinary kind -- pull more cubes, the interval tightens, the
proxy gets validated or refuted. The whole point of scaling 30TVN from 25 to 87
cubes was to settle it.

**Observed.** It does not tighten, and it cannot. `probes.cv.crossed_folds`
clamps the fold count to `min(n_years, n_cubes)`; GreenEarthNet's seasonal cubes
span 2017-2020, so **k = 4 whatever the cube count**. 3.5x the cubes moved the
interval width from 1.611 to 1.433, 11%.

```
                  folds   weather r2_vs_clim          CI width   per-fold sd    MDE
PROXY  (cube)         5   +0.192 [+0.139, +0.245]       0.105       0.042      0.053
REAL   (crossed)      4   -0.136 [-0.852, +0.581]       1.433       0.450      0.631
Stage B per fold: -0.385  -0.642  +0.192  +0.292   (one held-out year each)
```

The per-fold values are four draws of interannual variability and they disagree
in sign. That spread is the quantity being averaged, a t interval on 3 df around
it is irreducibly wide, and detecting the observed effect at 80% power would
need ~87 folds against a supply of 4.

This is not specific to our probe or our tile. Any leave-target-year-out
evaluation on this benchmark, scored under a protocol that holds the year out
honestly, inherits a 4-sample interannual denominator.

**Changed.** The claim structure, in three parts:

1. **The reported ceiling is the within-season PROXY**, on 32UNU, with the
   dataset limitation stated. It is well-powered and cube-limited: 87 cubes give
   a CI width of 0.105 against 20 cubes' 0.450, so **scaling 32UNU from 20 to
   its available 192 is the highest-value cheap action left for P4**, and it is
   now unblocked (the cubes reserved for P3 were never these).
2. **The leave-year-out run is reported as a consistency check that is
   structurally underpowered**, with the fold-count mechanism and the power
   calculation shown, not as a validation that passed. The proxy and the real
   climatology differ by 0.328 in point estimate; their intervals overlap only
   because one of them is enormous. `scripts/validate_proxy_climatology.py`
   prints "that is NOT evidence the proxy is sound" in its own output, so the
   overlap cannot be quietly read as agreement.
3. **H1 as originally scoped is retired** as a precise deliverable on this
   benchmark. What replaces it is a bound plus an explicit statement of why a
   point estimate is not available -- which is a stronger contribution than a
   number whose error bars nobody computed.

**Why this is worth the block it cost.** The result arrived from running the
experiment that was supposed to be routine. Had we scaled 32UNU first and
reported a precise proxy ceiling without ever running Stage B at scale, the
obvious reviewer question -- "how do you know your proxy approximates the
standard baseline?" -- would have had no answer, and the honest answer turns out
to be interesting: on this benchmark, nobody can know, and here is the
arithmetic.

**Commit.** `e5b7f43`

---

## 2026-08-11: Stage B's climatology has a second-order leak path, recorded before any H1 is quoted from it

**Assumed.** That holding the cube out under `crossed` made Stage B's
leave-target-year-out target fully clean, because the climatology for a test row
pools only that cube's other years and the whole cube is out of train.

**Observed.** True for TEST rows, not for TRAIN ones. A train cube's rows in
year Y' are detrended by a climatology pooling that cube's other years -- which
can include the fold's HELD-OUT year Y. So test-year information reaches the
train TARGETS through other cubes.

It is second order: test rows come from held-out cubes whose own climatologies
never touch train, and the climatology averages over years, so one year moves it
by roughly 1/(n_years - 1). With four years that is not negligible.

**Changed.** Nothing yet, deliberately -- Stage B is structurally underpowered
on this benchmark (see the year-limit entry above), so no number is being quoted
from it and fixing a second-order bias in an estimate whose interval spans 1.4
would be misplaced effort. What changed is that it is written down at the site
that would have to change: `probes/p4_ceiling.run_stage_b` carries the mechanism
and the cost of the fix (rebuild each cube's climatology per FOLD, excluding
every held-out year rather than only the row's own -- a climatology
recomputation per (cube, fold) instead of per (cube, year)).

**The condition.** This must be fixed and quantified BEFORE any H1-style number
is quoted from Stage B, on this benchmark or a future one with more years. It is
recorded here rather than in a TODO because the cost of discovering it after a
number is published is a retraction.

**Commit.** `68cce76`

---

## 2026-08-11: Phase 1.6, P2: the gap between two frames is measured in DAYS, and a run-time check proves the right column was used

**Assumed.** That naming the two axes in `encoders/manifest.py`'s docstring, and
fixing the E-OBS join that conflated them, had retired the hazard.

**Observed.** It had not. P2's whole Part B is built on "the gap between
consecutive retained frames", and there are THREE plausible readings of that
phrase in this manifest:

```
gap_days        daily_axis_index       min  5   median 10   max 35
gap_acq_steps   original_axis_index    min  1   median  2   max  6
frames between  array position         always 1
```

Only the first is a gap. `original_axis_index` is the EMBEDDING JOIN KEY and
counts acquisitions; the two disagree on **244 of 244 pairs**, by a factor of
five at the median — one Sentinel-2 orbit lattice. And the wrong column is
finite, integer, in range, strictly increasing within a cube, and *correlated
with the right one*, so nothing about its values looks wrong.

Gap length is also not an innocent covariate: a long gap exists because the
frames between it were dropped for cloud, cloud is precipitation, and
precipitation is weather. A gap measured in acquisitions launders that
dependence into the horizon itself.

**Changed.** `probes.p2_deltas.DeltaPairs` carries all three readings, and
`assert_gap_axes_disagree` runs on the real pairs before anything is fitted. It
asserts they **materially disagree** — not that the right one was used, which is
unfalsifiable from inside, but that the candidates are far enough apart for the
check to be able to tell them apart at all. A test that cannot distinguish the
correct column from the wrong one proves nothing.

Two things make this a control rather than a hypothesis. The synthetic fixture
in `tests/test_p2_deltas.py` uses `day_step=10, acq_step=2` — the same 5×
disagreement as the real data, so a probe reading the wrong column fails there
too; a fixture where the axes coincided would let the bug through every test in
the file. And the guard is exercised **against the wrong column**:
`test_the_axis_check_fails_when_the_gap_came_from_the_join_key` builds exactly
the bug and asserts the refusal names it.

**Precedent.** HANDOFF_P2 §4: "a guard that has never fired on real data is a
hypothesis, not a control." The `year` bug sat behind a correct refusal, a
passing test and a docstring naming the exact fix for four phases.

---

## 2026-08-11: Phase 1.6, P2: the gap-length-alone control is mandatory, and on the magnitude target it wins

**Assumed.** That the delta probe's controls were the raw-pixel baseline and the
permutation floor, and that a control built from elapsed time alone was a
formality.

**Observed.** It is not a formality; it is the strongest model in the table on
one of the two targets. Fitted inside each fold and scored on held-out pairs,
`gap_days` alone reaches Spearman **+0.209 [+0.066, +0.351]** against the
magnitude of common-masked NDVI change — **above every encoder**, including the
raw-pixel baseline (+0.078) and the best network (DINOv2, +0.174). On the sign
target it is **−0.118**, and every encoder clears it by a wide margin.

So the two targets have opposite readings, and only the control separates them.
Quoting DINOv2's +0.174 on magnitude without it would report a positive
correlation with elapsed time as evidence that the representation tracks the
rate of change.

**Changed.** `margin_over_control` is the headline, exactly as in P4; the raw
Spearman is reported beside it and is not the number to quote.
`assert_degenerate_control_present` requires the control for every fold mode,
target, feature level and read-out.

**Why degree 2 and not 1.** Spearman is rank-based, so a monotone fit has the
same score as the raw variable — a straight line in `gap_days` would make the
control's fitted coefficients irrelevant to its own reported value, and the
training-poison test vacuous. Degree 2 is also the honest shape: NDVI change
over a gap saturates.

**Precedent, now three phases deep.** P1: `[clear_frac, window_span_days]`, two
numbers and no image, decoded season at 0.646–0.658. P4: the same control still
absorbed a real share at 115 cubes and did not wash out. P2: gap length beats
every encoder on magnitude. **A target correlated with elapsed time carries a
competitor that needs no image**, and P3's horizons are defined in days.

---

## 2026-08-11: Phase 1.6, P2: the multi-image encoder is excluded from the delta ranking, for a sharper version of P1's reason

**Assumed.** That `satlas_s2_swinb_mi_rgb`'s `si_comparable=False` flag carried
the same weight in P2 as in P1 — a caveat about an ill-defined label.

**Observed.** It is worse here, and structurally so. The encoder pools **8
retained frames**. Two CONSECUTIVE embeddings therefore share **up to 7 of their
8 source frames**, so `E(t+1) − E(t)` is largely the difference between
"frames 1–8" and "frames 2–9": a sliding-window increment, not a state change
over the gap. Its measured delta scores are the lowest in the table
(sign +0.163 against DINOv2's +0.536), which is exactly what a mostly-cancelling
difference predicts and is *not* evidence about its representation.

**Changed.** Its scores are REPORTED — excluding it from the table would hide
the positive control — and it is flagged `si_comparable=False` on every row it
owns in both parts, and excluded from the ranking used for the structural
hypothesis. `assert_mi_flagged_and_excluded` enforces both halves, and
`structural_hypothesis` raises if it ever appears in the ranking.

**The distinction that matters.** This is a property of its INPUT WINDOW, not of
its training objective, so including it in a comparison *about* training
objectives would answer a different question than the one asked.

---

## 2026-08-11: Phase 1.6, P2: K2 is recorded twice, because `raw_features` contains the target

**Assumed.** That `raw_features` — the repo's mandatory not-a-network baseline —
could serve as the K2 comparison for "can this encoder's latents recover current
NDVI at all", as P2's spec names it.

**Observed.** At the MATCHED level it is the identity function on that target.
`encoders.raw_features.RAW_FEATURE_NAMES` ends in `NDVI_mean, NDVI_std,
NDVI_p10 … NDVI_p90`, computed over the same grid cell. Measured on this subset:

```
cell_mean  from grid_cell features   R2 +1.0000     <- the baseline IS the target
cube_p90   from pooled    features   R2 +1.0000
cube_mean  from pooled    features   R2 +0.9998
cube_mean  from grid_cell features   R2 +0.4403     <- PRIMARY, and not matched
```

The degeneracy is a property of the level PAIRING, not of the baseline as such:
predicting a cube's mean from one cell's statistics is not the identity, so the
primary configuration is a fair gate. The secondary views are not, and a verdict
read off them would measure which columns the baseline was handed.

**Changed.** Two verdicts, on every row. `k2_verdict` against `raw_features`,
exactly as specified. `k2_verdict_band_matched` against `raw_rgb_only` —
B02/B03/B04 statistics only, the same three bands every network encoder
receives, never degenerate at any level. **The band-matched verdict is the one
that decides P3 inclusion.** This is the same asymmetry P1 found and the same
fix: a column slice of an array already on disk, derived from the baseline's own
feature names and asserted to partition them, so a rename cannot repoint it.
`test_the_band_matched_columns_exclude_every_ndvi_column` fails if an NDVI or
B8A column ever enters the slice, and fails the other way if `raw_features` ceases
to carry NDVI at all — at which point this entry describes a baseline that no
longer exists.

---

## 2026-08-11: Phase 1.6, P2: a lossy verdict is not a rejection unless the paired difference is separable

**Assumed.** That "does not beat the baseline ⇒ mark `audited: lossy` ⇒ exclude
from P3" was a complete rule.

**Observed.** At 20 cubes it decides on noise. `dinov2_vitb14` scores **+0.4346**
against the baseline's **+0.4403** — 0.006 apart — with marginal cube-clustered
intervals of `[+0.144, +0.726]` and `[+0.169, +0.711]`, roughly 0.6 wide.
Applying the rule literally would drop DINOv2, the strongest encoder on the
entire delta probe, from P3 on a sixth of a percentage point.

**Changed.** A third column, `k2_separable`, computed as the **paired** per-fold
difference (encoder − baseline within each fold, then a cube-clustered interval
on that difference). Pairing is legitimate and much tighter here because both
models are scored on identical folds, and the difference is what the verdict is
actually about — comparing two overlapping marginal intervals by eye is the
wrong test.

The exclusion rule is now the AND: an encoder leaves P3 only if it is
`audited: lossy` **and** separable. Where the paired interval spans zero,
"audited: lossy" means *did not beat the baseline*, not *is measurably worse
than it*, and the table says so rather than leaving it to be argued.

**What this does not do.** It does not weaken the gate. The one encoder that is
both lossy and separable — `satlas_s2_swinb_mi_rgb` — is still excluded. It
prevents an exclusion that the data does not support.

**And it turned up something the point estimates hid.** Once the paired
intervals were computed, **no single-image encoder separates from the baseline
at all**:

```
satlas_s2_swinb_rgb     +0.105   [-0.118, +0.327]
imagenet_vit_b16        +0.081   [-0.061, +0.223]
dinov2_vitb14           -0.006   [-0.142, +0.131]
satlas_s2_swinb_mi_rgb  -0.363   [-0.592, -0.134]    <- the only separable one
```

So K2 on this subset is a **floor check** — "nothing is catastrophically lossy"
— and cannot be more than that. The `+0.545` vs `+0.440` spread at the top of
the table is not an encoder ranking and must not be reported as one. That is a
statement about the benchmark's resolving power at 20 cubes, and it is the same
statement P1 made about the same encoders.

---

## 2026-08-11: Phase 1.6, P2: common-masked pixel survival is reported per gap length, and it is not monotone

**Assumed.** That the intersection of two frames' valid masks would shrink with
gap length, and that long-gap pairs might have to be dropped for lack of shared
pixels.

**Observed.** It does not shrink, and the shape is the opposite of the
expectation. Over 244 pairs, **none** collapses to zero shared pixels; the
median is 88.8% of 16384 and the minimum 27.2%. By gap:

```
gap (d)    5      10     15     20     25     30     35
pairs    106      67     31     15     14     10      1
median  0.895   0.839  0.783  0.900  0.953  0.985  0.810
```

Survival is *worse* at 15 days than at 30. The mechanism is selection: a long
gap exists **because** the frames between it were dropped for cloud, which
leaves the two surviving endpoints unusually clear.

**Changed.** `summarise_pixel_survival` reports the table rather than a summary
statistic, and it is written to `data/phase1_6/results/`. Nothing is dropped for
low survival.

**The consequence for P3.** Do not model pixel survival, or any data-availability
quantity, as a decreasing function of horizon. On this benchmark the
relationship is non-monotone and driven by the observation process.

---

## 2026-08-11: Phase 1.6, P2: the control's value is written onto every row, not only into its own

**Assumed.** That emitting a control under each filter label, as P4 does, plus an
assertion that the copies agree, was enough to make "filtering the CSV cannot
drop the control" true.

**Observed.** It is enough only for the labels the control is emitted under. A
reader who filters to `encoder == "dinov2_vitb14"` sees no control row at all,
because the control's `encoder` is `"none"` — and a margin column they cannot
check is a margin they have to trust.

**Changed.** `add_margins` writes `control_score` (and `control_kind`) onto
**every** row. Whatever a reader filters to, the control travels with the row.
`assert_control_identical_across_views` then checks both that the control rows
agree across the labels they are duplicated under, and that every row's carried
copy is digit-for-digit the control's own value.

**And the key is stated as data, not as prose.** `_CONTROL_KEY` records that the
gap control is invariant to encoder / feature level / read-out but NOT to fold
mode, while the retention control also varies with feature level (frame
`clear_frac` versus per-cell `grid_clear_frac`, both of which P1 reports on
purpose). One assertion covers two controls with different invariances and no
special case. **`fold_mode` is part of the key, not one of the invariant
labels**: a control evaluated on a different set of held-out cubes is a
different number, and asserting one value across modes would be a false
identity rather than a consistency guarantee.

---

## 2026-08-11: Phase 1.6, P2: the structural hypothesis is NOT DETERMINABLE, because the ordering flips between fold modes

**Pre-registered hunch.** research_plan_v3 §3/P2: augmentation-invariance
training (DINOv2) may discard state-change more than reconstruction-style
training (Satlas).

**Observed.** On the primary delta configuration (`cube_mean` / sign / pooled /
linear) the SI-comparable ranking under `cube` folds is

```
1. raw_features          +0.801  [+0.753, +0.849]
2. dinov2_vitb14         +0.536  [+0.397, +0.674]
3. satlas_s2_swinb_rgb   +0.481  [+0.404, +0.557]
4. imagenet_vit_b16      +0.458  [+0.318, +0.597]
```

which contradicts the hunch — DINOv2 is *above* Satlas. But the ordering **does
not survive the fold mode**:

```
cube            dinov2 +0.536  >  satlas +0.481      hunch fails
loco            satlas +0.531  >  dinov2 +0.481      hunch holds
spatial_block   satlas +0.489  >  dinov2 +0.473      hunch holds
cell level      satlas +0.4596 ≈  dinov2 +0.4576     indistinguishable
```

The two encoders the hunch is *about* swap places, and every interval overlaps
every other. **This is the same pair P1 could not rank** (0.387 vs 0.386, a rank
that moved with a scipy version). Two phases, two probes, same conclusion: at
20 cubes of one tile these two encoders are not separable.

**Changed.** `structural_hypothesis` now ranks under **every** fold mode, not
just the primary, and returns `supported=None` when the pair's ordering is not
stable — neither confirmed nor refuted. Reporting `False` from the `cube` row
alone would have been as wrong as reporting `True` from the `loco` row, and the
first draft of this phase did exactly that before the other two modes were
looked at. `order_stable_across_fold_modes` and `verdict_by_fold_mode` are
returned so a reader can see the disagreement rather than take the summary.

**The caveat travels with every call:** 2 EO-relevant single-image points plus 2
anchors, one tile, one year, 20 cubes, overlapping cube-clustered intervals.

**Why it is recorded at all.** A hunch that is quietly dropped is a hunch that
gets re-proposed next phase. And "not determinable" is a finding about the
benchmark's resolving power, which is reusable; "false" would have been a claim
about DINOv2, which the data does not support.

---

## 2026-08-11: P2 at 115 cubes: three of the phase's four headline claims changed, and the two that changed most were the confident ones

**Assumed.** That the 20-cube P2 table was underpowered in a *uniform* way --
intervals too wide to separate encoders, but point estimates roughly right, so
scaling would sharpen the same picture.

**Observed.** Scaling did not sharpen the picture; it inverted two of its
claims, and in both cases the 20-cube version was confidently wrong rather than
merely wide.

1. **"K2 is a floor check, not a ranking."** At 115 cubes it IS a ranking:
   satlas SI, imagenet and dinov2 all separably beat `raw_features` on the
   paired per-fold difference. Every interval collapsed (DINOv2's by 6x,
   ImageNet's by 11x).
2. **"`satlas_s2_swinb_mi_rgb` is excluded from P3."** Retracted. It moved from
   -0.363 `[-0.592, -0.134]` (separably lossy) to -0.069 `[-0.192, +0.053]`
   (not separable) and passes band-matched. **No encoder is excluded.**
3. **"The gap-length control beats every encoder on magnitude."** Wrong, and
   wrong in the direction that flattered the control. The control itself was a
   small-sample artefact: +0.209 at 20 cubes, **+0.063 `[-0.004, +0.130]`** at
   115. The corrected finding is not that encoders win but that *nobody*
   recovers magnitude -- every rho is +0.06 to +0.12 and three of four encoders
   flip margin sign across fold modes.
4. **"The structural hypothesis is not determinable."** Now determinable and
   REFUTED: the ordering is identical under all three fold modes.

Only the SIGN result survived unchanged, and it survived well -- margins move by
less than 0.05 across fold modes.

**Changed.** Every quotable number in log.md, README.md and HANDOFF_P3.md now
comes from the 115-cube table. The 20-cube entry is kept in full with a
SUPERSEDED banner naming exactly which two claims died, because the comparison
is the evidence for the methodological point below.

**The methodological point, which is the reusable part.** `k2_separable` -- the
paired per-fold interval added late in the 20-cube run precisely because a
0.006 point estimate was about to exclude DINOv2 -- was right in BOTH
directions. It stopped a false exclusion (DINOv2, now separably ABOVE the
baseline) and it flagged the one exclusion that later reversed (the MI encoder,
which was separable at 20 and is not at 115). A binary verdict read off a point
estimate would have been wrong twice.

**And the trap this phase walked into anyway.** A CONTROL can be a small-sample
artefact exactly as a treatment can. Nothing in the 20-cube protocol was wrong:
the gap control was fitted in-fold, poisoned in both directions, reported with
a cube-clustered interval. It was still inflated by a factor of three, and
because it was a control, its inflation produced a *negative* headline that
looked conservative and therefore trustworthy. **A conservative-looking result
is not a safe result.** Controls need the same sample size the treatments do.

**Commit.** `930cc01`

---

## 2026-08-12: Phase 1.8, P3: the multi-image encoder gets ONE embedding as its context, and the code refuses a stack

**Assumed.** That "context = the k=3 most recent retained frames" was a uniform
rule, and that applying it to `satlas_s2_swinb_mi_rgb` alongside the four
single-image encoders would simply give it a wider input.

**Observed.** It would give it the SAME input three times. The MI encoder's
embedding at t already max-pools up to 8 preceding retained frames -- a lookback
of 0 to 105 days on this subset, median 55 -- so consecutive MI embeddings share
up to 7 of their 8 source frames. A k=3 stack is therefore three heavily
overlapping summaries of one window presented to the read-out as three
observations, and it double-counts the lookback the encoder was built to
contain. This is the sharper version of the same fact P1 flagged (a single
"month" label is ill-defined for a 105-day lookback) and P2 flagged harder
(`E(t+1) - E(t)` is a sliding-window increment, not a state change).

Nothing about the failure is visible downstream. The stacked design matrix has
the right shape, the right dtype, no non-finite entries, and produces a
plausible R-squared.

**Changed.** `context_frames_for(encoder)` is the single source of the k: 3 for
every single-image encoder, 1 for the MI encoder. `context_block` REFUSES a
3-frame request for the MI encoder with a message that states the reason, rather
than silently honouring or silently ignoring it -- a silently-ignored argument is
indistinguishable from one that was honoured, and the row would then claim a
3-frame context while carrying a 1-frame one. Every MI row carries
`si_comparable=False` and `context_frames=1`, and
`assert_mi_flagged_and_single_frame` asserts both halves on the table.

**What was deliberately NOT done.** The MI encoder is not dropped, and its rows
are not excluded from the results table. P2's 115-cube run RETRACTED its
exclusion (the encoder is no longer separably lossy on gate K2), so excluding it
here would re-impose a verdict the data withdrew. It is reported, flagged, and
kept out of any single-image ranking.

**A consequence worth stating.** The k>=3 rule defines the ROW SET, and it is
applied ONCE before any encoder is chosen -- so the MI encoder is scored on
exactly the rows the others are scored on, even though it needs only one of the
three. Letting its single-frame context buy it extra rows would make the table's
`n` depend on which column a reader filtered to.

**Commit.** `33fee78`

---

## 2026-08-12: Phase 1.8, P3: persistence is scored through common-masking, which makes its residual P2's delta by construction

**Assumed.** That persistence -- "predict NDVI(t+Delta) = NDVI(t)" -- was the one
baseline too simple to get wrong, and that it could be scored by taking each
frame's own spatial mean.

**Observed.** That is exactly P2's differencing trap, one level up. Frame t's own
mean is over t's valid pixels; frame t+Delta's own mean is over ITS valid pixels.
The difference between them is partly a comparison of two different pieces of
ground, attributed to time. The two answers agree only when the masks agree --
i.e. exactly when the distinction does not matter -- so no spot check finds it,
and on this subset the intersection keeps a median of only 82-88% of pixels.

It matters more for persistence than for anything else in the table, because
persistence's error IS a frame-to-frame difference. Every other row's error is a
model residual, in which a mask mismatch is one more noise term; in persistence
it is the whole quantity.

**Changed.** The TARGET itself is common-masked: `y` is the aggregate of NDVI at
t+Delta over the pixels valid in BOTH frames, and persistence predicts the
aggregate of NDVI at t over that same intersection. The persistence residual is
then `p2_deltas.common_masked_delta`'s change, **by construction rather than by
coincidence**, and every other model in the table is scored against the same
target definition, so the comparison is like-for-like.

`common_masked_levels` adds the two LEVELS -- which a forecast needs and a delta
probe did not -- on top of P2's imported, unmodified `common_masked_delta`, and
then PINS them to it: for all three aggregations it asserts
`level_b - level_a == the delta p2 returned` to 1e-12 on every pair. The
intersection arithmetic is not duplicated, and the two cannot drift.

**The test that can fail.** A synthetic pair with deliberately non-overlapping
masks, where the naive answer and the common-masked answer differ by more than
1e-6; the test asserts the residual equals p2's delta AND does not equal the
difference of the two frames' own means. A fixture whose masks agreed would pass
both ways and prove nothing.

**Commit.** `33fee78`

---

## 2026-08-12: Phase 1.8, P3: the horizon drop policy is a boundary drop, never a wrap, and the tolerance is measured rather than chosen

**Assumed.** That "the nearest retained acquisition to t + Delta_days" needed a
tolerance, and that +/-3 days was a mild, unimportant setting.

**Observed.** Two things, and the second changes how the first is read.

1. **The tolerance does no work here.** Every retained frame on this subset
   satisfies `doy % 5 == 2` -- one Sentinel-2 orbit lattice -- so every
   realisable horizon is a multiple of 5 days, and so are all four nominal
   horizons. A +/-3 day tolerance therefore accepts EXACT matches only: +/-2 and
   +/-3 select the identical 1653 rows, 0 of them off the nominal horizon.
   Loosening to +/-5 admits the neighbouring lattice point, which nearly doubles
   the row count (1653 -> 2936) and moves 1283 rows off their nominal horizon --
   by 5 days, which at Delta = 5 d is 100% of the horizon.
2. **The window boundary is where the rows actually go.** Cubes cover about 150
   days and their retained frames span a median of 135, so a 100-day horizon has
   almost nowhere to land: 196 rows over 94 cubes, against 518 rows over 115
   cubes at 5 days. **21 cubes contribute no 100-day pair at all.**

**Changed.** A row with no retained frame within the tolerance of t + Delta,
inside the SAME cube, is DROPPED. It is never wrapped to another cube, never
extrapolated past the cube's window, and never filled. `TOLERANCE_DAYS = 3` is
kept -- exact matching is what it means here -- and
`horizon_tolerance_sensitivity` prints the row counts and the off-nominal counts
at several tolerances so the choice is a measurement in the run log rather than
an unexamined constant.

The shrinkage is REPORTED, not hidden: `print_horizon_retention` prints it before
anything is fitted, `n_retained` and `n_cubes` travel on every row of the CSV,
and `assert_retention_shrinks` asserts it -- because if the count did NOT fall at
100 days over 135-day cubes, the row set would be including pairs the cubes
cannot support, and a wrap, an extrapolation or a too-loose tolerance is silent
in the score.

**Where the assertion binds, and why that is not a loophole.** A horizon that is
a small fraction of the cube's covered window loses nothing at the boundary and
has no reason to shrink. The requirement therefore applies when the longest
horizon reaches at least half the median cube day-span (100/135 = 74% here), and
otherwise the check PRINTS that it did not apply and gives the fraction. That is
the "or explained if it does not" half of the rule expressed as data. A blanket
requirement would be a false invariant that a future run with shorter horizons
would have to work around, and worked-around assertions stop being read.

**Commit.** `33fee78`

---

## 2026-08-12: Phase 1.8, P3: three cloud-contaminated frames set this table's R-squared, and they are reported rather than dropped

**Assumed.** That the clear-fraction filter plus the cached per-pixel mask left a
target clean enough that an R-squared over ~500 rows was a property of the bulk
of them.

**Observed.** It is not. **Three frames of 1580 carry a common-masked cube-mean
NDVI below ZERO in midsummer** -- days of year 177, 202 and 202, at clear
fractions of 0.587, 0.624 and 0.627, in three different cubes that share a pixel
column near the western edge of the tile. Bare soil is about 0.15 and dense
summer canopy about 0.85; NDVI near zero in July over Allgau farmland is cloud,
not vegetation. Both filters passed them: the frames are 59-63% "clear" so the
clear-fraction rule kept them, and the per-pixel mask marked the surviving pixels
valid, so the contamination sits INSIDE the pixels this phase, P2 and P4 all
treat as good data.

Those three frames produce about five forecast rows whose persistence error is
0.66 to 0.86 NDVI, against a median persistence error of 0.022 at Delta = 5 d.
**The worst 1% of rows carry 71% of the persistence sum of squares at Delta = 5
d**, 40% at 25 d and 33% at 50 d. An R-squared computed over them is a statement
about three frames.

**Changed, and what was deliberately NOT changed.** Nothing is dropped. A
plausibility screen here would be a filter P2 and P4 do not apply, which would
make P3's row set incomparable to theirs -- and their numbers rest on the same
three frames, so the right fix is in the SHARED frame-target code, not in a
private filter in this phase. Instead:

* `print_target_outliers` measures the concentration and names the three frames,
  before anything is fitted, with the physical floor stated.
* `sse_share_top1pct` travels on EVERY row of the CSV, so any R-squared can be
  read against how concentrated it is.
* `medae_pooled` -- a MEDIAN absolute error, which five rows cannot set -- sits
  beside every mean.

**Recorded as an open item for the shared code**, not for P3: a physical
plausibility screen at frame level in `p4_ceiling.cube_frame_targets` would
change P2's magnitude table and P4's ceiling as well as P3's headline, and that
is a decision about all three phases.

**The reusable point.** P2's lesson was that a CONTROL can be a small-sample
artefact. This is the neighbouring one: a squared-error metric can be an
OUTLIER artefact, and it looks exactly like a measurement -- finite, in range,
with a plausible interval and a cube-clustered CI. The diagnostic that catches it
is not a better model, it is printing which rows the score came from.

**Commit.** `33fee78`

---

## 2026-08-12: Phase 1.8, P3: the margins are taken on the POOLED out-of-fold R-squared, because leave-one-cube-out has two-row folds

**Assumed.** That P3 would report the mean of per-fold R-squareds with a
fold-clustered interval, as P1, P2 and P4 all do.

**Observed.** That works in those phases because a fold holds dozens of rows.
Here it does not. A P3 row is a (cube, t, Delta) triple, and a cube contributes
about 4-5 of them at Delta = 5 d and 2 at Delta = 100 d. Under leave-one-cube-out
a test fold is therefore two to five rows, and an R-squared against the mean of
three points is not a measurement -- `r2` is NaN for essentially every LOCO fold,
so `r2_mean` is NaN for every LOCO row and a margin built on it would go missing
for an entire fold mode.

**Changed.** Every margin -- `margin_over_control`, `margin_over_persistence`,
`margin_over_climatology`, `margin_over_horizon_control`,
`margin_over_permutation`, `margin_over_band_matched` -- and `control_score`
itself are taken on `r2_pooled`: the R-squared over every held-out prediction,
each used exactly once. It is defined wherever any row was predicted at all.

Its uncertainty is a **delete-one-FOLD jackknife**: recompute the pooled
statistic with each fold's rows removed and form a t interval from the spread.
Folds hold disjoint sets of cubes, so deleting one deletes a CLUSTER -- the same
clustering `p4_ceiling.fold_clustered_ci` applies to a mean, extended to a
statistic that is not one.

`r2_mean` and its fold-clustered interval stay on the table beside it and are the
right number to quote under `cube` and `spatial_block`. Where `r2_mean` is NaN
the row carries `n_folds_nan`, and `assert_results_complete` asserts that a NaN
mean is always accompanied by a non-zero NaN-fold count -- so the NaN is a
reported fact rather than a hole, and `print_headlines` prints `n/a` in that
column rather than a mean over whichever folds happened to survive.

**Commit.** `33fee78`

---

## 2026-08-12: Tier 1, P3: the ridge penalty is SELECTED per fold, and both rules stay in the table

**Assumed.** That `p4_ceiling.make_estimator`'s rule -- alpha = D, fixed a
priori, applied to standardised features -- was the safe choice for P3 too. It
needs no selection loop, and P1 had to prove a nested tuning loop clean with a
poisoning test; having no loop at all is strictly the stronger position.

**Observed.** The rule sets the penalty from the WIDTH of the design, and P3's
designs are not remotely comparable in width. Across the encoder views the
ridge's alpha runs from **79** (the band-matched `raw_rgb_only` baseline: 7
percentiles x 3 bands x 3 context frames, plus 16 weather columns) to **11536**
(a k=3 DINOv2 context) -- a **146-fold range**, and it is set by the
architecture's embedding dimension rather than by anything about the data.

That would be tolerable if the comparison ran across some other axis. It does
not: the comparison this phase exists to make is precisely between the narrowest
row in the table and the widest ones. Under alpha = D the band-matched baseline
is the most heavily penalised row relative to what it can express, so "the
hand-crafted baseline is not beaten" and "the hand-crafted baseline was
handicapped" are not distinguishable from the table.

Under a penalty selected by nested CV on the training fold, the band-matched
baseline moves **+0.483 -> +0.597** at Delta = 5 d and **+0.574 -> +0.693** at
Delta = 25 d, while the encoder rows barely move. The rule was worth more than
0.1 R-squared to one row and nearly nothing to the others, which is the
signature of a hyperparameter that is measuring design width.

**Changed.** Every ridge row is emitted TWICE, under `alpha_rule`:

* `fixed_alpha_D` -- P4's rule, unchanged. These are the published numbers.
* `nested_cv` -- `p2_deltas.select_ridge_alpha`, imported. The inner split is
  `probes.cv.folds` on the OUTER TRAINING FOLD's sub-manifest under cube
  grouping, ties break toward the stronger penalty by a stated rule, and it
  already carries poisoning tests in both directions.

Non-ridge rows say `not_a_ridge` rather than claiming a rule they do not have,
and `assert_alpha_rules_present` refuses a table where a configuration exists
under one rule and not the other. `alpha_per_fold`, `alpha_median` and
`n_folds_alpha_at_grid_edge` are on every row, so a penalty pinned by the end of
the grid is visible rather than assumed away -- that is P1's lesson, where the C
grid stopped at 1 and selected there in 57% of folds.

**The fixed-alpha rows are NOT deleted**, and that is the load-bearing half.
They are what the 2026-08-12 table is; a run that dropped them would not be
comparable with the run it supersedes, and the comparison is the point.

`p4_ceiling.make_estimator` gained an `alpha` override for the ridge and ONLY
the ridge -- passing it for `hgb` or the MLP raises, because a silently ignored
hyperparameter fits a different model than the caller asked for.

**Commit.** `51eb72a`

---

## 2026-08-12: Tier 1, P3: "X beats Y" is the PAIRED per-fold difference, and a marginal-CI comparison is refused by an assertion

**Assumed.** That reporting each row's R-squared with its own fold-clustered
interval was enough to compare two rows: print both, see whether the intervals
overlap.

**Observed.** It is not, and P2 had already established why on gate K2. Two
rows of this table are fitted on the SAME folds and scored on the SAME held-out
observations, so most of the width of each marginal interval is a fold effect the
two SHARE. Comparing the marginals counts that shared variation twice and
answers a question nobody asked. P2's measurement: DINOv2 sat 0.006 below the
baseline with marginal intervals **0.6 wide**, and only the paired difference
could say whether that was a result or noise.

It is also the form an edit reaches for first, because the marginal intervals are
already sitting on the row.

**Changed.** Every comparison in the table is now a paired difference with a
fold-clustered interval, and there is one such comparison per `margin_over_*`
column: the band-matched baseline, the observation control, persistence, the
proxy climatology, the horizon control, the permutation null, and -- new this
phase -- the row's own RGB twin.

The statistic is

    theta = R2_pooled(A) - R2_pooled(B) = (SSE_B - SSE_A) / SST

-- one expression, because A and B share `y` and therefore share `SST` -- over
the observations both rows predicted, keyed by (fold, feature row). Its interval
is the **delete-one-fold jackknife of theta itself**, not of either term: folds
hold disjoint sets of cubes, so deleting one deletes a cluster, and the shared
fold effect cancels inside theta before the spread is taken. This is the same
construction `_pooled_with_fold_jackknife` already applies to a single pooled
statistic, applied to the difference. The jackknife runs on per-fold sums rather
than by re-concatenating the arrays k times, because at 115 leave-one-cube-out
folds and seven references on 1500-odd rows the naive form is the difference
between seconds and an hour.

`assert_separability_is_paired` refuses a table where any verdict could have come
from marginal intervals. Its fourth check is the one a re-implementation fails:
a marginal rule produces a half-width exactly equal to the SUM or the
ROOT-SUM-SQUARE of the two marginal half-widths, and a jackknife of the
difference equals neither on any row. It also asserts each paired difference
EQUALS the margin it is the interval for -- an interval on a neighbouring
quantity is worse than no interval -- and that the unsuffixed `paired_diff` /
`separable` columns ARE the primary reference's rather than a copy that drifted.

**The reusable point.** An interval is not a property of a number, it is a
property of a COMPARISON. Two correct marginal intervals can make a real
difference look like noise, and the failure is invisible because every number on
the page is right.

**Commit.** `51eb72a`

---

## 2026-08-12: Tier 1, P3: a shared [NDVI(t), weather] base under every model row

**Assumed.** That `raw_features` holding `NDVI_mean(t)..NDVI_p90(t)` was
legitimate and needed no correction, because the target is at t+Delta: using
current NDVI to predict future NDVI is ordinary autoregression, not the K2
leakage case P2 had to separate out.

**Observed.** Both halves of that are true and they do not add up to a fair
comparison. The autoregression is legitimate; the problem is that **exactly one
row in the table was given it**. Every network row saw an embedding and the
weather, `raw_features` saw an embedding, the weather AND current NDVI, and the
two were then differenced and the difference attributed to the representation.
`raw_features` winning outright at 3 of 4 horizons is therefore not
interpretable: it mixes "hand-crafted band statistics are better" with "only this
row was handed the strongest single predictor in the problem".

The size of the effect is not small. On this run the base alone moves the
`weather_only` row from **+0.225 to +0.522** pooled R-squared at Delta = 5 d
under the tuned ridge -- a row with no image in it at all.

**Changed.** `feature_base` is on every row:

* `none` -- the published design. What each representation carries on its own.
* `ndvi_weather` -- [NDVI(t), weather] under EVERY model row, so each row answers
  one question: what does this representation add beyond current NDVI and the
  weather over the horizon?

NDVI(t) is taken from the target view's `persistence` array -- it IS the
persistence prediction, at the row's own aggregation and on the same common mask
-- so it cannot drift from the baseline it is derived from. It adds exactly one
column, which `assert_shared_base_present`'s companion check in the end-to-end
test pins (`D_with_base == D_without + 1`).

**Controls take no base, ever.** A control handed current NDVI is not a control;
it is a model, and `margin_over_control` -- the number P4 established as the one
to quote -- would become a margin over a forecast. The assertion refuses a table
where any control carries it.

The no-base rows stay. With both in the table the two questions are separable
for the first time, and `print_base_effect` prints the difference the base alone
made, per row, per horizon.

**Commit.** `51eb72a`

---

## 2026-08-12: Tier 1, P3: nine encoder views, and the plausibility screen APPLIED rather than reported

**Assumed.** That comparing frozen encoders against hand-crafted band statistics
on the same three bands (B04, B03, B02) was the like-for-like comparison, and
that the three cloud-contaminated frames were best handled by reporting their
concentration rather than by filtering them.

**Observed.** Two things, and they are the two ways the 2026-08-12 headline
could be wrong.

**Band access.** Every network encoder in this project is 3-channel and was fed
true colour, so all four were DENIED B8A -- the near-infrared band where the
vegetation signal mostly lives -- while `raw_features` reads all four bands plus
seven NDVI statistics. "Hand-crafted features beat learned representations" and
"NIR beats RGB, and the representation was never what decided it" both predict
exactly the table P3 produced.

**The three frames.** They passed both filters -- 59-63% "clear", per-pixel mask
valid -- and carried 71% of the persistence sum of squares at Delta = 5 d. The
2026-08-12 run reported that concentration and dropped nothing, because a private
filter would have made P3's row set incomparable with P2's and P4's, whose
targets come from the same `p4_ceiling.cube_frame_targets`.

**Changed.** Phase 1.9 re-encoded the four networks under the colour-infrared
composite (B8A, B04, B03) -- same weights, same extraction recipe, same frame
selection, only the band routing differs -- and this run scores all **nine**
views, 5 RGB and 4 `_cir`. The headline is the PAIRED difference between each
`_cir` row and its own `_rgb` twin, which isolates band access and nothing else.

`_assert_twins_are_distinct` refuses a run in which the two caches turn out to
hold the same arrays. That is not a hypothetical: reading the `_cir` views out of
the RGB directory produces a twin difference of exactly zero on every row, which
looks like a finding. Routing is done in ONE place (`encoder_embeddings_dir`) and
the composite is derivable from the encoder's name.

And the screen is now **applied**, not merely reported: `frame_plausible` moved
into shared code in `27bede5`, and P3 opts in. A forecast row is dropped if ANY
frame it touches -- context, t, or target -- fails it. The row is what goes, not
the frame: a shortened manifest would break the embedding join, whose contract is
`(cube_id, original_axis_index) == (cube, kept_idx)` against a cache built over
every retained frame; and dropping the frame from the SELECTION would let the
context reach one frame further back, which silently lengthens the lookback of
exactly the rows nearest the contamination -- one inhomogeneity traded for
another. Every row declares `plausibility_screen=True`, and
`assert_plausibility_screen_declared` refuses a table that mixes screened and
unscreened rows, because they are computed over different row sets and their
scores are not comparable.

**Commit.** `51eb72a`

---

## 2026-08-13: Screened P2/P4 re-runs — like-for-like with P3, published tables untouched

**Assumed.** That dropping the same three `frame_plausible=False` frames P3
already removed would move P2/P4 headlines the way it moved P3 persistence
(+0.52 at Δ=5 d).

**Observed.** Geometry matches the dry-run: 3/1580 frames; P4 loses 3 cube /
36 cell target rows; P2 drops 6 pairs (1465→1459). Headlines barely move, and
where they move they do not change the verdict:

- P4 Stage A `cell_mean` / HGB / weather: cube **+0.116** (was +0.130), LOCO
  **+0.096** (was +0.085).
- P2 sign: dinov2 **+0.604** (was +0.606); `raw_rgb_only` **+0.759** (was
  +0.695). Magnitude stays weak (networks ≤ +0.13; gap control rises to +0.145
  so several network *margins* flip more negative).

**Changed.** Opt-in screen in `probes/p2_deltas.py` / `probes/p4_ceiling.py`
(default off). Thin runners write new CSVs only. Paper and handoffs should cite
screened tables for cross-probe comparison; the original scaled CSVs stay as
the audit baseline. This closes HANDOFF §4 item 1 (consistency), not a novelty
claim — Scenario 1 trust, not Scenario 2 climate geography.

**Commit.** `5b00ea6`

---

## 2026-08-16: Tier-1 trigger metrics — the threshold is fitted INSIDE the fold, and persistence wins at short lead

**Assumed.** That the trigger re-slice was a reporting change over artefacts we
already had — the memo scored it "Low effort, same data, just re-sliced, no new
compute" — and that it would make the existing negative FM result legible to a
climate reader without changing what the result says.

**Observed.** Two things were wrong with that.

*The artefact did not exist.* `p3_tier1_results.csv` carries aggregated fold
statistics only; `probes/p3_forecast.py` discarded per-fold held-out
predictions after scoring. No threshold-crossing metric was computable from the
cache, so a re-run was required after all. It is cheap — 15.5 min for the
headline configs against 173.7 min for the full Tier-1 stack — but it is not
zero, and the memo's "no new compute" line was wrong.

*The result is not flat across lead time.* Persistence's Peirce skill decays
`+0.585 → +0.300 → +0.102 → +0.087` over Δ = 5/25/50/100 d while the encoders'
decays far more slowly, so the two curves **cross between 25 and 50 days**:

- **Δ = 5 d:** no encoder beats persistence; **23 of 32** forecast cells are
  separably WORSE. This is the paper's thesis in its strongest and most
  decision-relevant form.
- **Δ = 50 d:** 18 of 32 are numerically ahead but only **2** separably so, and
  a *different* pair is separable under `spatial_block`.
- **Δ = 100 d:** 21 of 32 ahead, **none** separable, and persistence's hit rate
  there is 0.087 — it barely fires, so the bar is low.

**Changed.**

1. `probes/p3_forecast.py` gains opt-in `emit_predictions=True` writing one
   tidy row per held-out observation (`PREDICTIONS_COLUMNS`). Opt-in because
   the published run did not have it; verified free because all **424** shared
   rows are **bit-identical to `p3_tier1_results.csv` on 36 scoring columns**.
   Also gains `fold_modes` / `alpha_rules`, which narrow a run and cannot widen
   it.
2. New `probes/p3_triggers.py`: hit rate, false-alarm rate (POFD, the quantity
   Peirce subtracts — `far` is reported separately), CSI and Peirce skill, each
   against persistence on the same rows with the fold-clustered PAIRED interval
   `paired_difference` uses. A test reproduces a `paired_difference` result
   through the generalised jackknife to 1e-12 rather than asserting the
   equivalence in prose.
3. **The threshold rule is p4's, but the FIT moved inside the fold.**
   `severity_reference_anomaly` fits on all rows and is right to — it LABELS
   held-out rows after the fact, which is a reporting choice. A threshold a
   forecast is SCORED against cannot: a 10th percentile over the full sample
   has seen the held-out rows, and the crossing rate it defines is then partly
   a property of the test side. `_trigger_reference` therefore fits curve and
   quantiles on the fold's training rows only, and
   `assert_thresholds_are_train_fitted` refuses a file four ways where they
   could have come from the full sample.
4. A narrowed run cannot carry margins: an unfitted baseline takes the ridge
   control at `fixed_alpha_D`, which this scope does not compute, so
   `add_margins` — and with it `assert_separability_is_paired` and
   `assert_control_identical_across_views` — is unavailable on the subset
   table. Recorded rather than worked around; nothing downstream reads margins.

**What this licenses.** "At 5-day lead, frozen EO embeddings do not beat
persistence for bottom-decile anomaly crossings on this tile, and mostly lose to
it." **Not** "foundation models are unnecessary at any lead time" — the 50–100 d
crossover is real in sign, unstable in significance, and is a scope boundary
rather than a finding. This closes memo item B1 (kill date Aug 15, one day
late, clean) and moves the submission into the memo's Scenario 2 band by
supplying the decision-linked evaluation; it does **not** touch the remaining
one-tile / one-year objection.

**Commit.** `9721223`

---

## 2026-08-17: Extreme-tile P4 pilot, 32UQC — the stressed tile lowers the ceiling and raises the confounding

**Assumed.** Memo item B3: that a 2018 heat/drought tile would make the
weather-attributability ceiling *more climate-relevant, and possibly larger*
than 32UNU's, and that this was the cheapest climate-facing extension because it
needs no re-encoding. Costed at **1.3–1.9 CPU-hours** for one tile by linear
scaling from the 115-cube run. Also assumed, in the run's own brief, that every
branch the run could hit already had a precedented answer.

**Observed.** Three pre-authorised rules were encoded and two of them did not
fire. `32UQC` returned 348 non-overlapping cubes, exactly its documented
capacity, so no fallback to `32UNC`. All eight E-OBS variables are present and
100% finite over every cube and `assert_weather_join` verifies exactly, so
`weather_full8` ran and the ceiling is directly comparable with 32UNU's rather
than needing the `weather_finite6` footnote 30TVN forced.

Four things were measured that the plan did not anticipate.

1. **The ceiling is lower, not higher.** `cell_mean` / HGB / `weather_full8`:
   `cube` **+0.111** `[+0.031, +0.191]` against 32UNU's +0.116, `loco`
   **+0.078** `[+0.037, +0.118]` against +0.096. The margin over the
   observation-process control falls further, +0.086 / +0.056 against +0.120 /
   +0.117. Three times the cubes tightened `loco` (CI width 0.081 vs 0.134) and
   left the fraction of weather rows whose interval spans zero unchanged (25/54
   vs 24/54). The control-beating rate did improve — rows at or below the
   observation control 14/54 vs 23/54, at or below DOY 14/54 vs 29/54.

2. **Day-of-year and weather are far more collinear here, and this outweighs
   the improved DOY margin.** Measured before any fit: 6990 rows land on **47
   distinct dates** over 295 days, every row satisfies `doy % 5 == 2` (one
   Sentinel-2 orbit lattice), up to **346 cubes share a single date**, and the
   across-cube spread of a typical windowed weather feature is **0.07 of its
   total spread — so ~93% of it is recoverable from the date alone, against
   ~61% on 32UNU**. 346 cubes packed into one MGRS tile read an E-OBS grid too
   coarse to separate them. The DOY control is 6 harmonics, 13 smooth features;
   it cannot fit a 47-level categorical, so it understates what the date alone
   can do, and understates it harder here than on 32UNU. The `+0.096` DOY
   margin is therefore weather beating a *smooth function of* timing, and is
   **not** evidence that this ceiling is less confounded than 32UNU's. It is
   more confounded, and the control as designed is not the instrument that
   would show it.

3. **Two cubes carry fill values the published cloud mask calls clear.**
   `cube_frame_targets` refused them: a grid cell had clear pixels and no finite
   NDVI. All **57 720** "clear" pixels in the 61 offending cells (0.055% of
   111 968, across 2 of 348 cubes) carry **exactly-zero reflectance in B04 and
   B8A**. `encoders.frames.finite_valid_mask` cannot demote them because the
   bands are finite, merely zero; `data.ndvi.ndvi`'s `|B8A+B04| < 1e-12` guard
   correctly returns NaN. The assertion was right, and caught a no-data block
   before it was averaged into a target.

4. **The cost model was wrong by 4–6x.** Measured on this tile at 20 and 40
   cubes, per fold mode: `cube` exponent 0.59, `spatial_block` 1.17, **`loco`
   1.72** — leave-one-cube-out grows its fold count and its per-fold training
   set together. Projected 4.7 CPU-hours where linear-in-cubes said 1.26;
   **actual 7.1** (`run_stage_a` 427.4 min vs 281.2 projected), because the
   `loco` exponent itself steepens over an 8.7x extrapolation.

**Changed.**

- New `scripts/run_p4_extreme.py`: one invocation, download → pre-flight →
  weather-set resolution → two-point runtime calibration → gated full run →
  report, with the three rules encoded and each announcing itself in the log
  and in the CSV (`tile_reason`, `weather_feature_set_reason`,
  `cubes_excluded_reason`).
- `probes/p4_ceiling.py` gains the visibility the runner needed and nothing
  else: a throttled per-cube heartbeat in `build_p4_data`'s previously silent
  loop (`CUBE_HEARTBEAT_EVERY`), and `print_doy_weather_collinearity` now reads
  `feature_sets[0]` rather than the module constant `FEATURE_SETS[0]` — with
  `verbose=True` on a `weather_finite6` run the old line was a `KeyError` on
  exactly the runs that most need the output. No scientific path changed; the
  63 `tests/test_p4_ceiling.py` tests pass.
- **The two defective cubes are excluded whole**, nothing filled, count and
  reason on every result row; the run is 346 cubes. **This was not one of the
  three pre-authorised rules** — the brief's premise that every branch had a
  precedent was false here. The properly correct fix is a zero-reflectance rule
  beside `finite_valid_mask`, but that is a shared path P1/P2/P3 and the
  published 32UNU tables all read through, and moving it would move numbers
  this run had no mandate to move. Recorded so it can be overruled.
- **Go/no-go on the gated slim extreme-tile P3 (Aug 18–21): NO-GO.** The bar,
  set against the pilot's own hypothesis, was a ceiling clearly above
  +0.116/+0.096 with a *growing* margin over the observation and DOY controls —
  roughly `cube` ≥ +0.15 at an observation margin ≥ +0.120. Measured +0.111 /
  +0.078 at +0.086 / +0.056: lower on both axes, on a tile where 93% of the
  weather is recoverable from the date. The memo's own escalation gate ("the
  one-tile result materially changes the climate story" or "is clearly cleaner
  than 32UNU") is not met. The compute case is worse still: a slim P3 here
  would carry the same `loco` exponent plus a GPU RGB cache build, and the
  memo's estimate for it comes from the same linear scaling that was just
  measured 4–6x low.
- 32UQC stays as the paper's extreme-tile *check*, reported as a null: the
  Limits paragraph's promised follow-on has been run.

**Commit.** `4af7edf`
