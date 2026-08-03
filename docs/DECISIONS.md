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

**Commit.** `436095a`

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

**Commit.** `96a0054` (wording), source switch in `436095a`

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

**Commit.** `b4a996e`

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

**Commit.** `dc1cdf7`

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

**Commit.** `73704ce` (encoders), notebook and bundle in `8ddf508`
