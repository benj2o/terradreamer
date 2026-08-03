# Working log

Running record of measurements and adopted definitions. Reverse chronological.
Decisions and their rationale live in [docs/DECISIONS.md](docs/DECISIONS.md).

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
