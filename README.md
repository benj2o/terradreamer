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
encoders/                    Phase 1.2: Tier A frozen encoders, one interface:
                             encode(frames[T, C, H, W]) -> [T, D]. base.py holds
                             the frozen/batched/asserted machinery; frames.py the
                             clear-fraction rule and the valid-reflectance check;
                             four wrappers (raw_features D=35, imagenet_vit_b16
                             D=768, dinov2_vitb14 D=768, satlas_s2_swinb_rgb
                             D=1024); pipeline.py cube -> saved .npz embeddings
encoders/manifest.py         (cube, frame) manifest: original_axis_index (horizons
                             are DAYS on the original axis, never retained frames),
                             pixel_bbox, and landcover_stratum from in-cube esawc_lc
probes/cv.py                 THE split definition. Year mode raises on one year
tests/                     test_ndvi.py was written before data/ndvi.py existed
notebooks/phase1_1_data_toy_load.ipynb
notebooks/phase1_2_encoders.ipynb
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
- 1.2 frozen encoder embeddings: **DONE**, exit test passed on a Colab T4
  (`notebooks/phase1_2_encoders.ipynb`, run recorded in [log.md](log.md)).
  Four Tier A wrappers behind one interface at D = 35 / 768 / 768 / 1024;
  80 cube x encoder embedding files, 264 retained frames each; frame selection
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
- 1.3 `probes/cv.py`: next. Until the fold generators exist, nothing here is a
  result -- the Phase 1.2 embeddings are inputs, not numbers.
