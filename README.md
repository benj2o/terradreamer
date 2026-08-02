# Probing frozen EO foundation-model representations for dynamics and forecastability

GreenEarthNet / EarthNet2021 minicubes. Munich first: everything is prototyped on
a small Bavarian / Alpine-foreland subset before any scale-up.

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
data/ndvi.py               canonical NDVI. Exactly one function. Never copy it.
data/loader.py             (values, timestamps, mask) per cube. Mask polarity is
                           decided here, once, in valid_mask_from_codes.
data/download_minicubes.py up to 20 non-overlapping S2 minicubes at (11.55, 48.15)
data/stackstac_compat.py   shim for stackstac >= 0.5 vs earthnet-minicuber 0.1.3
data/diagnose.py           four escalating checks, stops at the first failure
tests/                     test_ndvi.py was written before data/ndvi.py existed
notebooks/phase1_1_data_toy_load.ipynb
RUNBOOK.md                 Colab walkthrough: folders, restarts, expected output
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
python -m data.download_minicubes --out data/raw --n 20
```

`NDVI_VERBOSE=1` makes `ndvi()` print the shapes of every call.

For Colab, follow [RUNBOOK.md](RUNBOOK.md).

## Phase status

- 1.1 data toy-load: `ndvi()` unit test green, loader and downloader in place.
- 1.2 frozen encoder embeddings: next.
- 1.3 `probes/cv.py`: until it exists, nothing here is a result.
