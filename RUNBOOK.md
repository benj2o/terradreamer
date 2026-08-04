# Runbook

Phase 1.1 first, Phase 1.2 at the end of this file. Both phases share the same
Colab workflow: one manual file move, one runtime restart, one fast download.

The upload bundle is now `phase1_2_repo.zip` (built by `make_zip.sh`); it
contains everything the Phase 1.1 notebook needs too, so there is no reason to
keep an old `phase1_1_repo.zip` around.

# Phase 1.1 runbook

## Before Colab

Drag `phase1_1_repo.zip` into `My Drive/NeurIPS-CCAI-2026/`.

```
My Drive/
└── NeurIPS-CCAI-2026/
    └── phase1_1_repo.zip     <- drag it here, leave it zipped
```

Leave it zipped. Drive's web "Extract" makes a nested folder the notebook would
have to guess about, and Step 2 unzips it correctly anyway.

Do not create `data/raw/` yourself. Step 2 makes it.

The folder name is a default, not a requirement. Step 2 searches `My Drive` two
levels deep for the zip.

Set Runtime > Change runtime type > T4 GPU before Step 1. Changing it later
restarts the kernel and wastes the install.

## Updating the code later

Drop the new zip in the same folder, overwrite, re-run Step 2. It re-extracts
when the zip is newer than `data/ndvi.py` and prints `zip is newer`.

Do not hand-unzip. An unzipped checkout next to an old zip is how you end up
debugging last week's code.

## Steps

| step | what | time |
|---|---|---|
| 1 | install, then auto-restart | 2 min |
| 2 | bootstrap, also defines the `sh()` helper | 1 min |
| 3 | environment check | instant |
| 4 | unit tests, expect `48 passed` | 15 s |
| 5 | diagnostic | 30 s |
| 6 | download 20 cubes | 15 s |
| 7 to 9 | load, check, plot | 2 min |

Exactly one restart, at the end of Step 1, triggered automatically. Resume at
Step 2. Step 1 self-skips via `/content/.phase1_1_installed`.

Step 2 also defines `sh()`, the helper every later step uses to run shell
commands. It lives in the bootstrap cell on purpose, so it cannot be skipped. It
is called `sh` and not `run` because IPython has a `%run` magic: if a helper
named `run` is ever undefined, automagic rewrites `run("...")` into `%run` and
reports a confusing error about a missing script file.

If Colab disconnects during Step 6: reconnect, re-run Step 2, re-run Step 6.
Completed cubes are skipped. At 15 seconds it barely matters.

## Nothing is downloaded by hand

Two things arrive over the network, both automatically:

1. pip packages, in Step 1.
2. Pre-processed GreenEarthNet cubes from `s3.bgc-jena.mpg.de`, anonymous, no
   credentials.

Output lands in `My Drive/NeurIPS-CCAI-2026/data/raw/`, about 3.4 MB per cube,
so roughly 70 MB for 20.

## The data

Tile `32UNU`, 9.00 to 10.49 E, 47.76 to 48.75 N: Allgäu / Upper Swabia. It is
the closest Alpine-foreland tile GreenEarthNet contains. Do not go looking for
Munich cubes: `32UPU`, which holds Munich, is not in the dataset. 32UNU is the
same latitude band, about 135 km west, same landscape.

Each cube is 128 x 128 px at 20 m over a 150-day window in 2018, on a daily grid
where roughly 29 days carry a Sentinel-2 acquisition. The loader drops the empty
days, which is where the irregular time axis comes from.

Two consequences worth stating plainly:

- Every cube in this tile is from 2018. There is no interannual signal to probe.
- One 150-day window per cube means one seasonal green-up, not three annual
  cycles. This is the EarthNet benchmark's own setup, so it is conventional, but
  it does bound what Phase 1.2 can claim.

Selection takes 20 of the tile's 192 cubes, round-robin across the 16 available
time windows, requiring 64 px of separation so no two cubes are even adjacent.

## Expected output

### Step 4, unit tests

```
48 passed
```

A red `test_masked_pixel_is_nan_and_does_not_leak` means the cloud mask is not
being applied and every downstream number is void.

### Step 6, download

```
[list] earthnet/earthnet2021x/train/32UNU
[list] 192 cubes in tile, selected 20 non-overlapping
[list] time windows covered: 15
[01/20] 32UNU_2018-03-09_2018-08-05_1081_1209_3641_3769_16_96_56_136.nc (3.4 MB, 0.7s)
...
[done] 20/20 cubes in .../data/raw (67 MB, 11s)
[check] all of ['s2_B02', 's2_B03', 's2_B04', 's2_B8A', 's2_mask'] present
```

### Step 7, the per-cube table

| column | healthy | a bad value means |
|---|---|---|
| `T` | 25 to 35 | far off: the empty daily slots were not dropped |
| `masked_fraction` | 0.45 to 0.75 | near 0: masking off or inverted. Near 1: inverted the other way |
| `n_steps_gt50pct_clear` | 8 to 20 | under 6 leaves too little signal for Phase 1.2 |
| `dt_days_median` | 2 to 5 | 5.0 with `dt_days_max` also 5.0: grid regularised |
| `refl_min`, `refl_max` | about 0, under 2.5 | negative: BOA offset present. Above 1 is bright cloud, normal |
| `ndvi_median` | 0.4 to 0.8 | strongly negative: B04 and B8A swapped |

`masked_fraction` is higher than the downloader's `1 - clear fraction` because
the loader drops timesteps with no acquisition before averaging. Both being in
range is the check.

### Step 8, the time grid

```
dt/days: n=560  min=5  p50=5  p90=10  max=20  |  exactly 5 d: 78%
```

The point is that `exactly 5 d` is not 100%. The 5-day floor is the S2 revisit;
anything longer is a cloud gap.

### Step 9, NDVI

One seasonal green-up per cube, not three. About 0.3 to 0.4 in early spring
rising to 0.7 or 0.8 by midsummer.

- A flat line near 0.2 to 0.3 means you are averaging clouds.
- A curve that mirrors the clear-fraction panel means the mask is correlated
  with the signal.

## When something fails

Run Step 5, `python -m data.diagnose`, and read the first failure.

| stops at | meaning | fix |
|---|---|---|
| `imports` | pip resolution failed in Step 1 | re-run the install without `-q` and read the conflict |
| `s3` | cannot reach `s3.bgc-jena.mpg.de` | retry, it is a third-party host |
| `cube` | the download or the file is broken | paste the traceback |
| `loader` | the real pipeline bug | paste the traceback |

`no cubes matching '*.nc'` in Step 7 is never the real bug. It means the
download produced nothing. Go to Step 5.

## The live-extraction path

`data/download_minicubes.py` still builds cubes from Sentinel-2 via Planetary
Computer, at any location and date range. Use it only for a location
GreenEarthNet does not cover. It measured 14.7 hours for 20 cubes, because the
cloud-mask U-Net runs on CPU for 36 monthly queries per cube.

Two upstream bugs it works around, both still live:

- `earthnet-minicuber` 0.1.3 imports `sen2nbar` but omits it from
  `install_requires`.
- `earthnet-minicuber` 0.1.3 calls `stackstac.stack` without `rescale` or
  `fill_value`. stackstac 0.5 added three `np.can_cast(type(x), dtype)` guards
  that reject float32 stacks. `data/stackstac_compat.py` injects
  `rescale=False` and `fill_value=float32(nan)`. Neither changes a number.

Note that its masks are self-derived, so numbers from that path are not
comparable with GreenEarthNet benchmark numbers.

## Moving on

Phase 1.1 is done when Steps 4, 7, 8 and 9 pass their checks. Then:

- Do not commit the cubes. Commit their checksums:
  `sha256sum data/raw/*.nc > data/raw.sha256`
- Phase 1.2: frozen encoder embeddings. `.eval()` and `torch.no_grad()`, no
  fine-tuning, ever.
- Phase 1.3: `probes/cv.py`. Until it exists, no number produced here is a
  result.

---

# Phase 1.2 runbook

Same workflow as Phase 1.1. Build the bundle with `./make_zip.sh` (it now emits
`phase1_2_repo.zip`), drag it into `My Drive/NeurIPS-CCAI-2026/`, leave it
zipped, open `notebooks/phase1_2_encoders.ipynb`, set the T4 runtime BEFORE
Step 1, and run top to bottom. Exactly one restart, at the end of Step 1.

| step | what | time |
|---|---|---|
| 1 | install (adds `satlaspretrain-models`), auto-restart | 2 min |
| 2 | bootstrap, defines `sh()` | 1 min |
| 3 | environment check | instant |
| 4 | unit tests, expect `118 passed, 5 skipped` locally | 30 s |
| 5 | cubes (skips ones already on Drive) | 15 s |
| 6 | build 4 encoders, first run downloads ~900 MB of weights | 3 min |
| 7 | four asserted `[T_kept, D]` on one cube | 1 min |
| 8 | valid-pixel reflectance <= 1.2 on all 20 cubes | 30 s |
| 9 | malformed inputs refused loudly | instant |
| 10 | T=290 memory smoke, peak GPU printed | 1 min |
| 11 | encode 20 cubes x 4 encoders, save `.npz` to Drive | 5 min |

The 3 skipped tests in Step 4 are the wrapper tests that would download
pretrained weights inside pytest (`PHASE1_2_WEIGHTS=1` enables them); Steps
6-11 exercise the same wrappers on the real cubes, which is the stronger test.

### Expected output, Step 6

```
D per model:
  raw_features           D=35
  imagenet_vit_b16       D=768
  dinov2_vitb14          D=768
  satlas_s2_swinb_rgb    D=1024
```

Each wrapper also prints its explicit preprocessing (RGB band selection,
resize, normalisation). If you did not see a preprocessing block, the wrapper
did not build.

### Expected output, Step 7

`T_kept` should be 10-16 (the log's `frames >50% clear` range), never 0, and
identical for all four encoders, with identical kept timestamps. The
retained-frame count is cross-checked against an independent numpy computation
in the same cell. On cube 1 the reference run gives `T_kept = 14/29`.

### Expected output, Steps 8, 10 and 11

Measured on the reference T4 run (full numbers in [log.md](log.md)):

```
Step 6   D per model: 35 / 1536 / 3840 / 1024 / 1024  (five encoders)
Step 8   all 20 cubes pass, worst prevalence 1.89e-05 vs tolerance 1e-04
         all-finite max reaches 1.9817 (bright cloud, behind the mask)
Step 10  peak GPU memory 1.56 GB for T=290 at batch_size=16   (budget 12 GB)
Step 11  100 cube x encoder pairs, 264 frames per encoder
         T_kept min 10, median 13, max 16
```

If Step 4 reports a DIFFERENT number of collected tests than you get locally,
the bundle is stale -- `make_zip.sh` lists files with `git ls-files`, so an
uncommitted file is silently absent from it. **Commit before `make_zip.sh`.**
A changed collection count is a stale-bundle signal, not noise.

Before re-running a phase after any change to the mask, the frame-selection
rule or an encoder's feature recipe:

```python
from data.paths import reset_phase
reset_phase("phase1_2")     # clears ONLY this phase; data/raw is untouched
```

`zipfile.extractall` overwrites but never deletes, and the Step 11 cache
reuses whatever it finds, so a stale artefact is a wrong number no assertion
will catch.

A `cached` status in Step 11 on a first run means `data/embeddings/` still
holds files from an earlier attempt. Delete the folder and re-run: cached
embeddings predate any change to the mask or the frame-selection rule.

### What lands on Drive

`data/embeddings/<cube>__<encoder>.npz`, 80 files, ~10 MB total. Each carries
`embeddings [T_kept, D]`, `timestamps`, the exact per-frame `clear_frac`
(later probes filter on it more strictly WITHOUT re-encoding), `kept_idx`
into the original cube time axis, and the encoder name. Step 11 is resumable:
existing files are loaded and re-asserted, not re-encoded. Delete the folder
after any change to the mask definition or the frame-selection rule.

### When something fails

- Step 6 fails on `satlaspretrain_models`: re-run Step 1 without `-q` blinders,
  the pip conflict is printed there.
- An `implausible valid pixels ... above the 1e-4 tolerance` assertion in
  Step 8 means bright cloud is leaking THROUGH the mask over real area -- the
  s2_dlmask + SCL conjunction is not being applied. Nothing downstream of it is
  trustworthy; go back to `data/loader.py` and Phase 1.1's diagnostics. A
  printed prevalence at or below ~1.9e-5 is the expected, tolerated case
  (isolated specular pixels, see `log.md` 2026-08-03) and does not stop the run.
- `EMPTY BATCH` from an encoder means a fully-clouded cube reached `encode()`
  without going through `encoders.pipeline.encode_cube`. Use the pipeline.
