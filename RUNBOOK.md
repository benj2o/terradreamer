# Phase 1.1 runbook

One manual file move, one runtime restart, one fast download.

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

Tile `32UNU`, 9.00 to 10.49 E, 47.76 to 48.75 N: Allgaeu and Upper Swabia. It is
the closest Alpine-foreland tile GreenEarthNet contains. `32UPU`, which holds
Munich itself, is not in the dataset. Same latitude band as Munich, about 135 km
west, same landscape.

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
