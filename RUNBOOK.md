# Phase 1.1 runbook

One manual file move, one runtime restart, one long download.

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
| 1 | install, then auto-restart | 3 min |
| 2 | bootstrap, also defines the `sh()` helper | 1 min |
| 3 | environment check | instant |
| 4 | unit tests, expect `35 passed` | 15 s |
| 5 | diagnostic | 90 s |
| 6 | timing probe, 2 months | 1 to 3 min |
| 7 | download 20 cubes | measured in step 6 |
| 8 to 10 | load, check, plot | 3 min |

Exactly one restart, at the end of Step 1, triggered automatically. Resume at
Step 2. Step 1 self-skips via `/content/.phase1_1_installed`.

Step 2 also defines `sh()`, the helper every later step uses to run shell
commands. It lives in the bootstrap cell on purpose, so it cannot be skipped. It
is called `sh` and not `run` because IPython has a `%run` magic: if a helper
named `run` is ever undefined, automagic rewrites `run("...")` into `%run` and
reports a confusing error about a missing script file.

If Colab disconnects during Step 7: reconnect, re-run Step 2, re-run Step 7.
Completed cubes are skipped. Nothing is lost because `data/raw/` is on Drive,
not on the ephemeral `/content` disk.

## Nothing is downloaded by hand

Three things arrive over the network, all automatically:

1. pip packages, in Step 1.
2. The cloud-mask checkpoint `mobilenetv2_l2a_rgbnir.pth` from
   `nextcloud.bgc-jena.mpg.de`, fetched on first use and cached in
   `~/.cache/torch/hub/`.
3. Sentinel-2 L2A, streamed from Microsoft Planetary Computer via STAC, 36
   monthly queries per cube. No credentials needed.

Output lands in `My Drive/NeurIPS-CCAI-2026/data/raw/`, about 60 to 90 MB per
cube, so 1.2 to 1.8 GB for 20. Check your Drive quota first.

## How long

Step 6 measures it instead of guessing. Two months of one cube, then multiply by
18 for a full cube and by 360 for all 20.

Most of the time is CPU, not network. The cloud mask is a mobilenet-v2 U-Net run
at 2x upsampling on Colab's 2 vCPUs. A GPU runtime does not help, because
minicuber never moves the model to CUDA.

If the projected total is unacceptable, use the GreenEarthNet fallback at the
end of Step 7.

## Expected output

### Step 4, unit tests

```
35 passed
```

A red `test_masked_pixel_is_nan_and_does_not_leak` means the cloud mask is not
being applied and every downstream number is void.

### Step 7, download

Per cube: 36 `Loading Sentinel2 for ...` lines and a heartbeat every 30 s, then:

```
[01/20] downloading lon=11.44659 lat=48.09826 ...
          cube dims {'time': 387, 'lat': 128, 'lon': 128}
          vars ['s2_B02', 's2_B03', 's2_B04', 's2_B8A', 's2_SCL', 's2_avail', 's2_mask']
          s2_mask shape (387, 128, 128) dtype float32 | clear(=0) fraction 0.383
          s2_B04 min 0.0001 max 0.9812
          -> mc_00_lon11.4466_lat48.0983.nc (71.4 MB, 3.2 min) | ETA for the rest ~61 min
```

- `time` between 250 and 450. That is S2A plus S2B with overlapping orbits over
  three years. Under 150 means a time filter was left on.
- `clear(=0) fraction` around 0.25 to 0.55. Bavaria is cloudy.
- `s2_B04 min` must not be negative. Negative means the BOA offset survived and
  NDVI is systematically wrong.

If neither a month line nor a heartbeat appears for several minutes, it is
genuinely stuck.

### Step 8, the per-cube table

| column | healthy | a bad value means |
|---|---|---|
| `T` | 250 to 450 | under 150: time filter on, or partial coverage |
| `masked_fraction` | 0.45 to 0.75 | near 0: masking off or inverted. Near 1: inverted the other way |
| `n_steps_gt50pct_clear` | 60 to 150 | under 30 leaves too little signal for Phase 1.2 |
| `dt_days_median` | 2 to 5 | 5.0 with `dt_days_max` also 5.0: grid regularised |
| `refl_min`, `refl_max` | about 0, under 1 | negative: BOA offset present |
| `ndvi_median` | 0.4 to 0.8 | strongly negative: B04 and B8A swapped |

`masked_fraction` is higher than the downloader's `1 - clear fraction` because
the loader drops timesteps with no acquisition before averaging. Both being in
range is the check.

### Step 9, the time grid

```
dt/days: n=7412  min=1  p50=3  p90=10  max=41  |  exactly 5 d: 18.4%
```

The point is that `exactly 5 d` is nowhere near 100%. A `max` of 30 to 60 days
is normal, that is a winter cloud stretch.

### Step 10, NDVI

Three seasonal humps, one per year. About 0.2 to 0.4 in winter, 0.7 to 0.9 in
June through August.

- A flat line near 0.2 to 0.3 means you are averaging clouds.
- A curve that mirrors the clear-fraction panel means the mask is correlated
  with the signal.

## When something fails

Run Step 5, `python -m data.diagnose`, and read the first failure.

| stops at | meaning | fix |
|---|---|---|
| `imports` | pip resolution failed in Step 1 | re-run the install without `-q` and read the conflict |
| `stac` | cannot reach Planetary Computer | retry, PC has transient outages |
| `cloudmask` | `nextcloud.bgc-jena.mpg.de` unreachable, or a torch mismatch | retry, it is a third-party host |
| `cube` | the real pipeline bug | paste the traceback |

`no cubes matching '*.nc'` in Step 8 is never the real bug. It means the
download produced nothing. Go to Step 5.

## Known upstream bugs, already worked around

- `earthnet-minicuber` 0.1.3 imports `sen2nbar` but omits it from
  `install_requires`. The install cell adds it.
- `earthnet-minicuber` 0.1.3 calls `stackstac.stack` without `rescale` or
  `fill_value`. stackstac 0.5 added three `np.can_cast(type(x), dtype)` guards
  that all reject float32 stacks. `data/stackstac_compat.py` injects
  `rescale=False` and `fill_value=float32(nan)`. Neither changes a number, and a
  guard raises if Planetary Computer ever publishes a real scale or offset.

## Moving on

Phase 1.1 is done when Steps 4, 8, 9 and 10 pass their checks. Then:

- Do not commit the cubes. Commit their checksums:
  `sha256sum data/raw/*.nc > data/raw.sha256`
- Phase 1.2: frozen encoder embeddings. `.eval()` and `torch.no_grad()`, no
  fine-tuning, ever.
- Phase 1.3: `probes/cv.py`. Until it exists, no number produced here is a
  result.
