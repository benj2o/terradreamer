# Runbook

Phase 1.1 first, then Phase 1.2, then Phase 1.3, then Phase 1.4 at the end of
this file. Every phase shares the same Colab workflow: one manual file move,
one runtime restart, one fast download.

The upload bundle is now `phase1_4_repo.zip` (built by `make_zip.sh`); it
contains everything the earlier notebooks need too, so there is no reason to
keep an old `phase1_1_repo.zip`, `phase1_2_repo.zip` or `phase1_3_repo.zip`
around.

## ONE SUBFOLDER PER PHASE

From Phase 1.3 on, each phase gets its own subfolder under one project folder,
holding its own checkout. Deleting a phase's subfolder is then a complete undo
of that phase and cannot touch another phase's artefacts:

```
My Drive/
└── NeurIPS-CCAI-2026/
    ├── data/raw/*.nc         SHARED cubes. NOT a phase.
    ├── phase1_1/             checkout + notebook
    ├── phase1_2/             checkout; artefacts at data/phase1_2/{embeddings,masks}
    ├── phase1_3/             checkout; artefacts at data/phase1_3/folds
    └── phase1_4/             checkout
        └── phase1_4_repo.zip <- drag it here, leave it zipped
```

**`data/raw` is not filed under a phase, and that is deliberate.** The cubes
are phase-independent: Phase 1.2 and 1.3 both read them and scale-up will too.
`data/paths.py` encodes this -- `RAW_DIR` is never phase-scoped and
`reset_phase` refuses to touch it. Filing the cubes under `phase1_1/` would
force every later phase to reach into another phase's folder, which is the
coupling this layout removes.

The `phase1_2/data/phase1_2/` nesting is redundant but deliberate: the outer
name is the Drive checkout, the inner one is `data.paths.phase_dir`, which is
canonical and not worth bending for cosmetics.

The two rules that make this safe:

* **A phase writes only under its own subfolder**, through
  `data.paths.phase_dir(<phase>, <kind>)`. Never a hand-typed path.
* **A phase READS earlier artefacts in place.** The Phase 1.3 bootstrap
  searches this checkout first, then Drive up to THREE levels down, for
  `data/raw/*.nc` and `data/phase1_2/embeddings/*.npz`, and uses whichever it
  finds. Three levels because the nested layout puts the embeddings at
  `NeurIPS-CCAI-2026/phase1_2/data/phase1_2/embeddings`. No cubes are copied
  per phase and nothing is written into Phase 1.2's subfolder.

Both the nested layout above and the older flat one
(`NeurIPS-CCAI-2026/data/phase1_2/embeddings`) resolve, so moving existing
Phase 1.2 artefacts into `phase1_2/` is tidiness, not a requirement. Step 2
prints which path it resolved -- read it.

`reset_phase("phase1_3")` is the finer-grained version of the same undo, and
`data/raw` is shared and never cleared by either.

### Reorganising an existing folder into this layout

**On Colab, use `notebooks/organise_drive.ipynb`.** It is self-contained: it
imports nothing from the project, so it works on a Drive folder that has no
checkout in it yet. Five cells -- mount, inventory, plan, verify, full listing
-- and nothing moves until you set `APPLY = True` in Step 3.

Step 5 prints **every folder and every file** from the project root with no
depth limit, then checks the shape of the result: at the root there should be
nothing except `data/` (holding only `raw/`) and the `phase1_*` folders.
Anything else is reported as `TODO ... still loose at the root`. It writes
`tree.txt`, so two runs can be diffed, and it is standalone -- run it on its
own in a fresh runtime whenever you want to see what is actually on Drive.

This matters because of a bootstrap problem the shell scripts have and the
notebook does not: `scripts/inventory.py` lives *inside* the phase bundle,
so `python -m scripts.inventory` needs a checkout already extracted into the
very folder you are trying to reorganise. A tool that tidies the folder a
checkout lives in cannot depend on that checkout existing.

The notebook therefore carries its own copy of the classification rules. That
copy is pinned against `scripts/inventory.py` by
`tests/test_scripts_organise.py`, over a table of paths and both decision
functions, so the two cannot drift into disagreeing about where a file belongs.

### The same job from a shell, when a checkout IS present

Both run anywhere -- locally, or in a Colab cell against the mounted Drive.
Step one only LOOKS, step two only MOVES, and step two imports its
classification from step one, so the plan you approve in the listing is the
plan that runs.

```bash
python -m scripts.inventory --root "/content/drive/MyDrive/NeurIPS-CCAI-2026" --sha256 --json before.json
```

Prints the tree, then every movable unit with the phase that owns it:

```
UNIT               KIND         PHASE        FILES  SIZE
data/phase1_2      artefacts    phase1_2       120  10.1 MB
data/raw           shared       -                3  10.2 MB
phase1_2_repo.zip  bundle       phase1_2         1  2 B
probes             checkout     -                1  2 B
```

`data/` is expanded one level on purpose: it holds three different kinds at
once, and collapsing it would drag the shared cubes into a phase folder.

`checkout` units show phase `-` because a checkout is **not** classifiable from
the filesystem -- it looks identical whichever bundle produced it. Step two
asks rather than guessing from mtimes.

```bash
python -m scripts.organise_phases --root "/content/drive/MyDrive/NeurIPS-CCAI-2026" --checkout-phase phase1_2
```

**Dry run by default.** It prints the plan and moves nothing. Read it, then
re-run with `--apply`. Afterwards:

```bash
python -m scripts.inventory --root "/content/drive/MyDrive/NeurIPS-CCAI-2026" --sha256 --json after.json
```

Diff `before.json` against `after.json`: same set of hashes, different paths.
That is what proves the move lost nothing, and Drive is exactly the kind of
place where an interrupted move is not obviously distinguishable from a
completed one.

Four things the mover refuses to do, each pinned by a test rather than a
docstring:

1. Move `data/raw`. It is shared; `data.paths.reset_phase` refuses it too.
2. Write to any destination outside the root.
3. Overwrite an existing destination. A same-named leftover from an older
   layout is common on Drive, and replacing a current artefact with a stale one
   is silent.
4. Split a checkout, or guess which phase owns one.

Re-running after an `--apply` plans zero moves. It is idempotent, so a half
finished run is resumed by simply running it again.

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
| 4 | unit tests, expect `261 passed, 5 skipped` locally | 30 s |
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

---

# Phase 1.3 runbook

`probes/cv.py`, the leakage-safe split definition. **CPU is enough** — no GPU,
no encoder weights, nothing re-encoded. Build the bundle with `./make_zip.sh`
(it emits `phase1_4_repo.zip`; earlier phases extract from it too), drag it
into a NEW subfolder
`My Drive/NeurIPS-CCAI-2026/phase1_3/`, leave it zipped, open
`notebooks/phase1_3_cv.ipynb`, and run top to bottom. Exactly one restart, at
the end of Step 1.

Phase 1.2 must have run first: Step 10 asserts the join contract against
`data/phase1_2/embeddings/`. Step 2 finds that directory wherever it is on
Drive and reads it in place, so Phase 1.2's subfolder is never modified.

| step | what | time |
|---|---|---|
| 1 | install (no `satlaspretrain-models` here), auto-restart | 2 min |
| 2 | bootstrap, resolves RAW + EMB_IN read-only, defines `sh()` | 1 min |
| 3 | environment check; fails loudly if Phase 1.2 never ran | instant |
| 4 | cubes (skips ones already on Drive) | 15 s |
| 5 | unit tests, expect `261 passed, 5 skipped` | 30 s |
| 6 | build the REAL manifest from `data/raw/*.nc` | 1 min |
| 7 | the three runnable modes: cube, spatial_block, temporal | 10 s |
| 8 | the three refusals: year, tile, crossed | instant |
| 9 | the same-cube gate provoked; duplicate rows refused | instant |
| 10 | join contract on one real (cube, encoder) pair per encoder | 10 s |
| 11 | save fold indices under `data/phase1_3/folds/` | 5 s |
| 12 | full recursive listing of the project root, layout check | 5 s |

### What must RAISE, and is not a bug

The 20 cubes are one tile (32UNU) and one year (2018), so:

```
cube            runs      DEFAULT, GroupKFold on cube_id
spatial_block   runs      SUBSTITUTE for tile holdout at prototype scale
temporal        runs      P3 robustness variant only, starves training data
year            RAISES    SingleYearError  -> use cube, or crossed at scale-up
tile            RAISES    SingleTileError  -> use spatial_block
crossed         RAISES    SingleYearError  -> use cube (cube == year here)
```

Step 8 asserts all three refusals fire. A green Step 8 means the guards work.
Do not "fix" them with a random split: it would put the same season on both
sides, which is exactly the inflated number the grouping exists to prevent.

### Expected output, reference local CPU run (2026-08-05)

Full numbers in [log.md](log.md); the archived run is
`notebooks/runs/phase1_3_cv_2026-08-05_localCPU.ipynb`.

```
Step 5   261 passed, 5 skipped        (266 collected)
Step 6   manifest (264, 21), 20 cubes, tile ['32UNU'], years [2018]
Step 7   cube k=5      test 52-53 rows / 4 cubes per fold
         LOCO          20 folds, test sizes 10..16
         spatial_block 5 blocks sized [9, 3, 4, 3, 1]
         temporal      cutoff 2018-08-15, 17 cubes train / 3 test, 65 dropped
         31 folds re-checked independently: no cube on both sides
Step 8   SingleYearError / SingleTileError / SingleYearError
Step 9   LeakageError naming crossed; crossed handles the same manifest
Step 10  window_span_days min 0 median 38 max 85 days on the MI encoder,
         exactly 0 on all single-image encoders
Step 11  5 files, 0.20 MB under data/phase1_3/folds/
Step 12  full tree; PHASE 1.3 OUTPUTS all present; LAYOUT CHECK
```

Step 12 lists every folder and file from the **project root** -- the parent of
this checkout in the nested layout -- so the shared cubes and Phase 1.2's
artefacts appear alongside Phase 1.3's own outputs. Two things to read there:
`PHASE 1.3 OUTPUTS` should show all five files under `data/phase1_3/folds/`,
and under `READ-ONLY INPUTS` neither the cubes nor the Phase 1.2 embeddings
should carry the `[inside this checkout]` tag -- that tag means the layout is
still flat, not that anything is wrong. It writes `tree.txt` beside the folds,
so two runs can be diffed.

The archived 2026-08-05 local run has **four** encoders in Step 10, not five:
`dinov2_vitb14` was missing from the local cache because its `torch.hub` code
annotates a signature `float | None`, which Python 3.9 evaluates and rejects,
and the dev venv is 3.9.6.

**No longer a limitation (2026-08-08).** It needed Python >= 3.10, not a GPU:
all 20 cubes encode on CPU in 42 s. The 20 files were produced in a throwaway
3.10 virtualenv pinned to the same `torch==2.8.0`, driving
`encoders.pipeline.encode_cube` unchanged, so the local cache is now the full
100 files and a local Step 10 sees all five. See
[docs/DECISIONS.md](docs/DECISIONS.md). Re-running Phase 1.3 locally today
would print five encoders where the archived run printed four; that is the
cache being complete, not a change to the phase.

### When something fails

- Step 3 `No Phase 1.2 embeddings found`: run
  `notebooks/phase1_2_encoders.ipynb` first. Whichever subfolder it ran in is
  fine — Step 2 here searches up to three levels down for
  `data/phase1_2/embeddings` and reads it in place. Read the `EMB_IN` line
  Step 2 prints to see what it resolved.
- A `cache schema v<N>` assertion in Step 10 means the embeddings predate
  `window_span_days`. They are Phase 1.2's artefacts, so reset THAT phase in
  ITS folder (`reset_phase("phase1_2")`) and re-encode; Phase 1.3 never writes
  there.
- A `LeakageError` from Step 7 on real data would mean the manifest has two
  cubes sharing an id, or duplicate `(cube_id, timestamp)` rows. Read the
  message: it names the cube.
- Step 5 collecting a different number of tests than you get locally means the
  bundle is stale. `make_zip.sh` lists files with `git ls-files`, so an
  uncommitted file is silently absent. **Commit before `make_zip.sh`.**

### Re-running cleanly

```python
from data.paths import reset_phase
reset_phase("phase1_3")     # clears ONLY this phase; data/raw is untouched
```

Deleting the `phase1_3/` subfolder is the coarser version of the same undo.
Neither can touch Phase 1.2's artefacts or the shared cubes, because this
notebook only ever reads them.

---

# Phase 1.4 runbook

P1, the appearance sanity probe: month and season from ONE frame's frozen
embedding. **CPU is enough** — no GPU, no encoder weights, nothing re-encoded
and nothing fine-tuned; this phase reads Phase 1.2's `.npz` with sklearn.
Build the bundle with `./make_zip.sh`, drag `phase1_4_repo.zip` into a NEW
subfolder `My Drive/NeurIPS-CCAI-2026/phase1_4/`, leave it zipped, open
`notebooks/phase1_4_p1_appearance.ipynb`, and run top to bottom. Exactly one
restart, at the end of Step 1.

Phase 1.2 must have run first, and its cache must be COMPLETE: Step 8 asserts
all 20 × 5 pairs, because a cube silently missing one encoder turns a
per-encoder comparison into a comparison over different cubes.

| step | what | time |
|---|---|---|
| 1 | install (adds `scikit-learn`, `scipy`), auto-restart | 2 min |
| 2 | bootstrap, resolves RAW + EMB_IN read-only, defines `sh()` | 1 min |
| 3 | environment check; sklearn present, cheap cache audit | instant |
| 4 | cubes (skips ones already on Drive) | 15 s |
| 5 | unit tests, expect `323 passed, 5 skipped` at the time of writing; the count GROWS as later phases add tests (383 as of Phase 1.5). The invariant is 0 failed and 5 skipped. | 20 s |
| 6 | build the REAL manifest from `data/raw/*.nc` | 1 min |
| 7 | realised class distribution, chance DERIVED from it | instant |
| 8 | audit + join all 100 pairs, re-asserted against the `.npz` | 20 s |
| 9 | fold disjointness re-derived from the manifest, not from `cv` | 5 s |
| 10 | poisoned test fold: selected `C` must not move (EXHIBIT; the gate is Step 5) | 10 s |
| 11 | **the run** — 432 rows, 4320 outer folds, ~95k fits | 85 min |
| 12 | table invariants: baseline row, degenerate control, MI flags | instant |
| 13 | rank agreement between the three fold modes | instant |
| 14 | Figure 1, five panels on shared axes | 20 s |
| 15 | write + re-read the CSV, list what the phase wrote | 5 s |

Step 11 dominates. It is the only long cell; `N_JOBS` is set in Step 3 from
`os.cpu_count() - 1`, and parallelism changes wall-clock only — every estimator
here is exact and the folds carry no RNG, so the CSV is identical at any
`N_JOBS`.

### What the numbers should look like

Full detail in [log.md](log.md); the archived run is
`notebooks/runs/phase1_4_p1_appearance_2026-08-08_localCPU.ipynb`.

```
Step 5   323 passed, 5 skipped        (328 collected)
Step 7   month  8 REALISED classes (Apr..Nov), chance 1/8 = 0.1250, NOT 1/12
         season 3 REALISED classes (no DJF),   chance 1/3 = 0.3333, NOT 1/4
Step 8   100 .npz -> 100 usable (20 x 5) at v3
         MI window_span_days min 0 median 55 max 105 d; SI encoders exactly 0
Step 9   30 folds, every manifest row tested exactly once in each mode
Step 10  selected C identical on all 5 folds; test score moves (~10 s)
Step 11  432 rows in ~85 min on 7 workers
Step 12  month/cube/logreg/grid_cell, margin over the BAND-MATCHED baseline:
           raw_features +0.102 > dinov2 +0.059 ~ satlas_rgb +0.058
           > imagenet +0.021 > MI +0.014 > raw_rgb_only 0.000
           DEGENERATE CONTROL -0.046   (all five networks clear it here)
Step 13  weakest pairwise Spearman rho +0.400
Step 14  PC1+PC2 explained variance 26% (dinov2) .. 70% (raw_features)
```

**Two results to read, not skim.** Against the BAND-MATCHED baseline
(`raw_rgb_only` — the same three bands every network gets) the frozen encoders
**win** on month/cube by +0.02 to +0.06; full `raw_features` leads only because
it also sees B8A and NDVI. Against the degenerate control
(`[clear_frac, window_span_days]`, no image at all) they win on month under
`cube`/`loco` (0 of 20 rows at or below) and **lose on season** (10/20, 9/20,
and 17/20 under `spatial_block`). Cite P1 as month under cube or LOCO, and cite
the margin over the control. Both are recorded in
[docs/DECISIONS.md](docs/DECISIONS.md).

### When something fails

- Step 3 `No Phase 1.2 embeddings found`: run
  `notebooks/phase1_2_encoders.ipynb` first. Read the `EMB_IN` line Step 2
  prints to see what it resolved.
- Step 8 `N of 100 (cube, encoder) pairs have no current embedding`: the cache
  has holes. The `[audit]` lines above it name the defect and its remedy —
  usually `python -m scripts.restamp_cache` (no GPU) or a Phase 1.2 re-encode.
  Do not continue with a partial roster.
- Step 10 failing means the tuning loop saw test data. Nothing downstream of it
  is reportable; fix `select_hyperparameter` before re-running Step 11.
- Step 11 running much longer than an hour: check `N_JOBS` in Step 3. On a
  single-core runtime the run is roughly 4 hours.
- Step 5 collecting a different number of tests than you get locally means the
  bundle is stale. `make_zip.sh` lists files with `git ls-files`, so an
  uncommitted file is silently absent. **Commit before `make_zip.sh`.**

### Re-running cleanly

```python
from data.paths import reset_phase
reset_phase("phase1_4")     # clears ONLY this phase; data/raw is untouched
```

Deleting the `phase1_4/` subfolder is the coarser version of the same undo.
Neither can touch Phase 1.2's embeddings or Phase 1.3's folds, because this
notebook only ever reads them.

---

# Phase 1.5 runbook

P4, the weather-attributability ceiling: how much of the post-climatology NDVI
anomaly is explainable from weather alone. **CPU is enough** — ~4 minutes, no
GPU, no encoder weights, **no embeddings at all**. This phase reads the cubes
and nothing else. Build the bundle with `./make_zip.sh`, drag
`phase1_5_repo.zip` into a NEW subfolder `My Drive/NeurIPS-CCAI-2026/phase1_5/`,
leave it zipped, open `notebooks/phase1_5_p4_ceiling.ipynb`, run top to bottom.
Exactly one restart, at the end of Step 1.

Phase 1.2 does **not** need to have run. Step 2 resolves `EMB_IN` because the
bootstrap block is shared verbatim with Phases 1.3 and 1.4, and then prints that
it is deliberately unused — `window_span_days` is recomputed from the manifest
via `encoders.pipeline.window_span_days` rather than read from a cache.

| step | what | time |
|---|---|---|
| 1 | install (adds `scikit-learn`, `scipy`, `joblib`), auto-restart | 2 min |
| 2 | bootstrap, resolves RAW read-only, defines `sh()` | 1 min |
| 3 | environment check, `N_JOBS` from `os.cpu_count() - 1` | instant |
| 4 | cubes (skips ones already on Drive) | 15 s |
| 5 | unit tests, expect `383 passed, 5 skipped` | 50 s |
| 6 | the REAL manifest + **the E-OBS join VERIFIED against the cubes** | 1 min |
| 7 | day-of-year vs weather structure; severity bin edges and counts | 30 s |
| 8 | poisoned held-out NDVI: the climatology must not move (EXHIBIT; the gate is Step 5) | 5 s |
| 9 | fold disjointness re-derived; weather constant across the 16 cells | 10 s |
| 10 | **the run** — 270 rows, 2700 outer folds | 3 min |
| 11 | table invariants: four controls, effective n, Stage B ran-or-deferred | instant |
| 12 | the headline — margin over the observation control | instant |
| 13 | write + re-read the CSV, list what the phase wrote | 5 s |

### What the numbers should look like

Full detail in [log.md](log.md); the archived run is
`notebooks/runs/phase1_5_p4_ceiling_2026-08-10_localCPU.ipynb`.

```
Step 5   383 passed, 5 skipped
Step 6   264 rows x 22 cols; the two axes differ on 264/264 rows by 4..122
         steps (median 53); weather join max abs difference 0
Step 7   36 DISTINCT days of year over 235 days, every row doy % 5 == 2
         across-cube spread of the windowed features: median 0.375
         severity edges (cube_mean) -0.1006 / -0.0329 / +0.0398 / +0.1013
Step 8   curve bit-identical on all 5 folds when only TEST rows are poisoned,
         and it moves when a TRAIN row is
Step 10  270 rows in ~3 min on 7 workers
Step 11  linear day-of-year control, worst |r2| = 0.019
         permutation control negative everywhere
         Stage B: DEFERRED, explicitly, and NOT substituted
Step 12  cube_mean / cube / linear / weather_full8:
           weather              +0.066  [-0.159, +0.291]
           OBSERVATION CONTROL  +0.028
           margin_over_control  +0.038      effective n = 20 CUBES
```

**Three results to read, not skim.** (1) The fold-clustered CI **includes zero
for all 54 weather rows** — at 20 cubes, one tile and one year, this subset
cannot measure a ceiling, and that is the finding. (2) **33 of 54 weather rows
sit at or below the observation-process control**, so cite `margin_over_control`
and never the raw R². (3) **Stage A is not H1** — it is a within-season proxy
climatology on a single year, and every row says so in `climatology_def`. H1
comes from Stage B, on the seasonal split.

### When something fails

- Step 6 `AssertionError: ... wrong-axis bug`: the manifest was built by code
  predating 2026-08-10, so its E-OBS columns are indexed by the acquisition axis
  and hold another day's weather. Rebuild it with `build_manifest`; do not
  continue, because weather is this probe's entire input.
- Step 8 failing means the climatology saw the test fold. **Nothing downstream
  of it is reportable** — not the weather number and not any control, because a
  leaked target definition inflates all of them together. Fix
  `doy_climatology_within_fold` before re-running Step 10.
- Step 9 `the feature varies ... ACROSS THE CELLS OF A SINGLE FRAME`: the cell
  expansion is indexing the wrong frames. Weather is one value per (cube, frame)
  by construction, so this is a join bug and every number below it would be a
  confident wrong number of the same shape.
- Step 11 `missing [...]`: the run was truncated. The four controls are not
  optional — without the observation control the headline margin does not exist.
- Step 11 reporting Stage B rows on this subset means the proxy was relabelled
  as the real climatology. That is the one substitution this phase exists to
  prevent.
- Step 5 collecting a different number of tests than you get locally means the
  bundle is stale. `make_zip.sh` lists files with `git ls-files`, so an
  uncommitted file is silently absent. **Commit before `make_zip.sh`.**

### Re-running cleanly

```python
from data.paths import reset_phase
reset_phase("phase1_5")     # clears ONLY this phase; data/raw is untouched
```

---

# Phase 1.6 runbook

P2, dynamics in deltas: gate K2 (can a frozen embedding recover current NDVI at
all) and the delta probe (do embedding changes track real NDVI change). **CPU is
enough** — ~10 minutes, no GPU, no encoder weights loaded and no embedding
recomputed. Build the bundle with `./make_zip.sh`, drag `phase1_6_repo.zip` into
a NEW subfolder `My Drive/NeurIPS-CCAI-2026/phase1_6/`, leave it zipped, open
`notebooks/phase1_6_p2_deltas.ipynb`, run top to bottom. Exactly one restart, at
the end of Step 1.

Unlike Phase 1.5, this phase **does** read Phase 1.2's cache — both the
embeddings and the per-pixel masks from 1.2b, which common-masking cannot be
done without. Step 2 resolves `EMB_IN` and `MASKS_IN` and prints the file counts;
if either is 0 the run cannot proceed, and re-encoding is a Phase 1.2 job.

| step | what | time |
|---|---|---|
| 1 | install, auto-restart | 2 min |
| 2 | bootstrap, resolves RAW / `EMB_IN` / `MASKS_IN` read-only, defines `sh()` | 1 min |
| 3 | environment check | instant |
| 4 | cubes (skips ones already on Drive) | 15 s |
| 5 | unit tests, expect **0 failed, 5 skipped** | 60 s |
| 6 | the manifest, **rebuilt** from the cubes; E-OBS join re-verified | 1 min |
| 7 | the cache audited (100 `.npz`), joined, and the Part A target built | 40 s |
| 8 | **the gap-axis check on 244 real pairs** — must report 0 pairs where the axes agree | 5 s |
| 9 | common-masked pair targets + the survival table by gap length | 1 min |
| 10 | poisoned gap control: held-out poison must not move it, TRAIN poison must (EXHIBIT; the gate is Step 5) | 5 s |
| 11 | fold disjointness, and no PAIR straddling a fold | 10 s |
| 12 | **the run** — 600 rows | 9 min |
| 13 | six table invariants | instant |
| 14 | gate K2, the four controls, the headline, the structural hypothesis | instant |
| 15 | robustness: the same margins under every fold mode | instant |
| 16 | write + re-read the CSV, list what the phase wrote | 5 s |

### What the numbers should look like

Full detail in [log.md](log.md); the archived run is under `notebooks/runs/`.

```
Step 5   0 failed, 5 skipped
Step 6   264 rows x 22 cols, 20 cubes, tile 32UNU, 2018; weather join max
         abs difference 0
Step 8   244 pairs; gap_days median 10 vs gap_acq_steps median 2 vs
         frames-between 1 -> 5.0 days per acquisition step,
         0/244 pairs where the two axes agree
Step 9   244/244 pairs keep shared pixels, median 88.8% of 16384;
         survival NOT monotone in gap (0.783 at 15 d, 0.985 at 30 d)
Step 12  600 rows x 61 cols in ~9 min
Step 14  K2 at cube_mean/grid_cell/cube: satlas SI +0.545, imagenet +0.521,
         raw_features +0.440, dinov2 +0.435, band-matched floor +0.417,
         retention control -0.129; EXCLUDED FROM P3: the MI encoder only
         delta SIGN: dinov2 +0.536, satlas SI +0.481, control -0.118
         delta MAGNITUDE: the CONTROL wins at +0.209
         structural hypothesis: NOT DETERMINABLE (order flips by fold mode)
```

### Things that will trip you up

- **`gap_days` is not `original_axis_index`.** The latter is the embedding join
  key and counts ACQUISITIONS. If Step 8 ever reports a non-zero count of pairs
  where the two axes agree, stop: every gap in the phase is wrong by a factor
  of about five, and nothing downstream will notice.
- **`MASKS_IN` empty means common-masking is impossible.** It cannot be
  approximated from `clear_frac`, which is a per-frame scalar. Re-run Phase
  1.2b's mask cache.
- **Do not quote a raw Spearman.** Quote `margin_over_control`. On the magnitude
  target the gap-length control beats every encoder, so a raw correlation there
  is a calendar.
- **Do not quote `k2_verdict` without `k2_verdict_band_matched`.**
  `raw_features` contains `NDVI_mean`..`NDVI_p90`.
- Step 5 collecting a different number of tests than you get locally means the
  bundle is stale. **Commit before `make_zip.sh`.**

### Re-running cleanly

```python
from data.paths import reset_phase
reset_phase("phase1_6")     # clears ONLY this phase; data/raw is untouched
```


---

# Phase 1.7 runbook — the scaled embedding cache

**This phase computes no result.** It builds the frozen-encoder cache for the
115-cube 32UNU set so that P1, P2 and P3 can be re-run at 5.75x the sample size
**without any of them changing a line of code** — both already take `emb_dir`
and `mask_dir` as parameters.

**This is the one notebook in the project that wants a GPU.** P4 could scale
trivially because it reads no embeddings; P1/P2/P3 read the Phase 1.2 `.npz`
cache, which exists for exactly 20 cubes. Growing that means 1580 retained
frames through four real networks. Everything downstream stays CPU-only.

Build the bundle with `./make_zip.sh`, drag `phase1_7_repo.zip` into a NEW
subfolder `My Drive/NeurIPS-CCAI-2026/phase1_7/`, leave it zipped, open
`notebooks/phase1_7_scaled_encoding.ipynb`. **Set Runtime > Change runtime type
> T4 GPU before Step 1.** Exactly one restart, at the end of Step 1.

| step | what | time (T4) |
|---|---|---|
| 1 | install — **adds `satlaspretrain-models`**, which Phase 1.5 did not need | 3 min |
| 2 | bootstrap, resolves RAW and the READ-ONLY 20-cube `EMB_IN` | 1 min |
| 3 | environment check, GPU detect, **python >= 3.10 assertion** | instant |
| 4 | the 115 cubes (idempotent; ~363 MB on a cold Drive) | 0–10 min |
| 5 | unit tests, expect **0 failed** | 60 s |
| 6 | build the five encoders (downloads weights on first run) | 2 min |
| 7 | smoke test on ONE cube, all five, shapes asserted | 30 s |
| 8 | **the encode** — 575 `.npz`, resumable | 20–40 min |
| 9 | audit: 575 embeddings + 115 masks, no holes | 30 s |
| 10 | **the reproduction cross-check against the 20-cube cache** | 1 min |
| 11 | manifest, delta pairs, effective n in cubes | 2 min |
| 12 | report, and the exact call to re-run P2 | instant |

### The two things that will actually bite you

- **Python 3.9 silently loses ONE encoder.** `dinov2_vitb14` loads its code from
  `torch.hub`, and that code uses `X | None`, a syntax error before 3.10. The
  other four build fine, so you get a cache with a hole in it and a `TypeError`
  from inside a downloaded file. Step 3 asserts the version up front. Measured
  on this repo's own 3.9 venv: 4 of 5 encoders build, DINOv2 fails.
- **Write the cache to Drive, not the Colab container.** 575 files over ~30
  minutes; a disconnect at minute 25 with the output in `/content` loses all of
  it. Step 8 is resumable — re-running the cell skips everything already on disk
  and re-validates it on load — but only if the files survive the session.

### What the numbers should look like

```
Step 3   python 3.11+, GPU detected
Step 4   115 cubes, 20 shared with data/raw, 95 new
Step 5   0 failed
Step 7   all five agree on the retained frames of the smoke cube
Step 8   575 (cube, encoder) pairs, 0 failures
Step 9   575 embedding .npz + 115 mask .npz, audit COMPLETE
Step 10  frame selection / timestamps / clear_frac BIT-IDENTICAL on the
         20 shared cubes; pooled embeddings within 1e-3
Step 11  1580 retained frames (6.0x), ~1465 delta pairs (6.0x),
         effective n 115 CUBES (5.75x)
```

Step 10 is the one that must not be skipped: without it the scaled cache is a
different experiment and no scaled number is comparable to a published one.
A non-zero difference there is expected to be small (float32 on different
hardware — Phase 1.4's Colab reproduction of P1 moved scores by ±0.003) and is
printed rather than assumed.

### Re-running cleanly

The cache is an INPUT to later phases and lives beside the shared cubes, so
`reset_phase` does not touch it — that is deliberate. To force a full re-encode:

```python
import shutil, os
shutil.rmtree(os.path.join(SCALED_ROOT, "embeddings"))
shutil.rmtree(os.path.join(SCALED_ROOT, "masks"))
```

```python
from data.paths import reset_phase
reset_phase("phase1_7")     # clears only this phase's own report CSVs
```
