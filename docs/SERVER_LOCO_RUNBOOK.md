# Running the missing `loco` rows on lxhalle — a step-by-step

**What this is.** The local run produced 1232 of 1540 table rows. The missing
308 are `fold_mode=loco` (leave-one-cube-out): 342 cubes, each held out once
and refitted on the other 341. It is the single most expensive fold mode —
measured growth exponent **+1.31**, projected **~10 h on the 8-core Mac** — so
it gets done on the 64-core server and merged back.

**Nothing here needs a GPU.** Fitting only reads the frozen embedding caches.
No encoder is imported.

**The one thing that must not drift.** These 308 rows get concatenated with the
local 1232. For that to be legitimate they must be the *same experiment*: same
342 cubes, same library versions, same `k`, same screen. Step 2 pins the
libraries and Step 5 reproduces the roster from `cache_roster.csv`. Do not skip
either.

---

## Step 0 — first, stop any local `loco` run

If one is going on the Mac it will take ~29 h, compete for your laptop, and
eventually write the **same filename** you are about to copy back from the
server. Check and stop it:

```bash
ps -eo pid,etime,command | grep "[f]old-modes loco"
```
```bash
pkill -f "fold-modes loco"
```

## Step 0b — what you need before you start

On the Mac, in the repo. On the server, a shell (you are at "Connection
established"). Replace `YOU` with your TUM login throughout.

Two locations on the server, on purpose:

| what | where | why |
|---|---|---|
| code + venv | `~/p3` | small, and home is backed up |
| data + outputs | `/var/tmp/YOU/p3` | ~4 GB; your home is at 71% |

---

## Step 1 — on the SERVER: make the folders

```bash
mkdir -p ~/p3 /var/tmp/$USER/p3/data/scaled_32UQC && df -h /var/tmp | tail -1
```

You want to see comfortably more than 5 GB free.

---

## Step 2 — on the SERVER: build the Python environment

The versions are pinned deliberately. A different scikit-learn can change a
ridge or HGB fit slightly, and then the 308 new rows are not comparable with
the 1232 you already have.

```bash
python3 -m venv ~/p3/.venv && ~/p3/.venv/bin/pip install -q --upgrade pip
```
```bash
~/p3/.venv/bin/pip install -q "numpy==2.0.2" "pandas==2.3.3" "scikit-learn==1.6.1" "scipy==1.13.1" joblib xarray netCDF4
```

Now `torch`, and **use this exact line** — the CPU-only index:

```bash
~/p3/.venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
```

**Why torch at all, when nothing is encoded?** Because
`probes/p3_forecast.py` imports `encoders.pipeline` (for `load_masks`), and
that module imports `torch` at the top. Nothing *uses* it during fitting, but
the import chain needs it present. Verified empirically: without torch the
import fails with `ModuleNotFoundError`.

**Why the CPU index?** On Linux, a plain `pip install torch` pulls the CUDA
build and drags in ~2.5 GB of nvidia libraries. The CPU wheel is a fraction of
that and this job has no GPU work at all.

**Not installed, deliberately:** `torchvision`, `satlaspretrain-models`,
`earthnet`, `s3fs`. The per-encoder imports are lazy — they live inside
`build_encoder`, which fitting never calls. Verified in a clean environment
with torchvision absent.

Now prove the whole chain imports before you copy 4 GB:

```bash
cd ~/p3 && ~/p3/.venv/bin/python -c "from probes import p3_forecast, p3_triggers; from encoders.pipeline import load_masks; from data.loader import load_cube; import scripts.run_p3_extreme; import sys,numpy,pandas,sklearn,scipy; print('imports OK |', sys.version.split()[0], numpy.__version__, pandas.__version__, sklearn.__version__, scipy.__version__)"
```

Expect `imports OK | 3.12.3 2.0.2 2.3.3 1.6.1 1.13.1`. The Python minor version
differing from the Mac's 3.9.6 is fine — the numerics live in the pinned
libraries.

---

## Step 3 — on the MAC: push the code and the data

Code first (small, seconds). `rsync` is the default here because it needs no
GitHub credentials on a shared university box and it copies exactly your
working tree. If you already have an SSH key on lxhalle that can reach the
repo, `git clone git@github.com:benj2o/terradreamer.git ~/p3` is equivalent —
everything needed is committed as of `10e425a`.

```bash
cd "/Users/benji/Code/NeurIPS CCAI 2026" && rsync -av --exclude '.venv' --exclude '.git' --exclude 'data/' --exclude '*.zip' --exclude 'notebooks/runs' ./ YOU@lxhalle.cit.tum.de:~/p3/
```

Then the data (~4 GB, this is the slow one — expect 10–40 min):

```bash
rsync -avP data/scaled_32UQC/raw data/scaled_32UQC/embeddings data/scaled_32UQC/embeddings_cir data/scaled_32UQC/masks data/scaled_32UQC/cache_roster.csv YOU@lxhalle.cit.tum.de:/var/tmp/YOU/p3/data/scaled_32UQC/
```

`-P` means it resumes if the connection drops. Re-run the same line to resume.

Verify on the SERVER:

```bash
cd /var/tmp/$USER/p3/data/scaled_32UQC && for d in raw embeddings embeddings_cir masks; do printf "%-16s %s\n" "$d" "$(ls $d | wc -l)"; done && ls cache_roster.csv
```

Expect **348 / 1715 / 1372 / 343** and the roster file. If any count is short,
re-run the rsync — a partial cache silently changes which rows get fitted.

---

## Step 4 — on the SERVER: the 5-minute smoke test

Do not skip this. It proves the environment before you spend hours.

```bash
cd ~/p3 && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ~/p3/.venv/bin/python -m scripts.run_p3_extreme --out /var/tmp/$USER/p3/data/scaled_32UQC --fold-modes loco --aggregations cube_mean --skip-calibration --no-predictions --max-cubes 20 --encoders raw_features,imagenet_vit_b16_cir --n-jobs 8 --csv-name p3_smoke_loco.csv --log-name p3_smoke_loco.log
```

You want it to reach `FITTING 342 cubes`, print the smoke-test warning, and
finish without a traceback. The numbers it produces are meaningless — it is a
shape check. Delete the output afterwards:

```bash
rm -f /var/tmp/$USER/p3/data/scaled_32UQC/p3_smoke_loco.*
```

---

## Step 5 — on the SERVER: the real run, in tmux

`tmux` keeps the job alive after you disconnect. `nice` keeps you a good
citizen on a shared box.

```bash
tmux new -s loco
```

Then, inside tmux, paste this as one block:

```bash
cd ~/p3 && export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 && nice -n 10 ~/p3/.venv/bin/python -m scripts.run_p3_extreme --out /var/tmp/$USER/p3/data/scaled_32UQC --fold-modes loco --aggregations cube_mean --skip-calibration --no-predictions --n-jobs 32 --csv-name p3_extreme_loco_results.csv --log-name p3_extreme_loco_run.log 2>&1 | tee /var/tmp/$USER/p3/loco_console.txt
```

Detach with **Ctrl-b** then **d**. The job keeps running. Come back with
`tmux attach -t loco`.

### Why each flag is there

| flag | why |
|---|---|
| `--fold-modes loco` | the only thing missing locally |
| `--aggregations cube_mean` | **required.** `cube_p90` and `cell_mean` only run under `cube` folds, so leaving them in makes the probe abort with "leaves 'cube_p90' with no fold mode" |
| `--skip-calibration` | the valve already ran locally; re-measuring costs ~1.5 h and decides nothing here |
| `--no-predictions` | saves writing a multi-GB per-observation file. It changes **no number** in the table — paired separability comes from payloads, not predictions |
| `--n-jobs 32` | 32 workers, one BLAS thread each = 32 of 128 logical CPUs (~25%) |
| `OMP/MKL/OPENBLAS/NUMEXPR=1` | **the important one.** Without it each of the 16 workers spawns its own BLAS thread pool and you get 16 × N threads fighting each other — slower for you and hostile to everyone else |

`loco` has 342 folds, so anything up to 342 workers is usable in principle —
but 32 is the polite ceiling on a shared box and already gets you to ~6 h. Do
not raise the thread caps above 1: that is what keeps 32 workers to 32 CPUs
instead of 32 x N.

---

## Step 6 — watching it

```bash
tail -f /var/tmp/$USER/p3/data/scaled_32UQC/p3_extreme_loco_run.log | grep -E "p3-eta|rule|Traceback"
```

You get a `row N` line per table row with elapsed time and s/row. Expect
**308 rows**.

**How long — from a real measurement, not an extrapolation.** A local `loco`
attempt on 2026-08-19 reached row 25 of 308 in 143 minutes on 7 Mac workers:
**344 s/row**, i.e. a **~29-hour job on the Mac**. (The clean two-point
calibration had projected ~10 h. It was a 3x underestimate — the same failure
mode the P4 pilot recorded, and the reason this runs on the server at all.)

`loco` parallelises over its 342 folds, so worker count scales it almost
linearly:

| `--n-jobs` | logical CPUs used | expected wall |
|---|---|---|
| 16 | 16 of 128 (12%) | ~13 h |
| **32** | **32 of 128 (25%)** | **~6–7 h** |

**Use 32.** It is still a quarter of the box, and it turns an overnight job
into an afternoon. Drop to 16 if `uptime` shows the machine already loaded.

Check you are being polite:

```bash
uptime && ps -o pid,%cpu,nlwp,comm -u $USER --sort=-%cpu | head -5
```

`nlwp` is threads per process — it should be small, not 128.

---

## A caveat to carry into the write-up

`loco` on this tile is 342 folds of **two to five rows each** — the smoke test
shows folds with `test 3 rows / 1 cubes` and `R2 +nan`. That is inherent to
leave-one-cube-out at this cube size, not a bug. Two consequences:

- Many `loco` rows will carry NaN or wildly noisy per-fold R², and the
  fold-clustered intervals over them are weak.
- A contingency table over a three-row fold is not a contingency table, which
  is exactly why `scripts/rerun_p3_predictions.py` excludes `loco` from the
  trigger metrics, and why the 2026-08-16 32UNU trigger run used
  `cube,spatial_block` only.

So these 308 rows complete the grid and let the full-table assertions run. They
are **not** a second opinion on the headline, and the paper should not lean on
them for the early-warning claim.

---

## Step 7 — bring it home

On the SERVER, package it:

```bash
cd /var/tmp/$USER/p3/data/scaled_32UQC && tar czf ~/p3_loco_handoff.tar.gz p3_extreme_loco_results.csv p3_extreme_loco_run.log && ls -la ~/p3_loco_handoff.tar.gz
```

On the MAC:

```bash
cd "/Users/benji/Code/NeurIPS CCAI 2026" && scp YOU@lxhalle.cit.tum.de:~/p3_loco_handoff.tar.gz /tmp/ && tar xzf /tmp/p3_loco_handoff.tar.gz -C data/scaled_32UQC/
```

---

## Step 8 — on the MAC: merge into the full table

```bash
.venv/bin/python -m scripts.merge_loco
```

That checks the two tables describe the same experiment before concatenating,
and refuses if they do not. It writes `p3_extreme_results.csv` — the full
1540-row grid.

---

## If something goes wrong

| symptom | cause | fix |
|---|---|---|
| `leaves 'cube_p90' with no fold mode` | you dropped `--aggregations cube_mean` | add it back |
| `RULE 1 STOP: the P4 roster is N cubes, not 346` | `raw/` did not copy fully | re-run the Step 3 rsync |
| `RULE 2 STOP: ... missing (cube, view) pairs` | an embedding cache is short | re-run the Step 3 rsync; check the Step 3 counts |
| `no module named data` | you ran from the wrong directory | `cd ~/p3` first |
| merge refuses with a version mismatch | server libraries differ | redo Step 2 with the exact pins |
| job vanished after logout | you forgot tmux | Step 5, in tmux |
