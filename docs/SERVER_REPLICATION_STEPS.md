# Running a P3 fold mode on lxhalle — the steps that actually worked

Written 2026-08-19, immediately after the `loco` top-up succeeded. This is the
**condensed, corrected** sequence — every command below was run and verified.
`docs/SERVER_LOCO_RUNBOOK.md` is the longer version with the reasoning.

Substitute your own login for `go38mid`.

**Measured outcome:** 308 rows, 23.0 s/row, **116.7 min on 64 workers**, on a
machine already at load 48. The same work on an 8-core Mac measured 344 s/row
and was on course for ~29 h.

---

## The four mistakes that cost the most time

Read these first. Each one cost a round-trip or an hour.

| mistake | symptom | fix |
|---|---|---|
| `--exclude 'data/'` in the code rsync | `ModuleNotFoundError: No module named 'data'` | `--exclude 'data/*/'` — `data/` is BOTH a Python package and the home of the cubes |
| `--fold-modes loco` alone | `leaves 'cube_p90' with no fold mode` | add `--aggregations cube_mean`; the other two run only under `cube` folds |
| assuming no torch is needed | `ModuleNotFoundError: No module named 'torch'` | torch IS needed at fit time via `encoders.pipeline`; install the **CPU-only** wheel |
| wrong hostname | DNS failure | it is `lxhalle.cit.tum.de`, not `.in.tum.de` |

**And one counting fact:** a completed `loco` run reports **304** rows for a
**308**-row table. The 4 `horizon_only` rows are emitted without an
`evaluate()` call. *Row 304 means finished, not crashed.*

---

## 1. [SERVER] Folders

Home holds code and the venv. `/var/tmp` is scratch and holds the ~4 GB of data.

```bash
mkdir -p ~/p3 /var/tmp/$USER/p3/data/scaled_32UQC && df -h /var/tmp | tail -1
```

## 2. [SERVER] Environment — versions pinned to the Mac's

A different scikit-learn changes the fits, and then the new rows are not
comparable with the existing table.

```bash
python3 -m venv ~/p3/.venv && ~/p3/.venv/bin/pip install -q --upgrade pip
```
```bash
~/p3/.venv/bin/pip install -q "numpy==2.0.2" "pandas==2.3.3" "scikit-learn==1.6.1" "scipy==1.13.1" joblib xarray netCDF4
```
```bash
~/p3/.venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
```

NOT installed, deliberately: `torchvision`, `satlaspretrain-models`, `earthnet`,
`s3fs`, `pytest`. The per-encoder imports are lazy and fitting never calls them.

## 3. [MAC] Code (~3 MB, seconds)

```bash
cd "/Users/benji/Code/NeurIPS CCAI 2026" && rsync -av --exclude '.venv' --exclude '.git' --exclude 'data/*/' --exclude '.pycache' --exclude '.pytest_cache' --exclude '__pycache__' --exclude '.DS_Store' --exclude '.claude' --exclude 'vendor' --exclude 'paper' --exclude '*.zip' --exclude 'notebooks/runs' ./ go38mid@lxhalle.cit.tum.de:~/p3/
```

## 4. [SERVER] Prove the environment BEFORE the 4 GB copy

Costs one second. Catches an install problem before a 40-minute transfer.

```bash
cd ~/p3 && ~/p3/.venv/bin/python -c "from probes import p3_forecast; from encoders.pipeline import load_masks; import scripts.run_p3_extreme, sys, sklearn; print('imports OK', sys.version.split()[0], sklearn.__version__)"
```

Want: `imports OK 3.12.3 1.6.1`

## 5. [MAC] Data (~4.1 GB, 10–40 min, resumable)

```bash
cd "/Users/benji/Code/NeurIPS CCAI 2026" && rsync -avP data/scaled_32UQC/raw data/scaled_32UQC/embeddings data/scaled_32UQC/embeddings_cir data/scaled_32UQC/masks data/scaled_32UQC/cache_roster.csv go38mid@lxhalle.cit.tum.de:/var/tmp/go38mid/p3/data/scaled_32UQC/
```

`-P` resumes on drop — re-run the identical line.

## 6. [SERVER] Verify the transfer

```bash
cd /var/tmp/$USER/p3/data/scaled_32UQC && for d in raw embeddings embeddings_cir masks; do printf "%-16s %s\n" "$d" "$(ls $d | wc -l)"; done && ls cache_roster.csv
```

Want **348 / 1715 / 1372 / 343**. A short count silently fits a different set
of cubes — do not continue.

## 7. [SERVER] Smoke test, ~2 min

```bash
cd ~/p3 && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ~/p3/.venv/bin/python -m scripts.run_p3_extreme --out /var/tmp/$USER/p3/data/scaled_32UQC --fold-modes loco --aggregations cube_mean --skip-calibration --no-predictions --max-cubes 20 --encoders raw_features,imagenet_vit_b16_cir --n-jobs 8 --csv-name p3_smoke_loco.csv --log-name p3_smoke_loco.log 2>&1 | tail -25
```

Must end with `DONE` and no `Traceback`. Then:

```bash
rm -f /var/tmp/$USER/p3/data/scaled_32UQC/p3_smoke_loco.*
```

## 8. [SERVER] Check the load, THEN pick `--n-jobs`

lxhalle is shared, typically ~300 users logged in.

```bash
uptime; ps -eo user:16,pcpu,comm --sort=-pcpu --no-headers | head -8
```

| 1-min load | use |
|---|---|
| under ~10 | `--n-jobs 64` (all physical cores) |
| 10–50 | `--n-jobs 32` |
| over 60 | `--n-jobs 16`, or come back later |

64 was used at load 48 and completed in 1.95 h under `nice`. Going past 64 uses
SMT siblings, not real cores — typically +10–30%, not +100%.

## 9. [SERVER] The run, in tmux

tmux is what makes the job survive a dropped SSH connection.

```bash
tmux new -s loco
```

Inside tmux, one block:

```bash
cd ~/p3 && export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 && nice -n 10 ~/p3/.venv/bin/python -m scripts.run_p3_extreme --out /var/tmp/$USER/p3/data/scaled_32UQC --fold-modes loco --aggregations cube_mean --skip-calibration --no-predictions --n-jobs 64 --csv-name p3_extreme_loco_results.csv --log-name p3_extreme_loco_run.log 2>&1 | tee /var/tmp/$USER/p3/loco_console.txt
```

Detach: **Ctrl-b**, release, **d**. Reattach later: `tmux attach -t loco`.

**Keep the four thread caps at 1.** They are what make `--n-jobs N` mean "N
CPUs" instead of "N × 64 threads fighting each other".

## 10. [SERVER] Monitoring — grep, never scroll

The loader prints `dropping 242/300 timesteps` once per cube open — thousands
of harmless lines. Do not try to scroll; filter.

```bash
grep "p3-eta" /var/tmp/$USER/p3/data/scaled_32UQC/p3_extreme_loco_run.log | tail -3
```

ETA = `s/row` × 308 ÷ 3600 hours. Ignore the first ~20 rows.

A long silence early is normal: `_horizon_control_folds` prints 342 × 4 = 1368
lines *before* the first counted row.

Full health check:

```bash
echo "=== tmux ==="; tmux ls 2>&1; echo "=== process ==="; pgrep -af run_p3_extreme | head -3 || echo "NOT RUNNING"; L=/var/tmp/$USER/p3/data/scaled_32UQC/p3_extreme_loco_run.log; A=$(stat -c%s $L); sleep 10; B=$(stat -c%s $L); echo "grew $((B-A)) bytes in 10s"; grep -v "^\[loader\]" $L | tail -5; echo "rows:"; grep -c "p3-eta" $L
```

## 11. [SERVER] Package

```bash
cd /var/tmp/$USER/p3/data/scaled_32UQC && tar czf ~/p3_loco_handoff.tar.gz p3_extreme_loco_results.csv p3_extreme_loco_run.log && ls -lh ~/p3_loco_handoff.tar.gz
```

## 12. [MAC] Retrieve and merge

```bash
cd "/Users/benji/Code/NeurIPS CCAI 2026" && scp go38mid@lxhalle.cit.tum.de:~/p3_loco_handoff.tar.gz /tmp/ && tar xzf /tmp/p3_loco_handoff.tar.gz -C data/scaled_32UQC/
```
```bash
cd "/Users/benji/Code/NeurIPS CCAI 2026" && .venv/bin/python -m scripts.merge_loco
```

`merge_loco` refuses unless the halves are the same experiment. If it refuses,
read the message — do not work around it.

## 13. [SERVER] Clean up

See `docs/SERVER_CLEANUP.md`.

---

## Adapting this to another tile or fold mode

The only tile-specific things are the data paths and `cache_roster.csv`. For a
new tile you need its `raw/`, `embeddings/`, `embeddings_cir/`, `masks/` and
roster staged the same way, plus `EXPECTED_CUBES` in
`scripts/run_p3_extreme.py` set to that tile's P4 roster size (currently 346,
hard-coded for 32UQC — rule 1 will stop the run if it disagrees, which is the
intended behaviour, not a bug to route around).

For a different fold mode, drop `--aggregations cube_mean` only if the mode
runs under all three (`cube` does; `loco` and `spatial_block` do not).
