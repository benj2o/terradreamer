# Extreme-tile P3 — how to run it, in order

**One sentence:** the *encoding* happens on Colab (needs a GPU and Python ≥ 3.10),
the *fitting* happens on the Mac (needs 12 CPU-hours and no GPU), and you carry
three folders between them.

**Why it is split.** Two of the nine encoder views (`dinov2_vitb14` and
`dinov2_vitb14_cir`) **cannot be built on this Mac at all** — the dev venv is
Python 3.9.6, and DINOv2's `torch.hub` code uses `X | None`, which is a syntax
error before 3.10. This Mac also has no CUDA (Apple M1). Colab has both. The
fitting run is the opposite: it is pure CPU, takes up to 12 hours, and a Colab
session would die halfway through.

---

## Part A — on the Mac, before Colab (5 minutes)

### A1. Commit first. This is not optional.

`make_zip.sh` builds the bundle from `git ls-files`. **Anything not committed is
not in the bundle**, and Colab will silently run the old code.

```bash
git add -A && git commit -m "feat(p3): extreme-tile encoding notebook + local runner"
```

### A2. Build the bundle

```bash
./make_zip.sh
```

Produces `phase1_10_repo.zip` in the repo root. It contains code only — no cubes,
no embeddings.

### A3. Upload

Drag `phase1_10_repo.zip` into Google Drive at:

```
My Drive / NeurIPS-CCAI-2026 / phase1_10 /
```

Create the `phase1_10` folder if it does not exist. One folder per phase is
deliberate: deleting `phase1_10/` removes this phase's artefacts and nothing else.

---

## Part B — on Colab (about 1–3 hours, mostly unattended)

### B1. Open the notebook

Upload `notebooks/phase1_10_extreme_encoding.ipynb` to Colab (or open it from
Drive).

### B2. Set the GPU runtime BEFORE running anything

**Runtime → Change runtime type → T4 GPU → Save.**

If you skip this, Step 3 prints `DEVICE = 'cpu'` and the encode takes hours
instead of tens of minutes. It still *works* — nothing here requires CUDA — but
there is no reason to pay for it.

### B3. Run the cells in order, top to bottom

Step 1 installs and **restarts the runtime**. That is expected. After the
restart, carry on from Step 2 — do not re-run Step 1.

Steps 6 and 7 download the two Satlas weight files on first run. One-time,
standard, same as every prior cache build.

### B4. STOP if any of these fail

| Where | What it means |
|---|---|
| Step 3 says python < 3.10 | DINOv2 cannot build; you would get a 7-view cache with a hole in it. Do not continue. |
| Step 4 roster ≠ 346 | The split or the non-overlap selection differs from P4's. The two tables would not be about the same place. |
| Step 5 not `0 failed` | Fix the code before building an artefact from it. |
| Step 8 reports dropped cubes | Read the reason. A dropped cube is out of *both* caches and the masks — that is correct behaviour, but the count must be explainable. |
| Step 9 audit fails | The cache has holes. Re-run Step 8; it is resumable and skips what exists. |
| **Step 10 fails** | **This is the important one.** Either the two caches describe different frames (no `_cir`/`_rgb` comparison is valid) or the embeddings are identical (the band swap never reached the network). Either way the headline is void. |

### B5. What you should see when it worked

```
rgb embeddings 1730 .npz     (346 × 5)
cir embeddings 1384 .npz     (346 × 4)
masks           346 .npz
both caches cover the SAME 346 cubes
embeddings differ on all 100 pairs (0 identical)
```

---

## Part C — carry the caches down (10–20 minutes)

Copy these **four things** from Drive into `data/scaled_32UQC/` on the Mac:

```
embeddings/          ~1 GB
embeddings_cir/      ~1 GB
masks/               ~2 MB
cache_roster.csv     tiny, but it is the contract — copy it
```

**Do not copy the cubes.** `data/scaled_32UQC/raw/` already holds all 348 on the
Mac (2.2 GB).

Check you have room first — the volume was at 96% when this was written:

```bash
df -h . | tail -1
```

Then confirm the counts landed:

```bash
ls data/scaled_32UQC/embeddings/*.npz | wc -l && ls data/scaled_32UQC/embeddings_cir/*.npz | wc -l && ls data/scaled_32UQC/masks/*.npz | wc -l
```

Expect `1730`, `1384`, `346`.

---

## Part D — on the Mac, the fitting run (up to 12 hours)

### D1. Launch it in the background and walk away

```bash
nohup .venv/bin/python -m scripts.run_p3_extreme --n-jobs 7 > /dev/null 2>&1 &
```

It tees everything to `data/scaled_32UQC/p3_extreme_run.log`.

### D2. Watch it

```bash
tail -f data/scaled_32UQC/p3_extreme_run.log
```

### D3. What it does on its own — you do not answer any questions

1. **Rule 1 — roster.** Reads `cache_roster.csv` (or falls back to the P4 CSV),
   asserts 346 cubes, prints every exclusion and why. Stops if the count is wrong.
2. **Rule 2 — caches.** Asserts both caches are complete *and cover the same
   cubes*. Stops if not. Encodes nothing; imports no encoder.
3. **Rule 3 — the runtime valve.** Runs the grid on 20 then 40 cubes, measures
   seconds-per-row **per fold mode**, fits one growth exponent per mode, and
   projects the full-tile time. It does not assume linearity anywhere — `loco`
   was measured at exponent 1.72 on this tile, and a linear estimate has been
   measured 4–6× low.
4. **Rule 4 — narrowing, only if the projection exceeds 12 hours.** In this
   order: drop `loco`, then `cell_mean`, then `fixed_alpha_D`.
   **All nine encoder views survive every narrowing.** If it narrows, it writes
   `p3_extreme_subset_results.csv` instead, and says so in every table.
5. Runs the full grid with `emit_predictions=True`, then scores the trigger
   metrics — no second fitting run.

### D4. What you get

```
data/scaled_32UQC/p3_extreme_results.csv        (or _subset_ if narrowed)
data/scaled_32UQC/p3_extreme_predictions.csv
data/scaled_32UQC/p3_extreme_triggers.csv
data/scaled_32UQC/p3_extreme_run.log
```

---

## The three rules that never bend

1. **Never touch a 32UNU artefact**, and never write to `data/raw` or
   `data/phase1_2`. This run writes only under `data/scaled_32UQC/`.
2. **All nine encoder views survive every narrowing.** Model coverage is what
   this run was commissioned for; fold modes and aggregations are what pay for it.
3. **A narrowed table is a subset table.** It goes under its own filename, the
   full-table completeness assertions are not run on it, and it says so in the
   output. The only thing worse than a subset table is a subset table that
   passed the checks the full one is held to.

---

## If you only remember five commands

```bash
git add -A && git commit -m "feat(p3): extreme-tile encoding notebook + local runner"
```
```bash
./make_zip.sh
```
```bash
ls data/scaled_32UQC/embeddings/*.npz | wc -l
```
```bash
nohup .venv/bin/python -m scripts.run_p3_extreme --n-jobs 7 > /dev/null 2>&1 &
```
```bash
tail -f data/scaled_32UQC/p3_extreme_run.log
```
