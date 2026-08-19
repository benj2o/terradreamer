# Files worth flagging for deletion

Audited 2026-08-19, after the extreme-tile P3 table was completed. Repo is
**7.3 GB**, of which 7.2 GB is `data/` and gitignored.

**Nothing here is deleted automatically.** Each row says what it is, whether it
is reproducible, and what it would cost to get back.

---

## Safe to delete now — pure junk, zero cost

| path | size | what |
|---|---|---|
| `.pycache/` | **40 MB** | a mirror of the macOS stdlib `.pyc` tree. Not `__pycache__`; a stray directory. **Not in `.gitignore`** — it got rsynced to the server once because of that |
| `.pytest_cache/` | 64 KB | pytest scratch |
| `**/.DS_Store` | tiny | macOS folder metadata |

```bash
rm -rf .pycache .pytest_cache && find . -name '.DS_Store' -not -path './.git/*' -delete
```

Add `.pycache/` to `.gitignore` so it cannot be shipped again.

---

## Delete when you are finished with 32UQC triggers — the big one

| path | size | notes |
|---|---|---|
| `data/scaled_32UQC/p3_extreme_predictions.csv.gz` | **362 MB** | 10.3 M held-out predictions. Already consumed to produce `p3_extreme_triggers.csv` |

This is **half a working day to regenerate** (a full re-fit with
`emit_predictions=True`). Keep it until the paper's trigger figures are final,
then delete. It is the single largest deletable artefact.

---

## Delete only if you are truly done with that experiment

| path | size | regeneration cost |
|---|---|---|
| `data/validation_30TVN/` | **1.8 GB** | cubes + results for the proxy-climatology validation (2026-08-10). Re-downloadable via `scripts/validate_proxy_climatology.py` |
| `data/scaled_32UQC/raw/` | 2.1 GB | re-downloadable in ~3 min via `scripts/scale_p4.download` — **but** only if GreenEarthNet's `extreme` split selection is stable |
| `data/scaled_32UQC/{embeddings,embeddings_cir}/` | 2.1 GB | **needs a GPU Colab session (~30 min)** to rebuild. Keep while the paper is live |
| `data/scaled_32UNU/` | 912 MB | the published Tier-1 tile. **Keep** — every comparison in the paper reads against it |

---

## Keep — superseded but load-bearing for provenance

| path | size | why keep |
|---|---|---|
| `data/scaled_32UQC/p3_extreme_subset_results.csv` | 3.6 MB | the 1232-row local table. Superseded by the merged 1540, but it is the record of what the Mac actually computed, and `log.md` cites it |
| `data/scaled_32UQC/p3_extreme_loco_results.csv` | 3.4 MB | the 308 server rows, likewise |
| `notebooks/runs/*.txt` | 27 MB | archived run logs. `RUN_INDEX.md` points at them by name |
| `data/scaled_32UQC/cache_roster.csv` | 26 KB | the contract that fixes which 342 cubes every table covers. **Never delete** |

---

## Not mine to delete — flagging for your decision

| path | size | issue |
|---|---|---|
| `vendor/eowm/` | **5.9 MB** | unreleased third-party model code + `era5_climatology_all.pt`, shared privately. Gitignored deliberately. It was rsynced to lxhalle once and removed there. Consider whether it should live in this checkout at all |
| `docs/correspondence/2026-08-07-eowm-authors.md` | 5 KB | contents of a **private email** from the EO-WM corresponding author, marked "cite as personal communication". Unlike `vendor/`, this **is tracked in git** — so it is in the GitHub repo. If `benj2o/terradreamer` is public, that email is published. Worth checking before the paper goes out |
| `.claude/` | 16 KB | local editor/agent config incl. a `launch.json` pointing at a stale temp path from another session. Untracked; harmless but noise |

---

## One-shot: the safe deletions only

```bash
cd "/Users/benji/Code/NeurIPS CCAI 2026" && du -sh . && rm -rf .pycache .pytest_cache && find . -name '.DS_Store' -not -path './.git/*' -delete && echo "--- after ---" && du -sh .
```

Frees ~40 MB and removes the directory that caused the accidental server upload.
