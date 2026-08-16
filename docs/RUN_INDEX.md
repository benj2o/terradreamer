# Run Index

A lookup table, not a phase list. `docs/DECISIONS.md` explains *why* each
change happened; this file only answers "when I say X, which files am I
pointing at?" Two different naming systems coexist on purpose and are not
merged:

- **Phase 1.N** is load-bearing in code — `data/paths.py` resolves artefacts
  to `data/phase1_N/<kind>`. Renaming a phase means editing the resolver and
  moving caches. These names are frozen.
- **Everything else below** (Tier-1, Screened P2/P4, ...) is a rerun or
  correction layered on top, living in `data/scaled_32UNU/` with descriptive
  filenames rather than its own phase folder. That's a feature, not a gap:
  it marks these as corrections to an existing result, not new phases.

| Name (say this) | What it actually is | Code entry point | Artefacts | Date | DECISIONS.md anchor |
|---|---|---|---|---|---|
| **Phase 1.1** | Data source switch, live extraction → pre-processed GreenEarthNet | `data/download_greenearthnet.py` | `data/phase1_2/` (bundled with 1.2) | 2026-08-02 | `## 2026-08-02: Data source...` |
| **Phase 1.2 / 1.2b / 1.2c** | Frozen encoder wrappers, patch grid, land cover, weather stack | `encoders/*.py` | `data/phase1_2/embeddings/`, `notebooks/runs/phase1_2_encoders_*.ipynb` | 2026-08-03 | `## 2026-08-03: Phase 1.2...` (multiple entries) |
| **Phase 1.3** | Cross-validation modes (cube/loco/spatial_block/crossed) | `probes/cv.py` | `notebooks/runs/phase1_3_cv_*.ipynb` | 2026-08-05/06 | `## 2026-08-05: Phase 1.3...` |
| **Phase 1.4 (P1)** | Appearance probe — month/season classification | `probes/p1_appearance.py` | `notebooks/runs/phase1_4_p1_appearance_*.ipynb` | 2026-08-08/09 | `## 2026-08-08: Phase 1.4, P1...` |
| **Phase 1.5 (P4)** | Weather-attributability ceiling, proxy climatology | `probes/p4_ceiling.py` | `notebooks/runs/phase1_5_p4_ceiling_*.ipynb` | 2026-08-10 | `## 2026-08-10:` / `## 2026-08-11: Stage B...` |
| **Phase 1.6 (P2)** | Delta/dynamics probe — sign vs. magnitude | `probes/p2_deltas.py` | `notebooks/runs/phase1_6_p2_deltas_*.ipynb` | 2026-08-11 | `## 2026-08-11: Phase 1.6, P2...` (multiple) |
| **Phase 1.7** | 115-cube scale-up infrastructure | `scripts/scale_p2.py`, `scripts/scale_p4.py` | `data/scaled_32UNU/` created here | 2026-08-11 | see P2/P4 scale-up entries |
| **Phase 1.8 (P3, first run)** | First forecastability probe, fixed `alpha=D`, marginal CIs | `probes/p3_forecast.py` (pre-tier1) | `data/phase1_8/results/p3_forecast_results.csv`, `notebooks/runs/phase1_8_p3_forecast_2026-08-12_localCPU.ipynb` | 2026-08-12 | `## 2026-08-12: Phase 1.8, P3...` (multiple) |
| **Phase 1.9** | Colour-infrared (CIR) re-encode — cache build only, no new probe | `notebooks/phase1_9_cir_encoding.ipynb` | `data/scaled_32UNU/embeddings_cir/` | 2026-08-12 | commit `27bede5`, `a33db3d` |
| **Tier-1** | P3 *rerun* correcting Phase 1.8: nested-CV alpha selection, paired-delta significance, CIR views added, plausibility screen applied | `scripts/rerun_p3_tier1.py`, `probes/p3_forecast.py` (post-tier1) | `data/scaled_32UNU/p3_tier1_results.csv`, `data/scaled_32UNU/p3_tier1_run.log`, `notebooks/runs/2026-08-12_p3_tier1_32UNU_115cubes.txt` | 2026-08-12 | `## 2026-08-12: Tier 1, P3...` (4 entries) |
| **Screened P2/P4** | P2 and P4 rerun with the same plausibility screen Tier-1 applied, for like-for-like comparability | `scripts/rerun_p2_screened.py`, `scripts/rerun_p4_screened.py` | `notebooks/runs/2026-08-13_p2_screened_32UNU_115cubes.txt`, `notebooks/runs/2026-08-13_p4_screened_32UNU_115cubes.txt` | 2026-08-13 | `## 2026-08-13: Screened P2/P4...` |
| **The claim/impact memo** | Strategy doc for what to run before Aug 28, not an experiment | — | `docs/CLAIM_IMPACT_EXPERIMENTS.md` | 2026-08-13 | n/a (not a DECISIONS entry) |
| **Tier-1 trigger metrics** | Threshold-crossing skill (hit rate / false-alarm rate / CSI / Peirce, vs persistence) on the Tier-1 headline configs. Needs a P3 re-run with `emit_predictions=True`; the published Tier-1 run wrote no per-observation artefact | `probes/p3_triggers.py`, `scripts/rerun_p3_predictions.py` | `data/scaled_32UNU/p3_tier1_predictions.csv`, `data/scaled_32UNU/p3_tier1_triggers.csv`, `data/scaled_32UNU/p3_tier1_subset_results.csv` | 2026-08-16 | `## 2026-08-16: Tier-1 trigger metrics...` |

## Rule for adding a new row

Adding a new rerun or correction: append a row here with the same six
columns, using whatever descriptive name people actually say out loud. Do
**not** invent a new phase number for it unless it introduces new code
structure that `data/paths.py` needs to own — a correction to an existing
phase's probe is a rerun of that phase, not a new one.
