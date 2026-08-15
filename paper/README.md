# CCAI 2026 workshop paper (draft)

## Status (2026-08-15)

- `main.tex` + `appendix.tex` + `refs.bib` — anonymized 4-page Papers-track draft.
- `figures/` — plots extracted from Tier-1 / screened CSVs (plus P1 latent clock).
- Rebuild plots (needs local CSVs under `data/scaled_32UNU/`):
  `MPLCONFIGDIR=.pycache/mpl python paper/make_figures.py`
- Build: `make -C paper clean all` (uses `tectonic`, `latexmk`, or `pdflatex`).
- PDF: `paper/main.pdf` (target: 4 pages main + appendix).

### Figures

| file | source |
|---|---|
| `fig0_latent_clock.png` | copied from `data/phase1_4/figures/` (P1 notebook Step 14) |
| `fig1_headline_r2.{pdf,png}` | `data/scaled_32UNU/p3_tier1_results.csv` |
| `fig2_paired_vs_bandmatched.{pdf,png}` | same Tier-1 P3 CSV (paired vs `raw_rgb_only`) |
| `fig3_sign_vs_magnitude.{pdf,png}` | `data/scaled_32UNU/p2_screened_results.csv` |
| `fig4_extreme_tails.{pdf,png}` | Tier-1 P3 extreme-tail skill columns |

## Before submission (checklist)

1. **OpenReview account** — create immediately at openreview.net; CCAI warns
   creation can take up to 2 weeks (CFP recommended date: 15 Aug 2026).
2. Swap the self-contained `article` class for the **official CCAI workshop
   template** from https://www.climatechange.ai/events/neurips2026.
3. Keep the PDF double-blind (no names, affiliations, or identifying URLs).
4. Freeze experiments **26 Aug**; dry-run upload **27 Aug**; submit **28 Aug**
   (deadline 29 Aug 23:59 AoE — do not use that as the plan).
5. Public code + frozen CSVs: see `docs/CCAI_SUBMIT.md`.

## Claim spine

Resource-light evaluation: simple baselines + weather already forecast NDVI;
frozen EO FMs do not add short-horizon skill on this tile; direction is in the
pictures, rate is not; no frozen network clears both extreme tails at any
horizon.
