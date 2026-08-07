# 2026-08-07: EO-WM author correspondence

Source: private email reply from the EO-WM corresponding author
(arXiv:2606.27277), received 2026-08-07, in response to our request for
benchmark/eval code. Cite as personal communication.

---

## (a) Persistence is not a main-table baseline

Tab. 1 (Extreme Summer) and Tab. 2 (Seasonal Matched-Pair) do not use
persistence. "Copy last clear frame" and "persistence (cloud-free mean)"
appear only in Appendix A.1 as trivial reconstruction references for the
tokenizer diagnostic, not as forecasting baselines.

**Implication.** `research_plan_v3` repeatedly framed comparisons against
"EO-WM's published persistence rows". Those rows do not exist for
forecasting. Persistence remains the correct P3 baseline, but we establish
it ourselves rather than inheriting it.

## (b) Earthformer was self-trained, not an official checkpoint

Retrained on EarthNet2021 from the `CuboidEarthNet2021` module with meso
auxiliary input. Config: `in_len` 10 / `out_len` 20, 128x128, 4 optical
channels (B/G/R/NIR), `auxiliary_channels` 7, `base_units` 256,
`total_batch_size` 32, 200 epochs.

**Implication.** Reproducing their baseline rows requires retraining a
200-epoch model. See the K1 status change in
[docs/DECISIONS.md](../DECISIONS.md).

## (c) Data and masks

They use the public EarthNet2021 npz format and did NOT regenerate cloud
masks. EO-WM uses a cloud-probability threshold of 0.2 for clear-sky
pixels; the Earthformer baseline uses the npz binary mask directly.
NDVI = (NIR - Red)/(NIR + Red + eps). The EO-VAE finetune separately uses
frame-level quality filtering at `data_qua_thresh = 0.3`, which is NOT the
same setting as the cloud probability threshold.

**Implication.** Our valid-pixel basis differs from theirs. We use
GreenEarthNet's `s2_dlmask` AND the `s2_SCL` allow-list (see the
2026-08-02 log entry); they use EarthNet2021-era masks. Any cross-paper
number carries this caveat, stated once. This vindicates the mask
correction already recorded in DECISIONS.

## (d) Climatology file

`era5_climatology_all.pt` holds per-tile, per-month mean/std for 5 meso
channels (precipitation, pressure, mean/min/max temperature) over 100
tiles, shape `[100, 12, 5]`, plus a global `[12, 5]` fallback. Tile 32UNU
IS in the tile list. The generation script was not shared.

**Implication.** This is a WEATHER climatology, not an NDVI climatology.
It does not unblock P4's post-climatology anomaly target, which still
requires the multi-year seasonal split. What it does give is their exact
normalisation for 5 channels; our in-cube E-OBS has 8
(tg/tn/tx/rr/pp/fg/hu/qq), a superset, so P4 can report an
EO-WM-comparable 5-variable version alongside the full 8-variable version.

Also record: EO-WM aligns the meteorological sequence to the Sentinel-2
timeline at EarthNet2021's meso temporal resolution; the Earthformer
baseline uses nearest interpolation to align auxiliary data to model
input length and spatial resolution.

## (e) Reconstruction diagnostic and SSIM convention

Masked MAE/MSE are computed on valid pixels only, using EarthNet2021's
mask channel. EarthNet2021 ENS SSIM follows the original toolkit:
`skimage.structural_similarity` is called only on frames at least 70%
valid, `data_range` is deliberately NOT passed, and the raw SSIM is taken
before ENS's power scaling (scaling factor 10.31885115, which maps SSIM
0.8 to 0.1).

**Implication.** Dormant for now. Our probes predict NDVI, not images, so
ENS does not apply to P1-P4. It becomes binding only if H3 wins the
convergence gate and we compute EarthNetScore. Recorded so it is not
rediscovered then.

## (f) Release timing and inference cost

Core model and training code will be open-sourced after paper acceptance.
Reference cost, single A100-80GB, bf16, 30 denoising steps, 5 samples,
in10/out20, 128x128: approximately 8.2 s per 5-sample rollout, 1.82 s
single-sample.

**Implication.** This is an authoritative number for H3's FLOPs/cost
table, replacing our estimate. Attribute it to personal communication,
not to the paper.

---

## Artifacts (not committed)

The reply included a bundle containing `era5_climatology_all.pt`, the
EO-VAE tree, the Earthformer tree, and their configs. Both code trees are
Apache-2.0, so redistribution is permitted, but the core EO-WM model code
is unreleased pending acceptance and the climatology artifact was shared
privately with its generation script withheld. This repository is
PUBLIC.

The bundle lives locally, outside this repo, at
`/Users/benji/Downloads/earthnet_resource.zip`, extracted to
`vendor/eowm/` (gitignored, not tracked). It contains:

- `vendor/eowm/EO-VAE/` -- EO-VAE finetune config and reconstruction eval
  scripts (Apache-2.0).
- `vendor/eowm/earth-forecasting-transformer/` -- the self-trained
  Earthformer baseline, config `scripts/cuboid_transformer/earthnet_w_meso/cfg.yaml`
  (Apache-2.0).
- `vendor/eowm/era5_climatology_all.pt` -- the weather climatology
  described in (d) above, shared privately, generation script withheld.

None of these files are copied into any tracked path. Committing any of
them to this public repository requires the authors' permission first.
