"""Encoder 4 of 4: the raw-feature baseline. NOT a network.

This is the row that gives every later probe its meaning: if a frozen
foundation model cannot beat per-frame summary statistics, its embedding
carries no usable dynamics signal beyond brightness and greenness.

Per frame, D = 35 features in a fixed, named order:

* per band (4 bands x 7): spatial mean, std, and the 10th/25th/50th/75th/90th
  percentiles, computed over ALL pixels of the UNMODIFIED frame -- exactly the
  input the network encoders see, clouds included. Matching the input domain
  is the point: the baseline must summarise the same evidence, not cleaner
  evidence. The reductions are NaN-aware: cloudy pixels have values and are
  kept, but no-data pixels have nothing to contribute and a plain mean over
  them would return NaN for the whole frame. This baseline therefore needs no
  fill sentinel, unlike the network wrappers -- it can simply skip them.
* NDVI (7): the same seven statistics over VALID pixels only, computed via the
  canonical ``data.ndvi.ndvi`` (never re-implemented), which is masked by
  definition and returns NaN at masked pixels. NaN-aware reductions are used,
  and a frame with zero valid NDVI pixels is refused loudly: producing a NaN
  (never 0) is the contract, but such a frame cannot pass the clear-fraction
  rule, so reaching one here means the upstream selection was bypassed.

No PCA, deliberately: spatial PCA components do not align across
geographically distinct cubes, whereas per-frame percentiles are dimensionally
stable and geography-agnostic.
"""

from __future__ import annotations

import numpy as np
import torch

from data.loader import S2_BANDS
from data.ndvi import ndvi

from encoders.base import FrozenEncoder

__all__ = ["PERCENTILES", "RAW_FEATURE_NAMES", "NDVI_FEATURE_SLICE", "RawFeatureBaseline"]

PERCENTILES = (10, 25, 50, 75, 90)
_STAT_NAMES = ("mean", "std") + tuple(f"p{p}" for p in PERCENTILES)

# 4 bands x 7 stats, band-major, then 7 NDVI stats. len == 35 == embed_dim.
RAW_FEATURE_NAMES = tuple(
    f"{band}_{stat}" for band in S2_BANDS for stat in _STAT_NAMES
) + tuple(f"NDVI_{stat}" for stat in _STAT_NAMES)

# Columns holding the NDVI statistics (the mask-respecting half of the vector).
NDVI_FEATURE_SLICE = slice(len(S2_BANDS) * len(_STAT_NAMES), len(RAW_FEATURE_NAMES))


def _stats(flat: np.ndarray, nan_aware: bool) -> np.ndarray:
    """(..., N) -> (..., 7) [mean, std, p10, p25, p50, p75, p90] along the last axis."""
    if nan_aware:
        mean, std = np.nanmean(flat, axis=-1), np.nanstd(flat, axis=-1)
        pct = np.nanpercentile(flat, PERCENTILES, axis=-1)
    else:
        mean, std = flat.mean(axis=-1), flat.std(axis=-1)
        pct = np.percentile(flat, PERCENTILES, axis=-1)
    # np.percentile puts the percentile axis FIRST; move it last.
    pct = np.moveaxis(pct, 0, -1)
    out = np.concatenate([mean[..., None], std[..., None], pct], axis=-1)
    assert out.shape == flat.shape[:-1] + (len(_STAT_NAMES),)
    return out


class RawFeatureBaseline(FrozenEncoder):
    name = "raw_features"
    embed_dim = len(RAW_FEATURE_NAMES)  # 35
    requires_mask = True  # data.ndvi.ndvi requires the cloud mask

    def _build(self):
        return None  # not a network; the base class prints exactly that

    def _preprocessing_lines(self) -> list:
        i_nir, i_red = S2_BANDS.index("B8A"), S2_BANDS.index("B04")
        return [
            f"band stats over ALL pixels of the UNMODIFIED frame (clouds included, "
            f"same input the network encoders see), NaN-aware so no-data pixels are "
            f"skipped rather than filled: per band in {S2_BANDS}, "
            f"[{', '.join(_STAT_NAMES)}]",
            f"NDVI via the canonical data.ndvi.ndvi(B8A=ch{i_nir}, B04=ch{i_red}, mask): "
            f"masked pixels are NaN, stats are NaN-aware over VALID pixels only, "
            f"[{', '.join(_STAT_NAMES)}]",
            "no resize, no normalisation, no PCA (spatial PCA components do not "
            "align across geographically distinct cubes)",
        ]

    def _encode_batch(self, frames: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        assert mask is not None  # enforced upstream by _check_mask
        v = frames.cpu().numpy()
        m = mask.cpu().numpy()
        B, C, H, W = v.shape

        flat = v.reshape(B, C, H * W)
        n_finite = np.isfinite(flat).sum(axis=-1)
        assert (n_finite > 0).all(), (
            f"{self.name}: {int((n_finite == 0).sum())} (frame, band) pair(s) have no "
            "finite pixel at all, so every band statistic would be a reduction over "
            "nothing. Such a frame cannot pass the clear-fraction rule -- "
            "encoders.frames.select_clear_frames was bypassed. Fix the caller."
        )
        band_feats = _stats(flat, nan_aware=True).reshape(B, -1)
        assert band_feats.shape == (B, C * len(_STAT_NAMES))

        i_nir, i_red = S2_BANDS.index("B8A"), S2_BANDS.index("B04")
        nd = ndvi(v[:, i_nir], v[:, i_red], m)  # (B, H, W), NaN where masked
        assert nd.shape == (B, H, W)
        nd_flat = nd.reshape(B, H * W)
        n_valid = np.isfinite(nd_flat).sum(axis=-1)
        assert (n_valid > 0).all(), (
            f"{self.name}: {int((n_valid == 0).sum())} frame(s) have ZERO valid NDVI "
            "pixels. The contract for an empty spatial reduction is NaN (never 0), "
            "but a NaN embedding would poison every probe -- and such a frame "
            "cannot pass the clear-fraction rule, so its presence here means "
            "encoders.frames.select_clear_frames was bypassed. Fix the caller."
        )
        ndvi_feats = _stats(nd_flat, nan_aware=True)
        assert ndvi_feats.shape == (B, len(_STAT_NAMES))

        out = np.concatenate([band_feats, ndvi_feats], axis=-1).astype(np.float32)
        assert out.shape == (B, self.embed_dim)
        return torch.from_numpy(out)
