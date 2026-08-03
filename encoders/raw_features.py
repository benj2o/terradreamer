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

from encoders.base import GRID, GRID_CELLS, FrozenEncoder

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
    # The SAME 35 statistics computed independently per grid cell. The
    # baseline must stay spatially comparable to the networks or it stops
    # being a fair baseline: comparing a 35-dim whole-frame summary against a
    # 4x4x768 grid would confound representation quality with spatial
    # resolution.
    grid_dim = len(RAW_FEATURE_NAMES)   # 35 per cell, 35 x 16 = 560 flattened
    FEATURE_RECIPE = (
        "not a network; pooled = 35 whole-frame statistics; grid = the same 35 "
        "statistics recomputed independently per 4x4 cell (35 x 16 = 560), NDVI "
        "via data.ndvi.ndvi per cell"
    )
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

    def _features_batch(self, frames: torch.Tensor, mask: torch.Tensor | None) -> dict:
        assert mask is not None  # enforced upstream by _check_mask
        v = frames.cpu().numpy()
        m = mask.cpu().numpy()
        B, C, H, W = v.shape
        assert H % GRID == 0 and W % GRID == 0, (
            f"{self.name}: {H}x{W} is not divisible by the {GRID}x{GRID} grid"
        )
        ch, cw = H // GRID, W // GRID

        pooled = self._stats_block(v, m)                            # (B, 35)
        cells = []
        for gy in range(GRID):
            for gx in range(GRID):
                ys, xs = slice(gy * ch, (gy + 1) * ch), slice(gx * cw, (gx + 1) * cw)
                cells.append(self._stats_block(v[:, :, ys, xs], m[:, ys, xs]))
        grid = np.stack(cells, axis=1)                              # (B, 16, 35)
        assert grid.shape == (B, GRID_CELLS, self.grid_dim)
        return {"pooled": torch.from_numpy(pooled),
                "grid": torch.from_numpy(grid)}

    def _stats_block(self, v: np.ndarray, m: np.ndarray) -> np.ndarray:
        """(B, C, h, w) + (B, h, w) -> (B, 35). Used whole-frame and per cell."""
        B, C, H, W = v.shape

        flat = v.reshape(B, C, H * W)
        n_finite = np.isfinite(flat).sum(axis=-1)
        if not (n_finite > 0).all():
            # Whole-frame this cannot happen (select_clear_frames asserts it).
            # Per CELL a 32x32 block can be entirely no-data; substitute the
            # band's frame-level median so the reduction is defined, and let
            # grid_clear_frac carry the fact that the cell is empty.
            flat = flat.copy()
            for b in range(C):
                bad = n_finite[:, b] == 0
                if bad.any():
                    good = flat[~bad, b] if (~bad).any() else None
                    fill = float(np.nanmedian(good)) if good is not None and good.size else 0.0
                    flat[bad, b] = fill
        band_feats = _stats(flat, nan_aware=True).reshape(B, -1)
        assert band_feats.shape == (B, C * len(_STAT_NAMES))

        i_nir, i_red = S2_BANDS.index("B8A"), S2_BANDS.index("B04")
        nd = ndvi(v[:, i_nir], v[:, i_red], m)  # (B, H, W), NaN where masked
        assert nd.shape == (B, H, W)
        nd_flat = nd.reshape(B, H * W)
        n_valid = np.isfinite(nd_flat).sum(axis=-1)
        # Whole-frame, a zero-valid frame cannot pass the clear-fraction rule.
        # Per CELL it can and does: a 32x32 cell may be fully clouded inside an
        # otherwise clear frame. The contract is NaN, never 0 -- but a NaN
        # feature would poison every probe, so such cells fall back to the
        # frame-level statistics and are flagged through grid_clear_frac, which
        # is exactly what probes filter cells on.
        if not (n_valid > 0).all():
            empty = n_valid == 0
            frame_med = np.nanmedian(nd_flat[~empty]) if (~empty).any() else 0.0
            nd_flat = nd_flat.copy()
            nd_flat[empty] = frame_med
        ndvi_feats = _stats(nd_flat, nan_aware=True)
        assert ndvi_feats.shape == (B, len(_STAT_NAMES))

        out = np.concatenate([band_feats, ndvi_feats], axis=-1).astype(np.float32)
        assert out.shape == (B, self.embed_dim)
        assert np.isfinite(out).all(), f"{self.name}: non-finite statistic"
        return out
