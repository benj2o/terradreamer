"""Encoder 5: SatlasPretrain Sentinel-2 Swin-B MULTI-IMAGE. The positive control.

Every other wrapper in the roster is single-image: it sees one frame and
cannot, even in principle, represent change. A negative result from those
alone is weak, because "frozen EO representations lose dynamics" is not
distinguishable from "we only ever showed them one frame". This encoder is
explicitly temporal, so it is the positive control: if nothing in the roster
recovers dynamics INCLUDING this one, the null is about the representations;
if this one does and the single-image encoders do not, the finding is about
temporal context, which is a different and also publishable claim.

It needs no new dependency -- `satlaspretrain-models` is already installed and
ships `Sentinel2_SwinB_MI_RGB` alongside the SI backbone. That makes it the
cheapest available answer to the n=1 EO-native problem that dropping Prithvi
opened, and it does not wait on Clay.

INPUT CONVENTION, read out of the package rather than assumed:
`AggregationBackbone` declares `self.groups = [[0..7]]`, so an MI model takes
**8 images stacked channel-wise** (8 x 3 = 24 channels for RGB) and max-pools
the per-image features across the group. Output depth is therefore unchanged
at 1024.

TEMPORAL WINDOW. For retained frame t the window is the 8 most recent retained
frames ending at t. At the start of a cube there are fewer than 8, so the
earliest available frame is repeated to fill -- the alternative, dropping the
first 7 frames of every cube, would cost ~7/13 of this subset. The window is
built over RETAINED frames and the gaps between them are irregular; probes must
therefore keep reading `original_axis_index` for anything horizon-related, and
must not treat this embedding as a fixed-duration lookback.
"""

from __future__ import annotations

import torch

from encoders.base import FrozenEncoder, pool_to_grid, composite_from_s2, resize_bilinear

__all__ = ["SatlasS2SwinBMI"]


class SatlasS2SwinBMI(FrozenEncoder):
    name = "satlas_s2_swinb_mi_rgb"
    embed_dim = 1024
    grid_dim = 1024
    variant_dims: dict = {}
    model_identifier = "Sentinel2_SwinB_MI_RGB"
    n_images = 8          # AggregationBackbone.groups == [[0..7]]
    window_len = 8        # makes the pipeline cache window_span_days
    size_multiple = 32
    FEATURE_RECIPE = (
        "multi-image Swin-B, 8 retained frames stacked channel-wise (24 ch), "
        "features max-pooled across the group; pooled = global average pool of "
        "the final stage = 1024; grid = same map adaptive-avg-pooled to 4x4. "
        "No CLS token (Swin is hierarchical). THE POSITIVE CONTROL: the only "
        "encoder in the roster that can represent change at all"
    )

    def __init__(self, *a, **kw):
        self._ctx = None
        super().__init__(*a, **kw)

    def _build(self):
        import satlaspretrain_models

        return satlaspretrain_models.Weights().get_pretrained_model(
            self.model_identifier, fpn=False, device="cpu"
        )

    def _reset_state(self) -> None:
        # Context must never leak from one cube into the next.
        self._ctx = None

    def _preprocessing_lines(self) -> list:
        return [
            "RGB from S2 bands: (B04, B03, B02) -> (R, G, B), by name from S2_BANDS",
            "non-finite (no-data) pixels -> NONFINITE_FILL, counted; before the clamp",
            "SatlasPretrain S2-RGB convention: x = clamp(2.5 * reflectance, 0, 1)",
            f"temporal window: the {self.n_images} most recent RETAINED frames ending "
            "at t, stacked channel-wise; the earliest frame is repeated when fewer "
            "than 8 exist, rather than dropping the first 7 frames of every cube",
            "window spans IRREGULAR gaps -- use original_axis_index for horizons",
            f"embedding = global average pool of the final Swin-B stage, D={self.embed_dim}",
        ]

    def _prep(self, frames: torch.Tensor) -> torch.Tensor:
        x = composite_from_s2(frames, self.composite)
        x = self._sanitise(x)
        x = torch.clamp(2.5 * x, 0.0, 1.0)
        H, W = x.shape[-2:]
        if H % self.size_multiple or W % self.size_multiple:
            side = max(self.size_multiple,
                       self.size_multiple * round(max(H, W) / self.size_multiple))
            x = resize_bilinear(x, side)
        return x

    def _windows(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) prepared frames -> (B, 8*3, H, W) temporal windows.

        ``self._ctx`` carries the tail of the previous batch so that windows are
        identical regardless of batch_size -- the batching must not change a
        single number.
        """
        ctx = self._ctx if self._ctx is not None else x[:1]      # repeat earliest
        seq = torch.cat([ctx, x], dim=0)                          # (n_ctx + B, 3, H, W)
        n_ctx = ctx.shape[0]
        B = x.shape[0]

        out = []
        for i in range(B):
            end = n_ctx + i + 1
            start = end - self.n_images
            if start < 0:                                          # pad by repeating
                pad = seq[:1].expand(-1 * start, -1, -1, -1)
                win = torch.cat([pad, seq[:end]], dim=0)
            else:
                win = seq[start:end]
            assert win.shape[0] == self.n_images, win.shape
            out.append(win.reshape(1, self.n_images * 3, *win.shape[-2:]))

        self._ctx = seq[-(self.n_images - 1):].detach()            # tail for next batch
        stacked = torch.cat(out, dim=0)
        assert stacked.shape == (B, self.n_images * 3, x.shape[-2], x.shape[-1])
        return stacked

    def _features_batch(self, frames: torch.Tensor, mask) -> dict:
        x = self._prep(frames)
        feats = self._model(self._windows(x).to(self.device))
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        assert feats.ndim == 4 and feats.shape[:2] == (frames.shape[0], self.embed_dim), (
            f"{self.name}: got {tuple(feats.shape)}, expected "
            f"({frames.shape[0]}, {self.embed_dim}, h, w)"
        )
        return {"pooled": feats.mean(dim=(2, 3)), "grid": pool_to_grid(feats)}
