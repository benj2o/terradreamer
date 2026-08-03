"""Encoder 3 of 4: SatlasPretrain Sentinel-2 Swin-B (satlaspretrain-models). EO-native.

Model choice, recorded in docs/DECISIONS.md: ``Sentinel2_SwinB_SI_RGB``, the
single-image RGB variant, NOT the multi-spectral ``..._SI_MS`` one. The MS
variant expects 9 channels (TCI + B05, B06, B07, B08, B11, B12); our cubes
carry only (B02, B03, B04, B8A), and zero-filling six missing bands would be
inventing data the spec forbids filling. The RGB variant is fully served by
the bands we have.

Input convention: SatlasPretrain Sentinel-2 RGB models are trained on TCI
divided by 255, i.e. approximately ``clip(2.5 * BOA reflectance, 0, 1)``. That
clamp is this model's trained input distribution, applied and printed inside
this wrapper only -- the network encoders elsewhere get unclipped reflectance.

Embedding: global average pool of the last Swin-B stage (channels 1024), from
the backbone alone (``fpn=False``), D=1024.
"""

from __future__ import annotations

import torch

from encoders.base import FrozenEncoder, resize_bilinear, rgb_from_s2

__all__ = ["SatlasS2SwinB"]


class SatlasS2SwinB(FrozenEncoder):
    name = "satlas_s2_swinb_rgb"
    embed_dim = 1024  # Swin-B last-stage channels
    model_identifier = "Sentinel2_SwinB_SI_RGB"
    # Swin-B downsamples by 32; H and W must be multiples of it. 128 already is.
    size_multiple = 32

    def _build(self):
        import satlaspretrain_models

        weights_manager = satlaspretrain_models.Weights()
        # device="cpu" makes the package torch.load with map_location="cpu";
        # its default ("cuda") crashes on CPU-only machines. The base class
        # moves the built model to self.device afterwards either way.
        return weights_manager.get_pretrained_model(
            self.model_identifier, fpn=False, device="cpu"
        )

    def _preprocessing_lines(self) -> list:
        return [
            "RGB from S2 bands: (B04, B03, B02) -> (R, G, B), indices resolved "
            "from S2_BANDS by name",
            "SatlasPretrain S2-RGB input convention: x = clamp(2.5 * reflectance, 0, 1) "
            "(approximately TCI / 255, the model's trained input distribution; this "
            "clamp exists ONLY inside this wrapper)",
            f"if H or W is not a multiple of {self.size_multiple}, antialiased bilinear "
            f"resize up to the nearest multiple (128 x 128 -> no-op)",
            "masked pixels NOT filled",
            f"embedding = global average pool of the last Swin-B backbone stage "
            f"(fpn=False), D={self.embed_dim}",
        ]

    def _encode_batch(self, frames: torch.Tensor, mask) -> torch.Tensor:
        x = rgb_from_s2(frames)
        x = torch.clamp(2.5 * x, 0.0, 1.0)
        H, W = x.shape[-2:]
        if H % self.size_multiple or W % self.size_multiple:
            side = max(
                self.size_multiple,
                self.size_multiple * round(max(H, W) / self.size_multiple),
            )
            x = resize_bilinear(x, side)
        feats = self._model(x.to(self.device))
        # fpn=False, no head: the satlaspretrain Model returns the backbone's
        # multi-scale feature list. Take the deepest map and pool it.
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        assert isinstance(feats, torch.Tensor) and feats.ndim == 4, (
            f"{self.name}: expected a (B, C, h, w) feature map from the backbone, "
            f"got {type(feats)}"
            + (f" with shape {tuple(feats.shape)}" if isinstance(feats, torch.Tensor) else "")
        )
        assert feats.shape[:2] == (frames.shape[0], self.embed_dim), (
            f"{self.name}: last-stage features {tuple(feats.shape)}, expected "
            f"({frames.shape[0]}, {self.embed_dim}, h, w) -- wrong backbone variant?"
        )
        z = feats.mean(dim=(2, 3))
        assert z.shape == (frames.shape[0], self.embed_dim)
        return z
