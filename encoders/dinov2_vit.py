"""Encoder 2 of 4: DINOv2 ViT-B/14 (torch.hub). The generalist.

Self-supervised on a broad web-image corpus (LVD-142M); no labels, no
satellite specialisation. Sits between the ImageNet floor and the EO-native
encoder: strong generic visual features, zero Earth-observation prior.

Embedding: the model's forward output, the normalised class token, D=768.
Input side must be a multiple of the 14-px patch: 224 = 16 x 14.
"""

from __future__ import annotations

import torch

from encoders.base import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    FrozenEncoder,
    resize_bilinear,
    rgb_from_s2,
)

__all__ = ["DINOv2ViTB14"]


class DINOv2ViTB14(FrozenEncoder):
    name = "dinov2_vitb14"
    embed_dim = 768
    input_size = 224  # multiple of the 14-px patch: 224 = 16 * 14

    def _build(self):
        assert self.input_size % 14 == 0, (
            f"DINOv2 needs input a multiple of 14, got {self.input_size}"
        )
        # torch.hub fetches facebookresearch/dinov2 (code + weights) on first
        # use; trust_repo=True keeps it non-interactive on fresh runtimes.
        return torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14",
                              trust_repo=True)

    def _preprocessing_lines(self) -> list:
        return [
            "RGB from S2 bands: (B04, B03, B02) -> (R, G, B), indices resolved "
            "from S2_BANDS by name",
            f"antialiased bilinear resize H x W -> {self.input_size} x {self.input_size} "
            f"({self.input_size} = {self.input_size // 14} x 14, the ViT-B/14 patch size)",
            f"normalise with ImageNet mean={IMAGENET_MEAN} std={IMAGENET_STD} "
            "(DINOv2's own transform)",
            "reflectance fed as-is: no TCI brightening, no clipping, masked pixels "
            "NOT filled",
        ]

    def _encode_batch(self, frames: torch.Tensor, mask) -> torch.Tensor:
        x = rgb_from_s2(frames)
        x = resize_bilinear(x, self.input_size)
        mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        z = self._model(x.to(self.device))
        assert z.shape == (frames.shape[0], self.embed_dim), (
            f"{self.name}: got {tuple(z.shape)}, expected CLS embedding "
            f"({frames.shape[0]}, {self.embed_dim})"
        )
        return z
