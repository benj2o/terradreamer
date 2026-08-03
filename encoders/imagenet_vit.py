"""Encoder 1 of 4: ImageNet ViT-B/16 (torchvision). The non-Earth floor.

A generalist trained on object-centric photographs, no satellite imagery at
all. If a probe on its embeddings matches the EO-native encoders, the probe is
reading generic image statistics, not Earth observation structure. That is
exactly what this row exists to detect.

Embedding: the pre-classifier class token, D=768 (``model.heads`` replaced by
``Identity``; no ImageNet logits are ever produced).
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

__all__ = ["ImageNetViTB16"]


class ImageNetViTB16(FrozenEncoder):
    name = "imagenet_vit_b16"
    embed_dim = 768
    input_size = 224  # torchvision's ViT-B/16 asserts 224x224 input

    def _build(self):
        from torchvision.models import ViT_B_16_Weights, vit_b_16

        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads = torch.nn.Identity()  # class token, pre-classifier
        return model

    def _preprocessing_lines(self) -> list:
        return [
            "RGB from S2 bands: (B04, B03, B02) -> (R, G, B), indices resolved "
            "from S2_BANDS by name",
            "non-finite (no-data) pixels -> NONFINITE_FILL, counted and reported; "
            "not inpainting, and done before the resize so a NaN cannot smear",
            f"antialiased bilinear resize H x W -> {self.input_size} x {self.input_size} "
            "(torchvision ViT-B/16 accepts exactly 224)",
            f"normalise with ImageNet mean={IMAGENET_MEAN} std={IMAGENET_STD}",
            "reflectance fed as-is: no TCI brightening, no clipping (clipping would "
            "hide the >1.2 valid-reflectance assertion's target), masked pixels NOT "
            "filled",
        ]

    def _encode_batch(self, frames: torch.Tensor, mask) -> torch.Tensor:
        x = rgb_from_s2(frames)
        x = self._sanitise(x)  # before resize: a NaN would smear over its neighbours
        x = resize_bilinear(x, self.input_size)
        mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        z = self._model(x.to(self.device))
        assert z.shape == (frames.shape[0], self.embed_dim), (
            f"{self.name}: got {tuple(z.shape)}, expected CLS embedding "
            f"({frames.shape[0]}, {self.embed_dim}) -- heads not replaced?"
        )
        return z
