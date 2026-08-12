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
    composite_from_s2,
    tokens_to_grid,
)

__all__ = ["ImageNetViTB16"]


class ImageNetViTB16(FrozenEncoder):
    name = "imagenet_vit_b16"
    # Probe default is concat(CLS, mean patch token) of the last block, the
    # usual ViT linear-probe feature: CLS alone discards all spatial evidence,
    # and on a 128 px agricultural scene there is no object for CLS to centre
    # on. Both halves are also exposed separately so the ablation is free.
    embed_dim = 1536           # 768 CLS + 768 patch-mean
    grid_dim = 768
    variant_dims = {"cls_last": 768, "patch_mean_last": 768}
    FEATURE_RECIPE = (
        "last block, post-LayerNorm; pooled = concat(cls_last, patch_mean_last) "
        "= 1536; grid = 14x14 patch tokens adaptive-avg-pooled to 4x4"
    )
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
            f"extraction: {self.FEATURE_RECIPE}",
            "reflectance fed as-is: no TCI brightening, no clipping (clipping would "
            "hide the >1.2 valid-reflectance assertion's target), masked pixels NOT "
            "filled",
        ]

    def _tokens(self, frames: torch.Tensor) -> torch.Tensor:
        """All transformer tokens after the final encoder LayerNorm: (B, 1+N, 768).

        torchvision's ViT exposes no token-level forward, so the standard
        pre-head path is reproduced explicitly: patchify, prepend the class
        token, run the encoder (which adds the positional embedding and
        applies the final LayerNorm).
        """
        x = composite_from_s2(frames, self.composite)
        x = self._sanitise(x)  # before resize: a NaN would smear over its neighbours
        x = resize_bilinear(x, self.input_size)
        mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
        x = ((x - mean) / std).to(self.device)

        m = self._model
        t = m._process_input(x)                                   # (B, N, 768)
        cls = m.class_token.expand(t.shape[0], -1, -1)            # (B, 1, 768)
        t = m.encoder(torch.cat([cls, t], dim=1))                 # (B, 1+N, 768)
        assert t.ndim == 3 and t.shape[-1] == 768, tuple(t.shape)
        return t

    def _features_batch(self, frames: torch.Tensor, mask) -> dict:
        t = self._tokens(frames)
        cls_last = t[:, 0]                                        # (B, 768)
        patches = t[:, 1:]                                        # (B, N, 768)
        patch_mean_last = patches.mean(dim=1)                     # (B, 768)
        return {
            "cls_last": cls_last,
            "patch_mean_last": patch_mean_last,
            "pooled": torch.cat([cls_last, patch_mean_last], dim=1),
            "grid": tokens_to_grid(patches),
        }
