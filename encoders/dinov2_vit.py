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
    composite_from_s2,
    tokens_to_grid,
)

__all__ = ["DINOv2ViTB14"]


class DINOv2ViTB14(FrozenEncoder):
    """DINOv2 ViT-B/14, extracted by its OWN published linear-probe protocol.

    Oquab et al., "DINOv2: Learning Robust Visual Features without
    Supervision" (TMLR 2023), linear-evaluation protocol: concatenate the
    class token of the last FOUR blocks with the average-pooled patch tokens
    of the last block. Final-CLS-only is reported there as underperforming
    this, so probing final-CLS-only and then concluding that DINOv2 is lossy
    for dynamics would measure the extraction, not the representation. That
    is the single most attackable choice in a representation audit, so the
    published recipe is the default here and the weaker variants are exposed
    beside it for the ablation.
    """

    name = "dinov2_vitb14"
    embed_dim = 3840           # 4 x 768 CLS + 768 patch-mean
    grid_dim = 768
    variant_dims = {"cls_last": 768, "cls_last4_concat": 3072, "patch_mean_last": 768}
    FEATURE_RECIPE = (
        "DINOv2 published linear-probe protocol (Oquab et al., TMLR 2023): "
        "pooled = concat(cls_last4_concat, patch_mean_last) = 3840; "
        "grid = 16x16 last-block patch tokens adaptive-avg-pooled to 4x4"
    )
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
            "non-finite (no-data) pixels -> NONFINITE_FILL, counted and reported; "
            "not inpainting, and done before the resize so a NaN cannot smear",
            f"antialiased bilinear resize H x W -> {self.input_size} x {self.input_size} "
            f"({self.input_size} = {self.input_size // 14} x 14, the ViT-B/14 patch size)",
            f"normalise with ImageNet mean={IMAGENET_MEAN} std={IMAGENET_STD} "
            "(DINOv2's own transform)",
            f"extraction: {self.FEATURE_RECIPE}",
            "reflectance fed as-is: no TCI brightening, no clipping, masked pixels "
            "NOT filled",
        ]

    def _features_batch(self, frames: torch.Tensor, mask) -> dict:
        x = composite_from_s2(frames, self.composite)
        x = self._sanitise(x)  # before resize: a NaN would smear over its neighbours
        x = resize_bilinear(x, self.input_size)
        mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
        x = ((x - mean) / std).to(self.device)

        # n=4 -> the last four blocks, each as (patch_tokens, class_token).
        layers = self._model.get_intermediate_layers(x, n=4, return_class_token=True)
        assert len(layers) == 4, f"{self.name}: expected 4 blocks, got {len(layers)}"
        patches_last, cls_last = layers[-1]
        cls_last4 = torch.cat([c for _p, c in layers], dim=1)     # (B, 3072)
        patch_mean_last = patches_last.mean(dim=1)                # (B, 768)
        return {
            "cls_last": cls_last,
            "cls_last4_concat": cls_last4,
            "patch_mean_last": patch_mean_last,
            "pooled": torch.cat([cls_last4, patch_mean_last], dim=1),
            "grid": tokens_to_grid(patches_last),
        }
