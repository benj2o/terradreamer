"""Tier A frozen encoders: one satellite frame in, one fixed-length embedding out.

Uniform interface (encoders.base.FrozenEncoder):

    encode(frames: Tensor[T, C, H, W]) -> Tensor[T, D]

frozen (`.eval()` + `torch.no_grad()`, re-asserted on every call), batched over
the time axis so memory never scales with T, every shape printed and asserted.
Phase 1.2 proves each encoder loads, runs frozen, and emits an assertable
shape. NO quality comparison happens here; that is Phase 1.3, through
probes/cv.py, and any number produced outside probes/cv.py does not exist.

The four rows, in TIER_A order:

    raw_features        D=35    NOT a network. Per-band + NDVI summary stats.
                                The row that gives every later probe its meaning.
    imagenet_vit_b16    D=768   torchvision ViT-B/16, the non-Earth floor.
    dinov2_vitb14       D=768   torch.hub DINOv2 ViT-B/14, the generalist.
    satlas_s2_swinb_rgb D=1024  SatlasPretrain Sentinel-2 Swin-B, EO-native.

Imports of the heavyweight wrappers are lazy: ``build_encoder`` touches
torchvision / torch.hub / satlaspretrain_models only when that encoder is
actually requested, so importing this package stays cheap and offline-safe.
"""

from __future__ import annotations

from encoders.base import FrozenEncoder
from encoders.frames import (
    MIN_CLEAR_FRACTION,
    assert_valid_reflectance,
    clear_fraction,
    select_clear_frames,
)
from encoders.pipeline import EncodedCube, encode_cube, load_encoded, save_encoded

__all__ = [
    "TIER_A",
    "FrozenEncoder",
    "build_encoder",
    "build_tier_a",
    "MIN_CLEAR_FRACTION",
    "assert_valid_reflectance",
    "clear_fraction",
    "select_clear_frames",
    "EncodedCube",
    "encode_cube",
    "save_encoded",
    "load_encoded",
]

TIER_A = ("raw_features", "imagenet_vit_b16", "dinov2_vitb14",
          "satlas_s2_swinb_rgb", "satlas_s2_swinb_mi_rgb")


def build_encoder(name: str, device: str | None = None, verbose: bool = True) -> FrozenEncoder:
    """Instantiate one Tier A encoder by name (downloads weights on first use)."""
    if name == "raw_features":
        from encoders.raw_features import RawFeatureBaseline as cls
    elif name == "imagenet_vit_b16":
        from encoders.imagenet_vit import ImageNetViTB16 as cls
    elif name == "dinov2_vitb14":
        from encoders.dinov2_vit import DINOv2ViTB14 as cls
    elif name == "satlas_s2_swinb_rgb":
        from encoders.satlas_s2 import SatlasS2SwinB as cls
    elif name == "satlas_s2_swinb_mi_rgb":
        from encoders.satlas_s2_mi import SatlasS2SwinBMI as cls
    else:
        raise KeyError(f"unknown encoder {name!r}; Tier A is {TIER_A}")
    enc = cls(device=device, verbose=verbose)
    assert enc.name == name, f"registry name {name!r} != wrapper name {enc.name!r}"
    return enc


def build_tier_a(device: str | None = None, verbose: bool = True) -> dict:
    """All four Tier A encoders, keyed by name, in TIER_A order."""
    return {name: build_encoder(name, device=device, verbose=verbose) for name in TIER_A}
