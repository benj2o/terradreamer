"""The uniform Tier A wrapper interface:

    encode(frames: Tensor[T, C, H, W]) -> Tensor[T, D]

Rules baked in here, once, so no wrapper can forget them:

* FROZEN, ALWAYS. Any wrapped network is put in ``.eval()`` at construction,
  every parameter gets ``requires_grad_(False)``, and the forward pass runs
  under ``torch.no_grad()``. ``encode`` re-asserts all of this on every call,
  so a stray ``.train()`` between calls fails loudly instead of silently
  updating BatchNorm statistics.
* BATCHED OVER TIME. Frames go through the model in chunks of ``batch_size``,
  so peak memory is set by the batch, not by T. T=290 (the seasonal split)
  must not OOM where T=29 fits.
* LOUD FAILURES. Wrong rank, wrong channel count, channels-last layouts,
  non-finite pixels and empty batches are all asserted with messages that say
  what went wrong, before any tensor reaches a model that might broadcast it
  into a silently wrong answer.
* PER-MODEL PREPROCESSING IS PRINTED. Each wrapper implements
  ``_preprocessing_lines`` describing exactly what it does to a frame (band
  selection, resize, normalisation), and the base class prints it at
  construction. Nothing radiometric happens implicitly in shared code.

The raw-feature baseline is deliberately forced through this same interface:
it is "encoder number four", not a special case, so every later probe treats
it identically. It is the only wrapper with ``requires_mask = True`` -- the
canonical ``data.ndvi.ndvi`` requires the mask, and the mask travels as an
optional second argument that the network wrappers never touch (their frames
are fed unmodified, clouds included).
"""

from __future__ import annotations

import abc

import numpy as np
import torch

from data.loader import S2_BANDS

__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "FrozenEncoder",
    "rgb_from_s2",
    "resize_bilinear",
]

# Shared by the two ImageNet-normalised wrappers (torchvision ViT, DINOv2).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def rgb_from_s2(frames: torch.Tensor) -> torch.Tensor:
    """(T, 4, H, W) in S2_BANDS order -> (T, 3, H, W) RGB = (B04, B03, B02).

    Indices come from S2_BANDS by name, never hardcoded, so a band-order
    change upstream breaks here loudly instead of feeding blue as red.
    """
    idx = [S2_BANDS.index(b) for b in ("B04", "B03", "B02")]
    assert frames.shape[1] == len(S2_BANDS), (
        f"expected {len(S2_BANDS)} channels in {S2_BANDS} order, got {frames.shape}"
    )
    out = frames[:, idx]
    assert out.shape == (frames.shape[0], 3, frames.shape[2], frames.shape[3])
    return out


def resize_bilinear(x: torch.Tensor, size: int) -> torch.Tensor:
    """Antialiased bilinear resize of (B, C, H, W) to (B, C, size, size)."""
    assert x.ndim == 4, f"expected (B, C, H, W), got {tuple(x.shape)}"
    if x.shape[-2:] == (size, size):
        return x
    out = torch.nn.functional.interpolate(
        x, size=(size, size), mode="bilinear", align_corners=False, antialias=True
    )
    assert out.shape == (x.shape[0], x.shape[1], size, size)
    return out


class FrozenEncoder(abc.ABC):
    """One frame in, one fixed-length embedding out. Frozen, batched, asserted.

    Subclasses set ``name`` and ``embed_dim``, implement ``_build`` (return a
    torch module, or None for the non-network baseline), ``_encode_batch`` and
    ``_preprocessing_lines``. Everything else -- freezing, batching, shape and
    finiteness assertions, the empty-batch guard -- lives here.
    """

    name: str = "?"
    embed_dim: int = -1
    # Only the raw-feature baseline sets this: data.ndvi.ndvi requires the
    # cloud mask. Network wrappers get frames UNMODIFIED and never see it.
    requires_mask: bool = False

    def __init__(self, device: str | None = None, verbose: bool = True):
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        model = self._build()
        if model is not None:
            model = model.to(self.device).eval()
            for p in model.parameters():
                p.requires_grad_(False)
            n_params = sum(p.numel() for p in model.parameters())
            if verbose:
                print(f"[{self.name}] frozen: eval()=True, requires_grad=False on all "
                      f"{n_params / 1e6:.1f}M params, device={self.device}")
        else:
            if verbose:
                print(f"[{self.name}] not a network: no parameters, nothing to freeze")
        self._model = model
        self._assert_frozen()
        if verbose:
            print(f"[{self.name}] D={self.embed_dim}")
            print(f"[{self.name}] preprocessing (inside this wrapper, in this order; "
                  "nothing else is applied):")
            for i, line in enumerate(self._preprocessing_lines(), 1):
                print(f"[{self.name}]   {i}. {line}")

    # ------------------------------------------------------------------ hooks
    @abc.abstractmethod
    def _build(self):
        """Return the torch module to wrap, or None for a non-network encoder."""

    @abc.abstractmethod
    def _encode_batch(self, frames: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """(B, C, H, W) [+ optional (B, H, W) mask] -> (B, embed_dim)."""

    @abc.abstractmethod
    def _preprocessing_lines(self) -> list:
        """Human-readable, exhaustive description of this wrapper's preprocessing."""

    # ----------------------------------------------------------------- guards
    def _assert_frozen(self) -> None:
        if self._model is None:
            return
        assert not self._model.training, (
            f"{self.name}: model is in train() mode. Frozen means eval(), always; "
            "something called .train() after construction."
        )
        thawed = [n for n, p in self._model.named_parameters() if p.requires_grad]
        assert not thawed, (
            f"{self.name}: {len(thawed)} parameter(s) have requires_grad=True "
            f"(first: {thawed[0]}). No pretrained model is ever fine-tuned."
        )

    def _check_frames(self, frames) -> torch.Tensor:
        if isinstance(frames, np.ndarray):
            frames = torch.from_numpy(frames)
        assert isinstance(frames, torch.Tensor), (
            f"{self.name}: frames must be a torch.Tensor or np.ndarray, got {type(frames)}"
        )
        assert frames.ndim == 4, (
            f"{self.name}: frames must be (T, C, H, W), got rank-{frames.ndim} shape "
            f"{tuple(frames.shape)}. No implicit unsqueeze: a single frame must be "
            "passed as (1, C, H, W)."
        )
        assert frames.shape[1] == len(S2_BANDS), (
            f"{self.name}: expected C={len(S2_BANDS)} at axis 1 in {S2_BANDS} order, got "
            f"shape {tuple(frames.shape)}. A channels-last (T, H, W, C) layout would "
            "put H here and be silently broadcast into nonsense -- refuse it."
        )
        assert frames.shape[0] > 0, (
            f"{self.name}: EMPTY BATCH -- 0 frames reached the encoder. Frames failing "
            "the clear-fraction rule are dropped upstream "
            "(encoders.frames.select_clear_frames); a cube whose every frame is "
            "cloudy must be skipped by the caller, never handed to a model."
        )
        assert torch.is_floating_point(frames), (
            f"{self.name}: frames must be float reflectance, got dtype {frames.dtype}"
        )
        frames = frames.to(torch.float32)
        assert torch.isfinite(frames).all(), (
            f"{self.name}: non-finite reflectance in the input frames. A NaN pixel "
            "would spread through attention/pooling and poison the whole embedding "
            "silently. Frames are fed UNMODIFIED by design, so do not fill it here: "
            "the frame must be excluded upstream, and its presence after "
            "clear-fraction selection means the loader let a NaN through a "
            "'valid' timestep -- investigate before encoding."
        )
        return frames

    def _check_mask(self, mask, frames: torch.Tensor) -> torch.Tensor | None:
        if self.requires_mask:
            assert mask is not None, (
                f"{self.name}: this encoder computes NDVI statistics via the canonical "
                "data.ndvi.ndvi, which REQUIRES the cloud mask. Pass mask=(T, H, W) bool."
            )
        if mask is None:
            return None
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        assert isinstance(mask, torch.Tensor)
        assert mask.dtype == torch.bool, (
            f"{self.name}: mask must be bool with True == VALID, got {mask.dtype}"
        )
        T, _, H, W = frames.shape
        assert mask.shape == (T, H, W), (
            f"{self.name}: mask {tuple(mask.shape)} incompatible with frames "
            f"{tuple(frames.shape)}"
        )
        return mask

    # ------------------------------------------------------------------- API
    def encode(
        self,
        frames,
        mask=None,
        batch_size: int = 16,
        verbose: bool = True,
    ) -> torch.Tensor:
        """Tensor[T, C, H, W] -> Tensor[T, D] on CPU, float32.

        Batched over the time axis: peak memory scales with ``batch_size``,
        never with T. All shape/finiteness guards run on every call.
        """
        frames = self._check_frames(frames)
        mask = self._check_mask(mask, frames)
        self._assert_frozen()
        assert batch_size >= 1, f"batch_size must be >= 1, got {batch_size}"

        T = frames.shape[0]
        chunks = []
        with torch.no_grad():
            for i in range(0, T, batch_size):
                fb = frames[i:i + batch_size]
                mb = mask[i:i + batch_size] if mask is not None else None
                z = self._encode_batch(fb, mb)
                assert isinstance(z, torch.Tensor)
                assert z.shape == (fb.shape[0], self.embed_dim), (
                    f"{self.name}: batch embedding {tuple(z.shape)} != "
                    f"({fb.shape[0]}, {self.embed_dim})"
                )
                assert torch.isfinite(z).all(), (
                    f"{self.name}: non-finite values in the embedding for frames "
                    f"[{i}:{i + fb.shape[0]}]"
                )
                chunks.append(z.detach().to("cpu", torch.float32))

        out = torch.cat(chunks, dim=0)
        assert out.shape == (T, self.embed_dim), (
            f"{self.name}: output {tuple(out.shape)} != ({T}, {self.embed_dim})"
        )
        assert not out.requires_grad
        if verbose:
            print(f"[{self.name}] encode: frames {tuple(frames.shape)} -> "
                  f"embeddings {tuple(out.shape)}  (D={self.embed_dim}, "
                  f"batch_size={batch_size}, {len(chunks)} batch(es))")
        return out
