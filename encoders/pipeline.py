"""Cube -> embeddings: frame selection, integrity assertions, encoding, storage.

The one path from a loaded cube to a saved embedding file. Order of operations
is fixed and load-bearing:

1. ``assert_valid_reflectance`` on the FULL cube -- the mask-leak check runs on
   everything, including the frames about to be dropped.
2. ``select_clear_frames`` -- drop frames with clear-fraction <= 0.5, keep the
   exact clear-fraction of every survivor.
3. Refuse an empty selection HERE, with the cube's name in the message. The
   encoder's own empty-batch assertion is the second line of defence, not the
   first.
4. ``encoder.encode`` -- frozen, batched, asserted.

What gets stored alongside every embedding matrix: the survivors' timestamps,
their exact clear-fractions (so later probes can filter more strictly WITHOUT
re-encoding), and their indices into the original cube time axis.
"""

from __future__ import annotations

import os
from typing import NamedTuple

import numpy as np
import torch

from data.loader import CubeSample

from encoders.base import FrozenEncoder
from encoders.frames import (
    MIN_CLEAR_FRACTION,
    assert_valid_reflectance,
    select_clear_frames,
)

__all__ = ["EncodedCube", "encode_cube", "save_encoded", "load_encoded"]


class EncodedCube(NamedTuple):
    """One cube through one encoder, with everything a probe needs to filter."""

    embeddings: np.ndarray  # (T_kept, D) float32
    timestamps: np.ndarray  # (T_kept,) datetime64[ns], strictly increasing
    clear_frac: np.ndarray  # (T_kept,) float64, exact clear-fraction per frame
    kept_idx: np.ndarray    # (T_kept,) int, index into the ORIGINAL cube time axis
    encoder: str            # FrozenEncoder.name
    cube: str               # basename of the source cube file


def encode_cube(
    sample: CubeSample,
    encoder: FrozenEncoder,
    min_clear: float = MIN_CLEAR_FRACTION,
    batch_size: int = 16,
    verbose: bool = True,
) -> EncodedCube:
    """(values, timestamps, mask) -> [T_kept, D] embeddings plus bookkeeping."""
    cube_name = os.path.basename(sample.path)

    assert_valid_reflectance(sample.values, sample.mask, verbose=verbose)
    sel = select_clear_frames(
        sample.values, sample.timestamps, sample.mask, min_clear=min_clear, verbose=verbose
    )
    assert sel.values.shape[0] > 0, (
        f"{cube_name}: no frame has clear-fraction > {min_clear}; the cube cannot "
        "be encoded and must be skipped by the caller. Nothing downstream may "
        "receive an empty batch."
    )

    frames = torch.from_numpy(np.ascontiguousarray(sel.values))
    mask = torch.from_numpy(sel.mask) if encoder.requires_mask else None
    emb = encoder.encode(frames, mask=mask, batch_size=batch_size, verbose=verbose).numpy()

    T_kept = sel.values.shape[0]
    assert emb.shape == (T_kept, encoder.embed_dim), (
        f"{encoder.name} on {cube_name}: embeddings {emb.shape} != "
        f"({T_kept}, {encoder.embed_dim})"
    )
    assert emb.dtype == np.float32
    assert np.isfinite(emb).all()

    out = EncodedCube(
        embeddings=emb,
        timestamps=sel.timestamps,
        clear_frac=sel.clear_frac,
        kept_idx=sel.kept_idx,
        encoder=encoder.name,
        cube=cube_name,
    )
    if verbose:
        print(f"[pipeline] {cube_name} x {encoder.name}: embeddings {emb.shape} | "
              f"clear_frac {out.clear_frac.shape} | kept_idx {out.kept_idx.shape}")
    return out


def _npz_path(out_dir: str, cube: str, encoder: str) -> str:
    stem = os.path.splitext(cube)[0]
    return os.path.join(out_dir, f"{stem}__{encoder}.npz")


def save_encoded(out_dir: str, ec: EncodedCube, verbose: bool = True) -> str:
    """Write one EncodedCube as a compressed .npz; returns the path."""
    os.makedirs(out_dir, exist_ok=True)
    path = _npz_path(out_dir, ec.cube, ec.encoder)
    np.savez_compressed(
        path,
        embeddings=ec.embeddings,
        timestamps=ec.timestamps.astype("datetime64[ns]"),
        clear_frac=ec.clear_frac,
        kept_idx=ec.kept_idx,
        encoder=np.array(ec.encoder),
        cube=np.array(ec.cube),
    )
    if verbose:
        print(f"[pipeline] saved {path} ({os.path.getsize(path) / 1e3:.0f} kB)")
    return path


def load_encoded(path: str) -> EncodedCube:
    """Read an EncodedCube back, re-asserting every invariant that was saved."""
    with np.load(path) as z:
        ec = EncodedCube(
            embeddings=z["embeddings"],
            timestamps=z["timestamps"].astype("datetime64[ns]"),
            clear_frac=z["clear_frac"],
            kept_idx=z["kept_idx"],
            encoder=str(z["encoder"]),
            cube=str(z["cube"]),
        )
    T_kept, _D = ec.embeddings.shape
    assert ec.timestamps.shape == ec.clear_frac.shape == ec.kept_idx.shape == (T_kept,)
    assert ec.embeddings.dtype == np.float32
    assert np.isfinite(ec.embeddings).all()
    assert (ec.clear_frac > 0).all() and (ec.clear_frac <= 1).all()
    assert np.all(np.diff(ec.timestamps) > np.timedelta64(0, "ns"))
    return ec
