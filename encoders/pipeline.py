"""Cube -> embeddings: frame selection, integrity assertions, encoding, storage.

The one path from a loaded cube to a saved embedding file. Order of operations
is fixed and load-bearing:

1. ``assert_valid_reflectance`` on the FULL cube -- the mask-leak check runs on
   everything, including the frames about to be dropped.
2. ``select_clear_frames`` -- make the mask self-consistent with the
   reflectance, then drop frames with clear-fraction <= 0.5, keeping the exact
   clear-fraction of every survivor.
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

from encoders.base import GRID_CELLS, FrozenEncoder
from encoders.frames import (
    MIN_CLEAR_FRACTION,
    assert_valid_reflectance,
    grid_clear_fraction,
    select_clear_frames,
)

__all__ = ["EncodedCube", "CubeMasks", "SCHEMA_VERSION", "encode_cube",
           "save_encoded", "load_encoded", "cube_masks", "save_masks", "load_masks"]

# Bump whenever a stored field is ADDED, REMOVED or changes meaning.
#   1  Phase 1.2   pooled + timestamps + clear_frac + kept_idx
#   2  Phase 1.2b  + grid (fp16), grid_clear_frac, variants
#   3  Phase 1.2c  + window_span_days
#
# Why this exists. Step 11 is resumable, and np.load on an older file simply
# lacks the newer keys -- load_encoded would return None for them and continue
# silently, so a probe would read window_span_days, find nothing, and quietly
# drop the covariate. Encoder D values do NOT settle it either: the multi-image
# encoder and window_span_days landed in DIFFERENT commits, so a cache can
# contain MI files at the right dimensionality and still predate the covariate.
# A stamped version is the only thing that distinguishes "cached" from "stale".
SCHEMA_VERSION = 3


class EncodedCube(NamedTuple):
    """One cube through one encoder, with everything a probe needs to filter."""

    embeddings: np.ndarray  # (T_kept, D) float32   -- the pooled probe default
    timestamps: np.ndarray  # (T_kept,) datetime64[ns], strictly increasing
    clear_frac: np.ndarray  # (T_kept,) float64, exact clear-fraction per frame
    kept_idx: np.ndarray    # (T_kept,) int, index into the ORIGINAL cube time axis
    encoder: str            # FrozenEncoder.name
    cube: str               # basename of the source cube file
    # --- Phase 1.2b additions -------------------------------------------
    grid: np.ndarray = None        # (T_kept, 16, D_grid) FLOAT16 on disk
    grid_clear_frac: np.ndarray = None  # (T_kept, 16) float32, per-cell valid fraction
    variants: dict = {}            # {name: (T_kept, dim) float32} extraction ablation
    window_span_days: np.ndarray = None  # (T_kept,) float32, see window_span_days()


class CubeMasks(NamedTuple):
    """Per-cube per-pixel valid mask, cached ONCE per cube, not per encoder.

    Cube-mean NDVI at t averages the pixels valid at t; at t+delta it averages
    a DIFFERENT pixel set, so a measured "NDVI change" partly measures which
    pixels happened to be visible. Clear-fraction on this tile swings roughly
    0.44-0.63, so the confound is live. The fix -- common-masking, restricting
    compared frames to pixels valid in ALL of them -- is probe-side logic and
    is NOT implemented here. This exists only so it remains POSSIBLE: without
    the per-pixel mask it cannot be done at all after the fact.
    """

    mask: np.ndarray        # (T_kept, H, W) bool, True == valid AND finite
    kept_idx: np.ndarray    # (T_kept,) int, index into the ORIGINAL time axis
    timestamps: np.ndarray  # (T_kept,) datetime64[ns]
    cube: str


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
    # Every encoder gets the mask now: the baseline needs it for NDVI, and the
    # networks ignore it (their frames are still fed unmodified).
    mask = torch.from_numpy(sel.mask)
    bundle = encoder.encode_bundle(frames, mask=mask, batch_size=batch_size,
                                   verbose=verbose)

    T_kept = sel.values.shape[0]
    emb = bundle["pooled"].numpy()
    # float16 for the grid: it is ~16x the pooled size, and probe inputs are
    # standardised anyway, so fp16 precision is ample. Pooled stays float32.
    grid = bundle["grid"].numpy().astype(np.float16)
    gcf = grid_clear_fraction(sel.mask).astype(np.float32)
    wsd = window_span_days(sel.timestamps, encoder.window_len)
    variants = {k: v.numpy() for k, v in bundle.items() if k not in ("pooled", "grid")}

    assert emb.shape == (T_kept, encoder.embed_dim), (
        f"{encoder.name} on {cube_name}: pooled {emb.shape} != "
        f"({T_kept}, {encoder.embed_dim})"
    )
    assert grid.shape == (T_kept, GRID_CELLS, encoder.grid_dim), (
        f"{encoder.name} on {cube_name}: grid {grid.shape} != "
        f"({T_kept}, {GRID_CELLS}, {encoder.grid_dim})"
    )
    assert gcf.shape == (T_kept, GRID_CELLS)
    assert emb.dtype == np.float32 and np.isfinite(emb).all()
    assert np.isfinite(grid).all(), (
        f"{encoder.name} on {cube_name}: float16 downcast overflowed to inf; "
        "the grid features exceed the fp16 range (~65504)"
    )
    # The per-cell fractions must average back to the frame clear-fraction.
    np.testing.assert_allclose(gcf.mean(axis=1), sel.clear_frac, rtol=0, atol=1e-6)

    out = EncodedCube(
        embeddings=emb,
        timestamps=sel.timestamps,
        clear_frac=sel.clear_frac,
        kept_idx=sel.kept_idx,
        encoder=encoder.name,
        cube=cube_name,
        grid=grid,
        grid_clear_frac=gcf,
        variants=variants,
        window_span_days=wsd,
    )
    if verbose:
        print(f"[pipeline] {cube_name} x {encoder.name}: pooled {emb.shape} | "
              f"grid {grid.shape} fp16 | grid_clear_frac {gcf.shape} | "
              f"variants {sorted(variants)}")
        if encoder.window_len > 1:
            print(f"[pipeline] {cube_name} x {encoder.name}: window_span_days "
                  f"(lookback of {encoder.window_len} retained frames) "
                  f"min={wsd.min():.0f} median={np.median(wsd):.0f} max={wsd.max():.0f} "
                  "-- weather-correlated, pass it as a covariate")
    return out


def cube_masks(sample: CubeSample, min_clear=MIN_CLEAR_FRACTION,
               verbose: bool = True) -> CubeMasks:
    """The per-pixel valid mask for a cube's retained frames. See CubeMasks."""
    sel = select_clear_frames(sample.values, sample.timestamps, sample.mask,
                              min_clear=min_clear, verbose=False)
    cm = CubeMasks(mask=sel.mask, kept_idx=sel.kept_idx, timestamps=sel.timestamps,
                   cube=os.path.basename(sample.path))
    if verbose:
        print(f"[pipeline] {cm.cube}: mask {cm.mask.shape} bool")
    return cm


def save_masks(out_dir: str, cm: CubeMasks, verbose: bool = True) -> str:
    """Write one cube's per-pixel masks. Compresses hard: clear-fraction is
    bimodal, so frames are near-fully clear or near-fully clouded."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{os.path.splitext(cm.cube)[0]}__masks.npz")
    np.savez_compressed(path, mask=cm.mask, kept_idx=cm.kept_idx,
                        timestamps=cm.timestamps.astype("datetime64[ns]"),
                        cube=np.array(cm.cube))
    if verbose:
        raw = cm.mask.size / 8 / 1e3
        print(f"[pipeline] saved {os.path.basename(path)} "
              f"({os.path.getsize(path) / 1e3:.0f} kB on disk, "
              f"{raw:.0f} kB as packed bits)")
    return path


def load_masks(path: str) -> CubeMasks:
    with np.load(path) as z:
        cm = CubeMasks(mask=z["mask"], kept_idx=z["kept_idx"],
                       timestamps=z["timestamps"].astype("datetime64[ns]"),
                       cube=str(z["cube"]))
    assert cm.mask.dtype == np.bool_ and cm.mask.ndim == 3
    assert cm.kept_idx.shape == cm.timestamps.shape == (cm.mask.shape[0],)
    return cm


def window_span_days(timestamps: np.ndarray, window_len: int) -> np.ndarray:
    """(T_kept,) calendar days each embedding's input window actually spans.

    A multi-image encoder consumes ``window_len`` RETAINED frames, and retained
    frames are irregularly spaced, so the embedding's effective lookback is a
    variable number of days -- and the variation is WEATHER-CORRELATED, because
    a cloudier stretch drops more frames and therefore reaches further back in
    time for the same 8 frames. That is the same confound as the horizon issue,
    living inside the encoder rather than beside it.

    A docstring warning is not enough: a probe cannot control for a quantity
    that was never cached. This is computable only at encode time, so it is
    cached here for probes to pass as a covariate.

    Single-image encoders (``window_len == 1``) get exactly 0.0 -- their
    lookback is one frame by construction, so the covariate is constant and
    carries no information, which is the honest value rather than a NaN.
    """
    timestamps = np.asarray(timestamps)
    assert window_len >= 1, f"window_len must be >= 1, got {window_len}"
    T = timestamps.shape[0]
    idx = np.arange(T)
    first = np.maximum(0, idx - (window_len - 1))   # earliest frame in the window
    days = (timestamps - timestamps[first]) / np.timedelta64(1, "D")
    out = np.asarray(days, dtype=np.float32)
    assert out.shape == (T,)
    assert np.isfinite(out).all() and (out >= 0).all(), "negative or non-finite span"
    if window_len == 1:
        assert (out == 0).all()
    return out


def _npz_path(out_dir: str, cube: str, encoder: str) -> str:
    stem = os.path.splitext(cube)[0]
    return os.path.join(out_dir, f"{stem}__{encoder}.npz")


def save_encoded(out_dir: str, ec: EncodedCube, verbose: bool = True) -> str:
    """Write one EncodedCube as a compressed .npz; returns the path."""
    os.makedirs(out_dir, exist_ok=True)
    path = _npz_path(out_dir, ec.cube, ec.encoder)
    payload = dict(
        schema_version=np.array(SCHEMA_VERSION),
        embeddings=ec.embeddings,
        timestamps=ec.timestamps.astype("datetime64[ns]"),
        clear_frac=ec.clear_frac,
        kept_idx=ec.kept_idx,
        encoder=np.array(ec.encoder),
        cube=np.array(ec.cube),
    )
    if ec.grid is not None:
        payload["grid"] = ec.grid
    if ec.grid_clear_frac is not None:
        payload["grid_clear_frac"] = ec.grid_clear_frac
    if ec.window_span_days is not None:
        payload["window_span_days"] = ec.window_span_days
    for k, v in (ec.variants or {}).items():
        payload[f"variant__{k}"] = v
    np.savez_compressed(path, **payload)
    if verbose:
        print(f"[pipeline] saved {path} ({os.path.getsize(path) / 1e3:.0f} kB)")
    return path


def load_encoded(path: str) -> EncodedCube:
    """Read an EncodedCube back, re-asserting every invariant that was saved."""
    with np.load(path) as z:
        found = int(z["schema_version"]) if "schema_version" in z else 0
        assert found == SCHEMA_VERSION, (
            f"{os.path.basename(path)} was written with cache schema v{found}, "
            f"but this code expects v{SCHEMA_VERSION}. A cached file from an older "
            "schema is missing fields that probes read silently as absent -- "
            "window_span_days is the live example. Delete this phase's artefacts "
            "and re-encode:\n"
            "    from data.paths import reset_phase; reset_phase('phase1_2')\n"
            "Encoder dimensionality does NOT prove a cache is current: the "
            "multi-image encoder and window_span_days landed in different commits."
        )
        ec = EncodedCube(
            embeddings=z["embeddings"],
            timestamps=z["timestamps"].astype("datetime64[ns]"),
            clear_frac=z["clear_frac"],
            kept_idx=z["kept_idx"],
            encoder=str(z["encoder"]),
            cube=str(z["cube"]),
            grid=z["grid"] if "grid" in z else None,
            grid_clear_frac=z["grid_clear_frac"] if "grid_clear_frac" in z else None,
            variants={k[len("variant__"):]: z[k] for k in z.files
                      if k.startswith("variant__")},
            window_span_days=z["window_span_days"] if "window_span_days" in z else None,
        )
    T_kept, _D = ec.embeddings.shape
    if ec.grid is not None:
        assert ec.grid.shape[:2] == (T_kept, GRID_CELLS), ec.grid.shape
        assert ec.grid.dtype == np.float16, f"grid must be fp16, got {ec.grid.dtype}"
        assert np.isfinite(ec.grid).all()
    if ec.grid_clear_frac is not None:
        assert ec.grid_clear_frac.shape == (T_kept, GRID_CELLS)
        g = ec.grid_clear_frac
        assert (g >= 0).all() and (g <= 1).all(), "grid_clear_frac outside [0, 1]"
        np.testing.assert_allclose(g.mean(axis=1), ec.clear_frac, rtol=0, atol=1e-6)
    if ec.window_span_days is not None:
        assert ec.window_span_days.shape == (T_kept,)
        assert np.isfinite(ec.window_span_days).all()
        assert (ec.window_span_days >= 0).all()
    assert ec.timestamps.shape == ec.clear_frac.shape == ec.kept_idx.shape == (T_kept,)
    assert ec.embeddings.dtype == np.float32
    assert np.isfinite(ec.embeddings).all()
    assert (ec.clear_frac > 0).all() and (ec.clear_frac <= 1).all()
    assert np.all(np.diff(ec.timestamps) > np.timedelta64(0, "ns"))
    return ec
