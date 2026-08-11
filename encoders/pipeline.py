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

import glob
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

__all__ = ["EncodedCube", "CubeMasks", "EmbeddingAudit", "SCHEMA_VERSION",
           "REQUIRED_KEYS", "encode_cube", "save_encoded", "load_encoded",
           "assert_encoded", "inspect_encoded", "migrate_to_current",
           "audit_embeddings", "print_embedding_audit",
           "assert_embeddings_complete", "assert_caches_agree",
           "cube_masks", "save_masks", "load_masks"]

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

# Every field a current .npz must carry. Used by inspect_encoded and
# migrate_to_current to tell "unstamped but complete" apart from "missing a
# field", which need completely different remedies -- a re-stamp versus a
# GPU re-encode. variant__* is deliberately absent: raw_features and both
# Satlas wrappers legitimately emit none.
REQUIRED_KEYS = ("embeddings", "timestamps", "clear_frac", "kept_idx",
                 "encoder", "cube", "grid", "grid_clear_frac",
                 "window_span_days")


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


def _read_encoded(z) -> EncodedCube:
    """Build an EncodedCube from an open npz, without checking the version."""
    return EncodedCube(
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


def assert_encoded(ec: EncodedCube) -> EncodedCube:
    """Every invariant a saved EncodedCube must satisfy. Shared by load and
    migrate, so a re-stamped file is held to exactly the load-time standard."""
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


def inspect_encoded(path: str) -> dict:
    """What is actually in one .npz, WITHOUT enforcing the version.

    Diagnostic only -- never returns something a probe can consume. Use it to
    tell "written before the stamp existed but otherwise complete" apart from
    "genuinely missing a field", which have completely different remedies.

    Also reports whether the FILENAME is the one ``save_encoded`` would have
    chosen for this file's contents. It usually is, and when it is not the file
    is a copy: Google Drive renames a duplicate to "Copy of <name>" (or "Kopie
    von", "Copie de", ... -- the prefix is localised, so never match on it).
    Comparing against the name derived from the stored ``cube`` and ``encoder``
    is exact, locale-independent, and needs no list of Drive's prefixes.
    """
    with np.load(path) as z:
        keys = set(z.files)
        version = int(z["schema_version"]) if "schema_version" in z else 0
        cube = str(z["cube"]) if "cube" in z else None
        encoder = str(z["encoder"]) if "encoder" in z else None
    missing = [k for k in REQUIRED_KEYS if k not in keys]
    expected = (os.path.basename(_npz_path("", cube, encoder))
                if cube and encoder else None)
    return {"path": path, "file": os.path.basename(path), "schema_version": version,
            "missing": missing, "n_variants": len([k for k in keys
                                                   if k.startswith("variant__")]),
            "complete": not missing, "cube": cube, "encoder": encoder,
            "expected_file": expected,
            "canonical": expected is not None and os.path.basename(path) == expected}


class EmbeddingAudit(NamedTuple):
    """What a directory of .npz actually contains, partitioned by usability.

    ``current`` is the ONLY part a probe may consume: {(cube, encoder): path},
    every entry stamped v{SCHEMA_VERSION} and named as ``save_encoded`` would
    have named it. Everything else is a defect with its own remedy.
    """

    dir: str
    current: dict        # {(cube, encoder): path} -- usable
    unstamped: list      # complete but v0: re-stampable, no GPU
    incomplete: list     # missing a REQUIRED key: must be re-encoded
    foreign: list        # cube not in cube_ids: another tile, or a Drive copy
    duplicates: list     # a second file for a (cube, encoder) already covered
    unreadable: list     # [(path, error)]
    encoders: tuple
    cubes: tuple


def audit_embeddings(emb_dir: str, cube_ids=None, verbose: bool = True):
    """Partition an embeddings directory before any probe reads from it.

    WHY THIS EXISTS, AND WHY EVERY LATER PHASE SHOULD CALL IT FIRST. Globbing a
    directory and taking ``sorted(...)[0]`` is not a selection, it is a
    coin-flip that a shared Drive folder will eventually lose. Three things
    accumulate there that look like embeddings and are not:

      * Drive duplicates. Copying a file into a folder that already holds that
        name yields "Copy of <name>.npz", which still ends in the right
        ``__<encoder>.npz`` suffix and sorts BEFORE the real file, so
        ``sorted(...)[0]`` picks the copy every time. Detected by comparing the
        filename against the one derived from the file's own stored cube and
        encoder -- exact, and independent of Drive's UI language.
      * Artefacts from an older schema, which ``load_encoded`` refuses one file
        at a time, halting a run at an arbitrary point with the rest
        undiagnosed.
      * Files for cubes that are not in this manifest at all.

    ``cube_ids`` is the set of cube ids the caller actually intends to use --
    ``set(MANIFEST.cube_id)``. Pass it: without it, foreign files cannot be
    told from legitimate ones.
    """
    paths = sorted(glob.glob(os.path.join(emb_dir, "*.npz")))
    by_key, foreign, incomplete, unstamped, unreadable = {}, [], [], [], []

    for p in paths:
        try:
            info = inspect_encoded(p)
        except Exception as e:                       # noqa: BLE001 - reported
            unreadable.append((p, f"{type(e).__name__}: {e}"))
            continue
        if info["cube"] is None or info["encoder"] is None:
            unreadable.append((p, "no cube/encoder field -- not one of ours"))
            continue
        if cube_ids is not None and info["cube"] not in cube_ids:
            foreign.append(info)
            continue
        by_key.setdefault((info["cube"], info["encoder"]), []).append(info)

    current, duplicates = {}, []
    for key, infos in sorted(by_key.items()):
        # Prefer the canonically-named file; a copy only wins if it is alone.
        canon = [i for i in infos if i["canonical"]]
        chosen = (canon or infos)[0]
        duplicates.extend(i for i in infos if i is not chosen)
        if chosen["missing"]:
            incomplete.append(chosen)
        elif chosen["schema_version"] == 0:
            unstamped.append(chosen)
        elif chosen["schema_version"] != SCHEMA_VERSION:
            incomplete.append(chosen)                # declares a schema we lack
        else:
            current[key] = chosen["path"]

    audit = EmbeddingAudit(
        dir=emb_dir, current=current, unstamped=unstamped, incomplete=incomplete,
        foreign=foreign, duplicates=duplicates, unreadable=unreadable,
        encoders=tuple(sorted({e for _, e in current})),
        cubes=tuple(sorted({c for c, _ in current})),
    )
    if verbose:
        print_embedding_audit(audit, len(paths))
    return audit


def print_embedding_audit(audit: EmbeddingAudit, n_files: int | None = None) -> None:
    """Everything the audit found, with the remedy for each defect."""
    n_files = len(audit.current) if n_files is None else n_files
    print(f"[audit] {audit.dir}")
    print(f"[audit]   {n_files} .npz on disk -> {len(audit.current)} usable "
          f"({len(audit.cubes)} cubes x {len(audit.encoders)} encoders)")
    for enc in audit.encoders:
        n = sum(1 for _, e in audit.current if e == enc)
        print(f"[audit]     {enc:<24} {n:>3} cubes")
    if audit.duplicates:
        print(f"[audit]   {len(audit.duplicates)} DUPLICATE file(s) -- a second "
              "copy of a (cube, encoder)\n[audit]   already covered. The "
              "canonically-named file was used. Delete these:")
        for i in audit.duplicates[:6]:
            print(f"[audit]     {i['file']}   (should be {i['expected_file']})")
        if len(audit.duplicates) > 6:
            print(f"[audit]     ... and {len(audit.duplicates) - 6} more")
    if audit.foreign:
        print(f"[audit]   {len(audit.foreign)} file(s) for cubes NOT in this "
              "manifest -- another tile,\n[audit]   or a Drive copy whose cube "
              "id no longer parses. Ignored:")
        for i in audit.foreign[:6]:
            print(f"[audit]     {i['file']}")
        if len(audit.foreign) > 6:
            print(f"[audit]     ... and {len(audit.foreign) - 6} more")
    if audit.unstamped:
        print(f"[audit]   {len(audit.unstamped)} file(s) COMPLETE but UNSTAMPED "
              "-- re-stampable, no GPU:")
        print(f"[audit]     python -m scripts.restamp_cache --dir '{audit.dir}'")
        print("[audit]     (add --apply once the dry run looks right)")
    if audit.incomplete:
        print(f"[audit]   {len(audit.incomplete)} file(s) genuinely INCOMPLETE "
              "-- must be re-encoded:")
        for i in audit.incomplete[:6]:
            print(f"[audit]     {i['file']}  missing {i['missing']}")
    if audit.unreadable:
        print(f"[audit]   {len(audit.unreadable)} UNREADABLE file(s):")
        for p, e in audit.unreadable[:6]:
            print(f"[audit]     {os.path.basename(p)}: {e}")


def assert_embeddings_complete(audit: EmbeddingAudit, cube_ids, encoders=None) -> None:
    """Every (cube, encoder) the caller needs is present and current.

    This is the check that protects P1-P4. Asserting the join contract on ONE
    pair proves the contract; asserting it on ALL of them proves the cache is
    whole. A cube silently missing one encoder turns a per-encoder comparison
    into a comparison over different cubes, which is a confound no downstream
    assertion can detect.
    """
    cube_ids = sorted(set(cube_ids))
    # Emptiness is checked on the AUDIT, not on the requested encoder list: a
    # caller naming encoders it expects must still get "there is nothing here"
    # rather than a gap list enumerating every pair, which buries the cause.
    assert audit.current, (
        f"no usable embeddings in {audit.dir}. Nothing can be joined. See the "
        "audit above for which defect applies and its remedy; if every line is "
        "empty too, the directory itself is wrong -- check the path."
    )
    encoders = tuple(sorted(encoders)) if encoders is not None else audit.encoders
    gaps = [(c, e) for c in cube_ids for e in encoders if (c, e) not in audit.current]
    assert not gaps, (
        f"{len(gaps)} of {len(cube_ids) * len(encoders)} (cube, encoder) pairs "
        f"have no current embedding in {audit.dir}, e.g. {gaps[:5]}.\n"
        "A per-encoder comparison over a cache with holes is a comparison over "
        "DIFFERENT cubes,\nwhich no downstream assertion can detect. Fix the "
        "defects the audit named, then re-run."
    )
    print(f"[audit] COMPLETE: all {len(cube_ids)} x {len(encoders)} = "
          f"{len(cube_ids) * len(encoders)} (cube, encoder) pairs present at "
          f"v{SCHEMA_VERSION}")


def assert_caches_agree(old_dir: str, new_dir: str, cubes, encoders,
                        tol: float = 1e-3, verbose: bool = True) -> dict:
    """Two caches must describe the same experiment on the cubes they share.

    WHY THIS IS A FUNCTION AND NOT A ONE-OFF. A scaled cache is only useful if
    its numbers are comparable to the ones already published from the small one.
    Nothing about a re-encode guarantees that: a changed mask rule, a different
    clear-fraction threshold, a torch or weights version bump, or a wrapper
    edited between the two runs all produce a cache that is internally
    consistent, passes every audit, and quietly answers a different question.
    The only test is to compare the two on the cubes they have in common, which
    is what this does.

    TWO STANDARDS, AND THE DIFFERENCE IS THE POINT:

    * ``kept_idx``, ``timestamps`` and ``clear_frac`` must be **BIT-IDENTICAL**.
      These come from the cube file and the frame-selection rule, not from a
      network, so there is no legitimate source of variation. A difference here
      means the two caches describe DIFFERENT FRAMES and no result computed from
      one may be compared to a result computed from the other.
    * ``embeddings`` and ``grid`` must agree to ``tol``, and the measured
      difference is RETURNED and printed rather than merely asserted. These are
      float32 network outputs; the two caches are routinely written on different
      hardware (this project's 20-cube cache was verified on a Colab T4, its
      scaled one likewise, and probes are re-run on local CPU). Phase 1.4
      measured what that costs end to end: a Colab reproduction of P1 moved
      balanced accuracies by +-0.003. Demanding bit-equality of network outputs
      would fail for a reason that has nothing to do with correctness.

    Returns the per-encoder worst differences, so a caller can report the number
    instead of asserting a bound and saying nothing.
    """
    cubes = sorted(set(cubes))
    encoders = tuple(sorted(set(encoders)))
    assert cubes and encoders, "nothing to compare"

    rows, n_pairs = [], 0
    for cube in cubes:
        stem = os.path.splitext(cube)[0]
        for enc in encoders:
            old_p = _npz_path(old_dir, cube, enc)
            new_p = _npz_path(new_dir, cube, enc)
            if not (os.path.exists(old_p) and os.path.exists(new_p)):
                continue
            old, new = load_encoded(old_p), load_encoded(new_p)

            np.testing.assert_array_equal(
                old.kept_idx, new.kept_idx,
                err_msg=f"{cube} x {enc}: frame SELECTION differs between "
                        f"{old_dir} and {new_dir}. The two caches describe "
                        "different frames, so no number from one is comparable "
                        "to a number from the other. This is not device jitter: "
                        "kept_idx comes from the cube and the clear-fraction "
                        "rule, never from a network.")
            np.testing.assert_array_equal(
                old.timestamps, new.timestamps,
                err_msg=f"{cube} x {enc}: timestamps differ")
            np.testing.assert_allclose(
                old.clear_frac, new.clear_frac, rtol=0, atol=0,
                err_msg=f"{cube} x {enc}: clear_frac differs; the mask "
                        "definition changed between the two encodes")

            rows.append({
                "cube": cube, "encoder": enc,
                "max_abs_pooled": float(np.abs(old.embeddings - new.embeddings).max()),
                "max_abs_grid": float(np.abs(old.grid.astype(np.float64)
                                             - new.grid.astype(np.float64)).max()),
            })
            n_pairs += 1

    assert n_pairs, (
        f"no (cube, encoder) pair exists in BOTH {old_dir} and {new_dir}, so "
        "this check compared nothing. A vacuous pass is worse than no check: "
        "verify the two directories and the shared cube list."
    )
    worst = {}
    for r in rows:
        worst[r["encoder"]] = max(worst.get(r["encoder"], 0.0), r["max_abs_pooled"])
    worst_overall = max(r["max_abs_pooled"] for r in rows)
    worst_grid = max(r["max_abs_grid"] for r in rows)

    if verbose:
        print(f"[audit] {n_pairs} shared (cube, encoder) pairs compared between")
        print(f"[audit]   old {old_dir}")
        print(f"[audit]   new {new_dir}")
        print("[audit] kept_idx / timestamps / clear_frac: BIT-IDENTICAL on all "
              f"{n_pairs}")
        for enc in sorted(worst):
            print(f"[audit]   {enc:<24} max |pooled diff| {worst[enc]:.3g}")
        print(f"[audit] worst pooled {worst_overall:.3g}, worst grid "
              f"{worst_grid:.3g} (tolerance {tol})")
    assert worst_overall <= tol, (
        f"the two caches differ by {worst_overall:.3g} on the pooled "
        f"embeddings, above the {tol} tolerance. That is too large to be "
        "float32/device jitter -- something about the ENCODING changed between "
        "the two runs (weights, wrapper, preprocessing), and results computed "
        "from the two caches are not comparable."
    )
    return {"n_pairs": n_pairs, "per_encoder_max_pooled": worst,
            "max_abs_pooled": worst_overall, "max_abs_grid": worst_grid,
            "tol": tol}


def migrate_to_current(path: str, apply: bool = False, verbose: bool = True) -> str:
    """Re-stamp an UNSTAMPED .npz that already carries every current field.

    Returns one of: "current", "migrated", "would-migrate", "incomplete",
    "wrong-version".

    WHY THIS IS NOT THE HAZARD THE VERSION STAMP EXISTS TO PREVENT. That hazard
    was silence: ``np.load`` reports a missing key as merely absent, so
    ``load_encoded`` returned None for it and a probe read ``window_span_days``,
    found nothing, and quietly dropped the covariate. This function is the
    opposite -- it REQUIRES every key in ``REQUIRED_KEYS`` to be present and
    re-runs ``assert_encoded`` in full before it will re-stamp anything. A file
    missing a field is reported and left alone.

    It exists because of an accident of ordering: ``window_span_days`` landed in
    a1a6a12 and the stamp in f4ed234, a LATER commit. Artefacts written between
    the two are complete but unstamped, and re-encoding them costs a GPU run to
    reproduce bytes that are already correct.

    NOTHING IS INVENTED. A missing ``window_span_days`` is not recomputed even
    though the timestamps are present, because a value the artefact does not
    contain is not a value it recorded.
    """
    info = inspect_encoded(path)
    name = info["file"]
    if info["schema_version"] == SCHEMA_VERSION:
        if verbose:
            print(f"[migrate] {name}: already v{SCHEMA_VERSION}, untouched")
        return "current"
    if info["schema_version"] != 0:
        if verbose:
            print(f"[migrate] {name}: stamped v{info['schema_version']}, not 0 -- "
                  f"refusing to re-stamp a file that declares a different schema")
        return "wrong-version"
    if info["missing"]:
        if verbose:
            print(f"[migrate] {name}: INCOMPLETE, missing {info['missing']} -- "
                  "re-encode this one; nothing is invented here")
        return "incomplete"

    with np.load(path) as z:
        ec = _read_encoded(z)
        payload = {k: z[k] for k in z.files}
    assert_encoded(ec)            # the full load-time standard, not a subset

    if not apply:
        if verbose:
            print(f"[migrate] {name}: complete and would be stamped v{SCHEMA_VERSION} "
                  "(dry run)")
        return "would-migrate"

    payload["schema_version"] = np.array(SCHEMA_VERSION)
    # The suffix must stay .npz: np.savez_compressed APPENDS ".npz" to any name
    # that lacks it, so a plain ".tmp" would be written as "....tmp.npz" and the
    # rename below would look for a file that was never created.
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)         # atomic: a crash cannot leave a half-written npz
    load_encoded(path)            # prove the result loads under the real guard
    if verbose:
        print(f"[migrate] {name}: stamped v{SCHEMA_VERSION} and re-verified")
    return "migrated"


def load_encoded(path: str) -> EncodedCube:
    """Read an EncodedCube back, re-asserting every invariant that was saved."""
    with np.load(path) as z:
        found = int(z["schema_version"]) if "schema_version" in z else 0
        assert found == SCHEMA_VERSION, (
            f"{os.path.basename(path)} was written with cache schema v{found}, "
            f"but this code expects v{SCHEMA_VERSION}. A cached file from an older "
            "schema is missing fields that probes read silently as absent -- "
            "window_span_days is the live example.\n"
            "TWO DIFFERENT CAUSES, two different fixes:\n"
            "  (a) written BEFORE the stamp existed but otherwise complete "
            "(window_span_days\n"
            "      landed in a1a6a12, the stamp in f4ed234, a later commit). Check "
            "and\n"
            "      re-stamp in place, no GPU needed:\n"
            "          python -m scripts.restamp_cache --dir <embeddings dir>\n"
            "  (b) genuinely missing a field. Re-encode:\n"
            "          from data.paths import reset_phase; reset_phase('phase1_2')\n"
            "The command in (a) tells you which of the two you have, and refuses "
            "to stamp\nanything incomplete. Encoder dimensionality does NOT settle "
            "it: the multi-image\nencoder and window_span_days landed in different "
            "commits."
        )
        ec = _read_encoded(z)
    return assert_encoded(ec)
