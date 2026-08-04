"""Phase-scoped artefact directories.

Every phase writes its derived artefacts under ``data/<phase>/<kind>/``, so
re-running one phase clears exactly its own outputs -- not another phase's,
and never the cubes.

    data/raw/                       SHARED. The downloaded cubes.
                                    Phase-independent, never cleared by a reset.
    data/phase1_2/embeddings/       one .npz per (cube, encoder)
    data/phase1_2/masks/            one .npz per cube
    data/phase1_3/folds/            whatever Phase 1.3 produces

Why this exists. The Colab bootstrap extracts a zip over an existing checkout,
and ``zipfile.extractall`` OVERWRITES but never DELETES. Artefacts from an
earlier run of the same phase therefore survive into the next one, and the
resumable cache in ``encoders.pipeline`` reuses whatever it finds -- which is
exactly how an embedding computed under an older mask definition gets picked
up and silently believed. A stale artefact is not an inert leftover; it is a
wrong number that no assertion will catch, because it was internally
consistent when it was written.

``reset_phase`` makes clearing them one explicit, phase-scoped call rather
than a hand-typed ``rm -rf`` aimed at a path someone has to get right under
time pressure. It refuses to touch ``data/raw``: re-downloading 20 cubes to
re-run a probe is waste, not hygiene.
"""

from __future__ import annotations

import os
import shutil

__all__ = ["DATA_ROOT", "RAW_DIR", "phase_dir", "reset_phase", "describe_phase"]

DATA_ROOT = "data"
RAW_DIR = os.path.join(DATA_ROOT, "raw")   # shared, never phase-scoped


def _validate_phase(phase: str, root: str) -> str:
    assert isinstance(phase, str) and phase, "phase must be a non-empty string"
    assert "/" not in phase and "\\" not in phase and ".." not in phase, (
        f"phase must be a bare directory name, got {phase!r}"
    )
    # Compare resolved paths, so any spelling of data/raw is caught, not just
    # the literal string "raw".
    if os.path.abspath(os.path.join(root, phase)) == os.path.abspath(RAW_DIR):
        raise AssertionError(
            f"{phase!r} under {root!r} is the shared cube directory ({RAW_DIR}). "
            "It holds the downloaded cubes, is shared across every phase, and is "
            "never cleared by a phase reset -- re-downloading 20 cubes to re-run "
            "a probe is waste, not hygiene."
        )
    return phase


def phase_dir(phase: str, kind: str | None = None, root: str = DATA_ROOT) -> str:
    """``data/<phase>[/<kind>]``, created on demand.

    ``kind`` is the artefact class: "embeddings", "masks", "folds", ...
    Omit it for the phase root itself.
    """
    _validate_phase(phase, root)
    p = os.path.join(root, phase) if kind is None else os.path.join(root, phase, kind)
    os.makedirs(p, exist_ok=True)
    return p


def describe_phase(phase: str, root: str = DATA_ROOT, verbose: bool = True) -> dict:
    """What is currently on disk for one phase. Read-only."""
    _validate_phase(phase, root)
    d = os.path.join(root, phase)
    out = {"phase": phase, "dir": d, "files": 0, "bytes": 0, "by_kind": {}}
    if not os.path.isdir(d):
        return out
    for sub in sorted(os.listdir(d)):
        full = os.path.join(d, sub)
        if not os.path.isdir(full):
            continue
        files = [f for f in os.listdir(full) if not f.startswith(".")]
        size = sum(os.path.getsize(os.path.join(full, f)) for f in files)
        out["by_kind"][sub] = {"files": len(files), "bytes": size}
        out["files"] += len(files)
        out["bytes"] += size
    if verbose:
        print(f"[paths] {d}: {out['files']} file(s), {out['bytes'] / 1e6:.2f} MB")
        for k, v in out["by_kind"].items():
            print(f"[paths]   {k}/: {v['files']} file(s), {v['bytes'] / 1e6:.2f} MB")
    return out


def reset_phase(phase: str, root: str = DATA_ROOT, verbose: bool = True) -> int:
    """Delete every artefact of ONE phase. Returns the file count removed.

    Never touches ``data/raw`` or any other phase. Call this before re-encoding
    after a change to the mask definition, the frame-selection rule, or an
    encoder's feature recipe -- see the module docstring for why a leftover is
    worse than a missing file.
    """
    before = describe_phase(phase, root, verbose=False)
    d = before["dir"]
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)   # phase root survives, empty
    if verbose:
        print(f"[paths] reset {d}: removed {before['files']} file(s), "
              f"{before['bytes'] / 1e6:.2f} MB")
        for k, v in before["by_kind"].items():
            print(f"[paths]   {k}/: {v['files']} file(s)")
        print("[paths] data/raw untouched -- cubes are shared across phases")
    return before["files"]
