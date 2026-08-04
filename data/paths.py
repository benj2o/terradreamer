"""Per-phase artefact directories, so re-running one phase never touches another.

Layout:

    data/raw/                    SHARED. The downloaded cubes. Phase-independent,
                                 67 MB, reused by every phase; never reset.
    data/phase1_2/embeddings/    per-(cube, encoder) .npz
    data/phase1_2/masks/         per-cube valid masks
    data/phase1_3/folds/         Phase 1.3 artefacts, and so on

Why this shape: artefacts are invalidated by the phase that produced them, not
by the cubes. Re-running Phase 1.3 must not force a re-encode of Phase 1.2, and
re-running Phase 1.2 must not silently leave Phase 1.3's folds pointing at
embeddings that no longer exist. ``reset_phase`` deletes exactly one phase's
outputs and prints what it removed, so a re-run starts clean without anyone
reaching for ``rm -rf`` on a path they typed by hand.

``data/raw`` is deliberately NOT under a phase: re-downloading 20 cubes to
re-run a probe would be pure waste, and the cubes are identical for every phase.
"""

from __future__ import annotations

import os
import shutil

__all__ = ["DATA_ROOT", "RAW_DIR", "phase_dir", "reset_phase", "migrate_legacy"]

DATA_ROOT = "data"
RAW_DIR = os.path.join(DATA_ROOT, "raw")

# Directories that existed before phases were introduced, and where they go.
_LEGACY = {
    os.path.join(DATA_ROOT, "embeddings"): ("phase1_2", "embeddings"),
    os.path.join(DATA_ROOT, "masks"): ("phase1_2", "masks"),
}


def phase_dir(phase: str, kind: str, root: str = DATA_ROOT, create: bool = True) -> str:
    """``data/<phase>/<kind>``, e.g. phase_dir("phase1_2", "embeddings")."""
    assert phase and not phase.startswith("/"), f"bad phase {phase!r}"
    assert kind and not kind.startswith("/"), f"bad kind {kind!r}"
    p = os.path.join(root, phase, kind)
    if create:
        os.makedirs(p, exist_ok=True)
    return p


def reset_phase(phase: str, root: str = DATA_ROOT, verbose: bool = True) -> int:
    """Delete every artefact of ONE phase. Returns the number of files removed.

    Refuses to touch ``data/raw``: the cubes are shared and re-downloading them
    to re-run a probe is waste, not hygiene.
    """
    target = os.path.join(root, phase)
    assert os.path.abspath(target) != os.path.abspath(RAW_DIR), (
        "reset_phase will not delete the shared cube directory"
    )
    if not os.path.isdir(target):
        if verbose:
            print(f"[paths] nothing to reset: {target} does not exist")
        return 0
    n = sum(len(f) for _r, _d, f in os.walk(target))
    size = sum(os.path.getsize(os.path.join(r, f))
               for r, _d, fs in os.walk(target) for f in fs)
    shutil.rmtree(target)
    os.makedirs(target, exist_ok=True)
    if verbose:
        print(f"[paths] reset {target}: removed {n} file(s), {size / 1e6:.1f} MB")
    return n


def migrate_legacy(root: str = DATA_ROOT, verbose: bool = True) -> int:
    """Move pre-phase ``data/embeddings`` and ``data/masks`` under phase1_2.

    One-time and idempotent, so an existing Drive checkout is not forced into a
    needless re-encode just because the layout changed.
    """
    moved = 0
    for old, (phase, kind) in _LEGACY.items():
        old = os.path.join(root, os.path.basename(old))
        if not os.path.isdir(old):
            continue
        files = [f for f in os.listdir(old) if f.endswith(".npz")]
        if not files:
            continue
        new = phase_dir(phase, kind, root=root)
        for f in files:
            dst = os.path.join(new, f)
            if not os.path.exists(dst):
                shutil.move(os.path.join(old, f), dst)
                moved += 1
        if verbose:
            print(f"[paths] migrated {len(files)} file(s): {old} -> {new}")
    if verbose and not moved:
        print("[paths] no legacy artefacts to migrate")
    return moved
