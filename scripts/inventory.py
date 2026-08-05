"""Recursive inventory of a project tree, with every entry classified.

Step one of two. This script only LOOKS: it walks a root, lists every file and
folder, and labels each movable unit with the phase it belongs to.
``scripts/organise_phases.py`` consumes the result and does the moving.

    python -m scripts.inventory --root "/content/drive/MyDrive/NeurIPS-CCAI-2026"
    python -m scripts.inventory --root . --json inventory.json
    python -m scripts.inventory --root . --sha256          # slow, but see below

WHY THE CLASSIFICATION LIVES HERE AND NOT IN THE MOVER
------------------------------------------------------
"Which phase does this belong to" is the only hard question in the pair, and it
is answered exactly once, here, so the mover cannot disagree with the listing
you approved. The mover imports ``classify`` and ``iter_units``; it never
re-derives them.

WHAT A "UNIT" IS
----------------
Moves happen at unit granularity, never per file, because a checkout is only
useful whole. A unit is a top-level entry, EXCEPT that ``data/`` is expanded one
level -- it holds three different kinds at once:

    data/raw/           SHARED cubes, phase-independent, NEVER moved
    data/phase1_2/      Phase 1.2 artefacts
    data/ndvi.py        part of the checkout

Collapsing ``data/`` into one unit would drag the shared cubes into a phase
folder, which is precisely the coupling the per-phase layout exists to remove.

``--sha256`` is worth the wait once: run it before and after a move and diff the
two JSON files. A move that silently dropped or truncated a file cannot survive
that comparison, and Drive is exactly the kind of place where that happens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from typing import NamedTuple

__all__ = ["SKIP", "SHARED", "Unit", "classify", "iter_units", "walk",
           "format_tree", "sha256_of"]

# Never listed, never moved, never counted: caches and VCS metadata.
SKIP = {".git", "__pycache__", ".venv", ".pytest_cache", ".ipynb_checkpoints",
        ".DS_Store", ".mypy_cache", "node_modules"}

# The one path that is shared rather than owned by a phase. This mirrors
# data.paths.RAW_DIR and data.paths.reset_phase, which likewise refuse to treat
# the cubes as any phase's property: 1.2 and 1.3 both read them and scale-up
# will too, so filing them under a phase would force every later phase to reach
# into another phase's folder.
SHARED = ("data/raw",)

_PHASE_DIR = re.compile(r"^phase\d+_\d+$")
_PHASE_ZIP = re.compile(r"^(phase\d+_\d+)_repo\.zip$")
_PHASE_NB = re.compile(r"^(phase\d+_\d+)_.*\.ipynb$")


class Unit(NamedTuple):
    """One movable thing, plus what it is and where it belongs."""

    rel: str            # path relative to root, forward slashes
    is_dir: bool
    kind: str           # shared | artefacts | bundle | phase_folder | checkout
    phase: str | None   # "phase1_2", or None when the phase is undecidable
    n_files: int
    n_bytes: int


def classify(rel: str) -> tuple:
    """(kind, phase) for one unit path. Pure, and the single source of truth.

    kind is one of:
        shared        data/raw -- never moved, by contract
        artefacts     data/phase1_N -- one phase's outputs
        bundle        phase1_N_repo.zip -- the Colab upload for one phase
        phase_folder  phase1_N/ -- already filed correctly
        checkout      anything else: repo code, docs, notebooks

    phase is None for shared units and for checkout units, whose owning phase
    is NOT decidable from the filesystem -- a checkout looks identical whichever
    bundle produced it. The mover asks for it explicitly rather than guessing.
    """
    rel = rel.replace(os.sep, "/").strip("/")
    if rel in SHARED:
        return "shared", None
    if rel.startswith("data/") and _PHASE_DIR.match(rel.split("/", 1)[1]):
        return "artefacts", rel.split("/", 1)[1]
    m = _PHASE_ZIP.match(os.path.basename(rel))
    if m and "/" not in rel:
        return "bundle", m.group(1)
    if _PHASE_DIR.match(rel):
        return "phase_folder", rel
    return "checkout", None


def _dir_stats(path: str) -> tuple:
    """(n_files, n_bytes) under a directory, skipping SKIP entries."""
    n, b = 0, 0
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames[:] = [d for d in dirnames if d not in SKIP]
        for f in filenames:
            if f in SKIP:
                continue
            n += 1
            try:
                b += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return n, b


def iter_units(root: str) -> list:
    """The movable units under root, in a stable order.

    Top-level entries, with ``data/`` expanded one level. See the module
    docstring for why data/ is the exception.
    """
    units = []
    for name in sorted(os.listdir(root)):
        if name in SKIP:
            continue
        full = os.path.join(root, name)
        if name == "data" and os.path.isdir(full):
            for sub in sorted(os.listdir(full)):
                if sub in SKIP:
                    continue
                rel = f"data/{sub}"
                subfull = os.path.join(full, sub)
                is_dir = os.path.isdir(subfull)
                n, b = _dir_stats(subfull) if is_dir else (1, os.path.getsize(subfull))
                kind, phase = classify(rel)
                units.append(Unit(rel, is_dir, kind, phase, n, b))
            continue
        is_dir = os.path.isdir(full)
        n, b = _dir_stats(full) if is_dir else (1, os.path.getsize(full))
        kind, phase = classify(name)
        units.append(Unit(name, is_dir, kind, phase, n, b))
    return units


def walk(root: str, with_hash: bool = False) -> list:
    """Every file under root, recursively: [{path, bytes, sha256?}, ...].

    Paths are relative to root with forward slashes, so two inventories taken
    on different machines (or on Drive vs locally) compare directly.
    """
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP)
        for f in sorted(filenames):
            if f in SKIP:
                continue
            full = os.path.join(dirpath, f)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            rec = {"path": rel, "bytes": os.path.getsize(full)}
            if with_hash:
                rec["sha256"] = sha256_of(full)
            out.append(rec)
    return out


def sha256_of(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _human(n: int) -> str:
    for unit in ("B", "kB", "MB", "GB"):
        if n < 1000 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000.0


def format_tree(root: str, max_depth: int = 3) -> list:
    """A readable indented tree, depth-limited so a 100-file phase folder is
    one summarised line rather than a hundred."""
    lines = []

    def rec(path: str, prefix: str, depth: int):
        try:
            names = sorted(n for n in os.listdir(path) if n not in SKIP)
        except OSError as e:
            lines.append(f"{prefix}<unreadable: {e}>")
            return
        for i, name in enumerate(names):
            full = os.path.join(path, name)
            last = i == len(names) - 1
            branch = "`-- " if last else "|-- "
            if os.path.isdir(full):
                n, b = _dir_stats(full)
                lines.append(f"{prefix}{branch}{name}/   [{n} files, {_human(b)}]")
                if depth < max_depth:
                    rec(full, prefix + ("    " if last else "|   "), depth + 1)
            else:
                lines.append(f"{prefix}{branch}{name}   "
                             f"[{_human(os.path.getsize(full))}]")

    lines.append(f"{os.path.abspath(root)}/")
    rec(root, "", 1)
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=".", help="tree to inventory")
    ap.add_argument("--json", metavar="PATH", help="also write the inventory as JSON")
    ap.add_argument("--sha256", action="store_true",
                    help="checksum every file (slow; diff two runs to prove a "
                         "move lost nothing)")
    ap.add_argument("--depth", type=int, default=3, help="tree print depth")
    ap.add_argument("--no-tree", action="store_true", help="units and totals only")
    a = ap.parse_args(argv)

    root = os.path.abspath(a.root)
    assert os.path.isdir(root), f"not a directory: {root}"

    if not a.no_tree:
        print("\n".join(format_tree(root, a.depth)))
        print()

    units = iter_units(root)
    width = max((len(u.rel) for u in units), default=10)
    print(f"{'UNIT'.ljust(width)}  {'KIND':<12} {'PHASE':<10} {'FILES':>7}  SIZE")
    print("-" * (width + 42))
    for u in units:
        print(f"{u.rel.ljust(width)}  {u.kind:<12} {(u.phase or '-'):<10} "
              f"{u.n_files:>7}  {_human(u.n_bytes)}")

    files = walk(root, with_hash=a.sha256)
    total = sum(f["bytes"] for f in files)
    print("-" * (width + 42))
    print(f"{len(units)} units, {len(files)} files, {_human(total)} total")

    by_kind = {}
    for u in units:
        by_kind.setdefault(u.kind, [0, 0])
        by_kind[u.kind][0] += u.n_files
        by_kind[u.kind][1] += u.n_bytes
    for kind in sorted(by_kind):
        n, b = by_kind[kind]
        print(f"  {kind:<12} {n:>7} files  {_human(b)}")

    undecidable = [u.rel for u in units if u.kind == "checkout"]
    if undecidable:
        print(f"\n{len(undecidable)} checkout unit(s) have NO decidable phase -- a "
              "checkout looks the\nsame whichever bundle produced it. "
              "scripts/organise_phases.py asks you which\nphase owns them "
              "(--checkout-phase) rather than guessing:")
        print("  " + ", ".join(undecidable[:12])
              + (" ..." if len(undecidable) > 12 else ""))

    if a.json:
        payload = {"root": root, "units": [u._asdict() for u in units],
                   "files": files, "hashed": bool(a.sha256)}
        with open(a.json, "w") as fh:
            json.dump(payload, fh, indent=1)
        print(f"\nwrote {a.json} ({len(files)} files"
              f"{', hashed' if a.sha256 else ''})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
