"""File a project tree into the per-phase layout. Step two of two.

Consumes the classification from ``scripts/inventory.py`` -- it is imported,
never re-derived, so the plan you approve in the listing is the plan that runs.

    python -m scripts.organise_phases --root "/content/drive/MyDrive/NeurIPS-CCAI-2026"
    python -m scripts.organise_phases --root . --checkout-phase phase1_2 --apply

TARGET LAYOUT
-------------
    NeurIPS-CCAI-2026/
    |-- data/raw/           SHARED cubes -- NEVER moved
    |-- phase1_1/           checkout + notebook
    |-- phase1_2/           checkout; artefacts at data/phase1_2/{embeddings,masks}
    `-- phase1_3/           checkout

DRY RUN IS THE DEFAULT. Nothing moves until you pass ``--apply``. This script
relocates real artefacts that cost GPU-hours to produce, and on Drive an
interrupted move is not obviously distinguishable from a completed one, so the
plan is printed and approved before anything is touched.

FOUR REFUSALS, each enforced rather than documented:

1. ``data/raw`` is never moved. It is shared by every phase, and
   ``data.paths.reset_phase`` refuses to clear it for the same reason.
2. No destination may escape the root. A unit whose resolved destination lands
   outside is a bug in the plan, not something to execute.
3. An existing destination is never overwritten. Drive re-syncs make collisions
   real, and silently replacing an embedding with a same-named older one is the
   exact failure ``SCHEMA_VERSION`` exists to catch after the fact.
4. A checkout is moved WHOLE or not at all, and its owning phase must be stated
   with ``--checkout-phase``. A checkout looks identical whichever bundle
   produced it, so guessing it from mtimes would be a coin flip on which
   notebook you can still re-run.
"""

from __future__ import annotations

import argparse
import os
import shutil
from typing import NamedTuple

from scripts.inventory import SHARED, Unit, iter_units

__all__ = ["Move", "PlanError", "plan_moves", "apply_moves", "format_plan"]


class PlanError(RuntimeError):
    """Raised when a plan would be unsafe to execute."""


class Move(NamedTuple):
    src: str        # relative to root
    dst: str        # relative to root
    reason: str


def destination(u: Unit, checkout_phase: str | None) -> str | None:
    """Where one unit belongs, or None if it must stay put.

    Pure: takes a Unit, returns a relative path. Every decision the mover makes
    is visible here.
    """
    if u.kind == "shared":
        return None                       # refusal 1
    if u.kind == "phase_folder":
        return None                       # already filed
    if u.kind == "artefacts":
        # data/phase1_2  ->  phase1_2/data/phase1_2
        # The doubled name is deliberate: the outer is the Drive checkout, the
        # inner is data.paths.phase_dir, which is canonical.
        return f"{u.phase}/{u.rel}"
    if u.kind == "bundle":
        return f"{u.phase}/{u.rel}"
    if u.kind == "checkout":
        if checkout_phase is None:
            return None                   # refusal 4: caller must decide
        return f"{checkout_phase}/{u.rel}"
    raise PlanError(f"unclassified unit {u.rel!r} (kind {u.kind!r})")


def plan_moves(root: str, checkout_phase: str | None = None) -> list:
    """The full move plan for a root. Does not touch the filesystem."""
    moves = []
    for u in iter_units(root):
        dst = destination(u, checkout_phase)
        if dst is None or dst == u.rel:
            continue
        moves.append(Move(u.rel, dst, f"{u.kind} -> {u.phase or checkout_phase}"))
    _assert_safe(root, moves)
    return moves


def _assert_safe(root: str, moves) -> None:
    """Every refusal, checked against the plan before a single file moves."""
    root_abs = os.path.abspath(root)
    seen_dst = {}
    for m in moves:
        # 1. the shared cubes
        for shared in SHARED:
            assert not (m.src == shared or m.src.startswith(shared + "/")), (
                f"plan would move {m.src!r}, but {shared!r} is SHARED across "
                "every phase and is never filed under one. data.paths refuses "
                "to phase-scope it for the same reason."
            )
        # 2. no escaping the root
        dst_abs = os.path.abspath(os.path.join(root_abs, m.dst))
        if os.path.commonpath([root_abs, dst_abs]) != root_abs:
            raise PlanError(
                f"destination {m.dst!r} resolves outside the root ({dst_abs}). "
                "Refusing: a move that leaves the project tree is a bug in the "
                "plan, not an instruction."
            )
        # 3. no two units into one destination
        if m.dst in seen_dst:
            raise PlanError(
                f"two units both target {m.dst!r}: {seen_dst[m.dst]!r} and "
                f"{m.src!r}. Refusing rather than letting one overwrite the "
                "other."
            )
        seen_dst[m.dst] = m.src
        # 3b. destination must not already exist on disk
        if os.path.exists(dst_abs):
            raise PlanError(
                f"destination already exists: {m.dst!r}. Refusing to overwrite. "
                "Inspect it first -- on Drive a same-named leftover from an "
                "earlier layout is common, and replacing a current artefact "
                "with a stale one is silent."
            )
        # 4. source must exist
        if not os.path.exists(os.path.join(root_abs, m.src)):
            raise PlanError(f"source vanished between listing and plan: {m.src!r}")


def format_plan(root: str, moves, checkout_phase: str | None) -> list:
    lines = [f"root: {os.path.abspath(root)}",
             f"checkout phase: {checkout_phase or '(not given -- checkout units stay put)'}",
             ""]
    if not moves:
        lines.append("nothing to move: the tree is already in the phase layout.")
        return lines
    width = max(len(m.src) for m in moves)
    lines.append(f"{'FROM'.ljust(width)}  ->  TO")
    lines.append("-" * (width + 40))
    for m in moves:
        lines.append(f"{m.src.ljust(width)}  ->  {m.dst}")
    lines.append("-" * (width + 40))
    lines.append(f"{len(moves)} move(s)")
    lines.append("")
    lines.append("NOT moved, by contract: " + ", ".join(SHARED)
                 + "  (shared across every phase)")
    return lines


def apply_moves(root: str, moves, verbose: bool = True) -> int:
    """Execute a plan. Re-checks safety immediately before touching anything."""
    _assert_safe(root, moves)
    for m in moves:
        src = os.path.join(root, m.src)
        dst = os.path.join(root, m.dst)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            raise PlanError(f"destination appeared during the move: {m.dst!r}")
        shutil.move(src, dst)
        if verbose:
            print(f"[moved] {m.src}  ->  {m.dst}")
    return len(moves)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=".", help="project folder to reorganise")
    ap.add_argument("--checkout-phase", metavar="phase1_N",
                    help="which phase owns the loose checkout at the root. "
                         "Omit and checkout units stay where they are.")
    ap.add_argument("--apply", action="store_true",
                    help="actually move. Without this it is a dry run.")
    a = ap.parse_args(argv)

    root = os.path.abspath(a.root)
    assert os.path.isdir(root), f"not a directory: {root}"
    if a.checkout_phase is not None:
        assert a.checkout_phase.startswith("phase"), (
            f"--checkout-phase should look like phase1_2, got {a.checkout_phase!r}"
        )

    moves = plan_moves(root, a.checkout_phase)
    print("\n".join(format_plan(root, moves, a.checkout_phase)))

    if not moves:
        return 0
    if not a.apply:
        print("\nDRY RUN -- nothing was moved. Re-run with --apply to execute.")
        if a.checkout_phase is None:
            print("Checkout units were left out of the plan; pass "
                  "--checkout-phase phase1_N to include them.")
        return 0

    print()
    n = apply_moves(root, moves)
    print(f"\n{n} unit(s) moved. Re-run scripts/inventory.py to confirm, and if "
          "you took a\n--sha256 inventory beforehand, diff the two: that is what "
          "proves nothing was lost.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
