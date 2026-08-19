"""Merge the server-computed ``loco`` rows into the local P3 extreme table.

WHY THIS IS A SCRIPT AND NOT A ``pd.concat``
---------------------------------------------
The two halves were fitted on different machines, days apart, under different
Python minor versions. Concatenating them asserts they are the same
experiment. That claim is cheap to make and expensive to be wrong about: a
different cube roster, a different screen, or a different scikit-learn would
produce a table whose rows are not comparable to each other, and no downstream
assertion would catch it -- every row would still be internally consistent.

So the checks below run BEFORE the concat, and refuse rather than warn:

  * the two tables must not overlap on ``fold_mode``  (else the top-up is a
    re-run and one of them is stale)
  * ``n_cubes`` must agree                            (same 342 cubes)
  * ``cubes_excluded`` must agree                     (same exclusions, not
                                                       just the same count)
  * ``plausibility_screen`` must agree                (same screen)
  * the column sets must match, except for bookkeeping columns that are
    legitimately absent from an un-narrowed run

    .venv/bin/python -m scripts.merge_loco
"""

from __future__ import annotations

import argparse
import os
import sys

#: Columns a narrowed run carries and an explicitly-scoped run does not. Their
#: absence is expected and is filled in, never treated as a mismatch.
BOOKKEEPING = ("narrowed",)


def _fail(msg: str) -> None:
    raise SystemExit(f"\nREFUSING TO MERGE: {msg}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="data/scaled_32UQC")
    ap.add_argument("--base", default="p3_extreme_subset_results.csv",
                    help="the local table, everything except loco")
    ap.add_argument("--loco", default="p3_extreme_loco_results.csv",
                    help="the server table, loco only")
    ap.add_argument("--out", default="p3_extreme_results.csv",
                    help="the merged full grid")
    ap.add_argument("--expect-rows", type=int, default=1540)
    ap.add_argument("--expect-loco", type=int, default=308)
    args = ap.parse_args()

    import pandas as pd

    bp = os.path.join(args.root, args.base)
    lp = os.path.join(args.root, args.loco)
    for p in (bp, lp):
        if not os.path.exists(p):
            _fail(f"no table at {p}")

    base = pd.read_csv(bp)
    loco = pd.read_csv(lp)
    print(f"[merge] base {bp}: {len(base)} rows, "
          f"fold modes {sorted(base.fold_mode.unique())}")
    print(f"[merge] loco {lp}: {len(loco)} rows, "
          f"fold modes {sorted(loco.fold_mode.unique())}")

    # --- 1. no overlap: this is a top-up, not a re-run --------------------
    overlap = set(base.fold_mode.unique()) & set(loco.fold_mode.unique())
    if overlap:
        _fail(f"both tables contain fold_mode {sorted(overlap)}. This is a "
              "re-run, not a top-up, and one of them is stale.")

    if set(loco.fold_mode.unique()) != {"loco"}:
        _fail(f"the loco table holds {sorted(loco.fold_mode.unique())}, not "
              "just 'loco'.")

    # --- 2. the same experiment ------------------------------------------
    for col in ("n_cubes", "cubes_excluded", "plausibility_screen", "tile"):
        if col not in base.columns or col not in loco.columns:
            print(f"[merge] NOTE: {col!r} absent from one table, not compared")
            continue
        b = sorted(base[col].dropna().astype(str).unique())
        l = sorted(loco[col].dropna().astype(str).unique())
        if b != l:
            _fail(f"{col} differs between the tables.\n  base: {b}\n  loco: {l}\n"
                  "The two halves were not fitted on the same cubes or under "
                  "the same screen, so their rows are not comparable.")
        print(f"[merge] {col:<20} agrees: {b[0][:70] if b else '-'}")

    # --- 3. columns ------------------------------------------------------
    only_base = set(base.columns) - set(loco.columns)
    only_loco = set(loco.columns) - set(base.columns)
    unexpected = (only_base | only_loco) - set(BOOKKEEPING)
    if unexpected:
        _fail(f"column sets differ beyond bookkeeping: {sorted(unexpected)}. "
              "The two runs used different code.")
    for col in only_base:
        loco[col] = pd.NA
        print(f"[merge] filled {col!r} as NA on the loco rows (it is "
              "bookkeeping from the narrowed local run)")
    for col in only_loco:
        base[col] = pd.NA

    # --- 4. merge ---------------------------------------------------------
    out = pd.concat([base, loco[base.columns]], ignore_index=True)
    print(f"\n[merge] {len(base)} + {len(loco)} = {len(out)} rows")
    print("[merge] rows per fold mode:")
    print(out.fold_mode.value_counts().to_string())

    if len(loco) != args.expect_loco:
        print(f"\n[merge] WARNING: expected {args.expect_loco} loco rows, "
              f"got {len(loco)}. Merging anyway -- check the server log for a "
              "narrowing or a failed view before trusting this table.")
    if len(out) != args.expect_rows:
        print(f"[merge] WARNING: merged table is {len(out)} rows, not the "
              f"{args.expect_rows} of a full grid.")
    else:
        print(f"\n[merge] this is the FULL {args.expect_rows}-row grid. The "
              "full-table completeness assertions now apply to it, which they "
              "did not to the narrowed local table.")

    op = os.path.join(args.root, args.out)
    out.to_csv(op, index=False)
    print(f"[merge] wrote {op}")


if __name__ == "__main__":
    main()
