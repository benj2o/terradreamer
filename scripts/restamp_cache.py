"""Diagnose an embedding cache, and re-stamp files that are complete but unstamped.

    python -m scripts.restamp_cache --dir data/phase1_2/embeddings
    python -m scripts.restamp_cache --dir data/phase1_2/embeddings --apply

DRY RUN BY DEFAULT. It reports before it rewrites anything.

WHAT PROBLEM THIS SOLVES
------------------------
``load_encoded`` refuses any .npz whose ``schema_version`` is not current, and
that guard is correct: a cached file missing a newer field is read as merely
absent, so a probe would drop ``window_span_days`` -- a weather-correlated
covariate -- without a word. But the refusal cannot tell you WHICH of two very
different situations you are in, and they have very different costs:

  (a) The file predates the STAMP but carries every field anyway.
      ``window_span_days`` landed in a1a6a12; the stamp in f4ed234, a LATER
      commit. Anything written between the two is complete and unstamped.
      Fix: re-stamp in place. Seconds, no GPU.

  (b) The file genuinely lacks a field.
      Fix: re-encode. A GPU run.

This script measures which, per file, and only ever performs (a).

THIS IS NOT THE HAZARD THE STAMP EXISTS TO PREVENT
--------------------------------------------------
That hazard was silence. Here, ``migrate_to_current`` REQUIRES every key in
``REQUIRED_KEYS``, re-runs ``assert_encoded`` in full, writes atomically via a
temp file, and then re-opens the result through the real ``load_encoded``
guard. A file missing anything is reported and left untouched. Nothing is
recomputed, filled or inferred -- a missing ``window_span_days`` is NOT
regenerated from the timestamps, because a value the artefact does not contain
is not a value it recorded.

DRIVE DUPLICATES
----------------
Google Drive names a copied file ``Copy of <name>`` (localised: ``Copie de``,
``Kopie von``, ...). Those duplicates are reported separately rather than
migrated, because a probe that globs the directory can pick one at random --
which is exactly how a stale duplicate ends up being the file that loads.
"""

from __future__ import annotations

import argparse
import glob
import os

from encoders.pipeline import (
    REQUIRED_KEYS,
    SCHEMA_VERSION,
    inspect_encoded,
    migrate_to_current,
)

__all__ = ["scan", "looks_like_duplicate", "main"]

# Drive prefixes a copy with a localised "Copy of". Matching the shape rather
# than the wording keeps this from being an English-only check.
_DUP_PREFIXES = ("copy of ", "copie de ", "kopie von ", "copia de ",
                 "copia di ", "kopia av ", "kopie van ")


def looks_like_duplicate(name: str) -> bool:
    """True for a Drive-made copy, or a ``foo (1).npz`` style duplicate."""
    low = name.lower()
    if any(low.startswith(p) for p in _DUP_PREFIXES):
        return True
    stem = os.path.splitext(name)[0]
    return stem.endswith(")") and "(" in stem and stem.rsplit("(", 1)[1][:-1].isdigit()


def scan(directory: str) -> list:
    """One dict per .npz: what it is, what it is missing, whether it is a copy."""
    out = []
    for path in sorted(glob.glob(os.path.join(directory, "*.npz"))):
        name = os.path.basename(path)
        try:
            info = inspect_encoded(path)
        except Exception as e:                       # unreadable / truncated
            out.append({"path": path, "file": name, "schema_version": None,
                        "missing": list(REQUIRED_KEYS), "complete": False,
                        "duplicate": looks_like_duplicate(name), "error": str(e)})
            continue
        info["duplicate"] = looks_like_duplicate(name)
        info["error"] = None
        out.append(info)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", required=True, help="the embeddings directory")
    ap.add_argument("--apply", action="store_true",
                    help="re-stamp the complete-but-unstamped files. "
                         "Without this it is a dry run.")
    ap.add_argument("--include-duplicates", action="store_true",
                    help="also migrate 'Copy of ...' files. Off by default: "
                         "delete them instead.")
    a = ap.parse_args(argv)

    assert os.path.isdir(a.dir), f"not a directory: {a.dir}"
    rows = scan(a.dir)
    assert rows, f"no .npz files in {a.dir}"

    dups = [r for r in rows if r["duplicate"]]
    real = [r for r in rows if not r["duplicate"]]
    current = [r for r in real if r["schema_version"] == SCHEMA_VERSION]
    unstamped_ok = [r for r in real if r["schema_version"] == 0 and r["complete"]]
    incomplete = [r for r in real if r["schema_version"] == 0 and not r["complete"]]
    odd = [r for r in real if r["schema_version"] not in (0, SCHEMA_VERSION)]

    print(f"{a.dir}: {len(rows)} .npz")
    print(f"  current (v{SCHEMA_VERSION})        {len(current)}")
    print(f"  unstamped but COMPLETE     {len(unstamped_ok)}   <- re-stampable, no GPU")
    print(f"  unstamped and INCOMPLETE   {len(incomplete)}   <- must be re-encoded")
    print(f"  other schema version       {len(odd)}")
    print(f"  Drive duplicates           {len(dups)}")

    if dups:
        print("\nDRIVE DUPLICATES -- delete these. A probe that globs this directory\n"
              "can select one at random, which is how a stale copy becomes the file\n"
              "that loads:")
        for r in dups[:20]:
            print(f"  {r['file']}")
        if len(dups) > 20:
            print(f"  ... and {len(dups) - 20} more")

    if incomplete:
        print("\nINCOMPLETE -- these are genuinely missing fields, so they are NOT\n"
              "re-stamped. Nothing here is invented. Re-encode them:")
        for r in incomplete[:20]:
            print(f"  {r['file']}  missing {r['missing']}")
        if len(incomplete) > 20:
            print(f"  ... and {len(incomplete) - 20} more")

    if odd:
        print("\nDECLARES A DIFFERENT SCHEMA -- left alone:")
        for r in odd[:20]:
            print(f"  {r['file']}  v{r['schema_version']}")

    todo = unstamped_ok + (dups if a.include_duplicates else [])
    if not todo:
        print("\nnothing to re-stamp." + (
            "" if not incomplete else
            "\nRe-encode the incomplete files:\n"
            "    from data.paths import reset_phase; reset_phase('phase1_2')"))
        return 0

    print(f"\n{'RE-STAMPING' if a.apply else 'WOULD RE-STAMP'} {len(todo)} file(s). "
          "Every one is checked in full\n(all required keys present, plus the "
          "complete load-time assertion set) before\nit is rewritten; anything "
          "that fails is left untouched.\n")
    counts = {}
    for r in todo:
        status = migrate_to_current(r["path"], apply=a.apply, verbose=True)
        counts[status] = counts.get(status, 0) + 1

    print("\n" + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    if not a.apply:
        print("\nDRY RUN -- nothing was written. Re-run with --apply.")
    else:
        print(f"\nDone. Every rewritten file was re-opened through load_encoded, "
              f"so a v{SCHEMA_VERSION}\nresult here means it will load in a probe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
