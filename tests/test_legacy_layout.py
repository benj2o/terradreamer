"""The legacy-flat-layout detector in notebooks/organise_drive.ipynb Step 5.

Pins a real incident. A Phase 1.3 run had 100 correct, current-schema
embeddings sitting at ``phase1_2/data/embeddings`` -- the FLAT layout from
before ``data.paths`` phase-scoped artefact dirs (commit f4ed234) -- and every
resolver in this project looks for the nested ``data/phase1_2/embeddings``
only, so those 100 files were invisible. Nothing crashed; the resolver simply
had nothing to find there and fell back to a stale duplicate elsewhere. This
detector exists so that situation is diagnosed by name rather than rediscovered
by a failed join.
"""

from __future__ import annotations

import json
import os

NOTEBOOK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "notebooks", "organise_drive.ipynb")
_BEGIN = "# === LEGACY LAYOUT DETECTOR (pinned by tests/test_legacy_layout.py) -- BEGIN ==="
_END = "# === LEGACY LAYOUT DETECTOR -- END ==="


def _detector():
    with open(NOTEBOOK) as fh:
        nb = json.load(fh)
    for c in nb["cells"]:
        src = "".join(c["source"])
        if _BEGIN in src and _END in src:
            block = src.split(_BEGIN, 1)[1].split(_END, 1)[0]
            # The block also runs a print summary using ours/SCAN_ROOT/
            # EXPAND_ALL from the wider cell; supply harmless stand-ins so
            # exec() succeeds, then hand back only the function under test.
            ns = {"HIDE": set(), "ours": [], "SCAN_ROOT": os.getcwd(),
                 "EXPAND_ALL": False}
            exec("import os, re\n" + block, ns)   # noqa: S102
            return ns["find_legacy_layout"]
    raise AssertionError(
        f"no sentinel-fenced legacy-layout block in {NOTEBOOK}. Either Step 5 "
        "lost it or the sentinels were renamed."
    )


def _npz(path, n=1):
    os.makedirs(path, exist_ok=True)
    for i in range(n):
        with open(os.path.join(path, f"cube{i}__enc.npz"), "w") as fh:
            fh.write("x")


def test_flat_embeddings_under_a_named_phase_folder_is_flagged(tmp_path):
    find = _detector()
    proj = os.path.join(str(tmp_path), "NeurIPS-CCAI-2026")
    _npz(os.path.join(proj, "phase1_2", "data", "embeddings"), 5)
    got = find(proj)
    assert len(got) == 1
    path, n, target = got[0]
    assert path.endswith(os.path.join("phase1_2", "data", "embeddings"))
    assert n == 5
    assert target.replace(os.sep, "/").endswith(
        "phase1_2/data/phase1_2/embeddings")


def test_flat_masks_is_flagged_too(tmp_path):
    find = _detector()
    proj = os.path.join(str(tmp_path), "NeurIPS-CCAI-2026")
    _npz(os.path.join(proj, "phase1_2", "data", "masks"), 3)
    got = find(proj)
    assert len(got) == 1 and got[0][0].endswith(os.path.join("data", "masks"))


def test_the_correct_nested_layout_is_never_flagged(tmp_path):
    """The fix, applied: renaming to the nested path must make the flag
    disappear, or the detector would nag forever after being obeyed."""
    find = _detector()
    proj = os.path.join(str(tmp_path), "NeurIPS-CCAI-2026")
    _npz(os.path.join(proj, "phase1_2", "data", "phase1_2", "embeddings"), 5)
    assert find(proj) == []


def test_a_stale_but_correctly_nested_directory_is_not_legacy(tmp_path):
    """The OTHER real defect (a stale 'Copy of' duplicate sitting inside a
    phase1_3 checkout) is already nested correctly -- data/phase1_2/embeddings
    -- so this detector must stay silent about it. That defect belongs to
    encoders.pipeline.audit_embeddings, not this one."""
    find = _detector()
    proj = os.path.join(str(tmp_path), "NeurIPS-CCAI-2026-phase1_3")
    _npz(os.path.join(proj, "data", "phase1_2", "embeddings"), 1)
    assert find(proj) == []


def test_an_intentional_backup_is_flagged_but_not_mistaken_for_the_project(tmp_path):
    """A folder named 'Back-Up' has no phase-pattern name itself, but its
    ancestor does -- the suggested target must nest correctly under the
    backup, not under the project root."""
    find = _detector()
    proj = os.path.join(str(tmp_path), "NeurIPS-CCAI-2026")
    _npz(os.path.join(proj, "phase1_2", "Back-Up", "data", "embeddings"), 2)
    got = find(proj)
    assert len(got) == 1
    _, _, target = got[0]
    rel = os.path.relpath(target, proj).replace(os.sep, "/")
    assert rel == "phase1_2/Back-Up/data/phase1_2/embeddings"


def test_no_phase_ancestor_reports_target_none_rather_than_guessing(tmp_path):
    find = _detector()
    root = os.path.join(str(tmp_path), "SomeOtherFolder")
    _npz(os.path.join(root, "data", "embeddings"), 1)
    got = find(root)
    assert len(got) == 1 and got[0][2] is None


def test_an_empty_flat_directory_is_not_flagged(tmp_path):
    """No .npz means nothing is actually hidden there -- an empty
    data/embeddings is not a defect worth reporting."""
    find = _detector()
    proj = os.path.join(str(tmp_path), "NeurIPS-CCAI-2026")
    os.makedirs(os.path.join(proj, "phase1_2", "data", "embeddings"))
    assert find(proj) == []


def test_multiple_phases_are_each_reported_independently(tmp_path):
    find = _detector()
    proj = os.path.join(str(tmp_path), "NeurIPS-CCAI-2026")
    _npz(os.path.join(proj, "phase1_2", "data", "embeddings"), 4)
    _npz(os.path.join(proj, "phase1_3", "data", "embeddings"), 2)
    got = {os.path.relpath(p, proj).replace(os.sep, "/"): n for p, n, _ in find(proj)}
    assert got == {"phase1_2/data/embeddings": 4, "phase1_3/data/embeddings": 2}
