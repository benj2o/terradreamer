"""The Phase 1.3 bootstrap's input resolver, extracted from the notebook.

Notebook code is normally untestable, which is why this one block is fenced
with sentinels and exercised here: it decides WHICH directory the whole phase
reads its inputs from, and getting that wrong is silent.

The regression it pins is a real one. A stale copy of
``data/phase1_2/embeddings`` sat INSIDE the phase1_3 checkout; the old rule was
"this checkout first", so it beat the true Phase 1.2 folder, and the run died
on a pre-schema file nobody knew was there.
"""

from __future__ import annotations

import json
import os

import pytest

_NOTEBOOKS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "notebooks")
NOTEBOOK = os.path.join(_NOTEBOOKS, "phase1_3_cv.ipynb")
# Every phase notebook from 1.3 on bootstraps the same way, so each carries a
# copy of the block. A second copy free to drift is worse than no copy, so they
# are asserted character-identical below.
RESOLVER_NOTEBOOKS = ("phase1_3_cv.ipynb", "phase1_4_p1_appearance.ipynb")
_BEGIN = "# === RESOLVER (pinned by tests/test_notebook_resolver.py) -- BEGIN ==="
_END = "# === RESOLVER -- END ==="


def _resolver_source(notebook=None):
    notebook = notebook or NOTEBOOK
    with open(notebook) as fh:
        nb = json.load(fh)
    for c in nb["cells"]:
        src = "".join(c["source"])
        if _BEGIN in src and _END in src:
            return src.split(_BEGIN, 1)[1].split(_END, 1)[0]
    raise AssertionError(
        f"no sentinel-fenced resolver block in {notebook}. Either the bootstrap "
        "lost it or the sentinels were renamed; without them the precedence "
        "rule is untested and free to regress to 'first hit wins'."
    )


@pytest.mark.parametrize("name", RESOLVER_NOTEBOOKS[1:])
def test_every_phase_notebook_carries_the_same_resolver(name):
    """The block is copied, not imported -- a notebook cannot import from a
    checkout it is in the middle of unpacking. So the copies are pinned to each
    other here: fixing the precedence rule in one notebook and not the other is
    exactly how the 'first hit wins' bug would come back on the phase nobody
    re-read."""
    other = _resolver_source(os.path.join(_NOTEBOOKS, name))
    assert other == _resolver_source(), (
        f"the resolver in {name} has drifted from the one in "
        f"{os.path.basename(NOTEBOOK)}. Copy it back verbatim; anything that "
        "must differ between phases belongs in a variable (PHASE, INPUT_PHASE), "
        "not in the block."
    )


def run_resolver(repo, drive, phase="phase1_3", input_phase="phase1_2"):
    """Execute the notebook's resolver with an injected environment."""
    ns = {"os": os, "glob": __import__("glob"), "REPO": repo, "DRIVE": drive,
          "PHASE": phase, "INPUT_PHASE": input_phase,
          "RAW_DIR": os.path.join("data", "raw")}
    exec(compile(_resolver_source(), "resolver", "exec"), ns)   # noqa: S102
    return ns


def _touch(path, n=1, ext=".npz"):
    os.makedirs(path, exist_ok=True)
    for i in range(n):
        with open(os.path.join(path, f"f{i}{ext}"), "w") as fh:
            fh.write("x")
    return path


@pytest.fixture
def nested(tmp_path):
    """The target Drive layout, plus the stale copy that caused the failure.

        MyDrive/NeurIPS-CCAI-2026/
          data/raw/                              20 cubes   (shared)
          phase1_2/data/phase1_2/embeddings/    100 npz     (the real ones)
          phase1_3/data/phase1_2/embeddings/      5 npz     (STALE intruder)
    """
    drive = os.path.join(str(tmp_path), "MyDrive")
    proj = os.path.join(drive, "NeurIPS-CCAI-2026")
    _touch(os.path.join(proj, "data", "raw"), 20, ".nc")
    real = _touch(os.path.join(proj, "phase1_2", "data", "phase1_2", "embeddings"), 100)
    repo = os.path.join(proj, "phase1_3")
    stale = _touch(os.path.join(repo, "data", "phase1_2", "embeddings"), 5)
    return {"drive": drive, "proj": proj, "repo": repo, "real": real, "stale": stale}


def test_the_stale_copy_inside_the_checkout_does_not_win(nested):
    ns = run_resolver(nested["repo"], nested["drive"])
    assert ns["EMB_IN"] == os.path.abspath(nested["real"])
    assert ns["EMB_IN"] != os.path.abspath(nested["stale"])


def test_it_still_loses_even_when_it_holds_more_files(tmp_path):
    """The layout contract is not a popularity contest: another phase's
    artefacts inside this checkout lose whatever the counts say."""
    drive = os.path.join(str(tmp_path), "MyDrive")
    proj = os.path.join(drive, "NeurIPS-CCAI-2026")
    real = _touch(os.path.join(proj, "phase1_2", "data", "phase1_2", "embeddings"), 3)
    repo = os.path.join(proj, "phase1_3")
    stale = _touch(os.path.join(repo, "data", "phase1_2", "embeddings"), 500)
    ns = run_resolver(repo, drive)
    assert ns["EMB_IN"] == os.path.abspath(real)
    assert ns["EMB_IN"] != os.path.abspath(stale)


def test_the_sibling_layout_resolves_too(tmp_path):
    """The older layout, which is what the failing run was actually on:
    NeurIPS-CCAI-2026/ beside NeurIPS-CCAI-2026-phase1_3/."""
    drive = os.path.join(str(tmp_path), "MyDrive")
    real = _touch(os.path.join(drive, "NeurIPS-CCAI-2026", "data", "phase1_2",
                               "embeddings"), 100)
    repo = os.path.join(drive, "NeurIPS-CCAI-2026-phase1_3")
    _touch(os.path.join(repo, "data", "phase1_2", "embeddings"), 5)
    ns = run_resolver(repo, drive)
    assert ns["IS_PHASE_CHECKOUT"] is True, "'-phase1_3' suffix must be recognised"
    assert ns["EMB_IN"] == os.path.abspath(real)


def test_the_intruder_is_reported(nested, capsys):
    run_resolver(nested["repo"], nested["drive"])
    out = capsys.readouterr().out
    assert "candidate directories hold files" in out
    assert "inside this checkout" in out
    assert "phase1_2 artefacts belong in the" in out


def test_a_plain_dev_clone_is_not_penalised(tmp_path):
    """Locally the repo root IS where data/phase1_2 belongs. Demoting it there
    would be backwards, and a warning that fires when nothing is wrong is a
    warning nobody reads the second time."""
    repo = str(tmp_path)
    here = _touch(os.path.join(repo, "data", "phase1_2", "embeddings"), 80)
    _touch(os.path.join(repo, "data", "raw"), 20, ".nc")
    ns = run_resolver(repo, drive=None)
    assert ns["IS_PHASE_CHECKOUT"] is False
    assert ns["EMB_IN"] == os.path.abspath(here)


def test_no_warning_text_on_a_clean_dev_clone(tmp_path, capsys):
    repo = str(tmp_path)
    _touch(os.path.join(repo, "data", "phase1_2", "embeddings"), 80)
    _touch(os.path.join(repo, "data", "raw"), 20, ".nc")
    run_resolver(repo, drive=None)
    assert "WARNING" not in capsys.readouterr().out


def test_shared_cubes_resolve_to_the_project_root(nested):
    ns = run_resolver(nested["repo"], nested["drive"])
    assert ns["RAW"] == os.path.abspath(os.path.join(nested["proj"], "data", "raw"))


def test_a_missing_input_returns_a_path_rather_than_raising(tmp_path):
    """Step 3 is what reports the gap, with the remedy. The resolver must hand
    it a path to name, not blow up first."""
    repo = os.path.join(str(tmp_path), "NeurIPS-CCAI-2026", "phase1_3")
    os.makedirs(repo)
    ns = run_resolver(repo, drive=os.path.join(str(tmp_path), "MyDrive"))
    assert ns["EMB_IN"].endswith(os.path.join("data", "phase1_2", "embeddings"))
    assert not os.path.isdir(ns["EMB_IN"])


def test_choice_is_deterministic(nested):
    a = run_resolver(nested["repo"], nested["drive"])["EMB_IN"]
    b = run_resolver(nested["repo"], nested["drive"])["EMB_IN"]
    assert a == b
