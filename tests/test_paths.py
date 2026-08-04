"""Phase-scoped artefact directories: a reset must clear one phase and only one.

The canonical behaviour is pinned in tests/test_encoders.py; these cover the
edges around it.
"""

from __future__ import annotations

import os

import pytest

from data.paths import describe_phase, phase_dir, reset_phase


def _seed(root, phase, kinds=("embeddings", "masks"), n=3):
    for k in kinds:
        d = phase_dir(phase, k, root=root)
        for i in range(n):
            open(os.path.join(d, f"f{i}.npz"), "wb").write(b"x" * 100)


def test_describe_reports_per_kind_counts(tmp_path):
    root = str(tmp_path)
    _seed(root, "phase1_2", n=4)
    d = describe_phase("phase1_2", root, verbose=False)
    assert d["files"] == 8 and d["bytes"] == 8 * 100
    assert set(d["by_kind"]) == {"embeddings", "masks"}


def test_describe_on_a_phase_with_nothing_on_disk(tmp_path):
    d = describe_phase("phase9_9", str(tmp_path), verbose=False)
    assert d["files"] == 0 and d["by_kind"] == {}


@pytest.mark.parametrize("bad", ["../escape", "a/b", ""])
def test_phase_name_must_be_a_bare_directory(tmp_path, bad):
    with pytest.raises(AssertionError):
        phase_dir(bad, "embeddings", root=str(tmp_path))


def test_raw_is_refused_however_it_is_spelled(tmp_path):
    """The guard compares resolved paths, so it is not fooled by a different
    spelling of the same directory."""
    from data.paths import RAW_DIR

    with pytest.raises(AssertionError, match="shared cube directory"):
        reset_phase("raw", root=os.path.dirname(RAW_DIR) or ".", verbose=False)


def test_phase_root_survives_reset_but_is_empty(tmp_path):
    root = str(tmp_path)
    _seed(root, "phase1_2")
    assert reset_phase("phase1_2", root=root, verbose=False) == 6
    assert os.path.isdir(os.path.join(root, "phase1_2"))
    assert os.listdir(os.path.join(root, "phase1_2")) == []
