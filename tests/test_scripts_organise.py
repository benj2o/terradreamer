"""The inventory classifier and the phase mover.

The mover relocates artefacts that cost GPU-hours to produce, so its refusals
are pinned here rather than trusted to a docstring: the shared cubes are never
filed under a phase, nothing escapes the root, nothing is overwritten, and a
dry run moves nothing.
"""

from __future__ import annotations

import os

import pytest

from scripts.inventory import classify, iter_units, sha256_of, walk
from scripts.organise_phases import (
    PlanError,
    apply_moves,
    destination,
    plan_moves,
)


def _tree(root, files):
    """Materialise {relative path: content} under root."""
    for rel, content in files.items():
        full = os.path.join(root, rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as fh:
            fh.write(content)
    return root


@pytest.fixture
def drive(tmp_path):
    """A project folder shaped like the real one before reorganising."""
    return _tree(str(tmp_path), {
        "data/raw/32UNU_2018-03-09_a.nc": "cube",
        "data/raw/32UNU_2018-03-19_b.nc": "cube",
        "data/phase1_2/embeddings/a__raw_features.npz": "emb",
        "data/phase1_2/masks/a__masks.npz": "msk",
        "data/ndvi.py": "canonical",
        "data/loader.py": "loader",
        "encoders/manifest.py": "manifest",
        "probes/cv.py": "splits",
        "notebooks/phase1_2_encoders.ipynb": "nb",
        "README.md": "readme",
        "phase1_2_repo.zip": "bundle2",
        "phase1_3_repo.zip": "bundle3",
    })


# --- classification ---------------------------------------------------------

@pytest.mark.parametrize("rel,kind,phase", [
    ("data/raw", "shared", None),
    ("data/phase1_2", "artefacts", "phase1_2"),
    ("data/phase1_3", "artefacts", "phase1_3"),
    ("phase1_2_repo.zip", "bundle", "phase1_2"),
    ("phase1_3_repo.zip", "bundle", "phase1_3"),
    ("phase1_1", "phase_folder", "phase1_1"),
    ("data/ndvi.py", "checkout", None),
    ("probes", "checkout", None),
    ("README.md", "checkout", None),
])
def test_classify(rel, kind, phase):
    assert classify(rel) == (kind, phase)


def test_data_is_expanded_one_level_so_raw_is_its_own_unit(drive):
    rels = {u.rel for u in iter_units(drive)}
    assert "data/raw" in rels, "data/ must be expanded, or raw is dragged along"
    assert "data/phase1_2" in rels
    assert "data" not in rels


def test_units_carry_file_counts_and_sizes(drive):
    by_rel = {u.rel: u for u in iter_units(drive)}
    assert by_rel["data/raw"].n_files == 2
    assert by_rel["data/phase1_2"].n_files == 2
    assert by_rel["data/raw"].n_bytes > 0
    assert by_rel["phase1_2_repo.zip"].is_dir is False


def test_walk_lists_every_file_with_stable_relative_paths(drive):
    got = {f["path"] for f in walk(drive)}
    assert "data/raw/32UNU_2018-03-09_a.nc" in got
    assert "probes/cv.py" in got
    assert len(got) == 12
    assert all("\\" not in p for p in got), "paths must be comparable across OSes"


def test_walk_skips_caches(drive):
    os.makedirs(os.path.join(drive, "__pycache__"))
    with open(os.path.join(drive, "__pycache__", "x.pyc"), "w") as fh:
        fh.write("junk")
    assert not any("__pycache__" in f["path"] for f in walk(drive))


def test_sha256_lets_two_inventories_be_compared(drive):
    before = {f["path"]: f["sha256"] for f in walk(drive, with_hash=True)}
    assert len(before) == 12
    p = os.path.join(drive, "probes", "cv.py")
    assert before["probes/cv.py"] == sha256_of(p)
    with open(p, "w") as fh:
        fh.write("tampered")
    after = {f["path"]: f["sha256"] for f in walk(drive, with_hash=True)}
    assert after["probes/cv.py"] != before["probes/cv.py"]


# --- the plan ---------------------------------------------------------------

def test_shared_cubes_are_never_given_a_destination(drive):
    shared = [u for u in iter_units(drive) if u.rel == "data/raw"][0]
    assert destination(shared, "phase1_2") is None


def test_plan_moves_artefacts_and_bundles_but_not_raw(drive):
    moves = plan_moves(drive)
    pairs = {(m.src, m.dst) for m in moves}
    assert ("data/phase1_2", "phase1_2/data/phase1_2") in pairs
    assert ("phase1_2_repo.zip", "phase1_2/phase1_2_repo.zip") in pairs
    assert ("phase1_3_repo.zip", "phase1_3/phase1_3_repo.zip") in pairs
    assert not any(m.src.startswith("data/raw") for m in moves)


def test_checkout_stays_put_until_its_phase_is_stated(drive):
    without = {m.src for m in plan_moves(drive)}
    assert "probes" not in without
    assert "data/ndvi.py" not in without

    with_phase = {m.src: m.dst for m in plan_moves(drive, "phase1_2")}
    assert with_phase["probes"] == "phase1_2/probes"
    assert with_phase["data/ndvi.py"] == "phase1_2/data/ndvi.py"
    assert "data/raw" not in with_phase, "raw is shared even when a phase is named"


def test_plan_is_empty_on_an_already_tidy_tree(tmp_path):
    root = _tree(str(tmp_path), {
        "data/raw/a.nc": "cube",
        "phase1_2/data/phase1_2/embeddings/a.npz": "emb",
        "phase1_3/phase1_3_repo.zip": "bundle",
    })
    assert plan_moves(root, "phase1_2") == []


def test_plan_refuses_two_units_targeting_one_destination(tmp_path, monkeypatch):
    root = _tree(str(tmp_path), {"phase1_2_repo.zip": "a", "other.txt": "b"})
    import scripts.organise_phases as op

    def collide(u, checkout_phase):
        return "phase1_2/x"
    monkeypatch.setattr(op, "destination", collide)
    with pytest.raises(PlanError, match="both target"):
        op.plan_moves(root, "phase1_2")


def test_plan_refuses_a_destination_outside_the_root(tmp_path, monkeypatch):
    root = _tree(str(tmp_path), {"phase1_2_repo.zip": "a"})
    import scripts.organise_phases as op
    monkeypatch.setattr(op, "destination", lambda u, p: "../escaped/x")
    with pytest.raises(PlanError, match="outside the root"):
        op.plan_moves(root, "phase1_2")


def test_plan_refuses_to_overwrite_an_existing_destination(drive):
    os.makedirs(os.path.join(drive, "phase1_2"))
    with open(os.path.join(drive, "phase1_2", "phase1_2_repo.zip"), "w") as fh:
        fh.write("an older bundle")
    with pytest.raises(PlanError, match="already exists"):
        plan_moves(drive)


# --- applying ---------------------------------------------------------------

def test_apply_moves_units_and_leaves_raw_alone(drive):
    before = {f["path"]: f["sha256"] for f in walk(drive, with_hash=True)}
    moves = plan_moves(drive, "phase1_2")
    n = apply_moves(drive, moves, verbose=False)
    assert n == len(moves)

    assert os.path.isfile(os.path.join(drive, "data", "raw", "32UNU_2018-03-09_a.nc"))
    assert os.path.isfile(os.path.join(
        drive, "phase1_2", "data", "phase1_2", "embeddings", "a__raw_features.npz"))
    assert os.path.isfile(os.path.join(drive, "phase1_2", "probes", "cv.py"))
    assert os.path.isfile(os.path.join(drive, "phase1_3", "phase1_3_repo.zip"))

    # Nothing lost, nothing altered: same set of hashes, different paths.
    after = {f["path"]: f["sha256"] for f in walk(drive, with_hash=True)}
    assert sorted(before.values()) == sorted(after.values())
    assert len(after) == len(before)


def test_apply_is_idempotent_a_second_plan_is_empty(drive):
    apply_moves(drive, plan_moves(drive, "phase1_2"), verbose=False)
    assert plan_moves(drive, "phase1_2") == []


def test_dry_run_moves_nothing(drive):
    before = {f["path"] for f in walk(drive)}
    plan_moves(drive, "phase1_2")          # planning alone must not touch disk
    assert {f["path"] for f in walk(drive)} == before


def test_apply_rechecks_safety_rather_than_trusting_the_plan(drive):
    from scripts.organise_phases import Move
    forged = [Move("data/raw", "phase1_1/data/raw", "hand-written")]
    with pytest.raises(AssertionError, match="SHARED"):
        apply_moves(drive, forged, verbose=False)
    assert os.path.isdir(os.path.join(drive, "data", "raw"))
