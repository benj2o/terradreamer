"""Re-stamping a complete-but-unstamped cache.

This rewrites artefacts that cost a GPU run to produce, so the rule it must
never break is pinned here: a file missing ANY required field is reported and
left alone, never stamped and never filled. Nothing is recomputed -- a missing
window_span_days is not regenerated from the timestamps that sit beside it.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from encoders.pipeline import (
    REQUIRED_KEYS,
    SCHEMA_VERSION,
    EncodedCube,
    inspect_encoded,
    load_encoded,
    migrate_to_current,
    save_encoded,
)
from scripts.restamp_cache import looks_like_duplicate, scan


def _encoded(T=4, D=6, cube="32UNU_2018-03-09_a.nc", encoder="raw_features"):
    gcf = np.full((T, 16), 0.8, dtype=np.float32)
    return EncodedCube(
        embeddings=np.zeros((T, D), dtype=np.float32),
        timestamps=(np.datetime64("2018-04-01")
                    + np.arange(T) * np.timedelta64(5, "D")).astype("datetime64[ns]"),
        clear_frac=gcf.mean(axis=1).astype(np.float64),
        kept_idx=np.arange(T),
        encoder=encoder,
        cube=cube,
        grid=np.zeros((T, 16, D), dtype=np.float16),
        grid_clear_frac=gcf,
        variants={},
        window_span_days=np.zeros(T, dtype=np.float32),
    )


def _unstamp(path, drop=()):
    """Rewrite an .npz without schema_version, optionally dropping fields --
    exactly what a file written before f4ed234 looks like."""
    with np.load(path) as z:
        payload = {k: z[k] for k in z.files
                   if k != "schema_version" and k not in drop}
    np.savez_compressed(path, **payload)
    return path


@pytest.fixture
def cache(tmp_path):
    d = str(tmp_path)
    save_encoded(d, _encoded(), verbose=False)
    return d


# --- inspection -------------------------------------------------------------

def test_inspect_reports_version_and_missing_keys(cache):
    p = os.path.join(cache, os.listdir(cache)[0])
    info = inspect_encoded(p)
    assert info["schema_version"] == SCHEMA_VERSION
    assert info["missing"] == [] and info["complete"]

    _unstamp(p, drop=("window_span_days",))
    info = inspect_encoded(p)
    assert info["schema_version"] == 0
    assert info["missing"] == ["window_span_days"]
    assert not info["complete"]


def test_inspect_does_not_enforce_the_version(cache):
    """It must read a stale file, or it could not diagnose one."""
    p = _unstamp(os.path.join(cache, os.listdir(cache)[0]))
    assert inspect_encoded(p)["schema_version"] == 0     # no raise
    with pytest.raises(AssertionError):
        load_encoded(p)                                   # the guard still bites


# --- migration --------------------------------------------------------------

def test_complete_unstamped_file_is_restamped_and_then_loads(cache):
    p = os.path.join(cache, os.listdir(cache)[0])
    before = {k: v for k, v in np.load(p).items() if k != "schema_version"}
    _unstamp(p)
    with pytest.raises(AssertionError):
        load_encoded(p)

    assert migrate_to_current(p, apply=False, verbose=False) == "would-migrate"
    assert inspect_encoded(p)["schema_version"] == 0, "dry run must not write"

    assert migrate_to_current(p, apply=True, verbose=False) == "migrated"
    ec = load_encoded(p)                                  # the real guard passes
    assert ec.window_span_days is not None

    # Re-stamping changes the stamp and NOTHING else.
    after = {k: v for k, v in np.load(p).items() if k != "schema_version"}
    assert set(before) == set(after)
    for k in before:
        assert np.array_equal(before[k], after[k]), f"{k} changed during migration"


@pytest.mark.parametrize("dropped", list(REQUIRED_KEYS))
def test_a_file_missing_any_required_field_is_refused(cache, dropped):
    p = os.path.join(cache, os.listdir(cache)[0])
    _unstamp(p, drop=(dropped,))
    assert migrate_to_current(p, apply=True, verbose=False) == "incomplete"
    assert inspect_encoded(p)["schema_version"] == 0, "must not have been stamped"


def test_missing_window_span_days_is_never_recomputed(cache):
    """The timestamps are right there, and it still must not be regenerated:
    a value the artefact does not contain is not a value it recorded."""
    p = os.path.join(cache, os.listdir(cache)[0])
    _unstamp(p, drop=("window_span_days",))
    assert migrate_to_current(p, apply=True, verbose=False) == "incomplete"
    with np.load(p) as z:
        assert "window_span_days" not in z.files
        assert "timestamps" in z.files, "the input it refused to use is present"


def test_an_already_current_file_is_untouched(cache):
    p = os.path.join(cache, os.listdir(cache)[0])
    mtime = os.path.getmtime(p)
    assert migrate_to_current(p, apply=True, verbose=False) == "current"
    assert os.path.getmtime(p) == mtime


def test_a_file_declaring_another_schema_is_refused(cache):
    p = os.path.join(cache, os.listdir(cache)[0])
    with np.load(p) as z:
        payload = {k: z[k] for k in z.files}
    payload["schema_version"] = np.array(2)
    np.savez_compressed(p, **payload)
    assert migrate_to_current(p, apply=True, verbose=False) == "wrong-version"
    assert inspect_encoded(p)["schema_version"] == 2


def test_migration_refuses_a_file_that_fails_the_load_time_assertions(cache):
    """Present-but-wrong must fail too, not just present-or-absent."""
    p = os.path.join(cache, os.listdir(cache)[0])
    with np.load(p) as z:
        payload = {k: z[k] for k in z.files if k != "schema_version"}
    payload["grid_clear_frac"] = payload["grid_clear_frac"] * 0.5   # breaks the
    np.savez_compressed(p, **payload)                # gcf.mean == clear_frac tie
    with pytest.raises(AssertionError):
        migrate_to_current(p, apply=True, verbose=False)


# --- Drive duplicates -------------------------------------------------------

@pytest.mark.parametrize("name,is_dup", [
    ("Copy of 32UNU_a__dinov2_vitb14.npz", True),
    ("copy of x.npz", True),
    ("Kopie von 32UNU_a__raw_features.npz", True),
    ("Copie de 32UNU_a__raw_features.npz", True),
    ("32UNU_a__raw_features (1).npz", True),
    ("32UNU_a__raw_features.npz", False),
    ("32UNU_2018-05-03_1081_1209__satlas_s2_swinb_mi_rgb.npz", False),
])
def test_duplicate_detection(name, is_dup):
    assert looks_like_duplicate(name) is is_dup


def test_scan_separates_duplicates_from_originals(cache):
    p = os.path.join(cache, os.listdir(cache)[0])
    dup = os.path.join(cache, "Copy of " + os.path.basename(p))
    with open(p, "rb") as a, open(dup, "wb") as b:
        b.write(a.read())
    _unstamp(dup)

    rows = {r["file"]: r for r in scan(cache)}
    assert len(rows) == 2
    assert rows[os.path.basename(dup)]["duplicate"] is True
    assert rows[os.path.basename(p)]["duplicate"] is False


def test_scan_survives_an_unreadable_file(cache):
    bad = os.path.join(cache, "truncated__raw_features.npz")
    with open(bad, "wb") as fh:
        fh.write(b"not an npz")
    rows = {r["file"]: r for r in scan(cache)}
    assert rows["truncated__raw_features.npz"]["error"] is not None
    assert rows["truncated__raw_features.npz"]["complete"] is False
