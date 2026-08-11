"""audit_embeddings: a shared Drive folder accumulates things that look like
embeddings and are not, and a probe that globs it must not pick one.

The live failure this pins: Google Drive renamed a copy to
"Copy of <cube>__dinov2_vitb14.npz", which still ends in the right
"__<encoder>.npz" suffix and sorts BEFORE the real file, so sorted(...)[0]
selected the copy -- a pre-schema artefact -- and halted a run at Step 10.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from encoders.pipeline import (
    EncodedCube,
    SCHEMA_VERSION,
    assert_caches_agree,
    assert_embeddings_complete,
    audit_embeddings,
    inspect_encoded,
    save_encoded,
)

CUBES = ("32UNU_2018-03-09_2018-08-05_1081_1209_3641_3769_16_96_56_136.nc",
         "32UNU_2018-03-19_2018-08-15_1337_1465_5305_5433_20_100_82_162.nc")
ENCODERS = ("raw_features", "dinov2_vitb14")


def _encoded(cube, encoder, T=3, D=5):
    ts = np.array(["2018-04-02", "2018-04-12", "2018-04-22"], dtype="datetime64[ns]")[:T]
    return EncodedCube(
        embeddings=np.zeros((T, D), dtype=np.float32),
        timestamps=ts,
        clear_frac=np.full(T, 0.8),
        kept_idx=np.arange(T),
        encoder=encoder,
        cube=cube,
        grid=np.zeros((T, 16, D), dtype=np.float16),
        grid_clear_frac=np.full((T, 16), 0.8, dtype=np.float32),
        variants={},
        window_span_days=np.zeros(T, dtype=np.float32),
    )


@pytest.fixture
def emb(tmp_path):
    """A healthy cache: every cube x every encoder, canonically named."""
    d = str(tmp_path)
    for c in CUBES:
        for e in ENCODERS:
            save_encoded(d, _encoded(c, e), verbose=False)
    return d


def _drive_copy(emb_dir, cube, encoder, prefix="Copy of "):
    """Reproduce Drive's duplicate: same bytes, renamed."""
    stem = os.path.splitext(cube)[0]
    src = os.path.join(emb_dir, f"{stem}__{encoder}.npz")
    dst = os.path.join(emb_dir, f"{prefix}{stem}__{encoder}.npz")
    with open(src, "rb") as a, open(dst, "wb") as b:
        b.write(a.read())
    return dst


def _unstamp(path):
    """Strip schema_version, leaving every other field: the v0 case."""
    with np.load(path) as z:
        payload = {k: z[k] for k in z.files if k != "schema_version"}
    os.remove(path)
    np.savez_compressed(path, **payload)


# --- the healthy case -------------------------------------------------------

def test_healthy_cache_is_all_current(emb):
    a = audit_embeddings(emb, cube_ids=CUBES, verbose=False)
    assert len(a.current) == 4
    assert a.encoders == tuple(sorted(ENCODERS))
    assert a.cubes == tuple(sorted(CUBES))
    assert not (a.unstamped or a.incomplete or a.foreign
                or a.duplicates or a.unreadable)


def test_assert_complete_passes_on_a_whole_cache(emb, capsys):
    a = audit_embeddings(emb, cube_ids=CUBES, verbose=False)
    assert_embeddings_complete(a, CUBES, ENCODERS)
    assert "COMPLETE" in capsys.readouterr().out


# --- Drive duplicates: the live failure -------------------------------------

def test_drive_copy_is_flagged_and_the_real_file_still_wins(emb):
    dup = _drive_copy(emb, CUBES[0], "dinov2_vitb14")
    assert os.path.basename(dup).startswith("Copy of ")
    a = audit_embeddings(emb, cube_ids=CUBES, verbose=False)

    assert len(a.duplicates) == 1
    assert a.duplicates[0]["file"] == os.path.basename(dup)
    # The canonically-named file is the one selected, NOT the copy that sorts
    # first -- which is exactly what sorted(glob(...))[0] got wrong.
    chosen = a.current[(CUBES[0], "dinov2_vitb14")]
    assert os.path.basename(chosen) == a.duplicates[0]["expected_file"]
    assert "Copy of" not in os.path.basename(chosen)
    assert len(a.current) == 4


@pytest.mark.parametrize("prefix", ["Copy of ", "Kopie von ", "Copie de ",
                                    "Copia de "])
def test_duplicate_detection_is_locale_independent(emb, prefix):
    """Drive localises the prefix, so the rule must never match on the prefix.
    Comparing the name against the one derived from the file's own contents is
    exact in every language."""
    _drive_copy(emb, CUBES[0], "raw_features", prefix=prefix)
    a = audit_embeddings(emb, cube_ids=CUBES, verbose=False)
    assert len(a.duplicates) == 1
    assert "Copy of" not in os.path.basename(a.current[(CUBES[0], "raw_features")])


def test_a_copy_is_still_used_when_it_is_the_only_file(emb):
    """A renamed file is a defect, not a reason to lose data."""
    dup = _drive_copy(emb, CUBES[0], "raw_features")
    os.remove(os.path.join(emb, f"{os.path.splitext(CUBES[0])[0]}__raw_features.npz"))
    a = audit_embeddings(emb, cube_ids=CUBES, verbose=False)
    assert a.current[(CUBES[0], "raw_features")] == dup
    assert not a.duplicates


def test_a_stale_copy_does_not_mask_a_good_original(emb):
    """The exact shape of the reported failure: the copy is pre-schema and the
    original is fine. The cache must still be usable."""
    dup = _drive_copy(emb, CUBES[0], "dinov2_vitb14")
    _unstamp(dup)
    a = audit_embeddings(emb, cube_ids=CUBES, verbose=False)
    assert len(a.current) == 4, "the good original must still be selected"
    assert not a.unstamped, "the stale file is a duplicate, not a cache defect"
    assert len(a.duplicates) == 1
    assert_embeddings_complete(a, CUBES, ENCODERS)


# --- schema partitioning ----------------------------------------------------

def test_unstamped_is_reported_as_restampable_not_incomplete(emb):
    _unstamp(os.path.join(emb, f"{os.path.splitext(CUBES[0])[0]}__raw_features.npz"))
    a = audit_embeddings(emb, cube_ids=CUBES, verbose=False)
    assert len(a.unstamped) == 1 and not a.incomplete
    assert a.unstamped[0]["complete"] is True
    assert (CUBES[0], "raw_features") not in a.current


def test_missing_a_required_field_is_incomplete_not_restampable(emb):
    p = os.path.join(emb, f"{os.path.splitext(CUBES[0])[0]}__raw_features.npz")
    with np.load(p) as z:
        payload = {k: z[k] for k in z.files
                   if k not in ("schema_version", "window_span_days")}
    os.remove(p)
    np.savez_compressed(p, **payload)
    a = audit_embeddings(emb, cube_ids=CUBES, verbose=False)
    assert len(a.incomplete) == 1 and not a.unstamped
    assert "window_span_days" in a.incomplete[0]["missing"]


def test_unreadable_file_is_reported_not_raised(emb):
    with open(os.path.join(emb, "truncated__raw_features.npz"), "wb") as fh:
        fh.write(b"not a zip file at all")
    a = audit_embeddings(emb, cube_ids=CUBES, verbose=False)
    assert len(a.unreadable) == 1
    assert len(a.current) == 4, "one bad file must not cost the whole cache"


# --- foreign cubes ----------------------------------------------------------

def test_a_cube_outside_the_manifest_is_foreign(emb):
    other = "33TUN_2019-04-01_2019-08-28_1_129_1_129_0_80_0_80.nc"
    save_encoded(emb, _encoded(other, "raw_features"), verbose=False)
    a = audit_embeddings(emb, cube_ids=CUBES, verbose=False)
    assert len(a.foreign) == 1 and a.foreign[0]["cube"] == other
    assert len(a.current) == 4


def test_without_cube_ids_nothing_is_foreign(emb):
    save_encoded(emb, _encoded("33TUN_2019-04-01_x.nc", "raw_features"),
                 verbose=False)
    a = audit_embeddings(emb, cube_ids=None, verbose=False)
    assert not a.foreign and len(a.current) == 5


# --- completeness -----------------------------------------------------------

def test_a_hole_in_the_cache_fails_loudly(emb):
    os.remove(os.path.join(emb, f"{os.path.splitext(CUBES[0])[0]}__dinov2_vitb14.npz"))
    a = audit_embeddings(emb, cube_ids=CUBES, verbose=False)
    with pytest.raises(AssertionError, match="DIFFERENT cubes"):
        assert_embeddings_complete(a, CUBES, ENCODERS)


def test_an_empty_directory_fails_with_a_readable_message(tmp_path):
    a = audit_embeddings(str(tmp_path), cube_ids=CUBES, verbose=False)
    with pytest.raises(AssertionError, match="no usable embeddings"):
        assert_embeddings_complete(a, CUBES, ENCODERS)


def test_complete_defaults_to_the_encoders_actually_present(emb):
    """Not passing `encoders` must not silently pass on a cache missing one."""
    a = audit_embeddings(emb, cube_ids=CUBES, verbose=False)
    assert_embeddings_complete(a, CUBES)          # both encoders present
    for c in CUBES:
        os.remove(os.path.join(emb, f"{os.path.splitext(c)[0]}__dinov2_vitb14.npz"))
    a = audit_embeddings(emb, cube_ids=CUBES, verbose=False)
    assert a.encoders == ("raw_features",)
    assert_embeddings_complete(a, CUBES)          # honest: one encoder, whole


# --- inspect_encoded's new fields -------------------------------------------

def test_inspect_reports_the_canonical_name(emb):
    p = os.path.join(emb, f"{os.path.splitext(CUBES[0])[0]}__raw_features.npz")
    info = inspect_encoded(p)
    assert info["canonical"] is True
    assert info["cube"] == CUBES[0] and info["encoder"] == "raw_features"
    assert info["expected_file"] == os.path.basename(p)
    assert info["schema_version"] == SCHEMA_VERSION


def test_inspect_flags_a_non_canonical_name(emb):
    dup = _drive_copy(emb, CUBES[0], "raw_features")
    info = inspect_encoded(dup)
    assert info["canonical"] is False
    assert info["expected_file"] == f"{os.path.splitext(CUBES[0])[0]}__raw_features.npz"
    assert info["cube"] == CUBES[0], "contents are authoritative, not the name"


# ---------------------------------------------------------------------------
# assert_caches_agree: a scaled cache must measure the same thing as the small
# one, or the two tables cannot be compared and the scale-up proves nothing.
# ---------------------------------------------------------------------------

def _cache(tmp_path, name, **overrides):
    """A cache whose one cube/encoder can be perturbed field by field."""
    d = os.path.join(str(tmp_path), name)
    os.makedirs(d, exist_ok=True)
    ec = _encoded(CUBES[0], ENCODERS[0])
    if overrides:
        ec = ec._replace(**overrides)
    save_encoded(d, ec, verbose=False)
    return d


def test_two_identical_caches_agree(tmp_path):
    a, b = _cache(tmp_path, "a"), _cache(tmp_path, "b")
    out = assert_caches_agree(a, b, [CUBES[0]], [ENCODERS[0]], verbose=False)
    assert out["n_pairs"] == 1
    assert out["max_abs_pooled"] == 0.0


def test_float_jitter_inside_the_tolerance_is_accepted_and_reported(tmp_path):
    """Network outputs on different hardware differ slightly; that is expected
    and must be MEASURED rather than asserted away."""
    a = _cache(tmp_path, "a")
    b = _cache(tmp_path, "b",
               embeddings=np.full((3, 5), 1e-6, dtype=np.float32))
    out = assert_caches_agree(a, b, [CUBES[0]], [ENCODERS[0]], verbose=False)
    assert out["max_abs_pooled"] == pytest.approx(1e-6)
    assert out["max_abs_pooled"] <= out["tol"]


def test_an_embedding_difference_above_tolerance_is_refused(tmp_path):
    a = _cache(tmp_path, "a")
    b = _cache(tmp_path, "b", embeddings=np.ones((3, 5), dtype=np.float32))
    with pytest.raises(AssertionError, match="not comparable"):
        assert_caches_agree(a, b, [CUBES[0]], [ENCODERS[0]], verbose=False)


def test_a_different_frame_selection_is_refused_however_small(tmp_path):
    """kept_idx comes from the cube and the clear-fraction rule, never from a
    network, so there is no tolerance for it: a difference means the two caches
    describe DIFFERENT FRAMES."""
    a = _cache(tmp_path, "a")
    b = _cache(tmp_path, "b", kept_idx=np.array([0, 1, 3]))
    with pytest.raises(AssertionError, match="frame SELECTION differs"):
        assert_caches_agree(a, b, [CUBES[0]], [ENCODERS[0]], verbose=False)


def test_a_changed_mask_definition_is_refused(tmp_path):
    a = _cache(tmp_path, "a")
    b = _cache(tmp_path, "b", clear_frac=np.full(3, 0.80000001))
    with pytest.raises(AssertionError, match="clear_frac differs"):
        assert_caches_agree(a, b, [CUBES[0]], [ENCODERS[0]], verbose=False)


def test_comparing_nothing_is_refused_rather_than_passing(tmp_path):
    """A vacuous pass is worse than no check: it reads as evidence."""
    a, b = _cache(tmp_path, "a"), _cache(tmp_path, "b")
    with pytest.raises(AssertionError, match="compared nothing"):
        assert_caches_agree(a, b, ["not_a_cube.nc"], [ENCODERS[0]], verbose=False)
