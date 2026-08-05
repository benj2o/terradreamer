"""Phase 1.3 fold generators: every mode either yields leakage-free folds or
refuses loudly. Built on a synthetic manifest with KNOWN cube / tile / year /
bbox membership, so every leakage assertion is checked against an independent
computation, not against the code under test."""

import numpy as np
import pandas as pd
import pytest

from probes.cv import (
    LOCO,
    LeakageError,
    SingleTileError,
    SingleYearError,
    crossed_folds,
    cube_folds,
    filter_clear,
    folds,
    join_embeddings,
    leave_one_cube_out,
    spatial_block_folds,
    temporal_folds,
    tile_folds,
    year_folds,
)


def _manifest(cubes, frames_per_cube=6, clear=0.8):
    """Synthetic manifest. `cubes` is a list of dicts with cube_id, tile,
    year(s), bbox. Frames are spread inside each year at 10-day steps."""
    rows = []
    for spec in cubes:
        years = spec["years"] if "years" in spec else [spec["year"]]
        i = 0
        for y in years:
            for f in range(frames_per_cube):
                ts = np.datetime64(f"{y}-04-01") + np.timedelta64(10 * f, "D")
                rows.append({
                    "cube_id": spec["cube_id"],
                    "tile": spec["tile"],
                    "year": y,
                    "timestamp": ts,
                    "original_axis_index": i,
                    "pixel_bbox": spec["bbox"],
                    "clear_frac": spec.get("clear", clear),
                })
                i += 1
    return pd.DataFrame(rows)


def _multi(n_tiles=3, cubes_per_tile=4, years=(2018, 2019, 2020)):
    """Multi-tile, multi-year, single-year-per-cube (like the train split)."""
    cubes = []
    for t in range(n_tiles):
        for c in range(cubes_per_tile):
            cubes.append({
                "cube_id": f"T{t}_cube{c}.nc",
                "tile": f"32U{'NPQ'[t]}U",
                "year": years[(t * cubes_per_tile + c) % len(years)],
                "bbox": (t * 2000 + c * 300, t * 2000 + c * 300 + 128,
                         c * 400, c * 400 + 128),
            })
    return _manifest(cubes)


def _single_tile_single_year(n_cubes=8):
    """Like the 20-cube extreme-split prototype: one tile, one year."""
    return _manifest([{
        "cube_id": f"32UNU_cube{c}.nc", "tile": "32UNU", "year": 2018,
        "bbox": (c * 500, c * 500 + 128, (c % 3) * 700, (c % 3) * 700 + 128),
    } for c in range(n_cubes)])


def _seasonal(n_cubes=4):
    """Like the seasonal split: each cube spans years, per-row year is the
    timestamp's year (the CORRECT manifest for that split)."""
    return _manifest([{
        "cube_id": f"S_cube{c}.nc", "tile": f"33T{'AB'[c % 2]}N",
        "years": [2018, 2019, 2020], "bbox": (c * 400, c * 400 + 128, 0, 128),
    } for c in range(n_cubes)])


def _cubes_of(df, idx):
    return set(df["cube_id"].to_numpy()[idx].tolist())


def _tiles_of(df, idx):
    return set(df["tile"].to_numpy()[idx].tolist())


def _years_of(df, idx):
    ts = np.asarray(df["timestamp"].to_numpy(), dtype="datetime64[ns]")
    return set((ts[idx].astype("datetime64[Y]").astype(int) + 1970).tolist())


# --- no cube on both sides, in EVERY runnable mode -------------------------

@pytest.mark.parametrize("make_folds", [
    lambda df: cube_folds(df, k=3, verbose=False),
    lambda df: leave_one_cube_out(df, verbose=False),
    lambda df: crossed_folds(df, k=3, verbose=False),
    lambda df: tile_folds(df, k=3, verbose=False),
    lambda df: spatial_block_folds(df, k=3, verbose=False),
], ids=["cube", "loco", "crossed", "tile", "spatial_block"])
def test_no_cube_on_both_sides_in_any_mode(make_folds):
    df = _multi()
    n = 0
    for tr, te in make_folds(df):
        assert not (_cubes_of(df, tr) & _cubes_of(df, te))
        assert tr.size and te.size
        assert not np.intersect1d(tr, te).size
        n += 1
    assert n >= 2


def _staggered(n_early=3, n_late=3, n_straddle=1):
    """Staggered time windows like the real tile's 16 windows: early cubes,
    late cubes, and straddlers whose frames sit on both sides of any cutoff
    between the two groups."""
    rows = []

    def add(cube_id, bbox_row, t0, n_frames):
        for f in range(n_frames):
            ts = np.datetime64(t0) + np.timedelta64(10 * f, "D")
            rows.append({
                "cube_id": cube_id, "tile": "32UNU", "year": 2018,
                "timestamp": ts, "original_axis_index": f,
                "pixel_bbox": (bbox_row, bbox_row + 128, 0, 128),
                "clear_frac": 0.8,
            })

    for c in range(n_early):
        add(f"early{c}.nc", c * 500, "2018-04-01", 4)          # Apr 01 - May 01
    for c in range(n_late):
        add(f"late{c}.nc", 3000 + c * 500, "2018-06-10", 4)     # Jun 10 - Jul 10
    for c in range(n_straddle):
        add(f"straddle{c}.nc", 6000 + c * 500, "2018-04-20", 6)  # Apr 20 - Jun 09
    return pd.DataFrame(rows)


_TEMPORAL_CUTOFF = np.datetime64("2018-05-20")


def test_no_cube_on_both_sides_in_temporal_mode():
    df = _staggered()
    (tr, te), = temporal_folds(df, _TEMPORAL_CUTOFF, verbose=False)
    assert not (_cubes_of(df, tr) & _cubes_of(df, te))


# --- mode-specific guarantees ----------------------------------------------

def test_cube_folds_partition_and_default_k():
    df = _single_tile_single_year()
    seen = []
    for tr, te in cube_folds(df, k=5, verbose=False):
        seen.append(te)
    assert len(seen) == 5
    all_test = np.sort(np.concatenate(seen))
    assert all_test.tolist() == list(range(len(df))), \
        "every row must be tested exactly once across cube folds"


def test_leave_one_cube_out_holds_exactly_one_cube():
    df = _single_tile_single_year(n_cubes=6)
    folds_ = list(cube_folds(df, k=LOCO, verbose=False))
    assert len(folds_) == 6
    for tr, te in folds_:
        assert len(_cubes_of(df, te)) == 1
        assert len(_cubes_of(df, tr)) == 5


def test_crossed_leaves_cube_AND_year_unseen():
    df = _multi()
    n = 0
    for tr, te in crossed_folds(df, k=3, verbose=False):
        assert not (_cubes_of(df, tr) & _cubes_of(df, te))
        assert not (_years_of(df, tr) & _years_of(df, te))
        n += 1
    assert n >= 1


def test_crossed_works_on_a_correct_seasonal_manifest():
    df = _seasonal()
    for tr, te in crossed_folds(df, k=3, verbose=False):
        assert not (_cubes_of(df, tr) & _cubes_of(df, te))
        assert not (_years_of(df, tr) & _years_of(df, te))


def test_no_tile_on_both_sides_in_tile_and_spatial_block_modes():
    df = _multi()
    for gen in (tile_folds(df, k=3, verbose=False),
                spatial_block_folds(df, k=3, verbose=False)):
        for tr, te in gen:
            assert not (_tiles_of(df, tr) & _tiles_of(df, te))


def test_spatial_block_on_one_tile_holds_out_whole_blocks():
    df = _single_tile_single_year(n_cubes=9)
    folds_ = list(spatial_block_folds(df, k=3, verbose=False))
    assert len(folds_) == 3
    tested = set()
    for tr, te in folds_:
        assert not (_cubes_of(df, tr) & _cubes_of(df, te))
        tested |= _cubes_of(df, te)
    assert tested == set(df.cube_id), "every cube must be tested once"


def test_spatial_block_is_deterministic():
    df = _single_tile_single_year(n_cubes=9)
    a = [(tr.tolist(), te.tolist())
         for tr, te in spatial_block_folds(df, k=3, verbose=False)]
    b = [(tr.tolist(), te.tolist())
         for tr, te in spatial_block_folds(df, k=3, verbose=False)]
    assert a == b


def test_temporal_is_strictly_before_and_after():
    df = _staggered()
    (tr, te), = temporal_folds(df, _TEMPORAL_CUTOFF, verbose=False)
    ts = np.asarray(df["timestamp"].to_numpy(), dtype="datetime64[ns]")
    assert (ts[tr] < _TEMPORAL_CUTOFF).all()
    assert (ts[te] > _TEMPORAL_CUTOFF).all()


def test_temporal_drops_wrong_side_frames_not_reassigns_them():
    df = _staggered()
    (tr, te), = temporal_folds(df, _TEMPORAL_CUTOFF, verbose=False)
    # The straddling cube goes whole to its majority side (here: train, 3 of
    # its 6 frames are pre-cutoff vs 2 after, 1 dropped either way) and its
    # wrong-side frames are DROPPED, not smuggled into test.
    used = np.concatenate([tr, te])
    assert used.size < len(df), "straddling cubes must lose frames"
    straddle_rows = np.flatnonzero((df.cube_id == "straddle0.nc").to_numpy())
    assert not (set(straddle_rows) & set(te)), "straddler leaked into test"
    dropped = set(straddle_rows) - set(tr)
    assert dropped, "the straddler's post-cutoff frames must be dropped"


def test_temporal_raises_when_a_side_is_empty():
    df = _staggered()
    with pytest.raises(AssertionError, match="no test frames"):
        list(temporal_folds(df, "2019-01-01", verbose=False))
    with pytest.raises(AssertionError, match="no training frames"):
        list(temporal_folds(df, "2017-01-01", verbose=False))


# --- the three raise paths --------------------------------------------------

def test_year_mode_raises_on_single_year_with_actionable_message():
    df = _single_tile_single_year()
    with pytest.raises(SingleYearError) as e:
        list(year_folds(df, verbose=False))
    msg = str(e.value)
    assert "cube" in msg and "crossed" in msg, "must name the fallbacks"
    assert "random split" in msg, "must warn against the tempting wrong fallback"


def test_crossed_mode_raises_on_single_year_naming_cube_mode():
    df = _single_tile_single_year()
    with pytest.raises(SingleYearError) as e:
        list(crossed_folds(df, verbose=False))
    msg = str(e.value)
    assert "'cube' mode" in msg
    assert "random split" in msg


def test_tile_mode_raises_on_single_tile_naming_spatial_block():
    df = _single_tile_single_year()
    with pytest.raises(SingleTileError) as e:
        list(tile_folds(df, verbose=False))
    msg = str(e.value)
    assert "spatial_block" in msg
    assert "DEFERRED TO SCALE-UP" in msg


# --- the same-cube gate, deliberately provoked ------------------------------

def test_year_mode_on_seasonal_manifest_fires_the_same_cube_gate():
    """One cube spans 2018-2020, so year grouping MUST split within it -- the
    seasonal collision. The gate inside the splitter fires and names crossed
    mode as the resolution."""
    df = _seasonal()
    with pytest.raises(LeakageError, match="crossed"):
        list(year_folds(df, k=3, verbose=False))


def test_year_mode_rejects_a_lying_year_column():
    """build_manifest currently derives year from the cube id; on a seasonal
    cube that disagrees with the timestamps and year-aware modes must refuse
    rather than split on the wrong label."""
    df = _seasonal()
    df["year"] = 2018  # the id-derived (wrong) label
    with pytest.raises(AssertionError, match="build_manifest"):
        list(year_folds(df, k=3, verbose=False))


# --- manifest integrity -----------------------------------------------------

def test_duplicate_cube_timestamp_rows_fail_loudly():
    df = _single_tile_single_year()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(LeakageError, match="duplicate"):
        list(cube_folds(df, verbose=False))


def test_clear_frac_filter_removes_exactly_the_right_rows():
    df = _single_tile_single_year(n_cubes=6)
    rng = np.random.default_rng(0)
    df["clear_frac"] = rng.uniform(0.51, 1.0, size=len(df))
    expected = np.flatnonzero(df["clear_frac"].to_numpy() > 0.7)
    kept = filter_clear(df, min_clear=0.7, verbose=False)
    assert kept.tolist() == expected.tolist()
    # and the folds only ever use surviving rows
    for tr, te in cube_folds(df, k=3, min_clear=0.7, verbose=False):
        assert set(tr) | set(te) <= set(expected.tolist())


def test_clear_frac_filter_refuses_to_loosen_below_encoding_threshold():
    df = _single_tile_single_year()
    with pytest.raises(AssertionError, match="upward"):
        filter_clear(df, min_clear=0.3, verbose=False)


# --- dispatcher -------------------------------------------------------------

def test_folds_dispatcher_routes_every_mode():
    df = _multi()
    assert len(list(folds(df, "cube", k=3, verbose=False))) == 3
    assert len(list(folds(df, "tile", k=3, verbose=False))) == 3
    with pytest.raises(AssertionError, match="cutoff"):
        folds(df, "temporal", verbose=False)
    with pytest.raises(AssertionError, match="not in"):
        folds(df, "random", verbose=False)


# --- the join contract ------------------------------------------------------

def _fake_encoded(df, cube_id, D=7, with_wsd=True):
    from encoders.pipeline import EncodedCube
    sub = df[df.cube_id == cube_id]
    T = len(sub)
    return EncodedCube(
        embeddings=np.zeros((T, D), dtype=np.float32),
        timestamps=np.asarray(sub["timestamp"].to_numpy(), dtype="datetime64[ns]"),
        clear_frac=sub["clear_frac"].to_numpy().astype(float),
        kept_idx=sub["original_axis_index"].to_numpy().astype(int),
        encoder="fake",
        cube=cube_id,
        grid=np.zeros((T, 16, D), dtype=np.float16),
        grid_clear_frac=np.full((T, 16), 0.8, dtype=np.float32),
        variants={},
        window_span_days=np.zeros(T, dtype=np.float32) if with_wsd else None,
    )


def test_join_embeddings_asserts_the_contract_and_carries_wsd():
    df = _single_tile_single_year(n_cubes=3)
    ec = _fake_encoded(df, "32UNU_cube1.nc")
    out = join_embeddings(df, ec, verbose=False)
    sub_idx = np.flatnonzero((df.cube_id == "32UNU_cube1.nc").to_numpy())
    assert out["manifest_idx"].tolist() == sub_idx.tolist()
    assert out["window_span_days"].shape == (len(sub_idx),)
    assert out["embeddings"].shape[0] == len(sub_idx)


def test_join_embeddings_rejects_kept_idx_mismatch():
    df = _single_tile_single_year(n_cubes=3)
    ec = _fake_encoded(df, "32UNU_cube1.nc")
    ec = ec._replace(kept_idx=ec.kept_idx + 1)
    with pytest.raises(AssertionError, match="JOIN CONTRACT"):
        join_embeddings(df, ec, verbose=False)


def test_join_embeddings_refuses_to_drop_window_span_days():
    df = _single_tile_single_year(n_cubes=3)
    ec = _fake_encoded(df, "32UNU_cube1.nc", with_wsd=False)
    with pytest.raises(AssertionError, match="window_span_days"):
        join_embeddings(df, ec, verbose=False)
