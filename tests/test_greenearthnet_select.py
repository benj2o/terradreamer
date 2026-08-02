"""Selection must never hand two overlapping cubes to the probe stage."""

import pytest

from data.download_greenearthnet import parse_footprint, select_non_overlapping


def _id(r0, r1, c0, c1, start="2018-03-09", end="2018-08-05"):
    return f"32UNU_{start}_{end}_{r0}_{r1}_{c0}_{c1}_16_96_56_136.nc"


def test_parse_footprint():
    assert parse_footprint(_id(1081, 1209, 3641, 3769)) == (1081, 1209, 3641, 3769)


def test_parse_footprint_handles_full_s3_key():
    key = "earthnet/earthnet2021x/train/32UNU/" + _id(10, 138, 20, 148)
    assert parse_footprint(key) == (10, 138, 20, 148)


def test_disjoint_cubes_all_selected():
    keys = [_id(0, 128, 0, 128), _id(0, 128, 200, 328), _id(300, 428, 0, 128)]
    assert select_non_overlapping(keys, 3) == keys


def test_overlapping_cube_is_skipped():
    keys = [_id(0, 128, 0, 128), _id(64, 192, 64, 192), _id(500, 628, 500, 628)]
    got = select_non_overlapping(keys, 3)
    assert len(got) == 2, "the half-overlapping cube must be dropped"
    assert keys[1] not in got


def test_same_patch_different_dates_is_still_overlap():
    keys = [_id(0, 128, 0, 128, end="2018-08-05"),
            _id(0, 128, 0, 128, start="2018-05-28", end="2018-10-24")]
    assert len(select_non_overlapping(keys, 2)) == 1


def test_limit_respected():
    keys = [_id(0, 128, i * 200, i * 200 + 128) for i in range(10)]
    assert len(select_non_overlapping(keys, 4)) == 4


def test_raises_when_nothing_selectable():
    with pytest.raises(AssertionError):
        select_non_overlapping([], 5)


def test_windows_are_spread_not_clustered():
    """20 cubes must not all come from the first time window."""
    keys = []
    for w, start in enumerate(["2018-03-09", "2018-04-18", "2018-05-28"]):
        for i in range(10):
            keys.append(_id(w * 5000 + i * 200, w * 5000 + i * 200 + 128, 0, 128,
                            start=start))
    got = select_non_overlapping(keys, 6)
    windows = {k.split("_")[1] for k in got}
    assert len(windows) == 3, f"expected all 3 windows represented, got {windows}"


def test_spread_can_be_disabled():
    keys = [_id(i * 200, i * 200 + 128, 0, 128,
                start="2018-03-09" if i < 3 else "2018-05-28") for i in range(6)]
    got = select_non_overlapping(keys, 3, spread_windows=False)
    assert got == keys[:3]


def test_adjacent_cubes_are_rejected_by_default():
    """Edge-to-edge cubes share no pixel but are the same field."""
    keys = [_id(0, 128, 0, 128), _id(128, 256, 0, 128)]
    assert len(select_non_overlapping(keys, 2)) == 1


def test_gap_of_64px_is_enough():
    keys = [_id(0, 128, 0, 128), _id(192, 320, 0, 128)]
    assert len(select_non_overlapping(keys, 2)) == 2


def test_min_gap_can_be_disabled():
    keys = [_id(0, 128, 0, 128), _id(128, 256, 0, 128)]
    assert len(select_non_overlapping(keys, 2, min_gap_px=0)) == 2
