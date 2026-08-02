"""The two cross-year quantities must refuse to run on a single-year subset,
loudly, rather than silently degenerating."""

import numpy as np
import pandas as pd
import pytest

from data.climatology import SingleYearError, ndvi_climatology
from probes.cv import assert_multi_year, cube_years, year_groups


def _id(start="2018-03-09", end="2018-08-05", r0=0, r1=128):
    return f"32UNU_{start}_{end}_{r0}_{r1}_0_128_16_96_56_136.nc"


# --- probes/cv.py ---------------------------------------------------------

def test_cube_years_parses_the_window_start():
    assert cube_years([_id(), _id("2019-04-01", "2019-09-01")]).tolist() == [2018, 2019]


def test_cube_years_handles_full_paths():
    assert cube_years(["/a/b/" + _id()]).tolist() == [2018]


def test_year_grouped_cv_raises_on_the_current_single_year_subset():
    paths = [_id(r0=i * 200, r1=i * 200 + 128) for i in range(20)]
    with pytest.raises(SingleYearError) as e:
        year_groups(paths)
    msg = str(e.value)
    assert "2018" in msg
    assert "DEFERRED TO SCALE-UP" in msg
    assert "random split" in msg, "must warn against the tempting wrong fallback"


def test_year_grouped_cv_works_once_a_second_year_exists():
    paths = [_id(), _id("2019-04-01", "2019-09-01"), _id("2020-04-01", "2020-09-01")]
    assert year_groups(paths).tolist() == [2018, 2019, 2020]


def test_assert_multi_year_returns_years_when_valid():
    assert assert_multi_year([_id(), _id("2019-04-01", "2019-09-01")]).tolist() == [2018, 2019]


# --- data/climatology.py --------------------------------------------------

def _stack(years, h=2, w=2, per_year=8):
    t, frames = [], []
    rng = np.random.default_rng(0)
    for y in years:
        for k in range(per_year):
            t.append(pd.Timestamp(f"{y}-03-01") + pd.Timedelta(days=15 * k))
            frames.append(rng.uniform(0.2, 0.8, size=(h, w)))
    return np.stack(frames).astype(np.float32), np.array(t, dtype="datetime64[ns]")


def test_climatology_raises_on_single_year():
    stack, ts = _stack([2018])
    with pytest.raises(SingleYearError) as e:
        ndvi_climatology(stack, ts, target_year=2018)
    msg = str(e.value)
    assert "[2018]" in msg
    assert "within-year mean" in msg, "must name the substitute it refuses to make"


def test_climatology_excludes_the_target_year():
    stack, ts = _stack([2018, 2019, 2020])
    out = ndvi_climatology(stack, ts, target_year=2019)
    assert out.shape == (366, 2, 2)
    assert out.dtype == np.float32


def test_climatology_is_indexed_by_day_of_year():
    stack, ts = _stack([2018, 2019])
    out = ndvi_climatology(stack, ts, target_year=2020)  # neither year excluded
    assert out.shape[0] == 366, "must cover every day-of-year including leap day"


def test_climatology_shape_matches_input_grid():
    stack, ts = _stack([2018, 2019], h=5, w=3)
    assert ndvi_climatology(stack, ts, target_year=2018).shape == (366, 5, 3)


def test_climatology_rejects_wrong_rank():
    stack, ts = _stack([2018, 2019])
    with pytest.raises(AssertionError):
        ndvi_climatology(stack[:, 0], ts, target_year=2018)
