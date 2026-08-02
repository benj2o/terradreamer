"""Loader smoke tests on a synthetic minicube -- catches wiring bugs before
any Colab/GPU time is spent on real downloads."""

from __future__ import annotations

import numpy as np
import pytest

from data.loader import (
    S2_BANDS,
    cube_ndvi,
    describe_cube,
    iter_cubes,
    load_cube,
    valid_mask_from_codes,
)
from data.ndvi import ndvi


def test_valid_mask_polarity():
    codes = np.array([[0, 1], [2, 4]], dtype=np.uint8)
    m = valid_mask_from_codes(codes)
    print(f"[test] codes {codes.shape} -> valid mask {m.shape} {m.dtype}, clear={m.sum()}")
    assert m.dtype == np.bool_
    assert m.tolist() == [[True, False], [False, False]], "0 == clear, everything else masked"


def test_valid_mask_handles_nan_codes():
    codes = np.array([[0.0, np.nan]], dtype=np.float32)
    assert valid_mask_from_codes(codes).tolist() == [[True, False]]


def test_load_cube_shapes_and_time_axis(synthetic_cube):
    s = load_cube(synthetic_cube)
    T, C, H, W = s.values.shape
    assert C == len(S2_BANDS) == 4
    assert s.mask.shape == (T, H, W)
    assert s.timestamps.shape == (T,)
    assert s.timestamps.dtype == np.dtype("datetime64[ns]")
    assert np.all(np.diff(s.timestamps) > np.timedelta64(0, "ns")), "duplicates not removed"
    assert np.isfinite(s.values).any(axis=(1, 2, 3)).all(), "empty timesteps not dropped"


def test_time_grid_is_irregular(synthetic_cube):
    s = load_cube(synthetic_cube)
    dt = np.diff(s.timestamps) / np.timedelta64(1, "D")
    print(f"[test] dt/days = {dt}")
    assert dt.min() != dt.max(), "loader must not silently resample onto a regular grid"


def test_cube_ndvi_uses_canonical_function(synthetic_cube):
    s = load_cube(synthetic_cube)
    got = cube_ndvi(s)
    want = ndvi(s.values[:, S2_BANDS.index("B8A")], s.values[:, S2_BANDS.index("B04")], s.mask)
    assert got.shape == s.mask.shape
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
    np.testing.assert_allclose(got[~np.isnan(got)], want[~np.isnan(want)], rtol=1e-6)


def test_masked_pixels_are_nan_in_cube_ndvi(synthetic_cube):
    s = load_cube(synthetic_cube)
    nd = cube_ndvi(s)
    assert np.isnan(nd[~s.mask]).all(), "masked pixels leaked into cube NDVI"
    assert np.isfinite(nd[s.mask]).all(), "clear pixels became NaN"


def test_describe_cube_reports_mask_fraction(synthetic_cube, capsys):
    s = load_cube(synthetic_cube)
    stats = describe_cube(s)
    out = capsys.readouterr().out
    assert "CLOUD-MASK" in out
    assert 0.0 < stats["valid_fraction"] < 1.0
    assert np.isclose(stats["valid_fraction"] + stats["masked_fraction"], 1.0)


def test_iter_cubes_yields_triplets(tmp_path):
    from conftest import make_synthetic_cube

    for i in range(3):
        make_synthetic_cube(str(tmp_path / f"mc_{i:02d}.nc"), seed=i)
    samples = list(iter_cubes(str(tmp_path), limit=2))
    assert len(samples) == 2, "limit not respected"
    for values, timestamps, mask, _path, _bands in samples:
        assert values.shape[0] == timestamps.shape[0] == mask.shape[0]


def test_iter_cubes_errors_on_empty_dir(tmp_path):
    with pytest.raises(AssertionError):
        list(iter_cubes(str(tmp_path)))


def test_assert_no_overlap_passes_for_disjoint_cubes(tmp_path):
    from conftest import make_synthetic_cube
    from data.loader import assert_no_overlap
    import xarray as xr

    for i, lat0 in enumerate([48.15, 48.30]):   # far apart
        p = make_synthetic_cube(str(tmp_path / f"mc_{i:02d}.nc"), seed=i)
        ds = xr.load_dataset(p)
        ds = ds.assign_coords(lat=np.linspace(lat0, lat0 - 0.02, ds.sizes["lat"]))
        ds.to_netcdf(p)
    assert len(assert_no_overlap(str(tmp_path))) == 2


def test_assert_no_overlap_catches_cubes_sharing_pixels(tmp_path):
    """Two windows cut from one grid, offset by half a cube, share real rows."""
    from conftest import make_synthetic_cube
    from data.loader import assert_no_overlap
    import xarray as xr

    full = np.linspace(48.16, 48.10, 16)   # one grid both cubes are cut from
    for i, off in enumerate([0, 4]):       # 8-wide windows, so they share 4 rows
        p = make_synthetic_cube(str(tmp_path / f"mc_{i:02d}.nc"), seed=i)
        ds = xr.load_dataset(p)
        ds = ds.assign_coords(lat=full[off:off + ds.sizes["lat"]])
        ds.to_netcdf(p)
    with pytest.raises(AssertionError, match="overlapping cube pair"):
        assert_no_overlap(str(tmp_path))


def test_adjacent_cubes_sharing_no_pixel_are_allowed(tmp_path):
    """Cubes are gridded in UTM but carry lat/lon coords, so two ADJACENT cubes
    have overlapping lat/lon bounding boxes while sharing no pixel. That must
    not be reported as overlap."""
    from conftest import make_synthetic_cube
    from data.loader import assert_no_overlap
    import xarray as xr

    for i, lat0 in enumerate([48.16, 48.14]):
        p = make_synthetic_cube(str(tmp_path / f"mc_{i:02d}.nc"), seed=i)
        ds = xr.load_dataset(p)
        # skewed by a fraction of a pixel, exactly like the real cubes
        ds = ds.assign_coords(
            lat=np.linspace(lat0, lat0 - 0.02, ds.sizes["lat"]) + i * 0.0003)
        ds.to_netcdf(p)
    assert len(assert_no_overlap(str(tmp_path))) == 2
