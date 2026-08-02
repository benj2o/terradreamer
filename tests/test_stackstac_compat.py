"""The stackstac shim forces rescale=False. These tests pin down exactly when
that is allowed, so it can never silently discard a real scale/offset."""

from __future__ import annotations

import pytest

from data.stackstac_compat import _assert_identity_scaling, apply


def _item(raster_bands=None):
    """A STAC item as a plain dict, the shape stackstac receives."""
    asset = {"href": "s3://x/B04.tif"}
    if raster_bands is not None:
        asset["raster:bands"] = raster_bands
    return {"id": "S2A_TEST", "assets": {"B04": asset, "B8A": dict(asset)}}


def test_no_raster_bands_is_allowed():
    """Planetary Computer's sentinel-2-l2a: no raster:bands -> scale 1, offset 0."""
    _assert_identity_scaling([_item()], ["B04", "B8A"])


def test_explicit_identity_scaling_is_allowed():
    _assert_identity_scaling([_item([{"scale": 1, "offset": 0}])], ["B04"])


def test_non_identity_scale_raises():
    with pytest.raises(RuntimeError, match="scale=0.0001"):
        _assert_identity_scaling([_item([{"scale": 0.0001, "offset": 0}])], ["B04"])


def test_non_zero_offset_raises():
    """The S2 baseline >= 04.00 BOA_ADD_OFFSET case. Must never be dropped."""
    with pytest.raises(RuntimeError, match="offset=-1000"):
        _assert_identity_scaling([_item([{"scale": 1, "offset": -1000}])], ["B04"])


def test_pystac_style_item_is_understood():
    class Asset:
        def __init__(self, extra):
            self.extra_fields = extra

    class Item:
        assets = {"B04": Asset({"raster:bands": [{"scale": 2, "offset": 0}]})}

    with pytest.raises(RuntimeError, match="scale=2"):
        _assert_identity_scaling([Item()], ["B04"])


def test_guard_tolerates_missing_assets_and_odd_items():
    _assert_identity_scaling([_item()], ["NOT_AN_ASSET"])
    _assert_identity_scaling([_item()], None)
    _assert_identity_scaling(object(), ["B04"])  # not iterable -> no crash


@pytest.fixture
def fake_stackstac(monkeypatch):
    """A stand-in `stackstac` module that records the kwargs it is called with,
    so the shim is testable without installing the real (heavy) package."""
    import sys
    import types

    mod = types.ModuleType("stackstac")
    mod.calls = []

    def stack(items, **kw):
        mod.calls.append(kw)
        return "stacked"

    mod.stack = stack
    monkeypatch.setitem(sys.modules, "stackstac", mod)
    return mod


def test_shim_injects_both_kwargs(fake_stackstac):
    import numpy as np

    apply()
    apply()  # second call must not double-wrap
    assert fake_stackstac.stack("items", assets=["B04"], dtype="float32") == "stacked"
    kw = fake_stackstac.calls[-1]

    assert kw["rescale"] is False, "shim did not inject rescale=False"
    fv = kw["fill_value"]
    assert np.isnan(fv), "fill_value must still be NaN"
    assert fv.dtype == np.float32, f"fill_value must carry the output dtype, got {fv.dtype}"
    # The exact condition stackstac 0.5.1 to_dask.py:39 checks.
    assert np.can_cast(type(fv), np.float32), "fill_value would still trip the guard"


def test_explicit_kwargs_are_respected(fake_stackstac):
    apply()
    fake_stackstac.stack("items", assets=["B04"], dtype="float32",
                         rescale=True, fill_value=0)
    kw = fake_stackstac.calls[-1]
    assert kw["rescale"] is True, "explicit rescale must be respected"
    assert kw["fill_value"] == 0, "explicit fill_value must be respected"


def test_fill_value_matches_requested_dtype(fake_stackstac):
    """float64 stacks must get a float64 NaN, not a float32 one."""
    import numpy as np

    apply()
    fake_stackstac.stack("items", assets=["B04"], dtype="float64")
    assert fake_stackstac.calls[-1]["fill_value"].dtype == np.float64


def test_non_float_dtype_keeps_plain_nan(fake_stackstac):
    """uint16 stacks: dtype.type('nan') would be nonsense, so leave np.nan and
    let stackstac raise its own (correct) complaint."""
    import numpy as np

    apply()
    fake_stackstac.stack("items", assets=["B04"], dtype="uint16")
    assert fake_stackstac.calls[-1]["fill_value"] is np.nan


def test_guard_still_fires_through_the_shim(fake_stackstac):
    """A real scale/offset must raise even when going through the patched stack."""
    apply()
    with pytest.raises(RuntimeError, match="offset=-1000"):
        fake_stackstac.stack([_item([{"scale": 1, "offset": -1000}])],
                             assets=["B04"], dtype="float32")
