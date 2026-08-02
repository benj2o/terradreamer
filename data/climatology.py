"""Per-pixel NDVI climatology, adopted from GreenEarthNet.

The definition is taken verbatim in behaviour from `model_pixelwise/
climatology.py` in vitusbenson/greenearthnet, because it is the only version
whose numbers are comparable with the published baselines:

    For a target year, pool NDVI from ALL OTHER years, keep pixels where
    s2_dlmask == 0 and s2_SCL is in (1, 2, 4, 5, 6, 7), linearly interpolate
    the remaining gaps along time, group by day-of-year and average across
    years, then smooth with a 30-day centred rolling mean over a wrap-padded
    year and drop the 30 padding days at each end.

Do not invent a substitute. A within-year "climatology" is a different
quantity, it is not comparable with anything published, and on a single-season
cube it is close to a smoothed copy of the signal it would be a baseline for.
`ndvi_climatology` therefore raises `SingleYearError` rather than guessing.

NOT COMPUTABLE ON THE CURRENT SUBSET. Every cube in tile 32UNU is from 2018,
and the definition needs at least one year that is not the target year. This
module becomes usable at scale-up, when cubes from more than one year exist.

Two deliberate differences from the reference implementation:

* NDVI comes from data.ndvi.ndvi, the project's single definition, which
  returns NaN where |B8A + B04| < 1e-12. GreenEarthNet instead adds 1e-8 to
  the denominator. The two agree everywhere the denominator is not near zero.
* The mask is built by data.loader.greenearthnet_valid_mask, so mask polarity
  is still decided in exactly one place.
"""

from __future__ import annotations

import numpy as np

__all__ = ["SingleYearError", "ndvi_climatology", "CLIMATOLOGY_SMOOTH_DAYS"]

CLIMATOLOGY_SMOOTH_DAYS = 30


class SingleYearError(RuntimeError):
    """Raised when a cross-year quantity is asked of a single-year subset."""


def ndvi_climatology(ndvi_stack, timestamps, target_year: int,
                     smooth_days: int = CLIMATOLOGY_SMOOTH_DAYS):
    """Day-of-year NDVI climatology from every year except `target_year`.

    Parameters
    ----------
    ndvi_stack : (T, H, W) float array
        Masked NDVI, i.e. the output of data.ndvi.ndvi. NaN where not clear.
    timestamps : (T,) datetime64
        The irregular acquisition times that go with `ndvi_stack`.
    target_year : int
        The year being predicted. It is excluded from the pool, which is what
        makes the climatology an honest baseline rather than a leak.
    smooth_days : int
        Width of the centred rolling mean, 30 in the reference implementation.

    Returns
    -------
    (366, H, W) float32
        Climatological NDVI indexed by day-of-year 1..366.

    Raises
    ------
    SingleYearError
        If no year other than `target_year` is present.
    """
    import pandas as pd
    import xarray as xr

    ndvi_stack = np.asarray(ndvi_stack)
    timestamps = np.asarray(timestamps, dtype="datetime64[ns]")
    assert ndvi_stack.ndim == 3, f"expected (T, H, W), got {ndvi_stack.shape}"
    assert timestamps.shape[0] == ndvi_stack.shape[0], (
        f"{timestamps.shape[0]} timestamps vs {ndvi_stack.shape[0]} frames"
    )

    years = pd.DatetimeIndex(timestamps).year
    other = years != target_year
    present = sorted(set(years.tolist()))
    if not other.any():
        raise SingleYearError(
            f"climatology for target year {target_year} needs at least one other "
            f"year, but the input contains only {present}. Every cube in tile "
            "32UNU is from 2018, so this is expected on the current subset. "
            "Do not substitute a within-year mean: it is not the published "
            "quantity and it leaks the signal it is meant to baseline. "
            "Deferred to scale-up."
        )

    da = xr.DataArray(
        ndvi_stack[other].astype(np.float32),
        dims=("time", "y", "x"),
        coords={"time": timestamps[other]},
    )
    print(f"[climatology] pooling {int(other.sum())} frames from "
          f"{sorted(set(years[other].tolist()))}, excluding {target_year}")

    da = da.interpolate_na("time", method="linear")
    doy = da.groupby("time.dayofyear").mean("time")
    doy = doy.reindex(dayofyear=np.arange(1, 367))

    # Wrap-pad a whole year so the rolling mean is continuous across 31 Dec,
    # then drop the padding. Late-December climatology otherwise leans on
    # nothing but December.
    pad = smooth_days
    wrapped = xr.concat([doy[-pad:], doy, doy[:pad]], dim="dayofyear")
    smoothed = wrapped.rolling(dayofyear=smooth_days, center=True,
                               min_periods=1).mean()
    out = smoothed[pad:-pad].values.astype(np.float32)

    assert out.shape == (366,) + ndvi_stack.shape[1:], (
        f"expected {(366,) + ndvi_stack.shape[1:]}, got {out.shape}"
    )
    print(f"[climatology] out {out.shape} float32 | "
          f"finite {np.isfinite(out).mean():.3f}")
    return out
