"""THE single definition of leakage-safe evaluation splits.

Any number produced outside this module does not exist. Do not write an ad-hoc
train/test split anywhere else in the repo; import from here.

INCOMPLETE, Phase 1.3. Only the grouping keys and their guards are settled so
far. The fold generators, the nested-CV wrapper and the metric aggregation land
in Phase 1.3, and nothing in this file should be used to produce a reported
number before then.

Two grouping modes, because there are two ways this dataset leaks:

* spatial: cubes near each other share weather, phenology and often the same
  fields. Operative mode for Phase 1.3. Cubes are already non-overlapping by
  construction (data.download_greenearthnet.select_non_overlapping enforces a
  64 px gap, data.loader.assert_no_overlap verifies it on disk), so the
  remaining question is block size, not overlap.
* year: the same patch in two years is nearly the same sample. Implemented,
  but it RAISES on the current subset, which is single-year.
"""

from __future__ import annotations

import os

import numpy as np

from data.climatology import SingleYearError

__all__ = ["cube_years", "assert_multi_year", "year_groups", "SingleYearError"]


def cube_years(paths) -> np.ndarray:
    """Acquisition year per cube, parsed from the GreenEarthNet cube id.

    Ids look like 32UNU_2018-03-09_2018-08-05_1081_1209_3641_3769_16_96_56_136
    where field 1 is the window start.
    """
    years = []
    for p in paths:
        stem = os.path.basename(str(p))
        stem = stem[:-3] if stem.endswith(".nc") else stem
        parts = stem.split("_")
        assert len(parts) > 2, f"cannot parse a year from {stem!r}"
        years.append(int(parts[1][:4]))
    out = np.asarray(years, dtype=int)
    assert out.shape == (len(list(paths)),) or out.size == len(years)
    return out


def assert_multi_year(paths) -> np.ndarray:
    """Raise unless the manifest spans more than one year.

    The year-leakage check is deferred to scale-up: every cube in tile 32UNU is
    from 2018. Failing loudly here is the point. A year-grouped split that
    silently degenerates to a single group would report a leakage-free number
    that had never been tested for year leakage at all.
    """
    years = cube_years(paths)
    unique = sorted(set(years.tolist()))
    if len(unique) < 2:
        raise SingleYearError(
            f"year-grouped CV needs at least 2 years, the manifest has {unique}. "
            f"{len(years)} cubes, all from {unique[0]}. Every cube in tile 32UNU "
            "is from 2018, so this is expected on the current subset and the "
            "year-leakage check is DEFERRED TO SCALE-UP. Use spatial grouping "
            "for Phase 1.3. Do not fall back to a random split: it would put the "
            "same season on both sides."
        )
    return years


def year_groups(paths) -> np.ndarray:
    """Group labels for year-grouped CV, one per cube.

    Raises SingleYearError on a single-year manifest. See assert_multi_year.
    """
    years = assert_multi_year(paths)
    print(f"[cv] year groups: {len(years)} cubes over {sorted(set(years.tolist()))}")
    return years
