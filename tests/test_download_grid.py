"""The download grid must be pinned by the FULL grid, never by --n."""

from data.download_minicubes import RESOLUTION, XY_SHAPE, cube_centres


def test_preflight_cube_is_cube_00_of_the_real_grid():
    assert cube_centres(1)[0] == cube_centres(20)[0], (
        "--n 1 must download the first cube of the 20-grid, not a re-centred one"
    )


def test_prefix_property_holds_for_every_n():
    full = cube_centres(20)
    for n in range(1, 21):
        assert cube_centres(n) == full[:n], f"--n {n} is not a prefix of the grid"


def test_no_two_centres_overlap():
    width_km = XY_SHAPE[0] * RESOLUTION / 1000.0
    centres = cube_centres(20)
    assert len(set(centres)) == 20
    import math
    for i in range(len(centres)):
        for j in range(i + 1, len(centres)):
            dlon_km = abs(centres[i][0] - centres[j][0]) * 111.320 * math.cos(math.radians(48.15))
            dlat_km = abs(centres[i][1] - centres[j][1]) * 111.320
            assert dlon_km >= width_km or dlat_km >= width_km, (
                f"cubes {i} and {j} overlap: dlon {dlon_km:.2f} km, dlat {dlat_km:.2f} km"
            )
