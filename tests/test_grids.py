"""The prescribed grid is generated correctly and matches every delivered file.

Two things are checked. First, that the construction is self-consistent: the
weights sum to the range of sin(latitude), and the cell areas sum to the area
of the sphere. Second, that the generated grid is the grid GIAMIP actually
delivered its inputs on, which is the claim that lets a case be validated for
an experiment whose forcing file has not arrived.
"""

import numpy as np
import pytest

from gia_mip import grids

# Every delivered file, relative to data/original/. Marked `data` because they
# are not in the repository.
DELIVERED = [
    "giamip/ice-history/iceHistory-ICE-6G_D.nc",
    "giamip/ice-history/iceHistory-ICE-7G_NA.nc",
    "giamip/ice-history/iceHistory-PaleoMIST_1a.nc",
    "giamip/ice-history/iceHistory-GLAC3c_profile1.nc",
    "giamip/ice-history/iceHistory-GLAC3c_profile2.nc",
    "giamip/ice-history/iceHistory-GLAC3c_profile3.nc",
    "giamip/initial-topography/bedtopo_Bedmap2ANT_BedMach5GRL_GEBCO_GLQ256.nc",
    "giamip/initial-topography/bedtopo_Bedmap3ANT_BedMach5GRL_GEBCO_GLQ256.nc",
    "giamip/initial-topography/bedtopo_BedMach3ANT_BedMach5GRL_GEBCO_GLQ256.nc",
]


def test_shapes():
    assert grids.latitudes().shape == (grids.NLAT,)
    assert grids.longitudes().shape == (grids.NLON,)


def test_latitudes_are_not_equally_spaced():
    """The defining property. An equally spaced axis is a different grid."""
    spacing = np.diff(grids.latitudes())
    assert spacing.max() - spacing.min() > 1e-3, (
        "latitudes look equally spaced, which means leggauss was not used"
    )


def test_longitudes_do_not_repeat_the_meridian():
    """360 is not present as a second copy of 0."""
    lon = grids.longitudes()
    assert lon[0] == 0.0
    assert lon[-1] < 360.0
    assert np.isclose(lon[-1] + (lon[1] - lon[0]), 360.0)


def test_weights_sum_to_the_range_of_sine_latitude():
    assert np.isclose(grids.sin_latitude_weights().sum(), 2.0, rtol=0, atol=1e-12)


def test_cell_areas_sum_to_the_area_of_the_sphere():
    """The check that the area formula is right.

    Using cos(lat) * dlat * dlon instead of the quadrature weight fails here,
    which is the point of the test.
    """
    radius = 6371000.0
    total = grids.cell_areas(radius).sum()
    assert np.isclose(total, 4 * np.pi * radius**2, rtol=1e-12)


def test_integrate_rejects_a_transposed_field():
    """A transposed field has the same element count and a wrong integral."""
    field = np.ones((grids.NLON, grids.NLAT))
    with pytest.raises(ValueError, match="transposed"):
        grids.integrate(field, 6371000.0)


def test_check_rejects_an_equally_spaced_latitude_axis():
    """The mistake a shape check misses: right count, right extent, wrong grid."""
    lat = np.linspace(-89.46490769578145, 89.46490769578145, grids.NLAT)
    with pytest.raises(grids.GridMismatch, match="equally spaced"):
        grids.check(lat, grids.longitudes())


def test_check_accepts_the_generated_grid():
    grids.check(grids.latitudes(), grids.longitudes())


@pytest.mark.data
@pytest.mark.parametrize("relative_path", DELIVERED)
def test_delivered_files_are_on_the_prescribed_grid(data_root, relative_path):
    """Every file GIAMIP delivered sits on the grid this module generates."""
    from gia_mip import _netcdf
    path = data_root / "original" / relative_path
    if not path.exists():
        pytest.skip(f"{relative_path} not fetched")
    # Through h5netcdf, which applies no mask. See AGENTS.md: the ice files
    # declare a fill value of zero, and a masking read hides every ice-free cell.
    with _netcdf.open_read(path) as ds:
        grids.check(np.asarray(ds["lat"][...]), np.asarray(ds["lon"][...]), name=relative_path)


@pytest.mark.data
def test_generated_latitudes_match_a_delivered_file_to_machine_precision(data_root):
    """Tighter than `check`'s tolerance, to record how close the agreement is.

    `check` allows 1e-3 degrees because that is what the compliance checker
    allows. The delivered float64 axes agree far more closely than that, and
    this test records the real figure so a later drift is visible.
    """
    from gia_mip import _netcdf
    path = data_root / "original" / "giamip/ice-history/iceHistory-ICE-6G_D.nc"
    if not path.exists():
        pytest.skip("ICE-6G_D not fetched")
    with _netcdf.open_read(path) as ds:
        error = np.max(np.abs(np.asarray(ds["lat"][...]) - grids.latitudes()))
    assert error < 1e-12, f"latitudes differ by {error:g} degrees"
