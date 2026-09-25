"""The Gauss-Legendre grid that GIAMIP prescribes, and checks against it.

GIAMIP requires every two-dimensional output field on one global Gaussian grid
of 257 latitudes by 513 longitudes. That is the grid the ice histories and the
bed topographies are delivered on, and the grid the compliance checker tests
submitted output against.

The latitudes are **not** equally spaced. They are the Gauss-Legendre
quadrature nodes in sin(latitude), so the spacing runs from 0.693 degrees at
the pole to 0.699 at the equator. Two consequences follow, and both are easy
to get wrong without any error appearing:

* The area of a cell is the quadrature weight times the longitude step, not
  ``cos(lat) * dlat * dlon``. On a sphere the area element is
  ``a^2 dlambda d(sin phi)``, and the weights integrate ``d(sin phi)`` exactly,
  which is the whole point of choosing this grid. The wrong formula is close
  enough to look plausible and wrong enough to spoil a mass integral.
* A grid built with ``numpy.linspace`` has the right shape and the right
  extent, and is a different grid. It passes a shape check and fails the
  compliance checker.

The grid is generated here and not read from a delivered file, which is what
lets a case be validated for the experiments whose forcing file has not been
delivered. The delivered files agree with the generated values to 3e-14
degrees, which ``tests/test_grids.py`` checks.

Relation to spherical harmonics: this is the SHTOOLS ``GLQ`` grid for a
maximum degree of 256, where ``nlat = lmax + 1`` and ``nlon = 2 * lmax + 1``.
Quadrature on it integrates a spherical harmonic product of degree up to 256
exactly.
"""

from __future__ import annotations

import numpy as np

# The prescribed grid. Fixed by the GIAMIP protocol, section 2.2.1, and by the
# compliance checker, which hard-codes the same two numbers.
NLAT = 257
NLON = 513

# The maximum spherical harmonic degree the grid integrates exactly. Reported
# for the reader; nothing here depends on it.
LMAX = NLAT - 1

# The compliance checker matches a submitted axis against the generated one to
# 1e-3 degrees. Independent implementations of Gauss-Legendre quadrature agree
# to about 1e-12 degrees, while any grid that is not the prescribed one is off
# by 0.1 degrees or more, so the threshold separates the two cases with a wide
# margin. Reproduced here so a case is checked against the same number the
# submission will be checked against.
GRID_ABSTOL = 1e-3


def latitudes(nlat: int = NLAT) -> np.ndarray:
    """Latitudes of the prescribed grid in degrees, ordered south to north.

    The Gauss-Legendre nodes are in sin(latitude), so the latitude of node
    ``i`` is ``arcsin(mu_i)``. The array runs from about -89.465 to +89.465;
    neither pole is a grid point.

    Parameters
    ----------
    nlat
        Number of latitudes. Defaults to the prescribed 257. Another value is
        useful only for testing the construction itself.
    """
    nodes, _ = np.polynomial.legendre.leggauss(nlat)
    return np.degrees(np.arcsin(nodes))


def longitudes(nlon: int = NLON) -> np.ndarray:
    """Longitudes of the prescribed grid in degrees, ordered west to east.

    Equally spaced over ``[0, 360)``. With 513 nodes the step is 0.70175
    degrees and the last node is 359.298, so 360 is not repeated as a second
    copy of 0. A reader that assumes a wrapped duplicate will be one column
    out.
    """
    return np.arange(nlon) * (360.0 / nlon)


def sin_latitude_weights(nlat: int = NLAT) -> np.ndarray:
    """Gauss-Legendre quadrature weights in sin(latitude), ordered with `latitudes`.

    These sum to 2, which is the range of sin(latitude) over the sphere.
    """
    _, weights = np.polynomial.legendre.leggauss(nlat)
    return weights


def cell_areas(radius: float, nlat: int = NLAT, nlon: int = NLON) -> np.ndarray:
    """Area of each grid cell in square metres, shaped ``(nlat, nlon)``.

    The area element on a sphere is ``a^2 dlambda d(sin phi)``. The longitude
    step is uniform, so every cell on a given latitude row has the same area,
    and the row's area is the quadrature weight times ``a^2`` times the
    longitude step in radians. The total is ``4 pi a^2`` to machine precision,
    which ``tests/test_grids.py`` checks.

    Parameters
    ----------
    radius
        Earth radius in metres. Take it from the case file. GIAMIP prescribes
        6371000 m in protocol section 8; the benchmarks use the same value but
        state it separately, and nothing here assumes they agree.

    Returns
    -------
    numpy.ndarray
        Cell areas, ``(nlat, nlon)``. The full two-dimensional array is
        returned even though every row is constant, so that an integral is
        written as ``(field * areas).sum()`` with no broadcasting to reason
        about.
    """
    dlambda = 2.0 * np.pi / nlon
    row_area = radius**2 * dlambda * sin_latitude_weights(nlat)
    return np.repeat(row_area[:, None], nlon, axis=1)


def integrate(field: np.ndarray, radius: float) -> float:
    """Integrate a field over the sphere, in the field's units times m^2.

    A convenience over ``(field * cell_areas(radius)).sum()`` that checks the
    shape first, because a field that arrives transposed has the same number
    of elements and integrates to a wrong answer without complaint.
    """
    if field.shape != (NLAT, NLON):
        raise ValueError(
            f"field has shape {field.shape}, expected ({NLAT}, {NLON}). "
            "GIAMIP orders two-dimensional fields (lat, lon); a transposed "
            "field integrates to a wrong answer with no error."
        )
    return float((field * cell_areas(radius)).sum())


class GridMismatch(ValueError):
    """A file's axes are not the prescribed grid."""


def check(lat: np.ndarray, lon: np.ndarray, *, atol: float = GRID_ABSTOL,
          name: str = "dataset") -> None:
    """Raise `GridMismatch` unless the axes are the prescribed grid.

    Checks the node counts, then the values against the generated grid. The
    second check is what catches an equally spaced axis with the right count
    and the right extent, which is the mistake a shape check misses.

    Parameters
    ----------
    lat, lon
        The axes to check, in degrees.
    atol
        Absolute tolerance in degrees. Defaults to the value the compliance
        checker uses.
    name
        Used in the error message, so a failure names the file.
    """
    problems = []
    if lat.shape != (NLAT,):
        problems.append(f"latitude has shape {lat.shape}, expected ({NLAT},)")
    if lon.shape != (NLON,):
        problems.append(f"longitude has shape {lon.shape}, expected ({NLON},)")
    if problems:
        raise GridMismatch(f"{name}: " + "; ".join(problems))

    lat_err = float(np.max(np.abs(np.asarray(lat, dtype=float) - latitudes())))
    lon_err = float(np.max(np.abs(np.asarray(lon, dtype=float) - longitudes())))
    if lat_err > atol:
        problems.append(
            f"latitude differs from the prescribed grid by up to {lat_err:.3g} "
            f"degrees (tolerance {atol}). An equally spaced axis differs by "
            f"about 0.1 degrees."
        )
    if lon_err > atol:
        problems.append(
            f"longitude differs from the prescribed grid by up to {lon_err:.3g} "
            f"degrees (tolerance {atol})"
        )
    if problems:
        raise GridMismatch(f"{name}: " + "; ".join(problems))
