"""Real spherical harmonics on the GIAMIP grid, in the convention of the protocol.

GIAMIP asks for optional Stokes coefficients ``Clm`` and ``Slm`` of the geoid
height change (protocol v4.0 section 7). The basis is the fully normalised
real basis with 4 pi normalisation, the geodesy convention:

    Y_lm(theta, phi) = Pbar_l|m|(cos theta) * cos(m phi)      for m >= 0
                       Pbar_l|m|(cos theta) * sin(|m| phi)    for m < 0

    Pbar_lm = sqrt((2 - delta_m0) (2l + 1) (l - m)! / (l + m)!) P_lm

    int_sphere Y_lm^2 dOmega = 4 pi

where theta is the colatitude, so ``cos theta = sin(latitude)``. The
protocol writes the factor sqrt(2) of m > 0 next to the trigonometric
function and the rest in a normalisation factor N_lm; here both are folded
into ``Pbar_lm``, which is the same product.

The associated Legendre functions ``P_lm`` here carry **no** Condon-Shortley
phase (-1)^m. The protocol does not say so in words, but the checker's
``variables.md`` names the convention "fully_normalized_4pi", which is the
geodesy convention without the phase. ``scipy.special.lpmv`` and
``scipy.special.sph_harm_y`` both include the phase, so a value from either
must be multiplied by (-1)^m before it is compared with a value from here.

The coefficients are dimensionless, at the reference radius R (6371000 m in
GIAMIP, taken from the case file by the caller):

    f(theta, phi) = R * sum_lm (C_lm cos(m phi) + S_lm sin(m phi)) Pbar_lm(cos theta)

and so, from the orthogonality of the basis,

    C_lm = 1 / (4 pi R) int_sphere f Pbar_lm cos(m phi) dOmega

with ``sin`` for S_lm.

The integral is evaluated by quadrature on the GIAMIP Gauss-Legendre grid
(`giamip.grids`): the 257 nodes in sin(latitude) integrate a polynomial of
degree up to 513 exactly, and the 513 equally spaced longitudes integrate a
trigonometric polynomial of order up to 512 exactly. For a coefficient of
degree l <= 96, the integrand ``f Pbar_lm cos(m phi)`` is therefore integrated
exactly when the field has degree ``deg(f) + 96 <= 513`` and order
``m_f + 96 < 513``. A field sampled on this grid resolves degree 256 at most,
so the analysis to degree 96 is exact to round-off for any field on the grid.

The arrays are stored with the first index the degree and the second the
order, ``(degree, order)``, with zeros where order > degree, as the checker's
``variables.md`` asks. ``S[:, 0]`` is zero because sin(0 phi) vanishes.
"""

from __future__ import annotations

import numpy as np

from giamip import grids

# The highest degree GIAMIP asks for in Clm and Slm (data request: 97 x 97,
# degrees 0 to 96). The grid could resolve 256; the request stops at 96.
LMAX_STOKES = 96


def normalized_legendre(lmax: int, x: np.ndarray) -> np.ndarray:
    """Fully normalised (4 pi) associated Legendre functions, no Condon-Shortley phase.

    The values are computed with the standard forward-column recursion of
    geodesy, which uses no factorials, so nothing overflows at high degree
    (the unnormalised P_96^96 is about 1e174 and is never formed):

    * sectoral seed: ``Pbar_00 = 1``, ``Pbar_11 = sqrt(3) s`` with
      ``s = sqrt(1 - x^2)`` (the sin of the colatitude). The factor sqrt(2)
      of m > 0 enters here once, because sqrt(3) = sqrt(2) * sqrt(3/2);
    * ``Pbar_mm = sqrt((2m + 1) / (2m)) s Pbar_(m-1)(m-1)`` for m >= 2;
    * ``Pbar_(m+1)m = sqrt(2m + 3) x Pbar_mm``;
    * ``Pbar_lm = a_lm x Pbar_(l-1)m - b_lm Pbar_(l-2)m`` for l >= m + 2, with
      ``a_lm = sqrt((2l - 1)(2l + 1) / ((l - m)(l + m)))`` and
      ``b_lm = sqrt((2l + 1)(l + m - 1)(l - m - 1) / ((l - m)(l + m)(2l - 3)))``.

    Every step multiplies by a factor of order one, so the recursion is
    stable to degree 96. The smallest value it produces on the GIAMIP grid is
    the sectoral function of degree 96 at the outermost node, about
    ``(cos 89.46 deg)^96 ~ 1e-194``, far from the double-precision underflow.

    Parameters
    ----------
    lmax
        Highest degree.
    x
        cos(colatitude) = sin(latitude) at each point, in [-1, 1].

    Returns
    -------
    numpy.ndarray
        ``(lmax + 1, lmax + 1, len(x))``, indexed ``[l, m, point]``, with
        zeros for m > l.
    """
    x = np.asarray(x, dtype=float)
    # sin of the colatitude; non-negative for colatitude in [0, pi].
    s = np.sqrt(np.clip(1.0 - x * x, 0.0, None))
    p = np.zeros((lmax + 1, lmax + 1, x.size))
    p[0, 0] = 1.0
    if lmax == 0:
        return p
    # Sectoral terms Pbar_mm.
    p[1, 1] = np.sqrt(3.0) * s
    for m in range(2, lmax + 1):
        p[m, m] = np.sqrt((2.0 * m + 1.0) / (2.0 * m)) * s * p[m - 1, m - 1]
    # The first term below the diagonal, Pbar_(m+1)m, for every order.
    for m in range(0, lmax):
        p[m + 1, m] = np.sqrt(2.0 * m + 3.0) * x * p[m, m]
    # Two-term recursion in degree for each order.
    for m in range(0, lmax + 1):
        for l in range(m + 2, lmax + 1):
            a = np.sqrt((2.0 * l - 1.0) * (2.0 * l + 1.0) / ((l - m) * (l + m)))
            b = np.sqrt((2.0 * l + 1.0) * (l + m - 1.0) * (l - m - 1.0)
                        / ((l - m) * (l + m) * (2.0 * l - 3.0)))
            p[l, m] = a * x * p[l - 1, m] - b * p[l - 2, m]
    return p


def _grid_tables(lmax: int):
    """Legendre and trigonometric tables on the GIAMIP grid.

    The Legendre table is evaluated at the Gauss-Legendre nodes in the order
    of `giamip.grids.latitudes`, south first, so the row order of a field and
    of the table come from one array and cannot disagree. A north-first field
    would flip the sign of every coefficient with odd l + m, which is what
    `tests/test_harmonics.py` checks against.
    """
    mu, weights = np.polynomial.legendre.leggauss(grids.NLAT)
    lam = np.radians(grids.longitudes())
    orders = np.arange(lmax + 1)
    cos_table = np.cos(np.outer(orders, lam))    # (m, lon)
    sin_table = np.sin(np.outer(orders, lam))
    return normalized_legendre(lmax, mu), weights, cos_table, sin_table


def _check_shape(field: np.ndarray) -> None:
    """Raise unless the field is (lat, lon) on the GIAMIP grid.

    A transposed field has the same number of values and gives wrong
    coefficients with no error, so the shape is checked first.
    """
    if field.shape != (grids.NLAT, grids.NLON):
        raise ValueError(
            f"field has shape {field.shape}, expected ({grids.NLAT}, {grids.NLON}), "
            "ordered (lat, lon) with latitude south to north."
        )


def analyse(field: np.ndarray, radius: float, lmax: int = LMAX_STOKES):
    """Stokes coefficients of a field on the GIAMIP grid.

    Quadrature of ``C_lm = 1 / (4 pi R) int f Pbar_lm cos(m phi) dOmega``
    (and sin for S_lm) on the Gauss-Legendre grid: the solid angle element is
    ``d(sin lat) dphi``, so the weight of node (j, i) is
    ``w_j * 2 pi / 513``, the same weight as `giamip.grids.cell_areas`
    divided by R^2.

    Parameters
    ----------
    field
        ``(257, 513)``, latitude south to north, in metres for a geoid height.
    radius
        Reference radius in metres. GIAMIP uses 6371000 m; pass the case
        file's ``earth_radius_m``. The coefficients are the field divided by
        this radius, so they are dimensionless.
    lmax
        Highest degree. Defaults to 96, the degree of the data request.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        ``C`` and ``S``, each ``(lmax + 1, lmax + 1)`` float64, indexed
        ``[degree, order]``, zero where order > degree, and ``S[:, 0] = 0``.
    """
    field = np.asarray(field, dtype=float)
    _check_shape(field)
    p, weights, cos_table, sin_table = _grid_tables(lmax)
    dlambda = 2.0 * np.pi / grids.NLON
    # Longitude sums first: (lat, lon) @ (lon, m) -> (lat, m), weighted by the
    # longitude step. These are the cosine and sine transforms of each row.
    a = field @ cos_table.T * dlambda
    b = field @ sin_table.T * dlambda
    # Latitude quadrature with the Gauss weights: sum_j w_j Pbar_lm(mu_j) a_jm.
    c = np.einsum("lmj,j,jm->lm", p, weights, a) / (4.0 * np.pi * radius)
    s = np.einsum("lmj,j,jm->lm", p, weights, b) / (4.0 * np.pi * radius)
    # Zero the entries that are not harmonics, so round-off from the empty
    # upper triangle of the table (exact zeros already) and sin(0 phi) (zero
    # to round-off) does not appear in the file.
    lower = np.tril(np.ones((lmax + 1, lmax + 1), dtype=bool))
    c = np.where(lower, c, 0.0)
    s = np.where(lower, s, 0.0)
    s[:, 0] = 0.0
    return c, s


def synthesise(c: np.ndarray, s: np.ndarray, radius: float) -> np.ndarray:
    """The field on the GIAMIP grid from its Stokes coefficients; the inverse of `analyse`.

    ``f = R * sum_lm (C_lm cos(m phi) + S_lm sin(m phi)) Pbar_lm(sin lat)``.

    Parameters
    ----------
    c, s
        ``(lmax + 1, lmax + 1)``, indexed ``[degree, order]``.
    radius
        Reference radius in metres.

    Returns
    -------
    numpy.ndarray
        ``(257, 513)``, latitude south to north, in the units of R.
    """
    c = np.asarray(c, dtype=float)
    s = np.asarray(s, dtype=float)
    lmax = c.shape[0] - 1
    p, _, cos_table, sin_table = _grid_tables(lmax)
    # Sum over degree for each order and latitude: (m, lat).
    pc = np.einsum("lmj,lm->mj", p, c)
    ps = np.einsum("lmj,lm->mj", p, s)
    return radius * (pc.T @ cos_table + ps.T @ sin_table)
