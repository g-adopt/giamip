"""The spherical harmonics of `giamip.harmonics` follow protocol v4.0 section 7.

The reference values here are built independently of the module, from
``scipy.special.sph_harm_y`` (orthonormal, complex, with the Condon-Shortley
phase). The protocol basis is

    Pbar_lm(mu) = sqrt(2 - delta_m0) * sqrt(4 pi) * (-1)^m * Re(sph_harm_y(l, m, arccos mu, 0))

so a test that synthesises a field from this formula and analyses it with the
module catches the three conventions a same-library round trip cannot see:

* a reversed latitude axis: P_lm(-x) = (-1)^(l+m) P_lm(x), so every
  coefficient with odd l + m changes sign;
* a Condon-Shortley phase: every coefficient with odd m changes sign;
* a wrong normalisation (orthonormal, Schmidt) or a lost sqrt(2).

``sph_harm_y`` is used instead of ``lpmv`` because ``lpmv`` overflows to
inf for 77 of the (l, m) pairs to degree 96 (it returns the unnormalised
function, about 1e174 at (96, 96)).

Tolerances. The quadrature on the 257-node grid reproduces 4 pi to 2e-12
relative at worst (the round-off of the Gauss nodes and weights on a degree
182 integrand), so the normalisation test allows 1e-10. The recovery of unit
coefficients is accurate to a few 1e-13, so it allows 1e-11. Both are ten
orders of magnitude below the errors the tests catch (a sign flip is 2, a
lost sqrt(2) is 0.41).
"""

from __future__ import annotations

import numpy as np
import pytest

from giamip import grids, harmonics

R = 6371000.0


def _sph_harm_y():
    """`scipy.special.sph_harm_y`, which needs scipy 1.15 or later."""
    special = pytest.importorskip("scipy.special")
    if not hasattr(special, "sph_harm_y"):
        pytest.skip("scipy.special.sph_harm_y needs scipy >= 1.15")
    return special.sph_harm_y


def _pbar_reference(l: int, m: int, mu: np.ndarray) -> np.ndarray:
    """The protocol's Pbar_lm from scipy, with the Condon-Shortley phase removed."""
    ylm = _sph_harm_y()(l, m, np.arccos(mu), 0.0)
    factor = np.sqrt(4.0 * np.pi) * (1.0 if m == 0 else np.sqrt(2.0))
    return factor * (-1.0) ** m * np.real(ylm)


def _grid():
    """South-first sin(latitude) nodes and longitudes in radians, from the grid module."""
    mu = np.sin(np.radians(grids.latitudes()))
    lam = np.radians(grids.longitudes())
    return mu, lam


def _synthesise_by_hand(coefficients: dict, mu: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """A field from {(l, m, 'c' or 's'): value}, in units of R, written from the protocol formula."""
    field = np.zeros((mu.size, lam.size))
    for (l, m, kind), value in coefficients.items():
        trig = np.cos(m * lam) if kind == "c" else np.sin(m * lam)
        field += value * np.outer(_pbar_reference(l, m, mu), trig)
    return R * field


# Odd l + m: (1,0), (2,1), (5,2); odd m: (2,1), (3,3); cos and sin of (2,1)
# differ so a cos/sin swap is visible. High-degree entries test the far end.
COEFFICIENTS = {
    (1, 0, "c"): 0.7, (2, 1, "c"): -0.3, (2, 1, "s"): 0.45, (3, 3, "c"): 0.11,
    (3, 3, "s"): -0.25, (5, 2, "c"): 0.9, (0, 0, "c"): 0.2,
    (96, 95, "c"): 0.05, (95, 1, "s"): -0.07,
}


def _expected_arrays():
    c = np.zeros((97, 97))
    s = np.zeros((97, 97))
    for (l, m, kind), value in COEFFICIENTS.items():
        (c if kind == "c" else s)[l, m] = value
    return c, s


def test_hand_synthesis_is_recovered():
    """A field built from the protocol formula gives back its coefficients (R3)."""
    mu, lam = _grid()
    field = _synthesise_by_hand(COEFFICIENTS, mu, lam)
    c, s = harmonics.analyse(field, R)
    c_expected, s_expected = _expected_arrays()
    np.testing.assert_allclose(c, c_expected, rtol=0, atol=1e-11)
    np.testing.assert_allclose(s, s_expected, rtol=0, atol=1e-11)


def test_a_north_first_field_flips_the_odd_degree_plus_order_coefficients():
    """The first test can fail: the same field with its rows reversed is a different answer."""
    mu, lam = _grid()
    field = _synthesise_by_hand(COEFFICIENTS, mu, lam)[::-1]
    c, _ = harmonics.analyse(field, R)
    assert c[1, 0] == pytest.approx(-0.7, abs=1e-11)
    assert c[2, 1] == pytest.approx(0.3, abs=1e-11)
    assert c[5, 2] == pytest.approx(-0.9, abs=1e-11)
    assert c[3, 3] == pytest.approx(0.11, abs=1e-11)   # even l + m keeps its sign


def test_a_condon_shortley_field_flips_the_odd_order_coefficients():
    """A field written with the phase of lpmv gives the odd-m coefficients with the other sign."""
    mu, lam = _grid()
    with_phase = {k: v * (-1.0) ** k[1] for k, v in COEFFICIENTS.items()}
    c, _ = harmonics.analyse(_synthesise_by_hand(with_phase, mu, lam), R)
    assert c[2, 1] == pytest.approx(0.3, abs=1e-11)
    assert c[1, 0] == pytest.approx(0.7, abs=1e-11)


# (l, m, kind) with sin only for m > 0, since sin(0 phi) is not a basis function.
NORMALISATION_CASES = (
    [(l, m, "cos") for l, m in [(0, 0), (1, 0), (1, 1), (2, 1), (7, 4), (50, 0),
                               (91, 0), (96, 1), (96, 48), (96, 96)]]
    + [(l, m, "sin") for l, m in [(1, 1), (2, 1), (7, 4), (96, 1), (96, 48), (96, 96)]]
)


@pytest.mark.parametrize("l,m,kind", NORMALISATION_CASES)
def test_normalisation_is_four_pi(l, m, kind):
    """The quadrature of Y_lm^2 over the sphere is 4 pi, the protocol normalisation."""
    mu, lam = _grid()
    p = harmonics.normalized_legendre(l, mu)[l, m]
    trig = np.cos(m * lam) if kind == "cos" else np.sin(m * lam)
    y = np.outer(p, trig)
    total = float((y**2 * grids.cell_areas(R)).sum()) / R**2
    assert total == pytest.approx(4.0 * np.pi, rel=1e-10)


def test_recursion_matches_scipy_at_every_degree_and_order():
    """Every Pbar_lm to degree 96 agrees with sph_harm_y to 1e-12 of its own maximum."""
    mu, _ = _grid()
    p = harmonics.normalized_legendre(96, mu)
    worst = 0.0
    for l in range(97):
        for m in range(l + 1):
            ref = _pbar_reference(l, m, mu)
            scale = float(np.max(np.abs(ref)))
            worst = max(worst, float(np.max(np.abs(p[l, m] - ref))) / scale)
    assert worst < 1e-12, worst


def test_layout_and_round_trip():
    """(degree, order) 97 x 97, zero above the diagonal, S[:, 0] = 0, and synthesise inverts analyse."""
    c_expected, s_expected = _expected_arrays()
    field = harmonics.synthesise(c_expected, s_expected, R)
    c, s = harmonics.analyse(field, R)
    assert c.shape == s.shape == (97, 97)
    assert not np.any(np.triu(c, 1)) and not np.any(np.triu(s, 1))
    assert not np.any(s[:, 0])
    np.testing.assert_allclose(c, c_expected, atol=1e-11)
    np.testing.assert_allclose(s, s_expected, atol=1e-11)


def test_a_transposed_field_is_refused():
    with pytest.raises(ValueError, match="shape"):
        harmonics.analyse(np.zeros((grids.NLON, grids.NLAT)), R)
