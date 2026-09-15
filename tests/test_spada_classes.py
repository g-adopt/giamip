"""The classes a solver uses for Spada et al. (2011): case set-up, result, score.

None of these tests needs the archive. The result and scoring tests build
small synthetic results in memory and in a temporary directory.

The interface under test
------------------------
``gia_mip.benchmarks.inputs.SpadaCase(case)``
    ``case`` is ``"t02-01"`` or ``"spada2011-t02-01"``. Attributes:
    ``case_id``, ``earth_model`` (the model dict), ``ice_density``,
    ``water_density``, ``earth_radius_m``, ``loads`` (tuple of load names),
    ``load_centre`` (colatitude, longitude in degrees), ``epochs_kyr`` (ends
    with ``math.inf``), ``harmonic_degrees`` (min, max), ``outputs``.
    Methods: ``earth_layers()``, ``load_mass(load)``,
    ``surface_density(load, colatitude_deg)``,
    ``legendre_coefficients(load, degrees)``,
    ``near_field_colatitudes_deg()``, ``global_colatitudes_deg()``.

``gia_mip.benchmarks.results.SpadaResult(case, member)``
    ``add_love_numbers(quantity, degrees, elastic, fluid, residues)`` and
    ``love_numbers(quantity)`` with fields ``degrees``, ``elastic``,
    ``fluid``, ``residues`` (degrees by modes);
    ``add_relaxation_spectrum(degrees, s_per_kyr)`` and
    ``relaxation_spectrum()`` with fields ``degrees``, ``s_per_kyr``;
    ``add_spatial(load, quantity, epoch_kyr, colatitude_deg, values)``,
    ``spatial(load, quantity, epoch_kyr)`` returning a ``Curve``, and
    ``epochs(load, quantity)``; ``write(path)``, ``read(path, member, case)``,
    ``members(path)``, ``reference_path(case)``.

``gia_mip.benchmarks.metrics.spada_score(result, references)``
    ``references`` maps a member code to a ``SpadaResult``. One row per
    comparison, with ``kind``, ``load``, ``quantity``, ``epoch_kyr``,
    ``interval``, ``reference`` and the entries of ``difference_metrics``.

Units: colatitude in degrees, epochs in kyr, U, V, N in metres, rates in
metres per kyr, s_j in 1/kyr, densities in kg/m3, surface density and its
Legendre coefficients in kg/m2.
"""

import copy
import math

import numpy as np
import pytest

from gia_mip import cases

# The load mass TABOO prints in the header of every T02-01 cap and disc file,
# for a 1500 m cap and a 1000 m disc of 10 degrees radius, ice of 931 kg/m3,
# on a sphere of radius 6371 km. The analytic integrals give the same number
# for both loads, because (2/3) 1500 = 1000.
TABOO_LOAD_MASS_KG = 3.607171340900778e18


def _inputs():
    """The inputs module, imported when a test runs so collection never fails."""
    from gia_mip.benchmarks import inputs
    return inputs


def _results():
    from gia_mip.benchmarks import results
    return results


def _metrics():
    from gia_mip.benchmarks import metrics
    return metrics


def _legendre(n_max: int, x: np.ndarray) -> np.ndarray:
    """P_0 .. P_n_max at the points x, by the three-term recurrence."""
    p = np.zeros((n_max + 1, x.size))
    p[0] = 1.0
    if n_max >= 1:
        p[1] = x
    for n in range(1, n_max):
        p[n + 1] = ((2 * n + 1) * x * p[n] - n * p[n - 1]) / (n + 1)
    return p


# ---------------------------------------------------------------- SpadaCase

def test_a_case_accepts_the_short_and_the_full_id():
    inputs = _inputs()
    assert inputs.SpadaCase("t02-01").case_id == "spada2011-t02-01"
    assert inputs.SpadaCase("spada2011-t02-01").case_id == "spada2011-t02-01"


@pytest.mark.parametrize("case,model", [("t01-06", "vss96")]
                         + [(c, "m3-l70-v01") for c in ("t01-01", "t01-02", "t01-03", "t01-04",
                                                        "t01-05", "t01-07", "t02-01", "t02-02",
                                                        "t02-03")])
def test_the_earth_model_is_the_one_the_case_names(case, model):
    """Test 6/1 alone uses VSS96."""
    assert _inputs().SpadaCase(case).earth_model["id"] == model


def test_the_earth_layers_come_from_m3_l70_v01():
    """Five layers, surface first, an elastic lithosphere of infinite viscosity."""
    layers = _inputs().SpadaCase("t02-01").earth_layers()
    model = cases.earth_model("m3-l70-v01")
    assert len(layers["top_radius_m"]) == len(model["layers"])
    assert layers["top_radius_m"][0] == pytest.approx(6.371e6)
    assert math.isinf(layers["viscosity_pa_s"][0])


def test_the_densities_are_the_spada_values():
    """931 kg/m3 for ice and 1000 kg/m3 for water, table 2 of the paper."""
    case = _inputs().SpadaCase("t02-01")
    assert (case.ice_density, case.water_density) == (931.0, 1000.0)


def test_the_densities_are_read_from_the_load_table(monkeypatch):
    """A module constant would keep 931 after the table changes; the table must win.

    The load table returned by gia_mip.cases.loads is replaced by a copy with
    the GIAMIP values. SpadaCase must obtain its table through that function,
    as MartinecCase does.
    """
    original = cases.loads("spada2011")
    changed = copy.deepcopy(original)
    changed["constants"]["ice_density_kg_m3"] = 917
    changed["constants"]["water_density_kg_m3"] = 1025
    monkeypatch.setattr(cases, "loads",
                        lambda collection: changed if collection == "spada2011" else original)
    case = _inputs().SpadaCase("t02-01")
    assert (case.ice_density, case.water_density) == (917.0, 1025.0)


def test_the_spatial_cases_have_three_loads_and_the_wavenumber_cases_none():
    inputs = _inputs()
    assert tuple(inputs.SpadaCase("t02-01").loads) == ("cap", "disc", "point")
    assert tuple(inputs.SpadaCase("t01-02").loads) == ()


@pytest.mark.parametrize("case,centre", [("t02-01", (0.0, 0.0)), ("t02-02", (0.0, 0.0)),
                                         ("t02-03", (25.0, 75.0))])
def test_the_load_centre_is_the_pole_except_in_test_3_2(case, centre):
    assert tuple(_inputs().SpadaCase(case).load_centre) == centre


@pytest.mark.parametrize("load", ["cap", "disc"])
def test_the_cap_and_disc_masses_are_the_taboo_header_mass(load):
    """Both loads carry 3.607171340900778e18 kg.

    The cap mass is (4/3) pi a^2 rho h (1 - cos alpha) for the square-root
    profile, and the disc mass 2 pi a^2 rho h (1 - cos alpha). TABOO prints
    sixteen digits and both expressions are exact up to rounding, so 1e-9 is
    a safe tolerance. A linear cap profile, or ice at 917 kg/m3, misses by
    more than one per cent.
    """
    mass = _inputs().SpadaCase("t02-01").load_mass(load)
    assert mass == pytest.approx(TABOO_LOAD_MASS_KG, rel=1e-9)


def test_the_point_load_mass_is_1e18_kg():
    assert _inputs().SpadaCase("t02-01").load_mass("point") == pytest.approx(1e18, rel=1e-12)


def test_the_cap_surface_density_is_the_square_root_profile():
    """sigma = rho h sqrt((cos theta - cos alpha) / (1 - cos alpha)), zero beyond alpha.

    With rho = 931 kg/m3 and h = 1500 m the value at the centre is
    1.3965e6 kg/m2. Exact arithmetic, so the tolerance is rounding only.
    """
    case = _inputs().SpadaCase("t02-01")
    theta = np.array([0.0, 5.0, 9.99, 10.0, 12.0, 90.0])
    c_alpha = np.cos(np.radians(10.0))
    ratio = np.clip((np.cos(np.radians(theta)) - c_alpha) / (1.0 - c_alpha), 0.0, None)
    expected = 931.0 * 1500.0 * np.sqrt(ratio)
    np.testing.assert_allclose(case.surface_density("cap", theta), expected, rtol=1e-12, atol=1e-6)


def test_the_disc_surface_density_is_uniform_inside_alpha():
    case = _inputs().SpadaCase("t02-01")
    values = case.surface_density("disc", np.array([0.0, 5.0, 9.5, 10.5, 90.0]))
    np.testing.assert_allclose(values, [931e3, 931e3, 931e3, 0.0, 0.0], rtol=1e-12)


def test_the_point_load_has_no_surface_density():
    """A delta function has no value at a point, so asking for one raises."""
    with pytest.raises(ValueError):
        _inputs().SpadaCase("t02-01").surface_density("point", np.array([0.0, 1.0]))


@pytest.mark.parametrize("load", ["cap", "disc"])
def test_the_legendre_coefficients_are_the_projection_of_the_surface_density(load):
    """sigma_n = (2n + 1)/2 integral of sigma(theta) P_n(cos theta) over cos theta.

    The integral runs over x = cos theta from cos alpha to 1, where the load
    sits. The substitution x = cos alpha + (1 - cos alpha) u^2 turns the
    square root of the cap into u, so both integrands are polynomials in u of
    degree at most 2 n_max + 2. Gauss-Legendre quadrature with 400 nodes
    integrates them exactly up to rounding. The tolerance, 1e-8 relative with
    a floor of 1e-4 kg/m2 against a degree-1 value near 1e4 kg/m2, is
    therefore a statement about rounding. The linear cap profile gives
    coefficients that differ by tens of per cent.
    """
    case = _inputs().SpadaCase("t02-01")
    degrees = np.arange(1, 129)
    c_alpha = np.cos(np.radians(10.0))
    nodes, weights = np.polynomial.legendre.leggauss(400)
    u = 0.5 * (nodes + 1.0)
    w = 0.5 * weights
    x = c_alpha + (1.0 - c_alpha) * u ** 2
    jacobian = 2.0 * (1.0 - c_alpha) * u
    theta = np.degrees(np.arccos(np.clip(x, -1.0, 1.0)))
    sigma = case.surface_density(load, theta)
    p = _legendre(int(degrees[-1]), x)
    expected = (2 * degrees + 1) / 2.0 * (p[degrees] * sigma * jacobian * w).sum(axis=1)
    got = case.legendre_coefficients(load, degrees)
    np.testing.assert_allclose(got, expected, rtol=1e-8, atol=1e-4)


def test_the_point_load_coefficients_grow_with_2n_plus_1():
    """sigma_n = m (2n + 1) / (4 pi a^2): 5.88e3 kg/m2 at degree 2 for 1e18 kg."""
    case = _inputs().SpadaCase("t02-01")
    degrees = np.array([1, 2, 64, 128])
    expected = 1e18 * (2 * degrees + 1) / (4 * np.pi * 6.371e6 ** 2)
    np.testing.assert_allclose(case.legendre_coefficients("point", degrees), expected, rtol=1e-12)


def test_the_degree_0_cap_coefficient_carries_the_load_mass():
    """4 pi a^2 sigma_0 is the total mass, the check TABOO prints in its header."""
    case = _inputs().SpadaCase("t02-01")
    sigma_0 = case.legendre_coefficients("cap", np.array([0]))[0]
    assert 4 * np.pi * 6.371e6 ** 2 * sigma_0 == pytest.approx(TABOO_LOAD_MASS_KG, rel=1e-9)


@pytest.mark.parametrize("case", ["t02-01", "t02-02"])
def test_the_epochs_end_at_infinity(case):
    """0, 1, 2, 5, 10 kyr and the fluid limit. Infinity is a float, not the string."""
    epochs = tuple(_inputs().SpadaCase(case).epochs_kyr)
    assert epochs == (0.0, 1.0, 2.0, 5.0, 10.0, math.inf)


@pytest.mark.parametrize("case", ["t02-01", "t02-02"])
def test_the_near_field_grid_is_0_to_25_degrees_at_a_quarter_degree(case):
    """2.5 alpha with alpha = 10 degrees, 101 points. The T02-02 case file has no grid field."""
    grid = _inputs().SpadaCase(case).near_field_colatitudes_deg()
    np.testing.assert_allclose(grid, np.arange(101) * 0.25, atol=1e-12)


def test_the_global_grid_is_0_to_180_degrees_at_one_degree():
    grid = _inputs().SpadaCase("t02-01").global_colatitudes_deg()
    np.testing.assert_allclose(grid, np.arange(181.0), atol=1e-12)


@pytest.mark.parametrize("case,degrees", [("t02-01", (2, 128)), ("t01-02", (2, 256)),
                                          ("t01-04", (2, 2)), ("t01-07", (1, 1))])
def test_the_degree_range_comes_from_the_case_file(case, degrees):
    assert tuple(_inputs().SpadaCase(case).harmonic_degrees) == degrees


def test_the_outputs_are_those_of_the_case_file():
    assert list(_inputs().SpadaCase("t02-02").outputs) == ["U_rate", "V_rate", "N_rate"]


# -------------------------------------------------------------- SpadaResult

def _love(n_degrees: int = 3, modes: int = 9, seed: int = 0):
    """Arbitrary Love numbers of a plausible shape: degrees 2.., nine modes."""
    rng = np.random.default_rng(seed)
    degrees = np.arange(2, 2 + n_degrees)
    return (degrees, rng.normal(size=n_degrees), rng.normal(size=n_degrees),
            rng.normal(size=(n_degrees, modes)))


def test_love_numbers_survive_a_write_and_a_read(tmp_path):
    """NetCDF float64 round trip, so the arrays come back bit for bit."""
    results = _results()
    degrees, elastic, fluid, residues = _love()
    run = results.SpadaResult("t01-02", "radopt")
    run.add_love_numbers("k", degrees, elastic, fluid, residues)
    path = run.write(tmp_path / "spada2011-t01-02.nc")
    back = results.SpadaResult.read(path, "radopt").love_numbers("k")
    np.testing.assert_array_equal(back.degrees, degrees)
    np.testing.assert_array_equal(back.elastic, elastic)
    np.testing.assert_array_equal(back.fluid, fluid)
    np.testing.assert_array_equal(back.residues, residues)


def test_a_relaxation_spectrum_survives_a_write_and_a_read(tmp_path):
    results = _results()
    degrees = np.arange(2, 6)
    s = -np.linspace(1e-5, 3.0, 4 * 9).reshape(4, 9)
    run = results.SpadaResult("t01-01", "radopt")
    run.add_relaxation_spectrum(degrees, s)
    path = run.write(tmp_path / "spada2011-t01-01.nc")
    back = results.SpadaResult.read(path, "radopt").relaxation_spectrum()
    np.testing.assert_array_equal(back.degrees, degrees)
    np.testing.assert_array_equal(back.s_per_kyr, s)


def test_the_fluid_limit_and_1000_kyr_stay_two_epochs(tmp_path):
    """U at infinity and U at 1000 kyr differ by 7 m at the pole; both must survive.

    A layout that stored infinity as a large number, or matched epochs loosely,
    would merge them.
    """
    results = _results()
    theta = np.arange(0.0, 181.0)
    run = results.SpadaResult("t02-01", "radopt")
    run.add_spatial("cap", "U", 1000.0, theta, np.full(theta.size, -388.11))
    run.add_spatial("cap", "U", math.inf, theta, np.full(theta.size, -395.01))
    path = run.write(tmp_path / "spada2011-t02-01.nc")
    back = results.SpadaResult.read(path, "radopt")
    assert [float(e) for e in back.epochs("cap", "U")] == [1000.0, math.inf]
    assert back.spatial("cap", "U", math.inf).values[0] == -395.01
    assert back.spatial("cap", "U", 1000.0).values[0] == -388.11


def test_a_spatial_curve_is_held_on_an_ascending_colatitude():
    """A descending input is sorted, values with it."""
    run = _results().SpadaResult("t02-01", "radopt")
    run.add_spatial("disc", "V", 0.0, [2.0, 1.0, 0.0], [0.2, 0.1, 0.0])
    curve = run.spatial("disc", "V", 0.0)
    np.testing.assert_array_equal(curve.axis, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(curve.values, [0.0, 0.1, 0.2])


@pytest.mark.parametrize("load,quantity", [("ring", "U"), ("cap", "S")])
def test_an_unknown_load_or_quantity_is_refused(load, quantity):
    """Loads are cap, disc and point; quantities U, V, N and their rates."""
    run = _results().SpadaResult("t02-01", "radopt")
    with pytest.raises(ValueError):
        run.add_spatial(load, quantity, 0.0, [0.0, 1.0], [0.0, 0.0])


def test_two_members_share_one_file(tmp_path):
    """A reference file holds every contributor; members() lists them."""
    results = _results()
    path = tmp_path / "spada2011-t02-01.nc"
    for member in ("GS", "VK"):
        run = results.SpadaResult("t02-01", member)
        run.add_spatial("cap", "U", 0.0, [0.0, 1.0], [-27.7, -27.5])
        run.write(path)
    assert sorted(results.SpadaResult.members(path)) == ["GS", "VK"]


def test_the_reference_path_is_in_the_converted_spada_directory():
    path = _results().SpadaResult.reference_path("t02-01")
    assert path.parts[-3:] == ("converted", "spada2011", "spada2011-t02-01.nc")


def test_reading_a_result_turns_netcdf_masking_off(tmp_path, monkeypatch):
    """AGENTS.md: every reader calls set_auto_mask(False), and a test shows it.

    netCDF4.Dataset is replaced by a subclass that records the call. A masked
    read of a zero fill value returns the same numbers here, so only the call
    itself shows that masking is off.
    """
    import netCDF4

    results = _results()
    run = results.SpadaResult("t02-01", "radopt")
    run.add_spatial("cap", "V", 0.0, [0.0, 1.0], [0.0, 0.3])
    path = run.write(tmp_path / "spada2011-t02-01.nc")

    calls = []

    class Recording(netCDF4.Dataset):
        def set_auto_mask(self, value):
            calls.append(value)
            return super().set_auto_mask(value)

    monkeypatch.setattr(netCDF4, "Dataset", Recording)
    results.SpadaResult.read(path, "radopt")
    assert False in calls, calls


# ------------------------------------------------------------------ scoring

def _spatial_result(member: str, offset: float = 0.0, epoch: float = 1.0,
                    case: str = "t02-01"):
    """A cap U profile on the GS grid, 0 to 180 degrees at 0.25, shifted by `offset` m."""
    theta = np.arange(721) * 0.25
    run = _results().SpadaResult(case, member)
    run.add_spatial("cap", "U", epoch, theta, -95.0 * np.exp(-theta / 10.0) + offset)
    return run


def test_a_result_scored_against_itself_has_zero_difference():
    rows = _metrics().spada_score(_spatial_result("GS"), {"GS": _spatial_result("GS")})
    assert rows
    assert all(row["rms"] == 0.0 and row["max"] == 0.0 for row in rows)


def test_a_result_is_scored_against_each_reference_contributor():
    """No single reference code exists in Spada (2011); each contributor gets its rows."""
    references = {"GS": _spatial_result("GS"), "VK": _spatial_result("VK", offset=0.5)}
    rows = _metrics().spada_score(_spatial_result("radopt"), references)
    assert {row["reference"] for row in rows} == {"GS", "VK"}


def test_the_near_field_score_is_the_difference_metrics_of_the_two_curves():
    """The near field is 0 to 25 degrees; the numbers are metrics.difference_metrics.

    A constant offset of 0.25 m gives an RMS and a maximum of 0.25 m; the
    test compares with the function itself, so any other formula fails.
    """
    metrics = _metrics()
    candidate = _spatial_result("radopt", offset=0.25)
    reference = _spatial_result("GS")
    rows = metrics.spada_score(candidate, {"GS": reference})
    near = [row for row in rows if row["kind"] == "spatial" and tuple(row["interval"]) == (0.0, 25.0)]
    assert len(near) == 1, rows
    expected = metrics.difference_metrics(candidate.spatial("cap", "U", 1.0),
                                          reference.spatial("cap", "U", 1.0), (0.0, 25.0))
    assert near[0]["rms"] == pytest.approx(expected["rms"], rel=1e-12)
    assert near[0]["rms"] == pytest.approx(0.25, rel=1e-9)


def test_an_epoch_the_reference_does_not_hold_is_not_scored():
    """A solver's fluid limit is never compared with GS's 1000 kyr run."""
    candidate = _spatial_result("radopt", epoch=math.inf)
    reference = _spatial_result("GS", epoch=1000.0)
    assert _metrics().spada_score(candidate, {"GS": reference}) == []


def test_love_numbers_are_scored_per_degree():
    """An elastic h offset by 1e-3 at every degree scores 1e-3 against the reference."""
    results = _results()
    degrees, elastic, fluid, residues = _love(n_degrees=20)
    reference = results.SpadaResult("t01-02", "GS")
    reference.add_love_numbers("h", degrees, elastic, fluid, residues)
    candidate = results.SpadaResult("t01-02", "radopt")
    candidate.add_love_numbers("h", degrees, elastic + 1e-3, fluid, residues)
    rows = _metrics().spada_score(candidate, {"GS": reference})
    elastic_rows = [row for row in rows if row["kind"] == "love_numbers"
                    and row["quantity"] == "h_elastic"]
    assert len(elastic_rows) == 1, rows
    assert elastic_rows[0]["rms"] == pytest.approx(1e-3, rel=1e-9)
