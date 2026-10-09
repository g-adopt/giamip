"""The classes solvers use: case set-up, results, metrics and GIAMIP inputs.

Tests without a mark need only the repository. Tests marked `data` need the
converted Martinec files (tracked) or the GIAMIP originals (placed by hand),
and skip when those are absent.
"""

import numpy as np
import pytest

from giamip import grids
from giamip.benchmarks import metrics
from giamip.benchmarks.inputs import MartinecCase, angular_distance_deg
from giamip.benchmarks.results import BenchmarkResult, Curve
from giamip.mip import inputs as mip_inputs


# ----------------------------------------------------------- Martinec set-up

def test_the_step_scenario_is_full_after_zero_and_empty_before():
    case = MartinecCase("A")
    assert case.cap_size(0.0) == (0.0, 0.0)
    assert case.cap_size(1e-9) == (1500.0, 10.0)
    assert case.cap_size(10.0) == (1500.0, 10.0)


def test_the_growth_scenario_grows_linearly_to_ten_kyr_and_then_holds():
    case = MartinecCase("C")
    assert case.cap_size(5.0) == (250.0, 5.0)
    assert case.cap_size(10.0) == (500.0, 10.0)
    assert case.cap_size(15.0) == (500.0, 10.0)


def test_the_cap_profile_is_the_square_root_profile():
    case = MartinecCase("B")
    cap = case.cap
    c, l = cap["centre_colatitude_deg"], cap["centre_longitude_deg"]
    assert case.ice_thickness(c, l, 10.0) == pytest.approx(1500.0)
    assert case.ice_thickness(c + 10.5, l, 10.0) == 0.0
    expected = 1500.0 * np.sqrt((np.cos(np.radians(5)) - np.cos(np.radians(10)))
                                / (1 - np.cos(np.radians(10))))
    assert case.ice_thickness(c + 5.0, l, 10.0) == pytest.approx(expected, rel=1e-12)


def test_the_closed_form_mass_matches_a_numerical_integral():
    """The cap of case A sits on the pole, so a 1-D quadrature in colatitude suffices."""
    case = MartinecCase("A")
    x, w = np.polynomial.legendre.leggauss(400)
    theta = np.radians(5.0) * (x + 1.0)          # colatitude 0 to 10 degrees
    h = case.ice_thickness(np.degrees(theta), 0.0, 10.0)
    integral = case.ice_density * 2 * np.pi * case.earth_radius_m ** 2 * np.sum(
        w * np.radians(5.0) * h * np.sin(theta))
    assert integral == pytest.approx(case.ice_mass(10.0), rel=1e-5)


def _legendre(n_max: int, x: np.ndarray) -> np.ndarray:
    """P_0 .. P_n_max at the points x, by the three-term recurrence."""
    p = np.zeros((n_max + 1, x.size))
    p[0] = 1.0
    if n_max >= 1:
        p[1] = x
    for n in range(1, n_max):
        p[n + 1] = ((2 * n + 1) * x * p[n] - n * p[n - 1]) / (n + 1)
    return p


def test_the_cap_legendre_coefficients_are_the_projection_of_the_profile():
    """sigma_n = (2n + 1)/2 integral of sigma(psi) P_n(cos psi) over cos psi.

    The integral runs over x = cos psi from cos alpha to 1, where the cap
    sits. The substitution x = cos alpha + (1 - cos alpha) u^2 turns the
    square root of the profile into u, so the integrand is a polynomial in u
    of degree at most 2 n_max + 2. Gauss-Legendre quadrature with 400 nodes
    integrates it exactly up to rounding, so the tolerance, 1e-8 relative
    with a floor of 1e-4 kg/m2 against a degree-1 value near 1e5 kg/m2, is a
    statement about rounding and not about the closed form.
    """
    case = MartinecCase("B")
    degrees = np.arange(0, 129)
    height, alpha_deg = case.cap_size(10.0)
    c_alpha = np.cos(np.radians(alpha_deg))
    nodes, weights = np.polynomial.legendre.leggauss(400)
    u = 0.5 * (nodes + 1.0)
    w = 0.5 * weights
    x = c_alpha + (1.0 - c_alpha) * u ** 2
    jacobian = 2.0 * (1.0 - c_alpha) * u
    # The profile as a function of the angle from the cap centre, in kg/m2.
    sigma = case.ice_density * height * np.sqrt((x - c_alpha) / (1.0 - c_alpha))
    p = _legendre(int(degrees[-1]), x)
    expected = (2 * degrees + 1) / 2.0 * (p[degrees] * sigma * jacobian * w).sum(axis=1)
    got = case.legendre_coefficients(degrees, 10.0)
    np.testing.assert_allclose(got, expected, rtol=1e-8, atol=1e-4)


def test_the_degree_0_cap_coefficient_carries_the_cap_mass():
    """4 pi a^2 sigma_0 is the mass the closed form of `ice_mass` gives."""
    case = MartinecCase("C")
    sigma_0 = case.legendre_coefficients(np.array([0]), 15.0)[0]
    mass = 4 * np.pi * case.earth_radius_m ** 2 * sigma_0
    assert mass == pytest.approx(case.ice_mass(15.0), rel=1e-12)


def test_the_cap_coefficients_vanish_before_the_cap_exists():
    """Case C grows from nothing, so t = 0 has no load and every degree is zero."""
    got = MartinecCase("C").legendre_coefficients(np.arange(0, 5), 0.0)
    assert np.all(got == 0.0)


def test_a_negative_degree_is_refused():
    with pytest.raises(ValueError, match="non-negative"):
        MartinecCase("B").legendre_coefficients([-1, 0, 1], 10.0)


def test_the_basin_radius_and_area_match_the_paper():
    case = MartinecCase("D")
    assert case.basin_radius_deg() == pytest.approx(24.8503, abs=5e-5)
    assert case.initial_ocean_area_fraction() == pytest.approx(0.0462956, abs=5e-7)
    b = case.basin
    assert case.topography(b["centre_colatitude_deg"], b["centre_longitude_deg"]) == pytest.approx(-2200.0)
    assert case.topography(b["centre_colatitude_deg"] + case.basin_radius_deg(),
                           b["centre_longitude_deg"]) == pytest.approx(0.0, abs=1e-6)


def test_case_a_has_no_ocean_and_case_e_prescribes_the_terminal_topography():
    assert MartinecCase("A").ocean_function(np.array([0.0, 90.0, 180.0]), 0.0).sum() == 0
    assert MartinecCase("D").topography_time == "initial"
    assert MartinecCase("E").topography_time == "terminal"


def test_the_earth_layers_are_m3_l70_v01():
    layers = MartinecCase("C").earth_layers()
    assert layers["top_radius_m"][0] == 6371e3
    assert np.isinf(layers["viscosity_pa_s"][0])
    assert layers["viscosity_pa_s"][-1] == 0.0
    assert layers["density_kg_m3"].size == 5


def test_the_profiles_and_time_series_come_from_the_case_file():
    b = {p.name: p for p in MartinecCase("B").profiles}
    assert b["basin"].longitude_deg == 320 and b["basin"].colatitude_interval_deg == (65, 135)
    assert b["load"].quantities == ("U", "N")
    e = MartinecCase("E")
    assert e.profiles == []
    assert {q for s in e.time_series for q in s.quantities} == {"ocean_area", "h_UF"}


def test_ice_on_grid_follows_the_latitude_order_it_is_given():
    case = MartinecCase("A")
    lat = np.array([89.0, 0.0, -89.0])
    grid = case.ice_on_grid(lat, [0.0, 90.0], [10.0])
    assert grid.shape == (1, 3, 2)
    assert grid[0, 0, 0] > 0 and grid[0, 2, 0] == 0


def test_angular_distance_is_safe_at_the_centre_and_the_antipode():
    assert angular_distance_deg(25.0, 75.0, 25.0, 75.0) == pytest.approx(0.0, abs=1e-6)
    assert angular_distance_deg(155.0, 255.0, 25.0, 75.0) == pytest.approx(180.0, abs=1e-6)


# ------------------------------------------------------- results and metrics

def test_a_result_round_trips_through_netcdf(tmp_path):
    r = BenchmarkResult("C", "lovejx")
    x = np.linspace(0, 180, 181)
    r.add_profile("load", "U", x, np.sin(np.radians(x)))
    r.add_profile("load", "N", x[::-1], np.cos(np.radians(x[::-1])))
    r.add_time_series("h_UF", [0, 5, 15], [0.0, -10.0, -25.0], note="test")
    path = r.write(tmp_path / "out.nc")
    back = BenchmarkResult.read(path, "lovejx")
    assert back.case == "C"
    assert np.allclose(back.profile("load", "N").axis, x)
    assert np.allclose(back.profile("load", "N").values, np.cos(np.radians(x)))
    assert back.series("h_UF").attributes["note"] == "test"
    with pytest.raises(ValueError, match="already has"):
        r.write(path)


def test_reading_a_result_under_another_case_raises(tmp_path):
    """The file records case C, so a read as case D is refused; another spelling of C is not."""
    r = BenchmarkResult("C", "lovejx")
    r.add_time_series("h_UF", [0, 5, 15], [0.0, -10.0, -25.0])
    path = r.write(tmp_path / "out.nc")
    assert BenchmarkResult.read(path, "lovejx", case="martinec2018-c").case == "C"
    with pytest.raises(ValueError, match="holds case C, not D"):
        BenchmarkResult.read(path, "lovejx", case="D")


def test_difference_metrics_on_known_curves():
    ref = Curve([0, 1, 2, 3, 4], [0, 0, 0, 0, 0])
    cand = Curve([0, 4], [1, 5])                  # 1 + x
    m = metrics.difference_metrics(cand, ref, (1, 3))
    assert m["n"] == 3 and m["max"] == 4.0 and m["at"] == 3.0
    assert m["rms"] == pytest.approx(np.sqrt((4 + 9 + 16) / 3))
    with pytest.raises(ValueError, match="covers"):
        metrics.difference_metrics(Curve([2, 4], [0, 0]), ref, (1, 3))


def test_a_grid_that_stops_just_short_of_the_interval_is_accepted():
    """A colatitude grid starting at 0.044 is compared over 0 to 20 from its first node."""
    ref = Curve(np.linspace(0, 20, 201), np.zeros(201))
    cand = Curve(np.linspace(0.044, 180, 500), np.ones(500))
    m = metrics.difference_metrics(cand, ref, (0, 20))
    assert m["covered"][0] == pytest.approx(0.1) and m["max"] == 1.0
    with pytest.raises(ValueError, match="covers"):
        metrics.difference_metrics(Curve(np.linspace(1.0, 180, 500), np.ones(500)), ref, (0, 20))


def test_a_curve_refuses_a_repeated_axis_value():
    with pytest.raises(ValueError, match="repeats"):
        Curve([0, 1, 1], [0, 1, 2])


# ------------------------------------------------- against the converted data

def _need_converted():
    """Skip when the converted Martinec files are neither here nor downloadable."""
    from giamip import data
    try:
        BenchmarkResult.reference_path("E")
    except data.DataUnavailable:
        pytest.skip("the converted Martinec files are absent and cannot be downloaded")


@pytest.mark.data
def test_the_cap_mass_is_vega_total_ice_mass_at_every_time():
    """VEGA's grounded plus floating ice in case D is the full cap at every time.

    This checks the square-root profile, the density, the earth radius and
    the self-similar growth that the class assumes for T1. If only the height
    grew, the mass at 5 kyr would be four times larger. During growth the
    small cap is resolved by few VEGA grid nodes, and the two agree to 7e-4;
    after growth they agree to 4e-5.
    """
    _need_converted()
    vega = BenchmarkResult.reference("D", "ZM")
    g, f = vega.series("ice_mass_grounded"), vega.series("ice_mass_floating")
    case = MartinecCase("D")
    for t, rel in ((2.0, 1e-3), (5.0, 1e-3), (10.0, 1e-4), (15.0, 1e-4)):
        total = np.interp(t, g.axis, g.values) + np.interp(t, f.axis, f.values)
        assert total == pytest.approx(case.ice_mass(t), rel=rel), t


@pytest.mark.data
def test_scoring_ys_against_vega_reproduces_its_difference_file():
    """Case C: YS and VEGA share a grid, so the score equals the shipped difference."""
    _need_converted()
    ys = BenchmarkResult.reference("C", "YS")
    rows = {(r["profile"], r["quantity"]): r for r in metrics.score(ys) if r["kind"] == "profile"}
    assert len(rows) == 5
    for (profile, quantity), row in rows.items():
        spec = next(p for p in MartinecCase("C").profiles if p.name == profile)
        d = ys.differences[(profile, quantity)]
        inside = (d.axis >= spec.colatitude_interval_deg[0]) & (d.axis <= spec.colatitude_interval_deg[1])
        assert row["rms"] == pytest.approx(np.sqrt(np.mean(d.values[inside] ** 2)), abs=1e-4)


@pytest.mark.data
def test_case_a_scores_match_the_paper_against_the_recovered_reference():
    """The paper: all codes except SK within 0.7 m in U over 0 to 20 degrees."""
    _need_converted()
    for member in ("GA", "GS", "KS", "VB", "VK", "WW", "YS"):
        row = next(r for r in metrics.score(BenchmarkResult.reference("A", member))
                   if r["quantity"] == "U")
        assert row["max"] < 0.75, (member, row)
        assert "recovered" in row["reference"]


# ---------------------------------------------------------------- GIAMIP

def test_grounded_thickness_removes_only_floating_ice():
    h = np.array([100.0, 100.0, 100.0, 0.0])
    bed = np.array([10.0, -50.0, -200.0, -200.0])
    out = mip_inputs.grounded_thickness(h, bed, 917.0, 1025.0)
    # 100 m of ice floats in more than 89.5 m of water.
    assert out.tolist() == [100.0, 100.0, 0.0, 0.0]


def test_model_years_follow_the_protocol_steps():
    years = mip_inputs.GiamipExperiment("Exp01").model_years()
    assert years[0] == -122000 and years[-1] == 1000 and years.size == 463
    assert np.all(np.diff(years[years < -26000]) == 500)
    assert np.all(np.diff(years[years >= -26000]) == 100)
    assert mip_inputs.GiamipExperiment(6).model_years()[0] == -80000


def _ice_file(name="iceHistory-ICE-6G_D.nc"):
    """The relative path of an ice history, located (and fetched) through giamip.data."""
    from giamip import data
    rel = f"giamip/ice-history/{name}"
    try:
        data.locate(rel, "original")
    except data.DataUnavailable:
        pytest.skip("the GIAMIP originals are absent and cannot be downloaded")
    return rel


@pytest.mark.data
def test_an_ice_history_reads_unmasked_on_the_prescribed_grid():
    _ice_file()
    exp = mip_inputs.GiamipExperiment("Exp01")
    ice = exp.ice_history
    assert len(ice) == 463 and ice.years[0] == -122000 and ice.years[-1] == 1000
    present = ice.thickness(ice.step_of_year(0))
    assert not np.ma.isMaskedArray(present) and np.isfinite(present).all()
    assert (present == 0).sum() > 100000
    assert np.array_equal(ice.thickness(0, north_first=True), ice.thickness(0)[::-1])
    assert ice.latitudes(north_first=True)[0] > 0
    assert {"ice-fillvalue-zero", "ice-year-longname"} <= set(ice.correction_ids)
    assert ice.is_current_version
    # Present-day ice volume, integrated with the quadrature weights: about
    # 2.6e16 m3 for Antarctica and 2.9e15 m3 for Greenland.
    volume = grids.integrate(present, exp.constants["earth_radius_m"])
    assert 2.5e16 < volume < 3.6e16
    assert np.array_equal(exp.forcing_steps(), np.arange(463))


@pytest.mark.data
@pytest.mark.parametrize("experiment", ["Exp06", "Exp07"])
def test_the_80_ka_experiments_read_every_step_of_their_80_ka_file(experiment):
    """The `_80ka` files start where Exp06 and Exp07 start, so no step is skipped."""
    _ice_file()
    exp = mip_inputs.GiamipExperiment(experiment)
    steps = exp.forcing_steps()
    ice = exp.ice_history
    assert ice.years[steps[0]] == -80000 and ice.years[steps[-1]] == 1000
    assert np.array_equal(steps, np.arange(379))


@pytest.mark.data
def test_topography_names_its_antarctic_dataset_from_the_filename():
    from giamip import data
    rel = "giamip/initial-topography/bedtopo_BedMach3ANT_BedMach5GRL_GEBCO_GLQ256.nc"
    try:
        data.locate(rel, "original")
    except data.DataUnavailable:
        pytest.skip("the GIAMIP originals are absent and cannot be downloaded")
    topo = mip_inputs.Topography(rel)
    assert topo.antarctic_dataset == "BedMachine Antarctica v3"
    # The version of 2026-09-24 names its Antarctic dataset correctly, so the
    # source-attribute correction does not apply to it. The missing `crs`
    # variable is still missing.
    assert "topography-source-attribute" not in topo.correction_ids
    assert "topography-missing-crs" in topo.correction_ids
    z = topo.elevation()
    assert z.min() < -9000 and z.max() > 5000
