"""The classes solvers use: case set-up, results, metrics and GIAMIP inputs.

Tests without a mark need only the repository. Tests marked `data` need the
converted Martinec files (tracked) or the GIAMIP originals (placed by hand),
and skip when those are absent.
"""

import numpy as np
import pytest

from gia_mip import grids
from gia_mip.benchmarks import metrics
from gia_mip.benchmarks.inputs import MartinecCase, angular_distance_deg
from gia_mip.benchmarks.results import BenchmarkResult, Curve
from gia_mip.mip import inputs as mip_inputs


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
    r = BenchmarkResult("C", "radopt")
    x = np.linspace(0, 180, 181)
    r.add_profile("load", "U", x, np.sin(np.radians(x)))
    r.add_profile("load", "N", x[::-1], np.cos(np.radians(x[::-1])))
    r.add_time_series("h_UF", [0, 5, 15], [0.0, -10.0, -25.0], note="test")
    path = r.write(tmp_path / "out.nc")
    back = BenchmarkResult.read(path, "radopt")
    assert back.case == "C"
    assert np.allclose(back.profile("load", "N").axis, x)
    assert np.allclose(back.profile("load", "N").values, np.cos(np.radians(x)))
    assert back.series("h_UF").attributes["note"] == "test"
    with pytest.raises(ValueError, match="already has"):
        r.write(path)


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
    if not BenchmarkResult.reference_path("E").exists():
        pytest.skip("the converted Martinec files are absent")


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
    rel = f"giamip/ice-history/{name}"
    if not (mip_inputs.original_root() / rel).exists():
        pytest.skip("the GIAMIP originals are absent")
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
def test_the_truncated_experiments_start_at_80_ka():
    _ice_file()
    steps = mip_inputs.GiamipExperiment("Exp06").forcing_steps()
    ice = mip_inputs.GiamipExperiment("Exp06").ice_history
    assert ice.years[steps[0]] == -80000 and ice.years[steps[-1]] == 1000


@pytest.mark.data
def test_topography_names_its_antarctic_dataset_from_the_filename():
    rel = "giamip/initial-topography/bedtopo_BedMach3ANT_BedMach5GRL_GEBCO_GLQ256.nc"
    if not (mip_inputs.original_root() / rel).exists():
        pytest.skip("the GIAMIP originals are absent")
    topo = mip_inputs.Topography(rel)
    assert topo.antarctic_dataset == "BedMachine Antarctica v3"
    assert "topography-source-attribute" in topo.correction_ids
    z = topo.elevation()
    assert z.min() < -9000 and z.max() > 5000
