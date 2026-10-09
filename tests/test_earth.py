"""The radial earth of a GIAMIP case: interpolation, boundaries and case lookup.

The elastic structure files themselves are checked by their SHA-256 in the
manifest tests. These tests check the rules that `giamip.earth` applies to
them: linear interpolation between nodes, the side of a boundary, layers of
viscosity in depth that reach the core-mantle boundary, and which cases can be
built.
"""

import numpy as np
import pytest

from giamip import cases, earth

# The core-mantle boundary of PREM, which PREM-stable keeps, in metres.
CMB_M = 3480e3
SURFACE_M = 6371e3


@pytest.fixture(scope="module")
def stable():
    return earth.ElasticProfile.from_model("prem-stable")


def _boundaries(profile):
    """Indices k where rows k and k+1 share a radius."""
    return np.nonzero(np.diff(profile.radius_m) == 0)[0]


def test_the_profile_has_the_339_nodes_and_11_boundaries(stable):
    assert len(stable.radius_m) == 339
    assert len(_boundaries(stable)) == 11
    assert stable.surface_radius_m == SURFACE_M
    assert stable.core_mantle_boundary_m == CMB_M


def test_a_node_away_from_a_boundary_gives_its_row_exactly(stable):
    at_boundary = set(_boundaries(stable)) | set(_boundaries(stable) + 1)
    rows = np.array([k for k in range(len(stable.radius_m)) if k not in at_boundary])
    for side in earth.SIDES:
        values = stable(stable.radius_m[rows], side)
        for name, column in stable.columns.items():
            np.testing.assert_array_equal(values[name], column[rows], err_msg=name)


def test_on_a_boundary_the_side_picks_the_row(stable):
    """The first of two rows at one radius is below the boundary, the second above."""
    k = _boundaries(stable)
    r = stable.radius_m[k]
    above, below = stable(r, "above"), stable(r, "below")
    for name, column in stable.columns.items():
        np.testing.assert_array_equal(above[name], column[k + 1], err_msg=name)
        np.testing.assert_array_equal(below[name], column[k], err_msg=name)


def test_between_two_nodes_the_value_is_linear(stable):
    k = np.nonzero(np.diff(stable.radius_m) > 0)[0]
    lo, hi = stable.radius_m[k], stable.radius_m[k + 1]
    for t in (0.25, 0.5):
        values = stable(lo + t * (hi - lo))
        for name, column in stable.columns.items():
            expected = (1 - t) * column[k] + t * column[k + 1]
            np.testing.assert_allclose(values[name], expected, rtol=1e-12, err_msg=name)


def test_the_centre_and_the_surface_work_from_both_sides(stable):
    for side in earth.SIDES:
        v = stable([0.0, SURFACE_M], side)
        assert v["rho_kg_m3"][0] == stable.columns["rho_kg_m3"][0]
        assert v["rho_kg_m3"][1] == stable.columns["rho_kg_m3"][-1]


def test_the_core_mantle_boundary_is_fluid_below_and_solid_above(stable):
    assert stable(CMB_M, "below")["mu_Pa"] == 0.0
    assert stable(CMB_M, "above")["mu_Pa"] > 0.0


def test_the_output_keeps_the_shape_of_the_input(stable):
    r = np.linspace(0, SURFACE_M, 12).reshape(3, 4)
    assert stable(r)["rho_kg_m3"].shape == (3, 4)
    assert stable(5000e3)["rho_kg_m3"].shape == ()


@pytest.mark.parametrize("bad", [-1.0, SURFACE_M + 1.0, np.nan])
def test_a_radius_outside_the_earth_is_an_error(stable, bad):
    with pytest.raises(ValueError):
        stable([1e6, bad])


def test_an_unknown_side_is_an_error(stable):
    with pytest.raises(ValueError, match="side"):
        stable(5e6, "middle")


def test_a_million_radii_are_fast(stable):
    """One binary search per radius and no loop in Python."""
    import time
    r = np.random.default_rng(0).uniform(0, SURFACE_M, 1_000_000)
    start = time.perf_counter()
    stable(r)
    assert time.perf_counter() - start < 5.0


def test_prem_and_prem_stable_share_the_nodes_and_the_moduli():
    """Only the density was changed, so the moduli are those of PREM."""
    prem = earth.ElasticProfile.from_model("prem")
    stable = earth.ElasticProfile.from_model("prem-stable")
    np.testing.assert_array_equal(prem.radius_m, stable.radius_m)
    for name in ("kappa_Pa", "mu_Pa"):
        np.testing.assert_allclose(prem.columns[name], stable.columns[name], rtol=1e-9)
    assert not np.array_equal(prem.columns["rho_kg_m3"], stable.columns["rho_kg_m3"])


# ------------------------------------------------------------- viscosity


@pytest.mark.parametrize("model_id", ["vm5a", "vm7"])
def test_the_deepest_layer_ends_at_the_core_mantle_boundary(stable, model_id):
    """Protocol v4.0 gives the deepest layer as reaching the CMB."""
    layers = cases.earth_model(model_id)["layers"]
    assert layers[-1]["bottom_depth_km"] == (SURFACE_M - CMB_M) / 1e3
    v = earth.ViscosityProfile.from_model(model_id, stable)
    assert v.edges_m[0] == CMB_M


def test_vm5a_layers(stable):
    v = earth.ViscosityProfile.from_model("vm5a", stable)
    depth_km = np.array([30, 80, 200, 500, 1000, 2000, 2890])
    expected = [np.inf, 1e22, 5e20, 5e20, 1.57e21, 3.23e21, 3.23e21]
    np.testing.assert_array_equal(v(SURFACE_M - depth_km * 1e3), expected)


def test_a_layer_boundary_takes_the_side(stable):
    v = earth.ViscosityProfile.from_model("vm5a", stable)
    r = SURFACE_M - 670e3
    assert v(r, "above") == 5e20
    assert v(r, "below") == 1.57e21
    # The base of the elastic lid.
    assert v(SURFACE_M - 60e3, "above") == np.inf
    assert v(SURFACE_M - 60e3, "below") == 1e22


def test_the_core_has_no_viscosity(stable):
    v = earth.ViscosityProfile.from_model("vm7", stable)
    assert np.isnan(v(CMB_M, "below"))
    assert np.isnan(v(1000e3))
    assert v(CMB_M, "above") == 1.2e22


@pytest.mark.parametrize("model_id", ["gowan2021", "giamip-open-range-midpoint"])
def test_a_profile_from_parameters_has_three_layers(stable, model_id):
    m = cases.earth_model(model_id)
    v = earth.ViscosityProfile.from_model(model_id, stable)
    lith = m["lithosphere_thickness_km"] * 1e3
    boundary = m["upper_lower_mantle_boundary_depth_km"] * 1e3
    depths = np.array([lith / 2, (lith + boundary) / 2, 2000e3])
    np.testing.assert_array_equal(
        v(SURFACE_M - depths),
        [np.inf, m["upper_mantle_viscosity_pa_s"], m["lower_mantle_viscosity_pa_s"]])


def test_the_upper_lower_mantle_boundary_is_670_km():
    """The depth at which VM5a and VM7 of the protocol change, Sections 3.2.1-3.2.2."""
    for model_id in ("gowan2021", "giamip-open-range-midpoint"):
        assert cases.earth_model(model_id)["upper_lower_mantle_boundary_depth_km"] == 670


def test_a_range_is_not_a_structure(stable):
    with pytest.raises(ValueError, match="range"):
        earth.ViscosityProfile.from_model("giamip-open-range", stable)


# ------------------------------------------------------------- cases

ONE_D = ["giamip-exp01", "giamip-exp02", "giamip-exp03", "giamip-exp04", "giamip-exp05",
         "giamip-exp06", "giamip-exp07", "giamip-exp09", "giamip-exp10"]
THREE_D = ["giamip-exp08", "giamip-exp11", "giamip-exp12"]


@pytest.mark.parametrize("case_id", ONE_D)
def test_every_1d_case_builds_on_prem_stable(case_id):
    e = earth.for_case(case_id)
    assert e.elastic.model_id == "prem-stable"
    assert e.compressibility == "compressible"
    values = e(np.linspace(0, SURFACE_M, 101))
    assert np.isfinite(values["rho_kg_m3"]).all()


@pytest.mark.parametrize("case_id", THREE_D)
def test_a_laterally_varying_case_says_where_its_grid_comes_from(case_id):
    with pytest.raises(NotImplementedError, match="g-drift"):
        earth.for_case(case_id)


def test_an_open_structure_needs_the_gadopt_choice():
    with pytest.raises(ValueError, match="open"):
        earth.for_case("giamip-exp03", gadopt=False)
    # Exp01 prescribes both structures, so the protocol alone is enough.
    assert earth.for_case("giamip-exp01", gadopt=False).viscosity.model_id == "vm5a"


def test_a_benchmark_case_is_refused():
    with pytest.raises(ValueError, match="not a GIAMIP case"):
        earth.for_case("martinec2018-A")


# ------------------------------------------------------------- malformed tables


def _table(radius):
    """An elastic profile with one column, for checking the table rules."""
    r = np.asarray(radius, dtype=float)
    return earth.ElasticProfile(model_id="test", radius_m=r,
                                columns={"rho_kg_m3": np.ones_like(r)})


@pytest.mark.parametrize("radius", [
    [0.0],                 # one node, no interval
    [0.0, 0.0, 1.0],       # a boundary at the centre
    [0.0, 1.0, 1.0],       # a boundary at the surface
    [0.0, 1.0, 1.0, 1.0, 2.0],  # three rows at one radius
    [0.0, 2.0, 1.0],       # radii that decrease
    [1.0, 2.0],            # no node at the centre
])
def test_a_malformed_table_is_rejected(radius):
    """Each of these would give NaN or an unreachable row without an error."""
    with pytest.raises(ValueError):
        _table(radius)


def test_a_column_of_the_wrong_length_is_rejected():
    with pytest.raises(ValueError, match="column"):
        earth.ElasticProfile(model_id="test", radius_m=np.array([0.0, 1.0, 2.0]),
                             columns={"rho_kg_m3": np.ones(2)})


def test_a_scalar_radius_gives_0d_arrays_for_every_property():
    values = earth.for_case("giamip-exp01")(6000e3)
    for name, value in values.items():
        assert isinstance(value, np.ndarray) and value.shape == (), name


def test_exp07_takes_the_rheology_of_exp02():
    """The protocol table gives Exp07 the rheology "Same as Exp02"."""
    assert earth.for_case("giamip-exp07").rheology == earth.for_case("giamip-exp02").rheology
    assert earth.for_case("giamip-exp07").rheology == "maxwell"
    # Without the G-ADOPT choices Exp02, and so Exp07, leaves rheology open.
    assert earth.for_case("giamip-exp07", gadopt=False).rheology == "open"
