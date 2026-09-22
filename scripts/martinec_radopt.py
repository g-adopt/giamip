"""Run a Martinec et al. (2018) benchmark case with radopt.

This repository specifies the cases and holds the data to compare against; the
viscoelastic response is computed by radopt. This script is the piece between
them. It reads the case with `gia_mip.benchmarks.inputs.MartinecCase`, builds
the earth model and the load for radopt, runs radopt, and writes the result as
a `BenchmarkResult`, which is the same object the published contributors are
read through.

Run it from the repository root, with radopt on the path:

    PYTHONPATH=~/Workplace/radopt \
      ~/Workplace/python3.12/bin/python3.12 scripts/martinec_radopt.py A --norder 256

All five cases, A to E.

Case A does not solve the sea-level equation: there is no ocean, the load is
the ice alone, and the answer is one convolution of the load history against
the viscous kernel (`radopt.sealevel.surface_load`).

Cases B and C do solve it, with the ocean geometry and the topography both
fixed at the initial time (`radopt.sealevel.solver`).

Cases D and E let the coastline move and marine ice float. D prescribes the
topography at the first step; E prescribes it at the terminal time and solves
for the first step's topography with the iteration of eq. 34.

Runs matched to G-ADOPT (`NOTES/martinec-radopt/03-gadopt-matched.md`). Cases B
and C take `--ocean-function PATH`, the G-ADOPT mask `C0` on the Gauss-Legendre
grid, which replaces the sharp mask radopt builds itself. Cases B, C and D take
`--times PATH`, the G-ADOPT step times, which replace the uniform step. A
matched run is written under its own member name (`--member`), so that it does
not collide with the published run:

    PYTHONPATH=~/Workplace/radopt \
      ~/Workplace/python3.12/bin/python3.12 scripts/martinec_radopt.py B \
      --ocean-function ocean_function_C0.npz --times gadopt_times.txt \
      --member radopt-matched
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

import jax                                                           # noqa: E402
import jax.numpy as jnp                                              # noqa: E402

jax.config.update("jax_enable_x64", True)

from gia_mip.benchmarks.inputs import MartinecCase                   # noqa: E402
from gia_mip.benchmarks.results import BenchmarkResult               # noqa: E402
from radopt.lovenumbers.profile import layered_profile               # noqa: E402
from radopt.lovenumbers.propagate import StepPlan                    # noqa: E402
from radopt.lovenumbers.rheology import Maxwell                      # noqa: E402
from radopt.lovenumbers.solver import LoveNumberSolver               # noqa: E402
from radopt.lovenumbers.units import Units                           # noqa: E402
from radopt.sealevel.constants import LegacyConstants                # noqa: E402
from radopt.sealevel.sht import (GaussLegendreGrid, analysis,        # noqa: E402
                                 area_mean, evaluate, legendre_table,
                                 synthesis)
from radopt.lovenumbers.asymptotes import DegreeOneFrame             # noqa: E402
from radopt.sealevel.solver import SeaLevelConfig, solve_sea_level   # noqa: E402
from radopt.sealevel.surface_load import surface_load_response       # noqa: E402


def earth_model(case: MartinecCase, *, nodes_per_layer: int = 40, substeps: int = 2):
    """The radopt Love-number solver and rheology for the case's earth model.

    M3-L70-V01 is five uniform layers, incompressible, with an elastic
    lithosphere, three Maxwell mantle layers and an inviscid fluid core. Each
    of those three properties is carried explicitly:

    * incompressible through `compressible=False`, which substitutes the three
      incompressible limits into the coefficient matrix. It is not a large bulk
      modulus, and the bulk modulus of the profile is never read by it;
    * the elastic lithosphere as an infinite Maxwell viscosity, which is what
      "no viscous flow" means for a Maxwell body;
    * the fluid core through its zero shear modulus and the `fluid_solid`
      interface that `layered_profile` finds from it.

    Returns
    -------
    solver
        A `LoveNumberSolver` bound to the profile.
    rheology
        The `Maxwell` body, one fluidity per node.
    """
    layers = case.earth_layers()
    profile, layer_of_node = layered_profile(
        layers["top_radius_m"], layers["bottom_radius_m"],
        layers["density_kg_m3"], layers["shear_modulus_pa"],
        nodes_per_layer=nodes_per_layer,
        names=[str(layer["role"]) for layer in case.earth_model["layers"]])
    # The core is inviscid. A zero viscosity is an infinite fluidity, which no
    # array can hold, so the core is given a viscosity small enough to be fluid
    # on every timescale of the case and finite in the arithmetic. It carries no
    # shear modulus, so nothing in the mechanical problem reads it.
    viscosity = np.where(layers["viscosity_pa_s"] == 0.0, 1.0e-30,
                         layers["viscosity_pa_s"])[layer_of_node]
    solver = LoveNumberSolver(profile, compressible=False, plan=StepPlan(substeps=substeps))
    return solver, Maxwell(viscosity)


def constants(case: MartinecCase, solver) -> LegacyConstants:
    """The physical constants of the case, as the sea-level side of radopt takes them.

    Three of these decide the answer and must be the case's own, never a module
    default: the ice density (931 kg/m3 here, 917 in GIAMIP), the water density
    and the pair (earth radius, earth mass) that the load factor
    `T_l = 4 pi Re^3 / (Me (2l+1))` is built from.

    The mass is not read from anywhere: it is the mass of the earth model the
    Love numbers were computed on, recovered from that model's own integrated
    surface gravity as `Me = g(a) a^2 / G`. Taking it from a table instead would
    let the load factor and the Love numbers describe two different planets, and
    the resulting error scales every field by the same factor, which is exactly
    the kind of mistake that looks plausible.

    The rotation constants are left at their legacy values and are never read:
    every Martinec case is non-rotating.
    """
    gravity_surface = float(np.asarray(solver.profile.gravity)[-1])
    radius = float(np.asarray(solver.profile.radius)[-1])
    mass = gravity_surface * radius ** 2 / solver.units.G
    return LegacyConstants(
        pi=np.pi, Re=radius, Me=mass,
        rhoi=case.ice_density, rhow=case.water_density,
        gacc=gravity_surface,
        Omega=7.292e-5, G=solver.units.G,
        A=0.3296145 * mass * radius ** 2, C=0.3307007 * mass * radius ** 2,
        kf=0.9342 + 0.008, epsilon1=1.0e-5, epsilon2=1.0e-5)


def cap_coefficients(case: MartinecCase, degrees_max: int, t_kyr: float) -> np.ndarray:
    """Exact spherical-harmonic coefficients of the ice cap at time t, kg/m2.

    The cap is axially symmetric about its centre, so its expansion about that
    centre is the one-index `sigma_n` of `MartinecCase.legendre_coefficients`.
    The addition theorem moves it to the cap's actual position: in the 4 pi
    convention of `radopt.sealevel.sht`, where the harmonics satisfy
    `sum_m |Y_lm|^2 = 2l + 1`,

        P_l(cos psi) = (1 / (2l+1)) sum_m Y_lm(Omega) conj(Y_lm(Omega_0)),

    so a load `sum_l sigma_l P_l(cos psi)` has the coefficients

        U(l, m) = sigma_l / (2l + 1) * Pbar_lm(cos theta_0) * exp(-i m lambda_0).

    This is exact at every degree. Analysing the cap on a grid instead would
    alias its tail back onto the retained degrees, because the square-root edge
    of the profile makes the expansion converge slowly.

    Returns
    -------
    An `(L+1, L+1)` complex array, zero for `m > l`.
    """
    degrees = np.arange(degrees_max + 1)
    sigma = case.legendre_coefficients(degrees, t_kyr)
    centre = case.cap
    colatitude = np.radians(centre["centre_colatitude_deg"])
    longitude = np.radians(centre["centre_longitude_deg"])
    # Pbar_lm at the cap centre, shape (L+1, L+1, 1) from the one-point table.
    plm = np.asarray(legendre_table(degrees_max, np.cos([colatitude])))[:, :, 0]
    orders = np.arange(degrees_max + 1)
    phase = np.exp(-1j * orders * longitude)[None, :]
    return (sigma / (2.0 * degrees + 1.0))[:, None] * plm * phase


def run_case_a(*, norder: int = 256, nodes_per_layer: int = 40,
               substeps: int = 2, colatitude=None) -> BenchmarkResult:
    """Benchmark A: a Heaviside cap at the pole, no ocean, U and N at 10 kyr.

    Two time steps and no more. Every load increment enters the convolution as a
    step applied at its own time, and the kernel is the response to a unit step,
    so a load switched on at t = 0 and read at t = 10 kyr is exactly one term.
    There is no time-discretisation error in this case at all.

    Degrees 0 and 1 are left out, as the paper does (p. 399). Degree 0 is
    already absent, because the load factor of eq. B16 is zero there.
    """
    case = MartinecCase("A")
    solver, rheology = earth_model(case, nodes_per_layer=nodes_per_layer, substeps=substeps)
    consts = constants(case, solver)

    spectra = solver.spectra(rheology, np.arange(1, norder + 1))
    kernel = spectra.kernel(norder)

    # Three steps: the unloaded state, the step that applies the cap, and the
    # epoch to read. An increment carries its viscous response only at steps
    # strictly after the one it is applied at, so the load has to be applied
    # before the epoch and not at it. The first two are one year apart, which
    # the kernel never sees: the only lag that enters the answer is the 10 kyr
    # between the second step and the third.
    terminal = case.terminal_time_kyr * 1.0e3                        # years
    times = np.array([-1.0, 0.0, terminal])
    loaded = cap_coefficients(case, norder, case.terminal_time_kyr)
    load_lm = np.stack([np.zeros_like(loaded), loaded, loaded])
    response = surface_load_response(kernel, times, norder=norder, load_lm=load_lm,
                                     consts=consts, min_degree=2)

    # The profile of the paper: the meridian through the cap centre, which for
    # benchmark A sits at the pole, so the meridian is longitude 0. It is read at
    # the colatitudes the reference data carries, by summing the series there;
    # this is exact and is not an interpolation between grid rows.
    profile_spec = case.profiles[0]
    if colatitude is None:
        # The colatitudes of the published reference, so that the two curves are
        # sampled at the same points and the metrics of eq. 31 need no
        # interpolation of either side.
        from martinec_compare import published_reference
        colatitude = published_reference("A", "load", "U")[0]
    colatitude = np.asarray(colatitude, dtype=float)

    result = BenchmarkResult("A", "radopt")
    for quantity, coefficients in (("U", response.uplift_lm[-1]),
                                   ("N", response.geoid_lm[-1])):
        values = np.asarray(evaluate(coefficients, colatitude, profile_spec.longitude_deg))
        result.add_profile("load", quantity, colatitude, values,
                           time_kyr=profile_spec.time_kyr,
                           longitude_deg=profile_spec.longitude_deg)
    return result


def cap_on_grid(case: MartinecCase, grid, norder: int, t_kyr: float) -> np.ndarray:
    """Ice thickness in metres on the Gauss-Legendre grid, band limited at `norder`.

    The exact coefficients of `cap_coefficients`, synthesised onto the grid and
    divided by the ice density. The solver analyses whatever field it is given,
    and this one is band limited at `norder`, so that analysis returns the exact
    coefficients back: the quadrature of `nglv` latitudes is exact for
    polynomial degree `2 nglv - 1`, which is above `2 norder` here. Sampling the
    cap profile pointwise instead would alias its tail, because the square-root
    edge makes the expansion converge slowly.

    The truncation leaves the ripple of a step edge near the cap margin. That is
    the same truncation every contributor to the benchmark carries, and the
    paper reports the spikes it causes at the cap edge (p. 399).
    """
    coefficients = cap_coefficients(case, norder, t_kyr)
    return np.asarray(synthesis(grid, jnp.asarray(coefficients))) / case.ice_density


def ice_on_grid_at_step(case: MartinecCase, grid, norder: int, year: float) -> np.ndarray:
    """The ice that loads the step at `year`, in metres on the grid.

    A step at `t > 0` carries the cap of that time (`cap_on_grid`). The step at
    exactly `t = 0` is the step that applies the load, and it carries the right
    limit of the scenario at 0: the full cap for the step scenario T0, and the
    cap of `cap_size(0)` for the growth scenario T1, which is no ice. The
    unloaded state is the step before it, at a negative time.

    `cap_size` itself returns no cap at `t <= 0` for both scenarios. Evaluated at
    `t = 0` directly, a T0 load would then arrive one step late, at the first
    step after 0, and every viscous lag would be short by one step. The
    published case B run applies the cap at t = 0, and this function reproduces
    it bit for bit. For T1 the right limit at 0 is `cap_size(0)`, which is no
    ice, so cases C, D and E get the cap of `cap_on_grid` at every step.
    """
    t_kyr = year / 1.0e3
    if t_kyr == 0.0 and case.time_scenario == "T0":
        # The right limit of a step at 0 is the full cap, which cap_size gives
        # at any positive time. The terminal time is a positive time of the case.
        t_kyr = case.terminal_time_kyr
    return cap_on_grid(case, grid, norder, t_kyr)


def _grid_for(nglv: int, norder: int):
    """The transform of a run, with its tables on the device."""
    grid = GaussLegendreGrid.build(nglv, 2 * nglv, norder)
    return jax.tree_util.tree_map(jnp.asarray, grid)


# The largest difference, in degrees, between a coordinate of the exported mask
# and the same coordinate of the run's grid. The export and radopt compute the
# Gauss-Legendre nodes with two different codes, which agree to about 1e-12
# degrees; a grid of another size differs by more than 0.1 degree.
GRID_COORDINATE_TOLERANCE_DEG = 1.0e-8

# The largest difference, in years, allowed between the last step time of a
# ladder file and the terminal time of the case. A ladder built by summing
# floating-point steps lands on the terminal time to about 1e-9 years.
TERMINAL_TIME_TOLERANCE_YR = 1.0e-6


def read_step_times(times_path, case: MartinecCase) -> np.ndarray:
    """Read a ladder of step times and return the times the solver takes, in years.

    The file lists the times at the end of every step of the G-ADOPT run, from
    the load application at t = 0 to the terminal time of the case. Three
    formats are read:

    * `.npz` with an array `t_kyr`, in thousands of years, which is the name
      of the step time column of the G-ADOPT driver's step output;
    * `.npy` with one array, in years;
    * any other suffix as text with one number per line, in years (`#` starts
      a comment).

    The time 0 is added when the ladder does not start with it, because the
    load has to be applied at t = 0. One step before t = 0 is added in front,
    as in the uniform ladder: an increment carries its viscous response only
    at steps after the one it is applied at, so the unloaded step must exist.
    Its length is the first step after t = 0. The solver takes the lag between
    every pair of step times from the kernel, so the steps do not have to be
    equal (radopt `tests/test_prescribed_ocean.py`).

    Parameters
    ----------
    times_path
        The ladder file.
    case
        The case, for its terminal time.

    Returns
    -------
    An ascending array of step times in years that starts before 0 and ends at
    the terminal time of the case.

    Raises
    ------
    ValueError
        If a time is negative or not finite, if the times do not increase
        strictly, or if the last time is not the terminal time. A ladder that
        stops early would put the profiles of the terminal time at another
        time without an error.
    """
    path = pathlib.Path(times_path)
    if path.suffix == ".npz":
        with np.load(path) as data:
            if "t_kyr" not in data:
                raise ValueError(f"{path} has no array 't_kyr'; it holds {sorted(data.files)}")
            years = 1.0e3 * np.asarray(data["t_kyr"], dtype=float)
    elif path.suffix == ".npy":
        years = np.asarray(np.load(path), dtype=float)
    else:
        years = np.loadtxt(path, dtype=float, comments="#", ndmin=1)
    years = np.ravel(years)
    if years.size == 0 or not np.all(np.isfinite(years)):
        raise ValueError(f"{path}: the step times must be finite and there must be at least one")
    if np.any(years < 0.0):
        raise ValueError(f"{path}: a step time is negative; the ladder starts at the "
                         "load application, t = 0")
    if years[0] != 0.0:
        years = np.concatenate([[0.0], years])
    if np.any(np.diff(years) <= 0.0):
        raise ValueError(f"{path}: the step times must increase strictly")
    terminal = case.terminal_time_kyr * 1.0e3
    if abs(years[-1] - terminal) > TERMINAL_TIME_TOLERANCE_YR:
        raise ValueError(f"{path}: the last step time is {years[-1]} yr, and case "
                         f"{case.letter} ends at "
                         f"{terminal} yr; the profiles are read at the last step")
    # Put the terminal time exactly on the case's value, so that a rounding
    # difference in the file does not move the last kernel lag.
    years[-1] = terminal
    return np.concatenate([[-(years[1] - years[0])], years])


def uniform_step_times(case: MartinecCase, step_years: float) -> np.ndarray:
    """The uniform ladder of the published runs, in years.

    One step before the load, then the grid from t = 0 to the terminal time.
    An increment carries its viscous response only at steps after the one it
    is applied at, so the unloaded step has to be there.
    """
    terminal = case.terminal_time_kyr * 1.0e3
    loaded_times = np.arange(0.0, terminal + 0.5 * step_years, step_years)
    return np.concatenate([[-step_years], loaded_times])


def read_ocean_function(ocean_function_path, grid) -> np.ndarray:
    """Read a prescribed ocean function on the run's Gauss-Legendre grid.

    The file is the export of the G-ADOPT driver, `ocean_function_C0.npz`, with
    three arrays: `colatitude_deg` (the `nglv` Gauss-Legendre colatitudes, north
    to south), `longitude_deg` (the `2 nglv` longitudes, east from 0) and `C0`,
    shape `(nglv, 2 nglv)`, 1 in the ocean. The coordinates are compared with
    the run's grid, so that a mask of another grid, or with its rows reversed,
    is refused instead of being read as if it matched. The values are checked
    by `solve_sea_level`.

    Parameters
    ----------
    ocean_function_path
        The npz file.
    grid
        The `GaussLegendreGrid` of the run.

    Returns
    -------
    The mask as a `(nglv, 2 nglv)` float array.
    """
    path = pathlib.Path(ocean_function_path)
    with np.load(path) as data:
        missing = {"colatitude_deg", "longitude_deg", "C0"} - set(data.files)
        if missing:
            raise ValueError(f"{path} lacks the arrays {sorted(missing)}")
        colatitude = np.asarray(data["colatitude_deg"], dtype=float)
        longitude = np.asarray(data["longitude_deg"], dtype=float)
        mask = np.asarray(data["C0"], dtype=float)
    want_colatitude = 90.0 - np.asarray(grid.latitude_deg, dtype=float)
    want_longitude = np.asarray(grid.longitude_deg, dtype=float)
    if colatitude.shape != want_colatitude.shape or longitude.shape != want_longitude.shape:
        raise ValueError(f"{path} is on a grid of {colatitude.size} by {longitude.size} "
                         f"points, and the run's grid is {want_colatitude.size} by "
                         f"{want_longitude.size}")
    worst = max(np.max(np.abs(colatitude - want_colatitude)),
                np.max(np.abs(longitude - want_longitude)))
    if worst > GRID_COORDINATE_TOLERANCE_DEG:
        raise ValueError(f"{path}: the coordinates differ from the run's grid by up to "
                         f"{worst} degrees")
    if mask.shape != (want_colatitude.size, want_longitude.size):
        raise ValueError(f"{path}: C0 has shape {mask.shape}, expected "
                         f"{(want_colatitude.size, want_longitude.size)}")
    return mask


def mask_band_limit_error(grid, mask: np.ndarray) -> dict:
    """How far the mask is from its own expansion at the run's truncation.

    The solver multiplies grid fields by the mask point by point and never
    truncates the mask itself, but every product is analysed at `norder`, and
    a mask that is not band limited at `norder` aliases part of its tail onto
    the retained degrees. The size of that tail is the difference between the
    mask and the synthesis of its analysis. The ocean area, the `(0, 0)`
    coefficient, is exact by quadrature and is not affected.

    Returns
    -------
    A dictionary with `max_abs` and `rms_area` (area weighted) of the
    difference, dimensionless, and `ocean_area`, the area mean of the mask.
    """
    field = jnp.asarray(mask)
    difference = np.asarray(synthesis(grid, analysis(grid, field))) - mask
    return {"max_abs": float(np.max(np.abs(difference))),
            "rms_area": float(np.sqrt(np.asarray(area_mean(grid, jnp.asarray(difference ** 2))))),
            "ocean_area": float(np.asarray(area_mean(grid, field)))}


def run_fixed_coastline_case(letter: str, *, norder: int = 256, nglv: int = 512,
                             step_years: float = 100.0, nodes_per_layer: int = 40,
                             substeps: int = 2, numouter: int = 1,
                             ocean_function_path=None, times_path=None,
                             member: str = "radopt", verbose: bool = True) -> BenchmarkResult:
    """Benchmark B or C: a fixed coastline, with U, N and S at the terminal time.

    Both cases solve the sea-level equation with the ocean geometry held at its
    initial value (`fixed_ocean`) and the topography prescribed at the initial
    time (`topography_epoch="initial"`), which is the `SLE = 1` level of the
    paper. They differ in the ice: case B applies a thick cap inland as a step
    and case C grows a thinner one until it reaches the coast of a nearby
    basin, where the fixed coastline calves it.

    Under those two settings nothing inside the time loop depends on the
    sea-level field of the previous outer pass: the ocean function never
    changes, the prescribed topography is the one at the first step, and the
    shoreline term of eq. 70 is zero. One outer pass is therefore the whole
    answer, and `numouter` is exposed so that a second pass can confirm it.

    The time step is a real approximation in both, for two different reasons.
    Case B applies its ice as a step, but the water that ice moves is not a
    step: the ocean load changes at every step as the earth relaxes under it.
    Case C also grows its ice through time, so the load itself is a staircase.
    The first step sits one step before t = 0 so that a step load is applied at
    t = 0 exactly and carries its viscous response from the next step onward.

    Parameters
    ----------
    letter
        ``"B"`` or ``"C"``.
    nglv
        Gauss-Legendre latitudes. The default 512 is above the 288 that the
        truncation alone needs, and the extra rows are for the coastline: the
        ocean function is a step, its position quantises to the grid, and the
        vertical displacement changes by 15 m across 1.5 degrees at the basin
        edge. Going from 288 to 512 latitudes cuts the largest difference from
        VEGA on the basin profile from 1.53 m to 0.75 m and leaves every other
        number where it was. VEGA itself ran 1024 latitudes at this truncation.

        The profiles are read with `sht.evaluate`, which sums the series at the
        requested points, so no meridian has to land on a grid column.
    ocean_function_path
        None for the sharp mask `zeta0 < 0` that radopt forms itself, or the
        npz export of a prescribed mask `C0` (`read_ocean_function`). For a
        matched run this is the G-ADOPT mask, and the run then solves the
        smooth-mask equation of G-ADOPT. The distance of the mask from its
        expansion at `norder` is printed (`mask_band_limit_error`).
    times_path
        None for the uniform step `step_years`, or a ladder file of step times
        (`read_step_times`), for example the G-ADOPT ladder. `step_years` is
        then not read.
    member
        The member name of the result, `"radopt"` for the published run. A
        matched run takes another name, for example `"radopt-matched"`.
    verbose
        Print the distance of a prescribed mask from its expansion at `norder`
        (`mask_band_limit_error`). With False the run skips this computation
        and its print. Without a prescribed mask it has no effect.
    """
    if letter not in ("B", "C"):
        raise ValueError(f"only cases B and C have a fixed coastline, not {letter!r}")
    case = MartinecCase(letter)
    solver, rheology = earth_model(case, nodes_per_layer=nodes_per_layer, substeps=substeps)
    consts = constants(case, solver)
    grid = _grid_for(nglv, norder)

    # The step times: the uniform ladder of the published run, or a prescribed
    # ladder, both with one unloaded step before t = 0.
    if times_path is None:
        times = uniform_step_times(case, step_years)
    else:
        times = read_step_times(times_path, case)
    nsteps = times.size

    # The prescribed ocean function, if any, read and checked against the grid
    # before the ice history is built, so that a wrong file stops the run early.
    ocean_function = None
    if ocean_function_path is not None:
        ocean_function = read_ocean_function(ocean_function_path, grid)
        if verbose:
            error = mask_band_limit_error(grid, ocean_function)
            print(f"case {letter}: prescribed ocean function {ocean_function_path}, "
                  f"ocean area {error['ocean_area']:.7f}; distance from its degree "
                  f"{norder} expansion: max {error['max_abs']:.3e}, "
                  f"area rms {error['rms_area']:.3e}")

    # The ice at every step, from the cap of that step. `cap_size` gives the
    # step of scenario T0 and the linear growth of T1, so one expression covers
    # both cases; the step at t = 0 takes the right limit of the scenario
    # (`ice_on_grid_at_step`).
    ice = np.zeros((nsteps, nglv, 2 * nglv))
    for step, year in enumerate(times):
        ice[step] = ice_on_grid_at_step(case, grid, norder, year)

    # The initial topography, eq. 29, sampled on the grid. It enters only
    # through the ocean function, which is a threshold, so the pointwise values
    # are what it wants and no band limit applies.
    colatitude = 90.0 - grid.latitude_deg
    longitude = grid.longitude_deg
    colat2d, lon2d = np.meshgrid(colatitude, longitude, indexing="ij")
    topography = case.topography(colat2d, lon2d)

    config = SeaLevelConfig(norder=norder, nglv=nglv, nsteps=nsteps,
                            checkmarine=0, tpw=0, numouter=numouter, calcRG=1,
                            fixed_ocean=1, topography_epoch="initial")
    spectra = solver.spectra(rheology, np.arange(1, norder + 1))
    # The centre of mass of the whole system, earth plus ice plus ocean, which
    # is the frame the paper poses every case in (p. 390). In it the surface
    # potential perturbation of degree 1 is zero, so `1 + k` vanishes there and
    # the degree-1 geoid is zero. The default frame of the seam sets `k(1) = 0`
    # instead, which is what the Fortran consumes, and the two answers differ by
    # one rigid translation of the degree-1 load in U and in N.
    kernel = spectra.kernel(norder, degree_one_frame=DegreeOneFrame.KERNEL)
    result = solve_sea_level(config, kernel, ice, times, topography, consts=consts,
                             ocean_function=ocean_function)

    # The three fields of the paper at the terminal time. U is the radial
    # displacement and S is the geoid field of the solver, which is
    # `SL + R`, the change of the sea surface since the first step. The geoid
    # displacement N is S minus the uniform water layer: eq. 6 and eq. 8 give
    # S = h_UF + N over the ocean, and the solver's uniform term `conserv` of
    # eq. 78 is h_UF.
    h_uf = float(np.asarray(result.diagnostics.conserv)[-1])
    fields = {
        "U": np.asarray(result.radial[-1]),
        "S": np.asarray(result.geoid[-1]),
        "N": np.asarray(result.geoid[-1]) - h_uf,
    }

    out = BenchmarkResult(letter, member)
    for spec in case.profiles:
        reference = BenchmarkResult.reference(letter, "ZM")
        axis = np.asarray(reference.profile(spec.name, spec.quantities[0]).axis)
        for quantity in spec.quantities:
            # Analysed once and summed at the requested colatitudes, which is
            # exact for a field the grid carries and is not an interpolation
            # between rows.
            coefficients = analysis(grid, jnp.asarray(fields[quantity]))
            values = np.asarray(evaluate(coefficients, axis, spec.longitude_deg))
            out.add_profile(spec.name, quantity, axis, values,
                            time_kyr=spec.time_kyr, longitude_deg=spec.longitude_deg)
    out.add_time_series("h_UF", times / 1.0e3, np.asarray(result.diagnostics.conserv))
    out.add_time_series("ocean_area", times / 1.0e3, np.asarray(result.diagnostics.cstar00))
    out.add_time_series("ice_mass_grounded", times / 1.0e3,
                        (4.0 * np.pi * consts.Re ** 2 * case.ice_density
                         * np.asarray(result.diagnostics.istar00)))
    return out


# ---------------------------------------------------------------------------
# Cases D and E: the coastline moves and marine ice floats
# ---------------------------------------------------------------------------


def _moving_coastline_solution(topography: np.ndarray, *, consts: LegacyConstants,
                               kernel, times: np.ndarray, ice: np.ndarray,
                               norder: int, nglv: int, numouter: int):
    """Solve one moving-coastline run on a prescribed initial topography.

    This is the `SLE = 2` level of the paper: the ocean function is formed from
    the topography of every step instead of being frozen at the first one
    (`fixed_ocean=0`), and ice over marine bedrock is a load only when it is
    thick enough to stay grounded (`checkmarine=1`). The topography given here
    is the one of the first step (`topography_epoch="initial"`), which is what
    case D prescribes and what each iteration of case E supplies.

    The two settings make the outer iteration real. Every step's topography is
    the first step's plus the sea level that has accumulated since (eq. 39), so
    the ocean function and the flotation test of step `n` both read the
    solution of the previous outer pass. `numouter` therefore has to be raised
    until the answer stops moving; one pass is no longer the whole answer as it
    was in cases B and C.

    Parameters
    ----------
    topography
        `(nglv, 2 nglv)` solid Earth topography at the first step, metres,
        positive up.
    ice, times, consts, kernel
        The same objects the fixed-coastline driver builds, passed in so that
        the case E iteration can reuse them across its passes.

    Returns
    -------
    The `SeaLevelResult` of the run.
    """
    config = SeaLevelConfig(norder=norder, nglv=nglv, nsteps=times.size,
                            checkmarine=1, tpw=0, numouter=numouter, calcRG=1,
                            fixed_ocean=0, topography_epoch="initial")
    return solve_sea_level(config, kernel, ice, times, topography, consts=consts)


def _moving_setup(case: MartinecCase, *, norder: int, nglv: int, step_years: float,
                  nodes_per_layer: int, substeps: int, times_path=None):
    """Everything a moving-coastline run needs except the topography.

    The earth model, the constants, the transform, the step times, the ice on
    the grid at every step and the Love-number kernel in the frame the paper
    poses the cases in. Case E runs the solver several times on the same ice
    and the same kernel, so these are built once and handed to each pass.

    `times_path` is None for the uniform step `step_years`, or a ladder file of
    step times (`read_step_times`), for the case D run matched to G-ADOPT.
    """
    solver, rheology = earth_model(case, nodes_per_layer=nodes_per_layer, substeps=substeps)
    consts = constants(case, solver)
    grid = _grid_for(nglv, norder)

    # The step times, with one unloaded step before t = 0 (`uniform_step_times`).
    if times_path is None:
        times = uniform_step_times(case, step_years)
    else:
        times = read_step_times(times_path, case)

    # The whole cap at every step, with nothing removed at the coast: in cases
    # D and E the flotation test inside the solver decides what is a load, so
    # the ice handed over is the full cap of scenario T1.
    ice = np.zeros((times.size, nglv, 2 * nglv))
    for step, year in enumerate(times):
        ice[step] = ice_on_grid_at_step(case, grid, norder, year)

    spectra = solver.spectra(rheology, np.arange(1, norder + 1))
    # The centre of mass of the whole system, as in cases B and C.
    kernel = spectra.kernel(norder, degree_one_frame=DegreeOneFrame.KERNEL)
    return {"consts": consts, "grid": grid, "times": times, "ice": ice, "kernel": kernel}


def _basin_topography(case: MartinecCase, grid) -> np.ndarray:
    """The basin of eq. 29 sampled pointwise on the Gauss-Legendre grid, metres.

    It enters the solver only through a threshold, so the pointwise values are
    what it wants and no band limit applies. For case E this is the target at
    the terminal time and not the topography of the first step, which is what
    `MartinecCase.topography_time` records.
    """
    colatitude = 90.0 - grid.latitude_deg
    colat2d, lon2d = np.meshgrid(colatitude, grid.longitude_deg, indexing="ij")
    return case.topography(colat2d, lon2d)


def _profiles_and_series(case: MartinecCase, letter: str, result, times, grid,
                         consts: LegacyConstants, member: str = "radopt") -> BenchmarkResult:
    """Pack a solved run into the `BenchmarkResult` the comparison reads.

    The three profile fields at the terminal time and the four time series of
    the paper's figures 13, 16 and 17. `N` is the geoid field of the solver
    minus the uniform water layer, exactly as in cases B and C: eq. 6 and eq. 8
    give `S = h_UF + N` over the ocean, and the solver's `conserv` of eq. 78 is
    `h_UF`.

    The floating ice mass is the ice the flotation test removed: `ice00` is the
    area mean of the ice field before the test and `istar00` the area mean of
    what survived it, so their difference times `4 pi a^2 rho_ice` is the mass
    that floats.
    """
    h_uf = float(np.asarray(result.diagnostics.conserv)[-1])
    fields = {
        "U": np.asarray(result.radial[-1]),
        "S": np.asarray(result.geoid[-1]),
        "N": np.asarray(result.geoid[-1]) - h_uf,
    }
    out = BenchmarkResult(letter, member)
    for spec in case.profiles:
        reference = BenchmarkResult.reference(letter, "ZM")
        axis = np.asarray(reference.profile(spec.name, spec.quantities[0]).axis)
        for quantity in spec.quantities:
            coefficients = analysis(grid, jnp.asarray(fields[quantity]))
            values = np.asarray(evaluate(coefficients, axis, spec.longitude_deg))
            out.add_profile(spec.name, quantity, axis, values,
                            time_kyr=spec.time_kyr, longitude_deg=spec.longitude_deg)

    kyr = times / 1.0e3
    mass_factor = 4.0 * np.pi * consts.Re ** 2 * case.ice_density
    grounded = np.asarray(result.diagnostics.istar00)
    raw = np.asarray(result.diagnostics.ice00)
    out.add_time_series("h_UF", kyr, np.asarray(result.diagnostics.conserv))
    out.add_time_series("ocean_area", kyr, np.asarray(result.diagnostics.cstar00))
    out.add_time_series("ice_mass_grounded", kyr, mass_factor * grounded)
    out.add_time_series("ice_mass_floating", kyr, mass_factor * (raw - grounded))
    return out


def run_case_d(*, norder: int = 256, nglv: int = 512, step_years: float = 100.0,
               nodes_per_layer: int = 40, substeps: int = 2,
               numouter: int = 3, times_path=None,
               member: str = "radopt") -> BenchmarkResult:
    """Benchmark D: the coastline moves and marine ice floats.

    The same L2 cap, the same T1 growth and the same B2 basin as case C. What
    changes is the sea-level equation: the ocean function is rebuilt at every
    step from that step's topography (eq. 23) and ice over marine bedrock is a
    load only where it is thick enough to stay grounded (eq. 43). Ice that
    floats displaces water instead of loading the bedrock, and the coast can
    move over the shelf, which the paper reports as a difference of about 10
    per cent from case C in U and N.

    Kendall's ocean function and the paper's eq. 23 are the same mask. The
    topography of a step is `zeta_0 - SL(n)`, and `SL = S - U = N + h_UF - U`,
    so the solver's topography is the negative of the water column `h_w` of
    eq. 15. Its ocean test `topo < 0` is therefore `h_w > 0`, and the ice-free
    factor `beta` removes exactly the cells whose ice is grounded, which is
    `h_w* <= 0`. The ocean area against figure 17 is the measurement of that.

    `numouter` is a count of outer passes and not a convergence test. Three is
    the default because the profiles stop moving there; raise it to confirm.

    `times_path` replaces the uniform step with a ladder file
    (`read_step_times`), and `member` names the result. The run matched to
    G-ADOPT uses both and keeps the sharp mask: the coastline moves, so no mask
    can be prescribed, and the mask width stays in the G-ADOPT difference.
    """
    case = MartinecCase("D")
    if case.topography_time != "initial":
        raise ValueError("case D must prescribe its topography at the first step; "
                         f"the case file says {case.topography_time!r}, which needs "
                         "the iteration of run_case_e instead")
    setup = _moving_setup(case, norder=norder, nglv=nglv, step_years=step_years,
                          nodes_per_layer=nodes_per_layer, substeps=substeps,
                          times_path=times_path)
    topography = _basin_topography(case, setup["grid"])
    result = _moving_coastline_solution(
        topography, consts=setup["consts"], kernel=setup["kernel"],
        times=setup["times"], ice=setup["ice"],
        norder=norder, nglv=nglv, numouter=numouter)
    return _profiles_and_series(case, "D", result, setup["times"], setup["grid"],
                                setup["consts"], member=member)


def _l2_norm(grid, field: np.ndarray) -> dict:
    """Three readings of "the L2 norm" of a field on the sphere, in its units.

    The paper states the case E stopping test as an L2 norm below 0.01 m and
    does not say how the norm is normalised, which
    `docs/protocol/martinec2018.qmd` records as open value 6. Three readings
    are in use and they differ by a factor of 3.5, which is more than enough to
    decide whether a given iteration passes:

    * `mean_area`, the area-weighted root mean square
      `sqrt((1 / 4 pi) int f^2 dOmega)`;
    * `mean_grid`, the plain root mean square over the grid cells, which
      differs from the first only by the latitude weighting;
    * `integral`, the literal `sqrt(int f^2 dOmega)`, which is `sqrt(4 pi)`
      times the first and so is the largest of the three.

    All three are returned, and the caller stops on `integral`, so that the run
    satisfies the stopping test under every reading and not only the loosest.
    """
    mean_area = float(np.sqrt(np.asarray(area_mean(grid, jnp.asarray(field ** 2)))))
    return {"mean_area": mean_area,
            "mean_grid": float(np.sqrt(np.mean(field ** 2))),
            "integral": mean_area * float(np.sqrt(4.0 * np.pi))}


def run_case_e(*, norder: int = 256, nglv: int = 512, step_years: float = 100.0,
               nodes_per_layer: int = 40, substeps: int = 2, numouter: int = 3,
               max_iterations: int = 8, tolerance_m: float = 0.01,
               verbose: bool = True):
    """Benchmark E: the topography is prescribed at the terminal time.

    Case D prescribes the basin of eq. 29 at the first step. Case E prescribes
    the same shape at the terminal time, eq. 32, and the topography of the
    first step becomes the unknown. The two are not related by one subtraction,
    because the ocean geometry of every step is built from the first step's
    topography, so the whole run has to be repeated on an updated estimate.
    The paper's update is eq. 34,

        zeta_0(k+1) = zeta_0(k) + zeta_des - zeta_t(k),

    with the terminal topography of run `k` given by eq. 33,

        zeta_t(k) = zeta_0(k) + U(t_f) - N(t_f) - h_UF(t_f),

    which is `zeta_0(k) - SL(t_f)` in the solver's own fields, since
    `SL = S - U` and `N = S - h_UF`. The iteration stops when the norm of
    `zeta_des - zeta_t(k)` falls below `tolerance_m`. The paper reaches 0.01 m
    in at most six iterations (p. 409).

    The first estimate is the target itself, `zeta_0(0) = zeta_des`. The paper
    does not state what it started from (open value 6 of the protocol), and the
    fixed point does not depend on the start.

    The stopping test uses the largest of the three readings of the norm that
    `_l2_norm` returns, so that the run passes under every reading. Stopping on
    the area-weighted mean instead would end the iteration one pass early: at
    the production settings that mean is 0.0034 m at the fourth pass while the
    integral is 0.0121 m, which is above the paper's 0.01 m.

    Each iteration is a full case D run, so the cost is `max_iterations` times
    that of case D. Only the ocean area and `h_UF` are compared for this case.

    Returns
    -------
    result
        The `BenchmarkResult` of the last iteration.
    history
        One dictionary per iteration with the two residual norms, so that the
        stopping test can be reported instead of asserted.
    """
    case = MartinecCase("E")
    if case.topography_time != "terminal":
        raise ValueError("case E must prescribe its topography at the terminal time")
    setup = _moving_setup(case, norder=norder, nglv=nglv, step_years=step_years,
                          nodes_per_layer=nodes_per_layer, substeps=substeps)
    grid, consts, times = setup["grid"], setup["consts"], setup["times"]

    target = _basin_topography(case, grid)          # zeta_des at the terminal time, eq. 32
    initial = target.copy()                         # zeta_0(0), the first estimate
    history = []
    result = None
    for iteration in range(max_iterations):
        solution = _moving_coastline_solution(
            initial, consts=consts, kernel=setup["kernel"], times=times,
            ice=setup["ice"], norder=norder, nglv=nglv, numouter=numouter)
        # SL(t_f) = G(t_f) - R(t_f); eq. 33 is zeta_t = zeta_0 - SL(t_f).
        sea_level = np.asarray(solution.geoid[-1]) - np.asarray(solution.radial[-1])
        terminal_topography = initial - sea_level
        residual = target - terminal_topography
        norms = _l2_norm(grid, residual)
        history.append({"iteration": iteration, "max_m": float(np.max(np.abs(residual))),
                        **{f"l2_{k}_m": v for k, v in norms.items()}})
        if verbose:
            print(f"case E iteration {iteration}: residual L2 "
                  f"{norms['integral']:.5f} m as the integral, "
                  f"{norms['mean_area']:.5f} m area weighted, "
                  f"{norms['mean_grid']:.5f} m over the grid cells, "
                  f"max {history[-1]['max_m']:.4f} m")
        result = solution
        # The largest of the three readings, so that the run passes the paper's
        # test however the norm is meant.
        if norms["integral"] < tolerance_m:
            break
        initial = initial + residual                # eq. 34
    else:
        if verbose:
            print(f"case E did not reach {tolerance_m} m in {max_iterations} iterations")

    out = _profiles_and_series(case, "E", result, times, grid, consts)
    return out, history


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=["A", "B", "C", "D", "E"],
                        help="the published case letter")
    parser.add_argument("--norder", type=int, default=256,
                        help="spherical harmonic truncation (the paper's VEGA run used 256)")
    parser.add_argument("--nglv", type=int, default=512,
                        help="Gauss-Legendre latitudes; the grid is nglv by 2 nglv (case B)")
    parser.add_argument("--step-years", type=float, default=100.0,
                        help="time step of the sea-level run in years (cases B to E)")
    parser.add_argument("--nodes-per-layer", type=int, default=40,
                        help="radial nodes inside each layer of the earth model")
    parser.add_argument("--substeps", type=int, default=2,
                        help="integration substeps per radial interval")
    parser.add_argument("--numouter", type=int, default=3,
                        help="outer topography passes; real for cases D and E, "
                             "where every step's ocean geometry reads the previous pass")
    parser.add_argument("--max-iterations", type=int, default=8,
                        help="most iterations of the case E topography update (eq. 34)")
    parser.add_argument("--ocean-function", type=pathlib.Path, default=None,
                        help="npz with colatitude_deg, longitude_deg and C0 on the "
                             "Gauss-Legendre grid: a prescribed ocean function (cases B, C)")
    parser.add_argument("--times", type=pathlib.Path, default=None,
                        help="a ladder of step times from t = 0 to the terminal time: "
                             "text or .npy in years, or .npz with t_kyr (cases B, C, D)")
    # The default is None and not "radopt", so that an explicit --member is
    # visible to the refusal below for the cases whose runners take no member.
    parser.add_argument("--member", default=None,
                        help="member name of the result, radopt by default; a matched "
                             "run takes another name, for example radopt-matched "
                             "(cases B, C, D)")
    parser.add_argument("--output", type=pathlib.Path, default=None,
                        help="where to write the result; the default is "
                             "outputs/martinec2018/<member>/")
    args = parser.parse_args(argv)

    # The options of the matched runs apply only to the cases that can use
    # them, and are refused for the others instead of being ignored.
    if args.ocean_function is not None and args.case not in ("B", "C"):
        parser.error("--ocean-function applies to the fixed-coastline cases B and C only")
    if args.times is not None and args.case not in ("B", "C", "D"):
        parser.error("--times applies to cases B, C and D only")
    # `run_case_a` and `run_case_e` write the member "radopt" and take no member
    # name. A named member for them would put a "radopt" result under another
    # output directory, so it is refused.
    if args.member is not None and args.case not in ("B", "C", "D"):
        parser.error("--member applies to cases B, C and D only")
    member = args.member if args.member is not None else "radopt"

    if args.case == "A":
        result = run_case_a(norder=args.norder, nodes_per_layer=args.nodes_per_layer,
                            substeps=args.substeps)
    elif args.case in ("B", "C"):
        result = run_fixed_coastline_case(args.case, norder=args.norder, nglv=args.nglv,
                                          step_years=args.step_years,
                                          nodes_per_layer=args.nodes_per_layer,
                                          substeps=args.substeps,
                                          ocean_function_path=args.ocean_function,
                                          times_path=args.times, member=member)
    elif args.case == "D":
        result = run_case_d(norder=args.norder, nglv=args.nglv,
                            step_years=args.step_years,
                            nodes_per_layer=args.nodes_per_layer,
                            substeps=args.substeps, numouter=args.numouter,
                            times_path=args.times, member=member)
    else:
        result, _ = run_case_e(norder=args.norder, nglv=args.nglv,
                               step_years=args.step_years,
                               nodes_per_layer=args.nodes_per_layer,
                               substeps=args.substeps, numouter=args.numouter,
                               max_iterations=args.max_iterations)
    path = args.output or (REPOSITORY / "outputs" / "martinec2018" / member
                           / f"martinec2018-{args.case}.nc")
    path.parent.mkdir(parents=True, exist_ok=True)
    result.write(path)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
