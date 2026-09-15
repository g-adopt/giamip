"""Everything a solver needs to set up a benchmark case.

`MartinecCase` sets up one Martinec et al. (2018) case, and `SpadaCase`, at the
end of this module, one Spada et al. (2011) test.

A solver, radopt or G-ADOPT, needs the same five things for every case: the
earth model, the ice load as a function of position and time, the initial
topography, the physical constants, and the list of what to output where. The
case files, the load table and the earth model file record all of them, and
`MartinecCase` turns them into numbers, so no solver re-types a parameter.

The class evaluates the load and the topography at any points the caller
chooses. radopt passes the nodes of its Gauss-Legendre grid; G-ADOPT passes
the coordinates of its surface mesh. Nothing here assumes a grid, and nothing
here computes a response.

    >>> from gia_mip.benchmarks.inputs import MartinecCase
    >>> case = MartinecCase("C")
    >>> case.cap_size(5.0)          # half-grown in scenario T1
    (250.0, 5.0)
    >>> round(case.ice_mass(15.0) / 1e18, 4)
    1.2024

Conventions
-----------
Angles are in degrees, colatitude from the north pole and longitude east.
Time is in kyr since the load was first applied, t = 0. Heights are in metres.
Densities come from ``loads/martinec2018.json`` and never from a module
constant, because GIAMIP uses different values.
"""

from __future__ import annotations

import dataclasses
import math
import re

import numpy as np

from gia_mip import cases

COLLECTION = "martinec2018"


@dataclasses.dataclass(frozen=True)
class ProfileSpec:
    """One cross-section the paper compares.

    Attributes
    ----------
    name
        ``load`` for the meridian through the ice centre, ``basin`` for the
        meridian through the ocean basin centre. These are the group names of
        the converted data files.
    longitude_deg
        The meridian.
    quantities
        The fields compared on it: some of ``U``, ``N`` and ``S``.
    colatitude_interval_deg
        The interval over which the differences are measured.
    time_kyr
        The epoch, which is the terminal time of the case.
    figure
        The figure of the paper that shows it.
    """

    name: str
    longitude_deg: float
    quantities: tuple[str, ...]
    colatitude_interval_deg: tuple[float, float]
    time_kyr: float
    figure: int


@dataclasses.dataclass(frozen=True)
class TimeSeriesSpec:
    """One time series the paper compares, with its interval and figure."""

    quantities: tuple[str, ...]
    time_interval_kyr: tuple[float, float]
    figure: int
    note: str | None = None


def angular_distance_deg(colatitude_deg, longitude_deg, centre_colatitude_deg,
                         centre_longitude_deg) -> np.ndarray:
    """Great-circle angle between points and a centre, in degrees.

    cos(psi) = cos(theta) cos(theta_c) + sin(theta) sin(theta_c) cos(lambda - lambda_c),
    the relation the paper uses for the basin (eq. 30). The cosine is clipped
    to [-1, 1], because rounding can push it just outside at the centre and the
    antipode, where arccos would return NaN.
    """
    theta = np.radians(np.asarray(colatitude_deg, dtype=float))
    lam = np.radians(np.asarray(longitude_deg, dtype=float))
    theta_c = np.radians(centre_colatitude_deg)
    lam_c = np.radians(centre_longitude_deg)
    cos_psi = (np.cos(theta) * np.cos(theta_c)
               + np.sin(theta) * np.sin(theta_c) * np.cos(lam - lam_c))
    return np.degrees(np.arccos(np.clip(cos_psi, -1.0, 1.0)))


class MartinecCase:
    """The set-up of one Martinec et al. (2018) case, A to E.

    Parameters
    ----------
    case
        The published case letter, ``"C"``, or the case id, ``"martinec2018-C"``.
    """

    def __init__(self, case: str):
        letter = case.rsplit("-", 1)[-1].upper()
        self.letter = letter
        self.definition = cases.case(f"{COLLECTION}-{letter}")
        self.load_table = cases.loads(COLLECTION)
        self.earth_model = cases.earth_model(self.definition["earth_model"])
        spec = self.definition["specification"]
        # The option levels of table 4 of the paper.
        self.sea_level_equation: int = spec["sea_level_equation"]
        self.ice_height_model: str = spec["ice_height_model"]
        self.time_scenario: str = spec["time_scenario"]
        self.ocean_basin: str = spec["ocean_basin"]
        self.terminal_time_kyr: float = float(spec["terminal_time_kyr"])
        constants = self.load_table["constants"]
        self.ice_density: float = float(constants["ice_density_kg_m3"])
        self.water_density: float = float(constants["water_density_kg_m3"])
        self.surface_gravity: float = float(constants["surface_gravity_m_s2"])
        # The earth radius is the top of the outermost layer of M3-L70-V01.
        self.earth_radius_m: float = self.earth_model["layers"][0]["top_radius_km"] * 1e3

    def __repr__(self) -> str:
        return (f"MartinecCase({self.letter!r}: SLE={self.sea_level_equation}, "
                f"{self.ice_height_model}, {self.time_scenario}, {self.ocean_basin}, "
                f"t_f={self.terminal_time_kyr:g} kyr)")

    # ------------------------------------------------------------------ load

    @property
    def cap(self) -> dict:
        """Centre, full height and full angular radius of the ice cap (table 1)."""
        model = self.load_table["ice_height_models"][self.ice_height_model]
        return {
            "centre_colatitude_deg": float(model["centre_colatitude_deg"]),
            "centre_longitude_deg": float(model["centre_longitude_deg"]),
            "height_m": float(model["height_m"]),
            "radius_deg": float(self.load_table["load_shape"]["angular_radius_deg"]),
        }

    def cap_size(self, t_kyr: float) -> tuple[float, float]:
        """Height h(t) in metres and angular radius alpha(t) in degrees (table 2).

        T0 is a step: the full cap for t > 0 and no cap for t <= 0. T1 grows
        both h and alpha linearly from zero at t = 0 to their full values at
        the end of growth, 10 kyr, and holds them after that.

        The paper does not say how the cap profile scales while alpha grows
        (docs/protocol/martinec2018.qmd, open value 2). This class keeps the
        profile self-similar: the same square-root shape on the current radius
        and height. That gives an ice mass proportional to h alpha^2, which
        grows as the cube of time, as the paper reports for case C.
        """
        cap = self.cap
        scenario = self.load_table["time_scenarios"][self.time_scenario]
        t = float(t_kyr)
        if t <= 0.0:
            return 0.0, 0.0
        if self.time_scenario == "T0":
            return cap["height_m"], cap["radius_deg"]
        growth_end = float(scenario["growth_end_kyr"])
        fraction = min(t / growth_end, 1.0)
        return cap["height_m"] * fraction, cap["radius_deg"] * fraction

    def ice_thickness(self, colatitude_deg, longitude_deg, t_kyr: float) -> np.ndarray:
        """Ice thickness in metres at the given points and time.

        The cap of Spada et al. (2011), table 4, row 1:
        h(psi) = h(t) sqrt((cos psi - cos alpha(t)) / (1 - cos alpha(t))) for
        psi < alpha(t), and zero beyond, with psi the angle from the cap centre.
        The square root gives the infinite slope at the edge that the paper
        describes, and it is the profile whose Legendre coefficients
        loads/spada2011.json records.

        Parameters
        ----------
        colatitude_deg, longitude_deg
            Arrays of the same shape, or scalars.
        t_kyr
            Time since loading.

        Returns
        -------
        numpy.ndarray
            The thickness, broadcast to the shape of the inputs.
        """
        h, alpha = self.cap_size(t_kyr)
        cap = self.cap
        psi = angular_distance_deg(colatitude_deg, longitude_deg,
                                   cap["centre_colatitude_deg"], cap["centre_longitude_deg"])
        if alpha <= 0.0:
            return np.zeros_like(psi)
        cos_alpha = np.cos(np.radians(alpha))
        ratio = (np.cos(np.radians(psi)) - cos_alpha) / (1.0 - cos_alpha)
        return h * np.sqrt(np.clip(ratio, 0.0, None))

    def ice_on_grid(self, latitudes_deg, longitudes_deg, times_kyr) -> np.ndarray:
        """Ice thickness on a latitude-longitude grid, shaped (times, lat, lon).

        The rows follow the order of ``latitudes_deg``. radopt wants row 0 at
        the north, so pass its latitudes north first; GIAMIP files store them
        south first.
        """
        lat = np.asarray(latitudes_deg, dtype=float)
        lon = np.asarray(longitudes_deg, dtype=float)
        colat2d, lon2d = np.meshgrid(90.0 - lat, lon, indexing="ij")
        return np.stack([self.ice_thickness(colat2d, lon2d, t) for t in np.atleast_1d(times_kyr)])

    def ice_mass(self, t_kyr: float) -> float:
        """Total mass of the cap in kg, from the closed form of the profile.

        The integral of rho_ice h sqrt((cos psi - cos alpha)/(1 - cos alpha))
        over the cap is rho_ice h 2 pi a^2 (1 - cos alpha) 2/3. This is the
        mass before any ice is removed or floats, and it equals VEGA's total
        ice mass in cases D and E.
        """
        h, alpha = self.cap_size(t_kyr)
        if alpha <= 0.0:
            return 0.0
        return (self.ice_density * h * 2.0 * np.pi * self.earth_radius_m ** 2
                * (1.0 - np.cos(np.radians(alpha))) * 2.0 / 3.0)

    # ----------------------------------------------------------- topography

    @property
    def basin(self) -> dict:
        """Parameters of the ocean basin (table 3). B2* uses the B2 values."""
        table = self.load_table["ocean_basins"]
        basin = dict(table[self.ocean_basin.rstrip("*")])
        basin["sigma_b_deg"] = float(table["sigma_b_deg"])
        return basin

    @property
    def topography_time(self) -> str:
        """``initial`` when the basin gives the topography at t = 0, ``terminal`` for B2*.

        In case E the basin formula prescribes the topography at the terminal
        time, and the initial topography is the unknown the solver iterates for.
        """
        return "terminal" if self.ocean_basin.endswith("*") else "initial"

    def basin_radius_deg(self) -> float:
        """Angle from the basin centre at which the topography crosses zero.

        sigma_b sqrt(2 ln(b0 / bmax)): 24.8503 degrees for B1 and B2. Zero
        for B0, which has no ocean.
        """
        b = self.basin
        if b["b0_m"] <= b["bmax_m"]:
            return 0.0
        return b["sigma_b_deg"] * float(np.sqrt(2.0 * np.log(b["b0_m"] / b["bmax_m"])))

    def topography(self, colatitude_deg, longitude_deg) -> np.ndarray:
        """Solid Earth topography zeta in metres, positive up (eq. 29).

        zeta = bmax - b0 exp(-psi^2 / (2 sigma_b^2)), with psi the angle from the
        basin centre in degrees, as sigma_b is. For B0 both parameters are zero
        and the topography is zero everywhere, so there is no ocean. Whether
        this is the topography at t = 0 or at the terminal time is
        `topography_time`.
        """
        b = self.basin
        psi = angular_distance_deg(colatitude_deg, longitude_deg,
                                   b["centre_colatitude_deg"], b["centre_longitude_deg"])
        return b["bmax_m"] - b["b0_m"] * np.exp(-psi ** 2 / (2.0 * b["sigma_b_deg"] ** 2))

    def ocean_function(self, colatitude_deg, longitude_deg) -> np.ndarray:
        """1 where the topography is below zero, 0 elsewhere (eq. 1)."""
        return (self.topography(colatitude_deg, longitude_deg) < 0.0).astype(float)

    def initial_ocean_area_fraction(self) -> float:
        """Area of the basin as a fraction of the sphere, (1 - cos psi0) / 2.

        0.0462956 for B2, the value the case D and E figures mark at t = 0.
        """
        return (1.0 - np.cos(np.radians(self.basin_radius_deg()))) / 2.0

    # ---------------------------------------------------------- earth model

    def earth_layers(self) -> dict[str, np.ndarray]:
        """The layers of the earth model as arrays, surface first.

        Returns
        -------
        dict
            ``top_radius_m``, ``bottom_radius_m``, ``density_kg_m3``,
            ``shear_modulus_pa``, ``viscosity_pa_s`` and ``rheology``. An
            elastic layer has infinite viscosity, which is what "no viscous
            flow" means for a Maxwell body. The fluid core has zero shear
            modulus and zero viscosity.
        """
        layers = self.earth_model["layers"]
        viscosity = [np.inf if layer["rheology"] == "elastic" else float(layer["viscosity_pa_s"])
                     for layer in layers]
        return {
            "top_radius_m": np.array([l["top_radius_km"] * 1e3 for l in layers]),
            "bottom_radius_m": np.array([l["bottom_radius_km"] * 1e3 for l in layers]),
            "density_kg_m3": np.array([float(l["density_kg_m3"]) for l in layers]),
            "shear_modulus_pa": np.array([float(l["shear_modulus_pa"]) for l in layers]),
            "viscosity_pa_s": np.array(viscosity),
            "rheology": np.array([l["rheology"] for l in layers]),
        }

    # --------------------------------------------------------------- outputs

    @property
    def outputs(self) -> list[str]:
        """What the paper compares for this case."""
        return list(self.definition["outputs"])

    @property
    def profiles(self) -> list[ProfileSpec]:
        """The cross-sections to output and compare, from the case file."""
        out = []
        for p in self.definition["comparison"]["profiles"]:
            longitude = re.search(r"longitude ([\d.]+)", p["profile"])
            interval = re.fullmatch(r"colatitude ([\d.]+) to ([\d.]+) degrees", p["interval"])
            if not longitude or not interval:
                raise ValueError(f"{self.letter}: cannot read profile {p}")
            out.append(ProfileSpec(
                name="basin" if "basin" in p["profile"] else "load",
                longitude_deg=float(longitude.group(1)),
                quantities=tuple(p["quantities"]),
                colatitude_interval_deg=(float(interval.group(1)), float(interval.group(2))),
                time_kyr=self.terminal_time_kyr,
                figure=int(p["figure"]),
            ))
        return out

    @property
    def time_series(self) -> list[TimeSeriesSpec]:
        """The time series to output and compare, from the case file."""
        out = []
        for s in self.definition["comparison"]["time_series"]:
            interval = re.fullmatch(r"([\d.]+) to ([\d.]+) kyr", s["interval"])
            if not interval:
                raise ValueError(f"{self.letter}: cannot read time series {s}")
            out.append(TimeSeriesSpec(
                quantities=tuple(s["quantities"]),
                time_interval_kyr=(float(interval.group(1)), float(interval.group(2))),
                figure=int(s["figure"]),
                note=s.get("note"),
            ))
        return out


# =====================================================================
# Spada et al. (2011)
# =====================================================================

SPADA_COLLECTION = "spada2011"

# A grid description of a case file or of loads/spada2011.json, for example
# `0 to 25 degrees at 0.25 degree spacing` or
# `colatitude 0 to 2.5*alpha at 0.25 degree spacing`.
_GRID = re.compile(r"(?:colatitude )?(?P<lo>[\d.]+) to (?P<hi>[\d.]+(?:\*alpha)?)"
                   r"(?: degrees)? at (?P<step>[\d.]+) degree spacing")


class SpadaCase:
    """The set-up of one Spada et al. (2011) test, from its case file and the load table.

    The ten tests share one earth model family, three analytic loads and one
    time history. This class turns the case file, ``loads/spada2011.json`` and
    the earth model file into numbers, so a solver re-types no parameter. It
    computes no response.

    Parameters
    ----------
    case
        ``t02-01`` or ``spada2011-t02-01``.

    Conventions
    -----------
    Angles are in degrees; for a load, colatitude is the angle from the load
    centre. Epochs are in kyr since the load was applied, with ``math.inf``
    for the fluid limit. Surface density and its Legendre coefficients are in
    kg/m2, and sigma(theta) = sum_n sigma_n P_n(cos theta). Densities and the
    earth radius come from the load table and never from a module constant,
    because GIAMIP uses other values.
    """

    def __init__(self, case: str):
        short = case.lower().removeprefix(f"{SPADA_COLLECTION}-")
        self.case_id = f"{SPADA_COLLECTION}-{short}"
        self.definition = cases.case(self.case_id)
        # Looked up through gia_mip.cases at construction, so a changed table
        # is followed and no value is cached in this module.
        self.load_table = cases.loads(SPADA_COLLECTION)
        self.earth_model = cases.earth_model(self.definition["earth_model"])
        constants = self.load_table["constants"]
        self.ice_density: float = float(constants["ice_density_kg_m3"])
        self.water_density: float = float(constants["water_density_kg_m3"])
        # 6371 km, table 2 of the paper. Taken from the load table and not from
        # the earth model, because VSS96 has no layer table here.
        self.earth_radius_m: float = float(constants["earth_radius_m"])
        parameters = self.definition["parameters"]
        self.parameters = parameters
        self.loads: tuple[str, ...] = tuple(parameters.get("loads", ()))
        self.outputs: list[str] = list(self.definition["outputs"])

    def __repr__(self) -> str:
        return (f"SpadaCase({self.case_id!r}: {self.definition['title']}, "
                f"earth model {self.earth_model['id']}, loads {self.loads})")

    # ------------------------------------------------------------ set-up

    @property
    def load_centre(self) -> tuple[float, float] | None:
        """Colatitude and longitude of the load centre in degrees, None for no load.

        Tests 1/2 and 2/2 put the load on the north pole; test 3/2 moves its
        centroid to colatitude 25, longitude 75.
        """
        if not self.loads:
            return None
        centroid = self.parameters.get("load_centroid")
        if centroid is not None:
            return (float(centroid["colatitude_deg"]), float(centroid["longitude_deg"]))
        if self.parameters.get("placement") != "north pole":
            raise ValueError(f"{self.case_id}: no load placement in the case file")
        return (0.0, 0.0)

    @property
    def epochs_kyr(self) -> tuple[float, ...]:
        """The epochs to output, with the string ``infinity`` read as ``math.inf``."""
        return tuple(math.inf if e == "infinity" else float(e)
                     for e in self.parameters.get("epochs_kyr", ()))

    @property
    def harmonic_degrees(self) -> tuple[int, int] | None:
        """The lowest and highest harmonic degree, or None where the case gives none."""
        if "harmonic_degrees" in self.parameters:
            lo, hi = self.parameters["harmonic_degrees"]
            return (int(lo), int(hi))
        if "harmonic_degree" in self.parameters:
            n = int(self.parameters["harmonic_degree"])
            return (n, n)
        return None

    def earth_layers(self) -> dict[str, np.ndarray]:
        """The layers of the earth model as arrays, surface first.

        Same layout as `MartinecCase.earth_layers`. VSS96, the model of test
        6/1, has no layer table in ``models/vss96.json`` yet, and asking for its
        layers raises instead of returning a wrong model.
        """
        layers = self.earth_model.get("layers")
        if not layers:
            raise ValueError(f"earth model {self.earth_model['id']} has no layer table")
        viscosity = [np.inf if layer["rheology"] == "elastic" else float(layer["viscosity_pa_s"])
                     for layer in layers]
        return {
            "top_radius_m": np.array([l["top_radius_km"] * 1e3 for l in layers]),
            "bottom_radius_m": np.array([l["bottom_radius_km"] * 1e3 for l in layers]),
            "density_kg_m3": np.array([float(l["density_kg_m3"]) for l in layers]),
            "shear_modulus_pa": np.array([float(l["shear_modulus_pa"]) for l in layers]),
            "viscosity_pa_s": np.array(viscosity),
            "rheology": np.array([l["rheology"] for l in layers]),
        }

    # ------------------------------------------------------------- loads

    def _load(self, load: str) -> dict:
        """One load of the table, refusing a name the table does not hold."""
        table = self.load_table["loads"]
        if load not in table:
            raise ValueError(f"load must be one of {tuple(table)}, not {load!r}")
        return table[load]

    def load_mass(self, load: str) -> float:
        """Total mass of a load in kg, from its closed form.

        cap: rho h 2 pi a^2 (1 - cos alpha) 2/3, the integral of the square-root
        profile; disc: rho h 2 pi a^2 (1 - cos alpha); point: the mass of the
        table. With h = 1500 m for the cap and 1000 m for the disc the two are
        equal, 3.607171340900778e18 kg, which TABOO prints in every header.
        """
        spec = self._load(load)
        if load == "point":
            return float(spec["mass_kg"])
        cos_alpha = np.cos(np.radians(float(spec["angular_radius_deg"])))
        mass = (self.ice_density * float(spec["thickness_m"]) * 2.0 * np.pi
                * self.earth_radius_m ** 2 * (1.0 - cos_alpha))
        return float(mass * 2.0 / 3.0) if load == "cap" else float(mass)

    def surface_density(self, load: str, colatitude_deg) -> np.ndarray:
        """Surface mass density in kg/m2 at an angle from the load centre.

        cap: rho h sqrt((cos theta - cos alpha) / (1 - cos alpha)) inside alpha;
        disc: rho h inside alpha; both zero beyond. The point load is a delta
        function with no value at a point, so it raises ValueError.
        """
        spec = self._load(load)
        if load == "point":
            raise ValueError("the point load is a delta function and has no surface density")
        theta = np.radians(np.asarray(colatitude_deg, dtype=float))
        cos_alpha = np.cos(np.radians(float(spec["angular_radius_deg"])))
        rho_h = self.ice_density * float(spec["thickness_m"])
        if load == "disc":
            # The edge itself, theta = alpha, belongs to the disc.
            return np.where(np.cos(theta) >= cos_alpha, rho_h, 0.0)
        ratio = (np.cos(theta) - cos_alpha) / (1.0 - cos_alpha)
        return rho_h * np.sqrt(np.clip(ratio, 0.0, None))

    def legendre_coefficients(self, load: str, degrees) -> np.ndarray:
        """sigma_n in kg/m2 for the given degrees, from the closed forms of table 4.

        The coefficients satisfy sigma(theta) = sum_n sigma_n P_n(cos theta), so
        sigma_n = (2n + 1)/2 integral of sigma P_n d(cos theta), and 4 pi a^2 sigma_0
        is the load mass.

        cap: sigma_n = -rho h / (4 (1 - cos alpha)) [ (T_{n+1} - T_{n+2}) / (n + 3/2)
             - (T_{n-1} - T_n) / (n - 1/2) ], with T_k = cos(k alpha) as written
             in loads/spada2011.json. It also holds at n = 0, where it gives
             rho h (1 - cos alpha) / 3.
        disc: sigma_n = rho h / 2 [ P_{n-1}(cos alpha) - P_{n+1}(cos alpha) ], and
             rho h (1 - cos alpha) / 2 at n = 0.
        point: sigma_n = m (2n + 1) / (4 pi a^2).

        The cap expression subtracts nearly equal cosines at high degree; at
        n = 128 it agrees with a quadrature of the profile to 1e-10 relative.
        """
        spec = self._load(load)
        n = np.asarray(degrees, dtype=int)
        if np.any(n < 0):
            raise ValueError("degrees must be non-negative")
        if load == "point":
            return float(spec["mass_kg"]) * (2 * n + 1) / (4.0 * np.pi * self.earth_radius_m ** 2)
        alpha = np.radians(float(spec["angular_radius_deg"]))
        rho_h = self.ice_density * float(spec["thickness_m"])
        c = np.cos(alpha)
        if load == "cap":
            def t(k):
                return np.cos(k * alpha)
            return (-rho_h / (4.0 * (1.0 - c))
                    * ((t(n + 1) - t(n + 2)) / (n + 1.5) - (t(n - 1) - t(n)) / (n - 0.5)))
        # The disc needs P_{n-1} and P_{n+1} at cos alpha, from the three-term
        # recurrence up to the highest degree asked for.
        n_max = int(n.max()) + 1 if n.size else 1
        p = np.zeros(n_max + 1)
        p[0] = 1.0
        if n_max >= 1:
            p[1] = c
        for k in range(1, n_max):
            p[k + 1] = ((2 * k + 1) * c * p[k] - k * p[k - 1]) / (k + 1)
        out = np.where(n == 0, rho_h * (1.0 - c) / 2.0,
                       rho_h / 2.0 * (p[np.maximum(n - 1, 0)] - p[n + 1]))
        return out

    # ------------------------------------------------------------- grids

    def _grid(self, name: str) -> np.ndarray:
        """A colatitude grid from the case file, or else from the load table.

        Test 1/2 writes its grids into its case file; test 2/2 says only "as
        test 1/2", so the load table's `output_grids` is the fallback, where the
        near field ends at 2.5 alpha.
        """
        text = self.parameters.get("grids", {}).get(name) or self.load_table["output_grids"][name]
        match = _GRID.fullmatch(text)
        if not match:
            raise ValueError(f"{self.case_id}: cannot read the {name} grid {text!r}")
        lo, step = float(match.group("lo")), float(match.group("step"))
        hi_text = match.group("hi")
        if hi_text.endswith("*alpha"):
            radii = {float(self._load(l)["angular_radius_deg"]) for l in self.loads
                     if "angular_radius_deg" in self._load(l)}
            if len(radii) != 1:
                raise ValueError(f"{self.case_id}: the loads do not share one radius")
            hi = float(hi_text.removesuffix("*alpha")) * radii.pop()
        else:
            hi = float(hi_text)
        # A count from the rounded ratio, so that 0.25 degree steps land exactly
        # on 25.0 and are not lost to floating point in arange.
        count = int(round((hi - lo) / step)) + 1
        return lo + step * np.arange(count)

    def near_field_colatitudes_deg(self) -> np.ndarray:
        """0 to 2.5 alpha = 25 degrees at 0.25 degree, 101 points."""
        return self._grid("near_field")

    def global_colatitudes_deg(self) -> np.ndarray:
        """0 to 180 degrees at 1 degree, 181 points."""
        return self._grid("global")
