"""Everything a solver needs to set up one Martinec et al. (2018) benchmark case.

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
