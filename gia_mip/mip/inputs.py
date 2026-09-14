"""Read the GIAMIP inputs a solver needs: ice histories and bed topography.

The organisers deliver the inputs as NetCDF files on the prescribed
Gauss-Legendre grid (``gia_mip.grids``). Three things make a plain read give a
wrong answer with no error, and the readers here handle all three:

* every ice file declares ``_FillValue = 0.0``, so a default read masks every
  ice-free cell (correction ice-fillvalue-zero). Masking is always off here;
* the latitudes run south to north, while radopt and many solvers want the
  north first. Every field accessor takes ``north_first``;
* the files change under the same name (docs/data/corrections.qmd). Each
  reader computes the SHA-256 of the file it opens and looks up the
  corrections for that version only.

The experiment, the ice history and the topography of each case come from
``cases/giamip/expNN.json``, so `GiamipExperiment` is the entry point:

    >>> from gia_mip.mip.inputs import GiamipExperiment
    >>> exp = GiamipExperiment("Exp01")
    >>> exp.ice_history.years[:2]            # doctest: +SKIP
    array([-122000., -121500.])
    >>> h = exp.ice_history.thickness(0, north_first=True)   # (257, 513), metres
"""

from __future__ import annotations

import pathlib
import re

import numpy as np

from gia_mip import cases, corrections, data, grids

# The time unit of the forcing files: days on a 365-day calendar (correction
# ice-year-longname), so a year is exactly 365 days.
DAYS_PER_YEAR = 365.0


def original_root() -> pathlib.Path:
    """The ``original/`` directory: ``$GIA_MIP_DATA_ROOT/original`` or the checkout's."""
    import os
    root = os.environ.get("GIA_MIP_DATA_ROOT")
    return (pathlib.Path(root) if root else data.ROOT / "data") / "original"


class _GiamipFile:
    """A NetCDF input on the prescribed grid, opened with masking off.

    Parameters
    ----------
    relative_path
        Path under ``data/original/``, as the case file names it.
    root
        The ``original/`` directory. Default: `original_root`.
    """

    def __init__(self, relative_path: str, root: pathlib.Path | None = None):
        import netCDF4
        self.relative_path = relative_path
        self.path = (root or original_root()) / relative_path
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} is missing. GIAMIP inputs are not yet cleared for "
                f"redistribution; place them by hand as docs/data/fetching.qmd describes.")
        # The version of the file decides which corrections apply to it.
        self.sha256 = data.sha256(self.path)
        known = data.file_versions().get(relative_path)
        self.is_current_version = bool(known) and known["sha256"] == self.sha256
        self.corrections = corrections.applying_to(relative_path, self.sha256)
        self._ds = netCDF4.Dataset(self.path)
        self._ds.set_auto_mask(False)
        self.lat = np.asarray(self._ds["lat"][:], dtype=float)
        self.lon = np.asarray(self._ds["lon"][:], dtype=float)
        grids.check(self.lat, self.lon, name=relative_path)

    @property
    def correction_ids(self) -> list[str]:
        """The ids of the corrections that apply to this version of the file."""
        return [c["id"] for c in self.corrections]

    def latitudes(self, north_first: bool = False) -> np.ndarray:
        """The latitude axis in degrees, south first unless ``north_first``."""
        return self.lat[::-1].copy() if north_first else self.lat.copy()

    def close(self) -> None:
        self._ds.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class IceHistory(_GiamipFile):
    """An ice thickness history, (time, lat, lon) in metres.

    Attributes
    ----------
    years
        Time of each step in years from 2000 CE, model year 0; negative is the
        past. Computed from ``time`` in days on the 365-day calendar and
        checked against the file's ``year`` variable, whose long_name is wrong
        but whose values agree (correction ice-year-longname).
    """

    def __init__(self, relative_path: str, root: pathlib.Path | None = None):
        super().__init__(relative_path, root)
        days = np.asarray(self._ds["time"][:], dtype=float)
        calendar = getattr(self._ds["time"], "calendar", None)
        if calendar != "365_day":
            raise ValueError(f"{relative_path}: time calendar is {calendar!r}, expected '365_day'")
        self.years = days / DAYS_PER_YEAR
        stated = np.asarray(self._ds["year"][:], dtype=float)
        if np.max(np.abs(stated - self.years)) > 1e-6:
            raise ValueError(f"{relative_path}: time / 365 and year disagree")
        shape = self._ds["ice_thickness"].shape
        if shape != (self.years.size, grids.NLAT, grids.NLON):
            raise ValueError(f"{relative_path}: ice_thickness has shape {shape}")

    def __len__(self) -> int:
        return int(self.years.size)

    def thickness(self, step: int, north_first: bool = False) -> np.ndarray:
        """Ice thickness in metres at one step, (lat, lon), float64.

        Zero is a real value meaning no ice, never a missing value.
        """
        field = np.asarray(self._ds["ice_thickness"][step], dtype=float)
        return field[::-1].copy() if north_first else field

    def step_of_year(self, year: float) -> int:
        """The index of the step at a model year. No interpolation: an absent year raises."""
        hits = np.flatnonzero(np.isclose(self.years, year, atol=1e-6))
        if hits.size != 1:
            raise KeyError(f"{self.relative_path} has no step at year {year}")
        return int(hits[0])


class Topography(_GiamipFile):
    """A present-day bed topography, (lat, lon) in metres above mean sea level."""

    def elevation(self, north_first: bool = False) -> np.ndarray:
        """Bed elevation in metres, positive up, (lat, lon), float64."""
        field = np.asarray(self._ds["topg"][:], dtype=float)
        return field[::-1].copy() if north_first else field

    @property
    def antarctic_dataset(self) -> str:
        """The Antarctic bed dataset, read from the filename.

        The ``source`` attribute of one file names the wrong dataset
        (correction topography-source-attribute), and the filename is right
        in all three, so the filename is the authority.
        """
        m = re.search(r"bedtopo_(\w+?)ANT_", pathlib.Path(self.relative_path).name)
        names = {"Bedmap2": "Bedmap2", "Bedmap3": "Bedmap3", "BedMach3": "BedMachine Antarctica v3"}
        if not m or m.group(1) not in names:
            raise ValueError(f"{self.relative_path}: cannot read the Antarctic dataset from the name")
        return names[m.group(1)]


def grounded_thickness(thickness: np.ndarray, bed_elevation: np.ndarray,
                       ice_density: float, ocean_density: float) -> np.ndarray:
    """Ice thickness with floating ice removed.

    Ice is afloat where the bed is below sea level and the ice column weighs
    less than the water it would displace: thickness * rho_ice <
    -bed * rho_ocean. Only grounded ice loads the solid Earth (protocol
    section 2.1). The sea level is taken at zero on the topography's datum, so
    this is the flotation test at present-day sea level; a solver that
    evolves sea level applies the same test with its own sea level.

    Parameters
    ----------
    thickness, bed_elevation
        Fields on the same grid, in metres.
    ice_density, ocean_density
        kg/m3, from the case file (917 and 1025 for GIAMIP).
    """
    thickness = np.asarray(thickness, dtype=float)
    bed = np.asarray(bed_elevation, dtype=float)
    afloat = (bed < 0.0) & (thickness * ice_density < -bed * ocean_density)
    return np.where(afloat, 0.0, thickness)


class GiamipExperiment:
    """One GIAMIP experiment: its definition, forcing files and prescribed times.

    Parameters
    ----------
    experiment
        ``"Exp01"``, ``"giamip-exp01"`` or ``1``.
    root
        The ``original/`` directory. Default: `original_root`.
    """

    def __init__(self, experiment, root: pathlib.Path | None = None):
        number = int(re.search(r"(\d+)$", str(experiment)).group(1))
        self.id = f"giamip-exp{number:02d}"
        self.definition = cases.case(self.id)
        self.constants = self.definition["constants"]
        self._root = root
        self._ice = None
        self._topo = None

    def __repr__(self) -> str:
        return f"GiamipExperiment({self.definition['experiment_id']!r})"

    @property
    def ice_history(self) -> IceHistory | None:
        """The prescribed ice history, or None where the case leaves it open."""
        rel = self.definition["forcing"].get("ice_history_file")
        if rel and self._ice is None:
            self._ice = IceHistory(rel, self._root)
        return self._ice

    @property
    def topography(self) -> Topography | None:
        """The prescribed topography, or None where the case leaves it open."""
        rel = self.definition["topography"].get("file")
        if rel and self._topo is None:
            self._topo = Topography(rel, self._root)
        return self._topo

    def model_years(self) -> np.ndarray:
        """The prescribed output times in years from 2000 CE.

        500-year steps from the start to 26 ka, then 100-year steps to the end
        (protocol section 2.2.2).
        """
        period = self.definition["period"]
        start = -1000 * int(period["start_ka_before_2000ce"])
        end = int(period["end_years_after_2000ce"])
        coarse = np.arange(start, -26000, 500) if start < -26000 else np.array([], dtype=int)
        fine = np.arange(max(start, -26000), end + 1, 100)
        return np.concatenate([coarse, fine]).astype(float)

    def forcing_steps(self) -> np.ndarray:
        """Indices of the ice-history steps inside the experiment's period.

        For Exp06 and Exp07 the delivered file starts at 122 ka and the
        experiment at 80 ka, so this is the truncation the case file asks for.
        Raises if a prescribed year is not a step of the file.
        """
        ice = self.ice_history
        if ice is None:
            raise ValueError(f"{self.id} has no prescribed ice history")
        return np.array([ice.step_of_year(y) for y in self.model_years()])
