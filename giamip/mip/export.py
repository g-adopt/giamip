"""Read the intermediate file between the G-ADOPT export and the GIAMIP writer.

The G-ADOPT export script (the g-adopt branch ``sghelichkhani/giamip-exp01``)
evaluates the finite-element surface fields of a forward run at the nodes of
the GIAMIP Gauss-Legendre grid and writes one plain HDF5 file per
experiment, ``export_expXX.h5``. `giamip.mip.outputs` turns that file into
the GIAMIP NetCDF files. The split keeps Firedrake out of this package.

The layout of the file (part H of ``NOTES/GIAMIP-SUBMISSION-PLAN.md``, a
private working note of the G-ADOPT team) is
defined in code by `EXPORT_FIELDS`, `EXPORT_SCALARS` and `EXPORT_ATTRIBUTES`
below. The export script is expected to open its finished file with
`ExportFile` as its last check, so both sides use one definition:

    lat (257,) f8            giamip.grids.latitudes(), south to north, degrees
    lon (513,) f8            giamip.grids.longitudes(), [0, 360), degrees
    year (n,) f8             model output years, strictly increasing
    step (n,) i8             solver step of each output
    index (n,) i8            output index in the surface file
    fields/<name> (n, 257, 513) f8
    scalars/<name> (n,) f8
    check/ice_thickness_year0 (257, 513) f8

The file is plain HDF5 written with h5py, not NetCDF, so it is read with
h5py here. The rule that every NetCDF file is opened through h5netcdf
(AGENTS.md) is about NetCDF masking and fill values, which plain HDF5 does not
have.
"""

from __future__ import annotations

import pathlib

import numpy as np

from giamip import grids

# The six fields on the grid, by their GIAMIP names. Units: metres for the
# four displacement and geoid fields, 1 for the two fractions.
EXPORT_FIELDS = ("delta_bed", "delta_g", "delta_bed_east", "delta_bed_north",
                 "ocean_area_fraction", "land_ice_area_fraction")

# The six scalars, computed on the native mesh (protocol section 6.1) and
# matched to the output rows by the solver step on the export side.
# Units: m for mean_delta_g, kg for the masses, m2 for the areas.
EXPORT_SCALARS = ("mean_delta_g", "grd_ice_mass", "total_ice_mass", "maf",
                  "ocean_area", "ocean_area_grdice")

# The root attributes the writer needs. The export may add more; the writer
# copies every root attribute into the output files (`giamip.mip.outputs`),
# so an added attribute needs no change here.
EXPORT_ATTRIBUTES = ("experiment", "schedule", "git_head", "git_branch", "firedrake",
                     "petsc", "mesh", "bed_iteration", "bed_misfit_year0_rms_m",
                     "bed_misfit_year0_max_abs_m", "k_s", "C_minus_A_nondim",
                     "surface_file", "surface_file_size", "surface_file_mtime",
                     "ranks", "fraction_method", "fraction_samples", "created")

# The one check field: ice thickness at year 0 at the grid nodes, in metres.
# It is compared with the forcing slice of year 0, which tests the point
# order, the latitude direction and the longitude origin of the export.
CHECK_ICE = "check/ice_thickness_year0"


class ExportError(ValueError):
    """The intermediate file does not have the layout this module defines."""


def _decode(value):
    """An HDF5 attribute as a plain Python value.

    h5py returns text as ``str`` or ``bytes`` depending on how it was written,
    and numbers as NumPy scalars. One-element arrays become their value.
    Longer arrays stay arrays.
    """
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return _decode(value.reshape(-1)[0])
        return value
    if isinstance(value, np.generic):
        return value.item()
    return value


class ExportFile:
    """One intermediate file, opened read-only and checked against the layout above.

    The constructor checks the layout and raises `ExportError` naming the file
    on the first problem it finds. Field values are read row by row
    (`field`), so a 463-step field of 0.5 GB never sits in memory as a whole.
    NaN and inf are not checked here, because that would read every field
    once more; the writer checks each slice as it writes it.

    Parameters
    ----------
    path
        The file.
    experiment
        The experiment this file must be for, for example ``"Exp01"``. None
        accepts the experiment the file names.

    Attributes
    ----------
    experiment : str
        ``Exp01`` ... ``Exp12``, from the root attribute.
    attrs : dict
        Every root attribute, decoded.
    years : numpy.ndarray
        Model output years, float64, strictly increasing.
    steps, indices : numpy.ndarray
        Solver step and surface-file index of each row, int64.
    """

    def __init__(self, path, experiment: str | None = None):
        import h5py
        self.path = pathlib.Path(path)
        self._h5 = h5py.File(self.path, "r")
        try:
            self._validate(experiment)
        except Exception:
            self._h5.close()
            raise

    # ----------------------------------------------------------- checks

    def _fail(self, message: str):
        raise ExportError(f"{self.path}: {message}")

    def _validate(self, experiment: str | None) -> None:
        h5 = self._h5
        self.attrs = {k: _decode(v) for k, v in h5.attrs.items()}
        missing = [a for a in EXPORT_ATTRIBUTES if a not in self.attrs]
        if missing:
            self._fail(f"root attributes missing: {missing}")
        self.experiment = str(self.attrs["experiment"])
        if experiment is not None and self.experiment != experiment:
            self._fail(f"the file is for {self.experiment}, not {experiment}")

        for name in ("lat", "lon", "year", "step", "index"):
            if name not in h5:
                self._fail(f"dataset {name!r} missing")
        # grids.check compares the values with the generated grid, so a
        # north-first latitude axis or a shifted longitude axis is refused.
        try:
            grids.check(np.asarray(h5["lat"][...]), np.asarray(h5["lon"][...]),
                        name=str(self.path))
        except grids.GridMismatch as err:
            raise ExportError(str(err)) from None

        self.years = np.asarray(h5["year"][...], dtype=float)
        self.steps = np.asarray(h5["step"][...], dtype=np.int64)
        self.indices = np.asarray(h5["index"][...], dtype=np.int64)
        self.n = int(self.years.size)
        if self.years.ndim != 1 or self.n < 2:
            self._fail(f"year has shape {self.years.shape}; at least two outputs are needed")
        if not np.all(np.diff(self.years) > 0):
            self._fail("year is not strictly increasing")
        for name, values in (("step", self.steps), ("index", self.indices)):
            if values.shape != (self.n,):
                self._fail(f"{name} has shape {values.shape}, year has ({self.n},)")
        if not np.all(np.diff(self.steps) > 0):
            self._fail("step is not strictly increasing")

        for name in EXPORT_FIELDS:
            key = f"fields/{name}"
            if key not in h5:
                self._fail(f"dataset {key!r} missing")
            if h5[key].shape != (self.n, grids.NLAT, grids.NLON):
                self._fail(f"{key} has shape {h5[key].shape}, expected "
                           f"({self.n}, {grids.NLAT}, {grids.NLON})")
        for name in EXPORT_SCALARS:
            key = f"scalars/{name}"
            if key not in h5:
                self._fail(f"dataset {key!r} missing")
            if h5[key].shape != (self.n,):
                self._fail(f"{key} has shape {h5[key].shape}, expected ({self.n},)")
        if CHECK_ICE not in h5:
            self._fail(f"dataset {CHECK_ICE!r} missing")
        if h5[CHECK_ICE].shape != (grids.NLAT, grids.NLON):
            self._fail(f"{CHECK_ICE} has shape {h5[CHECK_ICE].shape}")

    # ----------------------------------------------------------- access

    def field(self, name: str, rows) -> np.ndarray:
        """Rows of one field, float64, ``(len(rows), 257, 513)``, or one row for an int.

        Parameters
        ----------
        name
            A member of `EXPORT_FIELDS`.
        rows
            An int, or a sorted sequence of row indices (h5py needs
            increasing indices for a fancy read).
        """
        if name not in EXPORT_FIELDS:
            raise KeyError(name)
        return np.asarray(self._h5[f"fields/{name}"][rows], dtype=float)

    def scalar(self, name: str) -> np.ndarray:
        """One scalar at every output row, float64 ``(n,)``."""
        if name not in EXPORT_SCALARS:
            raise KeyError(name)
        return np.asarray(self._h5[f"scalars/{name}"][...], dtype=float)

    def check_ice(self) -> np.ndarray:
        """The ice thickness at year 0 at the grid nodes, metres, ``(257, 513)``."""
        return np.asarray(self._h5[CHECK_ICE][...], dtype=float)

    def close(self) -> None:
        self._h5.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
