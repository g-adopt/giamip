"""Write the GIAMIP output files of one experiment from the G-ADOPT intermediate file.

Input: one intermediate file ``export_expXX.h5`` (`giamip.mip.export`), with
the surface fields of a forward run at the Gauss-Legendre nodes and the
scalars of the native mesh, at the model output years.

Output: one NetCDF-4 file per variable, named
``{variable}_{ExpXX}_G-ADOPT_g-adopt.nc`` (no ensemble member tag), that
passes every hard check of the compliance checker pinned in
``giamip/resources/checker.json``:

* the ten mandatory variables and the four optional ones (``delta_bed_east``,
  ``delta_bed_north``, ``Clm``, ``Slm``);
* fields float32 ``(time, lat, lon)``, zlib level 4, one chunk per time slice
  ``(1, 257, 513)``; scalars float64 ``(time,)``; Stokes coefficients float64
  ``(degree, order)``;
* no ``_FillValue`` and no ``missing_value`` anywhere: h5netcdf writes none
  unless asked, and nothing here asks;
* ``time`` in days since 2000-01-01 on the 365-day calendar, float64, and
  ``year`` int32, equal step by step to the ``year`` of the forcing file;
* the variable attributes of ``variables.json`` (generated from the checker,
  never retyped), the coordinate attributes of the checker source, and the
  global attributes that define the run.

Time axis. GIAMIP wants every variable at every forcing year. The protocol
runs output at every forcing year, so the rows are copied. The schedule-D
runs (Exp09, Exp10, Exp12) output at a subset of the forcing years; the
writer interpolates linearly in time between the two neighbouring outputs
(the decision of the G-ADOPT team for these three experiments), copies the
output years bit for bit, and refuses to
extrapolate. The interpolation runs in float64 and the result is cast to
float32 afterwards, so the cast does not add to the interpolation error.
The interpolated years are named in the ``comment`` of every time variable.

Constants. The densities and the radius come from the case file (AGENTS.md:
never from a module). The run values (k_s, C - A, rotation rate, provenance)
come from the root attributes of the intermediate file, which are all copied
into the global attributes with the prefix ``run_``.

Run it as ``python -m giamip.mip.outputs write --export E --out-dir D``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import re
import subprocess
import sys
from dataclasses import dataclass

import numpy as np

from giamip import __version__, _netcdf, cases, data, grids, harmonics
from giamip.mip.export import EXPORT_FIELDS, EXPORT_SCALARS, ExportFile

# ------------------------------------------------------------------ identity

# Group and model as registered with the organisers. The files carry no
# ensemble member tag, because G-ADOPT submits one member per experiment.
# Neither may contain an underscore: the checker splits the filename on
# underscores.
GROUP = "G-ADOPT"
MODEL = "g-adopt"
INSTITUTION = "Australian National University"

# The contributors, in the order the project lead gave them. One list, so
# the names and the e-mail addresses stay in the same order.
CONTRIBUTORS = (
    ("Sia Ghelichkhan", "siavash.ghelichkhan@anu.edu.au"),
    ("William Scott", "William.Scott1@anu.edu.au"),
    ("Mark Hoggard", "mark.hoggard@anu.edu.au"),
    ("Angus Gibson", "angus.gibson@anu.edu.au"),
    ("Dale S. Roberts", "dale.roberts1@anu.edu.au"),
    ("Stephan Kramer", "s.kramer@imperial.ac.uk"),
    ("Thomas Duvernay", "thomas.duvernay@anu.edu.au"),
    ("Rhodri Davies", "rhodri.davies@anu.edu.au"),
    ("G-ADOPT team", "help@gadopt.org"),
)
CONTRIBUTORS_NAME = ", ".join(name for name, _ in CONTRIBUTORS)
CONTRIBUTORS_EMAIL = ", ".join(email for _, email in CONTRIBUTORS)

REPOSITORY_URL = "https://github.com/g-adopt/giamip"
# The section of the conventions page that defines every output quantity.
CONVENTIONS_URL = "https://gadopt.org/giamip/docs/reference/conventions.html#giamip-output"

# Every root attribute of the intermediate file is copied into the global
# attributes under this prefix, so a new attribute of the export needs no
# change here, and a run value can never be mistaken for a case constant.
RUN_ATTRIBUTE_PREFIX = "run_"

# ------------------------------------------------------------------ the data request

# The coordinate variables, from the checker source (GIAMIP_COORD_LONG_NAMES,
# GIAMIP_COORD_STANDARD_NAMES, GIAMIP_COORD_DTYPES, the time units and
# calendar) and variables.md. `tests/test_giamip_outputs.py` compares this
# table with the pinned checker. Two entries are easy to get wrong: the
# `year` long_name is NOT the one of the forcing files, and `degree` and
# `order` are float64, not integers.
COORDINATE_ATTRIBUTES = {
    "time": {"dtype": "f8", "units": "days since 2000-01-01 00:00:00", "calendar": "365_day",
             "long_name": "Time", "standard_name": "time"},
    "year": {"dtype": "i4", "units": "year",
             "long_name": "Calendar year corresponding to each output time "
                          "(positive = CE, negative = BCE)"},
    "lat": {"dtype": "f8", "units": "degrees_north", "long_name": "Latitude",
            "standard_name": "latitude"},
    "lon": {"dtype": "f8", "units": "degrees_east", "long_name": "Longitude",
            "standard_name": "longitude"},
    "degree": {"dtype": "f8", "units": "1", "long_name": "Spherical harmonic degree"},
    "order": {"dtype": "f8", "units": "1", "long_name": "Spherical harmonic order"},
}

# The "Reference" column of the checker's variables.md: the datum of each
# variable. The QC document names "reference surface and time information"
# as a hard check that the checker code does not test, so it is written as a
# `reference` attribute. The variables whose column is a dash have none.
VARIABLE_REFERENCE_DATA_REQUEST = {
    "delta_bed": "Bedrock elevation at initial simulation timestep",
    "delta_g": "Geoid at initial simulation timestep",
    "mean_delta_g": "Geoid at initial simulation timestep",
    "delta_bed_east": "Geoid at initial simulation timestep",
    "delta_bed_north": "Geoid at initial simulation timestep",
    "Clm": "Initial simulation timestep",
    "Slm": "Initial simulation timestep",
}

# The data request gives "Geoid at initial simulation timestep" as the datum
# of the two horizontal displacement components. A horizontal displacement is
# measured from the position of the surface point at the first step; the
# geoid is not its datum. The organisers' text is a copy of the delta_g row,
# so the two components get the datum that matches their long name ("...
# relative to the initial simulation timestep"). The checker does not test
# this attribute, and the conventions page records the difference.
VARIABLE_REFERENCE = dict(
    VARIABLE_REFERENCE_DATA_REQUEST,
    delta_bed_east="Bedrock position at initial simulation timestep",
    delta_bed_north="Bedrock position at initial simulation timestep",
)

# The optional fields that are left out, never clipped, when a value is
# outside the checker bounds. Any other variable outside its bounds stops
# the writer.
OPTIONAL_SKIP_ON_BOUNDS = ("delta_bed_east", "delta_bed_north")

# Days per year of the 365-day calendar: time = year * 365 exactly.
DAYS_PER_YEAR = 365.0

# One time slice per chunk, with zlib level 4: a reader that wants one
# epoch decompresses one chunk.
FIELD_CHUNKS = (1, grids.NLAT, grids.NLON)
COMPRESSION_LEVEL = 4


class OutputError(ValueError):
    """A value cannot go into a GIAMIP file as it is (non-finite or out of bounds)."""


def variable_table() -> dict[str, dict]:
    """The GIAMIP variables of ``variables.json``, by name.

    ``variables.json`` is generated from ``GIAMIP_VARIABLES`` of the pinned
    checker, so the units, long names, standard names, precisions and bounds
    are the checker's own.
    """
    path = cases.RESOURCES / "variables.json"
    with open(path) as f:
        table = json.load(f)
    return {v["variable"]: v for v in table["variables"]}


def file_name(variable: str, experiment: str) -> str:
    """``{variable}_{experiment}_G-ADOPT_g-adopt.nc``."""
    return f"{variable}_{experiment}_{GROUP}_{MODEL}.nc"


# ------------------------------------------------------------------ time

@dataclass
class TimeMap:
    """How each forcing year is made from the model output rows.

    For forcing step t the value is ``(1 - a[t]) v[k0[t]] + a[t] v[k1[t]]``.
    At a model output year ``a = 0`` and ``k0 = k1``, and the row is copied
    bit for bit.

    Attributes
    ----------
    model_years : numpy.ndarray
        int64, the years of the output rows.
    forcing_years : numpy.ndarray
        int32, the years the files must carry.
    k0, k1 : numpy.ndarray
        int64, the two rows of each forcing step.
    a : numpy.ndarray
        float64, the weight of row k1, in [0, 1).
    """

    model_years: np.ndarray
    forcing_years: np.ndarray
    k0: np.ndarray
    k1: np.ndarray
    a: np.ndarray

    @classmethod
    def from_years(cls, model_years, forcing_years) -> "TimeMap":
        """Build the map, or raise `ValueError` if the model years cannot give the forcing years.

        Three conditions, each of which means the intermediate file is not
        the one the writer expects:

        * the model years are whole years and strictly increasing;
        * the first and the last forcing year are model years (no
          extrapolation);
        * every model year is a forcing year (an output at a year GIAMIP does
          not ask for points to a wrong schedule).
        """
        model = np.asarray(model_years, dtype=float)
        if np.any(np.abs(model - np.round(model)) > 1e-6):
            raise ValueError("model years are not whole years")
        model = np.round(model).astype(np.int64)
        forcing = np.asarray(forcing_years).astype(np.int64)
        if not np.all(np.diff(model) > 0):
            raise ValueError("model years are not strictly increasing")
        if not np.all(np.diff(forcing) > 0):
            raise ValueError("forcing years are not strictly increasing")
        if model[0] != forcing[0] or model[-1] != forcing[-1]:
            raise ValueError(
                f"model years run {model[0]} to {model[-1]}, forcing years {forcing[0]} to "
                f"{forcing[-1]}; the writer does not extrapolate")
        extra = np.setdiff1d(model, forcing)
        if extra.size:
            raise ValueError(f"model years that are not forcing years: {extra[:10].tolist()}")
        # k1: the first model row at or after the forcing year.
        k1 = np.searchsorted(model, forcing, side="left")
        exact = model[k1] == forcing
        k0 = np.where(exact, k1, k1 - 1)
        span = (model[k1] - model[k0]).astype(float)
        a = np.where(exact, 0.0, (forcing - model[k0]) / np.where(exact, 1.0, span))
        return cls(model, forcing.astype(np.int32), k0.astype(np.int64),
                   k1.astype(np.int64), a.astype(float))

    @property
    def is_identity(self) -> bool:
        """True when every forcing year is a model year (the protocol schedules)."""
        return bool(np.all(self.a == 0.0)) and self.model_years.size == self.forcing_years.size

    @property
    def interpolated(self) -> np.ndarray:
        """True at each forcing step that is interpolated, not copied."""
        return self.a != 0.0

    def interpolated_ranges(self) -> list[tuple[int, int, int]]:
        """Runs of consecutive interpolated forcing steps as ``(first year, last year, count)``."""
        runs = []
        flags = self.interpolated
        t = 0
        while t < flags.size:
            if not flags[t]:
                t += 1
                continue
            start = t
            while t < flags.size and flags[t]:
                t += 1
            runs.append((int(self.forcing_years[start]), int(self.forcing_years[t - 1]),
                         t - start))
        return runs

    def apply_scalar(self, values: np.ndarray) -> np.ndarray:
        """A scalar series at the forcing years, float64; copies at the model years."""
        v = np.asarray(values, dtype=float)
        mixed = (1.0 - self.a) * v[self.k0] + self.a * v[self.k1]
        return np.where(self.a == 0.0, v[self.k0], mixed)

    def apply_slice(self, read_row, t: int) -> np.ndarray:
        """The float64 field of forcing step t; `read_row(k)` returns model row k.

        The interpolation is ``(1 - a) * v0 + a * v1`` in float64; the caller
        casts to float32 afterwards, so the cast adds no error to the
        interpolation and a model year is stored exactly as its row cast once.
        """
        k0, k1, a = int(self.k0[t]), int(self.k1[t]), float(self.a[t])
        v0 = read_row(k0)
        if a == 0.0:
            return v0
        return (1.0 - a) * v0 + a * read_row(k1)

    def comment(self) -> str | None:
        """The sentence that names the interpolated years, or None for a copy."""
        if not np.any(self.interpolated):
            return None
        n = int(np.count_nonzero(self.interpolated))
        parts = []
        for first, last, count in self.interpolated_ranges():
            parts.append(str(first) if count == 1 else f"{first} to {last} ({count} years)")
        return (f"The model writes output at {self.model_years.size} of the "
                f"{self.forcing_years.size} forcing years. The values at the other {n} years "
                "are linear interpolations in time between the two neighbouring model "
                "outputs: " + "; ".join(parts) + ".")


class _RowCache:
    """Reads model rows of one field, keeping the last two, since consecutive forcing steps share rows."""

    def __init__(self, export: ExportFile, name: str):
        self._export = export
        self._name = name
        self._rows: dict[int, np.ndarray] = {}

    def __call__(self, k: int) -> np.ndarray:
        if k not in self._rows:
            if len(self._rows) >= 2:
                self._rows.pop(min(self._rows))
            self._rows[k] = self._export.field(self._name, k)
        return self._rows[k]


def forcing_years(experiment: str) -> np.ndarray:
    """The ``year`` of the forcing file of an experiment, int32.

    Read through `giamip.mip.inputs.IceHistory`, which takes the years from
    ``time / 365`` and checks them against the file's ``year`` variable. The
    checker compares the output ``year`` with the forcing ``year`` cast to
    int, so the years must be whole.
    """
    from giamip.mip.inputs import GiamipExperiment
    years = GiamipExperiment(experiment).ice_history.years
    whole = np.round(years)
    if np.any(np.abs(years - whole) > 1e-6):
        raise ValueError(f"{experiment}: forcing years are not whole years")
    return whole.astype(np.int32)


# ------------------------------------------------------------------ provenance

def writer_provenance(package_parent=None) -> dict:
    """The commit of the writer code, for the ``references`` attribute.

    Two sources, in this order:

    1. the git work tree that holds the package (a checkout or a worktree on
       phoenix): ``git rev-parse HEAD``, and ``-dirty`` appended when
       ``git status --porcelain -- giamip`` lists any change, untracked files
       included;
    2. a file ``WRITER_COMMIT`` in the directory above the package, written
       by the step that copies the package to Gadi (the copy has no ``.git``).

    Raises `RuntimeError` when neither gives a commit, so no file is written
    that names no commit.

    Parameters
    ----------
    package_parent
        The directory that holds the ``giamip`` package. None takes the one
        this module was imported from; tests pass another.
    """
    if package_parent is None:
        package_parent = pathlib.Path(__file__).resolve().parent.parent.parent
    package_parent = pathlib.Path(package_parent)
    if (package_parent / ".git").exists():
        try:
            head = subprocess.run(["git", "-C", str(package_parent), "rev-parse", "HEAD"],
                                  check=True, capture_output=True, text=True).stdout.strip()
            status = subprocess.run(["git", "-C", str(package_parent), "status", "--porcelain",
                                     "--", "giamip"],
                                    check=True, capture_output=True, text=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as err:
            raise RuntimeError(f"cannot read the git commit of {package_parent}: {err}") from None
        dirty = bool(status)
        return {"commit": head + ("-dirty" if dirty else ""), "dirty": dirty,
                "source": f"git work tree {package_parent}"}
    marker = package_parent / "WRITER_COMMIT"
    if marker.exists():
        commit = marker.read_text().strip()
        if commit:
            return {"commit": commit, "dirty": commit.endswith("-dirty"),
                    "source": str(marker)}
    raise RuntimeError(
        f"no writer commit: {package_parent} is not a git work tree and has no "
        "WRITER_COMMIT file")


def references_text(provenance: dict, version: str = __version__) -> str:
    """The ``references`` global attribute: repository, version, writer commit, conventions page.

    This is the only function that builds the attribute, and
    `rewrite_references` replaces it in finished files, so the text can name
    a release or a new commit later without rewriting any data.
    """
    return (f"Written by giamip {version} ({REPOSITORY_URL}), module giamip.mip.outputs, "
            f"commit {provenance['commit']}. Definitions of every variable and the "
            f"conventions of this submission: {CONVENTIONS_URL}")


# ------------------------------------------------------------------ attributes

def _attribute_value(value):
    """A value h5netcdf can store as an attribute: text, a number or a numeric array."""
    if isinstance(value, (bool, np.bool_, int, float, str, np.integer, np.floating)):
        return value
    if isinstance(value, np.ndarray) and value.dtype.kind in "iuf":
        return value
    return str(value)


# The protocol table gives some fields of an experiment as a reference to
# another experiment, for example the rheology of Exp07, "same as Exp02".
SAME_AS = re.compile(r"^same as (Exp\d{2})$", re.IGNORECASE)


def resolve_case_value(case: dict, *path: str, _seen: tuple = ()):
    """A field of a case with a "same as ExpNN" reference replaced by the value it names.

    The referenced case is read with the G-ADOPT choices applied
    (`cases.with_gadopt_choices`), which is how the run used it, and the
    same field is looked up there. This is the rule `giamip.earth.for_case`
    applies to the rheology, here for any field the writer copies into an
    attribute, so a file never says "same as Exp02" where the reader needs
    the value. A chain of references is followed; a loop raises.

    Parameters
    ----------
    case
        The case, with the G-ADOPT choices applied.
    *path
        Keys into the case, for example ``"earth", "rheology"``.

    Returns
    -------
    The value of the field after every reference is resolved.
    """
    value = case
    for key in path:
        value = value[key]
    match = SAME_AS.match(value) if isinstance(value, str) else None
    if match is None:
        return value
    other = f"giamip-exp{int(match.group(1)[3:]):02d}"
    if other in _seen or other == case["id"]:
        raise ValueError(f"{case['id']}: {'/'.join(path)} refers back to itself through {other}")
    return resolve_case_value(cases.with_gadopt_choices(other), *path,
                              _seen=_seen + (case["id"],))


def global_attributes(export: ExportFile, case: dict, provenance: dict,
                      time_map: TimeMap, export_sha256: str) -> dict:
    """The global attributes of every file of one experiment.

    The four the checker requires come first. Then what defines the run
    (everything that defines the run, so the files can be read without the
    model report): the case constants and the earth structure with the
    G-ADOPT choices, from the case file; the method sentences, each ending
    with the conventions page; and every root attribute of the intermediate
    file under `RUN_ATTRIBUTE_PREFIX`.
    """
    run = export.attrs
    constants = {k: v for k, v in case["constants"].items() if not k.startswith("$")}
    see = f" See {CONVENTIONS_URL}."
    if time_map.is_identity:
        interpolation = "none: the model writes output at every forcing year."
    else:
        # Two different counts: the years the model wrote, and the years
        # made by interpolation. They add up to the forcing years.
        n = int(np.count_nonzero(time_map.interpolated))
        interpolation = (f"the model writes output at {time_map.model_years.size} of the "
                         f"{time_map.forcing_years.size} forcing years; the values at the other "
                         f"{n} years are linear interpolations in time between the two "
                         "neighbouring model outputs. The interpolated years are named in the "
                         "comment of each variable.")
    attrs = {
        "group": GROUP,
        "model": MODEL,
        "contributors_name": CONTRIBUTORS_NAME,
        "contributors_email": CONTRIBUTORS_EMAIL,
        "institution": INSTITUTION,
        "title": f"GIAMIP {export.experiment}, {GROUP} {MODEL}",
        "experiment": export.experiment,
        "case": case["id"],
        "references": references_text(provenance),
        "source": (f"G-ADOPT (https://github.com/g-adopt/g-adopt) branch "
                   f"{run.get('git_branch')} commit {run.get('git_head')}; Firedrake "
                   f"{run.get('firedrake')}; PETSc {run.get('petsc')}; mesh {run.get('mesh')}"),
        # The token alone, so a script can compare it; the explanation follows
        # in its own attribute.
        "reference_frame": "CM",
        "reference_frame_comment": ("Centre of mass of the whole Earth system (solid Earth and "
                                    "surface load). The solver constrains the centre of mass to "
                                    "the origin at every step, so the outputs are in CM with no "
                                    "CE-to-CM conversion." + see),
        # Every case field is resolved, so a "same as ExpNN" of the protocol
        # table becomes the value that experiment used.
        "ice_history": resolve_case_value(case, "forcing", "ice_history"),
        "ice_history_file": resolve_case_value(case, "forcing", "ice_history_file"),
        "topography": resolve_case_value(case, "topography", "model"),
        "earth_viscosity": resolve_case_value(case, "earth", "viscosity"),
        "earth_elastic_structure": resolve_case_value(case, "earth", "elastic_structure"),
        "earth_rheology": resolve_case_value(case, "earth", "rheology"),
        "earth_compressibility": resolve_case_value(case, "earth", "compressibility"),
        **constants,
        "field_regridding": ("delta_bed, delta_g, delta_bed_east and delta_bed_north are "
                             "point values of the finite-element fields at the Gauss-Legendre "
                             "nodes. The two fractions are cell means; see the comment of each "
                             "fraction." + see),
        "scalars": ("The six scalar series are integrals on the native surface mesh of the "
                    "solver, before any regridding (protocol section 6.1)." + see),
        "delta_g_definition": ("delta_g is the change of N + Shift since the first output, "
                               "with N = (psi + psi_rot) / g_s the geoid including the "
                               "rotational potential and Shift the uniform sea-surface "
                               "change." + see),
        "sign_convention": ("delta_bed is positive upward; delta_g is positive when the sea "
                            "surface rises; delta_bed_east and delta_bed_north are positive "
                            "east and north." + see),
        "time_interpolation": interpolation + see,
        "export_file": str(export.path),
        "export_file_sha256": export_sha256,
        "history": (f"{datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')} "
                    f"written by giamip {__version__} giamip.mip.outputs from export_file."),
    }
    for key, value in run.items():
        attrs[RUN_ATTRIBUTE_PREFIX + key] = _attribute_value(value)
    return attrs


def _definition(name: str, case: dict, export: ExportFile) -> str | None:
    """One-sentence definition of a variable for its ``comment``, or None."""
    c = case["constants"]
    method = export.attrs.get("fraction_method", "not recorded")
    definitions = {
        "delta_bed": "Radial displacement u . rhat of the solid surface, positive upward, "
                     "relative to the first output, in the CM frame.",
        "delta_g": "Change of N + Shift relative to the first output, N = (psi + psi_rot) / g_s; "
                   "positive when the sea surface rises; includes the uniform term.",
        "delta_bed_east": "Eastward displacement u . e_east of the solid surface relative to "
                          "the first output.",
        "delta_bed_north": "Northward displacement u . e_north of the solid surface relative "
                           "to the first output.",
        "ocean_area_fraction": "Cell mean of B, which is 1 over open water and floating ice and "
                               "0 over land and grounded ice (the ocean of the sea-level load). "
                               f"Regridding method: {method}.",
        "land_ice_area_fraction": "Cell mean of [I > 1 mm], the indicator of grounded and "
                                  f"floating ice. Regridding method: {method}.",
        "mean_delta_g": "int(delta_g B dS) / int(B dS) on the native mesh, over the ocean B of "
                        "the same step (open water and floating ice).",
        "grd_ice_mass": f"rho_i int((1 - B) I dS) on the native mesh, rho_i = "
                        f"{c['ice_density_kg_m3']} kg/m3.",
        "total_ice_mass": f"rho_i int(I dS) on the native mesh, rho_i = "
                          f"{c['ice_density_kg_m3']} kg/m3.",
        "maf": f"rho_i int(max(0, I - (rho_w / rho_i) max(SL, 0)) dS) on the native mesh, SL the "
               f"water depth of the same step, rho_i = {c['ice_density_kg_m3']} kg/m3, rho_w = "
               f"{c['ocean_water_density_kg_m3']} kg/m3.",
        "ocean_area": "int(B dS) on the native mesh: open water and floating ice.",
        "ocean_area_grdice": "int(C dS) on the native mesh, C = 1 where the sea level is "
                             "positive, so marine regions under grounded ice are included.",
    }
    return definitions.get(name)


def variable_attributes(name: str, meta: dict, case: dict, export: ExportFile,
                        time_map: TimeMap | None) -> dict:
    """units, long_name, standard_name, reference and comment of one variable."""
    attrs = {"units": meta["units"], "long_name": meta["long_name"]}
    if meta.get("standard_name"):
        attrs["standard_name"] = meta["standard_name"]
    if name in VARIABLE_REFERENCE:
        attrs["reference"] = VARIABLE_REFERENCE[name]
    sentences = [s for s in (_definition(name, case, export),) if s]
    if time_map is not None and time_map.comment():
        sentences.append(time_map.comment())
    if sentences:
        attrs["comment"] = " ".join(sentences) + f" See {CONVENTIONS_URL}."
    return attrs


# ------------------------------------------------------------------ files

def _set_coordinate(f, name: str, values) -> None:
    """Create one coordinate or companion variable with the attributes of `COORDINATE_ATTRIBUTES`."""
    spec = COORDINATE_ATTRIBUTES[name]
    var = f.create_variable(name, (name if name != "year" else "time",), spec["dtype"],
                            data=np.asarray(values, dtype=spec["dtype"]))
    _netcdf.set_attributes(var, {k: v for k, v in spec.items() if k != "dtype"})


def _write_time_axis(f, time_map: TimeMap) -> None:
    """``time`` (days, float64) and ``year`` (int32) of the forcing years."""
    f.dimensions["time"] = int(time_map.forcing_years.size)
    years = time_map.forcing_years.astype(np.int32)
    _set_coordinate(f, "time", years.astype(np.float64) * DAYS_PER_YEAR)
    _set_coordinate(f, "year", years)


def _check_values(name: str, values: np.ndarray, meta: dict, where: str) -> None:
    """Raise `OutputError` on NaN, inf or a value outside the checker bounds."""
    if not np.all(np.isfinite(values)):
        bad = np.argwhere(~np.isfinite(values))[0].tolist()
        raise OutputError(f"{name}: non-finite value at {where}, index {bad}")
    bounds = meta.get("bounds")
    if bounds is not None:
        lo, hi = bounds
        vmin, vmax = float(np.min(values)), float(np.max(values))
        if vmin < lo or vmax > hi:
            raise OutputError(f"{name}: value outside the checker bounds [{lo}, {hi}] at "
                              f"{where}: min {vmin}, max {vmax}")


def _check_series(name: str, values: np.ndarray, meta: dict, years: np.ndarray) -> None:
    """`_check_values` for a scalar series, naming the year of the first bad value."""
    bad = ~np.isfinite(values)
    if meta.get("bounds") is not None:
        lo, hi = meta["bounds"]
        with np.errstate(invalid="ignore"):
            bad |= (values < lo) | (values > hi)
    if np.any(bad):
        k = int(np.argmax(bad))
        _check_values(name, values[k:k + 1], meta, f"year {int(years[k])}")


class _Writer:
    """Writes one file to ``<name>.nc.partial`` and renames it when it is complete.

    The checker reads only ``*.nc``, so a failure never leaves a file that
    looks finished; the partial file is removed on any error.
    """

    def __init__(self, path: pathlib.Path):
        self.path = path
        self.partial = path.with_name(path.name + ".partial")

    def __enter__(self):
        if self.partial.exists():
            self.partial.unlink()
        self.file = _netcdf.open_write(self.partial, append=False)
        return self.file

    def __exit__(self, exc_type, exc, tb):
        self.file.close()
        if exc_type is not None:
            self.partial.unlink(missing_ok=True)
            return False
        os.replace(self.partial, self.path)
        return False


def write_field_file(path: pathlib.Path, name: str, export: ExportFile, time_map: TimeMap,
                     global_attrs: dict, var_attrs: dict, meta: dict) -> dict:
    """One gridded variable, ``(time, lat, lon)`` float32, one compressed chunk per slice.

    The slices are made and written one at a time, so memory holds two
    model rows and one output slice. Each slice is cast to float32 after the
    interpolation and checked for finite values and the checker bounds as it
    is stored.

    Returns
    -------
    dict
        The minimum and maximum of the stored values.
    """
    rows = _RowCache(export, name)
    vmin, vmax = np.inf, -np.inf
    with _Writer(path) as f:
        _netcdf.set_attributes(f, global_attrs)
        _write_time_axis(f, time_map)
        f.dimensions["lat"] = grids.NLAT
        f.dimensions["lon"] = grids.NLON
        _set_coordinate(f, "lat", grids.latitudes())
        _set_coordinate(f, "lon", grids.longitudes())
        var = f.create_variable(name, ("time", "lat", "lon"), "f4", chunks=FIELD_CHUNKS,
                                compression="gzip", compression_opts=COMPRESSION_LEVEL)
        _netcdf.set_attributes(var, var_attrs)
        for t in range(time_map.forcing_years.size):
            values = time_map.apply_slice(rows, t).astype(np.float32)
            _check_values(name, values, meta, f"year {int(time_map.forcing_years[t])}")
            var[t, :, :] = values
            vmin, vmax = min(vmin, float(values.min())), max(vmax, float(values.max()))
    return {"min": vmin, "max": vmax}


def write_scalar_file(path: pathlib.Path, name: str, values: np.ndarray, time_map: TimeMap,
                      global_attrs: dict, var_attrs: dict) -> None:
    """One scalar series, ``(time,)`` float64, contiguous (463 values compress to nothing worth it)."""
    with _Writer(path) as f:
        _netcdf.set_attributes(f, global_attrs)
        _write_time_axis(f, time_map)
        var = f.create_variable(name, ("time",), "f8", data=np.asarray(values, dtype=np.float64))
        _netcdf.set_attributes(var, var_attrs)


def write_stokes_file(path: pathlib.Path, name: str, values: np.ndarray, global_attrs: dict,
                      var_attrs: dict) -> None:
    """``Clm`` or ``Slm``, ``(degree, order)`` float64 with float64 axes 0 .. 96, no time axis."""
    n = values.shape[0]
    with _Writer(path) as f:
        _netcdf.set_attributes(f, global_attrs)
        f.dimensions["degree"] = n
        f.dimensions["order"] = n
        _set_coordinate(f, "degree", np.arange(n, dtype=np.float64))
        _set_coordinate(f, "order", np.arange(n, dtype=np.float64))
        var = f.create_variable(name, ("degree", "order"), "f8",
                                data=np.asarray(values, dtype=np.float64))
        _netcdf.set_attributes(var, var_attrs)


# ------------------------------------------------------------------ one experiment

def _row_shift_numbers(export: ExportFile, radius: float) -> dict:
    """The grid integral of ``ocean_area_fraction`` against ``ocean_area``, at three rows.

    The writer cannot re-match the scalars to the fields by step; the export
    does that. These numbers make a misalignment visible: at the right row
    the relative difference is the regridding error of the fraction, and the
    same comparison with the neighbouring row is given beside it. A shifted
    scalar series shows as an aligned difference that is not smaller than
    the shifted one. Report numbers, not a pass criterion.
    """
    area = export.scalar("ocean_area")
    mean_dg = export.scalar("mean_delta_g")
    rows = sorted({0, int(np.argmax(np.abs(mean_dg))), export.n - 1})
    out = []
    for k in rows:
        grid = grids.integrate(export.field("ocean_area_fraction", k), radius)
        neighbour = k + 1 if k + 1 < export.n else k - 1
        out.append({"row": k, "year": float(export.years[k]),
                    "relative_difference": (grid - area[k]) / area[k],
                    "neighbour_row": neighbour,
                    "relative_difference_neighbour": (grid - area[neighbour]) / area[neighbour]})
    return {"ocean_area_grid_against_mesh": out}


def _ice_check(export: ExportFile, ice_year0: np.ndarray | None) -> dict:
    """RMS and largest difference of the exported ice at year 0 against the forcing slice.

    A mirrored or shifted grid, or points in the wrong order, give a
    difference of hundreds of metres, while a correct export differs only by
    the transfer of the ice onto the mesh and back.
    """
    if ice_year0 is None:
        return {"skipped": "no forcing ice slice given"}
    diff = export.check_ice() - ice_year0
    return {"rms_m": float(np.sqrt(np.mean(diff**2))), "max_abs_m": float(np.max(np.abs(diff)))}


def _write_report(report: dict, path) -> None:
    if path is not None:
        pathlib.Path(path).write_text(json.dumps(report, indent=2, default=str) + "\n")


def write_experiment(export_path, out_dir, *, experiment: str | None = None,
                     overwrite: bool = False, forcing_years_override=None,
                     ice_year0=None, report_path=None) -> dict:
    """Write the GIAMIP files of one experiment from its intermediate file.

    Order: the scalars, the mandatory fields, the two horizontal fields
    (left out, never clipped, when outside the bounds), then ``Clm`` and
    ``Slm`` from ``delta_g`` at the last output minus the first. A
    non-finite or out-of-bounds value of any other variable raises
    `OutputError`; the report is written first, so the stop names the
    variable, the year and the value. Files already written stay in place.

    Parameters
    ----------
    export_path
        The intermediate file.
    out_dir
        The experiment directory, for example ``.../G-ADOPT_g-adopt/Exp01``.
        Created if absent.
    experiment
        The experiment the file must be for. None takes it from the file.
    overwrite
        Replace existing output files. Without it an existing file raises
        before anything is written.
    forcing_years_override
        The forcing years, for tests on a short synthetic axis. None reads
        them from the forcing file of the case, which is what production uses.
    ice_year0
        The forcing ice thickness at year 0, ``(257, 513)``, for the check of
        the exported ice. None reads it from the forcing file, unless
        ``forcing_years_override`` is given, in which case the check is
        skipped.
    report_path
        Where to write the JSON report. None writes none.

    Returns
    -------
    dict
        The report: files, skipped variables, min and max of each variable,
        interpolated years, degree-1 Stokes coefficients, the ice check and
        the row-shift numbers.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = variable_table()
    report: dict = {"export_file": str(export_path), "out_dir": str(out_dir),
                    "files": {}, "skipped": [], "values": {}}
    with ExportFile(export_path, experiment) as export:
        exp = export.experiment
        number = int(re.search(r"(\d+)$", exp).group(1))
        case = cases.with_gadopt_choices(f"giamip-exp{number:02d}")
        radius = float(case["constants"]["earth_radius_m"])

        if forcing_years_override is not None:
            years = np.asarray(forcing_years_override)
        else:
            years = forcing_years(exp)
            if ice_year0 is None:
                from giamip.mip.inputs import GiamipExperiment
                ice = GiamipExperiment(exp).ice_history
                ice_year0 = ice.thickness(ice.step_of_year(0.0))
        time_map = TimeMap.from_years(export.years, years)

        names = list(EXPORT_SCALARS) + list(EXPORT_FIELDS) + ["Clm", "Slm"]
        paths = {n: out_dir / file_name(n, exp) for n in names}
        existing = [str(p) for p in paths.values() if p.exists()]
        if existing and not overwrite:
            raise FileExistsError(f"output files exist (pass overwrite=True): {existing}")
        # With overwrite, every output file of this experiment goes before the
        # first write, so the directory never holds files of two runs: a stop
        # half way leaves only files of this run, and the report lists them.
        if overwrite:
            for p in paths.values():
                p.unlink(missing_ok=True)
                p.with_name(p.name + ".partial").unlink(missing_ok=True)

        provenance = writer_provenance()
        sha = data.sha256(pathlib.Path(export_path))
        gattrs = global_attributes(export, case, provenance, time_map, sha)
        report.update({"experiment": exp, "writer": provenance,
                       "model_years": int(export.n),
                       "forcing_years": int(time_map.forcing_years.size),
                       "interpolated_years": int(np.count_nonzero(time_map.interpolated)),
                       "interpolated_ranges": time_map.interpolated_ranges()})

        def done(name, extra=None):
            report["files"][name] = {"path": str(paths[name]),
                                     "bytes": paths[name].stat().st_size}
            if extra:
                report["values"][name] = extra

        try:
            # Scalars: interpolate, check and write.
            for name in EXPORT_SCALARS:
                values = time_map.apply_scalar(export.scalar(name))
                _check_series(name, values, table[name], time_map.forcing_years)
                write_scalar_file(paths[name], name, values, time_map, gattrs,
                                  variable_attributes(name, table[name], case, export, time_map))
                done(name, {"min": float(values.min()), "max": float(values.max())})

            # Optional horizontal components: scanned over the model rows first.
            # A linear interpolation stays inside the range of its end points,
            # so the model rows decide whether the output can be in bounds.
            write_fields = []
            for name in EXPORT_FIELDS:
                if name in OPTIONAL_SKIP_ON_BOUNDS:
                    lo, hi = table[name]["bounds"]
                    vmin = vmax = 0.0
                    for k in range(export.n):
                        row = export.field(name, k).astype(np.float32)
                        vmin, vmax = min(vmin, float(row.min())), max(vmax, float(row.max()))
                    if vmin < lo or vmax > hi:
                        report["skipped"].append({
                            "variable": name, "reason": "outside the checker bounds; "
                            "not written and not clipped", "bounds": [lo, hi],
                            "min": vmin, "max": vmax})
                        print(f"SKIPPED {name}: values {vmin} .. {vmax} outside [{lo}, {hi}]")
                        if paths[name].exists():
                            paths[name].unlink()
                        continue
                write_fields.append(name)

            for name in write_fields:
                extremes = write_field_file(
                    paths[name], name, export, time_map, gattrs,
                    variable_attributes(name, table[name], case, export, time_map), table[name])
                done(name, extremes)

            # Stokes coefficients of delta_g between the first and last outputs
            # (protocol section 7), from the float64 rows of the export, the same
            # regridded field the organisers analyse. The first row should be
            # zero; subtracting it makes the definition exact either way.
            change = export.field("delta_g", export.n - 1) - export.field("delta_g", 0)
            c, s = harmonics.analyse(change, radius)
            for name, values in (("Clm", c), ("Slm", s)):
                attrs = variable_attributes(name, table[name], case, export, None)
                attrs["comment"] = (
                    "Fully normalised (4 pi) real spherical harmonic coefficients without the "
                    "Condon-Shortley phase, dimensionless at the reference radius "
                    f"{case['constants']['earth_radius_m']} m, of delta_g at the final output "
                    "minus delta_g at the first. Rows are the degree, columns the order; "
                    f"zero where order > degree. See {CONVENTIONS_URL}.")
                attrs["first_year"] = int(time_map.model_years[0])
                attrs["final_year"] = int(time_map.model_years[-1])
                write_stokes_file(paths[name], name, values, gattrs, attrs)
                done(name)
            report["degree_1"] = {"C10": float(c[1, 0]), "C11": float(c[1, 1]),
                                  "S11": float(s[1, 1]), "C00": float(c[0, 0])}
            report["ice_check_year0"] = _ice_check(export, ice_year0)
            report.update(_row_shift_numbers(export, radius))
        except OutputError as err:
            report["error"] = str(err)
            _write_report(report, report_path)
            raise
    _write_report(report, report_path)
    return report


def rewrite_references(directory, text: str | None = None) -> list[pathlib.Path]:
    """Replace the ``references`` attribute of every G-ADOPT file in a directory.

    The file is opened in append mode and only that attribute changes; the
    data and every other attribute stay as they are. Use it to name a
    released version or a new commit after the files have been checked.

    Parameters
    ----------
    directory
        An experiment directory.
    text
        The new attribute. None builds it with `references_text` from the
        current `writer_provenance`.
    """
    text = references_text(writer_provenance()) if text is None else text
    changed = []
    for path in sorted(pathlib.Path(directory).glob(f"*_{GROUP}_{MODEL}.nc")):
        with _netcdf.open_write(path, append=True) as f:
            f.attrs["references"] = text
        changed.append(path)
    return changed


# ------------------------------------------------------------------ command line

def main(argv=None) -> int:
    """``write`` one experiment, or ``rewrite-references`` in a directory."""
    parser = argparse.ArgumentParser(prog="python -m giamip.mip.outputs", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    w = sub.add_parser("write", help="write the GIAMIP files of one experiment")
    w.add_argument("--export", required=True, help="the intermediate file export_expXX.h5")
    w.add_argument("--out-dir", required=True, help="the experiment directory to write into")
    w.add_argument("--experiment", help="the experiment the export must be for, e.g. Exp01")
    w.add_argument("--overwrite", action="store_true", help="replace existing files")
    w.add_argument("--report", help="write the JSON report here")
    r = sub.add_parser("rewrite-references", help="replace the references attribute in place")
    r.add_argument("--dir", required=True)
    r.add_argument("--text", help="the new text; default: the current writer commit")
    args = parser.parse_args(argv)
    if args.command == "write":
        report = write_experiment(args.export, args.out_dir, experiment=args.experiment,
                                  overwrite=args.overwrite, report_path=args.report)
        print(json.dumps({k: report[k] for k in ("experiment", "model_years", "forcing_years",
                                                 "interpolated_years", "skipped", "degree_1",
                                                 "ice_check_year0")}, indent=2, default=str))
        print(f"{len(report['files'])} files written to {args.out_dir}")
    else:
        for path in rewrite_references(args.dir, args.text):
            print(f"rewrote references of {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
