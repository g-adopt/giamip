"""Read the Martinec et al. (2018) benchmark data and write it as NetCDF.

The archives Martinec sent in September 2026 (``resultsA_2026/`` to
``resultsE_2026/`` under ``data/original/martinec2018/``) hold, for each case,
every contributor's cross-sections and time series together with the GMT 4
scripts that drew the published figures. The data files come in about eight
layouts, most without a header, and several headers are wrong. This module
does not guess any of it. It takes three inputs:

* ``giamip/resources/column_layouts.json``, decoded from the scripts, which
  says for each plotted curve which file, which column, which sign and which
  scale factor;
* the ``time_series_columns`` table in the same file, which says what unit
  each time-series column is stored in;
* ``giamip/resources/corrections.json``, which lists the known defects.

and writes one NetCDF4 file per case with every plotted curve in SI units, on
an ascending axis, with its provenance in attributes.

Rules that the converter follows
--------------------------------
* A curve is converted only if a script draws it. A commented-out line in a
  script is a curve that was dropped from the figure, and it stays out.
* The sign is the script's sign. The scripts reconcile the contributors' sign
  conventions, which differ; see correction benchmarkA-geoid-sign-varies.
* A scale factor is applied only if it converts a unit: division by the
  surface gravity 9.8155 turns a potential into a height, and the time-series
  factors turn a stored unit into the unit on the axis. A factor that changes
  a curve for the figure only (``9*x/10`` for WW's axisymmetric run) is not
  applied, and is recorded in an attribute.
* Every correction that applies to a file the converter reads must have an
  entry in ``HANDLING``. A correction added to the table without a decision
  here makes the converter raise, so a defect cannot be forgotten.
* The converter never interpolates. Each curve keeps the grid its contributor
  used. Scoring on a common grid is a separate step, and correction
  martinec2018-caseD-vega-profile-grid explains why interpolation of a
  reference needs care.

Layout of the output
--------------------
``data/converted/martinec2018/martinec2018-<case>.nc`` holds three kinds of
group:

``profiles/<load|basin>/<member>``
    The cross-section through the ice centre (``load``) or the basin centre
    (``basin``): U, N and S in metres against colatitude in degrees. The group
    attributes give the longitude and the time from the figure title.
``differences/<load|basin>/<member>``
    The contributor minus VEGA, as shipped in ``Z_<code>_<profile>_diff.out``.
``time_series/<member>``
    h_UF and its components in metres, ice mass in kg, and ocean area as a
    fraction of 4 pi, against time in kyr since the load was first applied.

``<member>`` is the contributor's two-letter code, with ``_axisymmetric`` for
WW's axisymmetric run and ``_case_D`` for the case D run of VEGA that the case
E figures overlay. When two variables in one group sit on different axes, the
second axis is named ``colatitude_2`` or ``time_2``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import numpy as np

from giamip import _netcdf
from giamip import corrections
from giamip import data

COLLECTION = "martinec2018"
RESOURCES = pathlib.Path(__file__).resolve().parent.parent / "resources"
LAYOUTS_PATH = RESOURCES / "column_layouts.json"

# The surface gravity of the M3-L70-V01 gravity profile, in m/s^2. Every
# plotting script divides a potential by this value to obtain a geoid height,
# except the one case C script recorded in martinec2018-caseC-basin-gravity-factor.
GRAVITY = 9.8155

CASES = ("A", "B", "C", "D", "E")

# The prefix each contributor file carries inside a case directory. Case A has
# none. The evidence for B is in docs/collections/martinec2018.qmd. For C, D
# and E it is the directory itself: resultsC_2026/ holds only D3_ files, and its
# two cross-section scripts, which read them, are titled "Benchmark C".
INTERNAL_CODES = {"A": None, "B": "C2", "C": "D3", "D": "E2", "E": "F1"}

# Cross-section quantities by the y-axis label of their panel: the variable
# name and whether the panel shows a value or a difference from VEGA.
PROFILE_AXES = {
    "U(theta,t) (m)": ("U", "profiles"),
    "N(theta,t) (m)": ("N", "profiles"),
    "S(theta,t) (m)": ("S", "profiles"),
    "delta U(theta,t) (m)": ("U", "differences"),
    "delta N(theta,t) (m)": ("N", "differences"),
    "delta S(theta,t) (m)": ("S", "differences"),
}

# Time-series quantities by the y-axis label of their panel. `si_factor`
# converts the axis unit to SI: 10^17 kg to kg, per cent of 4 pi to a fraction.
# tests/test_layouts.py uses the same table to check the transcribed units.
TIME_SERIES_AXES = {
    "huf (m)": {"units": "m", "si_factor": 1.0},
    "Ice mass (10^17 kg)": {"units": "kg", "si_factor": 1e17},
    "Ocean area (% of 4pi)": {"units": "1", "si_factor": 0.01},
}

LONG_NAMES = {
    "U": "vertical displacement of the solid surface, positive upward",
    "N": "geoid displacement, positive upward",
    "S": "sea-surface variation, positive upward",
    "h_UF": "thickness of the uniform water layer",
    "h_UF_1": "component 1 of h_UF, -rho_ice h_ice / (O rho_w)",
    "h_UF_2": "component 2 of h_UF",
    "h_UF_3": "component 3 of h_UF",
    "h_UF_4": "component 4 of h_UF, water displaced by floating ice",
    "h_UF_1_plus_h_UF_4": "sum of components 1 and 4 of h_UF",
    "ice_mass_total": "total ice mass",
    "ice_mass_grounded": "grounded ice mass",
    "ice_mass_floating": "floating ice mass",
    "ocean_area": "ocean area as a fraction of the full solid angle 4 pi",
    "ocean_area_initial": "ocean area at t = 0 as a fraction of 4 pi",
}

# What the converter does about each correction that can apply to a file it
# reads. The key set is checked on every file: a correction that applies and
# is not listed here makes the converter raise.
HANDLING = {
    "benchmarkA-sk-superseded": "skip: the profile is a stale copy and is not converted",
    "benchmarkA-colatitude-ordering": "every axis is sorted ascending on read",
    "benchmarkA-fortran-exponent": "D exponents are rewritten as E before parsing",
    "benchmarkA-ww-axisymmetric-rescaled": "the figure-only factors 9*x/10 and 2*x/3 are not applied",
    "benchmarkA-geoid-sign-varies": "the sign is taken from column_layouts.json",
    "martinec2018-geoid-panel-is-negated": "the geoid is negated after the script's own sign",
    "benchmarkA-gs-repeated-last-row": "a point repeated with the same value is dropped",
    "benchmarkA-vega-supplied-run-differs": "converted, and flagged as a second VEGA run in an attribute",
    "martinec2018-caseC-basin-gravity-factor": "the potential is divided by 9.8155, not by 10",
    "martinec2018-vega-time-series-header": "column meaning and unit are taken from column_layouts.json",
    "martinec2018-secti-column5": "column 5 is written as h_UF_1_plus_h_UF_4",
    "martinec2018-vk-header-labels": "column 1 is read as colatitude or as time in kyr",
    "martinec2018-vk-huf-header": "column 5 is written as the total h_UF",
    "martinec2018-time-series-units": "column meaning and unit are taken from column_layouts.json",
    "martinec2018-ys-case-e-awk-exponent": "the exponent is read, and the figure's factor 10 is not taken as a unit",
    "martinec2018-caseD-vega-profile-grid": "converted on its own grid, and flagged in an attribute (also the case B basin)",
}

# Scale expressions that change a curve for the figure only. The data are
# written as submitted; see benchmarkA-ww-axisymmetric-rescaled.
FIGURE_ONLY_SCALES = {"9*x/10", "2*x/3"}


def converted_dir() -> pathlib.Path:
    """Where the converter writes by default: ``data/converted/martinec2018`` of the checkout.

    The converted files are tracked in the repository, so the converter writes
    into a checkout. An installed package has none, and then ``--out`` is
    required.
    """
    checkout = data.checkout_root()
    if checkout is None:
        raise SystemExit("giamip is installed, not checked out; pass an output directory")
    return checkout / "data" / "converted" / COLLECTION


def original_root(case: str) -> pathlib.Path:
    """The ``original/`` directory holding a case's unpacked archive, fetched if needed."""
    return data.original_tree(f"{COLLECTION}/results{case}_2026.zip")


def load_layouts(path: pathlib.Path = LAYOUTS_PATH) -> dict:
    """The decoded column layouts, as written by scripts/decode_gmt_layouts.py."""
    return json.loads(path.read_text())


def scale_factor(scale: str | None) -> float:
    """Evaluate a script's scale expression as a multiplier.

    The decoder writes the arithmetic an awk expression applies with the
    column replaced by ``x``: ``x/9.8155``, ``x*100``, ``x/1e17``, ``9*x/10``.
    Only a product of numbers and ``x`` with ``*`` and ``/`` is accepted, which
    covers every expression in the scripts. Anything else raises, so a new kind
    of arithmetic is noticed instead of being evaluated.

    Parameters
    ----------
    scale
        The expression, or None for a column used as it stands.

    Returns
    -------
    float
        The factor by which the column is multiplied.
    """
    if scale is None:
        return 1.0
    number = r"[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?"
    m = re.fullmatch(rf"(?:(?P<pre>{number})\*)?x(?P<ops>(?:[*/]{number})*)",
                     scale.replace(" ", ""))
    if not m:
        raise ValueError(f"unsupported scale expression {scale!r}")
    factor = float(m.group("pre")) if m.group("pre") else 1.0
    for op, value in re.findall(rf"([*/])({number})", m.group("ops")):
        factor = factor * float(value) if op == "*" else factor / float(value)
    return factor


def read_table(path: pathlib.Path) -> np.ndarray:
    """The numeric rows of one benchmark data file, as a 2-D float array.

    Comment lines starting with ``#`` and blank lines are skipped. Lines before
    the first numeric row that do not parse are a header and are skipped too,
    which covers the one-line column names of the GS files. A line that does
    not parse after numeric rows have started raises, because that is a
    corrupt file and not a header. Fortran ``D`` exponents are rewritten first
    (correction benchmarkA-fortran-exponent). Rows shorter than the widest
    row are padded with NaN.

    Parameters
    ----------
    path
        The file to read.

    Returns
    -------
    numpy.ndarray
        Shape (rows, columns). Column n of the file is index n - 1.
    """
    rows: list[list[float]] = []
    text = corrections.fortran_exponents(path.read_text(errors="replace"))
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            rows.append([float(v) for v in stripped.split()])
        except ValueError:
            if rows:
                raise ValueError(f"{path}: line {number} does not parse: {line!r}")
    if not rows:
        raise ValueError(f"{path}: no numeric rows")
    width = max(len(r) for r in rows)
    return np.array([r + [np.nan] * (width - len(r)) for r in rows])


def applying(relative_path: str) -> list[str]:
    """The ids of the corrections that apply to a file, checked against HANDLING.

    Raises
    ------
    KeyError
        If a correction applies and the converter has no decision for it.
    """
    ids = [e["id"] for e in corrections.applying_to(relative_path)]
    unknown = [i for i in ids if i not in HANDLING]
    if unknown:
        raise KeyError(f"{relative_path}: corrections {unknown} apply and "
                       f"giamip.benchmarks.martinec2018.HANDLING has no entry for them")
    return ids


def _column(table: np.ndarray, column: int, relative_path: str) -> np.ndarray:
    """Column n (1-based) of a table, with a clear error when it is absent."""
    if column < 1 or column > table.shape[1]:
        raise ValueError(f"{relative_path}: column {column} requested, "
                         f"file has {table.shape[1]}")
    return table[:, column - 1]


def _profile_name(script: str) -> str:
    """``load`` for a cross-section through the ice centre, ``basin`` otherwise."""
    name = script.rsplit("/", 1)[-1]
    if "_basin_" in name:
        return "basin"
    if "_load_" in name or name.startswith("A_fig10"):
        return "load"
    raise ValueError(f"{script}: not a cross-section script")


def _title_attributes(script: dict) -> dict:
    """Longitude and time of a cross-section, from the drawn figure title."""
    out = {}
    for title in script["titles"]:
        if not title["drawn"]:
            continue
        m = re.search(r"lambda=([\d.]+)°", title["text"])
        if m:
            out["longitude_deg"] = float(m.group(1))
        m = re.search(r"t=([\d.]+) kyr", title["text"])
        if m:
            out["time_kyr"] = float(m.group(1))
        out["figure_title"] = title["text"]
    return out


class Group:
    """The variables of one NetCDF group, each with its own axis until written.

    Variables that share an axis array exactly share a dimension in the file.
    """

    def __init__(self, axis_name: str, axis_units: str, axis_long_name: str):
        self.axis_name = axis_name
        self.axis_units = axis_units
        self.axis_long_name = axis_long_name
        self.variables: dict[str, dict] = {}
        self.attributes: dict[str, object] = {}

    def add(self, name: str, axis: np.ndarray, values: np.ndarray,
            attributes: dict) -> None:
        """Add one variable, sorted by its axis. A second, different source raises."""
        order = np.argsort(axis, kind="stable")
        axis, values = axis[order], values[order]
        # A point repeated with the same value is dropped, as in the case A GS
        # files, whose last row is written twice (benchmarkA-gs-repeated-last-row).
        # A repeated axis value with a different value is a real ambiguity and
        # raises.
        repeat = np.diff(axis) == 0
        if np.any(repeat):
            if not np.array_equal(values[1:][repeat], values[:-1][repeat], equal_nan=True):
                raise ValueError(f"{attributes['source_file']}: the axis repeats a "
                                 f"value with different data")
            keep = np.concatenate([[True], ~repeat])
            attributes["repeated_points_dropped"] = int((~keep).sum())
            axis, values = axis[keep], values[keep]
        if name in self.variables:
            old = self.variables[name]
            same_source = (old["attributes"]["source_file"] == attributes["source_file"]
                           and old["attributes"]["source_column"] == attributes["source_column"])
            if same_source and np.array_equal(old["values"], values):
                return
            raise ValueError(
                f"{name} arrives from {attributes['source_file']} column "
                f"{attributes['source_column']} and from "
                f"{old['attributes']['source_file']} column {old['attributes']['source_column']}")
        self.variables[name] = {"axis": axis, "values": values, "attributes": attributes}


def extract(case: str, root: pathlib.Path | None = None,
            layouts: dict | None = None) -> dict:
    """Read every drawn curve of one case into memory.

    Parameters
    ----------
    case
        The published case letter, A to E.
    root
        The ``original/`` directory. Default: `original_root`, which finds
        or fetches the case's archive through `giamip.data.locate`.
    layouts
        The decoded layouts. Default: read from the package resources.

    Returns
    -------
    dict
        ``groups`` maps a group path to a `Group`. ``skipped`` lists the files
        not converted, each with the correction that says why.
    """
    if case not in CASES:
        raise ValueError(f"unknown case {case!r}; expected one of {CASES}")
    root = root or original_root(case)
    layouts = layouts or load_layouts()
    prefix = f"martinec2018/results{case}_2026/"
    scripts = [s for s in layouts["scripts"] if s["script"].startswith(prefix)]
    if not scripts:
        raise ValueError(f"no decoded scripts for case {case} under {prefix}")
    table_units = layouts["time_series_columns"]

    groups: dict[str, Group] = {}
    skipped: dict[str, str] = {}
    tables: dict[str, np.ndarray] = {}
    digests: dict[str, str] = {}

    def table(relative_path: str) -> np.ndarray:
        if relative_path not in tables:
            full = root / relative_path
            tables[relative_path] = read_table(full)
            digests[relative_path] = data.sha256(full)
        return tables[relative_path]

    for script in scripts:
        titles = _title_attributes(script)
        for panel in script["panels"]:
            for series in panel["series"]:
                if not series["drawn"] or series["kind"] == "map":
                    continue
                rel = series["path"]
                ids = applying(rel)
                if "benchmarkA-sk-superseded" in ids:
                    skipped[rel] = "benchmarkA-sk-superseded"
                    continue
                member = series["contributor"]
                if series.get("variant"):
                    member += "_" + series["variant"]
                tab = table(rel)
                x = _column(tab, series["x"]["column"], rel)
                y = _column(tab, series["y"]["column"], rel)
                if series["x"]["negated"] or series["x"]["scale"]:
                    raise ValueError(f"{rel}: arithmetic on the abscissa is not expected")
                attributes = {
                    "source_file": rel,
                    "source_sha256": digests[rel],
                    "source_column": series["y"]["column"],
                    "abscissa_column": series["x"]["column"],
                    "script": script["script"],
                    "panel": panel["panel"] if panel["panel"] is not None else -1,
                    "axis_label": panel["quantity"],
                    "negated_by_script": int(series["y"]["negated"]),
                    "script_scale": series["y"]["scale"] or "",
                    "corrections": " ".join(ids),
                }

                if series["kind"] == "cross_section":
                    name, kind = PROFILE_AXES[panel["quantity"]]
                    scale = series["y"]["scale"]
                    if scale in FIGURE_ONLY_SCALES:
                        corrections.by_id("benchmarkA-ww-axisymmetric-rescaled")
                        factor = 1.0
                        attributes["figure_only_scale_not_applied"] = scale
                    elif scale == "x/10" and "martinec2018-caseC-basin-gravity-factor" in ids:
                        factor = 1.0 / GRAVITY
                    else:
                        factor = scale_factor(scale)
                        if scale is not None and abs(factor * GRAVITY - 1.0) > 1e-12:
                            raise ValueError(f"{rel}: unexpected scale {scale!r} on {name}")
                    sign = -1.0 if series["y"]["negated"] else 1.0
                    if name == "N":
                        # The geoid panel holds minus the geoid displacement,
                        # after the per-file signs above have brought the
                        # contributors into agreement with each other. The
                        # scripts draw the VEGA potential column as $5/9.8155,
                        # with no minus sign, while eq. 9 of the paper is
                        # N = -Phi/g0. See the correction
                        # martinec2018-geoid-panel-is-negated, which carries two
                        # independent checks: S = -N + h_UF in the shipped case C
                        # data, and 1 + k positive at every epoch for this earth
                        # model. Applied here so that every N in
                        # data/converted/ is positive upward, as
                        # docs/reference/conventions.qmd states.
                        corrections.by_id("martinec2018-geoid-panel-is-negated")
                        sign = -sign
                    values = sign * factor * y
                    if "benchmarkA-vega-supplied-run-differs" in ids:
                        attributes["note"] = ("A second VEGA run, not the one behind the "
                                              "published figure; see the correction.")
                    if "martinec2018-caseD-vega-profile-grid" in ids:
                        attributes["note"] = ("A coarse copy of the reference; do not "
                                              "interpolate it for scoring; see the correction.")
                    attributes.update(units="m", long_name=LONG_NAMES[name]
                                      if kind == "profiles" else
                                      f"{LONG_NAMES[name]}, contributor minus VEGA",
                                      scale_applied=factor * sign)
                    group_path = f"{kind}/{_profile_name(script['script'])}/{member}"
                    group = groups.setdefault(group_path, Group(
                        "colatitude", "degrees", "colatitude, from the north pole"))
                    group.attributes.update(titles)
                    group.add(name, x, values, attributes)

                elif series["kind"] == "time_series":
                    axis = TIME_SERIES_AXES[panel["quantity"]]
                    entry = table_units.get(rel)
                    if entry is None:
                        raise KeyError(f"{rel}: no time_series_columns entry")
                    column = entry["columns"].get(str(series["y"]["column"]))
                    if column is None:
                        raise KeyError(f"{rel}: column {series['y']['column']} not transcribed")
                    if entry["time_column"] != series["x"]["column"]:
                        raise ValueError(f"{rel}: the script's time column differs from the table")
                    # The transcribed unit must reproduce the figure: the stored
                    # value times the script's factor, in the axis unit, is the
                    # same quantity as the stored value times si_factor in SI.
                    implied = scale_factor(series["y"]["scale"]) * axis["si_factor"]
                    # awk ignores a Fortran D exponent and reads the mantissa, so
                    # a script can plot a value 10^n times larger than the file
                    # holds. Allowed only where the correction says so.
                    dropped = entry.get("awk_exponent_dropped", 0)
                    if dropped:
                        if "martinec2018-ys-case-e-awk-exponent" not in ids:
                            raise ValueError(f"{rel}: awk_exponent_dropped without its correction")
                        implied *= 10.0 ** dropped
                    if series["y"]["negated"] or abs(column["si_factor"] - implied) > 1e-12 * abs(implied):
                        raise ValueError(f"{rel}: si_factor {column['si_factor']} disagrees "
                                         f"with the figure, which implies {implied}")
                    if entry.get("run_of_case"):
                        member += f"_case_{entry['run_of_case']}"
                    name = column["name"]
                    attributes.update(units=axis["units"], long_name=LONG_NAMES[name],
                                      stored_units=column["units"],
                                      scale_applied=column["si_factor"])
                    group = groups.setdefault(f"time_series/{member}", Group(
                        "time", "kyr", "time since the load was first applied"))
                    group.add(name, x, column["si_factor"] * y, attributes)

    groups.update(reference_groups(case, groups))
    return {"groups": groups, "skipped": skipped}


# Tolerance, in degrees, for two colatitude axes to count as the same grid when
# a recovery subtracts a difference file from a profile. The archive writes
# colatitude with at least five decimals, so 1e-9 degrees accepts only the same
# numbers and never a neighbouring grid.
SAME_GRID_DEG = 1e-9


def reference_groups(case: str, groups: dict) -> dict:
    """The curves that scoring compares with, one group per profile and one for the time series.

    The case file's ``reference_data.reference_curves`` says, for each
    compared profile and for the time series, how the reference is built:

    ``as shipped``
        VEGA's own curve (member ZM) from the archive, copied.
    ``recovered``
        A contributor's profile minus that contributor's difference file, on
        the contributor's grid. Every difference file of the archive is the
        contributor minus one finer VEGA profile that the archive does not
        hold, so this gives that profile. YS is used: its 2048-point grid is
        VEGA's own grid in the profiles where the shipped VEGA file is fine,
        and it reproduces every other contributor's recovery to at most
        1.7e-4 m (docs/protocol/martinec2018.qmd, "How the reference is
        built").

    Only the quantities the case file compares are built, so the group holds
    exactly what `giamip.benchmarks.metrics.score` needs.

    Parameters
    ----------
    case
        The published case letter.
    groups
        The member groups of `extract`, keyed by group path.

    Returns
    -------
    dict
        ``reference/profiles/<load|basin>`` and ``reference/time_series`` to a
        `Group`, for the parts the case compares.

    Raises
    ------
    ValueError
        When a compared profile or time series has no entry in
        ``reference_curves``, when an entry names a member or a quantity the
        archive does not hold, or when a recovery's profile and difference
        file are on different grids.
    """
    # Imported here: inputs reads the case file through giamip.cases, and
    # the converter module is imported by the result classes, so a module-level
    # import would load the case machinery for every reader.
    from giamip.benchmarks.inputs import MartinecCase

    setup = MartinecCase(case)
    rules = setup.definition["reference_data"].get("reference_curves", {})
    out: dict[str, Group] = {}

    def put(group: Group, name: str, axis, values, attributes: dict) -> None:
        """Store a derived curve, ascending in its axis.

        `Group.add` tells two sources of one variable apart by their source
        file and column, which a derived curve does not have, so the curve
        is stored directly. Its axis comes from one member group, where
        `Group.add` already sorted it and removed repeated points.
        """
        axis = np.asarray(axis, dtype=float)
        if np.any(np.diff(axis) <= 0):
            raise ValueError(f"case {case}: the reference {name} axis does not ascend")
        group.variables[name] = {"axis": axis, "values": np.asarray(values, dtype=float),
                                 "attributes": attributes}

    def member_variable(kind: str, where: str, member: str, quantity: str) -> dict:
        path = f"{kind}/{where}/{member}" if where else f"{kind}/{member}"
        if path not in groups or quantity not in groups[path].variables:
            raise ValueError(f"case {case}: reference_curves needs {quantity} in {path}, "
                             f"which the archive does not hold")
        return groups[path].variables[quantity]

    for spec in setup.profiles:
        rule = rules.get("profiles", {}).get(spec.name)
        if rule is None:
            raise ValueError(f"case {case}: the {spec.name} profile is compared and "
                             f"reference_curves has no entry for it")
        group = Group("colatitude", "degrees", "colatitude, from the north pole")
        for quantity in spec.quantities:
            source = member_variable("profiles", spec.name, rule["member"], quantity)
            attributes = {"units": "m", "long_name": LONG_NAMES[quantity],
                          "method": rule["method"], "member": rule["member"],
                          "correction": rule.get("correction", "")}
            if rule["method"] == "as shipped":
                axis, values = source["axis"], source["values"]
                attributes["source_files"] = source["attributes"]["source_file"]
                attributes["built_from"] = (f"VEGA's {quantity} curve {source['attributes']['source_file']}"
                                            f" as shipped")
            elif rule["method"] == "recovered":
                difference = member_variable("differences", spec.name, rule["member"], quantity)
                if (source["axis"].shape != difference["axis"].shape
                        or np.max(np.abs(source["axis"] - difference["axis"])) > SAME_GRID_DEG):
                    raise ValueError(f"case {case}: {rule['member']}'s {spec.name} profile and "
                                     f"difference file are on different grids")
                axis = difference["axis"]
                # The profile minus its own difference, point by point on one
                # grid: no interpolation enters.
                values = source["values"] - difference["values"]
                files = (source["attributes"]["source_file"],
                         difference["attributes"]["source_file"])
                attributes["source_files"] = " ".join(files)
                attributes["built_from"] = (f"VEGA recovered as {rule['member']}'s {quantity} "
                                            f"({files[0]}) minus {rule['member']}'s difference "
                                            f"file ({files[1]}); correction "
                                            f"{rule.get('correction', '')}")
            else:
                raise ValueError(f"case {case}: unknown method {rule['method']!r}")
            put(group, quantity, axis, values, attributes)
        out[f"reference/profiles/{spec.name}"] = group

    compared_series = [q for s in setup.time_series for q in s.quantities]
    if compared_series:
        rule = rules.get("time_series")
        if rule is None:
            raise ValueError(f"case {case}: time series are compared and reference_curves "
                             f"has no entry for them")
        if rule["method"] != "as shipped":
            raise ValueError(f"case {case}: time series can only be taken as shipped")
        group = Group("time", "kyr", "time since the load was first applied")
        for quantity in compared_series:
            source = member_variable("time_series", "", rule["member"], quantity)
            attributes = {"units": source["attributes"]["units"],
                          "long_name": LONG_NAMES[quantity],
                          "method": rule["method"], "member": rule["member"],
                          "correction": rule.get("correction", ""),
                          "source_files": source["attributes"]["source_file"],
                          "built_from": (f"VEGA's {quantity} series "
                                         f"{source['attributes']['source_file']} as shipped")}
            put(group, quantity, source["axis"], source["values"], attributes)
        out["reference/time_series"] = group
    return out


def write(case: str, extracted: dict, destination: pathlib.Path) -> pathlib.Path:
    """Write one case, as returned by `extract`, to a netCDF-4 file through h5netcdf.

    Parameters
    ----------
    case
        The published case letter.
    extracted
        The result of `extract`.
    destination
        The file to write. Its directory is created. An existing file is
        replaced only after the new one is complete.

    Returns
    -------
    pathlib.Path
        The file written.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    partial.unlink(missing_ok=True)
    # The file carries no creation date, so that a conversion of the same
    # originals with the same library versions gives the same bytes and
    # therefore the same object key on the store (giamip/data.py).
    with _netcdf.open_write(partial, append=False) as ds:
        _netcdf.set_attributes(ds, {
            "title": f"Martinec et al. (2018) benchmark case {case}: every curve drawn in the published figures",
            "case": case,
            "internal_code": INTERNAL_CODES[case] or "",
            "source_directory": f"martinec2018/results{case}_2026",
            "source_citation": ("Martinec, Z. et al. (2018). A benchmark study of numerical "
                                "implementations of the sea level equation in GIA modelling. "
                                "Geophysical Journal International 215(1), 389-414. doi:10.1093/gji/ggy280"),
            "converter": "giamip.benchmarks.martinec2018",
            "column_layouts": "giamip/resources/column_layouts.json",
            "conventions_reference": "docs/reference/conventions.qmd",
            "skipped": "; ".join(f"{p}: {c}" for p, c in sorted(extracted["skipped"].items())),
        })
        for group_path, group in sorted(extracted["groups"].items()):
            g = _netcdf.create_group(ds, group_path)
            _netcdf.set_attributes(g, group.attributes)
            _netcdf.write_curves(
                g, [(name, variable["axis"], variable["values"], variable["attributes"])
                    for name, variable in sorted(group.variables.items())],
                axis_name=group.axis_name,
                coordinate_attributes={"units": group.axis_units,
                                       "long_name": group.axis_long_name})
    partial.replace(destination)
    return destination


def convert(case: str, root: pathlib.Path | None = None,
            out_dir: pathlib.Path | None = None) -> pathlib.Path:
    """Extract one case and write ``martinec2018-<case>.nc``.

    Parameters
    ----------
    case
        The published case letter, A to E.
    root
        The ``original/`` directory. Default: `original_root`.
    out_dir
        Where to write. Default: `converted_dir`.
    """
    out_dir = out_dir or converted_dir()
    return write(case, extract(case, root), out_dir / f"martinec2018-{case}.nc")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--case", action="append", choices=CASES,
                    help="case to convert; repeatable. Default: all five")
    ap.add_argument("--root", type=pathlib.Path, help="the original/ directory")
    ap.add_argument("--out", type=pathlib.Path, help="output directory")
    args = ap.parse_args(argv)
    for case in args.case or CASES:
        extracted = extract(case, args.root)
        out_dir = args.out or converted_dir()
        path = write(case, extracted, out_dir / f"martinec2018-{case}.nc")
        n = sum(len(g.variables) for g in extracted["groups"].values())
        print(f"case {case}: {len(extracted['groups'])} groups, {n} variables -> {path}")
        for p, c in sorted(extracted["skipped"].items()):
            print(f"  skipped {p} ({c})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
