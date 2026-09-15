"""Read the Spada et al. (2011) benchmark data and write it as NetCDF.

The COST Action ES0701 archive (``spada2011/spada2011-cost-es0701/`` under
``data/original/``) is a working copy: contributor data in about six layouts,
GMT 4 scripts, plotting copies of the data and runs that no figure used. This
module does not guess any of it. It takes two inputs:

* the ``files`` table of ``gia_mip/resources/spada_column_layouts.json``,
  written by ``scripts/decode_spada_layouts.py``, which names every file a
  reader uses and says what each column holds, in which unit and at which
  epoch, with the evidence;
* ``gia_mip/resources/corrections.json``, which lists the known defects.

and writes one NetCDF4 file per case in the layout of
``gia_mip.benchmarks.results.SpadaResult``, so a converted contributor is read
exactly as a solver's result is.

Rules that the converter follows
--------------------------------
* A file is converted only if the layout table names it for the case. For
  tests 1/2 and 2/2 that means the files the published script draws, which
  keeps out the runs with a different load
  (correction spada-t02-gs-runs-with-a-different-load).
* Every correction that applies to a file the converter reads, for the
  version of the file on disk, must have an entry in ``HANDLING``. A
  correction added to the table without a decision here makes the converter
  raise, so a defect cannot be forgotten.
* Units are converted to those of `SpadaResult`: U, V, N in m, rates in m/kyr,
  s_j and residues in 1/kyr, times in kyr, polar motion in degrees and its
  rate in degrees per Myr, the length of day in ms. The stored unit comes
  from the layout table, and the rate units are checked against the factor
  the published figure of test 2/2 applies.
* Residues are stored in the convention of the paper, h_f = h_e - sum r_j / s_j,
  with the three Maxwell modes of TABOO dropped by position, and modes are
  stored slowest first for every contributor.
* The converter never interpolates. Each curve keeps its contributor's grid,
  except that one code's files of one quantity are joined into one curve
  where they agree at their shared points: VK's fine and global VILMA grids
  of tests 1/2 and 2/2, and in test 3/2 GS's curves with its tables and VB's
  grid with its table.
* A contributor that the case file lists and that has no data in the archive
  must be named, with a reason, in the case file's
  ``reference_data.not_converted``; otherwise the converter raises.

Which files, per case
---------------------
``t01-01``  the relaxation spectra spec_T01_01_<member>.dat of GS, RR, VB, ZM.
``t01-02``  the loading Love numbers [hlk]_L_T01-02_<member>.dat, same four.
``t01-03``  the tidal Love numbers [hlk]_T_T01-03_<member>.dat of GS, RR, VB.
``t01-04``  the transfer functions GS-[yn]cw.dat and VB-[yn]cw.txt, with and
            without the Chandler wobble.
``t01-05``  Love numbers against time: the files T01-05/PAPER/love_PAP.gmt
            draws for figure 6, GS's and VK's [hlk]_L_T01-05_*.mod* and GSA's
            [hlk].mod*.
``t01-06``  the files T01-06/love_PAP.gmt draws for figure 7: VK's
            I106-M3-VSCAb-LLN-[hlk]-VK.mod*, GSA's [hlk].mod* and RR's
            res-RR-time[123]-[hkl].dat.
``t01-07``  the degree-one files T01-07/degree1.gmt draws for figure 8, for
            GS, GSA, RR, VB and VK.
``t02-01``  U, V, N: GS's per-epoch TABOO profiles of the cap and disc, GS's
            point load in T02/, and VK's VILMA files of the cap and disc.
``t02-02``  the same for the rates.
``t02-03``  polar motion: GS's PM_plot_*.dat and other_data/PM_table_*.dat of
            the three loads, VB's six tables and its two grids of the cap; the
            length of day lod_[GS|GSA|VK].dat of the cap.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np

from gia_mip import cases
from gia_mip import corrections
from gia_mip import data

RESOURCES = pathlib.Path(__file__).resolve().parent.parent / "resources"
LAYOUTS_PATH = RESOURCES / "spada_column_layouts.json"
ARCHIVE = "spada2011/spada2011-cost-es0701"

CASES = ("t01-01", "t01-02", "t01-03", "t01-04", "t01-05", "t01-06", "t01-07",
         "t02-01", "t02-02", "t02-03")

# How many m/kyr one stored rate unit is: 1 mm/yr = 1e-3 m / 1e-3 kyr = 1 m/kyr,
# and 1 m/yr = 1000 m/kyr. Displacements are stored in m and need no factor.
# This is the one definition in the package. scripts/decode_spada_layouts.py
# imports it to check each stored rate unit against the scale that
# T02-02/plot_PAP.gmt applies, before it writes the layouts this module reads.
# tests/test_spada_records.py checks the same units with a copy of its own, so
# that an error in this table cannot pass its own test.
IN_M_PER_KYR = {"mm/yr": 1.0, "m/yr": 1000.0, "m/kyr": 1.0}

# What the converter does about each correction that can apply to a file it
# reads. Entries for corrections on files it does not read say so, so that
# the table is a complete account of the Spada defects.
HANDLING = {
    "spada-paper-copy-differs-t01-03-vb": "not read: the converter reads the signed .dat files of the working directory",
    "spada-paper-copy-differs-t01-05-vk": "not read: the converter reads T01-05/PAPER/k_L_T01-05_VK.mod, which figure 6 draws and which holds k",
    "spada-t01-05-vk-k-column-holds-minus-one-minus-k": "not read: the k files of T01-05/PAPER/, which hold k, are read",
    "spada-vk-time-zero-written-as-1e-12-kyr": "a time of 1e-12 kyr is stored as 0, and the attribute time_zero_restored_from_kyr says so",
    "spada-vk-first-rows-read-as-header": "converted as drawn, from the first time the file holds, and the attribute rows_lost_at_the_top gives the count",
    "spada-t01-06-banner-names-gs-for-vk-files": "not read: the contributor comes from the file name, which says VK",
    "spada-t01-07-rr-six-of-nine-modes": "converted as drawn: the published curve, within 2.3e-5 of the sum over all nine modes",
    "spada-t01-07-gs-l-in-column-3": "l is read from column 3, as the layout table records",
    "spada-t02-03-k2-files-hold-the-length-of-day": "not read: the length of day comes from lod_<member>.dat",
    "spada-t02-03-gs-tables-with-twice-the-values": "not read: the layout table names only PM_table_<load>_<setting>.dat",
    "spada-t02-03-vb-point-mass-header": "the mass in the header is not read; the load comes from the load line, and its parameters from loads/spada2011.json",
    "spada-t02-01-contributors-without-data": "BL, PG and ZM are named in the case files' not_converted, and scoring is against GS and VK",
    "spada-t02-time-layout-differs-by-contributor": "the epoch of each column is taken from the layout table",
    "spada-t01-residue-files-absolute": "not read: residues come from the signed .dat files",
    "spada-t01-gs-twelve-residues": "residue positions 6, 9 and 12 are dropped by position, as the layout table marks them",
    "spada-t01-01-mod-relaxation-times-in-years": "not read: s_j comes from the .dat files, in 1/kyr",
    "spada-t01-01-gs-spectrum-header-unit": "the unit and the mode count are taken from the layout table",
    "spada-t01-01-zm-degrees-copy-degree-2": "the rows are kept, and degrees that copy an earlier row are listed in degrees_not_to_score",
    "spada-t01-zm-mode-order": "a file marked fastest_first in the layout table has its modes reversed",
    "spada-t01-rr-repeated-degrees": "the first row of a repeated degree is kept when the repeat agrees, and the degrees are listed in repeated_degrees",
    "spada-t01-vb-residue-sign-and-wrap": "records are joined across lines, and residues are multiplied by the layout table's residue_sign",
    "spada-t02-taboo-header-and-repeated-last-row": "comment lines are skipped, and a repeated point with the same values is dropped",
    "spada-t02-gs-rate-header-unit": "the unit mm/yr is taken from the layout table",
    "spada-t02-gs-runs-with-a-different-load": "not read: the layout table names only the files the published figure draws",
    "spada-t02-point-load-lmax-512": "converted, with the harmonic degree range of the file in an attribute",
    "spada-t02-vilma-rate-unit": "m/yr is multiplied by 1000 to give m/kyr",
}


def data_root() -> pathlib.Path:
    """Root of the data tree: ``$GIA_MIP_DATA_ROOT``, else ``data/`` in the checkout."""
    root = os.environ.get("GIA_MIP_DATA_ROOT")
    return pathlib.Path(root) if root else data.ROOT / "data"


def load_layouts(path: pathlib.Path = LAYOUTS_PATH) -> dict:
    """The ``files`` table of the decoded layouts."""
    files = json.loads(path.read_text()).get("files")
    if not files:
        raise KeyError(f"{path} has no `files` table; run scripts/decode_spada_layouts.py")
    return files


def applying(relative_path: str, sha256: str) -> list[str]:
    """The ids of the corrections that apply to one file version, checked against HANDLING.

    Raises
    ------
    KeyError
        If a correction applies and the converter has no decision for it. The
        message names the ids.
    """
    ids = [e["id"] for e in corrections.applying_to(relative_path, sha256)]
    unknown = [i for i in ids if i not in HANDLING]
    if unknown:
        raise KeyError(f"{relative_path}: corrections {unknown} apply and "
                       f"gia_mip.benchmarks.spada2011.HANDLING has no entry for them")
    return ids


def _records(path: pathlib.Path, per_record: int) -> np.ndarray:
    """The numbers of a table as records of ``per_record`` values.

    Lines starting with ``#`` (TABOO headers, correction
    spada-t02-taboo-header-and-repeated-last-row) or with the GMT segment
    marker ``>`` are skipped. Values are joined across lines before they are
    split into records, which reads VB's records that wrap onto two lines
    (correction spada-t01-vb-residue-sign-and-wrap). Fortran D exponents are
    rewritten first.
    """
    values: list[float] = []
    text = corrections.fortran_exponents(path.read_text(errors="replace"))
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped[0] in "#>":
            continue
        values.extend(float(v) for v in stripped.split())
    if not values or len(values) % per_record:
        raise ValueError(f"{path}: {len(values)} values do not form records of {per_record}")
    return np.array(values).reshape(-1, per_record)


def _one_row_per_degree(table: np.ndarray, compare: slice, relative: str) -> tuple[np.ndarray, list[int]]:
    """Keep the first row of each degree, after checking that repeats agree.

    RR writes degrees 29 and 40 twice (correction spada-t01-rr-repeated-degrees).
    ``compare`` selects the columns that must agree between the two rows: all
    of them for a spectrum, elastic and fluid for Love numbers. A repeat that
    disagrees there is a real ambiguity and raises.
    """
    degrees = table[:, 0].astype(int)
    keep, repeated = [], []
    first: dict[int, int] = {}
    for i, n in enumerate(degrees):
        if n in first:
            if not np.array_equal(table[first[n], compare], table[i, compare]):
                raise ValueError(f"{relative}: degree {n} repeats with different values")
            repeated.append(int(n))
            continue
        first[n] = i
        keep.append(i)
    return table[keep], sorted(set(repeated))


def _copied_degrees(table: np.ndarray) -> list[int]:
    """Degrees whose row, apart from the degree, equals the row of an earlier degree.

    ZM's spectrum gives degrees 3 and 4 the values of degree 2
    (correction spada-t01-01-zm-degrees-copy-degree-2).
    """
    copies = []
    for i in range(1, table.shape[0]):
        if any(np.array_equal(table[i, 1:], table[j, 1:]) for j in range(i)):
            copies.append(int(table[i, 0]))
    return copies


class _Source:
    """One file the converter reads: its layout, its sum and its corrections."""

    def __init__(self, root: pathlib.Path, relative: str, layout: dict):
        self.relative = relative
        self.layout = layout
        self.path = root / relative
        self.sha256 = data.sha256(self.path)
        # Every file read passes through HANDLING before any value is used.
        self.corrections = applying(relative, self.sha256)

    def attributes(self) -> dict:
        """Provenance attributes shared by everything read from this file."""
        return {"source_file": self.relative, "source_sha256": self.sha256,
                "contributor": self.layout["contributor"],
                "corrections": " ".join(self.corrections),
                "layout_evidence": self.layout["evidence"]}


def _spectrum(source: _Source, result) -> None:
    """Add one relaxation spectrum, modes slowest first, rows one per degree."""
    table = _records(source.path, source.layout["record_values"])
    table, repeated = _one_row_per_degree(table, slice(None), source.relative)
    columns = source.layout["columns"]
    order = [int(k) - 1 for k, c in sorted(columns.items(), key=lambda kv: int(kv[0]))
             if c["quantity"] == "s"]
    s = table[:, order]
    if source.layout["mode_order"] == "fastest_first":
        s = s[:, ::-1]
    extra = source.attributes()
    if repeated:
        extra["repeated_degrees"] = " ".join(map(str, repeated))
    copies = _copied_degrees(table)
    if copies:
        if "spada-t01-01-zm-degrees-copy-degree-2" not in source.corrections:
            raise ValueError(f"{source.relative}: degrees {copies} copy an earlier row "
                             f"and no correction records it")
        extra["degrees_not_to_score"] = " ".join(map(str, copies))
    extra.pop("source_file")
    result.add_relaxation_spectrum(table[:, 0], s, source_file=source.relative, **extra)


def _love_numbers(source: _Source, result) -> None:
    """Add one Love number table in the paper's residue convention, modes slowest first."""
    layout = source.layout
    table = _records(source.path, layout["record_values"])
    table, repeated = _one_row_per_degree(table, slice(0, 3), source.relative)
    columns = sorted(layout["columns"].items(), key=lambda kv: int(kv[0]))
    q = layout["quantity"]
    index = {c["quantity"]: int(k) - 1 for k, c in columns if "mode" not in c}
    # Residue columns in mode order; Maxwell columns carry no mode and are left out.
    residue_columns = [int(k) - 1 for k, c in sorted(
        ((k, c) for k, c in columns if c["quantity"] == f"{q}_residue"),
        key=lambda kv: kv[1]["mode"])]
    residues = table[:, residue_columns] * float(layout["residue_sign"])
    if layout["mode_order"] == "fastest_first":
        residues = residues[:, ::-1]
    extra = source.attributes()
    extra.pop("source_file")
    extra["love_number_type"] = layout["love_number_type"]
    extra["residue_sign_applied"] = int(layout["residue_sign"])
    if repeated:
        extra["repeated_degrees"] = " ".join(map(str, repeated))
    result.add_love_numbers(q, table[:, 0], table[:, index[f"{q}_elastic"]],
                            table[:, index[f"{q}_fluid"]], residues,
                            source_file=source.relative, **extra)


def _profile(axis: np.ndarray, values: np.ndarray, relative: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Sort a profile and drop a point repeated with the same value.

    TABOO writes colatitude 180 twice (correction
    spada-t02-taboo-header-and-repeated-last-row). A repeated colatitude with a
    different value raises.
    """
    order = np.argsort(axis, kind="stable")
    axis, values = axis[order], values[order]
    repeat = np.diff(axis) == 0
    if np.any(repeat):
        if not np.array_equal(values[1:][repeat], values[:-1][repeat]):
            raise ValueError(f"{relative}: a colatitude repeats with different values")
        keep = np.concatenate([[True], ~repeat])
        return axis[keep], values[keep], int((~keep).sum())
    return axis, values, 0


def _spatial_columns(source: _Source) -> list[tuple[str, float, np.ndarray, np.ndarray, dict]]:
    """Every (quantity, epoch, colatitude, values, attributes) a profile file holds.

    Only U, V, N and their rates are returned; TABOO's longitudinal column is
    zero along a meridian and has no place in the result.
    """
    columns = source.layout["columns"]
    table = _records(source.path, len(columns))
    axis_column = next(int(k) - 1 for k, c in columns.items() if c["quantity"] == "colatitude")
    out = []
    for key, column in sorted(columns.items(), key=lambda kv: int(kv[0])):
        quantity = column["quantity"]
        if quantity not in ("U", "V", "N", "U_rate", "V_rate", "N_rate"):
            continue
        if quantity.endswith("_rate"):
            factor = IN_M_PER_KYR[column["units"]]
        elif column["units"] == "m":
            factor = 1.0
        else:
            raise ValueError(f"{source.relative}: {quantity} in {column['units']}")
        axis, values, dropped = _profile(table[:, axis_column], factor * table[:, int(key) - 1],
                                         source.relative)
        attributes = source.attributes()
        attributes.update(source_column=int(key), stored_units=column["units"],
                          scale_applied=factor)
        if dropped:
            attributes["repeated_points_dropped"] = dropped
        if "harmonic_degrees" in source.layout:
            attributes["harmonic_degrees"] = " ".join(map(str, source.layout["harmonic_degrees"]))
        if "load_mass_kg" in source.layout:
            attributes["load_mass_kg"] = source.layout["load_mass_kg"]
        out.append((quantity, float(column["epoch_kyr"]), axis, values, attributes))
    return out


def _join_grids(pieces: list[tuple[np.ndarray, np.ndarray, dict]], label: str) -> tuple:
    """Join VK's fine and global grids into one curve.

    The fine grid runs from 0.25 to 15 degrees and the global grid from 1 to
    179; both omit the poles. Where they overlap the values must be the same to
    1e-9 relative (they are identical in the archive), and the fine grid's
    point is kept. The joined curve names the fine grid's file as its source and
    lists both files in ``joined_from``.
    """
    pieces = sorted(pieces, key=lambda p: p[2]["grid"] != "fine")
    axis, values, attributes = pieces[0][0], pieces[0][1], dict(pieces[0][2])
    for other_axis, other_values, other_attributes in pieces[1:]:
        shared = np.isin(other_axis, axis)
        mine = np.searchsorted(axis, other_axis[shared])
        if not np.allclose(values[mine], other_values[shared], rtol=1e-9, atol=0.0):
            raise ValueError(f"{label}: the fine and global grids disagree where they overlap")
        axis = np.concatenate([axis, other_axis[~shared]])
        values = np.concatenate([values, other_values[~shared]])
        order = np.argsort(axis)
        axis, values = axis[order], values[order]
        attributes["joined_from"] = " ".join(p[2]["source_file"] for p in pieces)
        attributes["joined_sha256"] = " ".join(p[2]["source_sha256"] for p in pieces)
    attributes.pop("grid", None)
    return axis, values, attributes


def _table(path: pathlib.Path) -> np.ndarray:
    """Rows of numbers as a 2-D array, skipping `#` lines and GMT segment markers.

    Every row must have the same width, else this raises: a table whose rows
    differ is not a table the layout describes.
    """
    rows = []
    for line in corrections.fortran_exponents(path.read_text(errors="replace")).splitlines():
        stripped = line.strip()
        if not stripped or stripped[0] in "#>":
            continue
        rows.append([float(v) for v in stripped.split()])
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise ValueError(f"{path}: rows of widths {sorted(widths)}")
    return np.array(rows)


def _segment_tables(path: pathlib.Path) -> list[tuple[str, np.ndarray]]:
    """A GMT multi-segment table as (marker line, rows) per segment, `#` lines skipped."""
    segments: list[tuple[str, list]] = []
    for line in path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith(">"):
            segments.append((stripped, []))
        elif segments:
            segments[-1][1].append([float(v) for v in stripped.split()])
        else:
            raise ValueError(f"{path}: data before the first segment marker")
    return [(marker, np.array(rows)) for marker, rows in segments]


def _restore_time_zero(source: _Source, time: np.ndarray, attributes: dict) -> np.ndarray:
    """Store as 0 the time that a VK file wrote as 1e-12 kyr.

    Only where the layout table records the substitution, and only when the
    correction that describes it applies to this version of the file
    (spada-vk-time-zero-written-as-1e-12-kyr); otherwise the time is left as
    it stands.
    """
    zero = source.layout.get("time_zero_written_as_kyr")
    if zero is None:
        return time
    if "spada-vk-time-zero-written-as-1e-12-kyr" not in source.corrections:
        raise ValueError(f"{source.relative}: the layout table records a time zero written as "
                         f"{zero} kyr, and no correction applies to this version")
    time = np.array(time, dtype=float)
    time[np.isclose(time, zero, rtol=1e-6, atol=0.0)] = 0.0
    attributes["time_zero_restored_from_kyr"] = zero
    return time


def _time_domain(source: _Source, result) -> None:
    """Add every degree of one time-domain Love number file, in the format its layout names.

    ``segments_degree_column``: one segment per degree, degree in column 3.
    ``segments_degree_marker``: the degree on each `>` line. ``columns_per_degree``
    and ``value_column``: a table whose value columns the layout lists with
    their degrees. The degrees found must be those of the layout table.
    """
    layout = source.layout
    quantity, form = layout["quantity"], layout["format"]
    curves: list[tuple[int, np.ndarray, np.ndarray, int]] = []
    if form == "segments_degree_column":
        for _, rows in _segment_tables(source.path):
            degrees = np.unique(rows[:, 2])
            if degrees.size != 1:
                raise ValueError(f"{source.relative}: a segment holds degrees {degrees}")
            curves.append((int(degrees[0]), rows[:, 0], rows[:, 1], 2))
    elif form == "segments_degree_marker":
        for marker, rows in _segment_tables(source.path):
            curves.append((int(marker.lstrip(">").split()[0]), rows[:, 0], rows[:, 1], 2))
    elif form in ("columns_per_degree", "value_column"):
        table = _table(source.path)
        for key, column in sorted(layout["columns"].items(), key=lambda kv: int(kv[0])):
            if column["quantity"] == quantity:
                curves.append((int(column["degree"]), table[:, 0], table[:, int(key) - 1], int(key)))
        # GS's l file of test 7/1 has a zero column 2 and l in column 3.
        if layout["columns"].get("2", {}).get("quantity") == "zero":
            if "spada-t01-07-gs-l-in-column-3" not in source.corrections:
                raise ValueError(f"{source.relative}: the layout skips a zero column 2 and no "
                                 f"correction applies to this version")
            if np.any(table[:, 1] != 0.0):
                raise ValueError(f"{source.relative}: column 2 is not zero")
    else:
        raise ValueError(f"{source.relative}: unknown format {form!r}")
    if [c[0] for c in curves] != list(layout["degrees"]):
        raise ValueError(f"{source.relative}: degrees {[c[0] for c in curves]}, and the layout "
                         f"table has {layout['degrees']}")
    if "rows_lost_at_the_top" in layout and "spada-vk-first-rows-read-as-header" not in source.corrections:
        raise ValueError(f"{source.relative}: the layout table records lost rows and no "
                         f"correction applies to this version")
    for degree, time, values, column in curves:
        attributes = source.attributes()
        attributes.update(source_column=column, stored_units="1")
        if "rows_lost_at_the_top" in layout:
            attributes["rows_lost_at_the_top"] = layout["rows_lost_at_the_top"]
        time = _restore_time_zero(source, time, attributes)
        result.add_time_domain_love_numbers(quantity, degree, time, values, **attributes)


def _polar_motion_pieces(source: _Source) -> list[tuple[tuple, np.ndarray, np.ndarray, dict]]:
    """Every (member, load, setting, quantity) curve one polar motion file holds.

    The curves are returned, not added, because one code can hold one quantity
    in two files, which `_join_files` joins.
    """
    layout = source.layout
    table = _table(source.path)
    columns = layout["columns"]
    time_column = next(int(k) - 1 for k, c in columns.items() if c["quantity"] == "time")
    pieces = []
    for key, column in sorted(columns.items(), key=lambda kv: int(kv[0])):
        if column["quantity"] == "time":
            continue
        attributes = source.attributes()
        attributes.update(source_column=int(key), stored_units=column["units"],
                          header_label=column["header_label"])
        pieces.append(((layout["contributor"], layout["load"], layout["chandler_wobble"],
                        column["quantity"]), table[:, time_column], table[:, int(key) - 1], attributes))
    return pieces


def _join_files(pieces: list[tuple[np.ndarray, np.ndarray, dict]], label: str) -> tuple:
    """Join one quantity of one code from several files into one curve.

    The file with the most points is the base and names the curve's source
    file. Points of the other files at times the base does not hold are added.
    At shared times the values must agree to 1e-6 relative, else this raises:
    GS's tables repeat its curves to every printed digit, and VB's grid repeats
    its table to 1e-12. All files are listed in ``joined_from``.
    """
    pieces = sorted(pieces, key=lambda p: (-p[0].size, p[2]["source_file"]))
    order = np.argsort(pieces[0][0], kind="stable")
    axis, values, attributes = pieces[0][0][order], pieces[0][1][order], dict(pieces[0][2])
    for other_axis, other_values, _ in pieces[1:]:
        shared = np.isin(other_axis, axis)
        mine = np.searchsorted(axis, other_axis[shared])
        if not np.allclose(values[mine], other_values[shared], rtol=1e-6, atol=1e-12):
            raise ValueError(f"{label}: the files disagree at their shared times")
        axis = np.concatenate([axis, other_axis[~shared]])
        values = np.concatenate([values, other_values[~shared]])
        order = np.argsort(axis, kind="stable")
        axis, values = axis[order], values[order]
    if len(pieces) > 1:
        attributes["joined_from"] = " ".join(p[2]["source_file"] for p in pieces)
        attributes["joined_sha256"] = " ".join(p[2]["source_sha256"] for p in pieces)
    return axis, values, attributes


def _length_of_day(source: _Source, result) -> None:
    """Add one length of day file: time in kyr and the change in length of day in ms."""
    table = _table(source.path)
    attributes = source.attributes()
    attributes.update(source_column=2, stored_units="ms")
    time = _restore_time_zero(source, table[:, 0], attributes)
    result.add_length_of_day(source.layout["load"], time, table[:, 1], **attributes)


def _polar_motion_transfer_function(source: _Source, result) -> None:
    """Add one transfer function, from GS's labelled blocks or VB's unlabelled rows.

    A part the file does not print, the imaginary A_e and A_s of GS's file
    without the Chandler wobble, is zero, and the attribute
    ``imaginary_parts_not_printed`` names it.
    """
    layout = source.layout
    text = source.path.read_text(errors="replace")
    parts: dict[str, list[float]] = {}
    if layout["format"] == "labelled_blocks":
        block = None
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                words = stripped.lstrip("#").split()
                if words and words[0] in ("ELA", "SEC", "ROT", "ISO"):
                    block = words[0]
                continue
            if block not in layout["blocks"]:
                continue
            values = [float(v) for v in stripped.split()[1:]]
            if len(values) != len(layout["blocks"][block]):
                raise ValueError(f"{source.relative}: a row of block {block} has {len(values)} values")
            for name, value in zip(layout["blocks"][block], values):
                parts.setdefault(name, []).append(value)
    elif layout["format"] == "rows":
        rows = [[float(v) for v in line.split()]
                for line in text.splitlines()[layout["header_lines"]:] if line.strip()]
        for number, row in enumerate(rows, start=1):
            names = layout["rows"][str(number)] if number <= 2 else layout["rows"]["3 and later"]
            if len(row) != len(names):
                raise ValueError(f"{source.relative}: row {number} has {len(row)} values")
            for name, value in zip(names, row):
                parts.setdefault(name, []).append(value)
    else:
        raise ValueError(f"{source.relative}: unknown format {layout['format']!r}")
    frequencies = np.array(parts["frequency_real"]) + 1j * np.array(parts["frequency_imag"])
    residues = np.array(parts["residue_real"]) + 1j * np.array(parts["residue_imag"])
    if frequencies.size != layout["modes"]:
        raise ValueError(f"{source.relative}: {frequencies.size} modes, and the layout table has {layout['modes']}")
    attributes = source.attributes()
    attributes.pop("source_file")
    missing = [name for name in ("elastic", "secular") if f"{name}_imag" not in parts]
    if missing:
        attributes["imaginary_parts_not_printed"] = " ".join(missing)
    elastic = complex(parts["elastic_real"][0], parts.get("elastic_imag", [0.0])[0])
    secular = complex(parts["secular_real"][0], parts.get("secular_imag", [0.0])[0])
    result.add_polar_motion_transfer_function(layout["chandler_wobble"], elastic, secular,
                                              frequencies, residues, source_file=source.relative,
                                              **attributes)


def extract(case: str, root: pathlib.Path | None = None, layouts: dict | None = None) -> dict:
    """Read every file of one case into one `SpadaResult` per contributor.

    Parameters
    ----------
    case
        A short case name of `CASES`, for example ``t02-01``.
    root
        The ``original/`` directory. Default: ``data_root() / "original"``.
    layouts
        The ``files`` table. Default: read from the package resources.

    Returns
    -------
    dict
        Member code to `SpadaResult`.
    """
    from gia_mip.benchmarks.results import SpadaResult

    case = case.lower().removeprefix("spada2011-")
    if case not in CASES:
        raise ValueError(f"case {case!r} is not converted; expected one of {CASES}")
    root = pathlib.Path(root) if root else data_root() / "original"
    layouts = layouts or load_layouts()
    chosen = {path: layout for path, layout in layouts.items() if case in layout["cases"]}
    if not chosen:
        raise ValueError(f"the layout table names no file for {case}")

    sources = [_Source(root, path, layout) for path, layout in sorted(chosen.items())]

    members: dict[str, SpadaResult] = {}
    pending: dict[tuple, list] = {}
    polar: dict[tuple, list] = {}
    for source in sources:
        member = source.layout["contributor"]
        result = members.setdefault(member, SpadaResult(case, member))
        kind = source.layout["kind"]
        if kind == "relaxation_spectrum":
            _spectrum(source, result)
        elif kind == "love_numbers":
            _love_numbers(source, result)
        elif kind == "spatial":
            load = source.layout["load"]
            for quantity, epoch, axis, values, attributes in _spatial_columns(source):
                if "grid" in source.layout:
                    attributes["grid"] = source.layout["grid"]
                    pending.setdefault((member, load, quantity, epoch), []).append(
                        (axis, values, attributes))
                else:
                    result.add_spatial(load, quantity, epoch, axis, values, **attributes)
        elif kind == "time_domain_love_numbers":
            _time_domain(source, result)
        elif kind == "polar_motion":
            for key, axis, values, attributes in _polar_motion_pieces(source):
                polar.setdefault(key, []).append((axis, values, attributes))
        elif kind == "length_of_day":
            _length_of_day(source, result)
        elif kind == "polar_motion_transfer_function":
            _polar_motion_transfer_function(source, result)
        else:
            raise ValueError(f"{source.relative}: unknown kind {kind!r}")
    for (member, load, quantity, epoch), pieces in sorted(pending.items()):
        axis, values, attributes = _join_grids(pieces, f"{member} {load} {quantity} {epoch}")
        members[member].add_spatial(load, quantity, epoch, axis, values, **attributes)
    for (member, load, setting, quantity), pieces in sorted(polar.items()):
        axis, values, attributes = _join_files(pieces, f"{member} {load} {setting} {quantity}")
        members[member].add_polar_motion(load, setting, quantity, axis, values, **attributes)

    # Every contributor of the case is converted or has a recorded reason.
    definition = cases.case(f"spada2011-{case}")
    reasons = definition["reference_data"].get("not_converted", {})
    for contributor in definition["reference_data"]["contributors"]:
        if contributor not in members and not str(reasons.get(contributor, "")).strip():
            raise ValueError(f"{case}: contributor {contributor} has no data and no reason "
                             f"in reference_data.not_converted")
    return members


def write(case: str, members: dict, destination: pathlib.Path) -> pathlib.Path:
    """Write every member of one case into one NetCDF4 file.

    The file is written beside the destination and moved into place only when
    complete, so a failed run never leaves a half-written reference file.
    """
    import netCDF4

    destination = pathlib.Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    partial.unlink(missing_ok=True)
    for member in sorted(members):
        members[member].write(partial)
    definition = cases.case(f"spada2011-{case}")
    with netCDF4.Dataset(partial, "a") as ds:
        ds.title = (f"Spada et al. (2011) {definition['published_name']}, "
                    f"{definition['title']}: every contributor in the archive")
        ds.source_citation = ("Spada, G. et al. (2011). A benchmark study for glacial isostatic "
                              "adjustment codes. Geophysical Journal International 185(1), "
                              "106-132. doi:10.1111/j.1365-246X.2011.04952.x")
        ds.source_archive = f"{ARCHIVE}.tar.gz"
        ds.converter = "gia_mip.benchmarks.spada2011"
        ds.column_layouts = "gia_mip/resources/spada_column_layouts.json"
        ds.not_converted = "; ".join(
            f"{code}: {reason}" for code, reason in
            sorted(definition["reference_data"].get("not_converted", {}).items()))
    partial.replace(destination)
    return destination


def convert(case: str, root: pathlib.Path | None = None,
            out_dir: pathlib.Path | None = None) -> pathlib.Path:
    """Extract one case and write ``spada2011-<case>.nc``.

    Parameters
    ----------
    case
        A short case name of `CASES`.
    root
        The ``original/`` directory. Default: ``data_root() / "original"``.
    out_dir
        Where to write. Default: ``data_root() / "converted" / "spada2011"``.
    """
    case = case.lower().removeprefix("spada2011-")
    members = extract(case, root)
    out_dir = pathlib.Path(out_dir) if out_dir else data_root() / "converted" / "spada2011"
    return write(case, members, out_dir / f"spada2011-{case}.nc")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--case", action="append", choices=CASES,
                    help="case to convert; repeatable. Default: all ten")
    ap.add_argument("--root", type=pathlib.Path, help="the original/ directory")
    ap.add_argument("--out", type=pathlib.Path, help="output directory")
    args = ap.parse_args(argv)
    for case in args.case or CASES:
        path = convert(case, args.root, args.out)
        print(f"case {case} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
