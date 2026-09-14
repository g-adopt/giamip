"""Decode what each Martinec et al. (2018) data file contains, from the GMT scripts.

For most of the Martinec et al. (2018) data files there is no header, or a
header that is wrong. The only reliable record of which column holds which
quantity is the GMT 4 script that drew the published figure, in lines of the
form::

    awk '{print $1, -$4}' $inpSK | psxy -R -JX $lineSK -K -O >> pom.ps

which says: for contributor SK, take column 1 as the abscissa and the negation
of column 4 as the ordinate, and plot it on the panel currently being drawn.
The panel's y-axis label says which quantity that is.

GMT 4 no longer runs, so this script reads the record and writes it down while
it can still be checked against the figures shipped alongside. It is a
decoder, not a plotter: nothing here draws anything.

Two families of script are read. The cross-section scripts
(``A_fig10_diff.gmt`` and ``figZ_uvf_{load,basin}_diff.gmt``) draw profiles
through the ice centre or the basin centre, mark each panel with a
``# Panel N`` comment, and take their axis labels from shell variables. The
time-series scripts of cases C, D and E (``fig_huf*``, ``fig_mass_ice*`` and
``fig_of00*``) draw the uniform water layer, the ice mass and the ocean area
against time. They have no panel comments and write their axis labels inline,
so a panel there starts at a ``psbasemap`` call with a new plot region.

Several things make the parse more than a regular expression:

* Shell variables holding filenames are reassigned between panels, so
  ``$inpWW`` means ``u_10ka.dat`` in one panel and ``n_10ka.dat`` in the next,
  and assignments have to be followed in source order.
* The awk expression can carry arithmetic: ``$5/9.8155`` turns a potential
  into a geoid height, ``$4*100`` turns a fraction into a percentage, and
  ``9*$2/10`` rescales a curve for the figure only.
* A ``psbasemap`` call is often split over two lines with a trailing
  backslash, and the y-axis label sits on the second line.
* The same filename, for example ``Z_ZM_load.out``, occurs in six directories
  with six different contents, so every file is identified by its path
  relative to ``data/original/`` and never by its name alone.

Some facts about the columns are not in any script: the layout of the VEGA
files, which are documented only in their own headers, and the unit and
meaning of each time-series column, which differ between contributors. Those
are transcribed below as constants, with the evidence for each, and written
into the same output so that a reader has one place to look. The output file
is generated. Do not edit it by hand; edit this script and run it again.

Usage::

    python3 scripts/decode_gmt_layouts.py [--root data/original] \\
        [--output gia_mip/resources/column_layouts.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

# `inpGS=A-fig10-TABOO_256.dat` and friends. Values are never quoted in these
# scripts and never contain spaces.
ASSIGN = re.compile(r"^\s*(\w+)=([^\s#]+)\s*$")

# `stit1="U(@~q@~,t) (m)"`. The value carries GMT escapes for Greek letters,
# which are translated below.
LABEL = re.compile(r'^\s*(stit\w*)="(.*)"\s*$')

# The line that does the work. Captures the awk expressions and the file. The
# time-series scripts sometimes put `grep -v "#" $inp1 |` in front, which
# changes nothing because awk is given the file again as its own argument.
AWK = re.compile(
    r"""awk\s+'\{print\s+(?P<fields>[^}]+?)\s*\}'\s+\$(?P<var>\w+)"""
)

# An axis label inside a -B option: `:"${stit4}":` or `:"huf (m)":`. GMT 4
# writes the axis settings as -B<x>/<y>, so when a basemap line carries two
# labels the first is the abscissa and the second is the ordinate. A panel is
# named by its ordinate, so the LAST label on the line is the one to take. A
# panel whose abscissa is unlabelled has one label, and it is the ordinate.
AXIS_LABEL = re.compile(r':"([^"]*)":')
LABEL_REFERENCE = re.compile(r"^\$\{(stit\w*)\}$")

# The plot region of a psbasemap call, `-R0/15/-50/10`. Used to tell a new
# panel from a second psbasemap that redraws the frame of the current one.
REGION = re.compile(r"-R(\S+)")

# `# Panel 3`
PANEL = re.compile(r"^#+\s*Panel\s+(\d+)", re.IGNORECASE)

# `120  -40 10  0  0  ML ZM` inside a pstext heredoc: the legend entry.
LEGEND = re.compile(r"^\s*[-\d.]+\s+[-\d.]+\s+\d+\s+\d+\s+\d+\s+\w+\s+(\w+)\s*$")

# `0.1  1.6  14  0  0  ML Benchmark C (@~l@~=75\260, t=15 kyr)`: the figure
# title, which names the published case, the longitude of the profile and the
# time. It is the evidence that ties a directory's internal file codes (D3,
# E2, F1) to the published case letters.
TITLE = re.compile(r"\bML\s+(Benchmark\b.*?)\s*$")

# The directory an archive unpacked into names the case: `benchmarkA/` from
# benchmarkA.zip, `resultsC_2026/` from resultsC_2026.zip.
CASE_DIRECTORY = re.compile(r"(?:^|/)(?:benchmark|results)([A-E])(?:_\d{4})?$")

# GMT 4 writes Greek by switching font: @~q@~ is theta, @~l@~ lambda, @~p@~ pi,
# @~S@~ capital sigma and @~D@~ capital delta. \260 is the octal code of the
# degree sign in the ISOLatin1 encoding GMT 4 uses. Sub- and superscripts are
# handled before this table is applied, see `clean_label`.
GREEK = {"@~q@~": "theta", "@~l@~": "lambda", "@~p@~": "pi",
         "@~S@~": "Sigma", "@~D@~": "delta ", "@~F@~": "Phi",
         "@~": "", "\\260": "°"}


def clean_label(raw: str) -> str:
    """Turn a GMT axis label or title into plain text.

    ``@+17@+`` is a superscript and becomes ``^17``; ``@-1@-`` is a subscript
    and becomes ``_1``. The Greek font switches are translated afterwards, so
    a subscript that holds a Greek letter, ``U@-@~q@~@-``, reads ``U_theta``.

    Parameters
    ----------
    raw
        The label as written between the quotes in the script.

    Returns
    -------
    str
        The label with escapes translated and runs of spaces collapsed.
    """
    text = re.sub(r"@\+(.*?)@\+", r"^\1", raw)
    text = re.sub(r"@-(.*?)@-", r"_\1", text)
    for token, plain in GREEK.items():
        text = text.replace(token, plain)
    return " ".join(text.split())


def join_continuations(lines: list[str]) -> list[str]:
    """Join shell lines that end in a backslash with the line that follows.

    The psbasemap calls put the y-axis label on the continuation line. Joining
    first means a basemap is always seen as one line, with both its region and
    its labels.
    """
    joined: list[str] = []
    buffer = ""
    for line in lines:
        stripped = line.rstrip()
        if stripped.endswith("\\"):
            buffer += stripped[:-1]
            continue
        joined.append(buffer + line)
        buffer = ""
    if buffer:
        joined.append(buffer)
    return joined


def parse_expression(expr: str) -> dict:
    """Split an awk print expression into a column and what is done to it.

    ``$2`` is column 2 unchanged. ``-$4`` is column 4 negated. ``$5/9.8155``,
    ``$4*100`` and ``9*$2/10`` carry a scale factor, recorded with the column
    replaced by ``x``. Anything with more than one column reference is recorded
    verbatim and left for a person to read.

    Parameters
    ----------
    expr
        One comma-separated field of the awk print statement.

    Returns
    -------
    dict
        ``column`` (1-based), ``negated``, ``scale`` (None when the column is
        used as it stands) and the original ``expression``.
    """
    columns = [int(c) for c in re.findall(r"\$(\d+)", expr)]
    if len(columns) != 1:
        return {"expression": expr.strip(), "columns": columns,
                "note": "more than one column, or none; read the expression"}
    column = columns[0]
    body = expr.strip()
    negated = body.startswith("-")
    scale = None
    stripped = body.lstrip("-")
    if stripped != f"${column}":
        scale = stripped.replace(f"${column}", "x")
    return {"column": column, "negated": negated, "scale": scale,
            "expression": body}


def contributor_of(variable: str) -> dict:
    """The contributor code a shell variable stands for.

    The scripts name their input variables after the contributor's two-letter
    code: ``inpGS`` is GS's profile and ``inpGS0`` GS's difference file.
    ``inpWWaxi`` and ``inpWW0axi`` are WW's axisymmetric run. VEGA, the
    reference solution by ZM, is ``inp2`` in the cross-section scripts and
    ``inp1`` or ``file1`` in the time-series scripts, and case E's scripts
    overlay a case D run of VEGA as ``fileE2`` or ``inp1E2``.

    Returns
    -------
    dict
        ``contributor`` (None when the variable follows no pattern) and, for
        WW's axisymmetric run, ``variant``.
    """
    name = re.sub(r"^(inp|file)", "", variable)
    if name in {"1", "2", "E2", "1E2"}:
        return {"contributor": "ZM"}
    m = re.fullmatch(r"([A-Z]{2})0?(axi)?", name)
    if not m:
        return {"contributor": None}
    out = {"contributor": m.group(1)}
    if m.group(2):
        out["variant"] = "axisymmetric"
    return out


def decode(path: pathlib.Path, root: pathlib.Path) -> dict:
    """Decode one GMT script into its panels and the series drawn on each.

    Parameters
    ----------
    path
        The script.
    root
        The ``data/original/`` directory, against which every path in the
        result is written.

    Returns
    -------
    dict
        ``script`` (the path relative to ``root``), ``case``, ``titles`` and
        ``panels``. Each panel has its number, ``quantity`` (the y-axis
        label), ``abscissa_label`` when the script sets one, ``region`` for
        time-series panels, ``inset`` when the panel is drawn inside another
        and inherits its labels, the ``legend`` and the ``series``.
    """
    rel = path.relative_to(root).as_posix()
    directory = rel.rsplit("/", 1)[0]
    lines = join_continuations(path.read_text(errors="replace").splitlines())
    # Scripts with `# Panel N` comments are divided by those. The others are
    # divided by psbasemap calls that change the plot region.
    has_panel_comments = any(PANEL.match(line) for line in lines)

    labels: dict[str, str] = {}
    files: dict[str, str] = {}
    panels: list[dict] = []
    titles: list[dict] = []
    current: dict | None = None

    def close() -> None:
        if current is not None and (current["series"] or current["panel"]):
            panels.append(current)

    def new_panel(number: int | None) -> dict:
        return {"panel": number, "quantity": None, "series": [], "legend": []}

    if has_panel_comments:
        current = new_panel(None)

    for line in lines:
        # A commented-out line is a contributor or a curve that was dropped
        # from the figure. Worth recording, and not as a live series.
        commented = line.lstrip().startswith("#")

        m = LABEL.match(line)
        if m and not commented:
            labels[m.group(1)] = clean_label(m.group(2))
            continue

        m = ASSIGN.match(line)
        if m and not commented:
            files[m.group(1)] = m.group(2)
            continue

        m = PANEL.match(line)
        if m and has_panel_comments:
            close()
            current = new_panel(int(m.group(1)))
            continue

        # Axis labels are read from psbasemap and from psscale, which is where
        # the one map script labels its colour bar.
        if ("psbasemap" in line or "psscale" in line) and not commented:
            found = []
            for raw in AXIS_LABEL.findall(line):
                ref = LABEL_REFERENCE.match(raw)
                found.append(labels.get(ref.group(1), ref.group(1)) if ref
                             else clean_label(raw))
            region = REGION.search(line)
            region = region.group(1) if region else None
            if not has_panel_comments and "psbasemap" in line:
                if current is not None and current.get("region") is None:
                    # The panel was opened by a series drawn before the first
                    # basemap, as the map script does; this basemap belongs
                    # to it.
                    current["region"] = region
                elif current is None or region != current.get("region"):
                    previous = current
                    close()
                    current = new_panel(len(panels) + 1)
                    current["region"] = region
                    # An inset has empty labels and shows the quantity of the
                    # panel it sits in, on a smaller region.
                    if previous is not None and not any(found):
                        current["quantity"] = previous["quantity"]
                        if "abscissa_label" in previous:
                            current["abscissa_label"] = previous["abscissa_label"]
                        current["inset"] = True
            if current is not None and found and found[-1]:
                if current["quantity"] is None:
                    current["quantity"] = found[-1]
                if len(found) > 1 and found[0]:
                    current.setdefault("abscissa_label", found[0])

        m = TITLE.search(line)
        if m:
            titles.append({"text": clean_label(m.group(1)), "drawn": not commented})
            continue

        m = AWK.search(line)
        if m:
            if current is None:
                # A series drawn before any basemap, as in the map script,
                # which grids its data before drawing the frame. The panel has
                # no number, as in a script with one panel.
                current = new_panel(None)
                current["region"] = None
            var = m.group("var")
            fields = [parse_expression(f) for f in m.group("fields").split(",")]
            file_name = files.get(var)
            entry = {
                "variable": var,
                **contributor_of(var),
                "file": file_name or f"<unresolved ${var}>",
                "path": f"{directory}/{file_name}" if file_name else None,
                "x": fields[0],
                "y": fields[1] if len(fields) > 1 else None,
                "drawn": not commented,
            }
            # A cross-section or a time series prints two fields. A map prints
            # three: longitude, latitude and the value to colour by.
            if len(fields) > 2:
                entry["z"] = fields[2]
                entry["kind"] = "map"
            elif current.get("abscissa_label", "").startswith("Time"):
                entry["kind"] = "time_series"
            else:
                entry["kind"] = "cross_section"
            current["series"].append(entry)
            continue

        m = LEGEND.match(line)
        if m and not commented and current is not None:
            current["legend"].append(m.group(1))

    close()
    case = CASE_DIRECTORY.search(directory)
    return {"script": rel, "case": case.group(1) if case else None,
            "titles": titles, "panels": panels}


# ---------------------------------------------------------------------------
# Transcribed layouts. Everything below is copied from a file header or
# established by arithmetic on the files, and the evidence is recorded with it.
# ---------------------------------------------------------------------------

VEGA_SEVEN_COLUMN_FORMAT = {
    "$comment": (
        "The seven-column format VEGA writes for a cross-section, taken "
        "verbatim from the header of martinec2018/benchmarkA-Z_ZM_load-full.out, "
        "which is the only file that carries one. The same format is used by "
        "every Z_ZM_{load,basin}.out and by the Z_*_diff.out files of cases B, "
        "C and D, where column n is the contributor minus VEGA. The case A "
        "difference files and the two WW files of case B hold four or five "
        "columns, which are a subset in the same order."),
    "source": "the header of benchmarkA-Z_ZM_load-full.out",
    "columns": [
        {"column": 1, "name": "abscissa", "units": "degrees",
         "description": "Longitude in a figY_uvf cross-section, colatitude in a figZ_uvf one"},
        {"column": 2, "name": "U", "units": "m", "description": "Vertical displacement"},
        {"column": 3, "name": "V_theta", "units": "m",
         "description": "Colatitude component of horizontal displacement"},
        {"column": 4, "name": "V_phi", "units": "m",
         "description": "Longitude component of horizontal displacement. NaN at colatitude 0 and 180, where it is undefined."},
        {"column": 5, "name": "potential", "units": "SI",
         "description": "Gravitational potential increment. Divide by the surface gravity 9.8155 to obtain geoid height N, which is what the plotting scripts do. In a difference file this column is already a geoid height difference in metres."},
        {"column": 6, "name": "S", "units": "m",
         "description": "Sea-surface variation with respect to the uniform water layer h_UF"},
        {"column": 7, "name": "SLE", "units": "m", "description": "Sea level equation result"},
    ],
    "applies_to": [
        "martinec2018/benchmarkA-Z_ZM_load-full.out",
        "martinec2018/benchmarkA/Z_ZM_load.out",
        "martinec2018/benchmarkB/Z_ZM_load.out",
        "martinec2018/benchmarkB/Z_ZM_basin.out",
        "martinec2018/benchmarkB/Z_{GA,GS,KS,SK,VB,VK,YS}_{load,basin}_diff.out",
        "martinec2018/results{A,B,C,D}_2026/Z_ZM_{load,basin}.out",
        "martinec2018/results{B,C,D}_2026/Z_*_{load,basin}_diff.out",
    ],
}

# The VEGA time-series files. Their header is transcribed as it stands in
# `header`, and `name`, `units` and `si_factor` say what the column holds once
# the two header errors in correction martinec2018-vega-time-series-header are
# taken into account.
VEGA_TIME_SERIES_COLUMNS = [
    {"column": 1, "header": "time index", "name": "index", "units": "1", "si_factor": 1},
    {"column": 2, "header": "time after surface load onset (kyr)", "name": "time",
     "units": "kyr", "si_factor": 1},
    {"column": 3, "header": "mass of ice load (10**18 kg)", "name": "ice_mass_total",
     "units": "1e18 kg", "si_factor": 1e18},
    {"column": 4, "header": "ocean area (per cent of 4\\pi)", "name": "ocean_area",
     "units": "fraction of 4 pi", "si_factor": 1,
     "note": "The header says per cent. The values are a fraction: 0.04628693 at t = 0, which every script multiplies by 100."},
    {"column": 5, "header": "huf1=-rhoice*hice00/ofj00/rhowat", "name": "h_UF_1",
     "units": "m", "si_factor": 1},
    {"column": 6, "header": "huf2=-fuj00/ofj00", "name": "h_UF_2", "units": "m", "si_factor": 1},
    {"column": 7, "header": "huf3=fuj100/ofj00", "name": "h_UF_3", "units": "m", "si_factor": 1},
    {"column": 8, "header": "huf4=mass of water displayed by ice/ofj00", "name": "h_UF_4",
     "units": "m", "si_factor": 1},
    {"column": 9, "header": "col5+col6+col7+col8", "name": "h_UF", "units": "m", "si_factor": 1},
    {"column": 10, "header": "grounded ice (10**18 kg)", "name": "ice_mass_grounded",
     "units": "1e18 kg", "si_factor": 1e18},
    {"column": 11, "header": "col3-col9", "name": "ice_mass_floating",
     "units": "1e18 kg", "si_factor": 1e18,
     "note": "The header says col3-col9, which subtracts a thickness from a mass. The column equals col3-col10 to 1e-8 in every file."},
]

VEGA_TIME_SERIES_FORMAT = {
    "$comment": (
        "The eleven-column time series VEGA writes for cases C, D and E. The "
        "header is transcribed from martinec2018/resultsC_2026/vega76.out. "
        "secti.out has no header and the same layout except column 5; see "
        "correction martinec2018-secti-column5."),
    "source": "the header of resultsC_2026/vega76.out",
    "columns": VEGA_TIME_SERIES_COLUMNS,
    "applies_to": [
        "martinec2018/resultsC_2026/vega76.out",
        "martinec2018/resultsD_2026/vega76.out",
        "martinec2018/resultsD_2026/secti.out",
        "martinec2018/resultsE_2026/vega76.out",
        "martinec2018/resultsE_2026/vega76_E2.out",
    ],
    "corrections": ["martinec2018-vega-time-series-header", "martinec2018-secti-column5"],
}


def _vega_columns(**replace: dict) -> dict:
    """The VEGA time-series columns keyed by number, with some replaced."""
    out = {}
    for c in VEGA_TIME_SERIES_COLUMNS:
        if c["name"] in ("index", "time"):
            continue
        out[str(c["column"])] = {k: c[k] for k in ("name", "units", "si_factor")}
    for column, entry in replace.items():
        out[column.lstrip("c")] = entry
    return out


# What each time-series column holds, per file. The scripts say which column is
# plotted and how it is scaled, but not the unit it is stored in, and the
# contributors use every unit available: ice mass in kg, 1e17 kg and 1e18 kg,
# ocean area as a fraction, a percentage and ten times a fraction. `si_factor`
# converts the stored value to kg, to a fraction of 4 pi, or to metres. A test
# checks, for every drawn time series, that si_factor equals the script's
# scale factor times the unit on the panel's axis, so an entry here that
# disagrees with the published figure is a test failure.
_YS_OCEAN_AREA = {
    "time_column": 1,
    "columns": {
        "2": {"name": "ocean_area_initial", "units": "fraction of 4 pi", "si_factor": 1,
              "evidence": "Constant in time in case D, and equal to VEGA column 4 at t = 0 to 1e-5."},
        "3": {"name": "ocean_area", "units": "10 x fraction of 4 pi", "si_factor": 0.1,
              "evidence": "Ten times VEGA column 4: 0.44377 against 0.044378 at 15 kyr in case D. The script multiplies it by 10 to plot per cent."},
    },
}
_SBK_TIME_SERIES = {
    "time_column": 2,
    "columns": {
        "3": {"name": "ice_mass_total", "units": "1e18 kg", "si_factor": 1e18,
              "evidence": "Equal to column 10 plus column 11 exactly, and 1.184 against VEGA's 1.202 at 15 kyr in case D."},
        "4": {"name": "ocean_area", "units": "fraction of 4 pi", "si_factor": 1,
              "evidence": "0.04631 at t = 0 against VEGA's 0.04629. The script multiplies it by 100."},
        "5": {"name": "h_UF_1", "units": "m", "si_factor": 1,
              "evidence": "Same position and size as VEGA column 5: -52.29 against -53.12 at 15 kyr in case D."},
        "6": {"name": "h_UF_2", "units": "m", "si_factor": 1,
              "evidence": "7.48 against VEGA's 7.59 at 15 kyr in case D."},
        "7": {"name": "h_UF_3", "units": "m", "si_factor": 1,
              "evidence": "4.93 against VEGA's 5.03 at 15 kyr in case D."},
        "8": {"name": "h_UF_4", "units": "m", "si_factor": 1,
              "evidence": "10.13 against VEGA's 10.30 at 15 kyr in case D."},
        "9": {"name": "h_UF", "units": "m", "si_factor": 1,
              "evidence": "Equal to the sum of columns 5 to 8 to 6e-7."},
        "10": {"name": "ice_mass_floating", "units": "1e18 kg", "si_factor": 1e18,
               "evidence": "0.2295 against VEGA's floating 0.2332 at 15 kyr in case D. Columns 10 and 11 are in the opposite order to VEGA's. No script plots them."},
        "11": {"name": "ice_mass_grounded", "units": "1e18 kg", "si_factor": 1e18,
               "evidence": "0.9549 against VEGA's grounded 0.9693 at 15 kyr in case D."},
    },
}

TIME_SERIES_COLUMNS = {
    "martinec2018/resultsC_2026/vega76.out": {"time_column": 2, "columns": _vega_columns()},
    "martinec2018/resultsD_2026/vega76.out": {"time_column": 2, "columns": _vega_columns()},
    "martinec2018/resultsE_2026/vega76.out": {"time_column": 2, "columns": _vega_columns()},
    "martinec2018/resultsE_2026/vega76_E2.out": {
        "time_column": 2, "columns": _vega_columns(),
        "run_of_case": "D",
        "evidence": "fig_hufE.gmt and fig_of00E.gmt label this curve 'Benchmark D'. It has 301 rows at 0.05 kyr, where resultsD_2026/vega76.out has 751 at 0.02 kyr.",
    },
    "martinec2018/resultsD_2026/secti.out": {
        "time_column": 2,
        "columns": _vega_columns(c5={
            "name": "h_UF_1_plus_h_UF_4", "units": "m", "si_factor": 1,
            "evidence": "Equal to vega76.out column 5 plus column 8 to 7e-15 over all 751 rows; see correction martinec2018-secti-column5."}),
    },
    "martinec2018/resultsC_2026/D3_fig06_YS.dat": {
        "time_column": 1,
        "columns": {
            "2": {"name": "ice_mass_grounded", "units": "1e17 kg", "si_factor": 1e17,
                  "evidence": "7.5154 at 15 kyr against VEGA's grounded ice, column 10 times 10, 7.5104. Case C has no floating ice."},
            "3": {"name": "ice_mass_floating", "units": "1e17 kg", "si_factor": 1e17,
                  "evidence": "Zero throughout, as VEGA column 11 is in case C."},
        },
    },
    "martinec2018/resultsD_2026/E2_fig06_YS.dat": {
        "time_column": 1,
        "columns": {
            "2": {"name": "ice_mass_floating", "units": "1e17 kg", "si_factor": 1e17,
                  "evidence": "2.4198 at 15 kyr against VEGA's floating 2.3317. The order differs from the case C file of the same contributor."},
            "3": {"name": "ice_mass_grounded", "units": "1e17 kg", "si_factor": 1e17,
                  "evidence": "9.6100 at 15 kyr against VEGA's grounded 9.6928."},
            "4": {"name": "ice_mass_total", "units": "1e17 kg", "si_factor": 1e17,
                  "evidence": "Equal to column 2 plus column 3 to 1e-8."},
        },
    },
    "martinec2018/resultsD_2026/E2_fig06_1_VK.dat": {
        "time_column": 1,
        "columns": {
            "2": {"name": "ice_mass_total", "units": "1e18 kg", "si_factor": 1e18,
                  "evidence": "The header, and equal to column 3 plus column 4 to 3e-6."},
            "3": {"name": "ice_mass_grounded", "units": "1e18 kg", "si_factor": 1e18,
                  "evidence": "The header."},
            "4": {"name": "ice_mass_floating", "units": "1e18 kg", "si_factor": 1e18,
                  "evidence": "The header."},
        },
    },
    "martinec2018/resultsD_2026/GS_mass-E2.dat": {
        "time_column": 1,
        "columns": {
            "2": {"name": "ice_mass_floating", "units": "kg", "si_factor": 1,
                  "evidence": "2.550e17 at 15 kyr against VEGA's floating 2.332e17."},
            "3": {"name": "ice_mass_grounded", "units": "kg", "si_factor": 1,
                  "evidence": "9.454e17 at 15 kyr against VEGA's grounded 9.693e17."},
        },
    },
    "martinec2018/resultsD_2026/E2_fig07_1_VK.dat": {
        "time_column": 1,
        "columns": {"2": {"name": "ocean_area", "units": "per cent of 4 pi", "si_factor": 0.01,
                          "evidence": "The header, and 4.6194 at t = 0 against VEGA's 4.6287 per cent."}},
    },
    "martinec2018/resultsE_2026/F1_fig07_1_VK.dat": {
        "time_column": 1,
        "columns": {"2": {"name": "ocean_area", "units": "per cent of 4 pi", "si_factor": 0.01,
                          "evidence": "The header, m^2/5.100646e14 *100."}},
    },
    "martinec2018/resultsD_2026/E2_fig07_YS.dat": _YS_OCEAN_AREA,
    # Same contributor, same filename pattern, different unit from case D. Column
    # 3 here is a plain fraction written with a D-01 exponent. awk reads only the
    # mantissa, ten times the value, so fig_of00E.gmt's factor 10 plots the right
    # per cent by accident; `awk_exponent_dropped` records that, and the unit
    # test and the converter allow for it only when the correction applies.
    "martinec2018/resultsE_2026/F1_fig07_YS.dat": {
        "time_column": 1,
        "awk_exponent_dropped": 1,
        "columns": {
            "3": {"name": "ocean_area", "units": "fraction of 4 pi", "si_factor": 1,
                  "evidence": "0.04696 at t = 0 and 0.04521 at 15 kyr, written 0.4696086368873D-01, against VEGA's 0.04696 and 0.04520. Column 2 is not plotted and changes from 0.04696 to 0.04630, so it is not given a name here. See correction martinec2018-ys-case-e-awk-exponent."},
        },
    },
    "martinec2018/resultsD_2026/E2_fig07_SBK.dat": _SBK_TIME_SERIES,
    "martinec2018/resultsE_2026/F1_fig07_SBK.dat": _SBK_TIME_SERIES,
    "martinec2018/resultsD_2026/GS_ocean-area-E2.dat": {
        "time_column": 1,
        "columns": {"2": {"name": "ocean_area", "units": "per cent of 4 pi", "si_factor": 0.01,
                          "evidence": "4.6148 at t = 0 against VEGA's 4.6287 per cent."}},
    },
    "martinec2018/resultsD_2026/GS_hUF-E2.dat": {
        "time_column": 1,
        "columns": {"2": {"name": "h_UF", "units": "m", "si_factor": 1,
                          "evidence": "-30.12 at 15 kyr against VEGA's -30.20."}},
    },
    "martinec2018/resultsD_2026/E2_fig08_YS.dat": {
        "time_column": 1,
        "columns": {"2": {"name": "h_UF", "units": "m", "si_factor": 1,
                          "evidence": "-30.23 at 15 kyr against VEGA's -30.20."}},
    },
    "martinec2018/resultsE_2026/F1_fig08_YS.dat": {
        "time_column": 1,
        "columns": {"2": {"name": "h_UF", "units": "m", "si_factor": 1,
                          "evidence": "-30.18 at 15 kyr against VEGA's -29.86."}},
    },
    "martinec2018/resultsE_2026/F1_fig08_1.dat": {
        "time_column": 1,
        "columns": {"5": {"name": "h_UF", "units": "m", "si_factor": 1,
                          "evidence": "Equal to column 2 plus column 3 plus column 4 to 7e-15; see correction martinec2018-vk-huf-header. Columns 2 to 4 are components whose split differs from VEGA's and are not given a name here."}},
    },
}


def main() -> int:
    here = pathlib.Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=pathlib.Path, default=here / "data" / "original")
    ap.add_argument("--output", type=pathlib.Path,
                    default=here / "gia_mip" / "resources" / "column_layouts.json")
    args = ap.parse_args()
    root = args.root.resolve()

    # The archives as first received (benchmarkA/, benchmarkB/) and the set
    # Martinec sent in September 2026 (resultsA_2026/ to resultsE_2026/). Both
    # are decoded: the two case B scripts are identical in both, and the case
    # A script differs only in the name of the PDF it writes.
    scripts = (sorted(root.glob("martinec2018/benchmark*/*.gmt"))
               + sorted(root.glob("martinec2018/results?_2026/*.gmt")))
    if not scripts:
        print(f"no GMT scripts under {root}/martinec2018/. "
              f"Fetch and unpack the benchmark archives first.", file=sys.stderr)
        return 1

    decoded = [decode(p, root) for p in scripts]

    # Invert to a per-file view, which is what a reader actually asks for.
    # Keyed by path, because the same filename holds different data in
    # different directories.
    per_file: dict[str, list[dict]] = {}
    for script in decoded:
        for panel in script["panels"]:
            for series in panel["series"]:
                record = {
                    "script": script["script"],
                    "case": script["case"],
                    "panel": panel["panel"],
                    "kind": series["kind"],
                    "quantity": panel["quantity"],
                    "contributor": series["contributor"],
                    "abscissa": series["x"],
                    "ordinate": series["y"],
                    "drawn_in_the_published_figure": series["drawn"],
                }
                if "variant" in series:
                    record["variant"] = series["variant"]
                if panel.get("inset"):
                    record["inset"] = True
                if "z" in series:
                    record["value"] = series["z"]
                key = series["path"] or series["file"]
                per_file.setdefault(key, []).append(record)

    doc = {
        "$comment": (
            "What each column of each Martinec et al. (2018) data file holds, "
            "decoded from the GMT 4 scripts that drew the published figures. "
            "Generated by scripts/decode_gmt_layouts.py; do not edit by hand. "
            "Paths are relative to data/original/. For files with no header "
            "these scripts are the only surviving record, and GMT 4 no longer "
            "runs. `negated` means the script plots the negative of the "
            "column, which is how several contributors' geoid sign is "
            "reconciled. `scale` records arithmetic applied before plotting, "
            "with the column written as x. The last three sections are "
            "transcribed from file headers or established by arithmetic, and "
            "carry their evidence."),
        "scripts": decoded,
        "by_file": dict(sorted(per_file.items())),
        "vega_seven_column_format": VEGA_SEVEN_COLUMN_FORMAT,
        "vega_time_series_format": VEGA_TIME_SERIES_FORMAT,
        "time_series_columns": TIME_SERIES_COLUMNS,
    }
    args.output.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")

    n_series = sum(len(v) for v in per_file.values())
    print(f"{len(decoded)} scripts, {sum(len(s['panels']) for s in decoded)} panels, "
          f"{len(per_file)} data files, {n_series} column assignments")
    unresolved = [f for f in per_file if f.startswith("<unresolved")]
    if unresolved:
        print(f"unresolved shell variables: {unresolved}", file=sys.stderr)
    try:
        shown = args.output.relative_to(here)
    except ValueError:
        shown = args.output
    print(f"written to {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
