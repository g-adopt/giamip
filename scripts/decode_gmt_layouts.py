"""Decode what each benchmark data file contains, from the GMT scripts.

For several of the Martinec et al. (2018) data files there is no header and no
documentation. The only surviving record of which column holds which quantity
is the GMT 4 script that drew the published figure, in lines of the form::

    awk '{print $1, -$4}' $inpSK | psxy -R -JX $lineSK -K -O >> pom.ps

which says: for contributor SK, take column 1 as the abscissa and the negation
of column 4 as the ordinate, and plot it on the panel currently being drawn.
The panel's y-axis label says which quantity that is.

GMT 4 no longer runs, so this script reads the record and writes it down while
it can still be checked against the figures shipped alongside. It is a
decoder, not a plotter: nothing here draws anything.

Three things make the parse more than a regular expression. Shell variables
holding filenames are reassigned between panels, so ``$inpWW`` means
``u_10ka.dat`` in one panel and ``n_10ka.dat`` in the next, and assignments
have to be followed in source order. The awk expression may carry arithmetic,
``$5/9.8155`` to divide potential by gravity, or ``9*$2/10`` to rescale a
contributor whose result was known to be off. And the legend is written with
separate ``pstext`` calls, which is what ties a line colour to a contributor.

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

# The line that does the work. Captures the two awk expressions and the file.
AWK = re.compile(
    r"""awk\s+'\{print\s+(?P<fields>[^}]+?)\s*\}'\s+\$(?P<var>\w+)"""
)

# `psbasemap ... :"${stitx}":/a2f1:"${stit4}":WeSn`. GMT 4 writes the axis
# settings as -B<x>/<y>, so when a basemap line carries two label references
# the first is the abscissa and the second is the ordinate. The panel is named
# by its ordinate, so the LAST reference on the line is the one to take. A
# panel whose abscissa is unlabelled has one reference, and it is the ordinate.
BASEMAP_LABEL = re.compile(r':"\$\{(stit\w*)\}":')

# `# Panel 3`
PANEL = re.compile(r"^#+\s*Panel\s+(\d+)", re.IGNORECASE)

# `120  -40 10  0  0  ML ZM` inside a pstext heredoc: the legend entry.
LEGEND = re.compile(r"^\s*[-\d.]+\s+[-\d.]+\s+\d+\s+\d+\s+\d+\s+\w+\s+(\w+)\s*$")

# GMT 4 writes Greek by switching font: @~q@~ is theta, @~l@~ lambda, and
# @~D@~ capital delta. Translated so the labels read as text.
GREEK = {"@~q@~": "theta", "@~l@~": "lambda", "@~D@~": "delta ",
         "@~F@~": "Phi", "@-": "_", "@+": "^", "@~": ""}


def clean_label(raw: str) -> str:
    """Turn a GMT axis label into plain text."""
    text = raw
    for token, plain in GREEK.items():
        text = text.replace(token, plain)
    return " ".join(text.split())


def parse_expression(expr: str) -> dict:
    """Split an awk print expression into a column and what is done to it.

    ``$2`` is column 2 unchanged. ``-$4`` is column 4 negated. ``$5/9.8155``
    and ``9*$2/10`` carry a scale factor. Anything with more than one column
    reference is recorded verbatim and left for a person to read.
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


def decode(path: pathlib.Path) -> dict:
    """Decode one GMT script into its panels and the series drawn on each."""
    labels: dict[str, str] = {}
    files: dict[str, str] = {}
    panels: list[dict] = []
    current = {"panel": None, "quantity": None, "series": [], "legend": []}
    pending_label: str | None = None

    for line in path.read_text().splitlines():
        # A commented-out line is a contributor that was dropped from the
        # figure. Worth recording, and not as a live series.
        commented = line.lstrip().startswith("#")

        m = LABEL.match(line)
        if m:
            labels[m.group(1)] = clean_label(m.group(2))
            continue

        m = ASSIGN.match(line)
        if m and not commented:
            files[m.group(1)] = m.group(2)
            continue

        m = PANEL.match(line)
        if m:
            if current["series"] or current["panel"]:
                panels.append(current)
            current = {"panel": int(m.group(1)), "quantity": None,
                       "series": [], "legend": []}
            continue

        found = BASEMAP_LABEL.findall(line)
        if found:
            pending_label = labels.get(found[-1], found[-1])
            if current["quantity"] is None:
                current["quantity"] = pending_label
            if len(found) > 1:
                current["abscissa_label"] = labels.get(found[0], found[0])

        m = AWK.search(line)
        if m:
            var = m.group("var")
            fields = [parse_expression(f) for f in m.group("fields").split(",")]
            entry = {
                "variable": var,
                "file": files.get(var, f"<unresolved ${var}>"),
                "x": fields[0],
                "y": fields[1] if len(fields) > 1 else None,
                "drawn": not commented,
            }
            # A cross-section prints two fields. A map prints three: longitude,
            # latitude and the value to colour by.
            if len(fields) > 2:
                entry["z"] = fields[2]
                entry["kind"] = "map"
            else:
                entry["kind"] = "cross_section"
            current["series"].append(entry)
            continue

        m = LEGEND.match(line)
        if m and not commented:
            current["legend"].append(m.group(1))

    if current["series"] or current["panel"]:
        panels.append(current)
    return {"script": path.name, "panels": panels}


def main() -> int:
    here = pathlib.Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=pathlib.Path, default=here / "data" / "original")
    ap.add_argument("--output", type=pathlib.Path,
                    default=here / "gia_mip" / "resources" / "column_layouts.json")
    args = ap.parse_args()

    scripts = sorted(args.root.glob("martinec2018/benchmark*/*.gmt"))
    if not scripts:
        print(f"no GMT scripts under {args.root}/martinec2018/. "
              f"Fetch and unpack the benchmark archives first.", file=sys.stderr)
        return 1

    decoded = [decode(p) for p in scripts]

    # Invert to a per-file view, which is what a reader actually asks for.
    per_file: dict[str, list[dict]] = {}
    for script in decoded:
        for panel in script["panels"]:
            for series in panel["series"]:
                record = {
                    "script": script["script"],
                    "panel": panel["panel"],
                    "kind": series["kind"],
                    "quantity": panel["quantity"],
                    "abscissa": series["x"],
                    "ordinate": series["y"],
                    "drawn_in_the_published_figure": series["drawn"],
                }
                if "z" in series:
                    record["value"] = series["z"]
                per_file.setdefault(series["file"], []).append(record)

    doc = {
        "$comment": (
            "What each column of each Martinec et al. (2018) data file holds, "
            "decoded from the GMT 4 scripts that drew the published figures. "
            "Generated by scripts/decode_gmt_layouts.py. For files with no "
            "header these scripts are the only surviving record, and GMT 4 no "
            "longer runs. `negated` means the script plots the negative of the "
            "column, which is how several contributors' geoid sign is "
            "reconciled. `scale` records arithmetic applied before plotting."),
        "scripts": decoded,
        "by_file": dict(sorted(per_file.items())),
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
