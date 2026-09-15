"""Decode what each Spada et al. (2011) data file contains, from the GMT scripts.

This is the companion to ``decode_gmt_layouts.py``, which does the same job for
the Martinec et al. (2018) archive. The two archives need two decoders because
the scripts are written in two different styles, and a single parser for both
would be harder to read than either.

The Martinec scripts assign filenames to shell variables and select columns
with ``awk``::

    inpSK=A_fig10_SBK.dat
    awk '{print $1, -$4}' $inpSK | psxy ...

The Spada scripts name the file on the command line, strip the header with
``sed`` into a temporary file, and let GMT take its default columns::

    psbasemap -R1/300/1e1/1e8 -Ba10f0.1:'Degree, n':/f1a1p:'Relaxation time (yrs)':WSen \\
        -JX9l/12l -K > spec_GS.ps
    sed '/#/ d' spec_T01_01_GS.mod > data.tmp
    psxy -M data.tmp -B -R -JX -Sc0.08 -G0 -O -K >> spec_GS.ps

Reading that requires four things that a regular expression alone does not give.

**The temporary file has to be followed back to the real one.** ``data.tmp`` is
written and overwritten many times in one script, so the mapping has to be
resolved in source order: ``data.tmp`` means one contributor's file above a
``psxy`` line and a different contributor's file below it. Fourteen of the
eighty-seven scripts add an ``awk`` stage on top, so the chain can be two steps
long and can carry a column selection and a scale factor.

**The columns are implicit.** With no ``awk``, GMT 4 ``psxy`` reads column 1 as
the abscissa and column 2 as the ordinate. That default is the layout for most
of these files, and it is recorded here explicitly so that no reader has to
know it.

**The panel is named by its output PostScript file**, and not by a comment.
A ``psbasemap`` line with ``>`` opens a panel; every later ``psxy`` with ``>>``
to the same file draws on it. The axis labels on the ``psbasemap`` line say
which quantity the panel holds.

**Two series on one panel can differ only by symbol.** In test 1/2 the elastic
and the fluid Love numbers are drawn on the same axes from a ``.ela`` and a
``.flu`` file, told apart by ``-Sc`` circles against ``-Ss`` squares. The
symbol is therefore part of the record, not decoration.

GMT 4 no longer runs, so for the files with no header these scripts are the
only surviving description of their contents.

The output has two views of the same archive. by_file is what each script
drew, keyed by the name the script wrote. files is what a reader uses,
keyed by the path under data/original/: for every file the converter
gia_mip.benchmarks.spada2011 reads, what each column holds, its unit and
its epoch, and the evidence for it. For the files no script describes (the
VILMA files, the signed Love number tables, the point load profiles, the
transfer functions of test 4/1 and the polar motion tables of test 3/2) that
evidence is the README files, the file headers, the Fortran programs that
wrote the plotted files, and arithmetic on the values, written out in
file_layouts.

Usage::

    python3 scripts/decode_spada_layouts.py [--root data/original] \\
        [--output gia_mip/resources/spada_column_layouts.json]
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys

import numpy as np

# The repository root goes first on sys.path, as in scripts/convert_spada2011.py,
# so this script uses the package of the checkout it sits in, also in a worktree.
# IN_M_PER_KYR, how many m/kyr one stored rate unit is, has its one definition
# in the converter. The rate unit check at the end of the `files` table uses it.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from gia_mip.benchmarks.spada2011 import IN_M_PER_KYR  # noqa: E402

# The eight contributor abbreviations of the paper's table 1, as recorded in
# the `reference_data.contributors` field of the ten case files, plus the one
# name the scripts use that table 1 does not. A filename is attributed by
# finding one of these as a whole token, which is why the longer codes are
# tested before the shorter ones: `GSA` has to win over `GS` in
# `h_L_T01-06_GSA.dat`, and `ALMA` over `BL` in `h-alma-deg1.dat`.
CONTRIBUTORS = ["GSA", "ALMA", "SBK", "BL", "GS", "PG", "RR", "VB", "VK", "ZM"]

# `ALMA` is the name of a program, and table 1 has no such contributor. Two
# independent pieces of evidence say it is GSA's series, and both are in the
# archive rather than assumed:
#
#   - Test 1/5 lists GS, GSA and VK as its three contributors. Its script
#     reads one file per contributor and names the three temporaries
#     `data_GS.tmp`, `data_VK.tmp` and `data_ALMA.tmp`, so the third is GSA.
#   - Test 1/7 keeps one directory per contributor, and `h-alma-deg1.dat`
#     and `l-alma-deg1.dat` are inside `./Gsa/`.
#
# The two names are kept apart in the search and joined here, so the mapping
# is a statement that can be checked instead of a spelling that hides it.
ALIASES = {"ALMA": "GSA"}

# `sed '/#/ d' spec_T01_01_GS.mod > data.tmp`. The sed expression itself is
# always a header strip in these scripts, and is captured only so that a
# different expression would be visible in the output instead of being applied
# silently.
SED = re.compile(r"\bsed\s+'([^']*)'\s+(?P<src>[^\s>]+)\s*>\s*(?P<dst>\S+)")

# `awk '{print $1, $2*1000}' data_VK_e.tmp > data_VK_e_0.tmp`.
AWK = re.compile(
    r"\bawk\s+'\{print\s+(?P<fields>[^}]+?)\s*\}'\s+(?P<src>[^\s>|]+)"
    r"(?:\s*>\s*(?P<dst>\S+))?"
)

# `psbasemap -X5 -Y5 -R1/300/1e1/1e8 -B... -JX9l/12l -K > spec_GS.ps`.
#
# Every psbasemap call draws one panel, and the redirection says whether the
# PostScript file is being created or added to. Tests 2/1, 2/2 and 2/3 put six
# panels in one file, the first opened with `>` and the rest appended with
# `>>`, offset by `-X` and `-Y`. Matching the single `>` alone would collapse
# those six panels into one and lose which quantity each holds.
PSBASEMAP = re.compile(r"\bpsbasemap\b(?P<flags>.*?)>>?\s*(?P<out>\S+\.ps)")

# `psxy -M data.tmp -B -R -JX -Sc0.08 -G0 -O -K >> spec_GS.ps`.
PSXY = re.compile(r"\bpsxy\b(?P<flags>.*?)>>\s*(?P<out>\S+\.ps)")

# `-R1/300/1e1/1e8`, the data range of the panel.
RANGE = re.compile(r"-R([\d.eE+/-]+)")

# `-JX9l/12l`: a linear projection whose width and height carry an `l` when
# that axis is logarithmic. Getting this wrong turns a spectrum into a
# straight line, so it is recorded per panel.
PROJECTION = re.compile(r"-JX([^\s]+)")

# The axis labels inside `-B`, each written as `:'Degree, n':`. GMT 4 writes
# the settings as -B<x>/<y>, so the first label on the line is the abscissa
# and the second is the ordinate. This is the same ordering rule the Martinec
# decoder applies, and the same one that produces a wrong answer if reversed.
AXIS_LABEL = re.compile(r":'([^']*)':|:\"([^\"]*)\":")

# `-Sc0.08` circles, `-Ss0.08` squares, `-St0.1` triangles. Captured whole.
SYMBOL = re.compile(r"-S(\w[\w.]*)")

# `-W5/255/0/0` and friends: a pen, which means the series is drawn as a line
# and not as symbols.
PEN = re.compile(r"-W([\w/.]+)")

# A legend entry inside a `pstext` heredoc:
# `1.35 -38 16 0 0 BL RR`, which is x, y, size, angle, font, justification and
# then the text. The same shape as the Martinec legend lines.
LEGEND = re.compile(r"^\s*[-\d.eE+]+\s+[-\d.eE+]+\s+\d+\s+\d+\s+\d+\s+\w+\s+(.+?)\s*$")

# `pstext ... <<END >> spec_GS.ps` opens a heredoc whose terminator is END.
HEREDOC = re.compile(r"<<\s*(\w+)")

# `echo '17 -200 10 0 2 LM Gs' | pstext -N -R -JX -O -K >> disp_PAP.ps`. Test
# 2/1 writes its legend this way instead of with a heredoc, so a parser that
# reads heredocs only loses every contributor label in that test.
ECHO_LEGEND = re.compile(r"\becho\s+'([^']*)'\s*\|\s*pstext\b.*?>>\s*(\S+\.ps)")

# A banner comment, which these scripts use as a section heading:
# `##############################  Vertical displacement u u u u u u`
# or `# ====================` followed by `# GS GS GS GS GS GS GS`.
# The text carries the quantity for panels whose y-axis label is blank.
BANNER = re.compile(r"^#+\s*(.*\S)\s*$")


# GMT 4 writes Greek by switching to the Symbol font, in which the roman
# letter stands for the Greek one at the same position: `@~q@~` is theta and
# `@~t@~` is tau. `@-` and `@+` open a subscript and a superscript.
#
# These two are the only escapes the eighty-seven scripts use. Tau is listed
# because the axis label of test 1/1 is the relaxation time tau_j, and the
# trailing rule that strips an unrecognised font switch would leave a bare
# `t`, which reads as time and means something else. The Martinec decoder
# carries its own table, for the letters its four scripts use.
GREEK = {"@~q@~": "theta", "@~t@~": "tau", "@~l@~": "lambda",
         "@~D@~": "delta ", "@~F@~": "Phi", "@-": "_", "@+": "^", "@~": ""}


def clean_label(raw: str) -> str:
    """Turn a GMT axis label into plain text."""
    text = raw
    for token, plain in GREEK.items():
        text = text.replace(token, plain)
    return " ".join(text.split())


def join_continuations(text: str) -> list[str]:
    """Join lines that a trailing backslash continues.

    Every `psbasemap` line in these scripts is split across two source lines,
    with the `-B` labels on the second. Parsing line by line would find a
    panel with no quantity for all eighty-seven scripts.
    """
    joined: list[str] = []
    buffer = ""
    for line in text.splitlines():
        if line.rstrip().endswith("\\"):
            buffer += line.rstrip()[:-1] + " "
            continue
        joined.append(buffer + line)
        buffer = ""
    if buffer:
        joined.append(buffer)
    return joined


def parse_expression(expr: str) -> dict:
    """Split one awk print expression into a column and what is done to it.

    ``$1`` is column 1 unchanged. ``-$4`` is column 4 negated. ``$2*1000``
    carries a scale factor, which in test 1/6 converts metres to millimetres.
    An expression naming no column or more than one is recorded verbatim for a
    person to read, because guessing at it is how a curve ends up plausible
    and wrong.
    """
    columns = [int(c) for c in re.findall(r"\$(\d+)", expr)]
    if len(columns) != 1:
        return {"expression": expr.strip(), "columns": columns,
                "note": "more than one column, or none; read the expression"}
    column = columns[0]
    body = expr.strip()
    negated = body.startswith("-")
    stripped = body.lstrip("-")
    scale = None if stripped == f"${column}" else stripped.replace(f"${column}", "x")
    return {"column": column, "negated": negated, "scale": scale,
            "expression": body}


def clean_filename(name: str) -> str:
    """Strip the quotes a few scripts put around a path.

    Test 1/7 writes `psxy "./Gs/h_time_domain.dat"`, and the quotes would
    otherwise become part of the filename recorded for that file, so the same
    data file would appear twice in the per-file index under two spellings.
    """
    return name.strip().strip("\"'")


def code_in(text: str) -> str | None:
    """Find a contributor abbreviation as a whole token inside `text`.

    The abbreviation is bounded by a separator or by the end of the string, so
    a plain substring test would attribute `h_L_T01-06_GSA.dat` to GS. The
    longer codes are therefore tried first. The match ignores case because the
    archive writes the same contributor as `GS`, `Gs` and `gs` in different
    places, for example the directory `./Gs/` and the file `h_Rr.dat`.
    """
    for code in CONTRIBUTORS:
        if re.search(rf"(?<![A-Za-z]){code}(?![A-Za-z])", text, re.IGNORECASE):
            return code
    return None


def contributor_of(name: str,
                   chain: list[str] | None = None) -> tuple[str | None, str | None]:
    """Name the contributor whose result a file holds, from its path.

    The archive names most files `spec_T01_01_GS.mod`, `h_L_T01-02_RR.ela` and
    so on. Test 1/7 instead puts each contributor in a directory of their own,
    `./Gs/h_time_domain.dat`, so the directories are tested when the filename
    carries no code. Test 1/5 names neither: its file is `h.dat`, and the only
    record of who produced it is the temporary the script copies it into,
    `data_ALMA.tmp`. So the temporaries are tested last.

    The order is filename, then directory, then the temporaries the data
    passed through, because a file named for its contributor keeps that
    contributor wherever it is read. Returns the code and which of the three
    answered, so that an attribution can be checked instead of trusted.
    """
    path = pathlib.PurePosixPath(clean_filename(name))
    candidates = [("filename", path.name)]
    candidates += [("directory", part) for part in reversed(path.parent.parts)]
    candidates += [("temporary file", temp) for temp in (chain or [])]
    for source, candidate in candidates:
        found = code_in(candidate)
        if found is not None:
            return ALIASES.get(found, found), source
    return None, None


def banner_contributor(comment: str) -> str | None:
    """Read a contributor out of a banner comment, if it names one twice.

    Test 2/1 names its files `disp_t0_cap.dat` and `./output_sec2parab/
    u2cost.dat`, neither of which carries a contributor code. The only record
    of who produced them is the comment above the `sed` line that reads them::

        # Removing the HEADER from VK files (starting with "#") VK VK VK VK VK

    A code repeated is a banner and is reliable. A code appearing once is
    ordinary prose, for example `# Plotting the curves for GS`, which can
    follow a different contributor's block, so a single mention is refused.
    """
    for code in CONTRIBUTORS:
        # The lookarounds match no characters, so two abbreviations
        # separated by a single space are both counted. A pattern that
        # consumed the separator would count `GS GS` once and refuse a real
        # banner.
        hits = re.findall(rf"(?<![A-Za-z]){code}(?![A-Za-z])", comment,
                          re.IGNORECASE)
        if len(hits) >= 2:
            return ALIASES.get(code, code)
    return None


class Provenance:
    """Where the data reaching a GMT command actually came from.

    A script writes temporary files and overwrites them, so this has to be
    resolved as the script is read and never afterwards. `record_sed` and
    `record_awk` add a step; `resolve` walks the chain back to a file that
    nothing in the script produced, which is the delivered data file.
    """

    def __init__(self) -> None:
        # destination filename -> {"source": str, "x": dict|None, "y": dict|None}
        self._steps: dict[str, dict] = {}

    def record_sed(self, source: str, destination: str, expression: str) -> None:
        """A header strip. Columns are untouched."""
        self._steps[destination] = {"source": source, "x": None, "y": None,
                                    "via": f"sed '{expression}'"}

    def record_awk(self, source: str, destination: str, fields: list[dict]) -> None:
        """A column selection, and possibly a scale factor or a negation."""
        self._steps[destination] = {
            "source": source,
            "x": fields[0] if fields else None,
            "y": fields[1] if len(fields) > 1 else None,
            "via": "awk",
        }

    def resolve(self, name: str) -> dict:
        """Follow `name` back to the delivered file and the columns taken.

        Returns the original filename, the list of temporary files passed
        through, and the abscissa and ordinate. When no `awk` stage set the
        columns, the GMT 4 `psxy` default applies: column 1 is the abscissa
        and column 2 is the ordinate.
        """
        chain: list[str] = []
        x = y = None
        current = name
        # A script that wrote a temporary file from itself would loop here.
        # None do, and the guard keeps a malformed script from hanging.
        for _ in range(16):
            step = self._steps.get(current)
            if step is None:
                break
            chain.append(current)
            # The first column selection encountered walking back from the
            # plot command is the one that reached GMT. An earlier stage that
            # also selected columns selected them from a different file, so it
            # must not overwrite what a later stage chose.
            if x is None and step["x"] is not None:
                x, y = step["x"], step["y"]
            current = step["source"]
        if x is None:
            x = {"column": 1, "negated": False, "scale": None,
                 "expression": "psxy default: first column"}
            y = {"column": 2, "negated": False, "scale": None,
                 "expression": "psxy default: second column"}
        return {"file": current, "via": chain, "abscissa": x, "ordinate": y}


def first_input(flags: str) -> str | None:
    """The filename a GMT command reads, which is its first non-flag argument.

    `psxy -M data.tmp -B -R -JX -Sc0.08` puts `-M` before the file, so taking
    the first token after the command name gives a flag. GMT flags all start
    with a hyphen, and these scripts never quote a filename.
    """
    for token in flags.split():
        if not token.startswith("-") and not token.startswith("<<"):
            return token
    return None


def decode(path: pathlib.Path, test: str, variant: str) -> dict:
    """Decode one Spada GMT script into its panels and the series on each."""
    provenance = Provenance()
    # Every panel in source order, and the panel a later `psxy` draws on for
    # each PostScript file, which is always the most recently opened one.
    panels: list[dict] = []
    current: dict[str, dict] = {}
    heredoc_terminator: str | None = None
    heredoc_target: str | None = None
    # The contributor named by the most recent banner comment. Test 2/1 and
    # test 2/2 name their data files for the quantity and the load shape only,
    # so the banner is the sole record of who produced them.
    banner_owner: str | None = None
    # Banner comments seen since the last `psbasemap`. Several panels carry a
    # blank y-axis label, because the published figure labels a row of panels
    # once, so these headings are what says which quantity the panel holds.
    sections: list[str] = []

    for line in join_continuations(path.read_text(errors="replace")):
        # Inside a pstext heredoc every line is legend text until the
        # terminator, so this test comes before the comment test: a legend
        # entry may itself begin with a hash.
        if heredoc_terminator is not None:
            if line.strip() == heredoc_terminator:
                heredoc_terminator = None
                heredoc_target = None
                continue
            match = LEGEND.match(line)
            if match and heredoc_target in current:
                current[heredoc_target]["legend"].append(clean_label(match.group(1)))
            continue

        stripped = line.lstrip()
        commented = stripped.startswith("#")
        # A commented-out plot command is a contributor dropped from the
        # figure. It is recorded with `drawn` false, because the file is still
        # part of the archive and its layout is still worth having.
        body = stripped.lstrip("# ") if commented else line

        if commented:
            match = BANNER.match(stripped)
            if match:
                text = match.group(1)
                owner = banner_contributor(text)
                if owner is not None:
                    banner_owner = owner
                # A row of `=` or `#` separates banners and carries no words.
                if re.search(r"[A-Za-z]", text):
                    sections.append(clean_label(text))

        match = ECHO_LEGEND.search(body)
        if match and not commented:
            entry = LEGEND.match(match.group(1))
            target = match.group(2)
            if entry and target in current:
                current[target]["legend"].append(clean_label(entry.group(1)))
            continue

        match = SED.search(body)
        if match:
            provenance.record_sed(clean_filename(match.group("src")),
                                  clean_filename(match.group("dst")),
                                  match.group(1))
            continue

        match = AWK.search(body)
        if match and match.group("dst"):
            fields = [parse_expression(f) for f in match.group("fields").split(",")]
            provenance.record_awk(clean_filename(match.group("src")),
                                  clean_filename(match.group("dst")), fields)
            continue

        match = PSBASEMAP.search(body)
        if match and not commented:
            out = match.group("out")
            flags = match.group("flags")
            # A label of `' '` is a deliberate blank: the published figure
            # labels one panel of a row and leaves its neighbours bare. An
            # empty string is kept out of the record so that a missing label
            # reads as missing and not as a quantity named by a space.
            labels = [clean_label(a or b) for a, b in AXIS_LABEL.findall(flags)]
            projection = PROJECTION.search(flags)
            axes = projection.group(1) if projection else ""
            width, _, height = axes.partition("/")
            range_match = RANGE.search(flags)
            panel = {
                "output": out,
                # Which panel of that PostScript file this is, counting from
                # one in source order.
                "index": sum(1 for p in panels if p["output"] == out) + 1,
                # The first label is the abscissa and the second the ordinate,
                # because GMT 4 writes the axis settings as -B<x>/<y>. Reading
                # them the other way round names half the panels by their
                # colatitude axis, which is the mistake the Martinec decoder
                # has a test against.
                "abscissa_label": labels[0] if labels and labels[0] else None,
                "quantity": labels[1] if len(labels) > 1 and labels[1] else None,
                "abscissa_log": width.endswith("l"),
                "ordinate_log": height.endswith("l"),
                "range": range_match.group(1) if range_match else None,
                # What the banner comments above this panel said. Read this
                # when `quantity` is null.
                "sections": sections[-4:],
                "series": [],
                "legend": [],
            }
            panels.append(panel)
            current[out] = panel
            sections = []
            # A new panel clears the contributor the last banner named. The
            # combined panels, which re-read every contributor's file after
            # the individual panels are drawn, would otherwise all carry the
            # name of whoever happened to be drawn last.
            banner_owner = None
            continue

        match = PSXY.search(body)
        if match:
            out = match.group("out")
            flags = match.group("flags")
            name = first_input(flags)
            if name is None:
                # `psxy` with no filename reads standard input, which in these
                # scripts is a heredoc of a few hand-typed points, for example
                # an axis marker. Nothing to attribute to a data file.
                continue
            resolved = provenance.resolve(clean_filename(name))
            symbol = SYMBOL.search(flags)
            pen = PEN.search(flags)
            if out not in current:
                # A `psxy` before any `psbasemap` for this file. None of the
                # eighty-seven scripts does this, and the placeholder keeps a
                # script that did from losing its series without a word.
                placeholder = {
                    "output": out, "index": 1, "abscissa_label": None,
                    "quantity": None, "abscissa_log": False,
                    "ordinate_log": False, "range": None,
                    "sections": sections[-4:], "series": [], "legend": [],
                }
                panels.append(placeholder)
                current[out] = placeholder
            # The path is asked first and the banner comment second. A name
            # written by the script for this particular file is stronger
            # evidence than the block the read happened to sit in: test 1/6
            # copies `h.mod` into `data_ALMA.tmp` inside a block whose banner
            # names a different contributor, and the temporary is right.
            # `contributor_source` records which answered.
            from_name, name_source = contributor_of(resolved["file"],
                                                     resolved["via"])
            if from_name is not None:
                contributor, source = from_name, name_source
            elif banner_owner is not None:
                contributor, source = banner_owner, "banner comment"
            else:
                contributor, source = None, None
            current[out]["series"].append({
                "file": resolved["file"],
                "via": resolved["via"],
                "contributor": contributor,
                "contributor_source": source,
                # What the banner comment above this block said. It is not
                # used to attribute the series when the path already answers,
                # and it is kept so that the two can be compared: a file whose
                # path names one contributor inside a block that names another
                # is either a mistake in this decoder or a mistake in the
                # archive, and either way it has to be looked at.
                "banner_contributor": banner_owner,
                "abscissa": resolved["abscissa"],
                "ordinate": resolved["ordinate"],
                # `-M` says the file holds more than one segment, separated by
                # a marker line. The relaxation spectra use this: one segment
                # per harmonic degree.
                "multisegment": " -M" in flags,
                "symbol": symbol.group(0) if symbol else None,
                "pen": pen.group(0) if pen else None,
                "drawn": not commented,
            })
            continue

        if not commented:
            match = HEREDOC.search(body)
            if match and "pstext" in body:
                heredoc_terminator = match.group(1)
                target = re.search(r">>\s*(\S+\.ps)", body)
                heredoc_target = target.group(1) if target else None

    return {
        "script": f"{test}/{variant}/{path.name}" if variant else f"{test}/{path.name}",
        "test": test,
        "variant": variant,
        "panels": panels,
    }


# ------------------------------------------------------------------------
# The per-file layouts a reader uses
# ------------------------------------------------------------------------
#
# `by_file` above answers "what did a script draw from this name". A reader
# needs a different answer: for one file, keyed by its path under
# data/original/, what every column holds, in which unit, and at which epoch.
# The scripts give that only partly. The Spada scripts plot GS's rate column
# 2 without saying its unit, they never draw the signed Love number tables or
# the point load, and the VILMA files are described only by README files and
# headers. So this part reads three kinds of evidence, all in the archive:
# the drawn series of the published scripts, the README files, and the file
# headers. Where a value is derived by arithmetic on the data, such as the
# sign convention of a residue, the arithmetic is written here and is the
# evidence.

# The path of the archive under data/original/, the key of every table.
ARCHIVE = "spada2011/spada2011-cost-es0701"

# The two scripts that drew the published figures of tests 1/2 and 2/2. The
# POSTER and test scripts read the same files; the PAP script is the one the
# paper printed, so its choice of files is the one a reader follows.
PUBLISHED_SPATIAL_SCRIPTS = {"T02-01/plot_PAP.gmt": "t02-01",
                             "T02-02/plot_PAP.gmt": "t02-02"}

# TABOO names its profile columns in a header line. Column 4 is the
# longitudinal component, zero along a meridian of an axisymmetric load, and
# is not the geoid (correction spada-t02-time-layout-differs-by-contributor).
TABOO_COLUMNS = {
    "Colat.": "colatitude",
    "u_rad": "U", "u_col": "V", "u_lon": "longitudinal_displacement", "geoid": "N",
    "d_rad": "U_rate", "d_col": "V_rate", "d_lon": "longitudinal_displacement_rate",
    "d_geoid": "N_rate",
}

# The load line of a TABOO header and of a VILMA header, as a load name of
# loads/spada2011.json.
TABOO_LOADS = {"PARABOLIC LOAD": "cap", "DISC LOAD": "disc", "POINT MASS": "point"}
VILMA_LOADS = {"parabolic load": "cap", "disc load": "disc"}

# The VILMA file stems, from the README: `[uve]2cost.dat` displacements and
# `[uev]dt2cost` rates, u vertical, v horizontal, e geoid.
VILMA_STEMS = {"u": "U", "v": "V", "e": "N", "udt": "U_rate", "vdt": "V_rate", "edt": "N_rate"}
# The words the VILMA header line uses for each quantity, checked so that a
# file renamed by hand cannot take the layout of another.
VILMA_WORDS = {"U": "vertical displacement", "V": "horizontal displacement",
               "N": "geoid displacement"}

# Test 1/1 prints nine modes per degree. A Love number record with twelve
# residues carries the three Maxwell modes at residue positions 6, 9 and 12
# (from one), the TABOO order (correction spada-t01-gs-twelve-residues).
MAXWELL_POSITIONS = (6, 9, 12)


def _header_lines(path: pathlib.Path) -> list[str]:
    """The comment lines of a data file, with the leading `#` removed."""
    out = []
    for line in path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            out.append(stripped.lstrip("#").strip())
    return out


def _records(path: pathlib.Path) -> list[list[float]]:
    """The data records of a Love number or spectrum table.

    A record starts at a line whose first token is an integer, the harmonic
    degree, and runs to the next such line. VB wraps each record of twelve
    values onto two lines (correction spada-t01-vb-residue-sign-and-wrap), so
    records are joined across lines here. Lines starting with `#` or with the
    GMT segment marker `>` are skipped, and Fortran D exponents are read.
    """
    records: list[list[float]] = []
    for line in path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped[0] in "#>":
            continue
        tokens = stripped.split()
        values = [float(t.replace("D", "E").replace("d", "e")) for t in tokens]
        if re.fullmatch(r"\d+", tokens[0]):
            records.append(values)
        elif records:
            records[-1].extend(values)
        else:
            raise ValueError(f"{path}: a continuation line before any record")
    return records


def _lines_per_record(path: pathlib.Path) -> int:
    """How many lines the first record of a table occupies."""
    count = 0
    for line in path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped[0] in "#>":
            continue
        if re.fullmatch(r"\d+", stripped.split()[0]) and count:
            return count
        count += 1
    return count


def _mode_order(records: list[list[float]], path: pathlib.Path) -> str:
    """`slowest_first` or `fastest_first`, from the frequencies of every row.

    s_j is negative, and the slowest mode is the one closest to zero. A row
    whose values strictly decrease lists the slowest mode first. A file whose
    rows do not all run the same way has no order a reader can rely on, and
    raises.
    """
    orders = set()
    for record in records:
        s = record[1:]
        if all(a > b for a, b in zip(s, s[1:])):
            orders.add("slowest_first")
        elif all(a < b for a, b in zip(s, s[1:])):
            orders.add("fastest_first")
        else:
            orders.add("none")
    if len(orders) != 1 or "none" in orders:
        raise ValueError(f"{path}: the modes are not in one monotonic order ({orders})")
    return orders.pop()


def _spectrum_layout(archive: pathlib.Path, path: pathlib.Path) -> dict:
    """The layout of one test 1/1 relaxation spectrum, checked against its .mod file.

    The .mod file of the same contributor holds the relaxation times tau_j in
    years (correction spada-t01-01-mod-relaxation-times-in-years). The unit of
    the .dat file is established by the arithmetic s_j = -1000 / tau_j at
    degree 2, which must hold to the eight printed digits.
    """
    records = _records(path)
    widths = {len(r) for r in records}
    if widths != {10}:
        raise ValueError(f"{path}: expected a degree and nine frequencies, found {widths}")
    order = _mode_order(records, path)
    tau = [r for r in _records(path.with_suffix(".mod")) if int(r[0]) == 2]
    s_from_tau = sorted(-1000.0 / r[1] for r in tau)
    s_printed = sorted(next(r for r in records if int(r[0]) == 2)[1:])
    if len(s_from_tau) != 9 or any(abs(a - b) > 1e-6 * abs(b)
                                   for a, b in zip(s_from_tau, s_printed)):
        raise ValueError(f"{path}: s_j is not -1000 / tau_j of the .mod file at degree 2")
    relative = str(path.relative_to(archive))
    contributor, _ = contributor_of(path.name)
    columns = {"1": {"quantity": "degree", "units": "1"}}
    for j in range(1, 10):
        columns[str(j + 1)] = {"quantity": "s", "mode": j, "units": "1/kyr"}
    return {
        "contributor": contributor,
        "kind": "relaxation_spectrum",
        "cases": ["t01-01"],
        "evidence": (f"{relative}: ten values per row; s_j = -1000 / tau_j of "
                     f"{path.with_suffix('.mod').name} (tau in years) at degree 2; "
                     f"the frequencies of every row are monotonic"),
        "record_values": 10,
        "lines_per_record": 1,
        "mode_order": order,
        "columns": columns,
    }


def _love_layout(archive: pathlib.Path, path: pathlib.Path, spectrum: tuple) -> dict:
    """The layout of one signed Love number table of test 2/1 or 3/1.

    ``spectrum`` is the tuple the caller builds from the same contributor's
    ``spec_T01_01_*.dat``: its mode order, its file name, and its frequencies
    s_j at degree 2 in 1/kyr.

    The record is degree, elastic, fluid and then nine or twelve residues.
    Twelve means the three Maxwell modes are included, at residue positions 6,
    9 and 12. The residues pair in order with the frequencies of the same
    contributor's spectrum, and their sign is found from the fluid limit of
    the normal mode sum at degree 2,

        h_f = h_e - sum_j r_j / s_j        (Spada et al. 2011, eq. 17),

    tried with the printed sign and with the opposite sign. Exactly one must
    hold to 1e-5 of max(1, |h_f|), else the file is not understood and this
    raises.
    """
    records = _records(path)
    widths = {len(r) for r in records}
    if len(widths) != 1 or widths.pop() not in (12, 15):
        raise ValueError(f"{path}: records of {sorted({len(r) for r in records})} values")
    width = len(records[0])
    residues = width - 3
    maxwell = MAXWELL_POSITIONS if residues == 12 else ()
    kept = [i for i in range(1, residues + 1) if i not in maxwell]
    # Item 2 of `spectrum`: the frequencies s_j at degree 2, in 1/kyr, in the
    # order the contributor's spec_T01_01_*.dat file lists them.
    frequencies = spectrum[2]
    row = next(r for r in records if int(r[0]) == 2)
    r = [row[2 + i] for i in kept]
    signs = []
    for sign in (1, -1):
        fluid = row[1] - sum(sign * rj / sj for rj, sj in zip(r, frequencies))
        if abs(fluid - row[2]) <= 1e-5 * max(1.0, abs(row[2])):
            signs.append(sign)
    if len(signs) != 1:
        raise ValueError(f"{path}: the fluid limit at degree 2 holds for signs {signs}")
    relative = str(path.relative_to(archive))
    contributor, _ = contributor_of(path.name)
    quantity, kind = path.name.split("_")[:2]
    columns = {"1": {"quantity": "degree", "units": "1"},
               "2": {"quantity": f"{quantity}_elastic", "units": "1"},
               "3": {"quantity": f"{quantity}_fluid", "units": "1"}}
    mode = 0
    for i in range(1, residues + 1):
        if i in maxwell:
            columns[str(3 + i)] = {"quantity": "maxwell_residue", "units": "1/kyr"}
        else:
            mode += 1
            columns[str(3 + i)] = {"quantity": f"{quantity}_residue", "mode": mode,
                                   "units": "1/kyr"}
    test = relative.split("/")[0]
    return {
        "contributor": contributor,
        "kind": "love_numbers",
        "cases": [test.lower()],
        "quantity": quantity,
        "love_number_type": {"L": "loading", "T": "tidal"}[kind],
        "evidence": (f"{relative}: header `Harmonic degree, elastic, fluid, residues`; "
                     f"{width} values per record over {_lines_per_record(path)} line(s); "
                     f"the fluid limit at degree 2 holds with residue sign {signs[0]:+d} "
                     f"and the modes of {spectrum[1]}"
                     + ("; residue positions 6, 9 and 12 are the Maxwell modes" if maxwell else "")),
        "record_values": width,
        "lines_per_record": _lines_per_record(path),
        # The printed residue times this sign is the residue in the paper's
        # convention.
        "residue_sign": signs[0],
        # The residues follow the mode order of the same contributor's spectrum.
        "mode_order": spectrum[0],
        "columns": columns,
    }


def _taboo_layout(archive: pathlib.Path, path: pathlib.Path, case: str,
                  drawn_by: list[str]) -> dict:
    """The layout of one TABOO (GS) profile: one epoch, colatitude and four fields.

    Column names and units come from the two header lines TABOO writes, and
    the epoch from `Time (kyrs) =`. The rate files label column 2 `(kyrs)`,
    which cannot be the unit of a rate; the unit is taken from the other rate
    columns, mm/yr, and the header's label is kept
    (correction spada-t02-gs-rate-header-unit).
    """
    header = _header_lines(path)
    names = units = None
    for i, line in enumerate(header):
        tokens = line.split()
        if tokens and tokens[0] == "Colat.":
            names, units = tokens, header[i + 1].split()
    if names is None or len(names) != 5 or len(units) != 5:
        raise ValueError(f"{path}: no TABOO column header")
    time = next(float(l.split("=")[1]) for l in header if l.startswith("Time"))
    mass = next(float(l.split("=")[1]) for l in header if "mass of the load" in l)
    lmin, lmax = (int(v) for v in next(l for l in header if l.startswith("Lmin")).split()[-2:])
    load = next(TABOO_LOADS[l] for l in header if l in TABOO_LOADS)
    rate_units = {u.strip("()") for n, u in zip(names, units) if n.startswith("d_") and u != "(kyrs)"}
    columns = {}
    for number, (name, unit) in enumerate(zip(names, units), start=1):
        quantity = TABOO_COLUMNS[name]
        entry = {"quantity": quantity, "units": unit.strip("()").replace("deg", "degrees")}
        if quantity != "colatitude":
            entry["epoch_kyr"] = time
            if unit == "(kyrs)":
                if len(rate_units) != 1:
                    raise ValueError(f"{path}: cannot read the unit of {name}")
                entry["units"] = next(iter(rate_units))
                entry["header_units"] = "kyrs"
        columns[str(number)] = entry
    relative = str(path.relative_to(archive))
    return {
        "contributor": "GS",
        "kind": "spatial",
        "cases": [case],
        "load": load,
        "evidence": (f"{relative}: TABOO header lines `Colat. ...` and their units, "
                     f"`Time (kyrs)`, the load line and the load mass"
                     + (f"; drawn by {', '.join(drawn_by)}" if drawn_by else
                        "; drawn by no script, the TABOO point load of tests 1/2 and 2/2")),
        "load_mass_kg": mass,
        "harmonic_degrees": [lmin, lmax],
        "columns": columns,
    }


def _vilma_layout(archive: pathlib.Path, path: pathlib.Path, case: str,
                  drawn_by: list[str]) -> dict:
    """The layout of one VILMA (VK) file: colatitude and one column per epoch.

    The README in the same directory gives the quantity of each stem, the
    units (m and m/yr) and the two grids; the header gives the load, the
    quantity in words and the epochs, `# Epoch (kyr): 0 1 2 5 10`. Both are
    checked against each other.
    """
    readme = (path.parent / "README").read_text(errors="replace")
    if "Units are m and m/yr" not in readme or path.name not in readme.split():
        raise ValueError(f"{path}: the README does not document this file")
    match = re.fullmatch(r"(udt|vdt|edt|u|v|e)2cost(g?)\.dat", path.name)
    if not match:
        raise ValueError(f"{path}: not a VILMA file name")
    quantity = VILMA_STEMS[match.group(1)]
    grid = "global" if match.group(2) else "fine"
    header = _header_lines(path)
    load = next(VILMA_LOADS[l] for l in header if l in VILMA_LOADS)
    base = quantity.removesuffix("_rate")
    words = VILMA_WORDS[base] + (" rate (m/a)" if quantity.endswith("_rate") else " (m)")
    if words not in header:
        raise ValueError(f"{path}: the header does not say `{words}`")
    epochs = next(l for l in header if l.startswith("Epoch (kyr):")).split(":")[1].split()
    units = "m/yr" if quantity.endswith("_rate") else "m"
    columns = {"1": {"quantity": "colatitude", "units": "degrees"}}
    for offset, epoch in enumerate(epochs):
        columns[str(offset + 2)] = {"quantity": quantity, "units": units,
                                    "epoch_kyr": float(epoch)}
    relative = str(path.relative_to(archive))
    return {
        "contributor": "VK",
        "kind": "spatial",
        "cases": [case],
        "load": load,
        "grid": grid,
        "evidence": (f"{relative.rsplit('/', 1)[0]}/README (`Units are m and m/yr`, "
                     f"`*cost.dat contain the fine grids and *costg contain the gobal grid`, "
                     f"`North and South pole are omitted`) and the header of {path.name} "
                     f"(`{words}`, `Epoch (kyr): {' '.join(epochs)}`); drawn by {', '.join(drawn_by)}"),
        "columns": columns,
    }


# ------------------------------------------------------------------------
# Tests 5/1, 6/1 and 7/1: Love numbers against time
# ------------------------------------------------------------------------
#
# The published figures of these three tests draw four kinds of file. A
# Fortran filter cut the files of GS and VK out of larger runs, one segment per
# degree with the degree in column 3. ALMA (GSA) writes one column per degree
# after the time. RR's files of test 6/1 name the degree on each segment
# marker line. Test 7/1 draws two-column tables of degree 1. Every script
# labels its time axis `time, kyr` and scales no column 1, so column 1 is in
# kyr in all of them.

# The scripts that drew figures 6, 7 and 8 of the paper.
PUBLISHED_TIME_DOMAIN_SCRIPTS = {"T01-05/PAPER/love_PAP.gmt": "t01-05",
                                 "T01-06/love_PAP.gmt": "t01-06",
                                 "T01-07/degree1.gmt": "t01-07"}

# The program that cut the multi-segment files of each directory. filter.f90
# and FIFI.f90 of T01-05/PAPER/ differ only in the file names they open;
# FRUFRU.f90 cut VK's VILMA run of test 6/1.
FILTER_PROGRAMS = {"T01-05/PAPER": "T01-05/PAPER/filter.f90", "T01-06": "T01-06/FRUFRU.f90"}
# The write statement that fixes the three columns of every segment.
FILTER_WRITE = "time(i), love_number(l,i), l"
# VK's times are in years and start at 0. The programs store 0 as 1e-9 years,
# so that a logarithmic axis can take it, and divide by 1000.
FILTER_TIME_ZERO = "if(time(i)==0.0) time(i)=1E-9"
TIME_ZERO_WRITTEN_AS_KYR = 1e-12
# The programs read n_header_lines + 1 = 11 lines as header, then one record
# of a time and 256 degrees, by Fortran list-directed input.
FILTER_HEADER_LINES = 11
FILTER_RECORD = 257

# The program that wrote the degree-one tables of RR, VB and VK in T01-07/.
DEGREE_ONE_PROGRAM = "T01-07/filter.f90"
DEGREE_ONE_SOURCES = {
    "RR": ("the degree-one normal modes of RR in riva-T01-deg1_inc/, summed by eq. 19 over "
           "the first MVB = 6 of RR's nine modes (correction spada-t01-07-rr-six-of-nine-modes)"),
    "VB": "the six degree-one normal modes of Vb/M3-L70-V01_t01-07-VALE.txt, summed by eq. 19",
    "VK": "VK's VILMA run Vk/I106-M3-L70-V01-f1-LLN-{q}.dat, with the time divided by 1000",
}


def _data_rows(path: pathlib.Path) -> list[list[float]]:
    """Rows of numbers, skipping blank lines, `#` comments and GMT segment markers."""
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped[0] in "#>":
            continue
        rows.append([float(t.replace("D", "E")) for t in stripped.split()])
    return rows


def _segments(path: pathlib.Path) -> list[tuple[str, list[list[float]]]]:
    """A GMT multi-segment table as (marker line, rows) per segment.

    A segment starts at a line beginning with `>`. `#` lines are skipped. Data
    before the first marker is an error, because its segment is unknown.
    """
    segments: list[tuple[str, list[list[float]]]] = []
    for line in path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith(">"):
            segments.append((stripped, []))
            continue
        if not segments:
            raise ValueError(f"{path}: data before the first segment marker")
        segments[-1][1].append([float(t) for t in stripped.split()])
    return segments


def _fortran_records(path: pathlib.Path, skip: int, per: int) -> np.ndarray:
    """Records as a Fortran list-directed read takes them.

    ``skip`` lines are read as header. Each read then takes ``per`` values,
    continuing onto the next line if a line is short, and a read that ends
    inside a line discards the rest of that line. This is how the filter
    programs read the larger runs, so a file cut by one of them can be
    reproduced from its source.
    """
    records, values = [], []
    for line in path.read_text(errors="replace").splitlines()[skip:]:
        if not line.strip():
            continue
        values.extend(float(t.replace("D", "E")) for t in line.split())
        if len(values) >= per:
            records.append(values[:per])
            values = []
    return np.array(records)


def _rows_lost_at_the_top(path: pathlib.Path, read: int) -> int:
    """How many data rows at the top of a file a reader with a fixed header count missed.

    The data start after the `#` lines, and after one more line if that line
    holds only integers: the VILMA runs of test 5/1 name their degree columns
    on a line without `#`. ``read`` is the number of records the reader got.
    """
    lines = [line for line in path.read_text(errors="replace").splitlines() if line.strip()]
    first = next(i for i, line in enumerate(lines) if not line.lstrip().startswith("#"))
    if all(re.fullmatch(r"\d+", t) for t in lines[first].split()):
        first += 1
    return len(lines) - first - read


def _filter_source(path: pathlib.Path, quantity: str, contributor: str, degrees: list[int],
                   segments: list) -> tuple[str, str, int]:
    """The run a filtered file was cut from, how its values relate to it, and the rows lost.

    The filter is replayed on the source: its times (VK's divided by 1000,
    with 0 stored as 1e-12 kyr) and the column of each selected degree. A
    segment is a ``copy`` of the column or holds ``-1 minus`` the column; any
    other relation raises. The comparison allows eight printed digits.
    """
    if path.parent.name == "PAPER":
        source = path.parent / f"{quantity}_L_T01-05_{contributor}.dat"
    else:
        source = path.parent / f"I107-M3-VSCAb-LLN-{quantity}.dat"
    records = _fortran_records(source, FILTER_HEADER_LINES, FILTER_RECORD)
    times = records[:, 0].copy()
    if contributor == "VK":
        times[times == 0.0] = 1e-9
        times = times / 1000.0
    relations = set()
    for degree, (_, rows) in zip(degrees, segments):
        rows = np.array(rows)
        if rows.shape[0] != records.shape[0] or not np.allclose(rows[:, 0], times, rtol=1e-6, atol=0):
            raise ValueError(f"{path}: degree {degree} does not follow the times of {source.name}")
        column = records[:, degree]
        if np.allclose(rows[:, 1], column, rtol=1e-7, atol=1e-12):
            relations.add("copy")
        elif np.allclose(rows[:, 1], -1.0 - column, rtol=1e-7, atol=1e-12):
            relations.add("-1 minus")
        else:
            raise ValueError(f"{path}: degree {degree} is neither the column of {source.name} "
                             f"nor -1 minus it")
    if len(relations) != 1:
        raise ValueError(f"{path}: the degrees relate to {source.name} in different ways {relations}")
    return source.name, relations.pop(), _rows_lost_at_the_top(source, records.shape[0])


def _filter_layout(archive: pathlib.Path, path: pathlib.Path, case: str, contributor: str,
                   drawn_by: str) -> dict:
    """The layout of one multi-segment file cut by a Fortran filter (GS and VK, tests 5/1 and 6/1).

    Each segment follows a `>` line and holds time in kyr, the Love number
    and the degree, as the program's write statement says. The degree must be
    one integer per segment. The file is checked against the run it was cut
    from (`_filter_source`), which also shows which of VK's copies of k holds
    k and which rows the program lost.
    """
    relative = str(path.relative_to(archive))
    program_path = FILTER_PROGRAMS[relative.rsplit("/", 1)[0]]
    program = (archive / program_path).read_text(errors="replace")
    if FILTER_WRITE not in program:
        raise ValueError(f"{program_path} does not write `{FILTER_WRITE}`")
    match = (re.fullmatch(r"([hlk])_L_T01-05_(?:GS|VK)\.mod2{0,2}", path.name)
             or re.fullmatch(r"I106-M3-VSCAb-LLN-([hlk])-VK\.mod2{0,2}", path.name))
    if not match:
        raise ValueError(f"{path}: not a filtered Love number file")
    quantity = match.group(1)
    segments = _segments(path)
    degrees = []
    for _, rows in segments:
        if {len(r) for r in rows} != {3}:
            raise ValueError(f"{path}: a segment without three columns")
        found = {r[2] for r in rows}
        if len(found) != 1 or not float(next(iter(found))).is_integer():
            raise ValueError(f"{path}: column 3 is not one degree per segment ({sorted(found)[:3]})")
        degrees.append(int(found.pop()))
    source, relation, lost = _filter_source(path, quantity, contributor, degrees, segments)
    if relation == "copy":
        values = f"the values are the columns of {source} for those degrees"
    else:
        values = (f"the values are -1 minus the columns of {source}, whose k column holds "
                  f"-(1 + k) (correction spada-t01-05-vk-k-column-holds-minus-one-minus-k)")
    layout = {
        "contributor": contributor,
        "kind": "time_domain_love_numbers",
        "cases": [case],
        "quantity": quantity,
        "format": "segments_degree_column",
        "degrees": degrees,
        "columns": {"1": {"quantity": "time", "units": "kyr"},
                    "2": {"quantity": quantity, "units": "1"},
                    "3": {"quantity": "degree", "units": "1"}},
    }
    notes = []
    first_times = {rows[0][0] for _, rows in segments}
    if contributor == "VK":
        if FILTER_TIME_ZERO not in program:
            raise ValueError(f"{program_path} does not convert VK's time zero")
        if all(abs(t - TIME_ZERO_WRITTEN_AS_KYR) <= 1e-6 * TIME_ZERO_WRITTEN_AS_KYR for t in first_times):
            layout["time_zero_written_as_kyr"] = TIME_ZERO_WRITTEN_AS_KYR
            notes.append("the first time, 1e-12 kyr, is t = 0 written by the program's "
                         f"`{FILTER_TIME_ZERO}` (correction spada-vk-time-zero-written-as-1e-12-kyr)")
    if lost:
        layout["rows_lost_at_the_top"] = lost
        notes.append(f"the program read the first {lost} data row(s) of {source} as header "
                     f"(correction spada-vk-first-rows-read-as-header)")
    time_unit = ("the program divides VK's years by 1000" if contributor == "VK"
                 else "the axis `time, kyr` scales no column")
    layout["evidence"] = (f"{relative}: written by {program_path} (`{FILTER_WRITE}`), one "
                          f"segment per degree after a `>` line with the degree in column 3; "
                          f"{values}; the time is in kyr, as {time_unit}"
                          + "".join(f"; {note}" for note in notes) + f"; drawn by {drawn_by}")
    return layout


def _alma_layout(archive: pathlib.Path, path: pathlib.Path, case: str, contributor: str,
                 drawn_by: str) -> dict:
    """The layout of one ALMA table (GSA): the time in kyr and one column per degree.

    The header names the quantity (`h load Love number`, `h(t) for degrees
    l= 12 ... 96`), the degree range and the number of times. ALMA computes
    the degrees from l1 to l2 at a fixed step ls (T01-05/PAPER/alma.inc), so
    with n value columns the step is (l2 - l1) / (n - 1), which must be a
    whole number.
    """
    relative = str(path.relative_to(archive))
    header = _header_lines(path)
    quantity = next(line for line in header if line.endswith("load Love number")).split()[0]
    label = next(line for line in header if "for degrees l=" in line)
    if "time (kyrs)" not in label or label.split()[2] != f"{quantity}(t)":
        raise ValueError(f"{path}: the column label `{label}` does not name {quantity}(t) against time in kyrs")
    lo, hi = (int(v) for v in re.findall(r"\d+", next(
        line for line in header if line.startswith("Range of harmonic degrees")).split(":", 1)[1]))
    points = int(next(line for line in header if line.startswith("Time points")).split(":", 1)[1])
    rows = _data_rows(path)
    widths = {len(r) for r in rows}
    if len(widths) != 1 or len(rows) != points:
        raise ValueError(f"{path}: {len(rows)} rows of widths {widths}, the header says {points}")
    count = widths.pop() - 1
    if count == 1:
        if lo != hi:
            raise ValueError(f"{path}: one column for degrees {lo} to {hi}")
        degrees = [lo]
    else:
        step, remainder = divmod(hi - lo, count - 1)
        if remainder or step <= 0:
            raise ValueError(f"{path}: {count} columns do not step evenly from {lo} to {hi}")
        degrees = [lo + i * step for i in range(count)]
    columns = {"1": {"quantity": "time", "units": "kyr"}}
    for offset, degree in enumerate(degrees):
        columns[str(offset + 2)] = {"quantity": quantity, "units": "1", "degree": degree}
    return {
        "contributor": contributor,
        "kind": "time_domain_love_numbers",
        "cases": [case],
        "quantity": quantity,
        "format": "columns_per_degree",
        "degrees": degrees,
        "columns": columns,
        "evidence": (f"{relative}: ALMA header `{quantity} load Love number`, `Range of harmonic "
                     f"degrees: {lo} - {hi}`, `time (kyrs) {quantity}(t) for degrees l= {lo} ... {hi}` "
                     f"and `Time points: {points}`; {count} value column(s), degrees at a fixed step "
                     f"as ALMA computes them (T01-05/PAPER/alma.inc: `min and max degrees, and step`); "
                     f"drawn by {drawn_by}"),
    }


def _marker_layout(archive: pathlib.Path, path: pathlib.Path, case: str, contributor: str,
                   drawn_by: str) -> dict:
    """The layout of one of RR's files of test 6/1: the degree on each segment marker.

    Each segment starts with `>    n` and holds the time in kyr and the Love
    number, the columns the script plots by the psxy default.
    """
    relative = str(path.relative_to(archive))
    match = re.fullmatch(r"res-RR-time\d-([hkl])\.dat", path.name)
    if not match:
        raise ValueError(f"{path}: not an RR time-domain file")
    quantity = match.group(1)
    degrees = []
    for marker, rows in _segments(path):
        found = re.fullmatch(r">\s*(\d+)", marker)
        if not found or {len(r) for r in rows} != {2}:
            raise ValueError(f"{path}: segment `{marker}` is not a degree and two columns")
        degrees.append(int(found.group(1)))
    return {
        "contributor": contributor,
        "kind": "time_domain_love_numbers",
        "cases": [case],
        "quantity": quantity,
        "format": "segments_degree_marker",
        "degrees": degrees,
        "columns": {"1": {"quantity": "time", "units": "kyr"},
                    "2": {"quantity": quantity, "units": "1"}},
        "evidence": (f"{relative}: one segment per degree, the degree on its `>` line, two "
                     f"columns; the script plots columns 1 and 2 on the axis `time, kyr`; "
                     f"drawn by {drawn_by}"),
    }


def _degree_one_layout(archive: pathlib.Path, path: pathlib.Path, case: str, contributor: str,
                       drawn_by: str) -> dict:
    """The layout of one degree-one table of test 7/1 (GS, RR, VB, VK).

    GS's TABOO files name the degree and the loading type in their header. GS's
    l file has three columns: column 2 is zero at every time and column 3 is
    l, while the script plots column 2 (correction
    spada-t01-07-gs-l-in-column-3). The tables of RR, VB and VK were written by
    T01-07/filter.f90, with the time in kyr and the Love number.
    """
    relative = str(path.relative_to(archive))
    quantity = path.name[0]
    rows = _data_rows(path)
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise ValueError(f"{path}: rows of widths {widths}")
    width = widths.pop()
    layout = {"contributor": contributor, "kind": "time_domain_love_numbers", "cases": [case],
              "quantity": quantity, "format": "value_column", "degrees": [1]}
    columns = {"1": {"quantity": "time", "units": "kyr"}}
    if path.name.endswith("_time_domain.dat"):
        header = _header_lines(path)
        if not any(re.fullmatch(r"Lmin, Lmax\s+1\s+1", line) for line in header) or \
                not any(f"These {quantity} numbers are of loading type" in line for line in header):
            raise ValueError(f"{path}: the TABOO header does not give loading degree 1")
        if width == 2:
            value_column, note = 2, ""
        elif width == 3 and all(r[1] == 0.0 for r in rows):
            value_column = 3
            columns["2"] = {"quantity": "zero", "units": "1"}
            note = ("; column 2 is zero at every time and column 3 is l, while the script "
                    "plots column 2, off the axis of figure 8 (correction spada-t01-07-gs-l-in-column-3)")
        else:
            raise ValueError(f"{path}: {width} columns")
        evidence = (f"{relative}: TABOO header `These {quantity} numbers are of loading type` and "
                    f"`Lmin, Lmax 1 1`; the time is in kyr on the axis `time, kyr`{note}")
    else:
        program = (archive / DEGREE_ONE_PROGRAM).read_text(errors="replace")
        variable = {"h": "hhh(i)", "l": "lll(i)"}[quantity]
        if width != 2 or variable not in program or path.name not in program:
            raise ValueError(f"{path}: not a two-column table of {DEGREE_ONE_PROGRAM}")
        value_column = 2
        evidence = (f"{relative}: written by {DEGREE_ONE_PROGRAM} (`time, {variable}`) from "
                    + DEGREE_ONE_SOURCES[contributor].format(q=quantity))
        if contributor == "VK":
            if FILTER_TIME_ZERO not in program:
                raise ValueError(f"{DEGREE_ONE_PROGRAM} does not convert VK's time zero")
            source = path.parent / "Vk" / f"I106-M3-L70-V01-f1-LLN-{quantity}.dat"
            # The program reads 6 header lines of VK's file and one time and
            # one value per record after them.
            lost = _rows_lost_at_the_top(source, len(rows))
            if abs(rows[0][0] - TIME_ZERO_WRITTEN_AS_KYR) <= 1e-6 * TIME_ZERO_WRITTEN_AS_KYR:
                layout["time_zero_written_as_kyr"] = TIME_ZERO_WRITTEN_AS_KYR
                evidence += ("; the first time, 1e-12 kyr, is t = 0 (correction "
                             "spada-vk-time-zero-written-as-1e-12-kyr)")
            if lost:
                layout["rows_lost_at_the_top"] = lost
                evidence += (f"; the program read the first {lost} data row(s) of {source.name} as "
                             f"header (correction spada-vk-first-rows-read-as-header)")
    columns[str(value_column)] = {"quantity": quantity, "units": "1", "degree": 1}
    layout["columns"] = columns
    layout["evidence"] = evidence + f"; drawn by {drawn_by}"
    return layout


# ------------------------------------------------------------------------
# Test 3/2: polar motion and the length of day
# ------------------------------------------------------------------------
#
# T02-03/m.gmt draws figure 13: GS's curves of the cap with and without the
# Chandler wobble, VB's points of the cap without it, and three curves of the
# length of day. Table 14 prints |m| and its rate for all three loads, both
# settings and both codes, at six epochs. GS's PM_plot files hold the curves
# of all three loads, and its tables in other_data/ and VB's six tables hold
# the table 14 epochs, so all of them are read. Where one code has two files
# for one quantity, the converter joins them after checking that they agree
# at their shared times.

POLAR_MOTION_SCRIPT = "T02-03/m.gmt"

# Column labels in the headers, split at runs of two or more spaces, and the
# quantity and unit each names. `deg/Ma` and `deg/Myr` are the same unit.
POLAR_MOTION_LABELS = {
    "m_1, deg": ("m_x", "degrees"), "m_2, deg": ("m_y", "degrees"), "|m|, deg": ("m_modulus", "degrees"),
    "dm_1, deg/Ma": ("m_rate_x", "degrees/Myr"), "dm_2, deg/Ma": ("m_rate_y", "degrees/Myr"),
    "|dm|, deg/Ma": ("m_rate_modulus", "degrees/Myr"),
    "|m(t)| deg": ("m_modulus", "degrees"), "|Dm(t)| deg/Ma": ("m_rate_modulus", "degrees/Myr"),
    "mx(t) (deg)": ("m_x", "degrees"), "my(t) (deg)": ("m_y", "degrees"),
    "|m(t)| (deg)": ("m_modulus", "degrees"),
    "mdotx (deg/Myr)": ("m_rate_x", "degrees/Myr"), "mdoty (deg/Myr)": ("m_rate_y", "degrees/Myr"),
    "|mdot| (deg/Myr)": ("m_rate_modulus", "degrees/Myr"),
}
POLAR_MOTION_TIME_LABELS = ("time kyrs", "time(kyrs)", "t (kyr)")
# GS writes the load as `>>> Cap`, `>>> Disc` or `Point`; VB as `cap`, `disk` or `point`.
POLAR_MOTION_LOADS = {"Cap": "cap", "Disc": "disc", "Point": "point"}
VB_POLAR_MOTION_LOADS = {"cap": "cap", "disk": "disc", "point": "point"}

# The line of lod.f90 that gives the length of day in ms from k_2 and the
# geometrical factor of the cap, gfactor(1).
LOD_PROGRAM = "T02-03/LOD/lod.f90"
LOD_FORMULA = "DLOD=(1.+ k2)*gfactor(1)*1000."
LOD_SOURCES = {
    "GS": "k_L_T01-05_GS.mod, the degree-2 segment of GS's test 5/1",
    "VK": "k_L_T01-05_VK.mod, the degree-2 segment of VK's test 5/1 as T01-05/PAPER/ holds it",
    "GSA": "LOD/k.mod, an ALMA run of k from 10^-1.5 kyr, not the T01-05/PAPER/k.mod of figure 6",
}


def _polar_motion_columns(path: pathlib.Path, label: str, width: int) -> dict:
    """The columns a polar motion header label names, checked against the width of the rows."""
    names = re.split(r"\s{2,}", label.strip())
    if names[0] not in POLAR_MOTION_TIME_LABELS or len(names) != width:
        raise ValueError(f"{path}: the label `{label}` does not name {width} columns from a time")
    columns = {"1": {"quantity": "time", "units": "kyr"}}
    for offset, name in enumerate(names[1:]):
        quantity, units = POLAR_MOTION_LABELS[name]
        columns[str(offset + 2)] = {"quantity": quantity, "units": units, "header_label": name}
    return columns


def _polar_motion_layout(archive: pathlib.Path, path: pathlib.Path, drawn: set[str]) -> dict:
    """The layout of one polar motion file of test 3/2, from its header.

    GS's files say `Contributor: Giorgio Spada`, `CW is ON` or `OFF`, and the
    load (`>>> Cap`, `>>> Disc`, `Point`). VB's tables say `Contributor:
    Valentina R. Barletta`, `Chandler wobble term is included` or `is not
    included`, and the load. VB's two grid files carry no contributor and no
    setting: T02-03/m.gmt reads them under the comment `Cw OFF - From
    Valentina` and draws them with the legend `Vb`, and their file names say
    `cwoff` and `cap`. `m_1` and `m_2` are the x and y components, which the
    script labels `x component` and `y component`.
    """
    relative = str(path.relative_to(archive))
    header = _header_lines(path)
    rows = _data_rows(path)
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise ValueError(f"{path}: rows of widths {widths}")
    width = widths.pop()
    label = next(line for line in header if line.startswith(POLAR_MOTION_TIME_LABELS))
    if any("Giorgio Spada" in line for line in header):
        contributor = "GS"
        setting = {"ON": "included", "OFF": "excluded"}[
            next(line for line in header if line.startswith("CW is")).split()[2]]
        load = next(POLAR_MOTION_LOADS[line.lstrip(">").strip()] for line in header
                    if line.lstrip(">").strip() in POLAR_MOTION_LOADS)
        source = "GS header lines `Contributor: Giorgio Spada`, `CW is ...` and the load"
    elif any("Valentina R. Barletta" in line for line in header):
        contributor = "VB"
        setting = ("excluded" if "Chandler wobble term is not included" in header
                   else "included" if "Chandler wobble term is included" in header else None)
        load = next(VB_POLAR_MOTION_LOADS[line] for line in header if line in VB_POLAR_MOTION_LOADS)
        source = "VB header lines `Contributor: Valentina R. Barletta`, the Chandler wobble line and the load"
    elif re.fullmatch(r"Rot_cwoff_GRID_cap_m(dot)?\.txt", path.name):
        script = (archive / POLAR_MOTION_SCRIPT).read_text(errors="replace")
        # The comment line directly above the sed line that reads the file,
        # and the legend the script writes for those points.
        if not re.search(r"Cw OFF - From Valentina\s*\n\s*sed '/#/ d' " + re.escape(path.name), script) \
                or "ML Vb" not in script:
            raise ValueError(f"{POLAR_MOTION_SCRIPT} does not read {path.name} as Valentina's")
        contributor, setting, load = "VB", "excluded", "cap"
        source = (f"{POLAR_MOTION_SCRIPT} reads it under `Cw OFF - From Valentina` and draws it "
                  f"with the legend `Vb`; the file name says cap and cwoff")
    else:
        raise ValueError(f"{path}: no contributor")
    if setting is None:
        raise ValueError(f"{path}: no Chandler wobble setting")
    return {
        "contributor": contributor,
        "kind": "polar_motion",
        "cases": ["t02-03"],
        "load": load,
        "chandler_wobble": setting,
        "columns": _polar_motion_columns(path, label, width),
        "evidence": (f"{relative}: {source}; the column label `{label}`; "
                     + (f"drawn by {POLAR_MOTION_SCRIPT}" if path.name in drawn
                        else "drawn by no script; its epochs are those of table 14")),
    }


def _polar_motion_agreement(archive: pathlib.Path, files: dict, first: str, second: str,
                            rtol: float) -> None:
    """Check that two files of one code agree at their shared times in every shared quantity.

    GS's tables in other_data/ must repeat its PM_plot curves at 0, 1, 2 and 5
    kyr, and VB's grid must repeat VB's table of the cap without the wobble.
    The files of other_data/ named PM_table_<setting>_<load>.dat hold twice
    these values (correction spada-t02-03-gs-tables-with-twice-the-values) and
    fail this check, which is why they are not in the table.
    """
    tables = []
    for name in (first, second):
        rows = np.array(_data_rows(archive / name))
        columns = files[f"{ARCHIVE}/{name}"]["columns"]
        tables.append({c["quantity"]: rows[:, int(k) - 1] for k, c in columns.items()})
    a, b = tables
    shared_times, ia, ib = np.intersect1d(a["time"], b["time"], return_indices=True)
    shared = sorted((set(a) & set(b)) - {"time"})
    if shared_times.size < 3 or not shared:
        raise ValueError(f"{first} and {second} share {shared_times.size} times and {shared}")
    for quantity in shared:
        if not np.allclose(a[quantity][ia], b[quantity][ib], rtol=rtol, atol=1e-12):
            raise ValueError(f"{first} and {second} disagree in {quantity} at their shared times")


def _length_of_day_layout(archive: pathlib.Path, path: pathlib.Path, contributor: str,
                          drawn: set[str]) -> dict:
    """The layout of one length of day file of test 3/2: time in kyr and Delta LOD in ms.

    T02-03/LOD/lod.f90 wrote it from the contributor's k_2(t) by eq. 33,
    `DLOD=(1.+ k2)*gfactor(1)*1000.`, where gfactor(1) is the factor of the cap
    at colatitude 25 degrees in seconds, so the value is in ms. The script
    draws it under the comment `LOD (ms) for ...`.
    """
    relative = str(path.relative_to(archive))
    program = (archive / LOD_PROGRAM).read_text(errors="replace")
    script = (archive / POLAR_MOTION_SCRIPT).read_text(errors="replace")
    if LOD_FORMULA not in program or f"file='{path.name}'" not in program \
            or f"LOD (ms) for {contributor}" not in script:
        raise ValueError(f"{path}: not a length of day in ms written by {LOD_PROGRAM}")
    rows = _data_rows(path)
    if {len(r) for r in rows} != {2}:
        raise ValueError(f"{path}: not two columns")
    layout = {
        "contributor": contributor,
        "kind": "length_of_day",
        "cases": ["t02-03"],
        "load": "cap",
        "columns": {"1": {"quantity": "time", "units": "kyr"},
                    "2": {"quantity": "delta_lod", "units": "ms"}},
    }
    evidence = (f"{relative}: written by {LOD_PROGRAM} (`{LOD_FORMULA}`, gfactor(1) the cap) from "
                f"{LOD_SOURCES[contributor]}; "
                + (f"drawn by {POLAR_MOTION_SCRIPT} as `LOD (ms) for {contributor}`"
                   if path.name in drawn else "drawn by no script"))
    if abs(rows[0][0] - TIME_ZERO_WRITTEN_AS_KYR) <= 1e-6 * TIME_ZERO_WRITTEN_AS_KYR:
        layout["time_zero_written_as_kyr"] = TIME_ZERO_WRITTEN_AS_KYR
        evidence += "; the first time, 1e-12 kyr, is t = 0 (correction spada-vk-time-zero-written-as-1e-12-kyr)"
    layout["evidence"] = evidence
    return layout


# ------------------------------------------------------------------------
# Test 4/1: the polar motion transfer function
# ------------------------------------------------------------------------
#
# No script draws these files; table 8 of the paper prints GS's and VB's
# coefficients with the Chandler wobble included. GS's files label their
# blocks. VB's files label nothing, and their rows are read by matching them
# with GS's labelled values of the same setting.

PMTF_BLOCK_NAMES = {
    "Re(A_E)": "elastic_real", "Im(A_E)": "elastic_imag", "A_E": "elastic_real",
    "Re(A_0)": "secular_real", "Im(A_0)": "secular_imag", "A_0": "secular_real",
    "Re(a_i)": "frequency_real", "Im(a_i)": "frequency_imag",
    "Re(A_i)": "residue_real", "Im(A_i)": "residue_imag",
}
PMTF_BLOCKS = ("ELA", "SEC", "ROT")


def _gs_pmtf_blocks(path: pathlib.Path) -> tuple[dict, dict]:
    """The labels and the rows of each block of a GS transfer function file.

    A block starts at a comment line `# ELA ...`, `# SEC ...`, `# ROT ...` or
    `# ISO ...`, whose further words label the columns. Each data row starts
    with a label token (`E`, `0` or the mode number) that is not a value.
    """
    labels: dict[str, list[str]] = {}
    rows: dict[str, list[list[float]]] = {}
    block = None
    for line in path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            words = stripped.lstrip("#").split()
            if words and words[0] in PMTF_BLOCKS + ("ISO",):
                block = words[0]
                labels[block] = words[1:]
                rows[block] = []
            continue
        if block is None:
            raise ValueError(f"{path}: data before the first block")
        rows[block].append([float(t) for t in stripped.split()[1:]])
    return labels, rows


def _gs_pmtf_layout(archive: pathlib.Path, path: pathlib.Path) -> tuple[dict, dict]:
    """The layout of GS-ycw.dat or GS-ncw.dat, and its values for checking VB's files.

    The Chandler wobble setting comes from `Number of isostatic/rotational
    modes: M / M'`: M' = M with the wobble included and M' = M - 1 without it
    (eqs. 13 and 14), and the file name must agree. The unit of the frequencies
    comes from the ISO block, whose isostatic frequencies s(i) contain the nine
    of spec_T01_01_GS.dat at degree 2, which are in 1/kyr.
    """
    relative = str(path.relative_to(archive))
    header = _header_lines(path)
    counts = next(line for line in header if line.startswith("Number of isostatic/rotational modes"))
    isostatic, rotational = (int(v) for v in re.findall(r"\d+", counts.split(":", 1)[1]))
    if rotational == isostatic:
        setting = "included"
    elif rotational == isostatic - 1:
        setting = "excluded"
    else:
        raise ValueError(f"{path}: {isostatic} isostatic and {rotational} rotational modes")
    if ("ycw" in path.name) != (setting == "included"):
        raise ValueError(f"{path}: the file name and `{counts}` disagree")
    labels, rows = _gs_pmtf_blocks(path)
    if len(rows["ROT"]) != rotational or len(rows["ELA"]) != 1 or len(rows["SEC"]) != 1:
        raise ValueError(f"{path}: the blocks do not hold one A_E, one A_0 and {rotational} modes")
    blocks = {name: [PMTF_BLOCK_NAMES[label] for label in labels[name]] for name in PMTF_BLOCKS}
    for name in PMTF_BLOCKS:
        if {len(r) for r in rows[name]} != {len(blocks[name])}:
            raise ValueError(f"{path}: block {name} rows do not match its labels {labels[name]}")
    spectrum = next(r for r in _records(archive / "T01-01" / "spec_T01_01_GS.dat") if int(r[0]) == 2)[1:]
    isostatic_s = [r[0] for r in rows["ISO"]]
    if not all(any(abs(s - t) <= 1e-6 * abs(s) for t in isostatic_s) for s in spectrum):
        raise ValueError(f"{path}: the ISO frequencies do not contain GS's degree-2 spectrum")
    values = {"setting": setting,
              "secular_real": rows["SEC"][0][0],
              "frequency_real": [r[0] for r in rows["ROT"]]}
    layout = {
        "contributor": "GS",
        "kind": "polar_motion_transfer_function",
        "cases": ["t01-04"],
        "chandler_wobble": setting,
        "format": "labelled_blocks",
        "blocks": blocks,
        "modes": rotational,
        "units": {"elastic": "1", "secular": "1/kyr", "frequency": "1/kyr", "residue": "1/kyr"},
        "evidence": (f"{relative}: header `{counts}`, so the Chandler wobble is {setting} (eqs. 13 "
                     f"and 14); blocks labelled {', '.join(' '.join([b] + labels[b]) for b in PMTF_BLOCKS)}; "
                     f"the ISO block's s(i) contain the degree-2 frequencies of "
                     f"T01-01/spec_T01_01_GS.dat, in 1/kyr; an unlabelled imaginary part is not "
                     f"printed and is zero; drawn by no script, the values of table 8"),
    }
    return layout, values


def _vb_pmtf_layout(archive: pathlib.Path, path: pathlib.Path, gs_values: dict) -> dict:
    """The layout of VB-ycw.txt or VB-ncw.txt, established against GS's file of the same setting.

    The file has text header lines up to a row of `*`, then A_e (real and
    imaginary), A_s (real and imaginary), and one row per rotational mode:
    a_j and A_j, real and imaginary. VB labels none of it. The reading holds
    because VB's A_s equals GS's to 1e-7 and the real part of every one of VB's
    frequencies equals one of GS's to 1e-6.
    """
    relative = str(path.relative_to(archive))
    lines = path.read_text(errors="replace").splitlines()
    end = next(i for i, line in enumerate(lines) if line.strip().startswith("****"))
    header = [line.strip() for line in lines[:end]]
    if "Chandler wobble term is not included" in header:
        setting = "excluded"
    elif "Chandler wobble term is included" in header:
        setting = "included"
    else:
        raise ValueError(f"{path}: no Chandler wobble line")
    if setting != gs_values["setting"]:
        raise ValueError(f"{path}: compared with a GS file of the other setting")
    rows = [[float(t) for t in line.split()] for line in lines[end + 1:] if line.strip()]
    if [len(r) for r in rows[:2]] != [2, 2] or {len(r) for r in rows[2:]} != {4}:
        raise ValueError(f"{path}: the rows are not A_e, A_s and four values per mode")
    if abs(rows[1][0] - gs_values["secular_real"]) > 1e-7 * abs(gs_values["secular_real"]):
        raise ValueError(f"{path}: row 2 is not GS's A_s")
    for row in rows[2:]:
        if not any(abs(row[0] - f) <= 1e-6 * abs(f) for f in gs_values["frequency_real"]):
            raise ValueError(f"{path}: frequency {row[0]} is none of GS's")
    return {
        "contributor": "VB",
        "kind": "polar_motion_transfer_function",
        "cases": ["t01-04"],
        "chandler_wobble": setting,
        "format": "rows",
        "header_lines": end + 1,
        "rows": {"1": ["elastic_real", "elastic_imag"], "2": ["secular_real", "secular_imag"],
                 "3 and later": ["frequency_real", "frequency_imag", "residue_real", "residue_imag"]},
        "modes": len(rows) - 2,
        "units": {"elastic": "1", "secular": "1/kyr", "frequency": "1/kyr", "residue": "1/kyr"},
        "evidence": (f"{relative}: header `Chandler wobble term is "
                     f"{'not ' if setting == 'excluded' else ''}included`; no labels, so the rows are "
                     f"read by agreement with GS-{'ycw' if setting == 'included' else 'ncw'}.dat: row 2 "
                     f"equals GS's A_s to 1e-7 and each mode's first value equals one of GS's "
                     f"frequencies to 1e-6; drawn by no script, the values of table 8"),
    }


def file_layouts(archive: pathlib.Path, decoded: list[dict]) -> dict:
    """Every file a Spada reader uses, keyed by its path under data/original/.

    The files are those the published scripts of tests 1/2 and 2/2 draw, the
    TABOO point load profiles in T02/, the four relaxation spectra of test 1/1
    and the signed Love number tables of tests 2/1 and 3/1, all in the working
    directories and not in PAPER/; the files the published scripts of tests
    5/1 (in PAPER/), 6/1 and 7/1 draw; the transfer functions of test 4/1;
    and the polar motion and length of day files of test 3/2. A file is chosen
    by the script that drew it where one exists, which is what keeps the runs
    with a different load out (correction spada-t02-gs-runs-with-a-different-load)
    and VK's working copy of k in test 5/1 out
    (correction spada-paper-copy-differs-t01-05-vk).
    """
    files: dict[str, dict] = {}

    # Tests 1/2 and 2/2: whatever the published script draws.
    drawn: dict[tuple[str, str], list[str]] = {}
    for script in decoded:
        case = PUBLISHED_SPATIAL_SCRIPTS.get(script["script"])
        if case is None:
            continue
        for panel in script["panels"]:
            for series in panel["series"]:
                if not series["drawn"]:
                    continue
                name = f"{script['test']}/{series['file'].removeprefix('./')}"
                drawn.setdefault((name, case), [])
                if script["script"] not in drawn[(name, case)]:
                    drawn[(name, case)].append(script["script"])
    for (name, case), scripts in sorted(drawn.items()):
        path = archive / name
        layout = (_vilma_layout if "/output_sec2" in name else _taboo_layout)(
            archive, path, case, scripts)
        files[f"{ARCHIVE}/{name}"] = layout

    # The point load, which no script draws.
    for prefix, case in (("disp", "t02-01"), ("rate", "t02-02")):
        for path in sorted((archive / "T02").glob(f"{prefix}_t*_point.dat")):
            files[f"{ARCHIVE}/{path.relative_to(archive)}"] = _taboo_layout(archive, path, case, [])

    # Test 1/1, and the Love numbers paired with it.
    spectra = {}
    for path in sorted((archive / "T01-01").glob("spec_T01_01_*.dat")):
        layout = _spectrum_layout(archive, path)
        files[f"{ARCHIVE}/{path.relative_to(archive)}"] = layout
        records = _records(path)
        row = next(r for r in records if int(r[0]) == 2)
        spectra[layout["contributor"]] = (layout["mode_order"], path.name, row[1:])
    for test, kind in (("T01-02", "L"), ("T01-03", "T")):
        for path in sorted((archive / test).glob(f"[hlk]_{kind}_{test}_*.dat")):
            contributor, _ = contributor_of(path.name)
            files[f"{ARCHIVE}/{path.relative_to(archive)}"] = _love_layout(
                archive, path, spectra[contributor])

    # Tests 5/1, 6/1 and 7/1: whatever the published script draws. A name is
    # relative to the script's own directory, and the contributor is the one
    # the script decoder found for the series, from the path or the temporary
    # file the data passed through.
    for script in decoded:
        case = PUBLISHED_TIME_DOMAIN_SCRIPTS.get(script["script"])
        if case is None:
            continue
        directory = f"{script['test']}/{script['variant']}" if script["variant"] else script["test"]
        chosen: dict[str, str] = {}
        for panel in script["panels"]:
            for series in panel["series"]:
                if series["drawn"]:
                    chosen.setdefault(f"{directory}/{series['file'].removeprefix('./')}",
                                      series["contributor"])
        for name, contributor in sorted(chosen.items()):
            path = archive / name
            if re.fullmatch(r"[hlk]\.mod2{0,2}|[hl]-alma-deg1\.dat", path.name):
                layout = _alma_layout(archive, path, case, contributor, script["script"])
            elif path.name.startswith("res-RR-time"):
                layout = _marker_layout(archive, path, case, contributor, script["script"])
            elif case == "t01-07":
                layout = _degree_one_layout(archive, path, case, contributor, script["script"])
            else:
                layout = _filter_layout(archive, path, case, contributor, script["script"])
            files[f"{ARCHIVE}/{name}"] = layout

    # Test 3/2: GS's curves and tables of all three loads, VB's tables and
    # grids, and the three lengths of day. The GS tables are those named
    # PM_table_<load>_<setting>.dat; the ones named the other way round hold
    # twice the values and are left out.
    drawn_3_2 = {series["file"] for script in decoded if script["script"] == POLAR_MOTION_SCRIPT
                 for panel in script["panels"] for series in panel["series"] if series["drawn"]}
    t02_03 = archive / "T02-03"
    polar = (sorted(t02_03.glob("PM_plot_*-[yn]cw.dat"))
             + sorted((t02_03 / "other_data").glob("PM_table_[cdp]*_[yn]cw.dat"))
             + sorted(t02_03.glob("M3-L70-V01_t02-03_*_cw*.txt"))
             + [t02_03 / "Rot_cwoff_GRID_cap_m.txt", t02_03 / "Rot_cwoff_GRID_cap_mdot.txt"])
    for path in polar:
        files[f"{ARCHIVE}/{path.relative_to(archive)}"] = _polar_motion_layout(archive, path, drawn_3_2)
    for short in ("cap", "dis", "poi"):
        for setting in ("ycw", "ncw"):
            _polar_motion_agreement(archive, files, f"T02-03/PM_plot_{short}-{setting}.dat",
                                    f"T02-03/other_data/PM_table_{short}_{setting}.dat", 1e-7)
    for grid in ("Rot_cwoff_GRID_cap_m.txt", "Rot_cwoff_GRID_cap_mdot.txt"):
        _polar_motion_agreement(archive, files, f"T02-03/{grid}",
                                "T02-03/M3-L70-V01_t02-03_cap_cwoff.txt", 1e-9)
    for contributor in ("GS", "GSA", "VK"):
        path = t02_03 / f"lod_{contributor}.dat"
        files[f"{ARCHIVE}/{path.relative_to(archive)}"] = _length_of_day_layout(
            archive, path, contributor, drawn_3_2)

    # Test 4/1: GS's labelled files first, because VB's are read against them.
    for setting in ("ycw", "ncw"):
        gs_path = archive / "T01-04" / f"GS-{setting}.dat"
        layout, values = _gs_pmtf_layout(archive, gs_path)
        files[f"{ARCHIVE}/{gs_path.relative_to(archive)}"] = layout
        vb_path = archive / "T01-04" / f"VB-{setting}.txt"
        files[f"{ARCHIVE}/{vb_path.relative_to(archive)}"] = _vb_pmtf_layout(archive, vb_path, values)

    # The units a reader converts rates with must reproduce the published axis:
    # the script multiplies a stored value by its scale, so the stored unit in
    # m/kyr divided by the scale is one number for every series on the axis.
    quotients = set()
    for script in decoded:
        if script["script"] != "T02-02/plot_PAP.gmt":
            continue
        for panel in script["panels"]:
            for series in panel["series"]:
                if not series["drawn"]:
                    continue
                name = f"{ARCHIVE}/T02-02/{series['file'].removeprefix('./')}"
                column = files[name]["columns"][str(series["ordinate"]["column"])]
                scale = series["ordinate"]["scale"]
                factor = 1.0 if scale is None else float(re.fullmatch(r"x\*([\d.]+)", scale).group(1))
                quotients.add(round(IN_M_PER_KYR[column["units"]] / factor, 9))
    if len(quotients) != 1:
        raise ValueError(f"the rate units disagree with T02-02/plot_PAP.gmt: {quotients}")
    return dict(sorted(files.items()))


def main() -> int:
    here = pathlib.Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=pathlib.Path,
                    default=here / "data" / "original")
    ap.add_argument("--output", type=pathlib.Path,
                    default=here / "gia_mip" / "resources"
                    / "spada_column_layouts.json")
    args = ap.parse_args()

    archive = args.root / "spada2011" / "spada2011-cost-es0701"
    scripts = sorted(archive.rglob("*.gmt"))
    if not scripts:
        print(f"no GMT scripts under {archive}. Fetch and unpack the Spada "
              f"archive first.", file=sys.stderr)
        return 1

    decoded = []
    for path in scripts:
        relative = path.relative_to(archive)
        # `T01-01/sa.gmt` is the working copy; `T01-01/PAPER/sa.gmt` is the
        # copy that drew the published figure. Which of the two a case uses is
        # a decision recorded in the case file, so both are decoded and the
        # difference is kept visible here.
        test = relative.parts[0]
        variant = "/".join(relative.parts[1:-1])
        decoded.append(decode(path, test, variant))

    # Invert to a per-file view, which is the question a reader actually asks:
    # given this data file, what do its columns hold?
    per_file: dict[str, list[dict]] = {}
    for script in decoded:
        for panel in script["panels"]:
            for series in panel["series"]:
                per_file.setdefault(series["file"], []).append({
                    "script": script["script"],
                    "test": script["test"],
                    "variant": script["variant"],
                    "panel": panel["output"],
                    "panel_index": panel["index"],
                    "quantity": panel["quantity"],
                    "abscissa_label": panel["abscissa_label"],
                    "abscissa": series["abscissa"],
                    "ordinate": series["ordinate"],
                    "contributor": series["contributor"],
                    "contributor_source": series["contributor_source"],
                    "banner_contributor": series["banner_contributor"],
                    "sections": panel["sections"],
                    "multisegment": series["multisegment"],
                    "symbol": series["symbol"],
                    "abscissa_log": panel["abscissa_log"],
                    "ordinate_log": panel["ordinate_log"],
                    "drawn_in_the_published_figure": series["drawn"],
                })

    # Where the path and the banner comment disagree, both are written out so
    # that a person can settle it. The path wins in the record because it
    # names the file itself; the banner names the block the read sat in, and a
    # combined panel re-reads every contributor's file inside one block.
    disagree = sorted({
        (f, e["script"], e["contributor"], e["banner_contributor"])
        for f, entries in per_file.items() for e in entries
        if e["banner_contributor"] is not None
        and e["contributor"] is not None
        and e["banner_contributor"] != e["contributor"]})
    print(f"{len(disagree)} file and script pairs have a banner comment that "
          f"names a different contributor from the path")

    doc = {
        "$comment": (
            "What each column of each Spada et al. (2011) data file holds, "
            "decoded from the GMT 4 scripts in the COST ES0701 archive. "
            "Generated by scripts/decode_spada_layouts.py. GMT 4 no longer "
            "runs and most of these files have no header, so these scripts "
            "are the only record. Where no awk stage selected columns, the "
            "GMT 4 psxy default applies and is written out explicitly: "
            "column 1 is the abscissa, column 2 the ordinate. `negated` and "
            "`scale` record arithmetic the script applied before plotting, "
            "which a data comparison must not repeat."),
        "scripts": decoded,
        "by_file": dict(sorted(per_file.items())),
        "attribution_disagreements": [
            {"file": f, "script": script, "path_says": path_code,
             "banner_says": banner_code}
            for f, script, path_code, banner_code in disagree],
        # The table a reader uses, keyed by the path under data/original/. It
        # is written by this script and never by hand, so the next run cannot
        # remove a table the reader depends on.
        "files": file_layouts(archive, decoded),
    }
    args.output.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")

    n_series = sum(len(v) for v in per_file.values())
    n_panels = sum(len(s["panels"]) for s in decoded)
    print(f"{len(decoded)} scripts, {n_panels} panels, {len(per_file)} data "
          f"files, {n_series} column assignments")
    print(f"{len(doc['files'])} files with a per-file layout for the reader")

    unnamed = [p["output"] for s in decoded for p in s["panels"]
               if p["quantity"] is None and p["series"]]
    if unnamed:
        print(f"{len(unnamed)} panels have no ordinate label: "
              f"{sorted(set(unnamed))[:8]}", file=sys.stderr)
    sources = collections.Counter(e["contributor_source"]
                                  for v in per_file.values() for e in v)
    print("contributor taken from: " + ", ".join(
        f"{source or 'nothing'} {count}" for source, count in sources.most_common()))
    unattributed = sorted({
        f for f, entries in per_file.items()
        if all(e["contributor"] is None for e in entries)})
    if unattributed:
        print(f"{len(unattributed)} files have no contributor: "
              f"{unattributed[:8]}", file=sys.stderr)


    try:
        shown = args.output.relative_to(here)
    except ValueError:
        shown = args.output
    print(f"written to {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
