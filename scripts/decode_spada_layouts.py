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
VILMA files, the signed Love number tables and the point load profiles) that
evidence is the README files, the file headers and arithmetic on the values,
written out in file_layouts.

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

# How many m/kyr one stored rate unit is. 1 mm/yr is 1e-3 m per 1e-3 kyr, so
# it is the same number in m/kyr; 1 m/yr is 1000 m/kyr.
IN_M_PER_KYR = {"mm/yr": 1.0, "m/yr": 1000.0, "m/kyr": 1.0}

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


def _love_layout(archive: pathlib.Path, path: pathlib.Path, spectrum: dict) -> dict:
    """The layout of one signed Love number table of test 2/1 or 3/1.

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


def file_layouts(archive: pathlib.Path, decoded: list[dict]) -> dict:
    """Every file a Spada reader uses, keyed by its path under data/original/.

    The files are those the published scripts of tests 1/2 and 2/2 draw, the
    TABOO point load profiles in T02/, the four relaxation spectra of test 1/1
    and the signed Love number tables of tests 2/1 and 3/1, all in the working
    directories and not in PAPER/. A file is chosen by the script that drew it
    where one exists, which is what keeps the runs with a different load out
    (correction spada-t02-gs-runs-with-a-different-load).
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
