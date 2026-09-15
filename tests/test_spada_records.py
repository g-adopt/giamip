"""The records around the Spada et al. (2011) data are complete and consistent.

These tests need no data. They check the files a person reads and CI runs:
the redistribution permission, the CI fetch step, the case files, the
documentation pages, and the per-file column layouts that the reader uses.

The column layouts table ``files`` in ``gia_mip/resources/spada_column_layouts.json``
is keyed by the path relative to ``data/original/``, the key the corrections
table and the manifest use. The existing ``by_file`` table is keyed by the
name a GMT script wrote, which is relative to the script's directory and so
names two different files in T02-01 and T02-02 with one key. A reader needs
the unambiguous key.

Each entry of ``files`` has the form::

    {"contributor": "VK",
     "evidence": "T02-01/output_sec2parab/README and the header of u2cost.dat",
     "columns": {"1": {"quantity": "colatitude", "units": "degrees"},
                 "2": {"quantity": "U", "units": "m", "epoch_kyr": 0}, ...}}
"""

import json
import pathlib
import re
import subprocess
import sys

import pytest

from gia_mip import cases, data

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESOURCES = pathlib.Path(data.__file__).parent / "resources"
ARCHIVE = "spada2011/spada2011-cost-es0701.tar.gz"
PREFIX = "spada2011/spada2011-cost-es0701"

# The status every Spada case file carried before a reader existed. A case
# file that still says this after the reader and converter exist is stale.
OLD_STATUS = "specified; no reader and no converted data yet"

# The VILMA epochs, from the header line `# Epoch (kyr): 0 1 2 5 10` of every
# *2cost.dat file. Column n + 2 holds epoch EPOCHS_VK[n].
EPOCHS_VK = (0, 1, 2, 5, 10)

# How many m/kyr one unit is. A rate in mm/yr equals the same number in m/kyr,
# because both are 1e-3 m per 1e-3 of the other time unit. A rate in m/yr is
# 1000 m/kyr. Used to tie each stored unit to the axis of the published figure.
IN_M_PER_KYR = {"m/yr": 1000.0, "mm/yr": 1.0, "m/kyr": 1.0}


@pytest.fixture(scope="module")
def provenance():
    return json.loads((RESOURCES / "provenance.json").read_text())


@pytest.fixture(scope="module")
def layouts():
    return json.loads((RESOURCES / "spada_column_layouts.json").read_text())


@pytest.fixture(scope="module")
def regenerated(data_root, tmp_path_factory):
    """The layouts document as scripts/decode_spada_layouts.py writes it now.

    The committed ``spada_column_layouts.json`` is generated: the script
    rewrites the whole document on every run. A table added to the JSON by
    hand, and not by the script, disappears without a warning at the next
    run, and the reader then loses the layouts it depends on. The decoder is
    therefore run into a temporary file and its output is the evidence.

    The script reads only the GMT scripts and README files of the unpacked
    archive, so the fixture skips when the archive is not unpacked. One run
    takes about a second, so the fixture is module-scoped and not slow.
    """
    archive = data_root / "original" / PREFIX
    if not archive.is_dir():
        pytest.skip(f"{archive} is not unpacked")
    output = tmp_path_factory.mktemp("spada_layouts") / "spada_column_layouts.json"
    # The same interpreter as the test run, so the script sees the same
    # packages. check=True makes a crash of the decoder an error of the
    # fixture, with the output of the script in the message.
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "decode_spada_layouts.py"),
         "--root", str(data_root / "original"), "--output", str(output)],
        capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return output


@pytest.fixture(scope="module")
def files(layouts):
    """The per-file layouts keyed by the path under data/original/.

    An absent table gives an empty mapping, so each test that needs an entry
    fails on its own claim instead of every test erroring in set-up.
    """
    return layouts.get("files", {})


# ------------------------------------------------------ permission and CI

def test_the_spada_archive_is_granted_for_redistribution(provenance):
    assert provenance["artifacts"][ARCHIVE]["permission"] == "granted"


def test_the_provenance_note_records_whose_decision_it_was_and_when(provenance):
    """The grant is Sia Ghelichkhani's decision of 2026-09-14, and it must say so."""
    notes = provenance["artifacts"][ARCHIVE]["notes"]
    assert "Sia Ghelichkhani" in notes
    assert "2026-09-14" in notes


def test_the_manifest_grants_the_spada_archive():
    """scripts/build_manifest.py copies permission; a stale manifest blocks the fetch.

    `python -m gia_mip.data` reads the manifest, not the provenance file, so a
    grant recorded in provenance.json alone still refuses the download in CI.
    """
    assert data.load_manifest()["artifacts"][ARCHIVE]["permission"] == "granted"


def test_the_data_job_in_ci_fetches_spada2011():
    """The `test-with-data` job must download the Spada archive before `pytest -m data`.

    The job is read as text: the fetch command is any line of that job that
    runs `gia_mip.data`, and one of them must name the spada2011 collection.
    """
    workflow = (ROOT / ".github" / "workflows" / "test.yml").read_text()
    job = workflow.split("test-with-data:", 1)
    assert len(job) == 2, "the workflow has no test-with-data job"
    fetch_lines = [line for line in job[1].splitlines() if "gia_mip.data" in line]
    assert fetch_lines, "the test-with-data job runs no fetch"
    assert any(re.search(r"--collection[ =]spada2011\b", line) for line in fetch_lines), fetch_lines


# ---------------------------------------------------------- case files

@pytest.mark.parametrize("case_id", [f"spada2011-t01-0{i}" for i in range(1, 8)]
                         + [f"spada2011-t02-0{i}" for i in range(1, 4)])
def test_every_spada_case_has_an_updated_status(case_id):
    """The reader exists after this task, so no case may still say it does not."""
    status = cases.case(case_id)["status"]
    assert status.strip() and status != OLD_STATUS, (case_id, status)


# ------------------------------------------------------------ documentation

def test_the_collection_page_no_longer_says_the_scripts_are_not_decoded():
    page = (ROOT / "docs" / "collections" / "spada2011.qmd").read_text()
    assert "## Not decoded yet" not in page


@pytest.mark.parametrize("name", ["SpadaCase", "SpadaResult"])
def test_the_python_interface_page_documents_the_spada_classes(name):
    page = (ROOT / "docs" / "reference" / "python-interface.qmd").read_text()
    assert name in page


# ------------------------------------------------------ per-file column layouts

def _vilma_paths(test: str, rates: bool) -> list[str]:
    """Every VILMA file of one test directory that the published figure reads."""
    stems = ("udt", "vdt", "edt") if rates else ("u", "v", "e")
    return [f"{PREFIX}/{test}/{directory}/{stem}2{grid}.dat"
            for directory in ("output_sec2parab", "output_sec2disc")
            for stem in stems for grid in ("cost", "costg")]


@pytest.mark.parametrize("path", _vilma_paths("T02-01", rates=False)
                         + _vilma_paths("T02-02", rates=True))
def test_every_vilma_file_has_a_decoded_layout(files, path):
    """The VILMA files are documented only by README files, and still need a layout."""
    assert path in files, path


@pytest.mark.parametrize("stem,quantity", [("u", "U"), ("v", "V"), ("e", "N")])
def test_the_vilma_displacement_columns_are_one_epoch_each(files, stem, quantity):
    """README: u vertical, v horizontal, e geoid, in metres; header: epochs 0 1 2 5 10 kyr."""
    columns = files[f"{PREFIX}/T02-01/output_sec2parab/{stem}2cost.dat"]["columns"]
    assert columns["1"]["quantity"] == "colatitude"
    assert columns["1"]["units"] == "degrees"
    for offset, epoch in enumerate(EPOCHS_VK):
        column = columns[str(offset + 2)]
        assert (column["quantity"], column["units"], column["epoch_kyr"]) == (quantity, "m", epoch)


@pytest.mark.parametrize("stem,quantity", [("udt", "U_rate"), ("vdt", "V_rate"), ("edt", "N_rate")])
def test_the_vilma_rate_columns_are_stored_in_metres_per_year(files, stem, quantity):
    """README: `Units are m and m/yr`, and the header of udt2cost.dat says `(m/a)`."""
    columns = files[f"{PREFIX}/T02-02/output_sec2parab/{stem}2costg.dat"]["columns"]
    for offset, epoch in enumerate(EPOCHS_VK):
        column = columns[str(offset + 2)]
        assert (column["quantity"], column["units"], column["epoch_kyr"]) == (quantity, "m/yr", epoch)


def test_the_gs_spatial_columns_skip_column_four(files):
    """disp_t0_cap.dat: U in column 2, V in 3, the geoid N in 5. Column 4 is not N.

    Correction spada-t02-time-layout-differs-by-contributor. Column 4 is zero
    along the meridian, so a geoid read from it is a flat line.
    """
    columns = files[f"{PREFIX}/T02-01/disp_t0_cap.dat"]["columns"]
    assert columns["1"]["quantity"] == "colatitude"
    assert (columns["2"]["quantity"], columns["2"]["units"]) == ("U", "m")
    assert (columns["3"]["quantity"], columns["3"]["units"]) == ("V", "m")
    assert (columns["5"]["quantity"], columns["5"]["units"]) == ("N", "m")
    assert columns.get("4", {}).get("quantity") not in ("U", "V", "N")


def test_the_gs_rate_files_are_stored_in_millimetres_per_year(files):
    """TABOO writes rates in mm/yr, and the figure draws them without a factor.

    The figure of test 2/2 plots GS's column 2 as it stands and VK's column
    times 1000, on one axis. With VK stored in m/yr, the axis is mm/yr, so GS
    is mm/yr.
    """
    column = files[f"{PREFIX}/T02-02/rate_t0_cap.dat"]["columns"]["2"]
    assert (column["quantity"], column["units"]) == ("U_rate", "mm/yr")


def test_the_stored_rate_units_reproduce_the_axis_of_the_figure(layouts, files):
    """The stored unit divided by the script scale is one axis unit for every series.

    The script multiplies a stored value by its scale, so one axis unit is
    (stored unit) / (scale) of physical rate. For every rate series the
    published T02-02 script draws, the layout's stored unit is converted to
    m/kyr and divided by the scale the script applies (x*1000 for VK, none for
    GS). All series share one axis, so the quotient must be one number. VK in
    m/yr gives 1000 / 1000 = 1 and GS in mm/yr gives 1 / 1 = 1. A unit
    transcribed as m/yr for GS gives 1000, and mm/yr for VK gives 0.001. The
    comparison is exact up to rounding, because every factor is a power of ten.
    """
    from gia_mip.benchmarks.martinec2018 import scale_factor

    seen = []
    for name, entries in layouts["by_file"].items():
        for entry in entries:
            if entry["script"] != "T02-02/plot_PAP.gmt" or not entry["drawn_in_the_published_figure"]:
                continue
            path = str(pathlib.PurePosixPath(PREFIX, "T02-02", name.removeprefix("./")))
            column = files[path]["columns"][str(entry["ordinate"]["column"])]
            factor = IN_M_PER_KYR[column["units"]] / scale_factor(entry["ordinate"]["scale"])
            seen.append((path, factor))
    assert seen, "no drawn series found in T02-02/plot_PAP.gmt"
    factors = {round(f, 9) for _, f in seen}
    assert len(factors) == 1, seen


# ------------------------------------------- the layouts are the decoder output

@pytest.mark.data
def test_the_decoder_writes_the_files_table(layouts, regenerated):
    """The ``files`` table the reader uses comes from the decoder, not a hand edit.

    Every test above that reads ``files`` passes for a hand-written table too.
    This test ties them to scripts/decode_spada_layouts.py: the table in a new
    run of the script must be non-empty and equal to the committed table.
    """
    produced = json.loads(regenerated.read_text()).get("files", {})
    assert produced, "scripts/decode_spada_layouts.py writes no `files` table"
    assert produced == layouts.get("files", {})


@pytest.mark.data
def test_the_committed_layouts_are_the_decoder_output(regenerated):
    """The whole committed document is byte-for-byte what the decoder writes.

    The script sorts its inputs and its tables and writes no time stamp, so
    its output is deterministic. Two runs with PYTHONHASHSEED 1 and 987 gave
    identical bytes. An exact comparison is therefore correct here, and any
    difference means that either the JSON was edited by hand or the script
    changed and was not run again.
    """
    committed = (RESOURCES / "spada_column_layouts.json").read_text()
    assert regenerated.read_text() == committed
