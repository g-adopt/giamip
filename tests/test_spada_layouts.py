"""The decoded Spada et al. (2011) column layouts are complete and consistent.

These guard the same thing `test_layouts.py` guards for the Martinec archive:
for data files with no header, the GMT 4 scripts are the only record of what
their columns hold, and GMT 4 no longer runs.

Two facts pinned here were checked by hand against material in the archive
that the decoder never reads, so a change in the decoder that breaks them is a
real regression and not a change of opinion:

- `T02-01/output_sec2disc/README` says the `[uve]2cost.dat` files hold
  vertical displacement, horizontal displacement and geoid, in metres, for
  the disc load, and that `*costg.dat` is the global grid while `*cost.dat`
  is the fine one. The header of `u2cost.dat` names its epochs as 0, 1, 2, 5
  and 10 kyr, which is five columns after the colatitude.
- `u2cost.dat` records `performed by spectral FE code, Klemann at GFZ`, which
  is the contributor the decoder attributes to VK.
"""

import collections
import json
import pathlib

import pytest

from giamip import data

LAYOUTS = (pathlib.Path(data.__file__).parent / "resources"
           / "spada_column_layouts.json")

# The contributor abbreviations of the paper's table 1, as recorded in the ten
# case files. `ALMA` is the name of a program and is mapped to GSA by the
# decoder, so it must never appear here.
TABLE_ONE = {"BL", "GS", "GSA", "PG", "RR", "VB", "VK", "ZM"}


@pytest.fixture(scope="module")
def layouts():
    return json.loads(LAYOUTS.read_text())


def test_every_script_in_the_archive_was_decoded(layouts):
    """Eighty-seven GMT scripts, across the ten test directories.

    A drop in this number means a script stopped matching the parser, and its
    column layouts are then missing rather than wrong, which is harder to
    notice.
    """
    assert len(layouts["scripts"]) == 87
    tests = {script["test"] for script in layouts["scripts"]}
    assert tests == {"T01-01", "T01-02", "T01-03", "T01-05", "T01-06",
                     "T01-07", "T02-01", "T02-02", "T02-03"}


def test_no_series_stops_at_a_temporary_file(layouts):
    """Every chain resolves back to a delivered file.

    The scripts strip headers into `data.tmp` and select columns into
    `data_GS_u_0.tmp`, overwriting both many times. A series left pointing at
    a temporary means the chain was not followed, and the layout recorded for
    it belongs to whichever contributor wrote that temporary last.
    """
    for name in layouts["by_file"]:
        assert not name.endswith(".tmp"), name
        assert not name.startswith("<"), name


def test_every_series_names_one_abscissa_and_one_ordinate_column(layouts):
    for name, entries in layouts["by_file"].items():
        for entry in entries:
            assert "column" in entry["abscissa"], f"{name}: {entry['abscissa']}"
            assert "column" in entry["ordinate"], f"{name}: {entry['ordinate']}"


def test_the_psxy_default_columns_are_written_out(layouts):
    """A file read with no awk stage gets GMT's default, recorded explicitly.

    Most of this archive relies on that default. Writing it into the record
    means no reader has to know a GMT 4 convention to use this file.
    """
    defaults = [entry for entries in layouts["by_file"].values()
                for entry in entries
                if "psxy default" in entry["abscissa"]["expression"]]
    assert defaults, "no series took the psxy default; the parser changed"
    for entry in defaults:
        assert entry["abscissa"]["column"] == 1
        assert entry["ordinate"]["column"] == 2
        assert entry["abscissa"]["negated"] is False
        assert entry["abscissa"]["scale"] is None


def test_every_contributor_is_one_from_table_one(layouts):
    """`ALMA` is a program name and must have been mapped to GSA."""
    found = {entry["contributor"]
             for entries in layouts["by_file"].values() for entry in entries}
    found.discard(None)
    assert found <= TABLE_ONE, sorted(found - TABLE_ONE)
    assert "ALMA" not in found


def test_gsa_is_not_read_as_gs(layouts):
    """`h_L_T01-06_GSA.dat` is GSA's and `h_L_T01-05_GS.dat` is GS's.

    A substring test attributes both to GS, which silently merges two
    contributors' results into one.
    """
    for name, entries in layouts["by_file"].items():
        stem = pathlib.PurePosixPath(name).name.upper()
        if "GSA" in stem:
            assert all(e["contributor"] == "GSA" for e in entries), name


def test_the_alma_files_are_attributed_to_gsa(layouts):
    """Test 1/5 and test 1/6 name their third series ALMA, which is GSA.

    The evidence is in the decoder's docstring: those tests list GS, GSA and
    VK as contributors and name the three temporaries `data_GS.tmp`,
    `data_VK.tmp` and `data_ALMA.tmp`, and test 1/7 keeps the same files in a
    directory called `Gsa`.
    """
    for name in ("h.dat", "l.dat", "h.mod", "l.mod"):
        entries = layouts["by_file"][name]
        assert {e["contributor"] for e in entries} == {"GSA"}, name


def test_test_one_one_is_a_log_log_spectrum(layouts):
    """The relaxation spectrum is drawn on two logarithmic axes.

    Reading either axis as linear turns the spectrum into a straight line, so
    the flags are part of the record.
    """
    panels = [panel for script in layouts["scripts"]
              if script["test"] == "T01-01" for panel in script["panels"]]
    assert panels
    # One test, four spellings of its y-axis and three of its x-axis, spread
    # across the working copy and the PAPER copy:
    #
    #   Relaxation time (yr) / (yrs) / Relaxation time, tau_j_ (yr) / (yrs)
    #   Degree, n / Harmonic degree, n / Harmonic degree, l
    #
    # They are one quantity against one axis. A reader keyed on the label
    # string would treat them as four quantities, which is why a case file
    # names the file it wants and not the label.
    for panel in panels:
        assert panel["quantity"].startswith("Relaxation time")
        assert "degree" in panel["abscissa_label"].lower()
        assert panel["abscissa_log"] is True
        assert panel["ordinate_log"] is True


def test_the_tau_of_the_relaxation_time_label_is_translated():
    """`@~t@~` is tau in GMT's Symbol font, and must not be left as `t`.

    Stripping the font switch without translating the letter turns the
    relaxation time tau_j into `t_j_`, which reads as a time and is a
    different quantity. No decoded label may carry a GMT font escape.
    """
    layouts = json.loads(LAYOUTS.read_text())
    labels = {panel["quantity"] for script in layouts["scripts"]
              for panel in script["panels"] if panel["quantity"]}
    labels |= {panel["abscissa_label"] for script in layouts["scripts"]
               for panel in script["panels"] if panel["abscissa_label"]}
    assert any("tau_j_" in label for label in labels)
    assert not any("@~" in label or "@-" in label for label in labels)


def test_the_relaxation_spectra_are_multisegment(layouts):
    """`psxy -M` says the file holds one segment per harmonic degree.

    A reader that ignores the segment markers concatenates all degrees into
    one column and produces a spectrum with no structure.
    """
    entries = [entry for name, entries in layouts["by_file"].items()
               for entry in entries
               if name.startswith("spec_T01_01_")]
    assert entries
    assert all(entry["multisegment"] for entry in entries)


def test_the_elastic_and_fluid_love_numbers_share_a_panel(layouts):
    """Test 1/2 draws `.ela` and `.flu` on one panel, told apart by symbol.

    Circles are the elastic Love numbers and squares the fluid ones. Without
    the symbol the two are one unlabelled scatter, because the panel carries a
    single axis label for both.
    """
    elastic = layouts["by_file"]["h_L_T01-02_RR.ela"]
    fluid = layouts["by_file"]["h_L_T01-02_RR.flu"]
    assert {e["symbol"] for e in elastic} == {"-Sc0.08"}
    assert {e["symbol"] for e in fluid} == {"-Ss0.08"}
    assert {e["panel"] for e in elastic} & {e["panel"] for e in fluid}


def test_the_two_contributors_of_test_two_one_store_time_differently(layouts):
    """GS writes one file per epoch; VK writes one column per epoch.

    This is the layout difference that a reader has to know and that nothing
    in the filenames says. Checked against `output_sec2disc/README` and the
    header of `u2cost.dat`, which names the epochs 0, 1, 2, 5 and 10 kyr.
    """
    entries = [e for e in layouts["by_file"]["./output_sec2disc/u2cost.dat"]
               if e["test"] == "T02-01"]
    assert {e["contributor"] for e in entries} == {"VK"}
    assert {e["abscissa"]["column"] for e in entries} == {1}
    # Five epochs, in the five columns after the colatitude.
    assert {e["ordinate"]["column"] for e in entries} == {2, 3, 4, 5, 6}

    for epoch in ("disp_t0_disc.dat", "disp_t1_disc.dat", "disp_t1000_disc.dat"):
        per_epoch = [e for e in layouts["by_file"][epoch] if e["test"] == "T02-01"]
        assert {e["contributor"] for e in per_epoch} == {"GS"}
        assert {e["abscissa"]["column"] for e in per_epoch} == {1}


def test_the_gs_column_order_of_test_two_one_is_pinned(layouts):
    """One GS file holds all three quantities, and column 4 is not one of them.

    `disp_t0_cap.dat` gives vertical displacement in column 2, horizontal
    displacement in column 3 and geoid in column 5. Column 4 is skipped by
    every panel of the published figure. Guessing that the three are columns
    2, 3 and 4 gives a geoid curve that is plausible and wrong.
    """
    entries = [e for e in layouts["by_file"]["disp_t0_cap.dat"]
               if e["script"] == "T02-01/plot_PAP.gmt"]
    by_quantity = {}
    for entry in entries:
        heading = next((s for s in entry["sections"]
                        if "displacement" in s or "Geoid" in s), None)
        by_quantity[heading] = entry["ordinate"]["column"]
    assert by_quantity == {
        "Vertical displacement u u u u u u u u u u u u u u": 2,
        "Horizontal displacement v v v v v v v v v v v v v": 3,
        "Geoid height variation n n n n n n n n n n n n n n": 5,
    }, by_quantity


def test_the_paper_and_working_copies_are_both_decoded(layouts):
    """Four tests ship a `PAPER` copy beside the working one.

    Which of the two a case uses is a decision that belongs in the case file.
    Decoding only one of them would make that decision here, invisibly.
    """
    variants = collections.defaultdict(set)
    for script in layouts["scripts"]:
        variants[script["test"]].add(script["variant"])
    for test in ("T01-01", "T01-02", "T01-03", "T01-05"):
        assert "" in variants[test], test
        assert "PAPER" in variants[test], test


def test_the_attribution_disagreements_are_recorded(layouts):
    """Where the path and the banner comment differ, both are kept.

    The path wins in the record, because it names the file itself while the
    banner names the block the read sat in, and a combined panel re-reads
    every contributor's file inside one block. The list is what a person reads
    to check that ruling, so it must not silently empty.
    """
    disagreements = layouts["attribution_disagreements"]
    assert disagreements
    for row in disagreements:
        assert row["path_says"] != row["banner_says"]
        assert row["path_says"] in TABLE_ONE
        assert row["banner_says"] in TABLE_ONE


def test_the_files_with_no_contributor_are_the_expected_ones(layouts):
    """Six files carry no contributor, and none of them is a result.

    Two hold the viscosity profile of an earth model and four hold polar
    motion tables. Anything else appearing here is a result whose author was
    lost, which is worth failing over.
    """
    unattributed = sorted({
        name for name, entries in layouts["by_file"].items()
        if all(entry["contributor"] is None for entry in entries)})
    assert unattributed == [
        "M3L70V01.dat",
        "PM_plot_cap-ncw.dat",
        "PM_plot_cap-ycw.dat",
        "Rot_cwoff_GRID_cap_m.txt",
        "Rot_cwoff_GRID_cap_mdot.txt",
        "vis-vss96.dat",
    ], unattributed


@pytest.mark.data
def test_every_decoded_file_exists_in_the_archive(layouts):
    """A filename the decoder invented would point at nothing.

    Needs the unpacked archive, so it is marked `data` and excluded from the
    default run.
    """
    root = (pathlib.Path(__file__).resolve().parent.parent / "data" / "original"
            / "spada2011" / "spada2011-cost-es0701")
    if not root.is_dir():
        pytest.skip(f"{root} is not unpacked")
    missing = []
    for name, entries in layouts["by_file"].items():
        # The record keeps the name as the script wrote it, which is relative
        # to the directory of the test the script belongs to.
        directories = {(entry["test"], entry["variant"]) for entry in entries}
        if not any((root / test / variant / name).exists()
                   for test, variant in directories):
            missing.append(name)
    assert not missing, missing
