"""The decoded column layouts are complete and self-consistent.

These guard the one piece of knowledge in this repository that cannot be
rebuilt from anywhere else: for the data files with no header, the GMT 4
scripts are the only record of what their columns hold, and GMT 4 no longer
runs.
"""

import json
import pathlib

import pytest

from gia_mip import data

LAYOUTS = pathlib.Path(data.__file__).parent / "resources" / "column_layouts.json"

# The five quantities the Martinec figures plot, after the Greek escapes in
# the GMT labels are translated.
QUANTITIES = {
    "U(theta,t) (m)", "delta U(theta,t) (m)",
    "N(theta,t) (m)", "delta N(theta,t) (m)",
    "S(theta,t) (m)", "delta S(theta,t) (m)",
    "m",
}


@pytest.fixture(scope="module")
def layouts():
    return json.loads(LAYOUTS.read_text())


def test_all_four_scripts_were_decoded(layouts):
    names = {s["script"] for s in layouts["scripts"]}
    assert names == {
        "A_fig10_diff.gmt",
        "figZ_uvf_load_diff.gmt",
        "figZ_uvf_basin_diff.gmt",
        "vega1D_iceB.gmt",
    }


def test_every_panel_is_named_by_a_known_quantity(layouts):
    """A panel named by its abscissa means the y-axis label was missed.

    GMT writes axis settings as -B<x>/<y>, so a basemap line with two label
    references puts the ordinate second. Taking the first gives
    'Co-latitude, theta (deg)' for half the panels, which is the bug this
    catches.
    """
    for script in layouts["scripts"]:
        for panel in script["panels"]:
            assert panel["quantity"] in QUANTITIES, (
                f"{script['script']} panel {panel['panel']}: "
                f"{panel['quantity']!r}")


def test_every_drawn_series_resolves_to_a_real_filename(layouts):
    """An unresolved shell variable means an assignment was missed."""
    for script in layouts["scripts"]:
        for panel in script["panels"]:
            for series in panel["series"]:
                if series["drawn"]:
                    assert not series["file"].startswith("<unresolved"), (
                        f"{script['script']} panel {panel['panel']}: "
                        f"{series['file']}")


def test_every_cross_section_names_exactly_one_ordinate_column(layouts):
    for file_name, entries in layouts["by_file"].items():
        for entry in entries:
            if entry["kind"] != "cross_section":
                continue
            assert "column" in entry["ordinate"], (
                f"{file_name}: ordinate {entry['ordinate']}")


def test_the_one_map_names_three_columns(layouts):
    """vega1D_iceB.gmt draws a map, so it prints longitude, latitude and a value."""
    entries = [e for v in layouts["by_file"].values() for e in v
               if e["kind"] == "map"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["abscissa"]["column"] == 1     # longitude
    assert entry["ordinate"]["column"] == 2     # latitude
    assert entry["value"]["column"] == 5        # the display field


def test_the_geoid_sign_split_is_recorded(layouts):
    """Six of the ten benchmark A contributors store the geoid negated.

    Pinned by name, so that a re-decode which loses a negation fails here
    instead of silently flipping a curve. One file, n_10ka_axi.dat, is both
    negated and divided by gravity, so the two sets below overlap.
    """
    negated = set()
    plain = set()
    for file_name, entries in layouts["by_file"].items():
        for entry in entries:
            if (entry["script"] == "A_fig10_diff.gmt"
                    and entry["quantity"] == "N(theta,t) (m)"
                    and entry["drawn_in_the_published_figure"]):
                (negated if entry["ordinate"]["negated"] else plain).add(file_name)
    assert negated == {
        "A_fig10_KS_256.dat", "A_fig10_SBK.dat", "case_A_VILMA_uve.dat",
        "fig10_n_VB_256.dat", "n_10ka.dat", "n_10ka_axi.dat",
    }, sorted(negated)
    assert plain == {
        "A-fig10-TABOO_256.dat", "A_fig10_YS.dat",
        "caseA_0.25.lmax256_GA.dat", "Z_ZM_load.out",
    }, sorted(plain)


def test_the_potential_to_geoid_division_is_recorded(layouts):
    """Two benchmark A series hold potential and are divided by gravity."""
    divided = {
        file_name
        for file_name, entries in layouts["by_file"].items()
        for entry in entries
        if entry["script"] == "A_fig10_diff.gmt"
        and "9.8155" in str(entry["ordinate"].get("scale"))
    }
    assert divided == {"Z_ZM_load.out", "n_10ka_axi.dat"}, sorted(divided)
