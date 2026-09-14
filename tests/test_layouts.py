"""The decoded column layouts are complete and self-consistent.

These guard the one piece of knowledge in this repository that cannot be
rebuilt from anywhere else: for the data files with no header, the GMT 4
scripts are the only record of what their columns hold, and GMT 4 no longer
runs.

Every file in the layout table is named by its path relative to
``data/original/``. The same filename, for example ``Z_ZM_load.out``, occurs
in six directories with six different contents, so a name alone is ambiguous.
"""

import json
import pathlib

import pytest

from gia_mip import corrections, data
from gia_mip.benchmarks import martinec2018

LAYOUTS = pathlib.Path(data.__file__).parent / "resources" / "column_layouts.json"

# The quantities the Martinec figures plot, after the Greek escapes in the GMT
# labels are translated. "m" is the colour bar of the one map.
QUANTITIES = {
    "U(theta,t) (m)", "delta U(theta,t) (m)",
    "N(theta,t) (m)", "delta N(theta,t) (m)",
    "S(theta,t) (m)", "delta S(theta,t) (m)",
    "huf (m)", "Ice mass (10^17 kg)", "Ocean area (% of 4pi)",
}

CASE_A_SCRIPTS = ("martinec2018/resultsA_2026/figZ_uvf_load_diff.gmt",)


@pytest.fixture(scope="module")
def layouts():
    return json.loads(LAYOUTS.read_text())


def test_every_script_was_decoded(layouts):
    names = {s["script"] for s in layouts["scripts"]}
    assert names == {
        "martinec2018/resultsA_2026/figZ_uvf_load_diff.gmt",
        "martinec2018/resultsB_2026/figZ_uvf_load_diff.gmt",
        "martinec2018/resultsB_2026/figZ_uvf_basin_diff.gmt",
        "martinec2018/resultsC_2026/figZ_uvf_load_diff.gmt",
        "martinec2018/resultsC_2026/figZ_uvf_basin_diff.gmt",
        "martinec2018/resultsC_2026/fig_hufC.gmt",
        "martinec2018/resultsC_2026/fig_mass_iceC.gmt",
        "martinec2018/resultsD_2026/figZ_uvf_load_diff.gmt",
        "martinec2018/resultsD_2026/figZ_uvf_basin_diff.gmt",
        "martinec2018/resultsD_2026/fig_hufD_secti_detail.gmt",
        "martinec2018/resultsD_2026/fig_mass_iceD.gmt",
        "martinec2018/resultsD_2026/fig_of00D.gmt",
        "martinec2018/resultsE_2026/fig_hufE.gmt",
        "martinec2018/resultsE_2026/fig_of00E.gmt",
    }


def test_files_are_named_by_path(layouts):
    """A bare filename as a key would merge six different Z_ZM_load.out files."""
    for key in layouts["by_file"]:
        assert key.startswith("martinec2018/") or key.startswith("<unresolved"), key


def test_every_panel_is_named_by_a_known_quantity(layouts):
    """A panel named by its abscissa means the y-axis label was missed.

    GMT writes axis settings as -B<x>/<y>, so a basemap line with two label
    references puts the ordinate second. Taking the first gives
    'Co-latitude, theta (deg)' or 'Time (kyr)' for half the panels, which is
    the bug this catches.
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
                    assert series["contributor"], (
                        f"{script['script']}: no contributor for {series['variable']}")


def test_every_cross_section_names_exactly_one_ordinate_column(layouts):
    for file_name, entries in layouts["by_file"].items():
        for entry in entries:
            if entry["kind"] not in ("cross_section", "time_series"):
                continue
            assert "column" in entry["ordinate"], (
                f"{file_name}: ordinate {entry['ordinate']}")


def test_each_directory_is_the_case_its_figure_titles_name(layouts):
    """The title is the evidence tying D3, E2 and F1 to published C, D and E.

    Only the cross-section scripts carry a live title. The time-series
    scripts of cases C and D keep a commented-out title copied from another
    script, and case E's scripts label an overlaid case D curve, so neither
    counts as evidence.
    """
    for script in layouts["scripts"]:
        if "figZ_uvf" not in script["script"]:
            continue
        drawn = [t["text"] for t in script["titles"] if t["drawn"]]
        assert len(drawn) == 1, script["script"]
        assert drawn[0].startswith(f"Benchmark {script['case']} "), (
            script["script"], drawn)


@pytest.mark.parametrize("script_path", CASE_A_SCRIPTS)
def test_the_geoid_sign_split_is_recorded(layouts, script_path):
    """Six of the ten benchmark A contributors store the geoid negated.

    Pinned by name, so that a re-decode which loses a negation fails here
    instead of silently flipping a curve. One file, n_10ka_axi.dat, is both
    negated and divided by gravity, so the two sets below overlap.
    """
    negated = set()
    plain = set()
    for file_name, entries in layouts["by_file"].items():
        for entry in entries:
            if (entry["script"] == script_path
                    and entry["quantity"] == "N(theta,t) (m)"
                    and entry["drawn_in_the_published_figure"]):
                name = file_name.rsplit("/", 1)[-1]
                (negated if entry["ordinate"]["negated"] else plain).add(name)
    assert negated == {
        "A_fig10_KS_256.dat", "A_fig10_SBK.dat", "case_A_VILMA_uve.dat",
        "fig10_n_VB_256.dat", "n_10ka.dat", "n_10ka_axi.dat",
    }, sorted(negated)
    assert plain == {
        "A-fig10-TABOO_256.dat", "A_fig10_YS.dat",
        "caseA_0.25.lmax256_GA.dat", "Z_ZM_load.out",
    }, sorted(plain)


@pytest.mark.parametrize("script_path", CASE_A_SCRIPTS)
def test_the_potential_to_geoid_division_is_recorded(layouts, script_path):
    """Two benchmark A series hold potential and are divided by gravity."""
    divided = {
        file_name.rsplit("/", 1)[-1]
        for file_name, entries in layouts["by_file"].items()
        for entry in entries
        if entry["script"] == script_path
        and "9.8155" in str(entry["ordinate"].get("scale"))
    }
    assert divided == {"Z_ZM_load.out", "n_10ka_axi.dat"}, sorted(divided)


def test_the_case_c_basin_script_divides_vega_by_ten(layouts):
    """The one script that uses 10 in place of 9.8155; see the correction.

    Pinned so that the decode keeps recording what the script does, and the
    reader keeps overriding it. Every other VEGA geoid series uses 9.8155.
    """
    factors = {}
    for key, entries in layouts["by_file"].items():
        if not key.endswith("/Z_ZM_load.out") and not key.endswith("/Z_ZM_basin.out"):
            continue
        for entry in entries:
            if entry["quantity"] == "N(theta,t) (m)":
                factors[(entry["script"], key)] = entry["ordinate"]["scale"]
    odd = {k: v for k, v in factors.items() if v != "x/9.8155"}
    assert odd == {("martinec2018/resultsC_2026/figZ_uvf_basin_diff.gmt",
                    "martinec2018/resultsC_2026/Z_ZM_basin.out"): "x/10"}


def test_the_ocean_area_scale_differs_by_contributor(layouts):
    """Five contributors, three storage units, one plotted percentage."""
    script = next(s for s in layouts["scripts"]
                  if s["script"] == "martinec2018/resultsD_2026/fig_of00D.gmt")
    scales = {x["contributor"]: x["y"]["scale"]
              for p in script["panels"] for x in p["series"] if x["drawn"]}
    assert scales == {"ZM": "x*100", "GS": None, "VK": None, "YS": "x*10", "SK": "x*100"}


def test_the_case_d_water_layer_inset_is_its_own_panel(layouts):
    """The comparison of h_UF between codes is drawn as an inset.

    Its basemap has empty labels, so without inheriting them it would be an
    unnamed panel, and without the region test it would merge with the main
    panel, which shows VEGA's components and no other code.
    """
    script = next(s for s in layouts["scripts"]
                  if s["script"] == "martinec2018/resultsD_2026/fig_hufD_secti_detail.gmt")
    main, inset = script["panels"]
    assert not main.get("inset") and inset["inset"]
    assert inset["quantity"] == "huf (m)"
    assert {x["contributor"] for x in main["series"] if x["drawn"]} == {"ZM"}
    assert {x["contributor"] for x in inset["series"] if x["drawn"]} == {"ZM", "GS", "YS", "SK"}


def _drawn_time_series(layouts):
    for script in layouts["scripts"]:
        for panel in script["panels"]:
            for series in panel["series"]:
                if series["drawn"] and series["kind"] == "time_series":
                    yield script, panel, series


def test_every_drawn_time_series_has_a_transcribed_unit(layouts):
    """The scripts give the column; the unit has to come from the table."""
    table = layouts["time_series_columns"]
    for script, panel, series in _drawn_time_series(layouts):
        entry = table.get(series["path"])
        assert entry, f"{script['script']}: no entry for {series['path']}"
        assert entry["time_column"] == series["x"]["column"], series["path"]
        assert str(series["y"]["column"]) in entry["columns"], (
            f"{series['path']} column {series['y']['column']}")


def test_transcribed_units_agree_with_the_plotting_scale(layouts):
    """si_factor must equal the script's scale times the unit of the axis.

    This ties every hand-transcribed unit to what the published figure shows.
    If YS's ocean area were recorded as a fraction, its si_factor would be 1
    while the script's factor of 10 and the per-cent axis give 0.1, and this
    test would name the file.
    """
    table = layouts["time_series_columns"]
    for script, panel, series in _drawn_time_series(layouts):
        entry = table[series["path"]]
        column = entry["columns"][str(series["y"]["column"])]
        axis = martinec2018.TIME_SERIES_AXES[panel["quantity"]]["si_factor"]
        expected = martinec2018.scale_factor(series["y"]["scale"]) * axis
        # awk drops a D exponent, so the figure can be right while the file is
        # 10^n smaller. Allowed only with the correction that documents it.
        if entry.get("awk_exponent_dropped"):
            ids = {e["id"] for e in corrections.applying_to(series["path"])}
            assert "martinec2018-ys-case-e-awk-exponent" in ids, series["path"]
            expected *= 10.0 ** entry["awk_exponent_dropped"]
        assert not series["y"]["negated"], series["path"]
        assert column["si_factor"] == pytest.approx(expected, rel=1e-12), (
            f"{script['script']}: {series['path']} column {series['y']['column']} "
            f"has si_factor {column['si_factor']}, the figure implies {expected}")
