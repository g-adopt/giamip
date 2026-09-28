"""The G-ADOPT choices for the open fields of the GIAMIP experiments.

The case files are the single record of these choices. Each GIAMIP case file
holds the protocol values and, in a `gadopt` block, the values the G-ADOPT team
chose for the fields that the protocol leaves open. The documentation pages
build their tables from the case files through `docs/_experiments.py`. These
tests keep the choices inside the open fields, keep every choice pointing at
something that exists, and keep the pages from carrying a typed copy.
"""

import pathlib
import sys

import pytest

from giamip import cases

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "docs"))
import _experiments  # noqa: E402

# The rheologies a choice may name. Each one needs a label on the protocol
# page, which the last test checks.
RHEOLOGIES = {"maxwell", "burgers", "power-law", "burgers-power-law"}

GIAMIP = cases.cases("giamip")
WITH_CHOICES = [c for c in GIAMIP if cases.gadopt_choices(c)]


def test_the_expected_experiments_carry_choices():
    """Exp01, Exp05, Exp06 and Exp07 leave nothing open, so they have none."""
    assert {c["experiment_id"] for c in WITH_CHOICES} == {
        "Exp02", "Exp03", "Exp04", "Exp08", "Exp09", "Exp10", "Exp11", "Exp12"}


@pytest.mark.parametrize("c", WITH_CHOICES, ids=lambda c: c["id"])
def test_a_choice_fills_only_a_field_the_protocol_leaves_open(c):
    """A G-ADOPT value must never overwrite a value the protocol prescribes."""
    for field in cases.gadopt_choices(c):
        assert field in cases.GADOPT_FIELDS, (c["id"], field)
        assert field in c["open_choices"], (c["id"], field)


@pytest.mark.parametrize("c", WITH_CHOICES, ids=lambda c: c["id"])
def test_every_choice_names_something_that_exists(c):
    chosen = cases.gadopt_choices(c)
    for field in ("viscosity", "elastic_structure"):
        if field in chosen:
            assert cases.earth_model(chosen[field])["id"] == chosen[field]
    if "rheology" in chosen:
        assert chosen["rheology"] in RHEOLOGIES
    if "topography" in chosen:
        assert chosen["topography"] in cases.topography_files()


@pytest.mark.parametrize("c", WITH_CHOICES, ids=lambda c: c["id"])
def test_every_choice_records_why(c):
    assert c["gadopt"].get("$comment"), c["id"]


def test_the_case_file_keeps_the_protocol_values():
    """The `gadopt` block adds values; the `earth` block is unchanged."""
    c = cases.case("giamip-exp03")
    assert c["earth"]["viscosity"] == "giamip-open-range"
    assert c["earth"]["rheology"] == "open"


def test_with_gadopt_choices_fills_the_open_fields():
    exp09 = cases.with_gadopt_choices("giamip-exp09")
    exp03 = cases.case("giamip-exp03")
    assert exp09["earth"]["rheology"] == "burgers"
    assert exp09["earth"]["viscosity"] == "giamip-open-range-midpoint"
    # The topography file comes from a case that the protocol gives bedmap3.
    assert exp09["topography"]["file"] == exp03["topography"]["file"]
    assert exp09["open_choices"] == []


def test_with_gadopt_choices_does_not_change_the_stored_case():
    cases.with_gadopt_choices("giamip-exp12")
    assert cases.case("giamip-exp12")["earth"]["viscosity"] == "open"


def test_the_midpoint_model_is_the_middle_of_the_range():
    """Log-scale middle for the viscosities, linear middle for the lithosphere."""
    r = cases.earth_model("giamip-open-range")
    m = cases.earth_model("giamip-open-range-midpoint")
    t = r["lithosphere_thickness_km"]
    assert m["lithosphere_thickness_km"] == (t["min"] + t["max"]) / 2
    for key in ("upper_mantle_viscosity_pa_s", "lower_mantle_viscosity_pa_s"):
        geometric_mean = (r[key]["min"] * r[key]["max"]) ** 0.5
        assert m[key] == pytest.approx(geometric_mean, rel=1e-12)


def test_no_page_types_a_red_value():
    """The red cells come from `docs/_experiments.py` only.

    A typed red value in a page would be a second copy of a choice, and it
    would go out of date without an error the first time the case file changes.
    """
    for page in (ROOT / "docs").rglob("*.qmd"):
        assert "color: red" not in page.read_text(), page


def test_the_protocol_table_shows_every_choice_in_red():
    table = _experiments.protocol_table()
    rows = [r for r in table.splitlines()[2:]]
    assert len(rows) == len(GIAMIP)
    n_choices = sum(len(cases.gadopt_choices(c)) for c in GIAMIP)
    assert table.count('{style="color: red;"}') == n_choices


def test_the_id_table_shows_every_choice_in_red():
    table = _experiments.id_table(link=True, open_count=True)
    n_choices = sum(len(cases.gadopt_choices(c)) for c in GIAMIP)
    assert table.count('{style="color: red;"}') == n_choices


def test_every_rheology_and_topography_has_a_protocol_label():
    for c in GIAMIP:
        values = {**{f: cases.protocol_value(c, f) for f in cases.GADOPT_FIELDS},
                  **cases.gadopt_choices(c)}
        assert values["rheology"] in _experiments.RHEOLOGY_LABELS, c["id"]
        assert values["topography"] in _experiments.TOPOGRAPHY_LABELS, c["id"]
