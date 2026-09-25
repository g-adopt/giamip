"""The reference curves of the Martinec cases, as the converter builds them.

The converter writes a group ``reference`` into each converted Martinec file,
following ``reference_data.reference_curves`` of the case file: VEGA as
shipped, or VEGA recovered as YS's profile minus YS's own difference file.
These tests check that every compared quantity has a reference curve, that
each stored curve is exactly what its rule says, and that the recovered
curves are the VEGA profile every contributor's difference file was computed
against: a cubic spline through the stored curve reproduces the recovery of
every other contributor, within twice the maximum measured for the plan
(NOTES/team/pypi-package/00-plan.md, section 1.3).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from giamip import cases
from giamip.benchmarks import martinec2018
from giamip.benchmarks.inputs import MartinecCase
from giamip.benchmarks.results import BenchmarkResult

LETTERS = ("A", "B", "C", "D", "E")

# The largest difference, in metres, between a cubic spline through the stored
# recovered reference and the recovery of any other contributor, over all of
# that contributor's points including colatitude 0 and 180. Measured for the
# plan and again by the numerics review (rounds 3 to 5). The largest values
# come from VK and WW_axisymmetric in A and from SK in D, whose own files carry
# the difference. The test allows twice each value.
MEASURED_MAXIMUM_M = {
    ("A", "load", "U"): 1.7e-4, ("A", "load", "N"): 5.9e-5,
    ("B", "basin", "U"): 3.7e-6, ("B", "basin", "N"): 6.0e-6, ("B", "basin", "S"): 8.7e-7,
    ("D", "load", "U"): 3.2e-6, ("D", "load", "N"): 1.3e-5,
    ("D", "basin", "U"): 8.6e-7, ("D", "basin", "N"): 2.0e-6, ("D", "basin", "S"): 2.0e-7,
}

# How many contributors have both a profile and a difference file for each
# recovered profile and quantity (YS and every ZM member left out). SK in case
# A has difference files and no profile, so it is not counted; WW has no S in
# the case B basin.
EXPECTED_MEMBERS = {
    ("A", "load", "U"): 7, ("A", "load", "N"): 7,
    ("B", "basin", "U"): 7, ("B", "basin", "N"): 7, ("B", "basin", "S"): 6,
    ("D", "load", "U"): 3, ("D", "load", "N"): 3,
    ("D", "basin", "U"): 3, ("D", "basin", "N"): 3, ("D", "basin", "S"): 3,
}


def _rules(letter: str) -> dict:
    return cases.case(f"martinec2018-{letter}")["reference_data"]["reference_curves"]


def _recovery(result: BenchmarkResult, profile: str, quantity: str):
    """A contributor's profile minus its difference file, on the difference file's grid.

    Where the two grids differ (SK in B, C and D: 361 profile rows, 360
    difference rows, the difference grid a subset of the profile grid), the
    profile is interpolated linearly onto the difference grid.
    """
    own = result.profiles[(profile, quantity)]
    difference = result.differences[(profile, quantity)]
    if own.axis.shape == difference.axis.shape and np.allclose(own.axis, difference.axis,
                                                               atol=1e-9):
        values = own.values
    else:
        values = np.interp(difference.axis, own.axis, own.values)
    return difference.axis, values - difference.values


@pytest.mark.parametrize("letter", LETTERS)
def test_every_compared_quantity_has_a_rule_and_a_reference_curve(letter):
    """The case file's comparison lists decide what the reference group must hold."""
    setup = MartinecCase(letter)
    rules = _rules(letter)
    reference = BenchmarkResult.reference(letter)
    assert reference.member == "VEGA"
    for spec in setup.profiles:
        assert spec.name in rules["profiles"], (letter, spec.name)
        for quantity in spec.quantities:
            curve = reference.profile(spec.name, quantity)
            assert curve.attributes["method"] == rules["profiles"][spec.name]["method"]
            assert curve.attributes["member"] == rules["profiles"][spec.name]["member"]
            assert curve.attributes["built_from"]
    for spec in setup.time_series:
        assert "time_series" in rules, letter
        for quantity in spec.quantities:
            assert reference.series(quantity).attributes["method"] == "as shipped"


@pytest.mark.parametrize("letter", LETTERS)
def test_each_stored_curve_is_what_its_rule_says(letter):
    """Shipped curves equal ZM exactly; recovered ones equal YS minus YS's difference."""
    rules = _rules(letter)
    reference = BenchmarkResult.reference(letter)
    for (profile, quantity), curve in reference.profiles.items():
        rule = rules["profiles"][profile]
        member = BenchmarkResult.reference(letter, rule["member"])
        if rule["method"] == "as shipped":
            source = member.profile(profile, quantity)
            np.testing.assert_array_equal(curve.axis, source.axis)
            np.testing.assert_array_equal(curve.values, source.values)
        else:
            own = member.profiles[(profile, quantity)]
            difference = member.differences[(profile, quantity)]
            np.testing.assert_array_equal(curve.axis, difference.axis)
            assert np.max(np.abs(curve.values - (own.values - difference.values))) < 1e-12
    for quantity, curve in reference.time_series.items():
        source = BenchmarkResult.reference(letter, "ZM").series(quantity)
        np.testing.assert_array_equal(curve.axis, source.axis)
        np.testing.assert_array_equal(curve.values, source.values)


@pytest.mark.parametrize("key", sorted(MEASURED_MAXIMUM_M), ids=lambda k: "-".join(k))
def test_a_recovered_reference_reproduces_every_other_recovery(key):
    """The stored curve is the VEGA profile every difference file was computed against.

    A cubic spline (not-a-knot ends) through the stored curve, evaluated on 0
    to 180 degrees, reproduces each other contributor's recovery at that
    contributor's own points. The bound is twice the measured maximum.
    """
    letter, profile, quantity = key
    stored = BenchmarkResult.reference(letter).profile(profile, quantity)
    spline = CubicSpline(stored.axis, stored.values)
    path = BenchmarkResult.reference_path(letter)
    found = {}
    for member in BenchmarkResult.members(path):
        if member == "YS" or member.startswith("ZM"):
            continue
        result = BenchmarkResult.read(path, member, case=letter)
        if (profile, quantity) not in result.profiles or (profile, quantity) not in result.differences:
            continue
        axis, recovered = _recovery(result, profile, quantity)
        found[member] = float(np.max(np.abs(spline(axis) - recovered)))
    assert len(found) == EXPECTED_MEMBERS[key], (key, sorted(found))
    bound = 2.0 * MEASURED_MAXIMUM_M[key]
    assert max(found.values()) <= bound, (key, bound, found)


def test_a_rule_that_names_an_absent_member_raises(monkeypatch):
    """A reference_curves entry the archive cannot satisfy stops the conversion."""
    definition = cases.case("martinec2018-A")
    definition["reference_data"]["reference_curves"]["profiles"]["load"]["member"] = "XX"
    monkeypatch.setattr(cases, "case", lambda case_id: definition)
    group = martinec2018.Group("colatitude", "degrees", "colatitude")
    group.add("U", np.array([0.0, 1.0]), np.array([1.0, 2.0]),
              {"source_file": "f", "source_column": 2})
    with pytest.raises(ValueError, match="does not hold"):
        martinec2018.reference_groups("A", {"profiles/load/YS": group,
                                            "differences/load/YS": group})


def test_a_compared_profile_without_a_rule_raises(monkeypatch):
    """A compared profile with no reference_curves entry stops the conversion."""
    definition = cases.case("martinec2018-B")
    del definition["reference_data"]["reference_curves"]["profiles"]["load"]
    monkeypatch.setattr(cases, "case", lambda case_id: definition)
    with pytest.raises(ValueError, match="no entry"):
        martinec2018.reference_groups("B", {})
