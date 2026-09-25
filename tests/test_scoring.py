"""The one scoring path for the Martinec cases, and the case entry point.

`giamip.benchmarks.metrics.score` compares a profile on the result's own
points, with the stored reference evaluated there by a cubic spline. The
strongest check of that rule is the archive itself: every contributor, scored
this way, must get the maximum and RMS of its own published difference file.
"""

from __future__ import annotations

import numpy as np
import pytest

import giamip
from giamip.benchmarks import metrics
from giamip.benchmarks.inputs import MartinecCase, SpadaCase
from giamip.benchmarks.results import BenchmarkResult, Curve
from giamip.mip.inputs import GiamipExperiment

# Bounds on |score - published difference file|, in metres. In A the
# reference is continued to the pole and VK differs from the other codes by
# 1.6e-4 m there; the numerics review measured at most 4.2e-5 m (A) and
# 3.7e-5 m (B, C, D) in a maximum or an RMS.
CONTRIBUTOR_BOUND_M = {"A": 2e-4, "B": 1e-4, "C": 1e-4, "D": 1e-4}


def _own_grid_members(letter: str):
    """Every contributor whose profile and difference file are on one grid.

    Such a contributor's difference file is known at exactly the points it
    would be scored on. SK (361 profile rows, 360 difference rows) is left out.
    """
    path = BenchmarkResult.reference_path(letter)
    for member in BenchmarkResult.members(path):
        if member.startswith("ZM"):
            continue
        result = BenchmarkResult.read(path, member, case=letter)
        pairs = [key for key in result.differences if key in result.profiles
                 and result.profiles[key].axis.shape == result.differences[key].axis.shape
                 and np.allclose(result.profiles[key].axis, result.differences[key].axis,
                                 atol=1e-9)]
        if pairs:
            yield member, result, pairs


def _published(difference: Curve, interval) -> tuple[float, float]:
    """Maximum and RMS of a difference file inside an interval."""
    inside = (difference.axis >= interval[0]) & (difference.axis <= interval[1])
    values = difference.values[inside]
    return float(np.max(np.abs(values))), float(np.sqrt(np.mean(values ** 2)))


# --------------------------------------------------------------- the rule

@pytest.mark.parametrize("letter", ["A", "B", "C", "D"])
def test_every_contributor_scores_its_own_published_difference_file(letter):
    compared = 0
    worst = 0.0
    for member, result, pairs in _own_grid_members(letter):
        rows = {(r["profile"], r["quantity"]): r for r in metrics.score(result)
                if r["kind"] == "profile"}
        for key in pairs:
            if key not in rows:
                continue
            row = rows[key]
            published_max, published_rms = _published(result.differences[key], row["interval"])
            error = max(abs(row["max"] - published_max), abs(row["rms"] - published_rms))
            worst = max(worst, error)
            assert error <= CONTRIBUTOR_BOUND_M[letter], (letter, member, key, row, error)
            compared += 1
    assert compared > 0, f"case {letter}: no contributor was compared"


@pytest.mark.parametrize("letter,profile", [("A", "load"), ("B", "basin"),
                                            ("D", "load"), ("D", "basin")])
def test_ys_scores_its_own_difference_file_exactly_on_the_recovered_profiles(letter, profile):
    """The reference is YS minus YS's difference, and the spline passes its own knots."""
    ys = BenchmarkResult.reference(letter, "YS")
    for row in metrics.score(ys):
        if row["kind"] != "profile" or row["profile"] != profile:
            continue
        published_max, published_rms = _published(ys.differences[(profile, row["quantity"])],
                                                  row["interval"])
        assert abs(row["max"] - published_max) < 1e-12
        assert abs(row["rms"] - published_rms) < 1e-12


@pytest.mark.parametrize("letter", ["A", "B", "C", "D", "E"])
def test_the_reference_scores_zero_and_a_constant_offset_scores_the_constant(letter):
    reference = BenchmarkResult.reference(letter)
    rows = metrics.score(reference)
    assert rows, letter
    assert all(r["max"] == 0.0 and r["rms"] == 0.0 for r in rows)
    # The offset is 0.01 m on the profiles and 1e-3 of the largest value on a
    # time series: an ice mass is about 1e18 kg, where 0.01 is below the
    # resolution of a float64.
    offsets = {}
    shifted = BenchmarkResult(letter, "shifted")
    for (profile, quantity), curve in reference.profiles.items():
        shifted.add_profile(profile, quantity, curve.axis, curve.values + 0.01)
        offsets[(profile, quantity)] = 0.01
    for quantity, curve in reference.time_series.items():
        offset = 1e-3 * float(np.max(np.abs(curve.values)))
        shifted.add_time_series(quantity, curve.axis, curve.values + offset)
        offsets[(None, quantity)] = offset
    for row in metrics.score(shifted):
        offset = offsets[(row["profile"], row["quantity"])]
        assert row["max"] == pytest.approx(offset, rel=1e-9), row
        assert row["rms"] == pytest.approx(offset, rel=1e-9), row
        assert row["compared_on"] == "result points"
        assert row["giamip_version"] == giamip.__version__


@pytest.mark.parametrize("letter,profile", [("B", "basin"), ("D", "load"), ("D", "basin")])
def test_the_shipped_vega_rows_score_as_samples_of_the_reference(letter, profile):
    """The 359 shipped rows are exact samples of the finer profile: no error to 1e-4 m."""
    shipped = BenchmarkResult.reference(letter, "ZM")
    result = BenchmarkResult(letter, "shipped")
    for (name, quantity), curve in shipped.profiles.items():
        if name == profile:
            result.add_profile(name, quantity, curve.axis, curve.values)
    rows = [r for r in metrics.score(result) if r["profile"] == profile]
    assert rows
    for row in rows:
        assert row["max"] < 1e-4, row


def test_a_result_row_at_the_pole_is_compared_in_case_a():
    """The A reference starts at 0.044 degrees; the spline reaches the pole."""
    reference = BenchmarkResult.reference("A")
    axis = np.linspace(0.0, 20.0, 21)
    curve = reference.profile("load", "U")
    from scipy.interpolate import CubicSpline
    result = BenchmarkResult("A", "grid")
    result.add_profile("load", "U", axis, CubicSpline(curve.axis, curve.values)(axis))
    row = next(r for r in metrics.score(result) if r["quantity"] == "U")
    assert row["n"] == 21 and row["covered"][0] == 0.0
    assert row["max"] < 1e-12


def test_a_result_outside_the_meridian_raises():
    result = BenchmarkResult("A", "bad")
    result.add_profile("load", "U", [-1.0, 5.0, 20.0], [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="outside 0 to 180"):
        metrics.score(result)


def test_a_result_that_stops_short_of_the_interval_raises():
    result = BenchmarkResult("B", "short")
    result.add_profile("load", "U", np.linspace(5.0, 30.0, 26), np.zeros(26))
    with pytest.raises(ValueError, match="covers"):
        metrics.score(result)


def test_every_row_reports_its_largest_point_spacing():
    reference = BenchmarkResult.reference("C")
    curve = reference.profile("load", "U")
    coarse = BenchmarkResult("C", "coarse")
    axis = np.arange(5.0, 45.5, 0.5)
    from scipy.interpolate import CubicSpline
    coarse.add_profile("load", "U", axis, CubicSpline(curve.axis, curve.values)(axis))
    row = metrics.score(coarse)[0]
    assert row["max_spacing"] == pytest.approx(0.5)


# ------------------------------------------------------------ uniform layer

def test_uniform_layer_of_b_reads_the_rebuilt_reference():
    """The shipped 359 rows give -119.896768879 m and a spread of 7.6e-6 m; both checks fail for them."""
    mean, spread = metrics.uniform_layer("B")
    assert mean == pytest.approx(-119.896770, abs=1e-7)
    assert spread < 1e-6


def test_uniform_layer_of_d_equals_the_vega_series():
    mean, spread = metrics.uniform_layer("D")
    series = BenchmarkResult.reference("D").series("h_UF")
    assert mean == pytest.approx(float(np.interp(15.0, series.axis, series.values)), abs=1e-8)
    assert spread < 1e-6


def test_uniform_layer_of_c_equals_the_vega_series():
    mean, _ = metrics.uniform_layer("C")
    series = BenchmarkResult.reference("C").series("h_UF")
    assert mean == pytest.approx(float(np.interp(15.0, series.axis, series.values)), abs=1e-6)


@pytest.mark.parametrize("letter", ["A", "E"])
def test_uniform_layer_is_not_defined_without_a_basin_profile(letter):
    with pytest.raises(ValueError):
        metrics.uniform_layer(letter)


# ------------------------------------------------------------ criterion 2

def test_feature_colatitudes_come_from_the_geometry():
    assert MartinecCase("B").feature_colatitudes("load") == (15.0, 35.0)
    assert MartinecCase("B").feature_colatitudes("basin") == pytest.approx((75.1497, 124.8503), abs=1e-4)
    for letter in ("C", "D"):
        assert MartinecCase(letter).feature_colatitudes("load") == (15.0, 35.0)
        assert MartinecCase(letter).feature_colatitudes("basin") == pytest.approx(
            (10.1497, 59.8503), abs=1e-4)
    assert MartinecCase("A").feature_colatitudes("load") == (10.0,)


def test_the_contributor_envelope_on_a_synthetic_file(tmp_path, monkeypatch):
    """Two members, case B basin (65 to 135 degrees, coast at 75.1497 and 124.8503)."""
    axis = np.arange(60.0, 141.0, 1.0)
    path = tmp_path / "martinec2018-B.nc"
    first = BenchmarkResult("B", "AA")
    # 5 m at colatitude 75, inside the band, removed; 0.3 m at 100, kept.
    values = np.where(axis == 75.0, 5.0, np.where(axis == 100.0, 0.3, 0.1))
    first.differences[("basin", "U")] = Curve(axis, values)
    first.write(path)
    second = BenchmarkResult("B", "BB")
    second.differences[("basin", "U")] = Curve(axis, np.full(axis.size, 0.2))
    second.write(path)
    vega = BenchmarkResult("B", "ZM")
    vega.differences[("basin", "U")] = Curve(axis, np.full(axis.size, 9.0))
    vega.write(path)
    monkeypatch.setattr(BenchmarkResult, "reference_path", staticmethod(lambda case: path))
    envelope = metrics.contributor_envelope("B", "basin", "U", band_half_width_deg=2.0)
    assert sorted(envelope["members"]) == ["AA", "BB"]
    # Kept: 65 to 135, less |x - 75.1497| <= 2 (74, 75, 76, 77) and
    # |x - 124.8503| <= 2 (123, 124, 125, 126): 71 - 8 = 63 points.
    keep = (axis >= 65) & (axis <= 135) & (np.abs(axis - 75.1497) > 2) & (np.abs(axis - 124.8503) > 2)
    assert envelope["members"]["AA"]["n"] == keep.sum() == 63
    assert envelope["members"]["AA"]["max"] == pytest.approx(0.3)
    expected_rms = np.sqrt(np.mean(values[keep] ** 2))
    assert envelope["members"]["AA"]["rms"] == pytest.approx(expected_rms)
    assert envelope["max"] == pytest.approx(0.3)
    assert envelope["rms"] == pytest.approx(max(expected_rms, 0.2))


def test_the_reference_is_inside_the_envelope_with_zero():
    rows = metrics.envelope_check(BenchmarkResult.reference("B"), band_half_width_deg=6.74)
    assert len(rows) == 5
    assert all(r["inside"] and r["max"] == 0.0 for r in rows)


def test_a_result_without_a_compared_curve_fails_criterion_2_loudly():
    reference = BenchmarkResult.reference("C")
    partial = BenchmarkResult("C", "partial")
    for (profile, quantity), curve in reference.profiles.items():
        if (profile, quantity) != ("basin", "N"):
            partial.add_profile(profile, quantity, curve.axis, curve.values)
    with pytest.raises(KeyError, match="basin/N"):
        metrics.envelope_check(partial, band_half_width_deg=6.74)


def test_the_band_width_has_no_default():
    with pytest.raises(TypeError):
        metrics.envelope_check(BenchmarkResult.reference("C"))
    with pytest.raises(TypeError):
        metrics.contributor_envelope("C", "basin", "U")


# --------------------------------------------------------- the entry point

def test_giamip_case_returns_the_class_of_its_collection():
    assert isinstance(giamip.case("martinec2018-B"), MartinecCase)
    assert isinstance(giamip.case("spada2011-t02-01"), SpadaCase)
    assert isinstance(giamip.case("giamip-exp01"), GiamipExperiment)
    with pytest.raises(KeyError, match="Known ids"):
        giamip.case("martinec2018-Z")


def test_a_case_gives_its_reference_and_its_reference_colatitudes():
    case = giamip.case("martinec2018-B")
    assert case.id == "martinec2018-B"
    assert case.reference().member == "VEGA"
    assert case.reference_colatitudes("basin").size == 2048


def test_a_case_refuses_to_score_a_result_of_another_case():
    with pytest.raises(ValueError, match="case C"):
        giamip.case("martinec2018-B").score(BenchmarkResult.reference("C"))


def _reference_plus(letter: str, change) -> BenchmarkResult:
    """The reference with one profile curve changed by ``change(axis, values)``."""
    reference = BenchmarkResult.reference(letter)
    result = BenchmarkResult(letter, "changed")
    for (profile, quantity), curve in reference.profiles.items():
        values = curve.values
        if (profile, quantity) == ("load", "U"):
            values = change(curve.axis, curve.values)
        result.add_profile(profile, quantity, curve.axis, values)
    return result


def test_criterion_2_removes_the_band_on_the_result_side_too():
    """A 1 m error inside the band around the cap margin at 35 degrees is not counted."""
    result = _reference_plus("B", lambda x, v: v + np.where(np.abs(x - 35.0) < 3.0, 1.0, 0.0))
    row = next(r for r in metrics.envelope_check(result, band_half_width_deg=6.74)
               if (r["profile"], r["quantity"]) == ("load", "U"))
    assert row["max"] == 0.0 and row["inside"]


def test_criterion_2_needs_both_the_maximum_and_the_rms_inside():
    """Each number alone outside the range of the published codes fails the criterion."""
    envelope = metrics.contributor_envelope("B", "load", "U", band_half_width_deg=6.74)
    # A constant between the codes' largest RMS and largest maximum: the
    # maximum is inside and the RMS is not.
    offset = 0.5 * (envelope["rms"] + envelope["max"])
    result = _reference_plus("B", lambda x, v: v + offset)
    row = next(r for r in metrics.envelope_check(result, band_half_width_deg=6.74)
               if (r["profile"], r["quantity"]) == ("load", "U"))
    assert row["max"] <= row["envelope_max"] and row["rms"] > row["envelope_rms"]
    assert not row["inside"]
    # One point beyond the codes' largest maximum, away from the features: the
    # RMS is inside and the maximum is not.
    spike = 2.0 * envelope["max"]
    result = _reference_plus("B", lambda x, v: v + np.where(np.argmin(np.abs(x - 25.0)) ==
                                                           np.arange(x.size), spike, 0.0))
    row = next(r for r in metrics.envelope_check(result, band_half_width_deg=6.74)
               if (r["profile"], r["quantity"]) == ("load", "U"))
    assert row["max"] > row["envelope_max"] and row["rms"] <= row["envelope_rms"]
    assert not row["inside"]


def test_a_colatitude_that_is_not_a_number_raises():
    """A `Curve` accepts NaN in its axis and sorts it last; scoring must not drop it quietly."""
    curve = BenchmarkResult.reference("B").profile("load", "U")
    axis = curve.axis.copy()
    axis[-1] = np.nan
    result = BenchmarkResult("B", "nan")
    result.add_profile("load", "U", axis, curve.values)
    with pytest.raises(ValueError, match="not a finite number"):
        metrics.score(result)


def test_a_time_that_is_not_a_number_raises():
    """The same check for a time series."""
    series = BenchmarkResult.reference("D").series("h_UF")
    axis = series.axis.copy()
    axis[-1] = np.nan
    result = BenchmarkResult("D", "nan")
    result.add_time_series("h_UF", axis, series.values)
    with pytest.raises(ValueError, match="not a finite number"):
        metrics.score(result)
