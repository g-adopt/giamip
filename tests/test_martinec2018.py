"""The Martinec (2018) converter: its rules, and the numbers it writes.

The tests without a mark need only the resources in the package. The tests
marked `data` convert the archives in data/original/martinec2018/ and check the
result against the difference files Martinec shipped, which were computed by
the benchmark organisers and are the independent evidence that each sign,
scale and unit is right.
"""

import json
import pathlib

import numpy as np
import pytest

from giamip import corrections
from giamip.benchmarks import martinec2018

LAYOUTS = martinec2018.load_layouts()


# ------------------------------------------------------------ no data needed

@pytest.mark.parametrize("scale,expected", [
    (None, 1.0), ("x/9.8155", 1 / 9.8155), ("x*100", 100.0), ("x*10", 10.0),
    ("x/1e17", 1e-17), ("9*x/10", 0.9), ("2*x/3", 2 / 3), ("x/10", 0.1),
])
def test_scale_factor_evaluates_the_script_arithmetic(scale, expected):
    assert martinec2018.scale_factor(scale) == pytest.approx(expected, rel=1e-15)


@pytest.mark.parametrize("scale", ["x+1", "sqrt(x)", "y*2", "x**2", "__import__('os')"])
def test_scale_factor_refuses_anything_else(scale):
    with pytest.raises(ValueError, match="unsupported"):
        martinec2018.scale_factor(scale)


def test_every_scale_in_the_scripts_is_understood():
    """A new kind of arithmetic in a decoded script must be seen, not evaluated."""
    for script in LAYOUTS["scripts"]:
        for panel in script["panels"]:
            for series in panel["series"]:
                if series["y"] and "column" in series["y"]:
                    martinec2018.scale_factor(series["y"]["scale"])


def test_every_correction_on_a_converted_file_has_a_decision():
    """A correction added without a decision in HANDLING must fail here first."""
    for script in LAYOUTS["scripts"]:
        if "/results" not in script["script"]:
            continue
        for panel in script["panels"]:
            for series in panel["series"]:
                if series["drawn"]:
                    martinec2018.applying(series["path"])


def test_handling_names_only_real_corrections():
    """A stale entry in HANDLING hides a correction that was renamed."""
    for correction_id in martinec2018.HANDLING:
        corrections.by_id(correction_id)


def test_each_results_directory_holds_one_internal_code():
    """resultsC_2026/ holds D3 files, and its figures are titled Benchmark C.

    Every contributor file in a directory carries the code, apart from the
    VEGA files and the difference files, which have none, and GS's case D
    files, which carry it as a suffix.
    """
    for case, code in martinec2018.INTERNAL_CODES.items():
        if code is None:
            continue
        names = {key.rsplit("/", 1)[-1] for key in LAYOUTS["by_file"]
                 if key.startswith(f"martinec2018/results{case}_2026/")}
        assert names, case
        for name in names:
            if name.startswith("Z_") or name in {"vega76.out", "vega76_E2.out", "secti.out"}:
                continue
            assert code in name, (case, name)


def test_read_table_skips_a_header_and_rewrites_fortran_exponents(tmp_path):
    f = tmp_path / "t.dat"
    f.write_text("# comment\n  n  lon  colat\n 1 0.5D+01 -0.25D-01\n\n 2 6.0 NaN 7\n")
    t = martinec2018.read_table(f)
    assert t.shape == (2, 4)
    assert t[0, 1] == 5.0 and t[0, 2] == -0.025
    assert np.isnan(t[0, 3]) and np.isnan(t[1, 2])


def test_read_table_refuses_text_after_data(tmp_path):
    f = tmp_path / "t.dat"
    f.write_text("1 2\n3 4\nthis is not a number\n")
    with pytest.raises(ValueError, match="line 3"):
        martinec2018.read_table(f)


def test_a_repeated_point_with_different_data_raises():
    group = martinec2018.Group("colatitude", "degrees", "colatitude")
    attrs = {"source_file": "f", "source_column": 2}
    with pytest.raises(ValueError, match="different data"):
        group.add("U", np.array([0.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0]), dict(attrs))
    group.add("U", np.array([0.0, 1.0, 1.0]), np.array([1.0, 2.0, 2.0]), dict(attrs))
    assert group.variables["U"]["attributes"]["repeated_points_dropped"] == 1


# --------------------------------------------------------------- data needed

def _read(path, group, name):
    """One variable and its axis, read through h5netcdf as AGENTS.md requires."""
    from giamip import _netcdf
    with _netcdf.open_read(path) as ds:
        g = ds[group]
        v = g.variables[name]
        axis = np.asarray(g.variables[v.dimensions[0]][...]).copy()
        return axis, np.asarray(v[...]).copy(), _netcdf.attributes(v)


@pytest.fixture(scope="module")
def converted(tmp_path_factory, data_root):
    root = data_root / "original"
    if not (root / "martinec2018" / "resultsE_2026").is_dir():
        pytest.skip("the resultsA_2026 to resultsE_2026 archives are not unpacked")
    out = tmp_path_factory.mktemp("martinec2018")
    return {case: martinec2018.convert(case, root, out) for case in martinec2018.CASES}


@pytest.mark.data
def test_every_axis_ascends_and_every_value_is_finite(converted):
    from giamip import _netcdf
    for case, path in converted.items():
        with _netcdf.open_read(path) as ds:
            stack = [ds]
            while stack:
                g = stack.pop()
                stack.extend(g.groups.values())
                for name, v in g.variables.items():
                    values = np.asarray(v[...])
                    assert np.all(np.isfinite(values)), (case, g.name, name)
                    if name in g.dimensions:
                        assert np.all(np.diff(values) > 0), (case, g.name, name)


@pytest.mark.data
@pytest.mark.parametrize("case,profile", [("B", "load"), ("C", "load"), ("C", "basin")])
def test_ys_minus_vega_reproduces_the_shipped_difference(converted, case, profile):
    """YS and VEGA share a 2048-point grid here, so no interpolation enters.

    This checks the geoid sign and the gravity factor of both codes at once,
    including the case C basin, where the script divides by 10. The case B
    basin and both case D profiles are left out because VEGA is on a coarser
    grid there.

    U and S agree to print precision. N agrees to 5e-6 of its size and no
    better, because the organisers computed the difference files with a
    surface gravity of 9.81555 where the scripts, and this converter, use
    9.8155. The tolerance allows that, and it is five orders of magnitude
    below the 1.8 per cent error the factor 10 would give.
    """
    path = converted[case]
    for name in ("U", "N") + (("S",) if profile == "basin" else ()):
        x_ys, ys, _ = _read(path, f"profiles/{profile}/YS", name)
        x_zm, zm, _ = _read(path, f"profiles/{profile}/ZM", name)
        x_d, d, _ = _read(path, f"differences/{profile}/YS", name)
        assert np.allclose(x_ys, x_zm, atol=1e-4) and np.allclose(x_ys, x_d, atol=1e-4)
        tolerance = 2e-8 + (1e-5 * np.abs(zm) if name == "N" else 0.0)
        assert np.all(np.abs(ys - zm - d) <= tolerance), (case, profile, name)


@pytest.mark.data
def test_the_case_a_stale_sk_profile_is_not_converted(converted):
    from giamip import _netcdf
    with _netcdf.open_read(converted["A"]) as ds:
        assert "SK" not in ds["profiles/load"].groups
        assert "SK" in ds["differences/load"].groups
        assert "benchmarkA-sk-superseded" in _netcdf.attributes(ds)["skipped"]


@pytest.mark.data
def test_the_figure_only_rescaling_is_not_applied(converted):
    _, values, attrs = _read(converted["A"], "differences/load/WW_axisymmetric", "U")
    assert attrs["figure_only_scale_not_applied"] == "9*x/10"
    assert attrs["scale_applied"] == 1.0


@pytest.mark.data
def test_case_d_time_series_agree_after_unit_conversion(converted):
    """Five codes, three storage units each for mass and area, one answer."""
    from giamip import _netcdf
    path = converted["D"]
    with _netcdf.open_read(path) as ds:
        members = set(ds["time_series"].groups)
    assert members == {"ZM", "GS", "VK", "YS", "SK"}
    for member in members:
        t, area, _ = _read(path, f"time_series/{member}", "ocean_area")
        assert 0.04614 < area[0] < 0.04632, (member, area[0])
    for member in ("ZM", "GS", "VK", "YS"):
        t, grounded, _ = _read(path, f"time_series/{member}", "ice_mass_grounded")
        t2, floating, _ = _read(path, f"time_series/{member}", "ice_mass_floating")
        i, j = np.argmin(abs(t - 15)), np.argmin(abs(t2 - 15))
        assert 1.18e18 < grounded[i] + floating[j] < 1.21e18, member
    for member in ("ZM", "GS", "YS", "SK"):
        t, huf, _ = _read(path, f"time_series/{member}", "h_UF")
        assert -30.3 < huf[np.argmin(abs(t - 15))] < -29.7, member


@pytest.mark.data
def test_case_e_keeps_the_case_d_run_apart(converted):
    from giamip import _netcdf
    with _netcdf.open_read(converted["E"]) as ds:
        assert {"ZM", "ZM_case_D", "VK", "YS", "SK"} == set(ds["time_series"].groups)
        assert "profiles" not in ds.groups
    _, huf, attrs = _read(converted["E"], "time_series/VK", "h_UF")
    assert attrs["source_column"] == 5


@pytest.mark.data
def test_secti_column_5_is_named_for_what_it_holds(converted):
    _, values, attrs = _read(converted["D"], "time_series/ZM", "h_UF_1_plus_h_UF_4")
    assert attrs["source_file"].endswith("secti.out")
    assert "martinec2018-secti-column5" in attrs["corrections"]


@pytest.mark.data
def test_case_e_ocean_area_is_a_fraction_for_every_member(converted):
    """Case E YS ocean area is written with a D exponent that awk drops.

    Copying the case D unit gives 0.0047 here, ten times too small, and the
    figure hides it; see martinec2018-ys-case-e-awk-exponent.
    """
    for member in ("ZM", "ZM_case_D", "VK", "YS", "SK"):
        _, area, _ = _read(converted["E"], f"time_series/{member}", "ocean_area")
        assert 0.0440 < area.min() and area.max() < 0.0471, (member, area.min(), area.max())


@pytest.mark.data
@pytest.mark.parametrize("case", ["A", "B", "C", "D"])
def test_every_contributor_profile_agrees_with_its_difference_file(converted, case):
    """Contributor minus VEGA, on the contributor's grid, against the shipped difference.

    VEGA is interpolated here, which costs up to a few centimetres near the ice
    edge on the 359-row grids, so the test compares the fit with the fit of the
    negated curve. A wrong sign or a wrong factor makes the two comparable; a
    right one makes the first far smaller. Case A is compared against the
    reference recovered from GS, because its shipped VEGA file is a second run.
    """
    from giamip import _netcdf
    path = converted[case]
    with _netcdf.open_read(path) as ds:
        profiles = {p: set(ds[f"differences/{p}"].groups) for p in ds["differences"].groups}
    for profile, members in profiles.items():
        for member in sorted(members):
            with _netcdf.open_read(path) as ds:
                if member not in ds[f"profiles/{profile}"].groups:
                    continue
                names = set(ds[f"profiles/{profile}/{member}"].variables) & set(
                    ds[f"differences/{profile}/{member}"].variables) & {"U", "N", "S"}
            for name in sorted(names):
                x, v, _ = _read(path, f"profiles/{profile}/{member}", name)
                xd, dv, _ = _read(path, f"differences/{profile}/{member}", name)
                ref_member = "GS" if case == "A" else "ZM"
                if case == "A":
                    xr, rv, _ = _read(path, f"profiles/{profile}/GS", name)
                    xg, dg, _ = _read(path, f"differences/{profile}/GS", name)
                    rv = rv - np.interp(xr, xg, dg)
                else:
                    xr, rv, _ = _read(path, f"profiles/{profile}/{ref_member}", name)
                inside = (xd > max(x.min(), xr.min())) & (xd < min(x.max(), xr.max()))
                xs = xd[inside]
                ours = np.interp(xs, x, v) - np.interp(xs, xr, rv)
                right = np.sqrt(np.mean((ours - dv[inside]) ** 2))
                flipped = np.sqrt(np.mean((-np.interp(xs, x, v) - np.interp(xs, xr, rv) - dv[inside]) ** 2))
                assert right < 0.2 * flipped, (case, profile, member, name, right, flipped)
