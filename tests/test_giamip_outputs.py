"""The GIAMIP writer `giamip.mip.outputs` makes files that pass the pinned checker.

Most tests run on a short synthetic time axis (`FORCING`, twelve years) so
they are fast; the last one, marked ``data``, runs on the real 463-year axis
of Exp01 with the real forcing file.

The checker itself is imported from a clone at the pinned commit, named by
the environment variable ``GIAMIP_CHECKER_DIR``. Without it, or without
xarray and tqdm (which the checker imports and the CI test extra does not
install), the checker tests skip. The fill-value test runs through netCDF4,
which the test extra has, and adds the xarray view when xarray is present.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import re

import numpy as np
import pytest

import giamip_export_fixture as fx
from giamip import _netcdf, cases, grids, harmonics
from giamip.mip import outputs
from giamip.mip.export import ExportError, ExportFile

# A short forcing axis with the protocol shape: coarse then fine steps.
FORCING = np.array([-3000, -2500, -2000, -1500, -1000, -900, -800, -700, -600, -500, 0, 100],
                   dtype=np.int32)
# A schedule-D-like subset: the first and last years, gaps of several steps.
SCHEDULE_D = np.array([-3000, -2000, -1000, -900, -500, 0, 100], dtype=float)

ALL_VARIABLES = ["delta_bed", "delta_g", "ocean_area_fraction", "land_ice_area_fraction",
                 "mean_delta_g", "grd_ice_mass", "total_ice_mass", "ocean_area_grdice",
                 "ocean_area", "maf", "delta_bed_east", "delta_bed_north", "Clm", "Slm"]
FIELDS = ["delta_bed", "delta_g", "ocean_area_fraction", "land_ice_area_fraction",
          "delta_bed_east", "delta_bed_north"]


# ------------------------------------------------------------------ helpers

def _checker():
    """The pinned checker module, or skip."""
    root = os.environ.get("GIAMIP_CHECKER_DIR")
    if not root:
        pytest.skip("GIAMIP_CHECKER_DIR is not set (a clone of the checker at the pinned commit)")
    pytest.importorskip("xarray")
    pytest.importorskip("tqdm")
    path = pathlib.Path(root) / "giamip_compliance_checker.py"
    spec = importlib.util.spec_from_file_location("giamip_compliance_checker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_checker(directory, forcing_path) -> dict:
    """Run the checker on one experiment directory and return its summary."""
    checker = _checker()
    summary = checker.run_checker(source_path=str(directory), forcing_path=str(forcing_path),
                                  commit_num="pinned")
    # The checker prints a warning and skips every comparison with the forcing
    # file when it cannot open that file, and the summary still counts zero
    # errors. Require that the comparisons ran: one per time variable (12 of
    # the 14 files; Clm and Slm have no time axis).
    assert summary["log_text"].count("Time axis matches forcing: passed") == 12, \
        "the checker did not compare the time axis with the forcing file"
    return summary


def _write(tmp_path, years=None, *, out="Exp01", experiment="Exp01", **kwargs):
    """A synthetic export and the files of the writer on the short axis."""
    years = FORCING.astype(float) if years is None else years
    export = fx.write_synthetic_export(tmp_path / f"export_{out}.h5", experiment, years,
                                       **kwargs)
    out_dir = tmp_path / out
    report = outputs.write_experiment(export, out_dir, forcing_years_override=FORCING,
                                      report_path=tmp_path / f"report_{out}.json")
    return export, out_dir, report


@pytest.fixture(scope="module")
def protocol(tmp_path_factory):
    """Files written from an export at every forcing year."""
    return _write(tmp_path_factory.mktemp("protocol"))


@pytest.fixture(scope="module")
def schedule_d(tmp_path_factory):
    """Files written from an export at a subset of the forcing years."""
    return _write(tmp_path_factory.mktemp("schedule_d"), SCHEDULE_D)


def _read(path, name):
    with _netcdf.open_read(path) as f:
        return np.asarray(f[name][...]), _netcdf.attributes(f[name]), _netcdf.attributes(f)


# ------------------------------------------------------------------ names and storage

def test_fourteen_files_with_the_giamip_names(protocol):
    _, out_dir, _ = protocol
    names = sorted(p.name for p in out_dir.glob("*.nc"))
    assert names == sorted(f"{v}_Exp01_G-ADOPT_g-adopt.nc" for v in ALL_VARIABLES)
    assert not list(out_dir.glob("*.partial"))


def test_names_parse_with_the_checker(protocol):
    checker = _checker()
    _, out_dir, _ = protocol
    for path in out_dir.glob("*.nc"):
        var, exp, group, model = checker._parse_giamip_filename(path.name)
        assert (exp, group, model) == ("Exp01", "G-ADOPT", "g-adopt")
        assert path.name.startswith(var + "_Exp01")


def test_values_round_trip_on_the_protocol_axis(protocol):
    """Fields are the export rows cast to float32, scalars the export values, bit for bit."""
    export, out_dir, _ = protocol
    with ExportFile(export) as e:
        for name in FIELDS:
            values, _, _ = _read(out_dir / outputs.file_name(name, "Exp01"), name)
            assert values.dtype == np.float32
            np.testing.assert_array_equal(values, e.field(name, list(range(e.n))).astype(np.float32))
        for name in ["mean_delta_g", "maf", "ocean_area"]:
            values, _, _ = _read(out_dir / outputs.file_name(name, "Exp01"), name)
            assert values.dtype == np.float64
            np.testing.assert_array_equal(values, e.scalar(name))
    year, _, _ = _read(out_dir / outputs.file_name("delta_bed", "Exp01"), "year")
    time, _, _ = _read(out_dir / outputs.file_name("delta_bed", "Exp01"), "time")
    assert year.dtype == np.int32 and time.dtype == np.float64
    np.testing.assert_array_equal(year, FORCING)
    np.testing.assert_array_equal(time, FORCING * 365.0)


def test_storage_layout(protocol):
    """float32 fields with one gzip-4 chunk per slice; float64 scalars and Stokes; checker dim order."""
    import h5py
    _, out_dir, _ = protocol
    for name in ALL_VARIABLES:
        with h5py.File(out_dir / outputs.file_name(name, "Exp01"), "r") as f:
            ds = f[name]
            if name in FIELDS:
                assert ds.dtype == np.float32
                assert ds.chunks == (1, 257, 513)
                assert ds.compression == "gzip" and ds.compression_opts == 4
            else:
                assert ds.dtype == np.float64
        with _netcdf.open_read(out_dir / outputs.file_name(name, "Exp01")) as f:
            dims = f[name].dimensions
        expected = (("time", "lat", "lon") if name in FIELDS else
                    ("degree", "order") if name in ("Clm", "Slm") else ("time",))
        assert dims == expected


def test_no_fill_value_anywhere(protocol):
    """No _FillValue or missing_value seen by netCDF-C, nor by xarray when it is installed."""
    netCDF4 = pytest.importorskip("netCDF4")
    _, out_dir, _ = protocol
    for path in out_dir.glob("*.nc"):
        with netCDF4.Dataset(path) as nc:
            for var in nc.variables.values():
                assert "_FillValue" not in var.ncattrs(), (path.name, var.name)
                assert "missing_value" not in var.ncattrs(), (path.name, var.name)
    try:
        import xarray
    except ImportError:
        return
    for path in out_dir.glob("*.nc"):
        with xarray.open_dataset(path, decode_times=False) as ds:
            for name in ds.variables:
                assert ds[name].encoding.get("_FillValue") is None, (path.name, name)


# ------------------------------------------------------------------ attributes

def test_required_and_identity_attributes(protocol):
    _, out_dir, _ = protocol
    for path in out_dir.glob("*.nc"):
        with _netcdf.open_read(path) as f:
            g = _netcdf.attributes(f)
        assert g["group"] == "G-ADOPT" and g["model"] == "g-adopt"
        assert g["contributors_name"] == (
            "Sia Ghelichkhan, William Scott, Mark Hoggard, Angus Gibson, Dale S. Roberts, "
            "Stephan Kramer, Thomas Duvernay, Rhodri Davies, G-ADOPT team")
        assert g["contributors_email"] == (
            "siavash.ghelichkhan@anu.edu.au, William.Scott1@anu.edu.au, mark.hoggard@anu.edu.au, "
            "angus.gibson@anu.edu.au, dale.roberts1@anu.edu.au, s.kramer@imperial.ac.uk, "
            "thomas.duvernay@anu.edu.au, rhodri.davies@anu.edu.au, help@gadopt.org")
        assert g["reference_frame"] == "CM"
        assert g["reference_frame_comment"].startswith("Centre of mass")
        assert "https://github.com/g-adopt/giamip" in g["references"]
        assert outputs.CONVENTIONS_URL in g["references"]
        assert re.search(r"commit [0-9a-f]{40}", g["references"])


def test_variable_and_coordinate_attributes(protocol):
    """units, long_name, standard_name from variables.json; coordinates from the table; reference."""
    _, out_dir, _ = protocol
    table = outputs.variable_table()
    for name in ALL_VARIABLES:
        with _netcdf.open_read(out_dir / outputs.file_name(name, "Exp01")) as f:
            attrs = _netcdf.attributes(f[name])
            assert attrs["units"] == table[name]["units"]
            assert attrs["long_name"] == table[name]["long_name"]
            assert attrs.get("standard_name") == table[name]["standard_name"]
            assert attrs.get("reference") == outputs.VARIABLE_REFERENCE.get(name)
            for coord in f.variables:
                if coord == name:
                    continue
                spec = outputs.COORDINATE_ATTRIBUTES[coord]
                got = _netcdf.attributes(f[coord])
                assert np.dtype(f[coord].dtype) == np.dtype(spec["dtype"]), coord
                for key, value in spec.items():
                    if key != "dtype":
                        assert got[key] == value, (name, coord, key)
    _, attrs, _ = _read(out_dir / outputs.file_name("Clm", "Exp01"), "Clm")
    assert attrs["first_year"] == -3000 and attrs["final_year"] == 100
    assert "6371000 m" in attrs["comment"]


def test_constants_from_the_case_and_run_values_from_the_export(protocol):
    """Densities and radius equal the case file; k_s is the fixture's unusual 0.123."""
    _, out_dir, _ = protocol
    _, _, g = _read(out_dir / outputs.file_name("maf", "Exp01"), "maf")
    constants = cases.case("giamip-exp01")["constants"]
    for key, value in constants.items():
        if not key.startswith("$"):
            assert g[key] == value, key
    assert g["run_k_s"] == pytest.approx(0.123)
    assert g["run_omega_rad_s"] == pytest.approx(7.292115e-5)   # not in EXPORT_ATTRIBUTES
    assert g["earth_elastic_structure"] == "prem-stable"
    assert g["earth_compressibility"] == "compressible"


def test_tables_agree_with_the_pinned_checker():
    checker = _checker()
    for coord, long_name in checker.GIAMIP_COORD_LONG_NAMES.items():
        assert outputs.COORDINATE_ATTRIBUTES[coord]["long_name"] == long_name
    for coord, standard_name in checker.GIAMIP_COORD_STANDARD_NAMES.items():
        assert outputs.COORDINATE_ATTRIBUTES[coord]["standard_name"] == standard_name
    for coord, dtype in checker.GIAMIP_COORD_DTYPES.items():
        assert np.dtype(outputs.COORDINATE_ATTRIBUTES[coord]["dtype"]) == np.dtype(dtype)
    assert outputs.COORDINATE_ATTRIBUTES["time"]["units"] == checker.GIAMIP_TIME_UNITS
    assert outputs.COORDINATE_ATTRIBUTES["time"]["calendar"] == checker.GIAMIP_TIME_CALENDAR
    table = outputs.variable_table()
    for v in checker.GIAMIP_VARIABLES:
        ours = table[v["variable"]]
        assert ours["units"] == v["units"] and ours["long_name"] == v["long_name"]
        assert ours["standard_name"] == v["standard_name"]
        assert np.dtype(ours["precision"]) == np.dtype(v["precision"])
        assert (ours["bounds"] is None) == (v["bounds"] is None)
        if v["bounds"] is not None:
            assert tuple(ours["bounds"]) == tuple(v["bounds"])
    # The "Reference" column of variables.md, the last column of the two variable tables.
    text = (pathlib.Path(os.environ["GIAMIP_CHECKER_DIR"]) / "variables.md").read_text()
    reference = {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 12 and cells[2].startswith("`"):
            reference[cells[2].strip("`")] = cells[11]
    assert len(reference) == 14
    for name, value in reference.items():
        # The table that transcribes the data request must equal it exactly.
        # The attribute that is written may differ only for the two horizontal
        # components, whose data-request datum is a copy of the delta_g row.
        expected = None if value in ("—", "-") else value
        assert outputs.VARIABLE_REFERENCE_DATA_REQUEST.get(name) == expected, name
        if name not in ("delta_bed_east", "delta_bed_north"):
            assert outputs.VARIABLE_REFERENCE.get(name) == expected, name


# ------------------------------------------------------------------ the checker

def test_checker_passes_on_the_protocol_axis(protocol, tmp_path):
    _, out_dir, _ = protocol
    forcing = fx.write_forcing_file(tmp_path / "forcing.nc", FORCING)
    summary = _run_checker(out_dir, forcing)
    assert summary["file_counter"] == 14
    assert summary["total_errors"] == 0, summary["log_text"]


def test_checker_passes_on_the_schedule_d_axis(schedule_d, tmp_path):
    _, out_dir, _ = schedule_d
    forcing = fx.write_forcing_file(tmp_path / "forcing.nc", FORCING)
    summary = _run_checker(out_dir, forcing)
    assert summary["file_counter"] == 14
    assert summary["total_errors"] == 0, summary["log_text"]


# ------------------------------------------------------------------ time interpolation

def test_interpolation_is_exact_for_a_linear_series():
    model = np.array([0.0, 300.0, 1000.0])
    forcing = np.arange(0, 1001, 100)
    tm = outputs.TimeMap.from_years(model, forcing)
    series = 2.5 + 1e-3 * model
    np.testing.assert_allclose(tm.apply_scalar(series), 2.5 + 1e-3 * forcing, rtol=0, atol=1e-12)
    assert tm.apply_scalar(series)[3] == series[1]       # year 300 is copied
    assert not tm.is_identity
    assert tm.interpolated_ranges() == [(100, 200, 2), (400, 900, 6)]


def test_interpolated_fields_are_cast_after_the_interpolation(schedule_d):
    """Every stored float32 value equals float32((1 - a) v0 + a v1) of the float64 rows, bit for bit.

    The fixture values are thirds, which are not float32 numbers, so casting
    the rows first and interpolating after differs by an ulp in many cells.
    """
    export, out_dir, _ = schedule_d
    tm = outputs.TimeMap.from_years(SCHEDULE_D, FORCING)
    with ExportFile(export) as e:
        for name in ("delta_bed", "ocean_area_fraction"):
            stored, _, _ = _read(out_dir / outputs.file_name(name, "Exp01"), name)
            wrong_order_differs = False
            for t in range(FORCING.size):
                v0, v1, a = e.field(name, int(tm.k0[t])), e.field(name, int(tm.k1[t])), tm.a[t]
                expected = v0.astype(np.float32) if a == 0 else np.float32((1 - a) * v0 + a * v1)
                np.testing.assert_array_equal(stored[t], expected)
                if a:
                    wrong = ((1 - a) * v0.astype(np.float32) + a * v1.astype(np.float32)).astype(np.float32)
                    wrong_order_differs |= bool(np.any(wrong != stored[t]))
            assert wrong_order_differs, f"{name}: the test cannot tell the two orders apart"


@pytest.mark.parametrize("model,message", [
    ([-2500.0, 0.0, 100.0], "extrapolate"),
    ([-3000.0, 0.0], "extrapolate"),
    ([-3000.0, -2900.0, 100.0], "not forcing years"),
    ([-3000.0, -1000.0, -2000.0, 100.0], "strictly increasing"),
    ([-3000.0, -999.5, 100.0], "whole years"),
])
def test_interpolation_refuses(model, message):
    with pytest.raises(ValueError, match=message):
        outputs.TimeMap.from_years(model, FORCING)


def test_interpolated_years_are_named_in_the_comment(protocol, schedule_d):
    _, out_dir_d, report = schedule_d
    _, attrs, g = _read(out_dir_d / outputs.file_name("maf", "Exp01"), "maf")
    assert "at 7 of the 12 forcing years" in attrs["comment"]
    assert "-2500; -1500; -800 to -600 (3 years)" in attrs["comment"]
    assert report["interpolated_years"] == 5
    assert g["time_interpolation"].startswith(
        "the model writes output at 7 of the 12 forcing years; the values at the other 5 "
        "years are linear interpolations")
    _, out_dir_p, _ = protocol
    _, attrs_p, g_p = _read(out_dir_p / outputs.file_name("maf", "Exp01"), "maf")
    assert "interpolation" not in attrs_p["comment"]
    assert g_p["time_interpolation"].startswith("none")


# ------------------------------------------------------------------ bounds and bad values

def test_a_horizontal_component_outside_the_bounds_is_skipped_not_clipped(tmp_path):
    def plant(name, k, row):
        if name == "delta_bed_east" and k == 3:
            row = row.copy()
            row[100, 200] = 250.0
        return row
    _, out_dir, report = _write(tmp_path, edit_row=plant)
    assert not (out_dir / outputs.file_name("delta_bed_east", "Exp01")).exists()
    assert (out_dir / outputs.file_name("delta_bed_north", "Exp01")).exists()
    assert len(list(out_dir.glob("*.nc"))) == 13
    skipped = report["skipped"]
    assert [s["variable"] for s in skipped] == ["delta_bed_east"]
    assert skipped[0]["max"] == 250.0
    assert json.loads((tmp_path / "report_Exp01.json").read_text())["skipped"] == skipped


def test_a_non_finite_mandatory_field_stops_with_a_report(tmp_path):
    def plant(name, k, row):
        if name == "delta_g" and k == 4:
            row = row.copy()
            row[10, 20] = np.nan
        return row
    with pytest.raises(outputs.OutputError, match="delta_g: non-finite value at year -1000"):
        _write(tmp_path, edit_row=plant)
    out_dir = tmp_path / "Exp01"
    assert not (out_dir / outputs.file_name("delta_g", "Exp01")).exists()
    assert not list(out_dir.glob("*.partial"))
    report = json.loads((tmp_path / "report_Exp01.json").read_text())
    assert "delta_g" in report["error"] and "year -1000" in report["error"]


def test_a_mandatory_scalar_outside_the_bounds_stops(tmp_path):
    scalars = fx.default_scalars(FORCING.astype(float))
    scalars["maf"] = scalars["maf"].copy()
    scalars["maf"][2] = 5e18
    with pytest.raises(outputs.OutputError,
                       match=r"maf: value outside the checker bounds \[1e\+19, 1e\+20\] at year -2000"):
        _write(tmp_path, scalars=scalars)
    assert not (tmp_path / "Exp01" / outputs.file_name("maf", "Exp01")).exists()


# ------------------------------------------------------------------ Stokes coefficients

def test_stokes_files_recover_the_known_coefficients(protocol):
    _, out_dir, report = protocol
    c_expected, s_expected = fx.stokes_arrays()
    c, _, _ = _read(out_dir / outputs.file_name("Clm", "Exp01"), "Clm")
    s, _, _ = _read(out_dir / outputs.file_name("Slm", "Exp01"), "Slm")
    assert c.shape == s.shape == (97, 97)
    np.testing.assert_allclose(c, c_expected, rtol=0, atol=1e-16)
    np.testing.assert_allclose(s, s_expected, rtol=0, atol=1e-16)
    degree, _, _ = _read(out_dir / outputs.file_name("Clm", "Exp01"), "degree")
    assert degree.dtype == np.float64
    np.testing.assert_array_equal(degree, np.arange(97.0))
    assert report["degree_1"]["C10"] == pytest.approx(1e-7, abs=1e-16)


# ------------------------------------------------------------------ references, overwrite, export

def test_references_are_rewritten_in_place(tmp_path):
    export, out_dir, _ = _write(tmp_path)
    before = {}
    for p in out_dir.glob("*.nc"):
        with _netcdf.open_read(p) as f:
            g = _netcdf.attributes(f)
            name = p.name.split("_Exp01")[0]
            before[p.name] = (np.asarray(f[name][...]), {k: v for k, v in g.items()
                                                          if k != "references"})
    changed = outputs.rewrite_references(out_dir, "giamip 0.3.0, a release")
    assert len(changed) == 14
    for p in out_dir.glob("*.nc"):
        with _netcdf.open_read(p) as f:
            g = _netcdf.attributes(f)
            name = p.name.split("_Exp01")[0]
            assert g["references"] == "giamip 0.3.0, a release"
            np.testing.assert_array_equal(np.asarray(f[name][...]), before[p.name][0])
            assert {k: v for k, v in g.items() if k != "references"} == before[p.name][1]
    if os.environ.get("GIAMIP_CHECKER_DIR"):
        forcing = fx.write_forcing_file(tmp_path / "forcing.nc", FORCING)
        assert _run_checker(out_dir, forcing)["total_errors"] == 0


def test_existing_files_are_not_overwritten(tmp_path):
    export, out_dir, _ = _write(tmp_path)
    with pytest.raises(FileExistsError):
        outputs.write_experiment(export, out_dir, forcing_years_override=FORCING)
    outputs.write_experiment(export, out_dir, forcing_years_override=FORCING, overwrite=True)


def test_overwrite_leaves_no_file_of_an_older_run(tmp_path):
    """A rerun with overwrite that stops half way leaves only files of the new run."""
    _, out_dir, _ = _write(tmp_path)
    assert (out_dir / outputs.file_name("delta_g", "Exp01")).exists()

    def plant(name, k, row):
        if name == "delta_bed" and k == 2:
            row = row.copy()
            row[0, 0] = np.inf
        return row
    bad = fx.write_synthetic_export(tmp_path / "bad.h5", "Exp01", FORCING.astype(float),
                                    edit_row=plant)
    with pytest.raises(outputs.OutputError, match="delta_bed: non-finite"):
        outputs.write_experiment(bad, out_dir, forcing_years_override=FORCING, overwrite=True)
    # delta_g comes after delta_bed, so this run never wrote it; the old file is gone.
    assert not (out_dir / outputs.file_name("delta_g", "Exp01")).exists()
    assert not (out_dir / outputs.file_name("Clm", "Exp01")).exists()
    assert sorted(p.name.split("_Exp01")[0] for p in out_dir.glob("*.nc")) == sorted(
        ["mean_delta_g", "grd_ice_mass", "total_ice_mass", "maf", "ocean_area",
         "ocean_area_grdice"])


@pytest.mark.parametrize("kwargs,message", [
    ({"lat": grids.latitudes()[::-1]}, "latitude differs"),
    ({"drop": ("scalars/maf",)}, "scalars/maf"),
    ({"attrs": {"experiment": "Exp02"}}, "for Exp02, not Exp01"),
    ({"steps": [0, 1, 2]}, "step has shape"),
    ({"drop": ("check/ice_thickness_year0",)}, "ice_thickness_year0"),
])
def test_a_wrong_export_is_refused(tmp_path, kwargs, message):
    path = fx.write_synthetic_export(tmp_path / "e.h5", "Exp01", FORCING.astype(float), **kwargs)
    with pytest.raises(ExportError, match=message):
        ExportFile(path, "Exp01")


def test_writer_provenance(tmp_path):
    """git work tree first, then a WRITER_COMMIT file, else an error."""
    here = outputs.writer_provenance()
    assert re.fullmatch(r"[0-9a-f]{40}(-dirty)?", here["commit"])
    copy = tmp_path / "copy"
    copy.mkdir()
    with pytest.raises(RuntimeError, match="no writer commit"):
        outputs.writer_provenance(copy)
    (copy / "WRITER_COMMIT").write_text("abc123-dirty\n")
    assert outputs.writer_provenance(copy) == {"commit": "abc123-dirty", "dirty": True,
                                               "source": str(copy / "WRITER_COMMIT")}


def test_report_has_the_row_shift_numbers(protocol):
    _, _, report = protocol
    rows = report["ocean_area_grid_against_mesh"]
    assert [r["row"] for r in rows] == [0, 11]   # |mean_delta_g| is largest at the last row
    for r in rows:
        assert set(r) == {"row", "year", "relative_difference", "neighbour_row",
                          "relative_difference_neighbour"}


# ------------------------------------------------------------------ the real Exp01 axis

@pytest.mark.data
def test_full_exp01_axis_passes_the_checker_with_the_real_forcing(tmp_path):
    """463 forcing years read from ICE-6G_D; the checker compares year with that file."""
    from giamip.mip.inputs import GiamipExperiment
    years = outputs.forcing_years("Exp01")
    assert years.size == 463 and years[0] == -122000 and years[-1] == 1000
    export = fx.write_synthetic_export(tmp_path / "export_exp01.h5", "Exp01",
                                       years.astype(float))
    out_dir = tmp_path / "G-ADOPT_g-adopt" / "Exp01"
    report = outputs.write_experiment(export, out_dir)
    assert len(report["files"]) == 14
    assert "rms_m" in report["ice_check_year0"]
    forcing_path = GiamipExperiment("Exp01").ice_history.path
    summary = _run_checker(out_dir, forcing_path)
    assert summary["total_errors"] == 0, summary["log_text"]


def test_time_interpolation_states_model_and_interpolated_counts(tmp_path):
    """An axis like Exp09: the sentence gives the model outputs and the interpolated years apart.

    The two numbers differ (here 4 model outputs and 8 interpolated years of
    12), so a sentence that puts one count in the place of the other fails.
    """
    model = np.array([-3000.0, -1000.0, -500.0, 100.0])
    _, out_dir, report = _write(tmp_path, model)
    _, _, g = _read(out_dir / outputs.file_name("delta_g", "Exp01"), "delta_g")
    assert report["model_years"] == 4 and report["interpolated_years"] == 8
    assert ("the model writes output at 4 of the 12 forcing years; the values at the other 8 "
            "years are linear interpolations in time") in g["time_interpolation"]


@pytest.mark.parametrize("number", range(1, 13))
def test_no_attribute_says_same_as_or_open(tmp_path, number):
    """Every case field copied into an attribute is resolved, for all twelve case files.

    Exp07 gives its rheology as "same as Exp02" in the protocol table; the
    file must carry the rheology of Exp02 with the G-ADOPT choices (maxwell).
    """
    exp = f"Exp{number:02d}"
    years = np.array([-3000.0, 100.0])
    path = fx.write_synthetic_export(tmp_path / "e.h5", exp, years)
    case = cases.with_gadopt_choices(f"giamip-exp{number:02d}")
    tm = outputs.TimeMap.from_years(years, years)
    table = outputs.variable_table()
    with ExportFile(path, exp) as e:
        attrs = outputs.global_attributes(e, case, {"commit": "0" * 40}, tm, "sha")
        texts = {k: v for k, v in attrs.items() if isinstance(v, str)}
        for name in ALL_VARIABLES:
            for k, v in outputs.variable_attributes(name, table[name], case, e, tm).items():
                if isinstance(v, str):
                    texts[f"{name}:{k}"] = v
    offenders = {k: v for k, v in texts.items() if "same as" in v.lower()}
    assert not offenders, offenders
    assert not [k for k in attrs if k.startswith("earth_") and attrs[k] == "open"]
    if number == 7:
        assert attrs["earth_rheology"] == "maxwell"
