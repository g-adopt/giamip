"""Every NetCDF file of the package is read and written through h5netcdf.

AGENTS.md: every reader in the package opens NetCDF through h5netcdf, which
applies no mask, scale or fill value, and no module of the package imports
netCDF4 or xarray. The six GIAMIP ice files declare ``_FillValue = 0.0`` on
``ice_thickness``, so a masking reader would turn every ice-free cell into a
missing value. These tests show:

* (a) a value equal to ``_FillValue`` comes back as that value through each
  reader: `BenchmarkResult.read`, `SpadaResult.read` and `IceHistory`;
* (b) netCDF-C (through netCDF4, a test dependency only) reads every tracked
  converted file and gives the same numbers as the package's reader;
* (c) no module of the package imports netCDF4 or xarray;
* (d) the attributes of a result survive a write and a read with their types;
* (e) a fresh conversion reproduces the tracked converted files, variable by
  variable and attribute by attribute (marked ``data``: it needs the
  originals).
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest

from gia_mip import _netcdf, grids
from gia_mip.benchmarks.results import BenchmarkResult, SpadaResult

ROOT = pathlib.Path(__file__).resolve().parent.parent
CONVERTED = ROOT / "data" / "converted"


def _tracked_converted_files() -> list[pathlib.Path]:
    """Every converted file tracked in the checkout."""
    return sorted(CONVERTED.glob("*/*.nc"))


# ------------------------------------------------------------------- (a)

def test_a_fill_value_comes_back_through_benchmark_result_read(tmp_path):
    """A curve whose values equal its _FillValue is read as those values."""
    import h5netcdf
    path = tmp_path / "martinec2018-B.nc"
    with h5netcdf.File(path, "w") as f:
        f.attrs["case"] = "B"
        g = f.create_group("profiles").create_group("load").create_group("X")
        g.dimensions["colatitude"] = 3
        g.create_variable("colatitude", ("colatitude",), "f8", data=[1.0, 2.0, 3.0])
        g.create_variable("U", ("colatitude",), "f8", data=[0.0, 0.5, 0.0], fillvalue=0.0)
    curve = BenchmarkResult.read(path, "X").profile("load", "U")
    assert not np.ma.isMaskedArray(curve.values)
    np.testing.assert_array_equal(curve.values, [0.0, 0.5, 0.0])


def test_a_fill_value_comes_back_through_spada_result_read(tmp_path):
    """A spatial curve whose values equal its _FillValue is read as those values."""
    import h5netcdf
    path = tmp_path / "spada2011-t02-01.nc"
    with h5netcdf.File(path, "w") as f:
        f.attrs["case"] = "t02-01"
        g = (f.create_group("spatial").create_group("cap").create_group("X")
             .create_group("epoch_0"))
        g.attrs["epoch_kyr"] = "0"
        g.dimensions["colatitude"] = 3
        g.create_variable("colatitude", ("colatitude",), "f8", data=[0.0, 1.0, 2.0])
        g.create_variable("V", ("colatitude",), "f8", data=[0.0, 0.3, 0.0], fillvalue=0.0)
    curve = SpadaResult.read(path, "X").spatial("cap", "V", 0.0)
    assert not np.ma.isMaskedArray(curve.values)
    np.testing.assert_array_equal(curve.values, [0.0, 0.3, 0.0])


def test_a_fill_value_comes_back_through_ice_history(tmp_path):
    """An ice file like the GIAMIP ones: zeros declared as _FillValue stay zeros."""
    import h5netcdf
    from gia_mip.mip.inputs import IceHistory
    relative = "giamip/ice-history/iceHistory-synthetic.nc"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    thickness = np.zeros((1, grids.NLAT, grids.NLON))
    thickness[0, 10, 20] = 1234.5
    with h5netcdf.File(path, "w") as f:
        f.dimensions.update({"time": 1, "lat": grids.NLAT, "lon": grids.NLON})
        f.create_variable("lat", ("lat",), "f8", data=grids.latitudes())
        f.create_variable("lon", ("lon",), "f8", data=grids.longitudes())
        time = f.create_variable("time", ("time",), "f8", data=[-365.0 * 1000])
        time.attrs["calendar"] = "365_day"
        f.create_variable("year", ("time",), "f8", data=[-1000.0])
        f.create_variable("ice_thickness", ("time", "lat", "lon"), "f8", data=thickness,
                          fillvalue=0.0)
    with IceHistory(relative, root=tmp_path) as ice:
        field = ice.thickness(0)
    assert not np.ma.isMaskedArray(field)
    assert np.count_nonzero(field == 0.0) == field.size - 1
    assert field[10, 20] == 1234.5
    assert not np.any(np.isnan(field))


# ------------------------------------------------------------------- (b)

@pytest.mark.parametrize("path", _tracked_converted_files(), ids=lambda p: p.name)
def test_netcdf_c_reads_every_converted_file_as_the_package_does(path):
    """netCDF4 (netCDF-C) and h5netcdf give the same groups, variables and numbers."""
    netCDF4 = pytest.importorskip("netCDF4")
    with netCDF4.Dataset(path) as nc, _netcdf.open_read(path) as h5:
        nc.set_auto_mask(False)
        stack = [(nc, h5)]
        compared = 0
        while stack:
            g_nc, g_h5 = stack.pop()
            assert set(g_nc.groups) == set(g_h5.groups), g_h5.name
            assert set(g_nc.variables) == set(g_h5.variables), g_h5.name
            for name in g_nc.variables:
                np.testing.assert_array_equal(np.asarray(g_nc.variables[name][...]),
                                              np.asarray(g_h5.variables[name][...]))
                compared += 1
            stack.extend((g_nc.groups[k], g_h5.groups[k]) for k in g_nc.groups)
    assert compared > 0


# ------------------------------------------------------------------- (c)

def test_no_module_of_the_package_imports_netcdf4_or_xarray():
    """The Gadi environment has no netCDF4; a stray import would fail only there."""
    offenders = []
    for source in sorted((ROOT / "gia_mip").rglob("*.py")):
        tree = ast.parse(source.read_text(), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name.split(".")[0] in ("netCDF4", "xarray"):
                    offenders.append(f"{source.relative_to(ROOT)}:{node.lineno} {name}")
    assert not offenders, offenders


# ------------------------------------------------------------------- (d)

def test_attributes_keep_their_values_through_a_benchmark_result_file(tmp_path):
    """Text, integers and floats written as curve attributes come back equal."""
    run = BenchmarkResult("C", "radopt")
    run.add_profile("basin", "U", [0.0, 1.0, 2.0], [0.1, 0.2, 0.3],
                    note="a note", source_column=5, scale_applied=0.5)
    run.add_time_series("h_UF", [0.0, 15.0], [0.0, -25.7], solver_step_years=100)
    path = run.write(tmp_path / "martinec2018-C.nc")
    back = BenchmarkResult.read(path, "radopt", case="C")
    attrs = back.profile("basin", "U").attributes
    assert attrs["note"] == "a note" and isinstance(attrs["note"], str)
    assert attrs["source_column"] == 5
    assert attrs["scale_applied"] == 0.5
    assert attrs["units"] == "m"
    assert back.series("h_UF").attributes["solver_step_years"] == 100
    np.testing.assert_array_equal(back.series("h_UF").values, [0.0, -25.7])


def test_a_spada_result_keeps_complex_values_and_group_attributes(tmp_path):
    """The transfer function is complex and stored as two real parts; it returns bit for bit."""
    run = SpadaResult("t01-04", "radopt")
    # Slowest mode first (smallest |Re a_j|), the order the class keeps.
    frequencies = np.array([-0.1 + 0.0j, -1.5 + 0.25j])
    residues = np.array([0.01 + 0.0j, 0.3 - 0.2j])
    run.add_polar_motion_transfer_function("included", 0.9 + 0.1j, -0.02 + 0.0j,
                                           frequencies, residues, source_file="f.dat",
                                           note="synthetic")
    path = run.write(tmp_path / "spada2011-t01-04.nc")
    back = SpadaResult.read(path, "radopt").polar_motion_transfer_function("included")
    assert back.elastic == 0.9 + 0.1j and back.secular == -0.02 + 0.0j
    np.testing.assert_array_equal(back.frequencies, frequencies)
    np.testing.assert_array_equal(back.residues, residues)
    assert back.source_file == "f.dat"
    assert back.attributes["note"] == "synthetic"


# ------------------------------------------------------------------- (e)

def _content(path: pathlib.Path) -> dict:
    """Every group's attributes and every variable's values and attributes.

    ``_NCProperties`` is left out by `_netcdf.attributes`: it records the
    library versions, which differ between machines and are not content.
    """
    out = {}
    with _netcdf.open_read(path) as ds:
        stack = [ds]
        while stack:
            group = stack.pop()
            out[(group.name, None)] = _netcdf.attributes(group)
            for name, variable in group.variables.items():
                out[(group.name, name)] = (tuple(variable.dimensions),
                                           np.asarray(variable[...]),
                                           _netcdf.attributes(variable))
            stack.extend(group.groups.values())
    return out


def _assert_same_content(fresh: pathlib.Path, tracked: pathlib.Path) -> None:
    a, b = _content(fresh), _content(tracked)
    assert set(a) == set(b), sorted(set(a) ^ set(b))
    for key in a:
        if key[1] is None:
            assert a[key] == b[key], key
            continue
        dims_a, values_a, attrs_a = a[key]
        dims_b, values_b, attrs_b = b[key]
        assert dims_a == dims_b, key
        np.testing.assert_array_equal(values_a, values_b, err_msg=str(key))
        assert set(attrs_a) == set(attrs_b), (key, set(attrs_a) ^ set(attrs_b))
        for name in attrs_a:
            np.testing.assert_array_equal(attrs_a[name], attrs_b[name], err_msg=f"{key} {name}")


@pytest.mark.data
@pytest.mark.parametrize("case", ["A", "B", "C", "D", "E"])
def test_a_fresh_martinec_conversion_equals_the_tracked_file(data_root, tmp_path, case):
    from gia_mip.benchmarks import martinec2018
    root = data_root / "original"
    if not (root / "martinec2018" / f"results{case}_2026").is_dir():
        pytest.skip("the Martinec archives are not unpacked")
    fresh = martinec2018.convert(case, root, tmp_path)
    _assert_same_content(fresh, CONVERTED / "martinec2018" / f"martinec2018-{case}.nc")


@pytest.mark.data
@pytest.mark.parametrize("case", ["t01-01", "t01-02", "t01-03", "t01-04", "t01-05",
                                  "t01-06", "t01-07", "t02-01", "t02-02", "t02-03"])
def test_a_fresh_spada_conversion_equals_the_tracked_file(data_root, tmp_path, case):
    from gia_mip.benchmarks import spada2011
    root = data_root / "original"
    if not (root / "spada2011").is_dir():
        pytest.skip("the Spada archive is not unpacked")
    fresh = spada2011.convert(case, root, tmp_path)
    _assert_same_content(fresh, CONVERTED / "spada2011" / f"spada2011-{case}.nc")
