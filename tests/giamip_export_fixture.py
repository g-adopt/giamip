"""Synthetic intermediate files in the layout of `giamip.mip.export`, for tests and the Gadi proof run.

The fields are analytic, so a test knows every value the writer must store:

* every field is ``A(lat, lon) + B(lat, lon) * s`` with
  ``s = (year - first) / (last - first)``, which is 0 at the first output
  and 1 at the last. The patterns are scaled by 1/3 so that almost no value
  is a float32 number, which lets a test tell an interpolation in float64
  followed by a cast from a cast followed by an interpolation;
* ``delta_g`` is ``s * R * sum(C_lm Y_lm)`` for the coefficients
  `DELTA_G_COEFFICIENTS`, so the Stokes files have a known answer;
* the two fractions stay in [0.1, 0.9];
* the scalars are inside the checker bounds, or are taken from a params log
  of a real run (``--params-log``), which gives the real model years too.

Used by ``tests/test_giamip_outputs.py`` through `write_synthetic_export`,
and on Gadi as a script:

    python giamip_export_fixture.py --experiment Exp09 \\
        --params-log .../params_forward_exp09.log --out export_exp09.h5
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

from giamip import grids, harmonics
from giamip.mip.export import EXPORT_ATTRIBUTES, EXPORT_FIELDS, EXPORT_SCALARS

R = 6371000.0

# Known Stokes coefficients of the final delta_g (degree, order, value), in
# the convention of `giamip.harmonics`. Odd l + m and odd m are present, as
# the R3 test of the harmonics module asks.
DELTA_G_COEFFICIENTS_C = {(0, 0): 2e-6, (1, 0): 1e-7, (2, 1): -3e-6, (3, 3): 1.1e-6, (5, 2): 4e-6}
DELTA_G_COEFFICIENTS_S = {(2, 1): 2.5e-6, (3, 3): -1.5e-6}

# Columns of params_forward_expXX.log, by the GIAMIP scalar they hold.
PARAMS_COLUMNS = {"mean_delta_g": "mean_delta_g_m", "grd_ice_mass": "grd_ice_mass_kg",
                  "total_ice_mass": "total_ice_mass_kg", "maf": "maf_kg",
                  "ocean_area": "ocean_area_m2", "ocean_area_grdice": "ocean_area_grdice_m2"}


def stokes_arrays():
    """The known coefficients as (97, 97) arrays, (degree, order)."""
    c = np.zeros((97, 97))
    s = np.zeros((97, 97))
    for (l, m), v in DELTA_G_COEFFICIENTS_C.items():
        c[l, m] = v
    for (l, m), v in DELTA_G_COEFFICIENTS_S.items():
        s[l, m] = v
    return c, s


def patterns() -> dict:
    """``(A, B)`` of each field, ``(257, 513)`` float64, metres or 1."""
    lat = np.radians(grids.latitudes())[:, None]
    lon = np.radians(grids.longitudes())[None, :]
    one = np.ones((grids.NLAT, grids.NLON))
    c, s = stokes_arrays()
    dg = harmonics.synthesise(c, s, R)
    return {
        "delta_bed": (0.0 * one, (100.0 / 3.0) * np.sin(lat) * np.cos(lon) + 1.0 / 3.0),
        "delta_g": (0.0 * one, dg),
        "delta_bed_east": (0.0 * one, (10.0 / 3.0) * np.cos(lat) * np.sin(2 * lon)),
        "delta_bed_north": (0.0 * one, (7.0 / 3.0) * np.sin(2 * lat) * np.cos(lon)),
        "ocean_area_fraction": (0.5 + (1.0 / 3.0) * np.cos(lat) ** 2 * np.cos(lon) * one,
                                (0.1 / 3.0) * np.sin(lon) * one),
        "land_ice_area_fraction": (0.4 + (1.0 / 3.0) * np.sin(lat) ** 2 * one,
                                   (0.1 / 3.0) * np.cos(2 * lon) * one),
    }


def default_scalars(years: np.ndarray) -> dict:
    """Scalars inside the checker bounds, smooth in time."""
    s = (years - years[0]) / (years[-1] - years[0])
    return {"mean_delta_g": -50.0 * s / 3.0, "grd_ice_mass": 3e19 + 1e19 * s / 3.0,
            "total_ice_mass": 3.1e19 + 1e19 * s / 3.0, "maf": 2.5e19 + 1e19 * s / 3.0,
            "ocean_area": 3.6e14 + 1e12 * s / 3.0, "ocean_area_grdice": 3.7e14 + 1e12 * s / 3.0}


def default_attributes(experiment: str) -> dict:
    """Root attributes of the intermediate file with recognisable values (k_s = 0.123 shows a hard-coded k_s)."""
    attrs = {
        "experiment": experiment, "schedule": "synthetic", "git_head": "0000000synthetic",
        "git_branch": "synthetic", "firedrake": "synthetic", "petsc": "synthetic",
        "mesh": "synthetic", "bed_iteration": 0, "bed_misfit_year0_rms_m": 0.0,
        "bed_misfit_year0_max_abs_m": 0.0, "k_s": 0.123, "C_minus_A_nondim": 0.00329,
        "surface_file": "none", "surface_file_size": 0, "surface_file_mtime": "none",
        "ranks": 1, "fraction_method": "synthetic test value", "fraction_samples": 0,
        "created": "synthetic", "omega_rad_s": 7.292115e-5,
    }
    assert set(EXPORT_ATTRIBUTES) <= set(attrs)
    return attrs


def read_params_log(path) -> dict:
    """``year``, ``step`` and the six scalars of a params_forward_expXX.log."""
    lines = [ln.split() for ln in pathlib.Path(path).read_text().splitlines() if ln.strip()]
    header, rows = lines[0], np.array(lines[1:], dtype=float)
    col = {name: rows[:, header.index(name)] for name in header}
    return {"year": col["year"], "step": col["step"].astype(np.int64),
            "scalars": {k: col[v] for k, v in PARAMS_COLUMNS.items()}}


def write_synthetic_export(path, experiment: str, years, *, steps=None, scalars=None,
                           attrs=None, edit_row=None, lat=None, drop=(),
                           compression=None) -> pathlib.Path:
    """Write one intermediate file.

    Parameters
    ----------
    path
        The file.
    experiment
        ``Exp01`` ... written as the ``experiment`` root attribute.
    years
        Model output years, increasing.
    steps
        Solver steps; default ``0 .. n-1``.
    scalars
        The six scalars; default `default_scalars`.
    attrs
        Root attributes that replace the defaults.
    edit_row
        ``f(name, k, row) -> row``, applied to each field row before it is
        written, to plant a bad value.
    lat
        A latitude axis to write instead of the grid's (to test the check).
    drop
        Dataset paths to leave out.
    compression
        h5py compression of the fields; None for the tests.
    """
    import h5py
    path = pathlib.Path(path)
    years = np.asarray(years, dtype=float)
    n = years.size
    s = (years - years[0]) / (years[-1] - years[0])
    steps = np.arange(n, dtype=np.int64) if steps is None else np.asarray(steps, dtype=np.int64)
    scalars = default_scalars(years) if scalars is None else scalars
    root = default_attributes(experiment)
    root.update(attrs or {})
    with h5py.File(path, "w") as f:
        for k, v in root.items():
            f.attrs[k] = v
        f["lat"] = grids.latitudes() if lat is None else lat
        f["lon"] = grids.longitudes()
        f["year"] = years
        f["step"] = steps
        f["index"] = np.arange(n, dtype=np.int64)
        for name, (a, b) in patterns().items():
            if f"fields/{name}" in drop:
                continue
            ds = f.create_dataset(f"fields/{name}", (n, grids.NLAT, grids.NLON), "f8",
                                  chunks=(1, grids.NLAT, grids.NLON), compression=compression)
            for k in range(n):
                row = a + b * s[k]
                if edit_row is not None:
                    row = edit_row(name, k, row)
                ds[k] = row
        for name in EXPORT_SCALARS:
            if f"scalars/{name}" in drop:
                continue
            f[f"scalars/{name}"] = np.asarray(scalars[name], dtype=float)
        if "check/ice_thickness_year0" not in drop:
            f["check/ice_thickness_year0"] = np.zeros((grids.NLAT, grids.NLON))
    assert set(EXPORT_FIELDS) == set(patterns())
    return path


def write_forcing_file(path, years) -> pathlib.Path:
    """A forcing file with the variables the checker reads: ``year`` int32, ``lat``, ``lon``."""
    import h5netcdf
    years = np.asarray(years).astype(np.int32)
    with h5netcdf.File(path, "w") as f:
        f.dimensions.update({"time": years.size, "lat": grids.NLAT, "lon": grids.NLON})
        f.create_variable("time", ("time",), "f8", data=years * 365.0)
        f.create_variable("year", ("time",), "i4", data=years)
        f.create_variable("lat", ("lat",), "f8", data=grids.latitudes())
        f.create_variable("lon", ("lon",), "f8", data=grids.longitudes())
    return pathlib.Path(path)


def main(argv=None) -> int:
    """Write a synthetic export with the years and scalars of a params log, or of the forcing."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--experiment", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--params-log", help="take the model years, steps and scalars from this log")
    a = p.parse_args(argv)
    if a.params_log:
        log = read_params_log(a.params_log)
        years, steps, scalars = log["year"], log["step"], log["scalars"]
    else:
        from giamip.mip.outputs import forcing_years
        years = forcing_years(a.experiment).astype(float)
        steps, scalars = None, None
    write_synthetic_export(a.out, a.experiment, years, steps=steps, scalars=scalars,
                           compression="gzip")
    print(f"wrote {a.out}: {len(years)} model years {years[0]:.0f} .. {years[-1]:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
