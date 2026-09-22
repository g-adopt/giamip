"""Score a G-ADOPT Martinec et al. (2018) run against the published reference.

This repository specifies the cases and holds the data to compare against; the
viscoelastic response is computed by G-ADOPT. This script is the piece between
them, and it is the G-ADOPT twin of `scripts/martinec_radopt.py`.

G-ADOPT runs the case on Gadi with
`demos/glacial_isostatic_adjustment/3d_martinec_sea_level/martinec_benchmark.py`
of the `sghelichkhani/sea-level` branch. That driver writes NumPy files and
nothing else, because a compute node must not read NetCDF and must not know
about this repository's readers:

``martinec-<label>-profiles_<t>kyr.npz``
    ``t_kyr``, ``shift_nondim``, ``h_UF_m``, and for each of the two
    comparison meridians ``<profile>_colatitude_deg``,
    ``<profile>_longitude_deg``, ``<profile>_U_m``, ``<profile>_N_m``,
    ``<profile>_S_m``, ``<profile>_RSL_m`` and ``<profile>_found``. The
    colatitudes are the colatitudes of the VEGA profile of that case, copied
    into the driver by `write_profile_json.py`, so no interpolation is needed
    anywhere in this file.
``martinec-<label>-timeseries.npz``
    One row per solved step: ``t_kyr``, ``h_UF_m``, ``ocean_area``,
    ``ice_mass_grounded_kg``, ``ice_mass_floating_kg`` and the solver
    counters.

This script reads those two files, builds a `BenchmarkResult` with the member
name ``g-adopt``, writes it to
``outputs/martinec2018/<member>/martinec2018-<case>.nc`` and scores it with
`scripts/martinec_compare.py`, exactly as a radopt run is scored.

Run it on the laptop, from the repository root:

    ~/Workplace/python3.12/bin/python3.12 scripts/martinec_gadopt.py B \
        --input ~/Workplace/gia-mip/outputs/martinec2018/g-adopt/g2 \
        --label g2 --member g-adopt

THE UNIFORM WATER LAYER OF CASE B AND CASE C

The paper compares `h_UF` against time for the moving-coastline cases only, so
the converted reference file of case B carries no time series at all
(``comparison.time_series`` of ``cases/benchmarks/martinec2018/B.json`` is
empty). The pass criterion of the G-ADOPT job G2 is nevertheless stated
against VEGA, so the VEGA value has to come from the profiles.

It does, exactly. Eq. 6 and eq. 8 of the paper give ``S = N + h_UF`` over the
ocean, and VEGA supplies both `S` and `N` on the basin meridian at the
terminal time on one common colatitude grid. Their difference is therefore the
VEGA uniform layer at the terminal time, and it is constant along the
meridian to 7.57e-06 m for case B, which is the evidence that the two curves
really do differ by the uniform layer and not by something else.
`vega_uniform_layer` returns that mean and that spread, and
`report_uniform_layer` prints the comparison with the run.

CAUTION: the `N` curves of ``data/converted/martinec2018/martinec2018-B.nc``
carry a sign fix that is modified in the working tree of this repository and is
committed nowhere. The number above, and the -119.896769 m it gives for case B,
need that version, MD5 ``4d256fd1387c993dda4d20a5e429531c``. The committed
version gives -120.344128 m and a spread of 3.02 m, which fails the constancy
argument. Commit the sign fix before you read an `N` number of this script.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from gia_mip.benchmarks.inputs import MartinecCase                   # noqa: E402
from gia_mip.benchmarks.results import BenchmarkResult               # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from martinec_compare import report_case, report_time_series         # noqa: E402

#: The driver's time-series array names, and the names `BenchmarkResult`
#: stores them under. `_time_series_units` of `gia_mip.benchmarks.results`
#: reads the unit from the stored name, so the names on the right are fixed:
#: metres for h_UF, kilograms for an ice mass, a fraction of 4 pi for the
#: ocean area. The driver's own names carry the unit as a suffix.
TIME_SERIES = {
    "h_UF_m": "h_UF",
    "ocean_area": "ocean_area",
    "ice_mass_grounded_kg": "ice_mass_grounded",
    "ice_mass_floating_kg": "ice_mass_floating",
}


def profiles_path(input_dir: pathlib.Path, label: str, epoch_kyr: float) -> pathlib.Path:
    """The driver's profile file of one epoch.

    The driver formats the epoch with ``%g``, so 10.0 becomes ``10`` and 2.5
    becomes ``2.5``. This reproduces that spelling instead of guessing.
    """
    return input_dir / f"martinec-{label}-profiles_{epoch_kyr:g}kyr.npz"


def timeseries_path(input_dir: pathlib.Path, label: str) -> pathlib.Path:
    """The driver's time-series file, which holds every solved step."""
    return input_dir / f"martinec-{label}-timeseries.npz"


def read_profiles(path: pathlib.Path, case: MartinecCase, letter: str,
                  result: BenchmarkResult) -> float:
    """Add the profile curves of one epoch file to a `BenchmarkResult`.

    Every quantity the case compares is taken from the file and checked
    against the VEGA colatitudes of the same case. Nothing is interpolated:
    the driver was given those colatitudes and evaluated the fields at them,
    so a mismatch here means the driver ran against a different reference file
    and is a reason to stop, not to resample.

    Args:
      path: the ``...-profiles_<t>kyr.npz`` file.
      case: the case, for the profile specification.
      letter: the case letter, for the VEGA reference and the messages.
      result: the result the curves are added to, in place.

    Returns:
      The epoch stored in the file, in kyr.

    Raises:
      FileNotFoundError: if the file is absent.
      ValueError: if a point of a profile was not located on the mesh, if a
        value is not finite, or if the colatitudes are not the VEGA ones.
    """
    if not path.exists():
        raise FileNotFoundError(f"no profile file {path}")
    data = np.load(path)
    epoch_kyr = float(data["t_kyr"])
    reference = BenchmarkResult.reference(letter, "ZM")
    for spec in case.profiles:
        # The driver marks a sample point it could not locate on either mesh.
        # Such a point silently carries whatever the evaluator returned, so a
        # single one invalidates the score of that profile.
        found = np.asarray(data[f"{spec.name}_found"], dtype=bool)
        if not found.all():
            raise ValueError(f"{path}: {int((~found).sum())} of {found.size} points of "
                             f"the {spec.name} profile were not located on the mesh")
        axis = np.asarray(data[f"{spec.name}_colatitude_deg"], dtype=float)
        published = np.asarray(reference.profile(spec.name, spec.quantities[0]).axis,
                               dtype=float)
        # `Curve` sorts its axis on construction, so the reference axis is
        # ascending while the driver's is in the order of the JSON. Sort the
        # driver's copy the same way before the comparison.
        order = np.argsort(axis, kind="stable")
        if axis.size != published.size or not np.allclose(axis[order], published, atol=1e-9):
            raise ValueError(f"{path}: the {spec.name} colatitudes are not the VEGA "
                             f"colatitudes of case {letter}")
        for quantity in spec.quantities:
            values = np.asarray(data[f"{spec.name}_{quantity}_m"], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"{path}: the {spec.name} {quantity} profile is not finite")
            result.add_profile(spec.name, quantity, axis, values,
                               time_kyr=epoch_kyr,
                               longitude_deg=float(data[f"{spec.name}_longitude_deg"]))
    return epoch_kyr


def read_time_series(path: pathlib.Path, result: BenchmarkResult) -> int:
    """Add every time series of the run to a `BenchmarkResult`.

    The driver writes one row per solved step, and for the Spada ladder of
    case B the first row is the elastic solve at t = 0. That row belongs in
    the series: the ladder continues from the elastic state, and only the
    viscous memory `solution_old` is reset after it.

    Args:
      path: the ``...-timeseries.npz`` file.
      result: the result the curves are added to, in place.

    Returns:
      The number of rows, which is the number of solved steps.

    Raises:
      FileNotFoundError: if the file is absent.
      ValueError: if a series is not finite.
    """
    if not path.exists():
        raise FileNotFoundError(f"no time-series file {path}")
    data = np.load(path)
    kyr = np.asarray(data["t_kyr"], dtype=float)
    for source, name in TIME_SERIES.items():
        if source not in data.files:
            continue
        values = np.asarray(data[source], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{path}: the {source} series is not finite")
        result.add_time_series(name, kyr, values)
    return int(kyr.size)


def read_run(input_dir, label: str, letter: str, member: str = "g-adopt",
             epoch_kyr: float | None = None, verbose: bool = True) -> BenchmarkResult:
    """Pack one G-ADOPT run into the `BenchmarkResult` the comparison reads.

    Args:
      input_dir: the directory that holds the driver's npz files.
      label: the driver's ``--label``, which every file name carries.
      letter: the case letter, B, C or D.
      member: the member name written into the NetCDF file. Use a separate
        name for a run that must not collide with the plain ``g-adopt`` one,
        for example ``g-adopt-K1000`` for the bulk-modulus run.
      epoch_kyr: the epoch whose profiles are read. The default is the
        terminal time of the case, which is the time the paper's profile
        figures are drawn at.
      verbose: print what was read.

    Returns:
      The result, with the profiles of one epoch and the time series of every
      step.
    """
    input_dir = pathlib.Path(input_dir)
    case = MartinecCase(letter)
    if epoch_kyr is None:
        # Every profile of a case is drawn at the same terminal time. Taking
        # it from the case file keeps the default right when a case changes.
        epoch_kyr = float(case.profiles[0].time_kyr)
    result = BenchmarkResult(letter, member)
    path = profiles_path(input_dir, label, epoch_kyr)
    stored = read_profiles(path, case, letter, result)
    if abs(stored - epoch_kyr) > 1e-9:
        raise ValueError(f"{path} holds t = {stored} kyr, not {epoch_kyr} kyr")
    rows = read_time_series(timeseries_path(input_dir, label), result)
    if verbose:
        print(f"case {letter}, member {member}: profiles at {epoch_kyr:g} kyr from "
              f"{path.name}, {rows} steps of time series")
    return result


def vega_uniform_layer(letter: str) -> tuple[float, float]:
    """The VEGA uniform water layer at the terminal time, from the profiles.

    The converted reference file of a fixed-coastline case carries no time
    series, so `h_UF` has to come from the curves that are there. Eq. 6 and
    eq. 8 give ``S = N + h_UF`` over the ocean, and VEGA supplies `S` and `N`
    on the basin meridian on one common colatitude grid, so their difference
    is the uniform layer.

    Args:
      letter: the case letter.

    Returns:
      The mean of ``S - N`` in metres, and the spread of that difference along
      the meridian, also in metres. A spread of the order of the rounding of
      the archive file (about 1e-05 m) is the evidence that the two curves
      differ by a constant.

    Raises:
      KeyError: if VEGA has no `S` or no `N` on the basin meridian of the case.
    """
    reference = BenchmarkResult.reference(letter, "ZM")
    surface = reference.profile("basin", "S")
    geoid = reference.profile("basin", "N")
    if not np.allclose(surface.axis, geoid.axis):
        raise ValueError(f"case {letter}: the VEGA S and N basin profiles are on "
                         "different colatitudes")
    difference = np.asarray(surface.values) - np.asarray(geoid.values)
    return float(difference.mean()), float(difference.max() - difference.min())


def report_uniform_layer(result: BenchmarkResult, letter: str,
                         epoch_kyr: float | None = None) -> dict:
    """Print and return the run's `h_UF` against the VEGA value.

    This is the pass criterion of the G-ADOPT job G2: the uniform water layer
    within 1 percent of VEGA at the terminal time.

    Args:
      result: the run.
      letter: the case letter.
      epoch_kyr: the time the comparison is read at. The default is the
        terminal time of the case.

    Returns:
      A dictionary with ``gadopt``, ``vega``, ``difference``, ``relative``
      (in percent) and ``vega_spread``, all in metres except ``relative``.
    """
    if epoch_kyr is None:
        epoch_kyr = float(MartinecCase(letter).profiles[0].time_kyr)
    series = result.series("h_UF")
    mine = float(np.interp(epoch_kyr, series.axis, series.values))
    theirs, spread = vega_uniform_layer(letter)
    relative = 100.0 * abs(mine - theirs) / abs(theirs)
    print(f"h_UF at {epoch_kyr:g} kyr: {result.member} {mine:.4f} m, VEGA {theirs:.4f} m, "
          f"difference {mine - theirs:+.4f} m, {relative:.3f} per cent")
    print(f"    the VEGA value is the mean of S - N on the basin meridian; it varies "
          f"by {spread:.2e} m along it")
    return {"gadopt": mine, "vega": theirs, "difference": mine - theirs,
            "relative": relative, "vega_spread": spread}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case", choices=["B", "C", "D"],
                        help="the published case letter")
    parser.add_argument("--input", type=pathlib.Path, required=True,
                        help="the directory that holds the driver's npz files")
    parser.add_argument("--label", required=True,
                        help="the driver's --label, which every file name carries")
    parser.add_argument("--member", default="g-adopt",
                        help="member name of the result; use another name for a run "
                             "that must not collide with the plain g-adopt one, "
                             "for example g-adopt-K1000")
    parser.add_argument("--epoch", type=float, default=None,
                        help="the epoch in kyr whose profiles are read; the default "
                             "is the terminal time of the case")
    parser.add_argument("--output", type=pathlib.Path, default=None,
                        help="where to write the result; the default is "
                             "outputs/martinec2018/<member>/martinec2018-<case>.nc")
    parser.add_argument("--no-write", action="store_true",
                        help="score only, and write no NetCDF file")
    parser.add_argument("--overwrite", action="store_true",
                        help="delete an existing output file first; without it a "
                             "second run of the same member raises, so that two runs "
                             "cannot mix in one file")
    args = parser.parse_args(argv)

    result = read_run(args.input, args.label, args.case, member=args.member,
                      epoch_kyr=args.epoch)

    print()
    print(f"profiles against VEGA, case {args.case}, member {args.member}")
    profile_scores = report_case(result, args.case)
    print()
    # Case B and case C publish no reference time series, so this prints
    # nothing for them and the uniform layer is scored from the profiles
    # instead, just below.
    series_scores = report_time_series(result, args.case)
    layer = report_uniform_layer(result, args.case, epoch_kyr=args.epoch)
    del profile_scores, series_scores, layer

    if not args.no_write:
        path = args.output or (REPOSITORY / "outputs" / "martinec2018" / args.member
                               / f"martinec2018-{args.case}.nc")
        path.parent.mkdir(parents=True, exist_ok=True)
        if args.overwrite and path.exists():
            path.unlink()
        result.write(path)
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
