"""Score a Martinec case run against the published reference.

The metrics are the two of the paper, eq. 31: the root mean square and the
maximum of the difference from the reference, over the colatitude interval the
case file records for that profile.

Benchmark A needs one extra step, which is why this file exists beside
`gia_mip.benchmarks.metrics`. The VEGA profile in the archive of case A,
``Z_ZM_load.out``, is a second VEGA run and not the one behind figure 3. Every
``Z_<code>_load_diff.out`` is that contributor minus the published reference, so
adding a contributor's own profile back onto its own difference returns the
reference. `published_reference` does that and checks that the contributors
agree on the answer.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from gia_mip.benchmarks.inputs import MartinecCase                   # noqa: E402
from gia_mip.benchmarks.results import BenchmarkResult               # noqa: E402


def published_reference(case: str, profile: str, quantity: str):
    """The reference curve of case A, recovered from the contributors.

    Returns
    -------
    colatitude
        The colatitudes of the recovered curve, in degrees.
    values
        The reference values in metres, the mean over the contributors that
        supplied both a profile and a difference file.
    spread
        The largest difference between any two contributors' recoveries, in
        metres. It is the evidence that the recovery is the published reference
        and not one code's own answer.
    """
    path = BenchmarkResult.reference_path(case)
    axis = None
    recovered = {}
    for member in BenchmarkResult.members(path):
        result = BenchmarkResult.read(path, member, case=case)
        if (profile, quantity) not in result.profiles:
            continue
        if (profile, quantity) not in result.differences:
            continue
        own = result.profiles[(profile, quantity)]
        diff = result.differences[(profile, quantity)]
        if not np.allclose(own.axis, diff.axis):
            continue
        if axis is None:
            axis = np.asarray(own.axis, dtype=float)
        elif own.axis.shape != axis.shape or not np.allclose(own.axis, axis):
            continue
        recovered[member] = np.asarray(own.values, dtype=float) - np.asarray(diff.values, dtype=float)
    if not recovered:
        raise KeyError(f"case {case}: no contributor supplies both a {quantity} "
                       f"profile and a difference on the {profile} meridian")
    stack = np.stack(list(recovered.values()))
    spread = float(np.nanmax(np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)))
    return axis, np.nanmean(stack, axis=0), spread


def score(values, reference, colatitude, interval) -> dict:
    """The two metrics of eq. 31 over one colatitude interval.

    `values` and `reference` must be sampled at the same colatitudes.
    """
    lo, hi = interval
    inside = (colatitude >= lo) & (colatitude <= hi)
    difference = np.asarray(values)[inside] - np.asarray(reference)[inside]
    return {
        "n": int(inside.sum()),
        "rms": float(np.sqrt(np.mean(difference ** 2))),
        "max": float(np.max(np.abs(difference))),
        "signal_max": float(np.max(np.abs(np.asarray(reference)[inside]))),
    }


def report_case(result: BenchmarkResult, case_letter: str) -> dict:
    """Score a run of case B, C or D against VEGA, profile by profile.

    The reference is VEGA (contributor ZM), which is the published reference for
    every case except A. The run must be sampled at the reference's own
    colatitudes: two of the VEGA profiles are coarse copies on a 359-point grid,
    and the correction martinec2018-caseD-vega-profile-grid says not to
    interpolate them.
    """
    case = MartinecCase(case_letter)
    reference = BenchmarkResult.reference(case_letter, "ZM")
    out = {}
    for spec in case.profiles:
        print(f"{spec.name} profile, longitude {spec.longitude_deg:g}, "
              f"colatitude {spec.colatitude_interval_deg[0]:g} to "
              f"{spec.colatitude_interval_deg[1]:g} degrees")
        for quantity in spec.quantities:
            curve = result.profile(spec.name, quantity)
            published = reference.profile(spec.name, quantity)
            axis = np.asarray(published.axis, dtype=float)
            if not np.allclose(axis, np.asarray(curve.axis, dtype=float)):
                raise ValueError(f"{spec.name}/{quantity}: the run and the reference "
                                 "are sampled at different colatitudes")
            row = score(curve.values, published.values, axis, spec.colatitude_interval_deg)
            out[(spec.name, quantity)] = row
            print(f"    {quantity}: rms {row['rms']:.4f} m, max {row['max']:.4f} m, "
                  f"signal {row['signal_max']:.2f} m, "
                  f"{100 * row['max'] / row['signal_max']:.2f} per cent of it, "
                  f"{row['n']} points")
    return out


def report_case_a(result: BenchmarkResult) -> dict:
    """Score a case A run and print the table the paper's figure 3 shows."""
    case = MartinecCase("A")
    spec = case.profiles[0]
    out = {}
    for quantity in spec.quantities:
        curve = result.profile("load", quantity)
        axis, reference, spread = published_reference("A", "load", quantity)
        if not np.allclose(axis, curve.axis):
            raise ValueError(f"{quantity}: the run is sampled at colatitudes the "
                             "recovered reference does not carry")
        row = score(curve.values, reference, axis, spec.colatitude_interval_deg)
        row["contributor_spread"] = spread
        out[quantity] = row
        print(f"{quantity}: over colatitude {spec.colatitude_interval_deg[0]} to "
              f"{spec.colatitude_interval_deg[1]} degrees, {row['n']} points\n"
              f"    rms {row['rms']:.4f} m, max {row['max']:.4f} m, "
              f"signal {row['signal_max']:.2f} m, "
              f"{100 * row['max'] / row['signal_max']:.2f} per cent of it\n"
              f"    the contributors recover the reference to {spread:.2e} m")
    return out


def report_time_series(result: BenchmarkResult, case_letter: str,
                       quantities=None, epochs=(5.0, 10.0, 15.0)) -> dict:
    """Score the time series of a run against VEGA, over 0 to 15 kyr.

    The reference is stored on a 751 point grid of 20 year steps and the run is
    on its own step grid, so the reference is read at the run's step times by
    linear interpolation. Interpolating the dense side is safe: 20 years is
    below every step any contributor used.

    The root mean square and maximum difference over the whole interval are the
    metrics of eq. 31 carried over to time. Each is printed against two scales,
    the largest absolute value of the reference and its range, because the two
    differ by a factor of 24 for the ocean area and by nothing at all for the
    series that start at zero. The values at `epochs` are printed as well,
    because the paper's figures 13, 16, 17, 18 and 19 are read at a few epochs
    and a relative difference there is easier to compare with the text.
    """
    reference = BenchmarkResult.reference(case_letter, "ZM")
    names = quantities if quantities is not None else sorted(reference.time_series)
    out = {}
    for name in names:
        if name not in reference.time_series or name not in result.time_series:
            continue
        published = reference.series(name)
        curve = result.series(name)
        axis = np.asarray(curve.axis, dtype=float)
        inside = (axis >= 0.0) & (axis <= 15.0)
        ours = np.asarray(curve.values, dtype=float)[inside]
        theirs = np.interp(axis[inside], np.asarray(published.axis, dtype=float),
                           np.asarray(published.values, dtype=float))
        difference = ours - theirs
        # Two scales, because they are far apart for the ocean area. The
        # largest absolute value is the right scale for h_UF and for the ice
        # masses, which all start at zero, so it is their range as well. The
        # ocean area starts at 0.0463 and moves by 0.0020 over the whole run,
        # so a difference divided by the absolute value is 24 times smaller
        # than the same difference divided by what the quantity actually does.
        # The range is the honest scale for it, and both are reported.
        scale = float(np.max(np.abs(theirs))) or 1.0
        span = float(np.max(theirs) - np.min(theirs)) or scale
        row = {"n": int(inside.sum()),
               "rms": float(np.sqrt(np.mean(difference ** 2))),
               "max": float(np.max(np.abs(difference))),
               "signal_max": scale,
               "signal_range": span}
        for epoch in epochs:
            mine = float(np.interp(epoch, axis[inside], ours))
            published_value = float(np.interp(epoch, np.asarray(published.axis, dtype=float),
                                              np.asarray(published.values, dtype=float)))
            row[f"at_{epoch:g}kyr"] = (mine, published_value)
        out[name] = row
        print(f"{name}: rms {row['rms']:.6g}, max {row['max']:.6g}, "
              f"largest value {scale:.6g} ({100 * row['max'] / scale:.3f} per cent), "
              f"range {span:.6g} ({100 * row['max'] / span:.3f} per cent)")
        for epoch in epochs:
            mine, published_value = row[f"at_{epoch:g}kyr"]
            relative = (abs(mine - published_value) / abs(published_value) * 100.0
                        if published_value else float("nan"))
            print(f"    {epoch:g} kyr: radopt {mine:.6g}, VEGA {published_value:.6g}, "
                  f"{relative:.3f} per cent")
    return out
