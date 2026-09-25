"""Print the score of a Martinec et al. (2018) run against the published reference.

The scoring lives in the package, in `giamip.benchmarks.metrics`, so that a
solver's own test calls the same functions as this script. This file only reads
a result and prints:

* the two metrics of eq. 31 for every compared profile and time series
  (`metrics.score`), on the result's own points, with the stored reference
  evaluated there;
* with ``--band``, criterion 2 of the G-ADOPT tests: the result's largest
  difference and RMS next to the largest of the published codes, with a band
  of that half width removed around the cap margin and the initial coastline
  (`metrics.envelope_check`);
* for cases B, C and D, the uniform water layer h_UF of the result at the
  terminal time next to the reference's, S - N on the basin meridian
  (`metrics.uniform_layer`).

Usage, from the repository root::

    ~/Workplace/python3.12/bin/python3.12 scripts/martinec_compare.py \\
        outputs/martinec2018/lovejx/martinec2018-C.nc --member lovejx --band 6.74
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from giamip.benchmarks import metrics                               # noqa: E402
from giamip.benchmarks.inputs import MartinecCase                   # noqa: E402
from giamip.benchmarks.results import BenchmarkResult               # noqa: E402


def print_score(rows: list[dict]) -> None:
    """One line per compared curve: RMS, maximum, the signal and the point count."""
    for row in rows:
        where = row["profile"] or "time series"
        scale = row.get("signal_range") or row["signal_max"]
        share = 100.0 * row["max"] / scale if scale else float("nan")
        print(f"  {where:12s} {row['quantity']:18s} rms {row['rms']:.6g}  max {row['max']:.6g} "
              f"at {row['at']:g}  ({share:.3f} per cent of {scale:.6g})  "
              f"n {row['n']}, spacing up to {row['max_spacing']:.3g}")
        print(f"  {'':12s} reference: {row['reference']}")


def print_envelope(rows: list[dict]) -> None:
    """One line per compared curve: the run against the range of the published codes."""
    for row in rows:
        verdict = "inside" if row["inside"] else "OUTSIDE"
        print(f"  {row['profile']:6s} {row['quantity']}: max {row['max']:.4g} of "
              f"{row['envelope_max']:.4g}, rms {row['rms']:.4g} of {row['envelope_rms']:.4g}  "
              f"{verdict}  (band {row['band_half_width_deg']:g} degrees around "
              f"{', '.join(f'{f:.4f}' for f in row['features'])}; codes "
              f"{', '.join(row['envelope_members'])})")


def report_uniform_layer(result: BenchmarkResult, epoch_kyr: float | None = None) -> dict:
    """Print and return the run's h_UF at the terminal time against the reference's.

    Args:
      result: the run, which must hold an ``h_UF`` time series.
      epoch_kyr: the time the run's series is read at, by linear
        interpolation. The default is the terminal time of the case.

    Returns:
      A dictionary with ``run``, ``reference``, ``difference``, ``relative``
      (in percent) and ``reference_spread``, all in metres except
      ``relative``.
    """
    case = MartinecCase(result.case)
    if epoch_kyr is None:
        epoch_kyr = case.terminal_time_kyr
    series = result.series("h_UF")
    mine = float(np.interp(epoch_kyr, series.axis, series.values))
    theirs, spread = metrics.uniform_layer(result.case)
    relative = 100.0 * abs(mine - theirs) / abs(theirs)
    print(f"  h_UF at {epoch_kyr:g} kyr: {result.member} {mine:.6f} m, reference {theirs:.6f} m, "
          f"difference {mine - theirs:+.6f} m, {relative:.3f} per cent")
    print(f"  the reference value is the mean of S - N on the basin meridian; it varies "
          f"by {spread:.2e} m along it")
    return {"run": mine, "reference": theirs, "difference": mine - theirs,
            "relative": relative, "reference_spread": spread}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("result", type=pathlib.Path, help="a BenchmarkResult file")
    parser.add_argument("--member", required=True, help="the member to score")
    parser.add_argument("--band", type=float, default=None,
                        help="half width in degrees of the band that criterion 2 removes "
                             "around each feature; criterion 2 is skipped without it, "
                             "because the width is not decided")
    args = parser.parse_args(argv)

    result = BenchmarkResult.read(args.result, args.member)
    print(f"case {result.case}, member {result.member}")
    print_score(metrics.score(result))
    if args.band is not None:
        print("criterion 2")
        print_envelope(metrics.envelope_check(result, band_half_width_deg=args.band))
    if result.case in ("B", "C", "D") and "h_UF" in result.time_series:
        print("uniform water layer")
        report_uniform_layer(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
