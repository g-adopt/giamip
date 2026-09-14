"""Convert the Martinec et al. (2018) benchmark archives to NetCDF4.

Run it after unpacking resultsA_2026.zip to resultsE_2026.zip into
data/original/martinec2018/ and after running scripts/decode_gmt_layouts.py.
It writes data/converted/martinec2018/martinec2018-<case>.nc. The work is done
in gia_mip.benchmarks.martinec2018, whose docstring describes the output.

Usage::

    python3 scripts/convert_martinec2018.py [--case C] [--root data/original] [--out DIR]
"""

import pathlib
import sys

# Run against this checkout's package, so a worktree converts with its own code.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from gia_mip.benchmarks import martinec2018  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(martinec2018.main())
