"""Convert the Spada et al. (2011) benchmark archive to NetCDF4.

Run it after unpacking spada2011-cost-es0701.tar.gz into
data/original/spada2011/ and after running scripts/decode_spada_layouts.py.
It writes data/converted/spada2011/spada2011-<case>.nc for each of the ten
tests. The work is done in giamip.benchmarks.spada2011, whose docstring
describes the rules and lists the files it reads per test.

Usage::

    python3 scripts/convert_spada2011.py [--case t02-01] [--root data/original] [--out DIR]
"""

import pathlib
import sys

# Run against this checkout's package, so a worktree converts with its own code.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from giamip.benchmarks import spada2011  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(spada2011.main())
