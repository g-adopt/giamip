"""Check an installed gia-mip wheel from a directory outside the checkout.

A solver's job on a compute node uses gia-mip as an installed package, with
no clone of this repository and no netCDF4. This script does what such a job
does, with the wheel installed into a fresh virtual environment that holds
only the wheel's own dependencies:

1. it checks that ``gia_mip`` is imported from the installation and not from
   this checkout, and that netCDF4 is absent;
2. it reads case B with `gia_mip.case`, gets its reference through the cache
   (downloaded on first use), scores a synthetic run (the reference plus
   0.01 m) and checks that every row gives 0.01 m, runs criterion 2 with a
   band of 6.74 degrees, and computes the uniform water layer;
3. it writes the synthetic run with `BenchmarkResult.write` and reads it back;
4. it scores the Spada t02-01 contributor GS against itself;
5. with ``--all-converted``, it locates every converted file of the manifest,
   which fails when one of them is not on the store.

``GIA_MIP_DATA_ROOT`` decides where the downloads go. When it is not set, a
temporary directory is used, so the check never fills a user cache.

With ``--store DIR`` the script first builds a store in DIR from this
checkout's ``data/converted/``, at the object keys of the installed manifest,
and points ``GIA_MIP_BASE_URL`` at it through ``file://``. The continuous
integration uses that, so that a change to a converted file is tested before
it is uploaded. Without ``--store`` the real store at data.gadopt.org is used.

Usage, with the wheel installed in the active environment::

    cd "$(mktemp -d)"
    python /path/to/gia-mip/scripts/check_installed.py [--store DIR] [--all-converted]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import shutil
import sys
import tempfile

# This file's own checkout. The script is run by its path, so Python puts
# scripts/ first on sys.path, not the checkout root, and `import gia_mip`
# below finds the installed package.
CHECKOUT = pathlib.Path(__file__).resolve().parent.parent


def fail(message: str) -> None:
    """Stop with a non-zero exit code and the reason."""
    print(f"FAILED: {message}", file=sys.stderr)
    raise SystemExit(1)


def build_store(store: pathlib.Path, manifest: dict) -> None:
    """Copy the checkout's converted files to their object keys under ``store``."""
    for key, entry in manifest["artifacts"].items():
        if entry["stage"] != "converted":
            continue
        source = CHECKOUT / "data" / "converted" / key
        if hashlib.sha256(source.read_bytes()).hexdigest() != entry["sha256"]:
            fail(f"{source} does not match the installed manifest; rebuild the manifest")
        target = store / entry["object"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    os.environ["GIA_MIP_BASE_URL"] = store.resolve().as_uri() + "/"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--store", type=pathlib.Path, default=None,
                        help="build a file:// store here from the checkout's converted files")
    parser.add_argument("--all-converted", action="store_true",
                        help="locate every converted file of the manifest")
    args = parser.parse_args(argv)

    if "GIA_MIP_DATA_ROOT" not in os.environ:
        os.environ["GIA_MIP_DATA_ROOT"] = tempfile.mkdtemp(prefix="gia-mip-check-")

    import numpy as np

    import gia_mip
    from gia_mip import data
    from gia_mip.benchmarks import metrics
    from gia_mip.benchmarks.results import BenchmarkResult, SpadaResult

    # 1. The installation, not the checkout, and no netCDF4.
    installed = pathlib.Path(gia_mip.__file__).resolve()
    if CHECKOUT in installed.parents:
        fail(f"gia_mip is imported from the checkout: {installed}")
    if data.checkout_root() is not None:
        fail("gia_mip.data sees a checkout")
    try:
        import netCDF4  # noqa: F401
    except ImportError:
        pass
    else:
        fail("netCDF4 is importable; the check must run with the wheel's dependencies only")
    print(f"gia_mip {gia_mip.__version__} from {installed.parent}")
    print(f"data root {os.environ['GIA_MIP_DATA_ROOT']}")

    if args.store is not None:
        build_store(args.store, data.load_manifest())
        print(f"store {os.environ['GIA_MIP_BASE_URL']}")

    # 2. A case, its reference through the cache, a synthetic run scored.
    case = gia_mip.case("martinec2018-B")
    reference = case.reference()
    path = BenchmarkResult.reference_path("B")
    if pathlib.Path(os.environ["GIA_MIP_DATA_ROOT"]) not in path.parents:
        fail(f"the reference was not read from the download root: {path}")
    run = BenchmarkResult("B", "synthetic")
    for (profile, quantity), curve in reference.profiles.items():
        run.add_profile(profile, quantity, curve.axis, curve.values + 0.01)
    rows = case.score(run)
    if len(rows) != 5:
        fail(f"case B gave {len(rows)} score rows, expected 5")
    for row in rows:
        if abs(row["max"] - 0.01) > 1e-9 or abs(row["rms"] - 0.01) > 1e-9:
            fail(f"a constant offset of 0.01 m scored {row}")
    envelope = case.envelope_check(run, band_half_width_deg=6.74)
    if len(envelope) != 5:
        fail(f"criterion 2 gave {len(envelope)} rows, expected 5")
    mean, spread = metrics.uniform_layer("B")
    if abs(mean + 119.89677) > 1e-5 or spread > 1e-6:
        fail(f"the uniform layer of B is {mean} m with a spread of {spread} m")
    print(f"case B scored: {len(rows)} rows, criterion 2 {sum(r['inside'] for r in envelope)} "
          f"of 5 inside, h_UF {mean:.6f} m")

    # 3. A result file written and read back through h5netcdf.
    written = run.write(pathlib.Path(tempfile.mkdtemp()) / "martinec2018-B.nc")
    back = BenchmarkResult.read(written, "synthetic", case="B")
    for key, curve in run.profiles.items():
        if not np.array_equal(back.profiles[key].values, curve.values):
            fail(f"{key} changed through a write and a read")
    print(f"wrote and read back {written.name}")

    # 4. Spada: a contributor against itself scores zero.
    gs = SpadaResult.reference("t02-01", "GS")
    spada_rows = metrics.spada_score(gs, {"GS": gs})
    if not spada_rows or any(r["max"] != 0.0 for r in spada_rows):
        fail("GS against itself does not score zero in t02-01")
    print(f"Spada t02-01: {len(spada_rows)} rows of zero")

    # 5. Every converted file of the manifest.
    if args.all_converted:
        manifest = data.load_manifest()
        keys = [k for k, e in manifest["artifacts"].items() if e["stage"] == "converted"]
        for key in keys:
            data.locate(key, "converted")
        print(f"located all {len(keys)} converted files")
    print("installed gia-mip: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
