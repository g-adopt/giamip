"""The package imports, and the parts that must not pull in anything heavy do not.

These checks are cheap and they catch the two mistakes that break a clean
install: a syntax error anywhere in the package, and an import added to
`gia_mip.data`, which has to run on a bare CI runner before the dependencies
are installed.
"""

import ast
import pathlib
import subprocess
import sys

import gia_mip


def test_version_is_reported():
    """The package exposes a version, so a submission can record which one made it."""
    assert gia_mip.__version__


def test_data_module_imports_only_the_standard_library():
    """`gia_mip.data` must run before numpy or netCDF4 exist on the machine.

    Checked by reading the source rather than by importing, because an import
    inside a function would pass an import-time check and still break a bare
    runner on first use.
    """
    source = (pathlib.Path(gia_mip.__file__).parent / "data.py").read_text()
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    outside = imported - set(sys.stdlib_module_names)
    assert not outside, f"gia_mip.data imports non-standard modules: {sorted(outside)}"


def test_data_module_runs_with_no_dependencies_importable():
    """`python -m gia_mip.data --list` works on a machine with nothing installed.

    Run under `-I`, which ignores the user site directory and PYTHONPATH, with
    the package's parent put back on the path explicitly. Without `-I` the
    ambient environment supplies numpy and netCDF4 and the test proves nothing
    about a bare runner.
    """
    package_parent = str(pathlib.Path(gia_mip.__file__).parent.parent)
    result = subprocess.run(
        [sys.executable, "-I", "-c",
         f"import sys; sys.path.insert(0, {package_parent!r}); "
         "import runpy; sys.argv = ['gia_mip.data', '--list']; "
         "runpy.run_module('gia_mip.data', run_name='__main__')"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "numpy" not in result.stderr
