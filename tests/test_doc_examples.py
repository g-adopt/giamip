"""The code examples a solver developer copies must run.

Two examples are executed here, taken from the documents themselves so that a
change to the API that breaks them fails a test:

* the example under "How to run a case" in README.md, which is also the page
  of the package on PyPI;
* the example marked ``# example: solver benchmark test`` in
  docs/reference/python-interface.qmd, both stages of a solver's benchmark
  test.

The examples name the solver's own pieces (``my_curve``, ``my_profile``,
``my_series``, the points and the output directory). Here each is a stand-in
that returns the reference itself, evaluated at the solver's points with a
cubic spline, so every check in the examples passes.
"""

from __future__ import annotations

import pathlib
import re

import numpy as np
from scipy.interpolate import CubicSpline

from gia_mip.benchmarks.results import BenchmarkResult

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _python_blocks(text: str) -> list[str]:
    """Every fenced ```python block of a Markdown or Quarto text."""
    return re.findall(r"^```python\n(.*?)^```", text, flags=re.MULTILINE | re.DOTALL)


def _reference_at(letter: str, profile: str, quantity: str, colatitude: np.ndarray) -> np.ndarray:
    """The stored reference curve at a solver's colatitudes."""
    curve = BenchmarkResult.reference(letter).profile(profile, quantity)
    return CubicSpline(curve.axis, curve.values)(colatitude)


def test_the_readme_example_runs(tmp_path, monkeypatch):
    blocks = [b for b in _python_blocks((ROOT / "README.md").read_text())
              if "gia_mip.case(" in b]
    assert len(blocks) == 1, "README.md should hold one example of gia_mip.case"
    monkeypatch.chdir(tmp_path)
    colatitude_deg = np.linspace(0.0, 180.0, 1801)
    namespace = {
        "colatitude": np.array([20.0, 100.0]), "longitude": np.array([75.0, 320.0]),
        "t_kyr": 5.0, "colatitude_deg": colatitude_deg,
        "my_curve": lambda profile, quantity: _reference_at("B", profile.name, quantity,
                                                           colatitude_deg),
    }
    exec(compile(blocks[0], "README.md", "exec"), namespace)
    assert (tmp_path / "martinec2018-B.nc").exists()
    assert namespace["rows"] and all(r["max"] < 1e-3 for r in namespace["rows"])
    assert all(r["inside"] for r in namespace["checks"])


def test_the_solver_benchmark_example_runs(tmp_path):
    text = (ROOT / "docs" / "reference" / "python-interface.qmd").read_text()
    blocks = [b for b in _python_blocks(text) if "# example: solver benchmark test" in b]
    assert len(blocks) == 1, "the interface page should hold the solver example once"
    colatitude_deg = np.linspace(0.0, 180.0, 1801)

    def my_profile(profile, quantity):
        return colatitude_deg, _reference_at("C", profile.name, quantity, colatitude_deg)

    def my_series(quantity):
        curve = BenchmarkResult.reference("C").series(quantity)
        time_kyr = np.linspace(0.0, 15.0, 151)
        return time_kyr, np.interp(time_kyr, curve.axis, curve.values)

    namespace = {"my_profile": my_profile, "my_series": my_series, "output_dir": tmp_path}
    exec(compile(blocks[0], "python-interface.qmd", "exec"), namespace)
    assert (tmp_path / "martinec2018-C.nc").exists()
    assert len(namespace["rows"]) == 7          # 5 profile curves and 2 time series
    assert namespace["vega"].values.size > 0
