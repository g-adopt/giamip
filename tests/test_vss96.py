"""VSS96, the earth model of Spada et al. (2011) test 6/1, is the archive's table.

`gia_mip/resources/models/vss96.json` is a transcription of `T01-06/VSS96.dat` in the Spada
archive. Two tests tie it to the archive. The first compares every value with
the file. The second compares the viscosity with the profile that figure 1 of
the paper draws, `T01-06/vis-vss96.dat`. The second test also fixes how a row
of the file is read: its radius is the top of a layer. Both need the unpacked
archive and are marked `data`.

The first test in this module needs no data. It checks the shape of the table
that a solver reads through `SpadaCase`.
"""

import hashlib
import math
import pathlib
import re

import numpy as np
import pytest

from gia_mip import cases
from gia_mip.benchmarks.inputs import SpadaCase

# The directory of test 6/1 inside the data tree.
T01_06 = pathlib.Path("original") / "spada2011" / "spada2011-cost-es0701" / "T01-06"


def _test_directory(data_root: pathlib.Path) -> pathlib.Path:
    """The unpacked T01-06 directory, or a skip when the archive is absent."""
    directory = data_root / T01_06
    if not directory.is_dir():
        pytest.skip(f"{directory} is not unpacked")
    return directory


def _rows(path: pathlib.Path) -> list[list[str]]:
    """The rows of VSS96.dat as text: radius, density, rigidity and viscosity.

    Lines that start with `!` are comments, and every value is quoted. The text
    is kept, so that a test compares with the file's own numbers.
    """
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.strip().startswith("!"):
            continue
        values = re.findall(r"'([^']*)'", line)
        assert len(values) == 4, line
        rows.append(values)
    return rows


def test_the_vss96_layers_are_an_elastic_lithosphere_28_maxwell_layers_and_a_fluid_core():
    """VSS96 read through SpadaCase: surface first, no gaps, and the three rheologies.

    The case file calls VSS96 a 28-layer model, and the 28 are the Maxwell
    layers of the mantle.
    """
    layers = SpadaCase("t01-06").earth_layers()
    assert list(layers["rheology"]) == ["elastic"] + ["maxwell"] * 28 + ["inviscid fluid"]
    assert layers["top_radius_m"][0] == pytest.approx(6.371e6)
    assert layers["bottom_radius_m"][-1] == 0.0
    # Each layer starts at the radius where the layer above it ends.
    np.testing.assert_array_equal(layers["bottom_radius_m"][:-1], layers["top_radius_m"][1:])
    assert np.all(layers["top_radius_m"] > layers["bottom_radius_m"])
    # The lithosphere has infinite viscosity, the core zero shear modulus and
    # zero viscosity, and every mantle layer a finite positive viscosity.
    assert math.isinf(layers["viscosity_pa_s"][0])
    assert layers["shear_modulus_pa"][-1] == 0.0 and layers["viscosity_pa_s"][-1] == 0.0
    mantle = layers["viscosity_pa_s"][1:-1]
    assert np.all(np.isfinite(mantle)) and np.all(mantle > 0.0)
    # The stored thickness is the difference of the two radii.
    for layer in cases.earth_model("vss96")["layers"]:
        assert layer["thickness_km"] == layer["top_radius_km"] - layer["bottom_radius_km"], layer


@pytest.mark.data
def test_the_vss96_table_is_the_archive_file(data_root):
    """Every radius, density, rigidity and viscosity equals VSS96.dat exactly.

    The file's SHA-256 must be the one the model records, so a changed file
    fails here before its values are compared.
    """
    path = _test_directory(data_root) / "VSS96.dat"
    model = cases.earth_model("vss96")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == model["source"]["transcribed_from"]["sha256"]
    rows = _rows(path)
    layers = model["layers"]
    assert len(rows) == len(layers)
    for i, ((radius, density, rigidity, viscosity), layer) in enumerate(zip(rows, layers)):
        assert layer["top_radius_km"] * 1e3 == float(radius), i
        # A layer extends down to the radius of the next row; the core to the centre.
        bottom = float(rows[i + 1][0]) if i + 1 < len(rows) else 0.0
        assert layer["bottom_radius_km"] * 1e3 == bottom, i
        assert layer["density_kg_m3"] == float(density), i
        assert layer["shear_modulus_pa"] == float(rigidity), i
        if viscosity == "infinity":
            assert layer["viscosity_pa_s"] is None and layer["rheology"] == "elastic", i
        else:
            assert layer["viscosity_pa_s"] == float(viscosity), i


@pytest.mark.data
def test_the_vss96_viscosity_is_the_profile_that_figure_1_draws(data_root):
    """Every sample of vis-vss96.dat equals the viscosity of the layer it lies in.

    T01-06/visco.gmt draws figure 1 of the paper from vis-vss96.dat: the
    viscosity at 5001 radii, in km, from the centre to the surface. That file
    writes the elastic lithosphere as 1e35 Pa s and the fluid core as 0. A
    sample at an interface belongs to the layer below it. Every layer holds at
    least one sample, so a layer with a wrong viscosity or a wrong radius fails.

    The same samples read against the other convention, a row's radius as the
    bottom of its layer, give a sample the viscosity of the next row down.
    Every sample above the core must then disagree, which shows that the
    comparison can tell the two readings apart. The core has no row below it,
    so its samples, more than half of the file, are left out of that count.
    """
    profile = np.loadtxt(_test_directory(data_root) / "vis-vss96.dat")
    layers = cases.earth_model("vss96")["layers"]
    top = np.array([layer["top_radius_km"] for layer in layers], dtype=float)
    bottom = np.array([layer["bottom_radius_km"] for layer in layers], dtype=float)
    viscosity = np.array([1e35 if layer["viscosity_pa_s"] is None else layer["viscosity_pa_s"]
                          for layer in layers], dtype=float)

    hits = np.zeros(len(layers), dtype=int)
    above_core = 0
    disagree_with_other_reading = 0
    for radius_km, value in profile:
        inside = np.flatnonzero((bottom < radius_km) & (radius_km <= top))
        # Only the centre, r = 0, lies in no half-open interval; it is core.
        i = int(inside[0]) if inside.size else len(layers) - 1
        hits[i] += 1
        assert value == pytest.approx(viscosity[i], rel=1e-12, abs=0.0), (radius_km, value, viscosity[i])
        # The other reading gives a sample the viscosity of the next row down.
        if i + 1 < len(layers):
            above_core += 1
            if value != pytest.approx(viscosity[i + 1], rel=1e-12, abs=0.0):
                disagree_with_other_reading += 1
    assert np.all(hits > 0), hits
    assert above_core > 0
    assert disagree_with_other_reading == above_core, (disagree_with_other_reading, above_core)
