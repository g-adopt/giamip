# %% [raw]
# ---
# title: "Spada et al. (2011), test 6/1"
# subtitle: "Love numbers for a multistratified model: the inputs"
# ---

# %% [markdown]
# Test 6/1 of Spada et al. (2011) asks for the quantities of test 5/1, the
# loading Love numbers $h$, $l$ and $k$ as functions of time, for the earth
# model VSS96 in place of M3-L70-V01. VSS96 has 28 viscoelastic mantle
# layers. With many layers the root-finding of the viscoelastic normal-mode
# method becomes hard, and the test checks that the codes still agree.
#
# This page shows the inputs of the test: the settings of the case file, the
# layers of VSS96, and VSS96 against M3-L70-V01. The page shows no results.
# The code cells read every value through
# `gia_mip.benchmarks.inputs.SpadaCase`, which reads
# `cases/benchmarks/spada2011/t01-06.json`, `models/vss96.json` and, for the
# comparison, `models/m3-l70-v01.json`. The test suite and the site build
# execute this page, so a change to those files or to `SpadaCase` that breaks
# a cell fails both.

# %%
#| include: false
# Quarto and tests/test_pages.py execute this page with its own directory as
# the working directory. Put the repository root first on the import path, so
# that the page runs against the checkout it belongs to. In a worktree this
# matters: the editable install points at the main checkout.
import pathlib
import sys

ROOT = pathlib.Path("../../..").resolve()
if (ROOT / "gia_mip").is_dir():
    sys.path.insert(0, str(ROOT))

# %%
import math

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown

from gia_mip.benchmarks.inputs import SpadaCase


def power_of_ten(value: float, digits: int) -> str:
    """Return a positive number as LaTeX, m × 10^e, for the tables on this page.

    Python writes 3.6072e+18, which reads badly in a table. The mantissa keeps
    ``digits`` significant figures, and a mantissa of exactly 1 is left out.
    """
    exponent = math.floor(math.log10(value))
    mantissa = f"{value / 10**exponent:.{digits}g}"
    # Rounding can carry the mantissa up to 10, for example 9.99999 with
    # three digits. Move that carry into the exponent.
    if float(mantissa) >= 10.0:
        exponent += 1
        mantissa = f"{value / 10**exponent:.{digits}g}"
    if mantissa == "1":
        return f"$10^{{{exponent}}}$"
    return f"${mantissa} \\times 10^{{{exponent}}}$"


case = SpadaCase("t01-06")
case

# %% [markdown]
# ## The settings of the case file
#
# The case file gives the outputs, the range of harmonic degrees and the time
# window.

# %%
lowest, highest = case.harmonic_degrees
start_kyr, end_kyr = case.parameters["time_window_kyr"]

Markdown("\n".join([
    "| Setting | Value |",
    "| --- | --- |",
    f"| Outputs | {', '.join(f'`{o}`' for o in case.outputs)} |",
    f"| Harmonic degrees | {lowest} to {highest} |",
    f"| Time window | {power_of_ten(start_kyr, 3)} to {power_of_ten(end_kyr, 3)} kyr |",
    f"| Earth model | {case.earth_model['name']} |",
]))

# %% [markdown]
# ## The earth model
#
# VSS96 is spherically symmetric, incompressible and self-gravitating. It has
# an elastic lithosphere, 28 Maxwell viscoelastic mantle layers and an
# inviscid fluid core. `models/vss96.json` holds its layers, transcribed from
# `T01-06/VSS96.dat` in the archive of the benchmark. `tests/test_vss96.py`
# checks the transcription against that file, and against the viscosity
# profile that figure 1 of the paper draws.

# %%
layers = case.earth_layers()
roles = [layer["role"] for layer in case.earth_model["layers"]]
# The case file calls VSS96 a 28-layer model. The 28 are the Maxwell layers
# of the mantle, between the elastic lithosphere and the fluid core.
assert int(np.sum(layers["rheology"] == "maxwell")) == 28


def viscosity_text(eta: float) -> str:
    """Return the viscosity of one layer as table text, naming the two limits.

    An elastic layer has infinite viscosity and the fluid core has zero
    viscosity. Both are written with the rheology they stand for, because a
    bare "inf" or "0" hides what the number means.
    """
    if math.isinf(eta):
        return "∞ (elastic)"
    if eta == 0.0:
        return "0 (inviscid fluid)"
    return power_of_ten(eta, 3)


rows = ["| Layer | Top radius (km) | Bottom radius (km) | Density (kg/m³) | Shear modulus (GPa) | Viscosity (Pa s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
for i, role in enumerate(roles):
    rows.append(f"| {role} | {layers['top_radius_m'][i] / 1e3:g} | {layers['bottom_radius_m'][i] / 1e3:g} | "
                f"{layers['density_kg_m3'][i]:g} | {layers['shear_modulus_pa'][i] / 1e9:.4g} | "
                f"{viscosity_text(layers['viscosity_pa_s'][i])} |")
Markdown("\n".join(rows))

# %% [markdown]
# ## VSS96 and M3-L70-V01
#
# Figure 1 of the paper compares the viscosity of VSS96 with that of
# M3-L70-V01. The figure below draws the viscosity, the density and the shear
# modulus of both models against radius. Each layer has one value, so each
# profile is a sequence of steps. The radius axis starts at 3000 km, below the
# core-mantle boundary, which the two models share, and the grey band marks
# the fluid core. The viscosity panel has no point in the elastic
# lithospheres, which differ in thickness between the two models.

# %%
# Test 5/1 asks for the quantities of test 6/1 on M3-L70-V01, so its case gives
# the model to compare with.
comparison = SpadaCase("t01-05")
assert comparison.earth_model["id"] == "m3-l70-v01", comparison.earth_model["id"]

# Name, layers and line colour of each model. The two colours are the first
# two slots of a categorical palette that stays distinct for readers with a
# colour vision deficiency.
models = [
    (case.earth_model["name"], layers, "#2a78d6"),
    (comparison.earth_model["name"], comparison.earth_layers(), "#eb6834"),
]

# The grey band for the core assumes that both models put the core-mantle
# boundary at the same radius, so check that they do.
core_tops = {float(model_layers["top_radius_m"][-1]) for _, model_layers, _ in models}
assert len(core_tops) == 1, core_tops
core_top_km = core_tops.pop() / 1e3
surface_km = float(layers["top_radius_m"][0]) / 1e3
lowest_radius_km = 3000.0


def steps(model_layers: dict, values: np.ndarray, keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return value and radius arrays that draw the kept layers of one model as steps.

    Each kept layer adds two points with its own value, one at its top radius
    and one at its bottom radius, both in km. Consecutive layers therefore join
    with a horizontal segment at the interface they share.
    """
    radius = np.column_stack([model_layers["top_radius_m"][keep],
                              model_layers["bottom_radius_m"][keep]]).ravel() / 1e3
    return np.repeat(values[keep], 2), radius


fig, axes = plt.subplots(1, 3, figsize=(9.0, 5.4), sharey=True, constrained_layout=True)
for name, model_layers, colour in models:
    maxwell = model_layers["rheology"] == "maxwell"
    every_layer = np.ones(maxwell.size, dtype=bool)
    # Values and the layers to draw, per panel. The viscosity panel draws only
    # the Maxwell layers, where the viscosity is finite and positive.
    per_panel = [
        (model_layers["viscosity_pa_s"], maxwell),
        (model_layers["density_kg_m3"], every_layer),
        (model_layers["shear_modulus_pa"] / 1e9, every_layer),
    ]
    for ax, (values, keep) in zip(axes, per_panel):
        value, radius = steps(model_layers, values, keep)
        ax.plot(value, radius, color=colour, linewidth=2.0, label=name, zorder=2)

for ax, label, scale in zip(axes, ("Viscosity (Pa s)", "Density (kg/m³)", "Shear modulus (GPa)"),
                            ("log", "linear", "linear")):
    ax.axhspan(lowest_radius_km, core_top_km, color="0.92", linewidth=0, zorder=0)
    ax.set_xscale(scale)
    ax.set_xlabel(label)
    ax.grid(axis="x", color="0.9", linewidth=0.5)

# The viscosity range of figure 1 of the paper, 1e20 to 1e23 Pa s.
axes[0].set_xlim(1e20, 1e23)
# Density from zero, with room for the cores near 11000 kg/m3. Ticks every
# 4000 kg/m3, because the default ticks every 2000 run into each other at the
# width of one panel.
axes[1].set_xlim(0.0, 12000.0)
axes[1].set_xticks([0.0, 4000.0, 8000.0, 12000.0])
# Shear modulus from just below zero, so that the core's zero stays clear of
# the left spine, to above the largest mantle value of VSS96.
axes[2].set_xlim(-10.0, 320.0)
axes[0].set_ylim(lowest_radius_km, surface_km)
axes[0].set_ylabel("Radius (km)")
axes[1].text(6000.0, 0.5 * (lowest_radius_km + core_top_km), "fluid core", ha="center",
             va="center", color="0.35")
# One legend for the figure, above the panels, because every panel holds the
# same two models.
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=2, frameon=False)
plt.show()
