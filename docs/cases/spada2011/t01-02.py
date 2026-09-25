# %% [raw]
# ---
# title: "Spada et al. (2011), test 2/1"
# subtitle: "Loading Love numbers: the inputs"
# ---

# %% [markdown]
# Test 2/1 of Spada et al. (2011) asks for the loading Love numbers $h$, $l$
# and $k$ of the earth model M3-L70-V01. For a surface load of harmonic degree
# $n$, $h$ scales the vertical displacement, $l$ the horizontal displacement
# and $k$ the perturbation of the gravitational potential.
#
# The test asks for the elastic and the fluid Love numbers, and for the residue
# of each relaxation mode. In the Laplace domain a Love number has the
# normal-mode form
#
# $$h(s) = h_e + \sum_{j} \frac{r_j}{s - s_j},$$
#
# with $h_e$ the elastic Love number, $s_j$ the relaxation frequencies of test
# 1/1 and $r_j$ the residues. The fluid Love number is the limit $s \to 0$,
# $h_f = h_e - \sum_j r_j / s_j$. The Love numbers $l$ and $k$ have the same
# form.
#
# This page shows the inputs of the test: the settings of the case file and
# the earth model. The page shows no results. The code cells read every value
# through `giamip.benchmarks.inputs.SpadaCase`, which reads
# `cases/benchmarks/spada2011/t01-02.json` and `models/m3-l70-v01.json`. The
# test suite and the site build execute this page, so a change to those files
# or to `SpadaCase` that breaks a cell fails both.

# %%
#| include: false
# Quarto and tests/test_pages.py execute this page with its own directory as
# the working directory. Put the repository root first on the import path, so
# that the page runs against the checkout it belongs to. In a worktree this
# matters: the editable install points at the main checkout.
import pathlib
import sys

ROOT = pathlib.Path("../../..").resolve()
if (ROOT / "giamip").is_dir():
    sys.path.insert(0, str(ROOT))

# %%
import math

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from IPython.display import Markdown

from giamip.benchmarks.inputs import SpadaCase


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


case = SpadaCase("t01-02")
case

# %% [markdown]
# ## The settings of the case file
#
# The case file gives the outputs and the range of harmonic degrees.

# %%
lowest, highest = case.harmonic_degrees

Markdown("\n".join([
    "| Setting | Value |",
    "| --- | --- |",
    f"| Outputs | {', '.join(f'`{o}`' for o in case.outputs)} |",
    f"| Harmonic degrees | {lowest} to {highest} |",
    f"| Relaxation modes per degree | {case.earth_model['relaxation_modes']['count']} |",
    f"| Earth model | {case.earth_model['name']} |",
]))

# %% [markdown]
# ## The earth model
#
# Test 2/1 uses M3-L70-V01, the earth model of both published benchmarks. It
# is spherically symmetric, incompressible and self-gravitating. It has five
# layers: an elastic lithosphere, three Maxwell viscoelastic mantle layers and
# an inviscid fluid core.

# %%
layers = case.earth_layers()
roles = [layer["role"] for layer in case.earth_model["layers"]]


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
# The figure draws each property against radius. Each layer has one value, so
# each profile is a sequence of steps. The viscosity axis is logarithmic and
# has no point for an infinite or a zero viscosity, so grey bands mark the
# elastic lithosphere and the fluid core in every panel. The names of the
# layers are beside the last panel.

# %%
top_km = layers["top_radius_m"] / 1e3
bottom_km = layers["bottom_radius_m"] / 1e3
maxwell = layers["rheology"] == "maxwell"
every_layer = np.ones(top_km.size, dtype=bool)


def steps(values: np.ndarray, keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return value and radius arrays that draw the kept layers as vertical steps.

    Each kept layer adds two points with its own value, one at its top radius
    and one at its bottom radius. Consecutive layers therefore join with a
    horizontal segment at the interface they share.
    """
    radius = np.column_stack([top_km[keep], bottom_km[keep]]).ravel()
    return np.repeat(values[keep], 2), radius


# Label, values, which layers to draw, and axis scale for each panel. The
# viscosity panel draws only the Maxwell layers, where the value is finite
# and positive.
panels = [
    ("Viscosity (Pa s)", layers["viscosity_pa_s"], maxwell, "log"),
    ("Density (kg/m³)", layers["density_kg_m3"], every_layer, "linear"),
    ("Shear modulus (GPa)", layers["shear_modulus_pa"] / 1e9, every_layer, "linear"),
]
line_colour = plt.get_cmap("Blues")(0.85)

fig, axes = plt.subplots(1, 3, figsize=(9.0, 5.0), sharey=True, constrained_layout=True)
for ax, (label, values, keep, scale) in zip(axes, panels):
    # Grey bands for the layers with no finite positive viscosity.
    for i in np.flatnonzero(~maxwell):
        ax.axhspan(bottom_km[i], top_km[i], color="0.92", linewidth=0, zorder=0)
    # A thin line at every interface, so that the three panels line up by eye.
    for radius in bottom_km[:-1]:
        ax.axhline(radius, color="0.78", linewidth=0.6, zorder=1)
    value, radius = steps(values, keep)
    ax.plot(value, radius, color=line_colour, linewidth=2.0, zorder=2)
    ax.set_xscale(scale)
    ax.set_xlabel(label)
    ax.grid(axis="x", color="0.9", linewidth=0.5)

# The mantle viscosities are 1e21 and 2e21 Pa s. A fixed range of decades
# keeps them readable on the log axis.
axes[0].set_xlim(3e20, 1e22)
# Density from zero, with room for the core at 10750 kg/m3.
axes[1].set_xlim(0.0, 12000.0)
# The core has zero shear modulus. The axis starts just below zero, so that
# the core's line stays clear of the left spine.
axes[2].set_xlim(-10.0, 250.0)
axes[0].set_ylim(0.0, top_km[0])
axes[0].set_ylabel("Radius (km)")

# Layer names beside the last panel, level with the middle of each layer. The
# x position is in axes units and the y position in radius.
beside = mtransforms.blended_transform_factory(axes[2].transAxes, axes[2].transData)
for i, role in enumerate(roles):
    rheology = layers["rheology"][i]
    name = role if rheology == "maxwell" else f"{role} ({rheology})"
    axes[2].text(1.03, 0.5 * (top_km[i] + bottom_km[i]), name, transform=beside,
                 ha="left", va="center", color="0.3", fontsize=9)
plt.show()
