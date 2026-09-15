# %% [raw]
# ---
# title: "Spada et al. (2011), test 1/2"
# subtitle: "Geodetic quantities: the inputs"
# ---

# %% [markdown]
# Test 1/2 of Spada et al. (2011) asks for the vertical displacement $U$, the
# horizontal displacement $V$ and the geoid displacement $N$ at the surface of
# the earth, as functions of colatitude. The test applies the load at $t = 0$
# as a Heaviside step and then holds it constant.
#
# This page shows the inputs of the test: the settings of the case file, the
# loads and the earth model. It shows no results. The code cells read every
# value through `gia_mip.benchmarks.inputs.SpadaCase`, which reads
# `cases/benchmarks/spada2011/t02-01.json`, `loads/spada2011.json` and
# `models/m3-l70-v01.json`. The test suite and the site build execute this
# page, so a change to those files or to `SpadaCase` that breaks a cell fails
# both.

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

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from IPython.display import Markdown
from matplotlib.cm import ScalarMappable

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


case = SpadaCase("t02-01")
case

# %% [markdown]
# ## The settings of the case file
#
# The case file gives the outputs, the epochs, the output grids and the range
# of harmonic degrees. `SpadaCase` turns each entry into numbers.

# %%
near = case.near_field_colatitudes_deg()
whole = case.global_colatitudes_deg()
lowest, highest = case.harmonic_degrees
# The fluid limit is math.inf, a different epoch from any finite time.
epochs = ", ".join("∞" if math.isinf(e) else f"{e:g}" for e in case.epochs_kyr)
centre_colatitude, centre_longitude = case.load_centre

Markdown("\n".join([
    "| Setting | Value |",
    "| --- | --- |",
    f"| Outputs | {', '.join(case.outputs)} |",
    f"| Loads | {', '.join(case.loads)} |",
    f"| Load centre | colatitude {centre_colatitude:g}°, longitude {centre_longitude:g}° |",
    f"| Epochs | {epochs} kyr |",
    f"| Harmonic degrees | {lowest} to {highest} |",
    f"| Near-field grid | {near[0]:g}° to {near[-1]:g}° at {near[1] - near[0]:g}°, {near.size} points |",
    f"| Global grid | {whole[0]:g}° to {whole[-1]:g}° at {whole[1] - whole[0]:g}°, {whole.size} points |",
    f"| Ice density | {case.ice_density:g} kg/m³ |",
    f"| Earth radius | {case.earth_radius_m / 1e3:g} km |",
]))

# %% [markdown]
# ## The loads
#
# The test uses three loads, each centred on the north pole. Each load is
# axisymmetric about its centre, so the colatitude $\theta$ is also the angle
# from the centre. The load table gives each load as a surface mass density
# $\sigma(\theta)$ in kg/m², with $\rho_i$ the ice density, $h$ the thickness
# at the centre, $\alpha$ the angular radius and $a$ the earth radius.
#
# The thickness of the cap falls from $h$ at the centre to zero at $\alpha$:
#
# $$\sigma(\theta) = \rho_i h \sqrt{\frac{\cos\theta - \cos\alpha}{1 - \cos\alpha}}, \qquad 0 \le \theta \le \alpha.$$
#
# The disc has the thickness $h$ everywhere inside $\alpha$:
#
# $$\sigma(\theta) = \rho_i h, \qquad 0 \le \theta \le \alpha.$$
#
# Both are zero for $\theta > \alpha$. The point load is a delta function at
# the pole with mass $m$:
#
# $$\sigma(\theta) = \frac{m}{2\pi a^2}\,\delta(1 - \cos\theta).$$

# %%
table = case.load_table["loads"]
rows = ["| Load | Thickness $h$ | Angular radius $\\alpha$ | $\\rho_i h$ | Mass |",
        "| --- | ---: | ---: | ---: | ---: |"]
for load in ("cap", "disc"):
    spec = table[load]
    rows.append(f"| {spec['name']} | {spec['thickness_m']:g} m | {spec['angular_radius_deg']:g}° | "
                f"{case.ice_density * spec['thickness_m']:,.0f} kg/m² | "
                f"{power_of_ten(case.load_mass(load), 5)} kg |")
rows.append(f"| {table['point']['name']} | | | | {power_of_ten(case.load_mass('point'), 5)} kg |")
Markdown("\n".join(rows))

# %% [markdown]
# ### The cap and the disc on the sphere
#
# The figure shows the sphere from the north pole to colatitude 60°, to scale.
# The colour gives the surface mass density, on one scale for both loads, and
# grey marks the surface with no load. The light falls along the axis of the
# loads. The surface darkens away from the pole, which shows its curvature,
# and the loads keep the colours of the colour bar.
#
# The point load has no surface density at a point, so the figure does not
# show it. A solver takes the point load through its Legendre coefficients,
# $\sigma_n = m\,(2n + 1) / (4\pi a^2)$.

# %%
# Colatitude from the load centre, in degrees. The figure stops at 60 degrees,
# far enough from the pole to show the curvature of the sphere. Both loads
# change only inside alpha = 10 degrees, so out to 15 degrees the grid uses the
# 0.25 degree near-field spacing of the test. Beyond that the density is zero,
# and a 1 degree grid draws the same colour with fewer faces.
colatitude_max = 60.0
colatitude = np.concatenate([np.arange(0.0, 15.0, 0.25),
                             np.arange(15.0, colatitude_max + 1e-9, 1.0)])
longitude = np.arange(0.0, 360.0 + 1e-9, 2.0)

# plot_surface colours each face with the value at its first corner. The
# density is therefore evaluated at the middle of each colatitude band, so that
# the band from 10.00 to 10.25 degrees, which lies outside the disc, does not
# take the disc's colour from its corner at exactly 10 degrees. The last row
# has no band below it and plot_surface never uses it for a face.
band_middle = np.append(0.5 * (colatitude[:-1] + colatitude[1:]), colatitude[-1])

# Points on the unit sphere. The picture needs only the shape, so the radius
# is 1 and not the earth radius.
theta, phi = np.meshgrid(np.radians(colatitude), np.radians(longitude), indexing="ij")
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)

# Surface mass density in 10^6 kg/m2, the unit of the colour bar.
density = {load: case.surface_density(load, band_middle) / 1e6 for load in ("cap", "disc")}

# One colour scale for both panels, so that one colour means one density. The
# ramp is a single hue from light to dark. It starts above white, so that the
# thin outer edge of the cap stays distinct from the grey of the unloaded
# surface.
norm = mcolors.Normalize(vmin=0.0, vmax=max(float(d.max()) for d in density.values()))
ramp = mcolors.LinearSegmentedColormap.from_list(
    "ice", plt.get_cmap("Blues")(np.linspace(0.25, 1.0, 256)))
no_load = mcolors.to_rgba("#d4d4d4")

# The light falls from above the pole, along the axis of the loads, and a face
# has the brightness 0.65 + 0.35 cos(theta). Inside alpha = 10 degrees that is
# at least 0.995, so the loads keep the colours of the colour bar to within
# half a per cent, and the grey surface darkens towards 60 degrees.
brightness = 0.65 + 0.35 * np.cos(theta)

fig = plt.figure(figsize=(9.0, 4.6))
axes = [fig.add_subplot(1, 2, i + 1, projection="3d") for i in range(2)]
for ax, load in zip(axes, ("cap", "disc")):
    sigma = density[load]
    # One colour per colatitude band, repeated round every longitude, because
    # the load is axisymmetric.
    band_colours = ramp(norm(sigma))
    band_colours[sigma == 0.0] = no_load
    # np.array makes a writable copy: broadcast_to returns a read-only view,
    # and the shading changes the colours in place.
    facecolors = np.array(np.broadcast_to(band_colours[:, None, :], theta.shape + (4,)))
    facecolors[..., :3] *= brightness[..., None]
    # rcount and ccount default to 50, which thins the grid and moves the edge
    # of the load. Setting them to the grid size draws every face. shade=False
    # leaves the colours as computed above.
    ax.plot_surface(x, y, z, facecolors=facecolors, rcount=theta.shape[0],
                    ccount=theta.shape[1], shade=False, linewidth=0, antialiased=False)
    spec = table[load]
    ax.set_title(f"{load.capitalize()}: $h$ = {spec['thickness_m']:g} m, "
                 f"$\\alpha$ = {spec['angular_radius_deg']:g}°")
    # Axis limits and box aspect from the extent of the surface, so that a unit
    # has the same length along every axis and the sphere keeps its shape.
    rim = math.sin(math.radians(colatitude_max))
    lowest_z = math.cos(math.radians(colatitude_max))
    ax.set_xlim(-rim, rim)
    ax.set_ylim(-rim, rim)
    ax.set_zlim(lowest_z, 1.0)
    ax.set_box_aspect((2.0 * rim, 2.0 * rim, 1.0 - lowest_z), zoom=1.4)
    ax.view_init(elev=40, azim=-60)
    ax.set_axis_off()

fig.colorbar(ScalarMappable(norm=norm, cmap=ramp), ax=axes, shrink=0.7, pad=0.02,
             label="Surface mass density (10⁶ kg/m²)")
plt.show()

# %% [markdown]
# ### A check on the loads
#
# The mass of an axisymmetric load follows from its surface density,
#
# $$M = 2\pi a^2 \int_0^\pi \sigma(\theta)\,\sin\theta\,\mathrm{d}\theta,$$
#
# and from its degree-0 Legendre coefficient, $M = 4\pi a^2 \sigma_0$. The next
# cell computes both with `SpadaCase` and compares them with the closed form of
# `SpadaCase.load_mass`. The load table gives the cap and the disc the same
# mass, and the cell checks that too.

# %%
a = case.earth_radius_m
rows = ["| Load | Integral of $\\sigma$ | $4\\pi a^2 \\sigma_0$ | Closed form |",
        "| --- | ---: | ---: | ---: |"]
for load in ("cap", "disc"):
    cos_alpha = math.cos(math.radians(float(table[load]["angular_radius_deg"])))
    # With x = cos(theta), sin(theta) dtheta = -dx, and the density is zero
    # for x < cos(alpha), so the integral runs over [cos(alpha), 1]. The
    # midpoint rule never evaluates the density exactly at the load edge,
    # where rounding in arccos could put the point on either side. The cap's
    # square-root edge makes the rule converge as N^-1.5, which leaves a
    # relative error of about 3e-9 with 10^5 intervals.
    edges = np.linspace(cos_alpha, 1.0, 100_001)
    middles = 0.5 * (edges[:-1] + edges[1:])
    sigma = case.surface_density(load, np.degrees(np.arccos(middles)))
    by_integral = 2.0 * math.pi * a**2 * float(np.sum(sigma * np.diff(edges)))
    by_coefficient = 4.0 * math.pi * a**2 * float(case.legendre_coefficients(load, [0])[0])
    closed = case.load_mass(load)
    assert math.isclose(by_integral, closed, rel_tol=1e-6), (load, by_integral, closed)
    assert math.isclose(by_coefficient, closed, rel_tol=1e-12), (load, by_coefficient, closed)
    rows.append(f"| {load} | {power_of_ten(by_integral, 8)} kg | "
                f"{power_of_ten(by_coefficient, 8)} kg | {power_of_ten(closed, 8)} kg |")
assert math.isclose(case.load_mass("cap"), case.load_mass("disc"), rel_tol=1e-12)
Markdown("\n".join(rows))

# %% [markdown]
# ## The earth model
#
# Test 1/2 uses M3-L70-V01, the earth model of both published benchmarks. It
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
