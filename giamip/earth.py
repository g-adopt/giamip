"""The radial earth structure of a GIAMIP case, at any radius.

A GIAMIP case names an elastic structure, for example PREM-stable, and a
viscosity structure, for example VM5a. This module reads both and returns, for
any array of radii, the density, the elastic moduli, the wave speeds, the
gravity and the viscosity at those radii. Every solver that runs a case can
therefore sample the same earth in the same way, on its own mesh or grid.

    >>> from giamip import earth
    >>> e = earth.for_case("giamip-exp01")                 # doctest: +SKIP
    >>> values = e([6371e3, 6000e3, 3480e3])                # doctest: +SKIP
    >>> values["rho_kg_m3"], values["viscosity_pa_s"]       # doctest: +SKIP

The elastic structure is a table of nodes (a CSV file located through
`giamip.data`). Every column is interpolated linearly in radius between two
neighbouring nodes, so the density of a G-ADOPT run is continuous within each
region and linear between nodes. For density this is the rule that protocol
v4.0 Section 3.1 gives for PREM-stable. Between the surface and the core-mantle
boundary the gravity column of that file is computed for this density, and
gravity is interpolated linearly as well: there it differs from the gravity of
the linearly varying density by less than 3e-6 of its value, because the nodes
are at most 10 km apart. In the core the file takes mass and gravity from the
PREM polynomials, and nodes are up to 100 km apart, so linear interpolation of
gravity there differs from the file's own gravity by up to about 1.2e-4 of its
value. A mantle model never samples the core.

The viscosity structure is a set of layers in depth below the surface, each of
constant viscosity. The deepest layer reaches the core-mantle boundary of the
elastic structure, as protocol v4.0 states for VM5a and VM7.

At a boundary, where a property jumps, the value is not defined by the radius
alone. Every call takes ``side``: ``"above"`` gives the value just above the
boundary and ``"below"`` the value just below it. The default is ``"above"``.
With it, a mantle mesh with nodes on the core-mantle boundary gets mantle
values there, and a node on the surface gets the surface value.

The interpolation is vectorised. It uses one binary search per radius
(`numpy.searchsorted`) and no loop in Python, so a million radii take a
fraction of a second. It involves no tolerance and no random choice, so the
same radii always give the same values on every machine.

Every G-ADOPT run is compressible, so the bulk modulus that this module returns
is the bulk modulus of the file. A code that makes the incompressible
approximation ignores it.

A scalar radius gives 0-d arrays, so every value has the shape of the input
whether the input is a number or an array.

Laterally varying models (Lloyd et al. 2024 and PREM-stable with GLAD-M25
variations) are not handled here. Their grids are g-drift datasets, which the
model files name with their SHA-256, and a solver reads them from g-drift.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from giamip import cases, data

# The columns of an elastic structure file that are interpolated. `depth_km`
# follows from the radius and `region` is an integer label of the PREM region,
# so neither is interpolated.
ELASTIC_COLUMNS = ("rho_kg_m3", "vp_m_s", "vs_m_s", "kappa_Pa", "mu_Pa", "g_m_s2")

# The two sides of a boundary that a caller can ask for.
SIDES = ("above", "below")


def _check_side(side: str) -> None:
    """Raise unless ``side`` is one of `SIDES`."""
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, not {side!r}")


def _radii(radius, surface: float) -> np.ndarray:
    """Check the radii a caller passed and return them as a float array.

    Parameters
    ----------
    radius
        A number or an array of radii in metres.
    surface
        The radius of the surface in metres. A radius above it is outside the
        earth model and is an error.

    Returns
    -------
    numpy.ndarray
        The radii as float64, with the shape of the input.
    """
    r = np.asarray(radius, dtype=float)
    if np.isnan(r).any():
        raise ValueError("a radius is NaN")
    if (r < 0).any() or (r > surface).any():
        raise ValueError(
            f"every radius must lie between 0 and the surface, {surface:.1f} m. "
            f"Got values from {r.min():.6g} to {r.max():.6g} m. Radii are in metres.")
    return r


@dataclasses.dataclass(frozen=True)
class ElasticProfile:
    """An elastic structure given at nodes, interpolated linearly in radius.

    Attributes
    ----------
    model_id
        The id of the earth model file it was read from, for example
        ``prem-stable``.
    radius_m
        The radii of the nodes in metres, from the centre to the surface. A
        boundary appears twice: the first of the two rows holds the values
        below it and the second the values above.
    columns
        One array per name in `ELASTIC_COLUMNS`, aligned with ``radius_m``.
    """

    model_id: str
    radius_m: np.ndarray
    columns: dict[str, np.ndarray]

    def __post_init__(self):
        r = self.radius_m
        # Linear interpolation needs at least one interval, and every column
        # must have a value at every node.
        if r.ndim != 1 or len(r) < 2:
            raise ValueError(f"{self.model_id}: at least two nodes are needed")
        for name, column in self.columns.items():
            if column.shape != r.shape:
                raise ValueError(f"{self.model_id}: column {name} has {column.shape} "
                                 f"values for {r.shape} nodes")
        # Non-decreasing order is what the binary search needs. Equal radii
        # mark a boundary, and three equal radii would leave the middle row
        # unreachable, so they are rejected as a malformed file.
        if np.any(np.diff(r) < 0):
            raise ValueError(f"{self.model_id}: the radii must not decrease")
        same = np.diff(r) == 0
        if np.any(same[1:] & same[:-1]):
            raise ValueError(f"{self.model_id}: three nodes share one radius")
        if r[0] != 0.0:
            raise ValueError(f"{self.model_id}: the first node must be at the centre")
        # A boundary at the centre or at the surface would make the end
        # interval that `__call__` falls back to have zero length, and the
        # value there 0/0. It is also physically meaningless, so it is
        # rejected here, which is what lets `__call__` divide without a check.
        if r[1] == r[0] or r[-1] == r[-2]:
            raise ValueError(f"{self.model_id}: a boundary at the centre or the surface")

    @classmethod
    def from_model(cls, model_id: str) -> "ElasticProfile":
        """Read the elastic structure of an earth model file.

        The model file names its node table under ``profile.file``. The table
        is found through `giamip.data.locate`, which uses the tracked copy in a
        checkout and downloads it, with its SHA-256 checked, otherwise.

        Parameters
        ----------
        model_id
            An earth model id, for example ``prem-stable`` or ``prem``.

        Raises
        ------
        NotImplementedError
            For a laterally varying elastic structure, whose grid is not in
            this package.
        ValueError
            For a model that is not an elastic structure given at nodes.
        """
        model = cases.earth_model(model_id)
        if "tomography" in model:
            raise NotImplementedError(
                f"{model_id} varies laterally. Its grid is the g-drift dataset "
                f"that the model file names. Use its "
                f"radial base, {model.get('base')!r}, for a 1D run.")
        profile = model.get("profile")
        if model["kind"] != "elastic_structure" or profile is None:
            raise ValueError(f"{model_id} is not an elastic structure given at nodes")
        path = data.locate(profile["file"], profile["stage"])
        table = np.genfromtxt(path, delimiter=",", names=True)
        return cls(model_id=model_id,
                   radius_m=np.asarray(table["radius_m"], dtype=float),
                   columns={c: np.asarray(table[c], dtype=float) for c in ELASTIC_COLUMNS})

    @property
    def surface_radius_m(self) -> float:
        """The radius of the top node, which is the surface, in metres."""
        return float(self.radius_m[-1])

    @property
    def core_mantle_boundary_m(self) -> float:
        """The radius of the core-mantle boundary in metres.

        It is the top of the outer core, the largest radius at which the S-wave
        speed is zero. Taking it from the file, not from a constant, keeps the
        viscosity structure on the same boundary as the elastic structure.
        """
        fluid = self.columns["vs_m_s"] == 0.0
        if not fluid.any():
            raise ValueError(f"{self.model_id} has no fluid core")
        return float(self.radius_m[fluid].max())

    def __call__(self, radius, side: str = "above") -> dict[str, np.ndarray]:
        """The elastic structure at the given radii.

        Parameters
        ----------
        radius
            A number or an array of radii in metres, between 0 and the surface.
        side
            ``"above"`` or ``"below"``: which value to give at a radius that
            lies exactly on a boundary. Elsewhere it has no effect.

        Returns
        -------
        dict
            One array per name in `ELASTIC_COLUMNS`, each with the shape of
            ``radius``, in the units of the column names: kg/m3, m/s, Pa and
            m/s2.
        """
        _check_side(side)
        r = _radii(radius, self.surface_radius_m)
        nodes = self.radius_m
        n = len(nodes)
        if side == "above":
            # The last node at or below r. On a boundary this is the second of
            # its two rows, the one that holds the values above it, so the
            # interval [k, k+1] lies above the boundary.
            k = np.searchsorted(nodes, r, side="right") - 1
        else:
            # The node before the first node at or above r. On a boundary the
            # first node at or above r is the first of its two rows, which
            # holds the values below it, so the interval [k, k+1] lies below.
            k = np.searchsorted(nodes, r, side="left") - 1
        # At the surface with side "above" k is the last node, and at the
        # centre with side "below" it is -1. Clipping keeps the interval
        # inside the table, and the weight below then puts the value on the
        # end node.
        k = np.clip(k, 0, n - 2)
        lo, hi = nodes[k], nodes[k + 1]
        # The interval is never of zero length: a boundary's two rows are
        # never the two ends of the interval chosen above, and `__post_init__`
        # rejects a boundary in the first or the last interval, which the
        # clipping can fall back to.
        t = (r - lo) / (hi - lo)
        # np.asarray keeps a scalar radius as a 0-d array, the same type that
        # the viscosity structure returns.
        return {c: np.asarray((1.0 - t) * v[k] + t * v[k + 1])
                for c, v in self.columns.items()}


@dataclasses.dataclass(frozen=True)
class ViscosityProfile:
    """A viscosity structure of layers of constant viscosity, in radius.

    Attributes
    ----------
    model_id
        The id of the earth model file it was read from, for example ``vm5a``.
    edges_m
        The radii of the layer boundaries in metres, increasing, from the
        core-mantle boundary to the surface. Layer ``j`` lies between
        ``edges_m[j]`` and ``edges_m[j + 1]``.
    viscosity_pa_s
        The viscosity of each layer in Pa s. An elastic layer has infinite
        viscosity.
    """

    model_id: str
    edges_m: np.ndarray
    viscosity_pa_s: np.ndarray

    def __post_init__(self):
        if len(self.edges_m) != len(self.viscosity_pa_s) + 1:
            raise ValueError(f"{self.model_id}: one more edge than layers is needed")
        if np.any(np.diff(self.edges_m) <= 0):
            raise ValueError(f"{self.model_id}: the layer boundaries must increase")

    @classmethod
    def from_model(cls, model_id: str, elastic: ElasticProfile) -> "ViscosityProfile":
        """Read the viscosity structure of an earth model file.

        Two forms are read. A layered profile (VM5a, VM7) lists each layer
        with its top and bottom depth. A profile given by parameters (Gowan et
        al. 2021, the middle of the GIAMIP range) gives a lithosphere
        thickness, an upper and a lower mantle viscosity, and the depth of the
        boundary between upper and lower mantle. In both forms the lithosphere
        is elastic and the deepest layer reaches the core-mantle boundary of
        ``elastic``.

        Parameters
        ----------
        model_id
            An earth model id, for example ``vm5a`` or ``gowan2021``.
        elastic
            The elastic structure. It gives the surface radius, from which
            depths are measured, and the core-mantle boundary.

        Raises
        ------
        NotImplementedError
            For a laterally varying viscosity, whose grid is not in this
            package.
        ValueError
            For a range of values, which is not one structure, and for a model
            that is not a viscosity structure.
        """
        model = cases.earth_model(model_id)
        kind = model["kind"]
        if kind == "viscosity_3d":
            raise NotImplementedError(
                f"{model_id} varies laterally. Its grid is the g-drift dataset "
                f"that the model file names.")
        if kind == "parameter_range":
            raise ValueError(
                f"{model_id} is a range of values. Choose one structure inside "
                f"it, as giamip-open-range-midpoint does.")
        if kind != "viscosity_profile":
            raise ValueError(f"{model_id} is not a viscosity structure")

        surface = elastic.surface_radius_m
        cmb_depth_km = (surface - elastic.core_mantle_boundary_m) / 1e3
        if model.get("layers"):
            # Depths in km below the surface, from the top layer down. The
            # bottom of the deepest layer is the core-mantle boundary of the
            # elastic structure, whatever depth the file writes for it.
            layers = model["layers"]
            for upper, lower in zip(layers, layers[1:]):
                if upper["bottom_depth_km"] != lower["top_depth_km"]:
                    raise ValueError(f"{model_id}: a gap or an overlap at "
                                     f"{upper['bottom_depth_km']} km")
            tops_km = [L["top_depth_km"] for L in layers]
            visc = [np.inf if L["viscosity_pa_s"] is None else float(L["viscosity_pa_s"])
                    for L in layers]
        else:
            # Three parameters and a boundary depth: an elastic lithosphere,
            # an upper mantle and a lower mantle. The protocol states no depth
            # for the upper/lower mantle boundary, so the model file must give
            # it, and its absence is an error, not a default.
            if "upper_lower_mantle_boundary_depth_km" not in model:
                raise KeyError(f"{model_id} gives no upper_lower_mantle_boundary_depth_km")
            tops_km = [0.0, float(model["lithosphere_thickness_km"]),
                       float(model["upper_lower_mantle_boundary_depth_km"])]
            visc = [np.inf, float(model["upper_mantle_viscosity_pa_s"]),
                    float(model["lower_mantle_viscosity_pa_s"])]
        if tops_km[0] != 0:
            raise ValueError(f"{model_id}: the top layer must start at the surface")
        if tops_km[-1] >= cmb_depth_km:
            raise ValueError(f"{model_id}: a layer starts below the core-mantle boundary")
        # Convert the depths of the layer tops to radii, and order the layers
        # from the core-mantle boundary upward, the order the search needs.
        depths_km = np.array(tops_km + [cmb_depth_km])
        edges = (surface - depths_km * 1e3)[::-1]
        return cls(model_id=model_id, edges_m=edges,
                   viscosity_pa_s=np.array(visc, dtype=float)[::-1])

    def __call__(self, radius, side: str = "above") -> np.ndarray:
        """The viscosity at the given radii, in Pa s.

        Parameters
        ----------
        radius
            A number or an array of radii in metres, between 0 and the surface.
        side
            ``"above"`` or ``"below"``: which layer a radius exactly on a
            layer boundary belongs to.

        Returns
        -------
        numpy.ndarray
            The viscosity with the shape of ``radius``. It is infinite in an
            elastic layer and NaN in the core, which the viscosity structure
            does not describe.
        """
        _check_side(side)
        edges = self.edges_m
        r = _radii(radius, edges[-1])
        # The index of the layer: with side "above" a radius on an edge goes
        # to the layer whose bottom is that edge, with side "below" to the
        # layer whose top it is.
        if side == "above":
            j = np.searchsorted(edges, r, side="right") - 1
        else:
            j = np.searchsorted(edges, r, side="left") - 1
        # The surface with side "above" gives one past the top layer.
        j = np.minimum(j, len(self.viscosity_pa_s) - 1)
        out = np.full(r.shape, np.nan)
        mantle = j >= 0
        out[mantle] = self.viscosity_pa_s[j[mantle]]
        return out


@dataclasses.dataclass(frozen=True)
class RadialEarth:
    """An elastic structure and a viscosity structure on the same radii.

    Attributes
    ----------
    elastic
        The elastic structure.
    viscosity
        The viscosity structure.
    rheology
        The rheology of the case, for example ``maxwell``, or ``open`` where
        the protocol leaves it to the group and the G-ADOPT choice is not
        used. It is recorded for the caller and does not change any value.
    compressibility
        ``compressible`` (every G-ADOPT choice), ``incompressible``, or
        ``open`` where the G-ADOPT choice is not used. It is recorded for the
        caller and does not change the bulk modulus.
    """

    elastic: ElasticProfile
    viscosity: ViscosityProfile
    rheology: str
    compressibility: str

    def __call__(self, radius, side: str = "above") -> dict[str, np.ndarray]:
        """Every property at the given radii.

        Parameters
        ----------
        radius
            A number or an array of radii in metres, between 0 and the surface.
        side
            ``"above"`` or ``"below"``: which value to give at a radius that
            lies exactly on a boundary of either structure.

        Returns
        -------
        dict
            The arrays of `ElasticProfile.__call__` and ``viscosity_pa_s``.
        """
        values = self.elastic(radius, side)
        values["viscosity_pa_s"] = self.viscosity(radius, side)
        return values


def for_case(case_id: str, gadopt: bool = True) -> RadialEarth:
    """The radial earth of a GIAMIP case.

    Parameters
    ----------
    case_id
        A GIAMIP case id, for example ``giamip-exp01``.
    gadopt
        True fills the fields the protocol leaves open with the G-ADOPT
        choices (`giamip.cases.with_gadopt_choices`). False uses the protocol
        values alone, which works only where the protocol prescribes both
        structures.

    Returns
    -------
    RadialEarth
        Call it with an array of radii in metres.

    Raises
    ------
    ValueError
        If a structure the case needs is left open, or the case is not a
        GIAMIP case.
    NotImplementedError
        If a structure varies laterally.
    """
    # The collection is checked first: a benchmark case has no open fields
    # and no G-ADOPT choices to fill in.
    if cases.case(case_id)["collection"] != "giamip":
        raise ValueError(f"{case_id} is not a GIAMIP case")
    c = cases.with_gadopt_choices(case_id) if gadopt else cases.case(case_id)
    e = c["earth"]
    for field in ("elastic_structure", "viscosity"):
        if e[field] == "open":
            raise ValueError(f"{case_id}: the protocol leaves {field} open. "
                             f"Pass gadopt=True for the G-ADOPT choice.")
    elastic = ElasticProfile.from_model(e["elastic_structure"])
    viscosity = ViscosityProfile.from_model(e["viscosity"], elastic)
    rheology = e["rheology"]
    # Exp07 takes its rheology from Exp02 (protocol table: "Same as Exp02").
    # The reference is resolved here, with the same use of the G-ADOPT
    # choices, so the caller always gets a rheology name or "open".
    reference = rheology.removeprefix("same as ")
    if reference != rheology:
        other = f"giamip-{reference.lower()}"
        rheology = (cases.with_gadopt_choices(other) if gadopt
                    else cases.case(other))["earth"]["rheology"]
    return RadialEarth(elastic=elastic, viscosity=viscosity,
                       rheology=rheology, compressibility=e["compressibility"])
