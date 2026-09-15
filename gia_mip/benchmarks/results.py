"""Profiles and time series of one run of one Martinec case, from any code.

The published contributors, VEGA, radopt and G-ADOPT all produce the same
things for a case: U, N and S along two meridians at the terminal time, and
h_UF, ice mass and ocean area against time. `BenchmarkResult` holds those
curves for one code, reads them from a NetCDF file and writes them to one, so
that plotting and scoring treat a reference contributor and our own solvers
the same way.

The file layout is the layout of the converted reference data
(``gia_mip.benchmarks.martinec2018``):

``profiles/<load|basin>/<member>``
    U, N, S in metres against ``colatitude`` in degrees.
``differences/<load|basin>/<member>``
    The member minus VEGA, where a contributor shipped one.
``time_series/<member>``
    h_UF in metres, ice mass in kg, ocean area as a fraction of 4 pi,
    against ``time`` in kyr.

The Spada et al. (2011) benchmark is held by `SpadaResult`, further down,
which adds Love numbers and relaxation spectra to the curves.

A radopt run of case C is therefore written as, for example,
``outputs/martinec2018/radopt/martinec2018-C.nc`` with member ``radopt``, and
``BenchmarkResult.read`` opens it exactly as it opens a reference file.

    >>> from gia_mip.benchmarks.results import BenchmarkResult
    >>> vega = BenchmarkResult.reference("C", "ZM")
    >>> curve = vega.profile("basin", "S")
    >>> curve.axis[:2], curve.values[:2]            # doctest: +SKIP
"""

from __future__ import annotations

import dataclasses
import gc
import math
import pathlib
import re

import numpy as np

from gia_mip.benchmarks import martinec2018

PROFILE_QUANTITIES = ("U", "N", "S")

def _time_series_units(quantity: str) -> str:
    """The unit a time-series quantity is stored in, from its name.

    A name starting with h_UF is a thickness, with ice_mass a mass, and with
    ocean_area a fraction of 4 pi. Any other name raises.
    """
    if quantity.startswith("h_UF"):
        return "m"
    if quantity.startswith("ice_mass"):
        return "kg"
    if quantity.startswith("ocean_area"):
        return "1"
    raise ValueError(f"unknown time-series quantity {quantity!r}")


@dataclasses.dataclass
class Curve:
    """One quantity against one axis, both 1-D, the axis strictly ascending.

    The constructor sorts by the axis, because several contributors store
    colatitude descending (correction benchmarkA-colatitude-ordering) and a
    comparison by interpolation needs it ascending.
    """

    axis: np.ndarray
    values: np.ndarray
    attributes: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        axis = np.asarray(self.axis, dtype=float).ravel()
        values = np.asarray(self.values, dtype=float).ravel()
        if axis.shape != values.shape:
            raise ValueError(f"axis has {axis.size} points and values have {values.size}")
        order = np.argsort(axis, kind="stable")
        self.axis, self.values = axis[order], values[order]
        if np.any(np.diff(self.axis) <= 0):
            raise ValueError("the axis repeats a value")


class BenchmarkResult:
    """The curves of one code for one case.

    Parameters
    ----------
    case
        The published case letter.
    member
        The code: a contributor's two-letter code, ``ZM`` for VEGA, or a name
        such as ``radopt`` or ``g-adopt``.
    """

    def __init__(self, case: str, member: str):
        self.case = case.rsplit("-", 1)[-1].upper()
        self.member = member
        self.profiles: dict[tuple[str, str], Curve] = {}
        self.differences: dict[tuple[str, str], Curve] = {}
        self.time_series: dict[str, Curve] = {}

    def __repr__(self) -> str:
        return (f"BenchmarkResult(case={self.case!r}, member={self.member!r}, "
                f"profiles={sorted(self.profiles)}, time_series={sorted(self.time_series)})")

    # ----------------------------------------------------------- building

    def add_profile(self, profile: str, quantity: str, colatitude_deg, values,
                    **attributes) -> None:
        """Add U, N or S in metres along the ``load`` or ``basin`` meridian."""
        if profile not in ("load", "basin"):
            raise ValueError(f"profile must be 'load' or 'basin', not {profile!r}")
        if quantity not in PROFILE_QUANTITIES:
            raise ValueError(f"quantity must be one of {PROFILE_QUANTITIES}, not {quantity!r}")
        self.profiles[(profile, quantity)] = Curve(colatitude_deg, values, dict(attributes))

    def add_time_series(self, quantity: str, time_kyr, values, **attributes) -> None:
        """Add h_UF (m), an ice mass (kg) or the ocean area (fraction of 4 pi)."""
        _time_series_units(quantity)
        self.time_series[quantity] = Curve(time_kyr, values, dict(attributes))

    # ------------------------------------------------------------- access

    def profile(self, profile: str, quantity: str) -> Curve:
        """One profile curve. Raises KeyError naming what is available."""
        try:
            return self.profiles[(profile, quantity)]
        except KeyError:
            raise KeyError(f"{self.member} case {self.case} has no {quantity} on the "
                           f"{profile} profile; it has {sorted(self.profiles)}") from None

    def series(self, quantity: str) -> Curve:
        """One time series. Raises KeyError naming what is available."""
        try:
            return self.time_series[quantity]
        except KeyError:
            raise KeyError(f"{self.member} case {self.case} has no {quantity} time series; "
                           f"it has {sorted(self.time_series)}") from None

    # --------------------------------------------------------------- files

    @staticmethod
    def reference_path(case: str) -> pathlib.Path:
        """The converted reference file of a case."""
        letter = case.rsplit("-", 1)[-1].upper()
        return martinec2018.data_root() / "converted" / "martinec2018" / f"martinec2018-{letter}.nc"

    @classmethod
    def reference(cls, case: str, member: str = "ZM") -> "BenchmarkResult":
        """A published contributor, or VEGA (``ZM``), from the converted data."""
        return cls.read(cls.reference_path(case), member, case=case)

    @staticmethod
    def members(path: pathlib.Path) -> list[str]:
        """Every member with at least one curve in a file."""
        import netCDF4
        found = set()
        with netCDF4.Dataset(path) as ds:
            for kind in ("profiles", "differences"):
                if kind in ds.groups:
                    for profile in ds[kind].groups.values():
                        found.update(profile.groups)
            if "time_series" in ds.groups:
                found.update(ds["time_series"].groups)
        return sorted(found)

    @classmethod
    def read(cls, path: pathlib.Path, member: str, case: str | None = None) -> "BenchmarkResult":
        """Read one member's curves from a file in the converted layout.

        Masking is turned off, as AGENTS.md requires of every reader.
        """
        import netCDF4
        with netCDF4.Dataset(path) as ds:
            ds.set_auto_mask(False)
            result = cls(case or getattr(ds, "case", "?"), member)

            def curves(group):
                for name, var in group.variables.items():
                    if name in group.dimensions:
                        continue
                    axis = group[var.dimensions[0]][:]
                    attrs = {k: var.getncattr(k) for k in var.ncattrs()}
                    yield name, Curve(axis, var[:], attrs)

            for kind, store in (("profiles", result.profiles), ("differences", result.differences)):
                if kind not in ds.groups:
                    continue
                for profile, pgroup in ds[kind].groups.items():
                    if member in pgroup.groups:
                        for name, curve in curves(pgroup[member]):
                            store[(profile, name)] = curve
            if "time_series" in ds.groups and member in ds["time_series"].groups:
                for name, curve in curves(ds["time_series"][member]):
                    result.time_series[name] = curve
        if not (result.profiles or result.differences or result.time_series):
            raise KeyError(f"{path} has no curves for member {member!r}")
        return result

    def write(self, path: pathlib.Path) -> pathlib.Path:
        """Write this member's curves into a file in the converted layout.

        The file is created if it does not exist. A member that already has
        a group in the file raises, so a second run cannot silently mix with
        the first.
        """
        import netCDF4
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with netCDF4.Dataset(path, "a" if path.exists() else "w", format="NETCDF4") as ds:
            if not hasattr(ds, "case"):
                ds.case = self.case
                ds.title = f"Martinec et al. (2018) benchmark case {self.case}"
                ds.conventions_reference = "docs/reference/conventions.qmd"
            elif ds.case != self.case:
                raise ValueError(f"{path} holds case {ds.case}, not {self.case}")
            groups: dict[str, list[tuple[str, Curve, str, str, str]]] = {}
            for (profile, name), curve in self.profiles.items():
                groups.setdefault(f"profiles/{profile}/{self.member}", []).append(
                    (name, curve, "m", "colatitude", "degrees"))
            for (profile, name), curve in self.differences.items():
                groups.setdefault(f"differences/{profile}/{self.member}", []).append(
                    (name, curve, "m", "colatitude", "degrees"))
            for name, curve in self.time_series.items():
                groups.setdefault(f"time_series/{self.member}", []).append(
                    (name, curve, _time_series_units(name), "time", "kyr"))
            for group_path, variables in sorted(groups.items()):
                if _has_group(ds, group_path):
                    raise ValueError(f"{path} already has {group_path}")
                g = ds.createGroup(group_path)
                axes: list[tuple[str, np.ndarray]] = []
                for name, curve, units, axis_name, axis_units in sorted(variables, key=lambda v: v[0]):
                    dim = next((d for d, a in axes if a.shape == curve.axis.shape
                                and np.array_equal(a, curve.axis)), None)
                    if dim is None:
                        dim = axis_name if not axes else f"{axis_name}_{len(axes) + 1}"
                        axes.append((dim, curve.axis))
                        g.createDimension(dim, curve.axis.size)
                        coord = g.createVariable(dim, "f8", (dim,), fill_value=False)
                        coord[:] = curve.axis
                        coord.units = axis_units
                    v = g.createVariable(name, "f8", (dim,), zlib=True, fill_value=False)
                    v[:] = curve.values
                    v.units = units
                    if name in martinec2018.LONG_NAMES:
                        v.long_name = martinec2018.LONG_NAMES[name]
                    for key, value in curve.attributes.items():
                        if key in ("units", "long_name"):
                            continue
                        v.setncattr(key, value)
        return path


def _read_dataset(path: pathlib.Path, function):
    """Open a NetCDF file with masking off, apply ``function`` to it, and free it.

    ``function`` does the reading in a frame of its own, so that when it
    returns no local name still refers to a group of the file. netCDF4 keeps
    each dataset in a reference cycle with itself and with every group it has
    opened, so a closed dataset is freed only by the cycle collector. If
    ``netCDF4.Dataset`` has been replaced by a subclass defined inside a
    function, as the masking test of tests/test_spada_classes.py does, a later
    collection can free that class and the dataset together, and netCDF4's
    deallocation then fails and can crash the interpreter. Collecting here,
    after the dataset is closed and while its class is still referenced by the
    caller, frees it safely. One full collection costs a few milliseconds.
    """
    import netCDF4
    ds = netCDF4.Dataset(path)
    try:
        ds.set_auto_mask(False)
        return function(ds)
    finally:
        ds.close()
        del ds
        gc.collect()


def _has_group(ds, group_path: str) -> bool:
    """Whether a slash-separated group path exists in an open dataset."""
    group = ds
    for part in group_path.split("/"):
        if part not in group.groups:
            return False
        group = group.groups[part]
    return True


# =====================================================================
# Spada et al. (2011)
# =====================================================================
#
# The Spada benchmark compares three kinds of result, and no one of them is a
# curve against colatitude alone: Love numbers per harmonic degree with one
# residue per relaxation mode (tests 2/1 and 3/1), the relaxation spectrum per
# degree and mode (test 1/1), and U, V, N or their rates against colatitude at
# several epochs for three loads (tests 1/2 and 2/2). `SpadaResult` holds all
# three for one code, so a published contributor, radopt and G-ADOPT are read,
# written and scored the same way.
#
# The file layout, shared with the converted reference data
# (``gia_mip.benchmarks.spada2011``):
#
# ``love_numbers/<h|l|k>/<member>``
#     ``elastic`` and ``fluid`` against ``degree``, and ``residues`` against
#     ``degree`` and ``mode``. Dimensionless, residues in 1/kyr.
# ``relaxation_spectrum/<member>``
#     ``s`` against ``degree`` and ``mode``, in 1/kyr.
# ``spatial/<load>/<member>/epoch_<epoch>``
#     U, V, N in m or U_rate, V_rate, N_rate in m/kyr against ``colatitude``
#     in degrees, for the epoch in kyr. The group attribute ``epoch_kyr`` is
#     the epoch as text, ``inf`` for the fluid limit, so that no variable
#     holds an infinite value.

SPADA_LOADS = ("cap", "disc", "point")
SPADA_LOVE_QUANTITIES = ("h", "l", "k")

# The unit a spatial quantity is stored in. A rate is in m/kyr, the unit that
# matches epochs in kyr and s_j in 1/kyr, and numerically equal to mm/yr.
SPADA_SPATIAL_UNITS = {"U": "m", "V": "m", "N": "m",
                       "U_rate": "m/kyr", "V_rate": "m/kyr", "N_rate": "m/kyr"}

SPADA_LONG_NAMES = {
    "U": "vertical displacement of the surface, positive upward",
    "V": "horizontal displacement, colatitude component, positive away from the pole",
    "N": "geoid displacement, positive upward",
    "U_rate": "rate of vertical displacement, positive upward",
    "V_rate": "rate of horizontal displacement, colatitude component",
    "N_rate": "rate of geoid displacement, positive upward",
}


def _short_spada_case(case: str) -> str:
    """``t02-01`` from ``t02-01``, ``T02-01`` or ``spada2011-t02-01``."""
    short = case.lower().removeprefix("spada2011-")
    if not re.fullmatch(r"t0[12]-0\d", short):
        raise ValueError(f"not a Spada et al. (2011) case: {case!r}")
    return short


def _epoch_label(epoch_kyr: float) -> str:
    """The text of an epoch in a group name and attribute: ``1000``, ``0.5``, ``inf``."""
    epoch = float(epoch_kyr)
    return "inf" if math.isinf(epoch) else f"{epoch:g}"


@dataclasses.dataclass
class LoveNumbers:
    """Love numbers of one kind (h, l or k) of one code, one row per degree.

    Attributes
    ----------
    degrees
        Harmonic degrees, strictly ascending.
    elastic, fluid
        The t = 0 and t -> infinity values, dimensionless.
    residues
        Shape (degrees, modes), in 1/kyr, in the convention of Spada et al.
        (2011), eq. 17: fluid = elastic - sum_j residues_j / s_j. Mode j is the
        j-th slowest mode, as in `RelaxationSpectrum`.
    source_file
        The path under data/original/ of the file the values come from, or
        None for a solver's own result.
    attributes
        Anything else recorded with the values.
    """

    degrees: np.ndarray
    elastic: np.ndarray
    fluid: np.ndarray
    residues: np.ndarray
    source_file: str | None = None
    attributes: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class RelaxationSpectrum:
    """The relaxation frequencies s_j of one code, in 1/kyr, one row per degree.

    ``s_per_kyr`` has shape (degrees, modes). s_j is negative, and mode 1 is the
    slowest, the value closest to zero. The relaxation time is -1/s_j in kyr.
    """

    degrees: np.ndarray
    s_per_kyr: np.ndarray
    source_file: str | None = None
    attributes: dict = dataclasses.field(default_factory=dict)


def _ascending_degrees(degrees, *arrays) -> tuple:
    """Sort rows by degree, and refuse a repeated degree.

    A repeated degree has no single value, and a reader that picked one
    silently would hide it. The converter resolves the one known case (RR's
    repeated rows, correction spada-t01-rr-repeated-degrees) before it gets
    here.
    """
    degrees = np.asarray(degrees).astype(int).ravel()
    order = np.argsort(degrees, kind="stable")
    degrees = degrees[order]
    if np.any(np.diff(degrees) <= 0):
        raise ValueError("a harmonic degree repeats")
    out = [degrees]
    for array in arrays:
        array = np.asarray(array, dtype=float)
        if array.shape[0] != degrees.size:
            raise ValueError(f"{degrees.size} degrees and {array.shape[0]} rows")
        out.append(array[order])
    return tuple(out)


class SpadaResult:
    """Love numbers, relaxation spectrum and surface fields of one code for one case.

    Parameters
    ----------
    case
        ``t02-01`` or ``spada2011-t02-01``.
    member
        The code: a contributor's two-letter code, or a name such as
        ``radopt`` or ``g-adopt``.
    """

    def __init__(self, case: str, member: str):
        self.case = _short_spada_case(case)
        self.member = member
        self.love: dict[str, LoveNumbers] = {}
        self.spectrum: RelaxationSpectrum | None = None
        self.fields: dict[tuple[str, str, float], Curve] = {}

    def __repr__(self) -> str:
        return (f"SpadaResult(case={self.case!r}, member={self.member!r}, "
                f"love_numbers={sorted(self.love)}, spectrum={self.spectrum is not None}, "
                f"spatial={len(self.fields)} curves)")

    # ----------------------------------------------------------- building

    def add_love_numbers(self, quantity: str, degrees, elastic, fluid, residues,
                         source_file: str | None = None, **attributes) -> None:
        """Add h, l or k: elastic and fluid per degree, residues per degree and mode."""
        if quantity not in SPADA_LOVE_QUANTITIES:
            raise ValueError(f"quantity must be one of {SPADA_LOVE_QUANTITIES}, not {quantity!r}")
        residues = np.asarray(residues, dtype=float)
        if residues.ndim != 2:
            raise ValueError("residues must have shape (degrees, modes)")
        degrees, elastic, fluid, residues = _ascending_degrees(degrees, np.ravel(elastic),
                                                               np.ravel(fluid), residues)
        self.love[quantity] = LoveNumbers(degrees, elastic, fluid, residues,
                                          source_file, dict(attributes))

    def add_relaxation_spectrum(self, degrees, s_per_kyr, source_file: str | None = None,
                                **attributes) -> None:
        """Add s_j in 1/kyr, shape (degrees, modes), mode 1 the slowest."""
        s = np.asarray(s_per_kyr, dtype=float)
        if s.ndim != 2:
            raise ValueError("s_per_kyr must have shape (degrees, modes)")
        degrees, s = _ascending_degrees(degrees, s)
        self.spectrum = RelaxationSpectrum(degrees, s, source_file, dict(attributes))

    def add_spatial(self, load: str, quantity: str, epoch_kyr: float, colatitude_deg,
                    values, **attributes) -> None:
        """Add one field along colatitude from the load centre at one epoch.

        ``epoch_kyr`` may be ``math.inf`` for the fluid limit, which is a
        different epoch from any finite one, 1000 kyr included. The unit is
        fixed by the quantity (`SPADA_SPATIAL_UNITS`); a ``units`` attribute
        that says otherwise raises, because a value in the wrong unit would be
        scored as if it were right.
        """
        if load not in SPADA_LOADS:
            raise ValueError(f"load must be one of {SPADA_LOADS}, not {load!r}")
        if quantity not in SPADA_SPATIAL_UNITS:
            raise ValueError(f"quantity must be one of {tuple(SPADA_SPATIAL_UNITS)}, "
                             f"not {quantity!r}")
        units = SPADA_SPATIAL_UNITS[quantity]
        if attributes.get("units", units) != units:
            raise ValueError(f"{quantity} is stored in {units}, not {attributes['units']}")
        attributes["units"] = units
        self.fields[(load, quantity, float(epoch_kyr))] = Curve(colatitude_deg, values,
                                                                dict(attributes))

    # ------------------------------------------------------------- access

    def love_numbers(self, quantity: str) -> LoveNumbers:
        """h, l or k. Raises KeyError naming what is available."""
        try:
            return self.love[quantity]
        except KeyError:
            raise KeyError(f"{self.member} case {self.case} has no {quantity} Love numbers; "
                           f"it has {sorted(self.love)}") from None

    def relaxation_spectrum(self) -> RelaxationSpectrum:
        """The spectrum. Raises KeyError when the result holds none."""
        if self.spectrum is None:
            raise KeyError(f"{self.member} case {self.case} has no relaxation spectrum")
        return self.spectrum

    def spatial(self, load: str, quantity: str, epoch_kyr: float) -> Curve:
        """One field at one epoch. The epoch must match exactly: 1000 is not infinity."""
        try:
            return self.fields[(load, quantity, float(epoch_kyr))]
        except KeyError:
            raise KeyError(f"{self.member} case {self.case} has no {quantity} for the {load} "
                           f"at {epoch_kyr} kyr; it has epochs "
                           f"{self.epochs(load, quantity)}") from None

    def epochs(self, load: str, quantity: str) -> list[float]:
        """The epochs held for one load and quantity, ascending; empty if none."""
        return sorted(e for (l, q, e) in self.fields if l == load and q == quantity)

    # --------------------------------------------------------------- files

    @staticmethod
    def reference_path(case: str) -> pathlib.Path:
        """The converted reference file of a case, which holds every contributor."""
        from gia_mip.benchmarks import spada2011
        short = _short_spada_case(case)
        return spada2011.data_root() / "converted" / "spada2011" / f"spada2011-{short}.nc"

    @classmethod
    def reference(cls, case: str, member: str) -> "SpadaResult":
        """A published contributor, from the converted data."""
        return cls.read(cls.reference_path(case), member, case=case)

    @staticmethod
    def members(path: pathlib.Path) -> list[str]:
        """Every member with at least one value in a file."""
        def collect(ds) -> list[str]:
            found = set()
            if "love_numbers" in ds.groups:
                for quantity in ds["love_numbers"].groups.values():
                    found.update(quantity.groups)
            if "relaxation_spectrum" in ds.groups:
                found.update(ds["relaxation_spectrum"].groups)
            if "spatial" in ds.groups:
                for load in ds["spatial"].groups.values():
                    found.update(load.groups)
            return sorted(found)
        return _read_dataset(path, collect)

    @classmethod
    def read(cls, path: pathlib.Path, member: str, case: str | None = None) -> "SpadaResult":
        """Read one member from a file in the converted layout.

        Masking is turned off, as AGENTS.md requires of every reader. The
        converted files hold no fill value, so a masked read would return the
        same numbers today; the call keeps it that way for a file that has one.
        """
        return _read_dataset(path, lambda ds: cls._read_open(ds, path, member, case))

    @classmethod
    def _read_open(cls, ds, path: pathlib.Path, member: str, case: str | None) -> "SpadaResult":
        """The body of `read`, on an open dataset whose masking is off."""
        result = cls(case or ds.getncattr("case"), member)

        def attrs(variable_or_group, skip=()):
            return {k: variable_or_group.getncattr(k) for k in variable_or_group.ncattrs()
                    if k not in skip}

        if "love_numbers" in ds.groups:
            for quantity, qgroup in ds["love_numbers"].groups.items():
                if member not in qgroup.groups:
                    continue
                g = qgroup[member]
                extra = attrs(g, skip=("source_file",))
                result.add_love_numbers(quantity, g["degree"][:], g["elastic"][:],
                                        g["fluid"][:], g["residues"][:],
                                        source_file=g.getncattr("source_file")
                                        if "source_file" in g.ncattrs() else None,
                                        **extra)
        if "relaxation_spectrum" in ds.groups and member in ds["relaxation_spectrum"].groups:
            g = ds["relaxation_spectrum"][member]
            result.add_relaxation_spectrum(
                g["degree"][:], g["s"][:],
                source_file=g.getncattr("source_file") if "source_file" in g.ncattrs() else None,
                **attrs(g, skip=("source_file",)))
        if "spatial" in ds.groups:
            for load, lgroup in ds["spatial"].groups.items():
                if member not in lgroup.groups:
                    continue
                for egroup in lgroup[member].groups.values():
                    epoch = float(egroup.getncattr("epoch_kyr"))
                    for name, var in egroup.variables.items():
                        if name in egroup.dimensions:
                            continue
                        result.add_spatial(load, name, epoch, egroup[var.dimensions[0]][:],
                                           var[:], **attrs(var, skip=("long_name",)))
        if not (result.love or result.spectrum is not None or result.fields):
            raise KeyError(f"{path} has nothing for member {member!r}")
        return result

    def write(self, path: pathlib.Path) -> pathlib.Path:
        """Write this member into a file in the converted layout.

        The file is created if it does not exist, so several members share one
        file. A member that already has a group in the file raises, so a second
        run cannot mix with the first.
        """
        import netCDF4
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with netCDF4.Dataset(path, "a" if path.exists() else "w", format="NETCDF4") as ds:
            if "case" not in ds.ncattrs():
                ds.case = self.case
                ds.title = f"Spada et al. (2011) benchmark case {self.case}"
                ds.conventions_reference = "docs/reference/conventions.qmd"
            elif ds.getncattr("case") != self.case:
                raise ValueError(f"{path} holds case {ds.getncattr('case')}, not {self.case}")

            def new_group(group_path):
                if _has_group(ds, group_path):
                    raise ValueError(f"{path} already has {group_path}")
                return ds.createGroup(group_path)

            def set_attrs(target, attributes):
                for key, value in attributes.items():
                    if value is None:
                        continue
                    target.setncattr(key, value)

            def degree_and_mode(g, degrees, modes):
                g.createDimension("degree", degrees.size)
                g.createDimension("mode", modes)
                d = g.createVariable("degree", "i4", ("degree",), fill_value=False)
                d[:] = degrees
                d.long_name = "spherical harmonic degree"
                m = g.createVariable("mode", "i4", ("mode",), fill_value=False)
                m[:] = np.arange(1, modes + 1)
                m.long_name = "relaxation mode, 1 the slowest"

            for quantity, love in sorted(self.love.items()):
                g = new_group(f"love_numbers/{quantity}/{self.member}")
                degree_and_mode(g, love.degrees, love.residues.shape[1])
                for name, values, dims, units in (
                        ("elastic", love.elastic, ("degree",), "1"),
                        ("fluid", love.fluid, ("degree",), "1"),
                        ("residues", love.residues, ("degree", "mode"), "1/kyr")):
                    v = g.createVariable(name, "f8", dims, zlib=True, fill_value=False)
                    v[:] = values
                    v.units = units
                set_attrs(g, {"source_file": love.source_file, **love.attributes})

            if self.spectrum is not None:
                g = new_group(f"relaxation_spectrum/{self.member}")
                degree_and_mode(g, self.spectrum.degrees, self.spectrum.s_per_kyr.shape[1])
                v = g.createVariable("s", "f8", ("degree", "mode"), zlib=True, fill_value=False)
                v[:] = self.spectrum.s_per_kyr
                v.units = "1/kyr"
                v.long_name = "relaxation frequency, negative; the relaxation time is -1/s"
                set_attrs(g, {"source_file": self.spectrum.source_file,
                              **self.spectrum.attributes})

            groups: dict[tuple[str, float], list[tuple[str, Curve]]] = {}
            for (load, quantity, epoch), curve in self.fields.items():
                groups.setdefault((load, epoch), []).append((quantity, curve))
            for (load, epoch), variables in sorted(groups.items()):
                g = new_group(f"spatial/{load}/{self.member}/epoch_{_epoch_label(epoch)}")
                g.epoch_kyr = _epoch_label(epoch)
                axes: list[tuple[str, np.ndarray]] = []
                for name, curve in sorted(variables, key=lambda v: v[0]):
                    dim = next((d for d, a in axes if a.shape == curve.axis.shape
                                and np.array_equal(a, curve.axis)), None)
                    if dim is None:
                        dim = "colatitude" if not axes else f"colatitude_{len(axes) + 1}"
                        axes.append((dim, curve.axis))
                        g.createDimension(dim, curve.axis.size)
                        coord = g.createVariable(dim, "f8", (dim,), fill_value=False)
                        coord[:] = curve.axis
                        coord.units = "degrees"
                        coord.long_name = "angular distance from the load centre"
                    v = g.createVariable(name, "f8", (dim,), zlib=True, fill_value=False)
                    v[:] = curve.values
                    v.long_name = SPADA_LONG_NAMES[name]
                    set_attrs(v, curve.attributes)
        return path
