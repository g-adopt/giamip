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
import pathlib

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


def _has_group(ds, group_path: str) -> bool:
    """Whether a slash-separated group path exists in an open dataset."""
    group = ds
    for part in group_path.split("/"):
        if part not in group.groups:
            return False
        group = group.groups[part]
    return True
