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
which adds Love numbers, relaxation spectra, Love numbers as functions of
time, polar motion, the length of day and the polar motion transfer function
to the curves.

A radopt run of case C is therefore written as, for example,
``outputs/martinec2018/radopt/martinec2018-C.nc`` with member ``radopt``, and
``BenchmarkResult.read`` opens it exactly as it opens a reference file.

    >>> from gia_mip.benchmarks.results import BenchmarkResult
    >>> vega = BenchmarkResult.reference("C", "ZM")
    >>> curve = vega.profile("basin", "S")
    >>> curve.axis[:2], curve.values[:2]            # doctest: +SKIP
"""

from __future__ import annotations

import contextlib
import dataclasses
import gc
import math
import pathlib
import re
import warnings

import numpy as np

from gia_mip.benchmarks import martinec2018

PROFILE_QUANTITIES = ("U", "N", "S")


@contextlib.contextmanager
def _without_netcdf4_shape_warning():
    """Suppress the one NumPy deprecation that netCDF4 1.7.4 raises on a 2D write.

    NumPy 2.5 deprecates assigning to ``ndarray.shape``. netCDF4 1.7.4 makes
    that assignment inside ``Variable.__setitem__`` whenever it writes a 2D
    array, whatever form the index takes, and the warning names our line. The
    values written are correct. netCDF4 fixed it upstream by calling
    ``reshape`` (issue #1468, commit 2c518fc1, 2026-02-16), but on 2026-09-15
    the latest release was still 1.7.4. Remove this function and its two uses
    in `SpadaResult.write` when pyproject.toml requires ``netCDF4>=1.7.5``.
    Only this message is suppressed, so every other warning still shows.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Setting the shape on a NumPy array has been deprecated",
            category=DeprecationWarning)
        yield


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

        Masking is turned off, as AGENTS.md requires of every reader. Raises
        ValueError when ``case`` is given and the file records another case.
        """
        import netCDF4
        with netCDF4.Dataset(path) as ds:
            ds.set_auto_mask(False)
            stored = getattr(ds, "case", None)
            result = cls(case or stored or "?", member)
            # A file holds one case, and `write` refuses to mix cases in it. A
            # `case` argument that disagrees with the file would score these
            # curves over the other case's intervals and report no error.
            if case is not None and stored is not None and result.case != cls(stored, member).case:
                raise ValueError(f"{path} holds case {stored}, not {result.case}")

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
        # Keep this collection. netCDF4 holds every dataset in a reference
        # cycle, so only the cycle collector frees a closed file. Collecting
        # now, while the Dataset class is still referenced, frees it safely. A
        # later collection that frees a locally defined Dataset subclass and
        # its dataset together can crash the interpreter (see the docstring).
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
# The Spada benchmark compares seven kinds of result, and no one of them is a
# curve against colatitude alone: Love numbers per harmonic degree with one
# residue per relaxation mode (tests 2/1 and 3/1), the relaxation spectrum per
# degree and mode (test 1/1), Love numbers against time per degree (tests 5/1,
# 6/1 and 7/1), the coefficients of the polar motion transfer function (test
# 4/1), U, V, N or their rates against colatitude at several epochs for three
# loads (tests 1/2 and 2/2), and polar motion and the length of day against
# time (test 3/2). `SpadaResult` holds all of them for one code, so a
# published contributor, radopt and G-ADOPT are read, written and scored the
# same way.
#
# The file layout, shared with the converted reference data
# (``gia_mip.benchmarks.spada2011``):
#
# ``love_numbers/<h|l|k>/<member>``
#     ``elastic`` and ``fluid`` against ``degree``, and ``residues`` against
#     ``degree`` and ``mode``. Dimensionless, residues in 1/kyr.
# ``relaxation_spectrum/<member>``
#     ``s`` against ``degree`` and ``mode``, in 1/kyr.
# ``time_domain_love_numbers/<h|l|k>/<member>``
#     One variable ``degree_<n>`` per degree, the Heaviside Love number,
#     dimensionless, with the attribute ``degree``, against ``time`` in kyr
#     since the load was applied. Degrees sampled at the same times share one
#     axis, and further axes are ``time_2`` and so on. One group per member
#     keeps the file small: a group per degree costs more storage than the
#     values themselves.
# ``polar_motion_transfer_function/<member>/chandler_wobble_<included|excluded>``
#     The scalars ``elastic_real`` and ``elastic_imag`` (A_e, dimensionless)
#     and ``secular_real`` and ``secular_imag`` (A_s, 1/kyr), and
#     ``frequency_real``, ``frequency_imag``, ``residue_real`` and
#     ``residue_imag`` (a_j and A_j, 1/kyr) against ``mode``, mode 1 the
#     slowest. NetCDF has no complex type, so each complex number is two
#     variables.
# ``spatial/<load>/<member>/epoch_<epoch>``
#     U, V, N in m or U_rate, V_rate, N_rate in m/kyr against ``colatitude``
#     in degrees, for the epoch in kyr. The group attribute ``epoch_kyr`` is
#     the epoch as text, ``inf`` for the fluid limit, so that no variable
#     holds an infinite value.
# ``polar_motion/<load>/<member>/chandler_wobble_<included|excluded>``
#     ``m_x``, ``m_y`` and ``m_modulus`` in degrees and ``m_rate_x``,
#     ``m_rate_y`` and ``m_rate_modulus`` in degrees per Myr, against ``time``
#     in kyr, with further axes ``time_2`` and so on where the variables are
#     sampled differently.
# ``length_of_day/<load>/<member>``
#     ``delta_lod`` in ms against ``time`` in kyr.

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

# The Heaviside Love numbers of eq. 18 of the paper: the response to a load
# applied at t = 0 and held, so h(0) is the elastic value and h(t) tends to the
# fluid value.
SPADA_TIME_DOMAIN_LONG_NAMES = {
    "h": "Heaviside loading Love number h(t)",
    "l": "Heaviside loading Love number l(t)",
    "k": "Heaviside loading Love number k(t)",
}

# Test 4/1 and test 3/2 are solved twice: with the Chandler wobble term of the
# Liouville equations (eq. 8) and without it (eq. 7). The paper's words for
# the two, in table 14, are "included" and "excluded".
SPADA_CHANDLER_WOBBLE = ("included", "excluded")

# Polar motion is m = m_x + i m_y = (omega_x + i omega_y) / Omega (eq. 3), an
# angle in degrees. The axes are those of longitude: x towards longitude 0 and
# y towards 90 degrees east. The evidence is eq. 31, where the geometrical
# factor carries exp(i lambda_c) for the load centroid at east longitude
# lambda_c = 75 degrees, and the phase of GS's and VB's m is -105 degrees, which
# is 75 - 180. The rate is in degrees per Myr, the unit of table 14 and of the
# files; per kyr would be 1000 times smaller.
SPADA_POLAR_MOTION_UNITS = {
    "m_x": "degrees", "m_y": "degrees", "m_modulus": "degrees",
    "m_rate_x": "degrees/Myr", "m_rate_y": "degrees/Myr", "m_rate_modulus": "degrees/Myr",
}

SPADA_POLAR_MOTION_LONG_NAMES = {
    "m_x": "polar motion, component towards longitude 0",
    "m_y": "polar motion, component towards longitude 90 E",
    "m_modulus": "modulus of polar motion",
    "m_rate_x": "rate of polar motion, component towards longitude 0",
    "m_rate_y": "rate of polar motion, component towards longitude 90 E",
    "m_rate_modulus": "modulus of the rate of polar motion",
}

# Eq. 32 and 33: Delta LOD / LOD = Delta I_zz / C with LOD = 86400 s. The paper
# notes that a positive value is a longer day.
SPADA_LENGTH_OF_DAY_LONG_NAME = "change in length of day, positive for a longer day"


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


@dataclasses.dataclass
class PolarMotionTransferFunction:
    """The coefficients of the polar motion transfer function of one code (eq. 12).

    M(s) = A_e + A_s / s + sum_j A_j / (s - a_j), for one setting of the
    Chandler wobble. With the wobble excluded A_e is not zero and the code has
    one rotational mode less than with it included (eqs. 13 and 14).

    Attributes
    ----------
    elastic
        A_e, dimensionless and complex.
    secular
        A_s in 1/kyr, complex.
    frequencies
        a_j in 1/kyr, a complex array. The real part is negative, and mode 1 is
        the slowest, the one whose real part is closest to zero. The mode that
        carries the Chandler wobble has an imaginary part of about 5e3 per kyr,
        a period of about 1.2 years.
    residues
        A_j in 1/kyr, a complex array in the order of ``frequencies``.
    source_file, attributes
        As in `LoveNumbers`.
    """

    elastic: complex
    secular: complex
    frequencies: np.ndarray
    residues: np.ndarray
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
    """Everything one code computed for one Spada et al. (2011) test.

    A result holds whichever of the seven kinds its test asks for: Love numbers
    with residues, a relaxation spectrum, Love numbers against time, a polar
    motion transfer function, surface fields, polar motion and the length of
    day. The block comment above this class gives the file layout of each.

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
        # (quantity, degree) -> h, l or k against time in kyr.
        self.time_domain: dict[tuple[str, int], Curve] = {}
        # (load, Chandler wobble setting, quantity) -> polar motion against time.
        self.polar: dict[tuple[str, str, str], Curve] = {}
        # load -> change in length of day in ms against time.
        self.lod: dict[str, Curve] = {}
        # Chandler wobble setting -> the transfer function.
        self.pmtf: dict[str, PolarMotionTransferFunction] = {}

    def __repr__(self) -> str:
        return (f"SpadaResult(case={self.case!r}, member={self.member!r}, "
                f"love_numbers={sorted(self.love)}, spectrum={self.spectrum is not None}, "
                f"spatial={len(self.fields)} curves, time_domain={len(self.time_domain)} curves, "
                f"polar_motion={len(self.polar)} curves, length_of_day={sorted(self.lod)}, "
                f"transfer_function={sorted(self.pmtf)})")

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

    def add_time_domain_love_numbers(self, quantity: str, degree: int, time_kyr, values,
                                     **attributes) -> None:
        """Add one Heaviside Love number h, l or k of one degree against time in kyr.

        Time starts at the loading, so a negative time raises. t = 0 is
        allowed and is the elastic value. The values are dimensionless, and a
        ``units`` attribute other than ``1`` raises.
        """
        if quantity not in SPADA_LOVE_QUANTITIES:
            raise ValueError(f"quantity must be one of {SPADA_LOVE_QUANTITIES}, not {quantity!r}")
        degree = int(degree)
        if degree < 1:
            raise ValueError(f"a Love number of degree {degree} has no place in these tests")
        if attributes.get("units", "1") != "1":
            raise ValueError(f"a Love number is dimensionless, not {attributes['units']}")
        attributes["units"] = "1"
        curve = Curve(time_kyr, values, dict(attributes))
        if curve.axis[0] < 0:
            raise ValueError("a time before the loading has no Heaviside Love number")
        self.time_domain[(quantity, degree)] = curve

    def add_polar_motion(self, load: str, chandler_wobble: str, quantity: str, time_kyr,
                         values, **attributes) -> None:
        """Add one polar motion quantity against time in kyr, for one load and wobble setting.

        ``chandler_wobble`` is ``included`` or ``excluded``. The unit is fixed
        by the quantity (`SPADA_POLAR_MOTION_UNITS`): degrees, or degrees per
        Myr for a rate. A ``units`` attribute that says otherwise raises.
        """
        if load not in SPADA_LOADS:
            raise ValueError(f"load must be one of {SPADA_LOADS}, not {load!r}")
        if chandler_wobble not in SPADA_CHANDLER_WOBBLE:
            raise ValueError(f"chandler_wobble must be one of {SPADA_CHANDLER_WOBBLE}, "
                             f"not {chandler_wobble!r}")
        if quantity not in SPADA_POLAR_MOTION_UNITS:
            raise ValueError(f"quantity must be one of {tuple(SPADA_POLAR_MOTION_UNITS)}, "
                             f"not {quantity!r}")
        units = SPADA_POLAR_MOTION_UNITS[quantity]
        if attributes.get("units", units) != units:
            raise ValueError(f"{quantity} is stored in {units}, not {attributes['units']}")
        attributes["units"] = units
        self.polar[(load, chandler_wobble, quantity)] = Curve(time_kyr, values, dict(attributes))

    def add_length_of_day(self, load: str, time_kyr, delta_lod_ms, **attributes) -> None:
        """Add the change in length of day in ms against time in kyr, for one load."""
        if load not in SPADA_LOADS:
            raise ValueError(f"load must be one of {SPADA_LOADS}, not {load!r}")
        if attributes.get("units", "ms") != "ms":
            raise ValueError(f"the length of day is stored in ms, not {attributes['units']}")
        attributes["units"] = "ms"
        self.lod[load] = Curve(time_kyr, delta_lod_ms, dict(attributes))

    def add_polar_motion_transfer_function(self, chandler_wobble: str, elastic, secular,
                                           frequencies, residues,
                                           source_file: str | None = None,
                                           **attributes) -> None:
        """Add A_e, A_s, and a_j with A_j per mode, for one Chandler wobble setting.

        The modes are sorted slowest first, by the real part of a_j from the
        value closest to zero, and each residue moves with its frequency. VB
        lists the modes fastest first and GS in an order of its own, so the
        sort gives every code one order.
        """
        if chandler_wobble not in SPADA_CHANDLER_WOBBLE:
            raise ValueError(f"chandler_wobble must be one of {SPADA_CHANDLER_WOBBLE}, "
                             f"not {chandler_wobble!r}")
        f = np.asarray(frequencies, dtype=complex).ravel()
        r = np.asarray(residues, dtype=complex).ravel()
        if f.shape != r.shape:
            raise ValueError(f"{f.size} frequencies and {r.size} residues")
        # lexsort takes its last key as the primary one: the real part,
        # descending, and the imaginary part only to order an exact tie.
        order = np.lexsort((f.imag, -f.real))
        self.pmtf[chandler_wobble] = PolarMotionTransferFunction(
            complex(elastic), complex(secular), f[order], r[order], source_file, dict(attributes))

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

    def time_domain_love_numbers(self, quantity: str, degree: int) -> Curve:
        """h, l or k of one degree against time. Raises KeyError naming the degrees held."""
        try:
            return self.time_domain[(quantity, int(degree))]
        except KeyError:
            raise KeyError(f"{self.member} case {self.case} has no {quantity}(t) of degree "
                           f"{degree}; it has degrees {self.time_domain_degrees(quantity)}") from None

    def time_domain_degrees(self, quantity: str) -> list[int]:
        """The degrees held for one time-domain quantity, ascending; empty if none."""
        return sorted(n for (q, n) in self.time_domain if q == quantity)

    def polar_motion(self, load: str, chandler_wobble: str, quantity: str) -> Curve:
        """One polar motion quantity. Raises KeyError naming what is available."""
        try:
            return self.polar[(load, chandler_wobble, quantity)]
        except KeyError:
            raise KeyError(f"{self.member} case {self.case} has no {quantity} for the {load} with "
                           f"the Chandler wobble {chandler_wobble}; it has {sorted(self.polar)}") from None

    def length_of_day(self, load: str) -> Curve:
        """The change in length of day for one load. Raises KeyError naming the loads held."""
        try:
            return self.lod[load]
        except KeyError:
            raise KeyError(f"{self.member} case {self.case} has no length of day for the {load}; "
                           f"it has {sorted(self.lod)}") from None

    def polar_motion_transfer_function(self, chandler_wobble: str) -> PolarMotionTransferFunction:
        """The transfer function for one Chandler wobble setting. Raises KeyError when absent."""
        try:
            return self.pmtf[chandler_wobble]
        except KeyError:
            raise KeyError(f"{self.member} case {self.case} has no transfer function with the "
                           f"Chandler wobble {chandler_wobble}; it has {sorted(self.pmtf)}") from None

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
            # Groups keyed by a quantity or a load hold one group per member
            # below that key; the transfer function holds the members directly.
            for kind in ("spatial", "time_domain_love_numbers", "polar_motion", "length_of_day"):
                if kind in ds.groups:
                    for key in ds[kind].groups.values():
                        found.update(key.groups)
            if "polar_motion_transfer_function" in ds.groups:
                found.update(ds["polar_motion_transfer_function"].groups)
            return sorted(found)
        return _read_dataset(path, collect)

    @classmethod
    def read(cls, path: pathlib.Path, member: str, case: str | None = None) -> "SpadaResult":
        """Read one member from a file in the converted layout.

        Masking is turned off, as AGENTS.md requires of every reader. The
        converted files hold no fill value, so a masked read would return the
        same numbers today; the call keeps it that way for a file that has one.
        Raises ValueError when ``case`` is given and the file records another
        case.
        """
        return _read_dataset(path, lambda ds: cls._read_open(ds, path, member, case))

    @classmethod
    def _read_open(cls, ds, path: pathlib.Path, member: str, case: str | None) -> "SpadaResult":
        """The body of `read`, on an open dataset whose masking is off."""
        stored = ds.getncattr("case") if "case" in ds.ncattrs() else None
        result = cls(case or stored, member)
        # A file holds one case, and `write` refuses to mix cases in it. A
        # `case` argument that disagrees with the file would score these
        # curves over the other test's intervals and report no error.
        if case is not None and stored is not None and result.case != _short_spada_case(stored):
            raise ValueError(f"{path} holds case {stored}, not {result.case}")

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
        if "time_domain_love_numbers" in ds.groups:
            for quantity, qgroup in ds["time_domain_love_numbers"].groups.items():
                if member not in qgroup.groups:
                    continue
                g = qgroup[member]
                for name, var in g.variables.items():
                    if name in g.dimensions:
                        continue
                    result.add_time_domain_love_numbers(
                        quantity, int(var.getncattr("degree")), g[var.dimensions[0]][:], var[:],
                        **attrs(var, skip=("long_name", "degree")))
        if "polar_motion" in ds.groups:
            for load, lgroup in ds["polar_motion"].groups.items():
                if member not in lgroup.groups:
                    continue
                for sgroup in lgroup[member].groups.values():
                    setting = sgroup.getncattr("chandler_wobble")
                    for name, var in sgroup.variables.items():
                        if name in sgroup.dimensions:
                            continue
                        result.add_polar_motion(load, setting, name, sgroup[var.dimensions[0]][:],
                                                var[:], **attrs(var, skip=("long_name",)))
        if "length_of_day" in ds.groups:
            for load, lgroup in ds["length_of_day"].groups.items():
                if member in lgroup.groups:
                    g = lgroup[member]
                    var = g["delta_lod"]
                    result.add_length_of_day(load, g[var.dimensions[0]][:], var[:],
                                             **attrs(var, skip=("long_name",)))
        if "polar_motion_transfer_function" in ds.groups and \
                member in ds["polar_motion_transfer_function"].groups:
            for sgroup in ds["polar_motion_transfer_function"][member].groups.values():
                def pair(name, g=sgroup):
                    # Real and imaginary parts are two variables, joined here
                    # without arithmetic, so the complex number is bit for bit
                    # the one written.
                    real = np.asarray(g[f"{name}_real"][...], dtype=float)
                    imag = np.asarray(g[f"{name}_imag"][...], dtype=float)
                    return real + 1j * imag
                result.add_polar_motion_transfer_function(
                    sgroup.getncattr("chandler_wobble"), complex(pair("elastic")),
                    complex(pair("secular")), pair("frequency"), pair("residue"),
                    source_file=sgroup.getncattr("source_file")
                    if "source_file" in sgroup.ncattrs() else None,
                    **attrs(sgroup, skip=("source_file", "chandler_wobble")))
        if not (result.love or result.spectrum is not None or result.fields or result.time_domain
                or result.polar or result.lod or result.pmtf):
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
                    # The residues are 2D, and netCDF4 warns on a 2D write.
                    with _without_netcdf4_shape_warning():
                        v[:] = values
                    v.units = units
                set_attrs(g, {"source_file": love.source_file, **love.attributes})

            if self.spectrum is not None:
                g = new_group(f"relaxation_spectrum/{self.member}")
                degree_and_mode(g, self.spectrum.degrees, self.spectrum.s_per_kyr.shape[1])
                v = g.createVariable("s", "f8", ("degree", "mode"), zlib=True, fill_value=False)
                with _without_netcdf4_shape_warning():
                    v[:] = self.spectrum.s_per_kyr
                v.units = "1/kyr"
                v.long_name = "relaxation frequency, negative; the relaxation time is -1/s"
                set_attrs(g, {"source_file": self.spectrum.source_file,
                              **self.spectrum.attributes})

            def curves(g, variables, axis_name, axis_units, axis_long_name, long_names,
                       keep_order=False, compress=True):
                """Write curves into one group, sharing an axis wherever two are equal.

                The first distinct axis is ``axis_name``, the next
                ``axis_name_2`` and so on, so a group whose variables are
                sampled alike has one axis. The variables are written in
                order of name, or in the order given when ``keep_order`` is
                true. With ``compress`` false the values are stored
                uncompressed, which for curves of a few hundred points is
                smaller: compression needs chunked storage, whose index costs
                more than it saves on short arrays.
                """
                axes: list[tuple[str, np.ndarray]] = []
                ordered = variables if keep_order else sorted(variables, key=lambda v: v[0])
                for name, curve in ordered:
                    dim = next((d for d, a in axes if a.shape == curve.axis.shape
                                and np.array_equal(a, curve.axis)), None)
                    if dim is None:
                        dim = axis_name if not axes else f"{axis_name}_{len(axes) + 1}"
                        axes.append((dim, curve.axis))
                        g.createDimension(dim, curve.axis.size)
                        coord = g.createVariable(dim, "f8", (dim,), fill_value=False)
                        coord[:] = curve.axis
                        coord.units = axis_units
                        coord.long_name = axis_long_name
                    v = g.createVariable(name, "f8", (dim,), zlib=compress, fill_value=False)
                    v[:] = curve.values
                    v.long_name = long_names[name]
                    set_attrs(v, curve.attributes)

            groups: dict[tuple[str, float], list[tuple[str, Curve]]] = {}
            for (load, quantity, epoch), curve in self.fields.items():
                groups.setdefault((load, epoch), []).append((quantity, curve))
            for (load, epoch), variables in sorted(groups.items()):
                g = new_group(f"spatial/{load}/{self.member}/epoch_{_epoch_label(epoch)}")
                g.epoch_kyr = _epoch_label(epoch)
                curves(g, variables, "colatitude", "degrees",
                       "angular distance from the load centre", SPADA_LONG_NAMES)

            by_quantity: dict[str, list[tuple[str, Curve]]] = {}
            long_names: dict[str, str] = {}
            for (quantity, degree), curve in sorted(self.time_domain.items()):
                # The degree goes into the attributes of the variable, so that
                # a reader takes it from there and never from the name.
                named = Curve(curve.axis, curve.values, {**curve.attributes, "degree": np.int32(degree)})
                by_quantity.setdefault(quantity, []).append((f"degree_{degree}", named))
                long_names[f"degree_{degree}"] = SPADA_TIME_DOMAIN_LONG_NAMES[quantity]
            for quantity, variables in sorted(by_quantity.items()):
                g = new_group(f"time_domain_love_numbers/{quantity}/{self.member}")
                # Sorted by degree, so that the axes are numbered in degree order.
                variables.sort(key=lambda v: int(v[1].attributes["degree"]))
                curves(g, variables, "time", "kyr", "time since the load was applied", long_names,
                       keep_order=True, compress=False)

            polar: dict[tuple[str, str], list[tuple[str, Curve]]] = {}
            for (load, setting, quantity), curve in self.polar.items():
                polar.setdefault((load, setting), []).append((quantity, curve))
            for (load, setting), variables in sorted(polar.items()):
                g = new_group(f"polar_motion/{load}/{self.member}/chandler_wobble_{setting}")
                g.chandler_wobble = setting
                curves(g, variables, "time", "kyr", "time since the load was applied",
                       SPADA_POLAR_MOTION_LONG_NAMES)

            for load, curve in sorted(self.lod.items()):
                g = new_group(f"length_of_day/{load}/{self.member}")
                curves(g, [("delta_lod", curve)], "time", "kyr", "time since the load was applied",
                       {"delta_lod": SPADA_LENGTH_OF_DAY_LONG_NAME})

            for setting, pmtf in sorted(self.pmtf.items()):
                g = new_group(f"polar_motion_transfer_function/{self.member}/chandler_wobble_{setting}")
                g.chandler_wobble = setting
                modes = pmtf.frequencies.size
                g.createDimension("mode", modes)
                m = g.createVariable("mode", "i4", ("mode",), fill_value=False)
                m[:] = np.arange(1, modes + 1)
                m.long_name = "rotational mode, 1 the slowest"
                for name, value, units, long_name in (
                        ("elastic", pmtf.elastic, "1", "elastic residue A_e"),
                        ("secular", pmtf.secular, "1/kyr", "secular residue A_s")):
                    for part in ("real", "imag"):
                        v = g.createVariable(f"{name}_{part}", "f8", (), fill_value=False)
                        v.assignValue(getattr(value, part))
                        v.units = units
                        v.long_name = f"{long_name}, {part} part"
                for name, values, long_name in (
                        ("frequency", pmtf.frequencies, "rotational frequency a_j"),
                        ("residue", pmtf.residues, "rotational residue A_j")):
                    for part in ("real", "imag"):
                        v = g.createVariable(f"{name}_{part}", "f8", ("mode",), zlib=True,
                                             fill_value=False)
                        v[:] = getattr(values, part)
                        v.units = "1/kyr"
                        v.long_name = f"{long_name}, {part} part"
                set_attrs(g, {"source_file": pmtf.source_file, **pmtf.attributes})
        return path
