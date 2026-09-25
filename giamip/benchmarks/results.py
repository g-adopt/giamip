"""Profiles and time series of one run of one Martinec case, from any code.

The published contributors, VEGA, lovejx and G-ADOPT all produce the same
things for a case: U, N and S along two meridians at the terminal time, and
h_UF, ice mass and ocean area against time. `BenchmarkResult` holds those
curves for one code, reads them from a NetCDF file and writes them to one, so
that plotting and scoring treat a reference contributor and our own solvers
the same way.

The file layout is the layout of the converted reference data
(``giamip.benchmarks.martinec2018``):

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

A lovejx run of case C is therefore written as, for example,
``outputs/martinec2018/lovejx/martinec2018-C.nc`` with member ``lovejx``, and
``BenchmarkResult.read`` opens it exactly as it opens a reference file.

    >>> from giamip.benchmarks.results import BenchmarkResult
    >>> vega = BenchmarkResult.reference("C", "ZM")
    >>> curve = vega.profile("basin", "S")
    >>> curve.axis[:2], curve.values[:2]            # doctest: +SKIP
"""

from __future__ import annotations

import dataclasses
import math
import pathlib
import re

import numpy as np

from giamip import _netcdf
from giamip.benchmarks import martinec2018

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
        such as ``lovejx`` or ``g-adopt``.
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
        """Add U, N or S along the ``load`` or ``basin`` meridian of the case.

        Parameters
        ----------
        profile
            ``"load"`` (the meridian through the ice centre) or ``"basin"``
            (through the basin centre); `MartinecCase.profiles` gives the
            longitude of each.
        quantity
            ``"U"`` (vertical displacement of the solid surface), ``"N"``
            (geoid displacement) or ``"S"`` (sea-surface variation), all in
            metres and positive upward (docs/reference/conventions.qmd).
        colatitude_deg
            The colatitudes of the values, in degrees, in any order and at any
            spacing; the curve is sorted on construction.
        values
            The values, one per colatitude.
        **attributes
            Anything to keep with the curve, for example ``time_kyr``; it is
            written to the file as attributes of the variable.
        """
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
        """The converted reference file of a case, fetched and cached on first use.

        `giamip.data.locate` finds it in a checkout or in the download root,
        or downloads it from data.gadopt.org and checks its SHA-256.
        """
        from giamip import data
        letter = case.rsplit("-", 1)[-1].upper()
        return data.locate(f"martinec2018/martinec2018-{letter}.nc", "converted")

    @classmethod
    def reference(cls, case: str, member: str | None = None) -> "BenchmarkResult":
        """The reference curves of a case, or one published contributor.

        Parameters
        ----------
        case
            The case letter or id.
        member
            None for the curves that scoring compares with: the group
            ``reference`` of the converted file, returned as a result with
            member ``"VEGA"``. Each curve's attributes say how it was built
            (``method``, ``member``, ``source_files``, ``correction`` and
            ``built_from``). A contributor's code, ``"ZM"`` included, reads
            that contributor's curves as the archive shipped them.
        """
        path = cls.reference_path(case)
        if member is not None:
            return cls.read(path, member, case=case)
        return cls.read_reference(path, case=case)

    @classmethod
    def read_reference(cls, path: pathlib.Path, case: str | None = None) -> "BenchmarkResult":
        """Read the group ``reference`` of a converted file, as member ``"VEGA"``.

        Raises KeyError when the file has no such group, and ValueError when
        ``case`` disagrees with the case the file records.
        """
        with _netcdf.open_read(path) as ds:
            stored = _netcdf.attributes(ds).get("case")
            result = cls(case or stored or "?", "VEGA")
            if case is not None and stored is not None and result.case != cls(stored, "VEGA").case:
                raise ValueError(f"{path} holds case {stored}, not {result.case}")
            if "reference" not in ds.groups:
                raise KeyError(f"{path} has no reference group")
            reference = ds.groups["reference"]
            if "profiles" in reference.groups:
                for profile, pgroup in reference.groups["profiles"].groups.items():
                    for name, curve in _curves(pgroup):
                        result.profiles[(profile, name)] = curve
            if "time_series" in reference.groups:
                for name, curve in _curves(reference.groups["time_series"]):
                    result.time_series[name] = curve
        return result

    @staticmethod
    def members(path: pathlib.Path) -> list[str]:
        """Every member with at least one curve in a file.

        The ``reference`` group of a converted file holds the reference curves
        that scoring uses, not a code's curves, so it is not a member.
        """
        found = set()
        with _netcdf.open_read(path) as ds:
            for kind in ("profiles", "differences"):
                if kind in ds.groups:
                    for profile in ds.groups[kind].groups.values():
                        found.update(profile.groups)
            if "time_series" in ds.groups:
                found.update(ds.groups["time_series"].groups)
        return sorted(found)

    @classmethod
    def read(cls, path: pathlib.Path, member: str, case: str | None = None) -> "BenchmarkResult":
        """Read one member's curves from a file in the converted layout.

        The file is read through h5netcdf, which applies no mask, as AGENTS.md
        requires of every reader. Raises ValueError when ``case`` is given and
        the file records another case.
        """
        with _netcdf.open_read(path) as ds:
            stored = _netcdf.attributes(ds).get("case")
            result = cls(case or stored or "?", member)
            # A file holds one case, and `write` refuses to mix cases in it. A
            # `case` argument that disagrees with the file would score these
            # curves over the other case's intervals and report no error.
            if case is not None and stored is not None and result.case != cls(stored, member).case:
                raise ValueError(f"{path} holds case {stored}, not {result.case}")
            for kind, store in (("profiles", result.profiles), ("differences", result.differences)):
                if kind not in ds.groups:
                    continue
                for profile, pgroup in ds.groups[kind].groups.items():
                    if member in pgroup.groups:
                        for name, curve in _curves(pgroup.groups[member]):
                            store[(profile, name)] = curve
            if "time_series" in ds.groups and member in ds.groups["time_series"].groups:
                for name, curve in _curves(ds.groups["time_series"].groups[member]):
                    result.time_series[name] = curve
        if not (result.profiles or result.differences or result.time_series):
            raise KeyError(f"{path} has no curves for member {member!r}")
        return result

    def write(self, path: pathlib.Path) -> pathlib.Path:
        """Write this member's curves into a file in the converted layout.

        The file is created if it does not exist. A member that already has
        a group in the file raises, so a second run cannot silently mix with
        the first. The file is written through h5netcdf and is an ordinary
        netCDF-4 file.
        """
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with _netcdf.open_write(path, append=path.exists()) as ds:
            stored = _netcdf.attributes(ds).get("case")
            if stored is None:
                _netcdf.set_attributes(ds, {
                    "case": self.case,
                    "title": f"Martinec et al. (2018) benchmark case {self.case}",
                    "conventions_reference": "docs/reference/conventions.qmd"})
            elif stored != self.case:
                raise ValueError(f"{path} holds case {stored}, not {self.case}")
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
                if _netcdf.has_group(ds, group_path):
                    raise ValueError(f"{path} already has {group_path}")
                g = _netcdf.create_group(ds, group_path)
                variables = sorted(variables, key=lambda v: v[0])
                # Every variable of one group shares the axis name and units
                # (colatitude in degrees or time in kyr).
                axis_name, axis_units = variables[0][3], variables[0][4]
                curves = []
                for name, curve, units, _, _ in variables:
                    attrs = {"units": units}
                    if name in martinec2018.LONG_NAMES:
                        attrs["long_name"] = martinec2018.LONG_NAMES[name]
                    attrs.update({k: v for k, v in curve.attributes.items()
                                  if k not in ("units", "long_name")})
                    curves.append((name, curve.axis, curve.values, attrs))
                _netcdf.write_curves(g, curves, axis_name=axis_name,
                                     coordinate_attributes={"units": axis_units})
        return path


def _curves(group):
    """Every curve of one group, as ``(name, Curve)``, with its attributes decoded."""
    for name, var in _netcdf.data_variables(group).items():
        yield name, Curve(_netcdf.axis_of(group, var), np.asarray(var[...]),
                          _netcdf.attributes(var))


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
# published contributor, lovejx and G-ADOPT are read, written and scored the
# same way.
#
# The file layout, shared with the converted reference data
# (``giamip.benchmarks.spada2011``):
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
        ``lovejx`` or ``g-adopt``.
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
        """The converted reference file of a case, which holds every contributor.

        Fetched and cached on first use through `giamip.data.locate`.
        """
        from giamip import data
        short = _short_spada_case(case)
        return data.locate(f"spada2011/spada2011-{short}.nc", "converted")

    @classmethod
    def reference(cls, case: str, member: str) -> "SpadaResult":
        """A published contributor, from the converted data."""
        return cls.read(cls.reference_path(case), member, case=case)

    @staticmethod
    def members(path: pathlib.Path) -> list[str]:
        """Every member with at least one value in a file."""
        found = set()
        with _netcdf.open_read(path) as ds:
            groups = ds.groups
            if "love_numbers" in groups:
                for quantity in groups["love_numbers"].groups.values():
                    found.update(quantity.groups)
            if "relaxation_spectrum" in groups:
                found.update(groups["relaxation_spectrum"].groups)
            # Groups keyed by a quantity or a load hold one group per member
            # below that key; the transfer function holds the members directly.
            for kind in ("spatial", "time_domain_love_numbers", "polar_motion", "length_of_day"):
                if kind in groups:
                    for key in groups[kind].groups.values():
                        found.update(key.groups)
            if "polar_motion_transfer_function" in groups:
                found.update(groups["polar_motion_transfer_function"].groups)
        return sorted(found)

    @classmethod
    def read(cls, path: pathlib.Path, member: str, case: str | None = None) -> "SpadaResult":
        """Read one member from a file in the converted layout.

        The file is read through h5netcdf, which applies no mask, as AGENTS.md
        requires of every reader. Raises ValueError when ``case`` is given and
        the file records another case.
        """
        with _netcdf.open_read(path) as ds:
            return cls._read_open(ds, path, member, case)

    @classmethod
    def _read_open(cls, ds, path: pathlib.Path, member: str, case: str | None) -> "SpadaResult":
        """The body of `read`, on an open file."""
        stored = _netcdf.attributes(ds).get("case")
        result = cls(case or stored, member)
        # A file holds one case, and `write` refuses to mix cases in it. A
        # `case` argument that disagrees with the file would score these
        # curves over the other test's intervals and report no error.
        if case is not None and stored is not None and result.case != _short_spada_case(stored):
            raise ValueError(f"{path} holds case {stored}, not {result.case}")
        groups = ds.groups

        def values(group, name):
            return np.asarray(group.variables[name][...])

        if "love_numbers" in groups:
            for quantity, qgroup in groups["love_numbers"].groups.items():
                if member not in qgroup.groups:
                    continue
                g = qgroup.groups[member]
                extra = _netcdf.attributes(g, skip=("source_file",))
                result.add_love_numbers(quantity, values(g, "degree"), values(g, "elastic"),
                                        values(g, "fluid"), values(g, "residues"),
                                        source_file=_netcdf.attributes(g).get("source_file"),
                                        **extra)
        if "relaxation_spectrum" in groups and member in groups["relaxation_spectrum"].groups:
            g = groups["relaxation_spectrum"].groups[member]
            result.add_relaxation_spectrum(
                values(g, "degree"), values(g, "s"),
                source_file=_netcdf.attributes(g).get("source_file"),
                **_netcdf.attributes(g, skip=("source_file",)))
        if "spatial" in groups:
            for load, lgroup in groups["spatial"].groups.items():
                if member not in lgroup.groups:
                    continue
                for egroup in lgroup.groups[member].groups.values():
                    epoch = float(_netcdf.attributes(egroup)["epoch_kyr"])
                    for name, var in _netcdf.data_variables(egroup).items():
                        result.add_spatial(load, name, epoch, _netcdf.axis_of(egroup, var),
                                           np.asarray(var[...]),
                                           **_netcdf.attributes(var, skip=("long_name",)))
        if "time_domain_love_numbers" in groups:
            for quantity, qgroup in groups["time_domain_love_numbers"].groups.items():
                if member not in qgroup.groups:
                    continue
                g = qgroup.groups[member]
                for name, var in _netcdf.data_variables(g).items():
                    result.add_time_domain_love_numbers(
                        quantity, int(_netcdf.attributes(var)["degree"]),
                        _netcdf.axis_of(g, var), np.asarray(var[...]),
                        **_netcdf.attributes(var, skip=("long_name", "degree")))
        if "polar_motion" in groups:
            for load, lgroup in groups["polar_motion"].groups.items():
                if member not in lgroup.groups:
                    continue
                for sgroup in lgroup.groups[member].groups.values():
                    setting = _netcdf.attributes(sgroup)["chandler_wobble"]
                    for name, var in _netcdf.data_variables(sgroup).items():
                        result.add_polar_motion(load, setting, name, _netcdf.axis_of(sgroup, var),
                                                np.asarray(var[...]),
                                                **_netcdf.attributes(var, skip=("long_name",)))
        if "length_of_day" in groups:
            for load, lgroup in groups["length_of_day"].groups.items():
                if member in lgroup.groups:
                    g = lgroup.groups[member]
                    var = g.variables["delta_lod"]
                    result.add_length_of_day(load, _netcdf.axis_of(g, var), np.asarray(var[...]),
                                             **_netcdf.attributes(var, skip=("long_name",)))
        if "polar_motion_transfer_function" in groups and \
                member in groups["polar_motion_transfer_function"].groups:
            for sgroup in groups["polar_motion_transfer_function"].groups[member].groups.values():
                def pair(name, g=sgroup):
                    # Real and imaginary parts are two variables, joined here
                    # without arithmetic, so the complex number is bit for bit
                    # the one written.
                    real = np.asarray(g.variables[f"{name}_real"][...], dtype=float)
                    imag = np.asarray(g.variables[f"{name}_imag"][...], dtype=float)
                    return real + 1j * imag
                group_attributes = _netcdf.attributes(sgroup)
                result.add_polar_motion_transfer_function(
                    group_attributes["chandler_wobble"], complex(pair("elastic")),
                    complex(pair("secular")), pair("frequency"), pair("residue"),
                    source_file=group_attributes.get("source_file"),
                    **_netcdf.attributes(sgroup, skip=("source_file", "chandler_wobble")))
        if not (result.love or result.spectrum is not None or result.fields or result.time_domain
                or result.polar or result.lod or result.pmtf):
            raise KeyError(f"{path} has nothing for member {member!r}")
        return result

    def write(self, path: pathlib.Path) -> pathlib.Path:
        """Write this member into a file in the converted layout.

        The file is created if it does not exist, so several members share one
        file. A member that already has a group in the file raises, so a second
        run cannot mix with the first. The file is written through h5netcdf and
        is an ordinary netCDF-4 file.
        """
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with _netcdf.open_write(path, append=path.exists()) as ds:
            stored = _netcdf.attributes(ds).get("case")
            if stored is None:
                _netcdf.set_attributes(ds, {
                    "case": self.case,
                    "title": f"Spada et al. (2011) benchmark case {self.case}",
                    "conventions_reference": "docs/reference/conventions.qmd"})
            elif stored != self.case:
                raise ValueError(f"{path} holds case {stored}, not {self.case}")

            def new_group(group_path):
                if _netcdf.has_group(ds, group_path):
                    raise ValueError(f"{path} already has {group_path}")
                return _netcdf.create_group(ds, group_path)

            def degree_and_mode(g, degrees, modes):
                g.dimensions["degree"] = degrees.size
                g.dimensions["mode"] = modes
                d = g.create_variable("degree", ("degree",), "i4", data=degrees)
                d.attrs["long_name"] = "spherical harmonic degree"
                m = g.create_variable("mode", ("mode",), "i4", data=np.arange(1, modes + 1))
                m.attrs["long_name"] = "relaxation mode, 1 the slowest"

            for quantity, love in sorted(self.love.items()):
                g = new_group(f"love_numbers/{quantity}/{self.member}")
                degree_and_mode(g, love.degrees, love.residues.shape[1])
                for name, data, dims, units in (
                        ("elastic", love.elastic, ("degree",), "1"),
                        ("fluid", love.fluid, ("degree",), "1"),
                        ("residues", love.residues, ("degree", "mode"), "1/kyr")):
                    v = g.create_variable(name, dims, "f8", data=np.asarray(data, dtype=float),
                                          compression="gzip")
                    v.attrs["units"] = units
                _netcdf.set_attributes(g, {"source_file": love.source_file, **love.attributes})

            if self.spectrum is not None:
                g = new_group(f"relaxation_spectrum/{self.member}")
                degree_and_mode(g, self.spectrum.degrees, self.spectrum.s_per_kyr.shape[1])
                v = g.create_variable("s", ("degree", "mode"), "f8",
                                      data=np.asarray(self.spectrum.s_per_kyr, dtype=float),
                                      compression="gzip")
                _netcdf.set_attributes(v, {
                    "units": "1/kyr",
                    "long_name": "relaxation frequency, negative; the relaxation time is -1/s"})
                _netcdf.set_attributes(g, {"source_file": self.spectrum.source_file,
                                           **self.spectrum.attributes})

            def curves(g, variables, axis_name, axis_units, axis_long_name, long_names,
                       keep_order=False, compress=True):
                """Write ``(name, Curve)`` pairs with `_netcdf.write_curves`.

                The variables are written in order of name, or in the order
                given when ``keep_order`` is true. Each variable's long name
                comes first and its own attributes after it.
                """
                ordered = variables if keep_order else sorted(variables, key=lambda v: v[0])
                _netcdf.write_curves(
                    g, [(name, curve.axis, curve.values,
                         {"long_name": long_names[name], **curve.attributes})
                        for name, curve in ordered],
                    axis_name=axis_name,
                    coordinate_attributes={"units": axis_units, "long_name": axis_long_name},
                    compress=compress)

            groups: dict[tuple[str, float], list[tuple[str, Curve]]] = {}
            for (load, quantity, epoch), curve in self.fields.items():
                groups.setdefault((load, epoch), []).append((quantity, curve))
            for (load, epoch), variables in sorted(groups.items()):
                g = new_group(f"spatial/{load}/{self.member}/epoch_{_epoch_label(epoch)}")
                g.attrs["epoch_kyr"] = _epoch_label(epoch)
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
                g.attrs["chandler_wobble"] = setting
                curves(g, variables, "time", "kyr", "time since the load was applied",
                       SPADA_POLAR_MOTION_LONG_NAMES)

            for load, curve in sorted(self.lod.items()):
                g = new_group(f"length_of_day/{load}/{self.member}")
                curves(g, [("delta_lod", curve)], "time", "kyr", "time since the load was applied",
                       {"delta_lod": SPADA_LENGTH_OF_DAY_LONG_NAME})

            for setting, pmtf in sorted(self.pmtf.items()):
                g = new_group(f"polar_motion_transfer_function/{self.member}/chandler_wobble_{setting}")
                g.attrs["chandler_wobble"] = setting
                modes = pmtf.frequencies.size
                g.dimensions["mode"] = modes
                m = g.create_variable("mode", ("mode",), "i4", data=np.arange(1, modes + 1))
                m.attrs["long_name"] = "rotational mode, 1 the slowest"
                for name, value, units, long_name in (
                        ("elastic", pmtf.elastic, "1", "elastic residue A_e"),
                        ("secular", pmtf.secular, "1/kyr", "secular residue A_s")):
                    for part in ("real", "imag"):
                        # A scalar variable: no dimension, one value.
                        v = g.create_variable(f"{name}_{part}", (), "f8",
                                              data=np.float64(getattr(value, part)))
                        _netcdf.set_attributes(v, {"units": units,
                                                   "long_name": f"{long_name}, {part} part"})
                for name, data, long_name in (
                        ("frequency", pmtf.frequencies, "rotational frequency a_j"),
                        ("residue", pmtf.residues, "rotational residue A_j")):
                    for part in ("real", "imag"):
                        v = g.create_variable(f"{name}_{part}", ("mode",), "f8",
                                              data=np.asarray(getattr(data, part), dtype=float),
                                              compression="gzip")
                        _netcdf.set_attributes(v, {"units": "1/kyr",
                                                   "long_name": f"{long_name}, {part} part"})
                _netcdf.set_attributes(g, {"source_file": pmtf.source_file, **pmtf.attributes})
        return path
