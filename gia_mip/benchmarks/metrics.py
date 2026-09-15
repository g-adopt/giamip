"""The differences the paper uses to score a code against the reference.

Martinec et al. (2018), eq. 31: for a quantity q sampled at n points inside a
colatitude interval at a fixed time,

    RMS = sqrt( (1/n) sum_i (q_i - q_i_ref)^2 ),   MAX = max_i |q_i - q_i_ref|.

The paper does not say which points. Here they are the points of the reference
inside the interval, and the candidate is interpolated linearly onto them.
Extrapolation is refused. A grid that stops just short of an end of the
interval, such as a colatitude grid whose first node is 0.044 degrees where the
interval starts at 0, is accepted when the gap is at most 1 per cent of the
interval; the comparison then uses the reference points the candidate covers,
and reports that range.

The reference is VEGA, contributor ZM. For three profiles the VEGA file that
Martinec shipped is not the one the published differences were computed
against, so the reference is recovered as a contributor's profile minus that
contributor's own difference file, which reproduces the published reference:

* case A (correction benchmarkA-vega-supplied-run-differs), recovered from GS;
* the case B basin and both case D profiles
  (correction martinec2018-caseD-vega-profile-grid), recovered from YS, whose
  2048-point grid is the finest.

Spada et al. (2011) has no such reference code; `spada_score` at the end of
this module scores against each contributor separately.
"""

from __future__ import annotations

import warnings

import numpy as np

from gia_mip.benchmarks.inputs import MartinecCase, SpadaCase
from gia_mip.benchmarks.results import BenchmarkResult, Curve, SpadaResult

# Which contributor the reference is recovered from, where VEGA as shipped
# cannot be used. The key is (case, profile).
RECOVER_REFERENCE_FROM = {
    ("A", "load"): "GS",
    ("B", "basin"): "YS",
    ("D", "load"): "YS",
    ("D", "basin"): "YS",
}


def difference_metrics(candidate: Curve, reference: Curve,
                       interval: tuple[float, float] | None = None) -> dict:
    """RMS and maximum absolute difference on the reference points in an interval.

    Parameters
    ----------
    candidate, reference
        The two curves, on any grids.
    interval
        Closed interval of the axis. Default: the whole reference axis.

    Returns
    -------
    dict
        ``rms``, ``max``, ``n`` (points used), ``at`` (axis value of the
        largest difference) and ``covered`` (the axis range compared).

    Raises
    ------
    ValueError
        If the candidate leaves more than 1 per cent of the interval uncovered
        at either end, or no reference point lies in the covered range.
    """
    lo, hi = interval if interval is not None else (reference.axis[0], reference.axis[-1])
    allowed_gap = 0.01 * (hi - lo)
    start = max(lo, reference.axis[0])
    end = min(hi, reference.axis[-1])
    if candidate.axis[0] - start > allowed_gap or end - candidate.axis[-1] > allowed_gap:
        raise ValueError(f"the candidate covers [{candidate.axis[0]}, {candidate.axis[-1]}] "
                         f"and the comparison needs [{start}, {end}]")
    lo_used, hi_used = max(start, candidate.axis[0]), min(end, candidate.axis[-1])
    inside = (reference.axis >= lo_used) & (reference.axis <= hi_used)
    x = reference.axis[inside]
    if x.size == 0:
        raise ValueError(f"no reference point in [{lo_used}, {hi_used}]")
    # Linear interpolation of the candidate onto the reference points.
    d = np.interp(x, candidate.axis, candidate.values) - reference.values[inside]
    i = int(np.argmax(np.abs(d)))
    return {"rms": float(np.sqrt(np.mean(d ** 2))), "max": float(np.abs(d[i])),
            "n": int(x.size), "at": float(x[i]), "covered": (float(x[0]), float(x[-1]))}


def reference_profile(case: str, profile: str, quantity: str) -> Curve:
    """The reference curve for one profile quantity of one case.

    VEGA as shipped, except where `RECOVER_REFERENCE_FROM` names a
    contributor. The returned curve's ``attributes["reference"]`` says which.
    """
    letter = case.rsplit("-", 1)[-1].upper()
    via = RECOVER_REFERENCE_FROM.get((letter, profile))
    if via is None:
        curve = BenchmarkResult.reference(letter, "ZM").profile(profile, quantity)
        return Curve(curve.axis, curve.values, {"reference": "VEGA as shipped"})
    contributor = BenchmarkResult.reference(letter, via)
    value = contributor.profile(profile, quantity)
    difference = contributor.differences[(profile, quantity)]
    if not np.allclose(value.axis, difference.axis, atol=1e-4):
        raise ValueError(f"{via} profile and difference file are on different grids")
    return Curve(value.axis, value.values - difference.values,
                 {"reference": f"VEGA recovered as {via} minus its difference file"})


def reference_time_series(case: str, quantity: str) -> Curve:
    """VEGA's time series of one quantity."""
    letter = case.rsplit("-", 1)[-1].upper()
    curve = BenchmarkResult.reference(letter, "ZM").series(quantity)
    return Curve(curve.axis, curve.values, {"reference": "VEGA as shipped"})


def score(result: BenchmarkResult) -> list[dict]:
    """Score every profile and time series a result holds, as the paper does.

    Profiles are compared over the colatitude interval of the case file, and
    time series over their whole time interval. A quantity the result does not
    hold is skipped, so a solver can be scored on part of a case.

    Returns
    -------
    list of dict
        One row per comparison: ``kind``, ``profile`` (None for a time
        series), ``quantity``, ``interval``, ``reference``, and the entries of
        `difference_metrics`.
    """
    setup = MartinecCase(result.case)
    rows = []
    for spec in setup.profiles:
        for quantity in spec.quantities:
            if (spec.name, quantity) not in result.profiles:
                continue
            ref = reference_profile(result.case, spec.name, quantity)
            m = difference_metrics(result.profiles[(spec.name, quantity)], ref,
                                   spec.colatitude_interval_deg)
            rows.append({"kind": "profile", "profile": spec.name, "quantity": quantity,
                         "interval": spec.colatitude_interval_deg,
                         "reference": ref.attributes["reference"], **m})
    for spec in setup.time_series:
        for quantity in spec.quantities:
            if quantity not in result.time_series:
                continue
            ref = reference_time_series(result.case, quantity)
            m = difference_metrics(result.time_series[quantity], ref, spec.time_interval_kyr)
            rows.append({"kind": "time_series", "profile": None, "quantity": quantity,
                         "interval": spec.time_interval_kyr,
                         "reference": ref.attributes["reference"], **m})
    return rows


# =====================================================================
# Spada et al. (2011)
# =====================================================================
#
# Spada et al. (2011) has no reference code. Its figures overlay the
# contributors and the paper judges agreement by eye, so no one of them is
# right by definition, and an average of them would be a result that no code
# produced. `spada_score` therefore scores a result against each reference
# contributor separately and labels every row with the contributor's code. A
# reader who wants one number chooses the contributor and says so.
#
# Every row carries the same keys, whatever its kind, so the rows form one
# table: a key that does not apply to a kind is None.

def _restricted(curve: Curve, keep: np.ndarray) -> Curve:
    """A curve on a subset of its axis values."""
    mask = np.isin(curve.axis, keep)
    return Curve(curve.axis[mask], curve.values[mask], dict(curve.attributes))


def _spatial_intervals(case: str) -> list[tuple[float, float]]:
    """The near-field and global colatitude intervals of a spatial case, in degrees."""
    setup = SpadaCase(case)
    near, whole = setup.near_field_colatitudes_deg(), setup.global_colatitudes_deg()
    return [(float(near[0]), float(near[-1])), (float(whole[0]), float(whole[-1]))]


def _degree_rows(candidate_degrees, reference_degrees, excluded) -> np.ndarray:
    """The degrees both hold, less those the reference marks as not to score."""
    common = np.intersect1d(np.asarray(candidate_degrees), np.asarray(reference_degrees))
    return common[~np.isin(common, np.asarray(excluded, dtype=int))]


def _row(kind: str, quantity: str, interval, code: str, metrics: dict, *, load=None,
         epoch_kyr=None, degree=None, chandler_wobble=None) -> dict:
    """One score row with every key present, None where a key does not apply."""
    return {"kind": kind, "load": load, "quantity": quantity, "epoch_kyr": epoch_kyr,
            "degree": degree, "chandler_wobble": chandler_wobble, "interval": interval,
            "reference": code, **metrics}


def _log_time(curve: Curve) -> Curve:
    """The curve against log10 of time in kyr, with the times at or below zero left out.

    Love numbers in the time domain are sampled logarithmically from 1e-3 to
    1e6 kyr. Interpolating linearly in t between such points misses the
    relaxation, and a coverage rule of 1 per cent of the interval would be
    1e4 kyr wide on a linear axis, so the comparison runs in log10 t. The
    elastic value at t = 0 has no logarithm and is not part of it.
    """
    keep = curve.axis > 0
    return Curve(np.log10(curve.axis[keep]), curve.values[keep], dict(curve.attributes))


def _time_domain_interval(window: tuple[float, float] | None, reference: Curve) -> tuple[float, float]:
    """The times compared, in kyr: the reference's positive times, inside the case's window if it has one."""
    positive = reference.axis[reference.axis > 0]
    if positive.size == 0:
        raise ValueError("the reference has no time after the loading")
    lo, hi = float(positive[0]), float(positive[-1])
    if window is not None:
        lo, hi = max(lo, window[0]), min(hi, window[1])
    return lo, hi


def _excluded_degrees(attributes: dict) -> list[int]:
    """Degrees listed in ``degrees_not_to_score``, text such as ``3 4``, or none.

    The converter lists ZM's spectrum rows that copy degree 2
    (correction spada-t01-01-zm-degrees-copy-degree-2).
    """
    text = attributes.get("degrees_not_to_score", "")
    return [int(v) for v in str(text).split()]


def spada_score(result: SpadaResult, references: dict[str, SpadaResult]) -> list[dict]:
    """Score one result against each reference contributor, with `difference_metrics`.

    Spatial fields are compared at every epoch both results hold, exactly: a
    solver's fluid limit (``math.inf``) is never paired with GS's run at 1000
    kyr. Each field gets two rows, one over the near field (0 to 25 degrees)
    and one over the whole meridian (0 to 180 degrees), measured at the
    reference's points. Love numbers are compared per degree over the degrees
    both hold: elastic and fluid for each of h, l and k. The relaxation
    spectrum is compared mode by mode over the common degrees. Modes pair by
    position, so a spectrum is scored only when both results hold the same
    number of modes; otherwise it adds no rows and a warning names the two
    counts. Residues are not
    scored, because their size spans twenty orders of magnitude over the
    degrees and a difference in one number says little.

    Love numbers in the time domain are compared per quantity and degree in
    log10 t, at the reference's times after the loading that lie inside the
    case's time window (tests 5/1 and 6/1) or over the whole reference (test
    7/1); the interval, ``at`` and ``covered`` are reported in kyr. Polar
    motion is compared per load, Chandler wobble setting and quantity, and
    the length of day per load, both linearly in time over the whole
    reference. With the Chandler wobble included, polar motion oscillates
    with a period of about 1.2 years, and the references are sampled once a
    year, so a candidate is compared meaningfully only at the reference's own
    times. The transfer function scores A_e and A_s as single values, and
    a_j and A_j mode by mode when both results hold the same number of modes;
    otherwise the modes add no rows and a warning names both counts.

    Parameters
    ----------
    result
        The candidate, for example a radopt run.
    references
        Member code to reference result, for example the GS and VK members of
        ``SpadaResult.reference_path(case)``.

    Returns
    -------
    list of dict
        One row per comparison: ``kind`` (``spatial``, ``love_numbers``,
        ``relaxation_spectrum``, ``time_domain_love_numbers``,
        ``polar_motion``, ``length_of_day`` or
        ``polar_motion_transfer_function``), ``load``, ``quantity`` (for Love
        numbers ``h_elastic`` and so on, for the spectrum ``s_1`` and so on,
        for the transfer function ``frequency_real`` and so on),
        ``epoch_kyr``, ``degree``, ``chandler_wobble``, ``interval``,
        ``reference`` (the member code), and the entries of
        `difference_metrics`. A key that does not apply to a kind is None.
    """
    rows: list[dict] = []
    intervals = _spatial_intervals(result.case) if result.fields else []
    window = SpadaCase(result.case).time_window_kyr if result.time_domain else None
    for code, reference in references.items():
        # Surface fields, per load, quantity and epoch.
        for (load, quantity, epoch), curve in sorted(result.fields.items()):
            if (load, quantity, epoch) not in reference.fields:
                continue
            ref = reference.fields[(load, quantity, epoch)]
            for interval in intervals:
                m = difference_metrics(curve, ref, interval)
                rows.append(_row("spatial", quantity, interval, code, m, load=load, epoch_kyr=epoch))
        # Love numbers in the time domain, per quantity and degree, in log10 t.
        for (quantity, degree), curve in sorted(result.time_domain.items()):
            if (quantity, degree) not in reference.time_domain:
                continue
            ref = reference.time_domain[(quantity, degree)]
            lo, hi = _time_domain_interval(window, ref)
            m = difference_metrics(_log_time(curve), _log_time(ref), (np.log10(lo), np.log10(hi)))
            # Back from log10 t to kyr, so every time in a row reads in kyr.
            m["at"] = float(10.0 ** m["at"])
            m["covered"] = tuple(float(10.0 ** c) for c in m["covered"])
            rows.append(_row("time_domain_love_numbers", quantity, (lo, hi), code, m,
                             degree=degree))
        # Polar motion, per load, wobble setting and quantity.
        for (load, setting, quantity), curve in sorted(result.polar.items()):
            if (load, setting, quantity) not in reference.polar:
                continue
            ref = reference.polar[(load, setting, quantity)]
            m = difference_metrics(curve, ref)
            rows.append(_row("polar_motion", quantity, (float(ref.axis[0]), float(ref.axis[-1])),
                             code, m, load=load, chandler_wobble=setting))
        # The length of day, per load.
        for load, curve in sorted(result.lod.items()):
            if load not in reference.lod:
                continue
            ref = reference.lod[load]
            m = difference_metrics(curve, ref)
            rows.append(_row("length_of_day", "delta_lod", (float(ref.axis[0]), float(ref.axis[-1])),
                             code, m, load=load))
        # The polar motion transfer function, per wobble setting.
        for setting, mine in sorted(result.pmtf.items()):
            if setting not in reference.pmtf:
                continue
            ref = reference.pmtf[setting]
            for name in ("elastic", "secular"):
                for part in ("real", "imag"):
                    d = abs(getattr(getattr(mine, name), part) - getattr(getattr(ref, name), part))
                    single = {"rms": d, "max": d, "n": 1, "at": None, "covered": None}
                    rows.append(_row("polar_motion_transfer_function", f"{name}_{part}", None,
                                     code, single, chandler_wobble=setting))
            if mine.frequencies.size != ref.frequencies.size:
                # Modes pair by position, slowest first, as in the spectrum.
                # GS keeps three Maxwell modes that VB leaves out, so no
                # pairing by position is right between those two.
                warnings.warn(
                    f"{result.member} has {mine.frequencies.size} rotational modes and {code} has "
                    f"{ref.frequencies.size}; the modes of the transfer function are not scored "
                    f"against {code}", stacklevel=2)
                continue
            modes = np.arange(1, mine.frequencies.size + 1, dtype=float)
            for name, values, ref_values in (("frequency", mine.frequencies, ref.frequencies),
                                             ("residue", mine.residues, ref.residues)):
                for part in ("real", "imag"):
                    m = difference_metrics(Curve(modes, getattr(values, part)),
                                           Curve(modes, getattr(ref_values, part)))
                    rows.append(_row("polar_motion_transfer_function", f"{name}_{part}",
                                     (1.0, float(modes[-1])), code, m, chandler_wobble=setting))
        # Love numbers, per degree.
        for quantity, love in sorted(result.love.items()):
            if quantity not in reference.love:
                continue
            ref = reference.love[quantity]
            degrees = _degree_rows(love.degrees, ref.degrees, _excluded_degrees(ref.attributes))
            if degrees.size == 0:
                continue
            interval = (float(degrees[0]), float(degrees[-1]))
            for part in ("elastic", "fluid"):
                candidate = _restricted(Curve(love.degrees, getattr(love, part)), degrees)
                reference_curve = _restricted(Curve(ref.degrees, getattr(ref, part)), degrees)
                m = difference_metrics(candidate, reference_curve, interval)
                rows.append(_row("love_numbers", f"{quantity}_{part}", interval, code, m))
        # Relaxation spectrum, per mode.
        if result.spectrum is not None and reference.spectrum is not None:
            mine, ref = result.spectrum, reference.spectrum
            if mine.s_per_kyr.shape[1] != ref.s_per_kyr.shape[1]:
                # Modes pair by position, s_1 with s_1 and so on, both slowest
                # first. With different mode counts no pairing is right, so the
                # spectrum gets no rows. The warning names both counts, because
                # a missing group of rows in a long score table is easy to miss.
                warnings.warn(
                    f"{result.member} has {mine.s_per_kyr.shape[1]} relaxation modes and "
                    f"{code} has {ref.s_per_kyr.shape[1]}; the relaxation spectrum is not "
                    f"scored against {code}", stacklevel=2)
            else:
                degrees = _degree_rows(mine.degrees, ref.degrees, _excluded_degrees(ref.attributes))
                if degrees.size:
                    interval = (float(degrees[0]), float(degrees[-1]))
                    for j in range(mine.s_per_kyr.shape[1]):
                        m = difference_metrics(
                            _restricted(Curve(mine.degrees, mine.s_per_kyr[:, j]), degrees),
                            _restricted(Curve(ref.degrees, ref.s_per_kyr[:, j]), degrees),
                            interval)
                        rows.append(_row("relaxation_spectrum", f"s_{j + 1}", interval, code, m))
    return rows
