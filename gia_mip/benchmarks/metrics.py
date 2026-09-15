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
    spectrum is compared mode by mode over the common degrees. Residues are not
    scored, because their size spans twenty orders of magnitude over the
    degrees and a difference in one number says little.

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
        One row per comparison: ``kind`` (``spatial``, ``love_numbers`` or
        ``relaxation_spectrum``), ``load`` (None unless spatial), ``quantity``
        (for Love numbers ``h_elastic`` and so on, for the spectrum ``s_1``
        and so on), ``epoch_kyr`` (None unless spatial), ``interval``,
        ``reference`` (the member code), and the entries of
        `difference_metrics`.
    """
    rows: list[dict] = []
    intervals = _spatial_intervals(result.case) if result.fields else []
    for code, reference in references.items():
        # Surface fields, per load, quantity and epoch.
        for (load, quantity, epoch), curve in sorted(result.fields.items()):
            if (load, quantity, epoch) not in reference.fields:
                continue
            ref = reference.fields[(load, quantity, epoch)]
            for interval in intervals:
                m = difference_metrics(curve, ref, interval)
                rows.append({"kind": "spatial", "load": load, "quantity": quantity,
                             "epoch_kyr": epoch, "interval": interval,
                             "reference": code, **m})
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
                rows.append({"kind": "love_numbers", "load": None,
                             "quantity": f"{quantity}_{part}", "epoch_kyr": None,
                             "interval": interval, "reference": code, **m})
        # Relaxation spectrum, per mode.
        if result.spectrum is not None and reference.spectrum is not None:
            mine, ref = result.spectrum, reference.spectrum
            if mine.s_per_kyr.shape[1] == ref.s_per_kyr.shape[1]:
                degrees = _degree_rows(mine.degrees, ref.degrees, _excluded_degrees(ref.attributes))
                if degrees.size:
                    interval = (float(degrees[0]), float(degrees[-1]))
                    for j in range(mine.s_per_kyr.shape[1]):
                        m = difference_metrics(
                            _restricted(Curve(mine.degrees, mine.s_per_kyr[:, j]), degrees),
                            _restricted(Curve(ref.degrees, ref.s_per_kyr[:, j]), degrees),
                            interval)
                        rows.append({"kind": "relaxation_spectrum", "load": None,
                                     "quantity": f"s_{j + 1}", "epoch_kyr": None,
                                     "interval": interval, "reference": code, **m})
    return rows
