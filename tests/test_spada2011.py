"""The Spada et al. (2011) reader and converter: its rules, and the numbers it writes.

The tests without a mark need only the resources in the package. The tests
marked `data` convert the archive unpacked in
data/original/spada2011/spada2011-cost-es0701/ into a temporary directory and
check the result against evidence the converter does not use: the fluid limit
of the normal mode expansion, the relaxation times file, the load mass that
TABOO prints in each header, and the agreement of two independent codes.

The converter lives in ``gia_mip.benchmarks.spada2011`` and follows the
pattern of ``gia_mip.benchmarks.martinec2018``:

``CASES``
    The short case names it converts, for example ``t02-01``.
``HANDLING``
    One decision per correction that can apply to a file it reads.
``applying(relative_path, sha256)``
    The correction ids for one file version; raises KeyError for a correction
    with no decision in HANDLING.
``convert(case, root, out_dir)``
    Writes ``out_dir/spada2011-<case>.nc`` and returns the path. It calls
    ``applying`` for every file it reads, so a correction without a decision
    stops the conversion.

Every value read from a source file names that file: a spatial curve in
``attributes["source_file"]``, and the objects returned by
``SpadaResult.love_numbers(q)`` and ``SpadaResult.relaxation_spectrum()`` in
a ``source_file`` field. Each is a path relative to ``data/original/``.

The converted files are read back with ``gia_mip.benchmarks.results.SpadaResult``,
so these tests also check that a reference contributor is read the way a
solver's result is.
"""

import fnmatch
import json
import os
import pathlib
import re

import numpy as np
import pytest

from gia_mip import cases, corrections, data

PREFIX = "spada2011/spada2011-cost-es0701"
RESOURCES = pathlib.Path(data.__file__).parent / "resources"

# The five cases whose data the result class is built for: relaxation spectra
# (1/1), loading and tidal Love numbers (2/1, 3/1), displacements (1/2) and
# their rates (2/2). The other five may or may not be converted, and say so
# in their case files either way.
REQUIRED_CASES = ("t01-01", "t01-02", "t01-03", "t02-01", "t02-02")


def _spada2011():
    """The converter module, imported when a test runs so collection never fails."""
    from gia_mip.benchmarks import spada2011
    return spada2011


def _current_sha(relative_path: str) -> str:
    """The SHA-256 the manifest records for one archive member."""
    return data.file_versions()[relative_path]["sha256"]


# ------------------------------------------------------------ no data needed

def test_the_required_cases_are_converted():
    assert set(REQUIRED_CASES) <= set(_spada2011().CASES)


def test_handling_names_only_real_corrections():
    """A stale entry in HANDLING hides a correction that was renamed."""
    for correction_id in _spada2011().HANDLING:
        corrections.by_id(correction_id)


def test_every_correction_on_a_file_with_a_layout_has_a_decision():
    """Every file the reader has a layout for is checked against HANDLING, by version."""
    spada2011 = _spada2011()
    layouts = json.loads((RESOURCES / "spada_column_layouts.json").read_text())
    assert layouts.get("files"), "spada_column_layouts.json has no `files` table"
    for path in layouts["files"]:
        spada2011.applying(path, _current_sha(path))


def test_every_spada_correction_is_pinned_to_file_versions():
    """The archive is frozen, and a correction names the versions it was found in."""
    for entry in corrections.load():
        if any(p.startswith("spada2011/") for p in entry["applies_to"]):
            assert entry.get("sha256"), entry["id"]


def test_every_spada_correction_pattern_matches_a_file():
    """A pattern naming a directory matches no file, so the correction never applies.

    `applying_to` matches member paths, for example
    spada2011/spada2011-cost-es0701/T02-01/disp_t0_cap.dat. A pattern written
    as the directory T02-01 matches none of them.
    """
    members = set(data.file_versions())
    for entry in corrections.load():
        for pattern in entry["applies_to"]:
            if not pattern.startswith("spada2011/"):
                continue
            assert any(fnmatch.fnmatch(m, pattern) for m in members), (entry["id"], pattern)


# Each file carries a defect found in the archive. The claim is only that the
# corrections table records one for it; which id and which wording is the
# implementer's decision.
KNOWN_DEFECTS = {
    # .ela/.flu/.res hold absolute values of the residues, for a log plot.
    "T01-02/h_L_T01-02_GS.ela": "absolute values",
    # Twelve residues for nine modes; positions 6, 9, 12 are Maxwell modes.
    "T01-02/h_L_T01-02_GS.dat": "twelve residues",
    # tau_j in years, not s_j in 1/kyr.
    "T01-01/spec_T01_01_GS.mod": "relaxation times in years",
    # Degrees 3 and 4 repeat the spectrum of degree 2 to every printed digit.
    "T01-01/spec_T01_01_ZM.dat": "degrees 3 and 4 copy degree 2",
    # Degrees 29 and 40 are written twice.
    "T01-01/spec_T01_01_RR.dat": "repeated degrees",
    # Residues carry the opposite sign to GS, RR and ZM; the fluid limit is
    # elastic + sum r/s in this file. Each record also wraps onto two lines.
    "T01-02/h_L_T01-02_VB.dat": "opposite residue sign",
    # '#' header lines and the last row, colatitude 180, written twice.
    "T02-01/disp_t0_cap.dat": "header and repeated last row",
    # Header says load mass 5.793e18 kg; the table-4 cap is 3.607e18 kg.
    "T02-01/disp_tinf_cap.dat": "a different load",
    # Epoch 3 kyr is not in the case, and the header mass is 3.875e18 kg.
    "T02-01/disp_t3_disc.dat": "an extra epoch with a different load",
    # Header says Lmax 512; the case says degrees 2 to 128.
    "T02/disp_t0_point.dat": "truncation degree 512",
    # VK stores rates in m/yr, GS in mm/yr; the script multiplies VK by 1000.
    "T02-02/output_sec2parab/udt2cost.dat": "rate unit",
}


@pytest.mark.parametrize("relative", sorted(KNOWN_DEFECTS))
def test_each_known_defect_is_recorded_as_a_correction(relative):
    path = f"{PREFIX}/{relative}"
    found = corrections.applying_to(path, _current_sha(path))
    assert found, f"no correction applies to {path} ({KNOWN_DEFECTS[relative]})"


def test_a_case_that_is_not_converted_records_every_contributor():
    """Where no reader exists for a case, the case file says why, per contributor.

    `reference_data.not_converted` maps a contributor code to the reason its
    data is not converted. A case outside CASES must account for all of them.
    """
    spada2011 = _spada2011()
    for case in cases.cases("spada2011"):
        short = case["id"].removeprefix("spada2011-")
        if short in spada2011.CASES:
            continue
        reasons = case["reference_data"].get("not_converted", {})
        for contributor in case["reference_data"]["contributors"]:
            assert str(reasons.get(contributor, "")).strip(), (case["id"], contributor)


# --------------------------------------------------------------- data needed

# The unpacked archive, under the same data root the converter uses:
# $GIA_MIP_DATA_ROOT if set, else data/ in the checkout (tests/conftest.py).
ARCHIVE_DIR = (pathlib.Path(os.environ["GIA_MIP_DATA_ROOT"]) if os.environ.get("GIA_MIP_DATA_ROOT")
               else pathlib.Path(__file__).resolve().parent.parent / "data") / "original" / PREFIX


def _numeric_rows(path: pathlib.Path, per_record: int) -> np.ndarray:
    """The numbers of a whitespace table as records of `per_record` values.

    Written here, independently of the reader, to give the tests their own
    view of the raw files. Lines starting with '#' or '>' (a GMT segment
    marker) are skipped, Fortran D exponents are read, and values are
    concatenated before being split into records, which handles VB's records
    that wrap onto a second line.
    """
    values: list[float] = []
    for line in path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped[0] in "#>":
            continue
        values += [float(v.replace("D", "E").replace("d", "e")) for v in stripped.split()]
    return np.array(values).reshape(-1, per_record)


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    """Convert the required cases into a temporary directory, once for the module."""
    root = _spada2011().data_root() / "original"
    if not (root / PREFIX / "T02-01").is_dir():
        pytest.skip("the Spada archive is not unpacked")
    out = tmp_path_factory.mktemp("spada2011")
    return {case: _spada2011().convert(case, root, out) for case in REQUIRED_CASES}


def _result(converted, case, member):
    from gia_mip.benchmarks.results import SpadaResult
    return SpadaResult.read(converted[case], member, case=case)


@pytest.mark.data
def test_the_converted_file_is_named_for_the_case(converted):
    for case, path in converted.items():
        assert pathlib.Path(path).name == f"spada2011-{case}.nc"


@pytest.mark.data
def test_every_value_is_finite_and_every_axis_ascends(converted):
    import netCDF4
    for case, path in converted.items():
        with netCDF4.Dataset(path) as ds:
            ds.set_auto_mask(False)
            stack = [ds]
            while stack:
                group = stack.pop()
                stack.extend(group.groups.values())
                for name, variable in group.variables.items():
                    values = np.asarray(variable[:], dtype=float)
                    assert np.all(np.isfinite(values)), (case, group.path, name)
                    if name in group.dimensions and values.size > 1:
                        assert np.all(np.diff(values) > 0), (case, group.path, name)


@pytest.mark.data
@pytest.mark.parametrize("case", REQUIRED_CASES)
def test_every_contributor_is_converted_or_recorded(converted, case):
    """Table 1 names contributors whose data is not in the archive; each needs a reason.

    Correction spada-t02-01-contributors-without-data: a contributor listed in
    the case file with no data must fail loudly, not be skipped.
    """
    from gia_mip.benchmarks.results import SpadaResult
    definition = cases.case(f"spada2011-{case}")
    members = set(SpadaResult.members(converted[case]))
    reasons = definition["reference_data"].get("not_converted", {})
    for contributor in definition["reference_data"]["contributors"]:
        assert contributor in members or str(reasons.get(contributor, "")).strip(), (
            case, contributor, sorted(members))


# Two files the converter reads, one per kind of case, for the stop test. The
# T02-01 file is a TABOO spatial profile; the T01-02 file is a Love number
# table. A converter that skips `applying` for either kind fails.
UNHANDLED_TARGETS = [("t02-01", "T02-01/disp_t0_cap.dat"),
                     ("t01-02", "T01-02/h_L_T01-02_GS.dat")]


@pytest.mark.data
@pytest.mark.parametrize("case,relative", UNHANDLED_TARGETS)
def test_a_correction_without_a_decision_stops_the_converter(monkeypatch, tmp_path, case, relative):
    """A defect added to the table and forgotten in the converter must stop `convert`.

    The table is the real one plus one fake correction on a file the case
    reads. The fake has no `sha256`, so it applies to every version of the
    file. HANDLING has no entry for it, so `convert` must raise KeyError naming
    the id. The real corrections stay in the table, so that nothing else in
    the conversion changes.
    """
    spada2011 = _spada2011()
    root = spada2011.data_root() / "original"
    if not (root / PREFIX / "T02-01").is_dir():
        pytest.skip("the Spada archive is not unpacked")
    path = f"{PREFIX}/{relative}"
    fake = {"id": "spada-test-only-unhandled", "kind": "trap", "applies_to": [path],
            "as_shipped": "", "correct": "", "evidence": "", "action": "",
            "status": "confirmed", "discovered": "2026-09-14"}
    real = list(corrections.load())
    monkeypatch.setattr(corrections, "load", lambda: real + [fake])
    with pytest.raises(KeyError, match="spada-test-only-unhandled"):
        spada2011.convert(case, root, tmp_path)


# Loads and quantities a converted spatial case can hold. A member holds a
# subset; `epochs` of a pair the member does not hold is empty or raises
# KeyError, and either means "not held".
SPATIAL_QUANTITIES = {"t02-01": ("U", "V", "N"), "t02-02": ("U_rate", "V_rate", "N_rate")}


def _held_epochs(result, load, quantity):
    """The epochs of one (load, quantity), or an empty list when the result holds none."""
    try:
        return list(result.epochs(load, quantity))
    except KeyError:
        return []


@pytest.mark.data
def test_every_file_the_converter_reads_has_a_layout(converted):
    """Every source file named in the converted data is a key of the `files` layout table.

    The source files are collected from the converted NetCDF, not from a list
    in this test: every spatial curve of every member, load, quantity and
    epoch, and the `source_file` field of every Love number table (t01-02,
    t01-03) and relaxation spectrum (t01-01). A file read without a decoded
    layout fails here. It would also be missing from the HANDLING check in
    test_every_correction_on_a_file_with_a_layout_has_a_decision, which loops
    over `files`.
    """
    from gia_mip.benchmarks.results import SpadaResult
    layouts = json.loads((RESOURCES / "spada_column_layouts.json").read_text())
    files = layouts.get("files", {})
    sources: dict[str, tuple] = {}
    for case, path in converted.items():
        for member in SpadaResult.members(path):
            result = SpadaResult.read(path, member, case=case)
            if case == "t01-01":
                sources[result.relaxation_spectrum().source_file] = (case, member, "spectrum")
            elif case in ("t01-02", "t01-03"):
                for quantity in "hlk":
                    sources[result.love_numbers(quantity).source_file] = (case, member, quantity)
            else:
                for load in ("cap", "disc", "point"):
                    for quantity in SPATIAL_QUANTITIES[case]:
                        for epoch in _held_epochs(result, load, quantity):
                            curve = result.spatial(load, quantity, epoch)
                            sources[curve.attributes["source_file"]] = (case, member, load, quantity, epoch)
    # A lower bound on the distinct files, so an empty or partial walk fails:
    # 4 spectra and 21 Love tables, and per spatial case 18 GS files (one per
    # load and epoch, each holding U, V and N or their rates) and 6 VK files
    # (one per quantity and load; a merged curve names one of its two grids).
    assert len(sources) >= 25 + 2 * (18 + 6), len(sources)
    missing = {source: where for source, where in sources.items() if source not in files}
    assert not missing, missing


@pytest.mark.data
@pytest.mark.parametrize("member", ["GS", "RR", "VB", "ZM"])
def test_the_spectrum_degrees_are_the_distinct_degrees_of_the_raw_file(converted, member):
    """One row per degree, ascending, and every degree the contributor printed.

    RR's spec_T01_01_RR.dat writes degrees 29 and 40 twice (257 rows, 255
    distinct degrees). A converter that copies the rows keeps the repeat and
    the degree axis does not strictly ascend. A converter that drops rows by
    another rule loses a degree. ZM prints degrees 2 to 5 only, VB 2 to 257.
    """
    spectrum = _result(converted, "t01-01", member).relaxation_spectrum()
    degrees = np.asarray(spectrum.degrees, dtype=float)
    assert np.all(np.diff(degrees) > 0), degrees
    raw = _numeric_rows(ARCHIVE_DIR / "T01-01" / f"spec_T01_01_{member}.dat", 10)
    np.testing.assert_array_equal(degrees, np.unique(raw[:, 0]))


# ----------------------------------------------------- test 1/1, spectra

@pytest.mark.data
def test_the_gs_spectrum_is_minus_one_over_the_relaxation_times(converted):
    """spec_T01_01_GS.mod holds tau_j in years; the converted s_j is in 1/kyr.

    s_j = -1000 / tau_j for every degree. The two files are printed with eight
    significant digits, which agree to 3e-8 here; the tolerance of 1e-6 allows
    that, and a unit slip gives a factor of 1000.
    """
    spectrum = _result(converted, "t01-01", "GS").relaxation_spectrum()
    tau = _numeric_rows(ARCHIVE_DIR / "T01-01" / "spec_T01_01_GS.mod", 2)
    for i, degree in enumerate(spectrum.degrees):
        expected = np.sort(-1000.0 / tau[tau[:, 0] == degree][:, 1])
        assert expected.size == 9, degree
        got = np.sort(np.asarray(spectrum.s_per_kyr)[i])
        np.testing.assert_allclose(got, expected, rtol=1e-6, err_msg=f"degree {degree}")


@pytest.mark.data
def test_the_slowest_degree_2_mode_is_27_thousand_years(converted):
    """s = -3.6215236e-5 per kyr, so tau = 27 612 kyr: the unit check in one number.

    The slowest mode is the largest s, the one closest to zero. In 1/yr the
    same number would read -3.6e-8, and as tau in years 2.76e7.
    """
    spectrum = _result(converted, "t01-01", "GS").relaxation_spectrum()
    row = np.asarray(spectrum.s_per_kyr)[list(spectrum.degrees).index(2)]
    assert np.max(row) == pytest.approx(-3.6215236e-5, rel=1e-6)


@pytest.mark.data
@pytest.mark.parametrize("member", ["RR", "VB", "ZM"])
def test_the_spectra_agree_mode_by_mode_at_degree_2(converted, member):
    """Four codes, one mode order in the converted file.

    ZM lists its modes fastest first and the others slowest first. After
    conversion, mode j of every member is the same mode, so the columns agree
    one to one. Printed digits limit the agreement to 2e-7 (ZM); the
    tolerance is 1e-5. A reversed order is wrong by a factor of 1e5.
    """
    gs = _result(converted, "t01-01", "GS").relaxation_spectrum()
    other = _result(converted, "t01-01", member).relaxation_spectrum()
    row_gs = np.asarray(gs.s_per_kyr)[list(gs.degrees).index(2)]
    row_other = np.asarray(other.s_per_kyr)[list(other.degrees).index(2)]
    np.testing.assert_allclose(row_other, row_gs, rtol=1e-5)


# ------------------------------------------------- tests 2/1 and 3/1, Love numbers

FLUID_LIMIT_CASES = [("t01-02", m, q) for m in ("GS", "RR", "VB", "ZM") for q in "hlk"] + \
                    [("t01-03", m, q) for m in ("GS", "RR", "VB") for q in "hlk"]


@pytest.mark.data
@pytest.mark.parametrize("case,member,quantity", FLUID_LIMIT_CASES)
def test_the_fluid_limit_is_elastic_minus_the_residues_over_the_frequencies(
        converted, case, member, quantity):
    """h_f = h_e - sum_j r_j / s_j, the t -> infinity limit of the normal mode sum.

    This ties the elastic, fluid and residue columns to the spectrum of the
    same code, in the convention of Spada et al. (2011), eq. 17. It fails for
    each trap: the absolute values in .ela/.flu/.res files, VB's opposite
    residue sign, a residue aligned with the wrong mode, and s in the wrong
    unit. The spectrum is the same code's test 1/1 result, because the
    relaxation frequencies depend only on the earth model.

    The tolerance is 1e-5 of max(1, |h_f|). Eight printed digits leave a
    measured misfit of at most 7e-7 of the value; every trap above gives a
    misfit of order one.
    """
    love = _result(converted, case, member).love_numbers(quantity)
    spectrum = _result(converted, "t01-01", member).relaxation_spectrum()
    s_by_degree = {int(n): np.asarray(spectrum.s_per_kyr)[i] for i, n in enumerate(spectrum.degrees)}
    checked = 0
    for i, degree in enumerate(love.degrees):
        if int(degree) not in s_by_degree:
            continue
        residues = np.asarray(love.residues)[i]
        s = s_by_degree[int(degree)]
        assert residues.shape == s.shape == (9,), (degree, residues.shape, s.shape)
        expected = love.elastic[i] - np.sum(residues / s)
        assert abs(love.fluid[i] - expected) <= 1e-5 * max(1.0, abs(love.fluid[i])), (
            degree, love.fluid[i], expected)
        checked += 1
    assert checked >= 2, (case, member, quantity)


def _raw_love_table(case: str, member: str, quantity: str) -> np.ndarray:
    """The records of one raw signed Love number file.

    GS prints degree, elastic, fluid and twelve residues (15 values). RR, VB
    and ZM print degree, elastic, fluid and nine residues (12 values); VB's
    records wrap onto a second line, which `_numeric_rows` joins.
    """
    kind = "L" if case == "t01-02" else "T"
    directory = case.upper()
    path = ARCHIVE_DIR / directory / f"{quantity}_{kind}_{directory}_{member}.dat"
    return _numeric_rows(path, 15 if member == "GS" else 12)


@pytest.mark.data
@pytest.mark.parametrize("case,member,quantity", FLUID_LIMIT_CASES)
def test_the_love_number_degrees_are_the_distinct_degrees_of_the_raw_file(
        converted, case, member, quantity):
    """One row per degree, ascending, equal to the degrees the contributor printed.

    RR's Love number files repeat degrees 29 and 40, as its spectrum does.
    """
    love = _result(converted, case, member).love_numbers(quantity)
    degrees = np.asarray(love.degrees, dtype=float)
    assert np.all(np.diff(degrees) > 0), degrees
    raw = _raw_love_table(case, member, quantity)
    np.testing.assert_array_equal(degrees, np.unique(raw[:, 0]))


@pytest.mark.data
@pytest.mark.parametrize("case,member,quantity", FLUID_LIMIT_CASES)
def test_the_elastic_and_fluid_love_numbers_are_the_printed_columns(
        converted, case, member, quantity):
    """Converted elastic and fluid equal columns 2 and 3 of the signed .dat file.

    The fluid limit identity uses converted values only, so a converter that
    computes the fluid value from the residues passes it. This test reads the
    printed columns. Every raw row of a degree is compared, which covers RR's
    repeated rows (their elastic and fluid columns are equal). The tolerance
    rtol 1e-7 allows a float32 round trip of eight printed digits and nothing
    more; the Love numbers are dimensionless, from 1e-5 to about 60 in size,
    and none printed is zero.
    """
    love = _result(converted, case, member).love_numbers(quantity)
    raw = _raw_love_table(case, member, quantity)
    degrees = [int(n) for n in love.degrees]
    for record in raw:
        i = degrees.index(int(record[0]))
        np.testing.assert_allclose([love.elastic[i], love.fluid[i]], record[1:3], rtol=1e-7,
                                   err_msg=f"{case} {member} {quantity} degree {int(record[0])}")


@pytest.mark.data
@pytest.mark.parametrize("case", ["t01-02", "t01-03"])
@pytest.mark.parametrize("quantity", ["h", "l", "k"])
def test_the_vb_residues_equal_the_gs_residues_mode_by_mode_at_degree_2(converted, case, quantity):
    """VB prints its residues with the opposite sign; converted, they are GS's.

    At degree 2 the raw VB h residue of the slowest mode is +3.0189252e-06 and
    the GS one is -3.0189264e-06. The converted file holds residues in the
    paper's convention, in one mode order for all members, so the VB and GS
    residues agree mode by mode. The measured misfit after negation is at most
    4.7e-7 of the value (eight printed digits); the tolerance is rtol 1e-5. A
    residue not negated is wrong by 200 per cent.
    """
    gs = _result(converted, case, "GS").love_numbers(quantity)
    vb = _result(converted, case, "VB").love_numbers(quantity)
    row_gs = np.asarray(gs.residues)[list(gs.degrees).index(2)]
    row_vb = np.asarray(vb.residues)[list(vb.degrees).index(2)]
    np.testing.assert_allclose(row_vb, row_gs, rtol=1e-5)


@pytest.mark.data
def test_the_maxwell_modes_are_dropped_by_position_at_degree_128(converted):
    """At degree 128 a real mode is as small as a Maxwell mode; position decides.

    The raw GS file holds twelve residues. Positions 6, 9 and 12 (from one) are
    the Maxwell modes. The converted file must hold the other nine, each with
    the relaxation frequency of its own mode, whatever order the converter
    writes the modes in. Dropping by magnitude removes a real mode at this
    degree (residue -1.4e-14) and keeps a Maxwell one.
    """
    raw_love = _numeric_rows(ARCHIVE_DIR / "T01-02" / "h_L_T01-02_GS.dat", 15)
    raw_spec = _numeric_rows(ARCHIVE_DIR / "T01-01" / "spec_T01_01_GS.dat", 10)
    row = raw_love[raw_love[:, 0] == 128][0]
    kept = np.delete(row[3:], [5, 8, 11])
    s_raw = raw_spec[raw_spec[:, 0] == 128][0, 1:]
    expected = sorted(zip(s_raw, kept))

    love = _result(converted, "t01-02", "GS").love_numbers("h")
    spectrum = _result(converted, "t01-01", "GS").relaxation_spectrum()
    i = list(love.degrees).index(128)
    j = list(spectrum.degrees).index(128)
    got = sorted(zip(np.asarray(spectrum.s_per_kyr)[j], np.asarray(love.residues)[i]))
    np.testing.assert_allclose(np.array(got), np.array(expected), rtol=1e-6, atol=1e-30)


@pytest.mark.data
def test_the_love_numbers_come_from_the_signed_file(converted):
    """l_e at degree 2 is -0.12822149; the .ela file stores its absolute value."""
    love = _result(converted, "t01-02", "GS").love_numbers("l")
    assert love.elastic[list(love.degrees).index(2)] == pytest.approx(-0.12822149, abs=1e-8)


# --------------------------------------------------------- test 1/2, displacements

GS_EPOCHS = [0.0, 1.0, 2.0, 5.0, 10.0, 1000.0]
VK_EPOCHS = [0.0, 1.0, 2.0, 5.0, 10.0]


@pytest.mark.data
def test_the_gs_cap_profile_has_721_points_at_a_quarter_degree(converted):
    """disp_t0_cap.dat has 722 rows; the last, colatitude 180, is a copy."""
    curve = _result(converted, "t02-01", "GS").spatial("cap", "U", 0.0)
    np.testing.assert_allclose(curve.axis, np.arange(721) * 0.25, atol=1e-9)


@pytest.mark.data
def test_the_gs_cap_uplift_at_the_pole_at_loading(converted):
    """U(0 deg, t = 0) = -27.77392 m, printed to seven decimals as -0.27773920E+02."""
    curve = _result(converted, "t02-01", "GS").spatial("cap", "U", 0.0)
    assert curve.values[0] == pytest.approx(-27.77392, abs=1e-6)


@pytest.mark.data
def test_the_gs_geoid_is_read_from_column_5(converted):
    """N at colatitude 180, t = 0, is 1.3567549 m in column 5; column 4 is zero there."""
    curve = _result(converted, "t02-01", "GS").spatial("cap", "N", 0.0)
    assert curve.values[-1] == pytest.approx(1.3567549, abs=1e-7)


@pytest.mark.data
@pytest.mark.parametrize("load", ["cap", "disc", "point"])
def test_the_gs_epochs_are_the_case_epochs_with_1000_kyr_for_the_end(converted, load):
    """0, 1, 2, 5, 10 and 1000 kyr, and nothing else.

    GS's last file is a run at 1000 kyr, not the fluid limit: U at the pole is
    -388.1 m there against -395.0 m in the limit, so it is stored at 1000 and
    never at infinity. disp_tinf_cap.dat and disp_t3_disc.dat are runs with a
    different load and must not appear.
    """
    epochs = _result(converted, "t02-01", "GS").epochs(load, "U")
    assert [float(e) for e in epochs] == GS_EPOCHS


@pytest.mark.data
@pytest.mark.parametrize("load", ["cap", "disc"])
def test_the_vk_epochs_are_the_five_columns(converted, load):
    epochs = _result(converted, "t02-01", "VK").epochs(load, "U")
    assert [float(e) for e in epochs] == VK_EPOCHS


def _header_mass(path: pathlib.Path) -> float:
    """The load mass TABOO prints: `# The mass of the load (kg) is =  3.607E+018`."""
    match = re.search(r"mass of the load \(kg\) is =\s*([-+0-9.Ee]+)", path.read_text())
    assert match, path
    return float(match.group(1))


@pytest.mark.data
@pytest.mark.parametrize("case,quantity", [("t02-01", "U"), ("t02-02", "U_rate")])
@pytest.mark.parametrize("load", ["cap", "disc", "point"])
def test_every_gs_file_converted_carries_the_load_of_the_case(converted, case, quantity, load):
    """The header mass of each source file equals the mass of the table-4 load.

    Every epoch of GS_EPOCHS must be present first.
    SpadaCase computes the mass from loads/spada2011.json: 3.607171340900778e18 kg
    for the cap and the disc, 1e18 kg for the point. TABOO prints sixteen
    digits, and the two agree to rounding, so the tolerance is 1e-9. The files
    of a different run (5.79e18 kg, 3.87e18 kg) differ by 7 to 60 per cent.
    """
    from gia_mip.benchmarks.inputs import SpadaCase
    result = _result(converted, case, "GS")
    expected = SpadaCase(case).load_mass(load)
    root = _spada2011().data_root() / "original"
    epochs = result.epochs(load, quantity)
    # An empty epoch list would pass the loop below without checking a file.
    # GS ships all six epochs for every load in both cases, including
    # T02-02/rate_t1000_cap.dat, rate_t1000_disc.dat and T02/rate_tinf_point.dat,
    # whose header says Time = 1000 kyr.
    assert [float(e) for e in epochs] == GS_EPOCHS, (case, load, quantity, epochs)
    for epoch in epochs:
        source = result.spatial(load, quantity, epoch).attributes["source_file"]
        assert _header_mass(root / source) == pytest.approx(expected, rel=1e-9), source


@pytest.mark.data
def test_the_vk_profile_joins_both_grids_and_omits_the_poles(converted):
    """VILMA: 0.25 to 15 degrees at 0.25 on the fine grid, 1 to 179 at 1 on the global.

    The two grids hold identical values where they overlap, so one curve holds
    both, with 60 + 164 = 224 points and neither pole.
    """
    axis = _result(converted, "t02-01", "VK").spatial("cap", "U", 0.0).axis
    expected = np.union1d(np.arange(1, 61) * 0.25, np.arange(1.0, 180.0))
    np.testing.assert_allclose(axis, expected, atol=1e-9)


@pytest.mark.data
@pytest.mark.parametrize("case,quantity", [("t02-01", q) for q in ("U", "V", "N")]
                         + [("t02-02", q) for q in ("U_rate", "V_rate", "N_rate")])
@pytest.mark.parametrize("load", ["cap", "disc"])
def test_gs_and_vk_agree_within_the_published_spread(converted, case, quantity, load):
    """Two independent codes, one sign and one unit, at every common epoch.

    TABOO (GS) and VILMA (VK) differ by up to 1.7 m in U near the cap edge at
    10 kyr, which is 0.6 per cent of the peak, and by at most 6 per cent of
    the peak in the smallest field (V rate at 10 kyr). The test allows 10 per
    cent of the GS peak. A sign error gives 200 per cent. The rate files
    differ by a factor of 1000 in unit, which gives 100 per cent or more. GS is
    on a 0.25 degree grid that contains every VK point, so the comparison
    interpolates nothing.
    """
    gs = _result(converted, case, "GS")
    vk = _result(converted, case, "VK")
    for epoch in VK_EPOCHS:
        a = gs.spatial(load, quantity, epoch)
        b = vk.spatial(load, quantity, epoch)
        difference = np.interp(b.axis, a.axis, a.values) - b.values
        assert np.max(np.abs(difference)) < 0.1 * np.max(np.abs(a.values)), (
            load, quantity, epoch, np.max(np.abs(difference)), np.max(np.abs(a.values)))


@pytest.mark.data
def test_rates_are_stored_in_metres_per_kyr(converted):
    """dU/dt at the pole at loading is -77.948066 in the GS file, in mm/yr = m/kyr."""
    curve = _result(converted, "t02-02", "GS").spatial("cap", "U_rate", 0.0)
    assert curve.attributes["units"] == "m/kyr"
    assert curve.values[0] == pytest.approx(-77.948066, abs=1e-6)


@pytest.mark.data
def test_every_spatial_curve_names_the_version_of_its_source(converted):
    """source_file and source_sha256 let a curve be traced to one file version."""
    root = _spada2011().data_root() / "original"
    result = _result(converted, "t02-01", "VK")
    curve = result.spatial("disc", "N", 5.0)
    source = curve.attributes["source_file"]
    assert source.startswith(f"{PREFIX}/T02-01/output_sec2disc/")
    assert curve.attributes["source_sha256"] == data.sha256(root / source)


@pytest.mark.data
def test_the_point_load_is_on_the_one_degree_grid(converted):
    """T02/disp_t*_point.dat: 182 rows at 1 degree, the last a copy of colatitude 180."""
    curve = _result(converted, "t02-01", "GS").spatial("point", "U", 0.0)
    np.testing.assert_allclose(curve.axis, np.arange(181.0), atol=1e-9)
    assert curve.values[0] == pytest.approx(-973.28409, abs=1e-5)
