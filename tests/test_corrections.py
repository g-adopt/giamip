"""The corrections table is well formed, and the rules it names actually work."""

import json

import pytest

from gia_mip import corrections

REQUIRED = {"id", "kind", "applies_to", "as_shipped", "correct",
            "evidence", "action", "status", "discovered"}


def test_every_entry_is_complete():
    for entry in corrections.load():
        missing = REQUIRED - set(entry)
        assert not missing, f"{entry.get('id')}: missing {sorted(missing)}"


def test_ids_are_unique():
    ids = [e["id"] for e in corrections.load()]
    assert len(ids) == len(set(ids))


def test_kinds_and_statuses_are_from_the_vocabulary():
    for entry in corrections.load():
        assert entry["kind"] in corrections.KINDS, entry["id"]
        assert entry["status"] in corrections.STATUSES, entry["id"]


def test_by_id_raises_for_an_unknown_id():
    with pytest.raises(KeyError, match="Known ids"):
        corrections.by_id("no-such-correction")


def test_applying_to_matches_the_ice_files():
    found = corrections.applying_to("giamip/ice-history/iceHistory-ICE-6G_D.nc")
    ids = {e["id"] for e in found}
    assert "ice-fillvalue-zero" in ids
    assert "ice-year-longname" in ids
    # The topography corrections must not match an ice file.
    assert "topography-missing-crs" not in ids


def test_applying_to_matches_one_topography_file_only():
    """The source-attribute defect is in the BedMachine Antarctica file alone."""
    bad = corrections.applying_to(
        "giamip/initial-topography/bedtopo_BedMach3ANT_BedMach5GRL_GEBCO_GLQ256.nc")
    good = corrections.applying_to(
        "giamip/initial-topography/bedtopo_Bedmap2ANT_BedMach5GRL_GEBCO_GLQ256.nc")
    assert "topography-source-attribute" in {e["id"] for e in bad}
    assert "topography-source-attribute" not in {e["id"] for e in good}


def test_fortran_exponents_rewrites_a_real_line():
    line = "   0.04395   0.04395     -0.3024125086287D+03      0.2883565323878D+00"
    fixed = corrections.fortran_exponents(line)
    values = [float(v) for v in fixed.split()]
    assert values[2] == pytest.approx(-302.4125086287)
    assert values[3] == pytest.approx(0.2883565323878)


def test_fortran_exponents_leaves_words_alone():
    """A D inside a header word is not an exponent."""
    assert corrections.fortran_exponents("# ICE-6G_D column") == "# ICE-6G_D column"
