"""The corrections table is well formed, and the rules it names actually work."""

import json

import pytest

from gia_mip import corrections, data

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


def _current(path):
    return data.file_versions()[path]["sha256"]


def test_applying_to_matches_the_ice_files():
    path = "giamip/ice-history/iceHistory-ICE-6G_D.nc"
    found = corrections.applying_to(path, _current(path))
    ids = {e["id"] for e in found}
    assert "ice-fillvalue-zero" in ids
    assert "ice-year-longname" in ids
    # The topography corrections must not match an ice file.
    assert "topography-missing-crs" not in ids


def test_applying_to_matches_one_topography_file_only():
    """The source-attribute defect is in the BedMachine Antarctica file alone."""
    bad_path = "giamip/initial-topography/bedtopo_BedMach3ANT_BedMach5GRL_GEBCO_GLQ256.nc"
    good_path = "giamip/initial-topography/bedtopo_Bedmap2ANT_BedMach5GRL_GEBCO_GLQ256.nc"
    bad = corrections.applying_to(bad_path, _current(bad_path))
    good = corrections.applying_to(good_path, _current(good_path))
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


# ------------------------------------------------ files that change under one name

def test_a_pinned_correction_needs_the_file_version():
    """Without the sum the answer is unknown, so the reader must supply it."""
    with pytest.raises(ValueError, match="sha256"):
        corrections.applying_to("giamip/ice-history/iceHistory-ICE-6G_D.nc")


def test_a_pinned_correction_does_not_apply_to_another_version():
    """A repaired file with the same name must not be corrected again."""
    path = "giamip/initial-topography/bedtopo_BedMach3ANT_BedMach5GRL_GEBCO_GLQ256.nc"
    found = corrections.applying_to(path, "0" * 64)
    assert "topography-source-attribute" not in {e["id"] for e in found}
    assert "topography-missing-crs" not in {e["id"] for e in found}


def test_every_pin_names_a_version_the_manifest_knows():
    """A mistyped sum would make a correction silently never apply."""
    known = set()
    for v in data.file_versions().values():
        known.add(v["sha256"])
        known.update(v["previous"])
    for entry in corrections.load():
        for digest in entry.get("sha256", []) + entry.get("checked_clean_sha256", []):
            assert digest in known, (entry["id"], digest)


def test_every_current_file_version_was_checked_against_its_corrections():
    """Fails when a supplier replaces a file, until someone has looked at it.

    For each pinned correction and each file it matches, the version now in
    the manifest must be listed either as having the defect (``sha256``) or
    as checked and free of it (``checked_clean_sha256``).
    """
    import fnmatch
    for entry in corrections.load():
        if "sha256" not in entry:
            continue
        reviewed = set(entry["sha256"]) | set(entry.get("checked_clean_sha256", []))
        for path, version in data.file_versions().items():
            if any(fnmatch.fnmatch(path, p) for p in entry["applies_to"]):
                assert version["sha256"] in reviewed, (
                    f"{path} has a new version {version['sha256'][:16]}. Check whether "
                    f"correction {entry['id']} still holds and add the sum to "
                    f"`sha256` or `checked_clean_sha256`.")


def test_content_corrections_on_giamip_files_are_pinned():
    """The GIAMIP inputs are not frozen, so a statement about content needs a version."""
    for entry in corrections.load():
        if entry["kind"] in ("metadata", "naming", "truncated") and any(
                p.startswith("giamip/") for p in entry["applies_to"]):
            assert "sha256" in entry, entry["id"]
