"""The manifest and the provenance record agree, and say what the pages need."""

import json
import pathlib

import pytest

from gia_mip import data

PROVENANCE = pathlib.Path(data.__file__).parent / "resources" / "provenance.json"
PERMISSIONS = {"granted", "requested", "not_asked"}

# Fields the documentation pages index without checking first, so a missing one
# is a broken page and not a caught error.
REQUIRED = {"collection", "stage", "description", "source_citation",
            "supplier", "received", "permission", "bytes", "sha256"}


@pytest.fixture(scope="module")
def manifest():
    return data.load_manifest()


@pytest.fixture(scope="module")
def provenance():
    return json.loads(PROVENANCE.read_text())


def test_manifest_and_provenance_cover_the_same_artifacts(manifest, provenance):
    assert set(manifest["artifacts"]) == set(provenance["artifacts"])


def test_every_entry_has_the_fields_the_pages_use(manifest):
    for rel, entry in manifest["artifacts"].items():
        missing = REQUIRED - set(entry)
        assert not missing, f"{rel}: missing {sorted(missing)}"


def test_permissions_are_from_the_vocabulary(manifest):
    for rel, entry in manifest["artifacts"].items():
        assert entry["permission"] in PERMISSIONS, f"{rel}: {entry['permission']}"


def test_stages_are_from_the_vocabulary(manifest):
    for rel, entry in manifest["artifacts"].items():
        assert entry["stage"] in data.STAGES, f"{rel}: {entry['stage']}"


def test_sums_look_like_sha256(manifest):
    for rel, entry in manifest["artifacts"].items():
        assert len(entry["sha256"]) == 64, rel
        int(entry["sha256"], 16)


def test_an_original_that_we_changed_says_so(manifest):
    """`original` means a third-party input, and two of them we repacked.

    A reader who wants to check our copy against the supplier's needs to know
    which ones cannot be compared byte for byte, and why.
    """
    for rel, entry in manifest["artifacts"].items():
        if entry["stage"] != "original":
            continue
        if "modified_from_supplied" in entry:
            assert entry["modified_from_supplied"].strip(), rel


def test_the_two_repacked_artifacts_are_flagged(manifest):
    """Named explicitly, so removing the flag from either is a failure."""
    for rel in ("spada2011/spada2011-cost-es0701.tar.gz",
                "giamip/giamip-protocol-v2.4.md"):
        assert "modified_from_supplied" in manifest["artifacts"][rel], rel


def test_archive_members_are_inventoried(manifest):
    for rel, entry in manifest["artifacts"].items():
        if rel.endswith((".zip", ".tar.gz")):
            assert entry.get("member_count", 0) > 0, rel
            assert len(entry["members"]) == entry["member_count"], rel


def test_member_names_map_to_a_path_on_disk(manifest):
    """Every member has a defined landing place under the unpack target.

    The inventory exists so that one file out of an unpacked archive can be
    verified later. That needs a single rule from member name to path, which
    `member_path` is.
    """
    for rel, entry in manifest["artifacts"].items():
        members = entry.get("members")
        if not members:
            continue
        top = data.common_top([m["name"] for m in members])
        paths = {data.member_path(m["name"], top) for m in members}
        assert "" not in paths, f"{rel}: a member maps to an empty path"
        assert len(paths) == len(members), f"{rel}: two members share a path"


def test_a_changed_file_keeps_the_sum_of_its_earlier_version():
    """build_manifest.carry_history records the version that is replaced."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_manifest", pathlib.Path(__file__).resolve().parent.parent / "scripts" / "build_manifest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    old = {"generated": "2026-09-11", "artifacts": {
        "giamip/t.zip": {"sha256": "a" * 64, "bytes": 1, "members": [{"name": "T/x.nc", "sha256": "b" * 64, "bytes": 1}]}}}
    new = {"giamip/t.zip": {"sha256": "c" * 64, "bytes": 2, "members": [{"name": "T/x.nc", "sha256": "d" * 64, "bytes": 2}]}}
    module.carry_history(old, new)
    assert new["giamip/t.zip"]["previous_versions"] == [{
        "sha256": "a" * 64, "bytes": 1, "recorded_in_manifest_of": "2026-09-11",
        "members": [{"name": "T/x.nc", "sha256": "b" * 64}]}]
    versions = data.file_versions({"artifacts": new})
    assert versions["giamip/t/x.nc"] == {"sha256": "d" * 64, "previous": ["b" * 64]}
    # A second rebuild with no change carries the history forward unchanged.
    again = {"giamip/t.zip": dict(new["giamip/t.zip"])}
    again["giamip/t.zip"].pop("previous_versions")
    module.carry_history({"generated": "2026-09-14", "artifacts": new}, again)
    assert again["giamip/t.zip"]["previous_versions"] == new["giamip/t.zip"]["previous_versions"]
