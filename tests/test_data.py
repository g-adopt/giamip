"""The fetcher: selection, archive handling, and the permission gate.

The archive tests carry most of the weight. Unpacking replaces a directory,
and an earlier version of it deleted the existing tree before extracting, so a
corrupt download destroyed the only copy of the data and then raised. These
tests pin the three properties that stop that happening again: a failed
extraction leaves the previous tree alone, a second run changes nothing, and
a member can never be written outside the target.
"""

import io
import json
import pathlib
import tarfile
import zipfile

import pytest

from giamip import data


# ---------------------------------------------------------------- selection

def test_expand_resolves_the_alias():
    assert data.expand(["benchmarks"]) == {"spada2011", "martinec2018"}
    assert data.expand(["giamip"]) == {"giamip"}
    assert data.expand(["benchmarks", "giamip"]) == {
        "spada2011", "martinec2018", "giamip"}


def test_select_rejects_an_unknown_collection():
    manifest = data.load_manifest()
    with pytest.raises(SystemExit, match="unknown collection"):
        data.select(manifest, {"nosuch"}, {"original"})


def test_select_rejects_an_unknown_stage():
    manifest = data.load_manifest()
    with pytest.raises(SystemExit, match="unknown stage"):
        data.select(manifest, None, {"nosuch"})


def test_select_allows_a_stage_with_no_files_yet():
    """A known stage with no files gives an empty result, not an error.

    The real manifest has files in both stages, so a synthetic one with only
    originals stands in for a stage that is still empty.
    """
    manifest = {"artifacts": {"x/a.dat": {"collection": "giamip", "stage": "original",
                                          "permission": "granted"}}}
    assert data.select(manifest, None, {"converted"}) == {}


def test_select_filters_by_collection():
    manifest = data.load_manifest()
    chosen = data.select(manifest, {"giamip"}, {"original"})
    assert chosen
    assert all(e["collection"] == "giamip" for e in chosen.values())


# ------------------------------------------------------------ member naming

@pytest.mark.parametrize("names,expected", [
    (["Ice History/a.nc", "Ice History/b.nc"], "Ice History"),
    (["l721j256/a.dat", "l721j256/b.dat"], "l721j256"),
    (["./T01-01/x", "./T01-02/y"], None),        # several top-level directories
    (["a.nc", "b.nc"], None),                     # members at the archive root
    (["foo.nc"], None),                           # one file, and it is not a directory
])
def test_common_top(names, expected):
    assert data.common_top(names) == expected


def test_member_path_is_the_documented_mapping():
    assert data.member_path("Ice History/a.nc", "Ice History") == "a.nc"
    assert data.member_path("./T01-01/x", None) == "T01-01/x"
    assert data.member_path("foo.nc", None) == "foo.nc"


@pytest.mark.parametrize("name", [
    "../../evil.txt",        # the case a lstrip-based check lets through
    "a/../../evil.txt",
    "/etc/passwd",
])
def test_check_member_names_rejects_an_escape(name):
    with pytest.raises(ValueError, match="escapes"):
        data.check_member_names([name])


def test_check_member_names_accepts_ordinary_names():
    data.check_member_names(["./T01-01/x", "Ice History/a.nc", "a/b/c.dat"])


# ----------------------------------------------------------------- unpacking

def make_zip(path: pathlib.Path, members: dict[str, bytes]) -> list[str]:
    with zipfile.ZipFile(path, "w") as z:
        for name, payload in members.items():
            z.writestr(name, payload)
    return list(members)


def make_tar(path: pathlib.Path, members: dict[str, bytes]) -> list[str]:
    with tarfile.open(path, "w:gz") as t:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            t.addfile(info, io.BytesIO(payload))
    return list(members)


def test_unpack_strips_one_top_level_directory(tmp_path):
    archive = tmp_path / "thing.zip"
    names = make_zip(archive, {"Top/a.nc": b"a", "Top/sub/b.nc": b"b"})
    target = data.unpack(archive, names, "deadbeef")
    assert (target / "a.nc").read_bytes() == b"a"
    assert (target / "sub" / "b.nc").read_bytes() == b"b"
    assert not (target / "Top").exists()


def test_unpack_keeps_the_layout_when_there_is_no_single_top(tmp_path):
    archive = tmp_path / "thing.tar.gz"
    names = make_tar(archive, {"./T01/a": b"a", "./T02/b": b"b"})
    target = data.unpack(archive, names, "deadbeef")
    assert (target / "T01" / "a").read_bytes() == b"a"
    assert (target / "T02" / "b").read_bytes() == b"b"


def test_unpack_handles_a_single_file_at_the_archive_root(tmp_path):
    """An earlier version treated the file's own name as a directory to strip."""
    archive = tmp_path / "thing.zip"
    names = make_zip(archive, {"only.nc": b"x"})
    target = data.unpack(archive, names, "deadbeef")
    assert (target / "only.nc").read_bytes() == b"x"


def test_unpack_writes_a_stamp_and_is_then_recognised_as_current(tmp_path):
    archive = tmp_path / "thing.zip"
    names = make_zip(archive, {"Top/a.nc": b"a"})
    assert not data.is_unpacked(archive, "abc123")
    data.unpack(archive, names, "abc123")
    assert data.is_unpacked(archive, "abc123")
    assert not data.is_unpacked(archive, "a-different-sum")


def test_unpack_replaces_the_tree_and_a_second_unpack_is_identical(tmp_path):
    archive = tmp_path / "thing.zip"
    names = make_zip(archive, {"Top/a.nc": b"a"})
    target = data.unpack(archive, names, "abc123")
    (target / "a.nc").write_bytes(b"tampered")
    data.unpack(archive, names, "abc123")
    assert (target / "a.nc").read_bytes() == b"a"


def test_a_corrupt_archive_leaves_the_previous_tree_intact(tmp_path):
    """The property that matters most: the old tree may be the only copy."""
    archive = tmp_path / "thing.zip"
    names = make_zip(archive, {"Top/a.nc": b"a" * 4096, "Top/b.nc": b"b" * 4096})
    target = data.unpack(archive, names, "abc123")
    before = sorted(p.name for p in target.iterdir())

    raw = bytearray(archive.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    archive.write_bytes(raw)

    with pytest.raises((zipfile.BadZipFile, OSError, ValueError)):
        data.unpack(archive, names, "def456")

    assert sorted(p.name for p in target.iterdir()) == before
    assert (target / "a.nc").read_bytes() == b"a" * 4096


def test_unpack_refuses_an_escaping_member_before_touching_the_target(tmp_path):
    archive = tmp_path / "thing.zip"
    make_zip(archive, {"ok.nc": b"x"})
    target = data.unpack_target(archive)
    target.mkdir()
    (target / "precious.nc").write_bytes(b"keep me")
    with pytest.raises(ValueError, match="escapes"):
        data.unpack(archive, ["../../evil.txt"], "abc123")
    assert (target / "precious.nc").read_bytes() == b"keep me"


# ------------------------------------------------------------ the permission gate

def test_the_permission_gate_blocks_a_download(tmp_path, monkeypatch, capsys):
    """Nothing whose supplier has not agreed is fetched, whatever the base URL."""
    manifest = {
        "base_url": "https://example.invalid/giamip/",
        "artifacts": {
            "x/blocked.dat": {
                "collection": "giamip", "stage": "original", "bytes": 3,
                "sha256": "0" * 64, "permission": "requested",
            },
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(data, "load_manifest", lambda *a, **k: manifest)
    # A base URL an attacker might set cannot get past the gate, because the
    # gate is checked before the URL is used at all.
    monkeypatch.setenv("GIAMIP_BASE_URL", "file:///etc/")
    code = data.main(["--dest", str(tmp_path), "--collection", "giamip"])
    assert code == 1
    assert "SKIPPED" in capsys.readouterr().err
    assert not (tmp_path / "original" / "x" / "blocked.dat").exists()


def test_a_granted_artifact_is_fetched_from_the_base_url(tmp_path, monkeypatch):
    """The gate lets `granted` through, checked against a local file:// base."""
    import hashlib

    payload = b"reference data\n"
    served = tmp_path / "served" / "x"
    served.mkdir(parents=True)
    (served / "ok.dat").write_bytes(payload)

    manifest = {
        "base_url": (tmp_path / "served").as_uri() + "/",
        "artifacts": {
            "x/ok.dat": {
                "collection": "giamip", "stage": "converted",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "permission": "granted",
            },
        },
    }
    monkeypatch.setattr(data, "load_manifest", lambda *a, **k: manifest)
    dest = tmp_path / "dest"
    code = data.main(["--dest", str(dest), "--collection", "giamip",
                      "--stage", "converted"])
    assert code == 0
    assert (dest / "converted" / "x" / "ok.dat").read_bytes() == payload
