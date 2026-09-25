"""Where `giamip.data.locate` finds a data file, and when it downloads one.

All offline. A temporary directory stands in for the object store and is
reached through a ``file://`` base URL (``GIAMIP_BASE_URL``), which
``urllib`` opens like an https one. A synthetic manifest replaces the real
one, except in the tests that are about the real checkout.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import pathlib
import subprocess
import sys
import textwrap
import zipfile

import pytest

from giamip import data

REPOSITORY = pathlib.Path(data.__file__).resolve().parent.parent


def _sum(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _converted_entry(key: str, payload: bytes, **extra) -> dict:
    """A manifest entry of a converted file, with its content-addressed object key."""
    digest = _sum(payload)
    return {"collection": key.split("/")[0], "stage": "converted", "bytes": len(payload),
            "sha256": digest, "object": f"versions/{digest}/{key}", "permission": "granted",
            **extra}


def _original_entry(key: str, payload: bytes, **extra) -> dict:
    return {"collection": key.split("/")[0], "stage": "original", "bytes": len(payload),
            "sha256": _sum(payload), "permission": "granted", **extra}


def _zip(members: dict[str, bytes]) -> bytes:
    """A zip archive in memory, members in order."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        for name, payload in members.items():
            z.writestr(name, payload)
    return buffer.getvalue()


def _serve(store: pathlib.Path, key: str, entry: dict, payload: bytes) -> None:
    """Put an artifact on the fake store at the object key the manifest names."""
    path = store / entry.get("object", key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """No checkout, an empty download root, a fake store, and a manifest to fill.

    Returns a dict with the ``store`` and ``root`` directories and the
    ``manifest`` that `data.load_manifest` returns.
    """
    store, root = tmp_path / "store", tmp_path / "root"
    store.mkdir()
    manifest = {"base_url": "https://unused.invalid/", "artifacts": {}}
    monkeypatch.setattr(data, "load_manifest", lambda *a, **k: manifest)
    monkeypatch.setattr(data, "checkout_root", lambda: None)
    monkeypatch.setenv("GIAMIP_DATA_ROOT", str(root))
    monkeypatch.setenv("GIAMIP_BASE_URL", store.as_uri() + "/")
    data._VERIFIED.clear()
    return {"store": store, "root": root, "manifest": manifest}


def _count_fetches(monkeypatch) -> list:
    """Wrap `data.fetch` so that every call is recorded."""
    calls = []
    real = data.fetch

    def recording(url, dest):
        calls.append(url)
        return real(url, dest)

    monkeypatch.setattr(data, "fetch", recording)
    return calls


# ----------------------------------------------------------- the checkout

def test_a_tracked_converted_file_is_used_and_not_downloaded(monkeypatch):
    """In this checkout the tracked file matches the manifest, so no fetch happens."""
    def refuse(url, dest):
        raise AssertionError(f"fetch called for {url}")

    monkeypatch.setattr(data, "fetch", refuse)
    path = data.locate("martinec2018/martinec2018-B.nc", "converted")
    assert path == REPOSITORY / "data" / "converted" / "martinec2018" / "martinec2018-B.nc"


def test_a_corrupt_tracked_converted_file_raises_and_names_the_fix(tmp_path, monkeypatch):
    """A development state of the checkout is reported, never overwritten."""
    checkout = tmp_path / "checkout"
    (checkout / "data" / "converted" / "x").mkdir(parents=True)
    (checkout / "pyproject.toml").write_text("")
    (checkout / "data" / "converted" / "x" / "f.nc").write_bytes(b"edited")
    manifest = {"base_url": "https://unused.invalid/",
                "artifacts": {"x/f.nc": _converted_entry("x/f.nc", b"published")}}
    monkeypatch.setattr(data, "load_manifest", lambda *a, **k: manifest)
    monkeypatch.setattr(data, "ROOT", checkout)
    monkeypatch.delenv("GIAMIP_DATA_ROOT", raising=False)
    with pytest.raises(ValueError, match="build_manifest"):
        data.locate("x/f.nc", "converted")
    assert (checkout / "data" / "converted" / "x" / "f.nc").read_bytes() == b"edited"


def test_main_in_a_checkout_reports_a_tracked_converted_file_and_fetches_nothing(monkeypatch, capsys):
    calls = _count_fetches(monkeypatch)
    monkeypatch.delenv("GIAMIP_DATA_ROOT", raising=False)
    assert data.main(["--stage", "converted", "--collection", "martinec2018"]) == 0
    assert calls == []
    assert "ok       martinec2018/martinec2018-B.nc" in capsys.readouterr().out


# -------------------------------------------------------- the download root

def test_a_missing_converted_file_is_downloaded_to_its_versioned_path(isolated, monkeypatch):
    payload = b"converted v1"
    entry = _converted_entry("x/f.nc", payload)
    isolated["manifest"]["artifacts"]["x/f.nc"] = entry
    _serve(isolated["store"], "x/f.nc", entry, payload)
    path = data.locate("x/f.nc", "converted")
    assert path == isolated["root"] / "versions" / entry["sha256"] / "x" / "f.nc"
    assert path.read_bytes() == payload


def test_two_versions_of_one_converted_file_live_at_two_paths(isolated):
    """A cache shared by two releases keeps both versions."""
    paths = []
    for payload in (b"version one", b"version two"):
        entry = _converted_entry("x/f.nc", payload)
        isolated["manifest"]["artifacts"]["x/f.nc"] = entry
        _serve(isolated["store"], "x/f.nc", entry, payload)
        paths.append(data.locate("x/f.nc", "converted"))
    assert paths[0] != paths[1]
    assert paths[0].read_bytes() == b"version one"
    assert paths[1].read_bytes() == b"version two"


def test_a_corrupt_cached_converted_file_is_downloaded_again(isolated):
    """Its path names its sum, so a wrong sum means a damaged copy."""
    payload = b"converted"
    entry = _converted_entry("x/f.nc", payload)
    isolated["manifest"]["artifacts"]["x/f.nc"] = entry
    _serve(isolated["store"], "x/f.nc", entry, payload)
    path = data.cache_path(isolated["root"], "x/f.nc", entry)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"damaged")
    assert data.locate("x/f.nc", "converted").read_bytes() == payload


def test_an_original_with_a_wrong_sum_is_not_replaced_and_the_named_command_fixes_it(isolated, capsys):
    """An original can be the only copy of a previous version; `locate` leaves it."""
    payload = b"current version"
    entry = _original_entry("giamip/in.nc", payload)
    isolated["manifest"]["artifacts"]["giamip/in.nc"] = entry
    _serve(isolated["store"], "giamip/in.nc", entry, payload)
    path = isolated["root"] / "original" / "giamip" / "in.nc"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"previous version")
    with pytest.raises(ValueError) as caught:
        data.locate("giamip/in.nc", "original")
    assert path.read_bytes() == b"previous version"
    message = str(caught.value)
    command = message[message.index("python -m giamip.data"):].rstrip(".").split()
    assert command[:3] == ["python", "-m", "giamip.data"]
    assert data.main(command[3:]) == 0
    assert data.locate("giamip/in.nc", "original").read_bytes() == payload


def test_an_artifact_without_permission_is_refused(isolated):
    payload = b"secret"
    entry = _original_entry("giamip/closed.nc", payload, permission="requested")
    isolated["manifest"]["artifacts"]["giamip/closed.nc"] = entry
    _serve(isolated["store"], "giamip/closed.nc", entry, payload)
    with pytest.raises(PermissionError):
        data.locate("giamip/closed.nc", "original")
    assert not (isolated["root"] / "original" / "giamip" / "closed.nc").exists()


def test_an_unreachable_store_raises_with_the_command_and_a_second_try_fails_at_once(isolated, monkeypatch):
    payload = b"never served"
    isolated["manifest"]["artifacts"]["x/f.nc"] = _converted_entry("x/f.nc", payload)
    with pytest.raises(data.DataUnavailable, match="python -m giamip.data --stage converted"):
        data.locate("x/f.nc", "converted")
    calls = _count_fetches(monkeypatch)
    with pytest.raises(data.DataUnavailable, match="failed .* s ago"):
        data.locate("x/f.nc", "converted")
    assert calls == []


def test_a_request_for_the_wrong_stage_raises(isolated):
    isolated["manifest"]["artifacts"]["x/f.nc"] = _converted_entry("x/f.nc", b"c")
    with pytest.raises(KeyError, match="converted file"):
        data.locate("x/f.nc", "original")
    with pytest.raises(KeyError):
        data.locate("x/unknown.nc", "converted")


def test_an_archive_member_is_located_by_unpacking_its_archive(isolated):
    """The GIAMIP inputs are members of zip archives, as the case files name them."""
    member = b"ice history bytes"
    archive = _zip({"Ice History/iceHistory-X.nc": member, "Ice History/other.nc": b"o"})
    entry = _original_entry("giamip/ice-history.zip", archive, members=[
        {"name": "Ice History/iceHistory-X.nc", "bytes": len(member), "sha256": _sum(member)},
        {"name": "Ice History/other.nc", "bytes": 1, "sha256": _sum(b"o")}])
    isolated["manifest"]["artifacts"]["giamip/ice-history.zip"] = entry
    _serve(isolated["store"], "giamip/ice-history.zip", entry, archive)
    path = data.locate("giamip/ice-history/iceHistory-X.nc", "original")
    assert path == isolated["root"] / "original" / "giamip" / "ice-history" / "iceHistory-X.nc"
    assert path.read_bytes() == member


def test_an_archive_key_is_returned_unpacked_and_stamped(isolated):
    archive = _zip({"top/a.txt": b"a"})
    entry = _original_entry("x/tree.zip", archive,
                            members=[{"name": "top/a.txt", "bytes": 1, "sha256": _sum(b"a")}])
    isolated["manifest"]["artifacts"]["x/tree.zip"] = entry
    _serve(isolated["store"], "x/tree.zip", entry, archive)
    path = data.locate("x/tree.zip", "original")
    assert data.is_unpacked(path, entry["sha256"])
    assert (data.unpack_target(path) / "a.txt").read_bytes() == b"a"
    assert data.original_tree("x/tree.zip") == isolated["root"] / "original"


def test_a_download_leaves_no_other_file_except_dotted_ones(isolated):
    """Lock, failure and temporary files start with "." and so never become artifacts."""
    archive = _zip({"top/a.txt": b"a"})
    entry = _original_entry("x/tree.zip", archive,
                            members=[{"name": "top/a.txt", "bytes": 1, "sha256": _sum(b"a")}])
    isolated["manifest"]["artifacts"]["x/tree.zip"] = entry
    _serve(isolated["store"], "x/tree.zip", entry, archive)
    data.locate("x/tree.zip", "original")
    found = {str(p.relative_to(isolated["root"])) for p in isolated["root"].rglob("*")
             if p.is_file() and not p.name.startswith(".")}
    assert found == {"original/x/tree.zip", "original/x/tree/a.txt"}


def test_the_temporary_download_name_starts_with_a_dot(isolated, monkeypatch):
    payload = b"converted"
    entry = _converted_entry("x/f.nc", payload)
    isolated["manifest"]["artifacts"]["x/f.nc"] = entry
    _serve(isolated["store"], "x/f.nc", entry, payload)
    names = []
    real = pathlib.Path.replace

    def recording(self, target):
        names.append(self.name)
        return real(self, target)

    monkeypatch.setattr(pathlib.Path, "replace", recording)
    data.locate("x/f.nc", "converted")
    assert names == [f".f.nc.part-{os.getpid()}"]


def test_the_default_download_root_is_under_home_and_ignores_xdg(tmp_path, monkeypatch):
    """On Gadi XDG_CACHE_HOME is job storage, deleted after the job."""
    monkeypatch.setattr(data, "checkout_root", lambda: None)
    monkeypatch.delenv("GIAMIP_DATA_ROOT", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "jobfs"))
    assert data.download_root() == tmp_path / "home" / ".cache" / "giamip"


# ----------------------------------------------- two processes, one download

PROCESS = textwrap.dedent("""
    import json, os, pathlib, sys, time
    sys.path.insert(0, {repository!r})
    from giamip import data
    manifest = json.loads(pathlib.Path({manifest!r}).read_text())
    data.load_manifest = lambda *a, **k: manifest
    data.checkout_root = lambda: None
    real = data.fetch
    def slow(url, dest):
        with open({count!r}, "a") as f:
            f.write(str(os.getpid()) + "\\n")
        time.sleep(1.0)
        return real(url, dest)
    data.fetch = slow
    # Both processes are running before either reaches locate.
    pathlib.Path({markers!r}, str(os.getpid())).touch()
    while len(list(pathlib.Path({markers!r}).iterdir())) < 2:
        time.sleep(0.01)
    print(data.locate("x/f.nc", "converted"))
""")


def test_two_processes_download_one_missing_artifact_once(tmp_path):
    """The lock and the check after it: the second process finds the first one's file."""
    store, root, markers = tmp_path / "store", tmp_path / "root", tmp_path / "markers"
    markers.mkdir()
    payload = b"shared" * 1000
    entry = _converted_entry("x/f.nc", payload)
    _serve(store, "x/f.nc", entry, payload)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"base_url": store.as_uri() + "/",
                                         "artifacts": {"x/f.nc": entry}}))
    count = tmp_path / "count.txt"
    script = PROCESS.format(repository=str(REPOSITORY), manifest=str(manifest_path),
                            count=str(count), markers=str(markers))
    env = {**os.environ, "GIAMIP_DATA_ROOT": str(root)}
    env.pop("GIAMIP_BASE_URL", None)
    runs = [subprocess.Popen([sys.executable, "-c", script], env=env, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(2)]
    outputs = [run.communicate(timeout=60) for run in runs]
    for run, (out, err) in zip(runs, outputs):
        assert run.returncode == 0, err
    assert len(count.read_text().split()) == 1
    assert outputs[0][0] == outputs[1][0]


# ------------------------------------------ archives already on disk

def _two_versions_of_an_archive():
    """An older archive with one member, and the current one with two, as a GIAMIP zip can grow."""
    old = _zip({"top/a.nc": b"old a"})
    new = _zip({"top/a.nc": b"new a", "top/b.nc": b"new b"})
    entry = _original_entry("giamip/grow.zip", new, members=[
        {"name": "top/a.nc", "bytes": 5, "sha256": _sum(b"new a")},
        {"name": "top/b.nc", "bytes": 5, "sha256": _sum(b"new b")}])
    return old, new, entry


def test_an_older_archive_is_not_replaced_when_a_member_is_missing(isolated):
    """Downloading over it would remove a previous version and its tree without a word."""
    old, new, entry = _two_versions_of_an_archive()
    isolated["manifest"]["artifacts"]["giamip/grow.zip"] = entry
    _serve(isolated["store"], "giamip/grow.zip", entry, new)
    archive = isolated["root"] / "original" / "giamip" / "grow.zip"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(old)
    with pytest.raises(ValueError) as caught:
        data.locate("giamip/grow/b.nc", "original")
    assert archive.read_bytes() == old
    message = str(caught.value)
    command = message[message.index("python -m giamip.data"):].rstrip(".").split()
    assert data.main(command[3:]) == 0
    assert data.locate("giamip/grow/b.nc", "original").read_bytes() == b"new b"


def test_main_ignores_a_recent_failure_and_clears_it(isolated):
    """A person who runs the command wants the download tried."""
    payload = b"converted"
    entry = _converted_entry("x/f.nc", payload)
    isolated["manifest"]["artifacts"]["x/f.nc"] = entry
    _serve(isolated["store"], "x/f.nc", entry, payload)
    path = data.cache_path(isolated["root"], "x/f.nc", entry)
    path.parent.mkdir(parents=True)
    marker = data._failure_marker(path)
    marker.write_text("an earlier failure\n")
    assert data.main(["--stage", "converted", "--collection", "x"]) == 0
    assert path.read_bytes() == payload
    assert not marker.exists()
    assert data.locate("x/f.nc", "converted") == path


def test_a_root_that_cannot_be_written_raises_data_unavailable(isolated):
    """A shared read-only directory that lacks a file gives the way out, not a bare OSError."""
    if os.geteuid() == 0:
        pytest.skip("root can write anywhere")
    isolated["manifest"]["artifacts"]["x/f.nc"] = _converted_entry("x/f.nc", b"c")
    isolated["root"].mkdir()
    isolated["root"].chmod(0o500)
    try:
        with pytest.raises(data.DataUnavailable, match="cannot write"):
            data.locate("x/f.nc", "converted")
    finally:
        isolated["root"].chmod(0o700)


UNPACKING = textwrap.dedent("""
    import json, os, pathlib, sys, time
    sys.path.insert(0, {repository!r})
    from giamip import data
    manifest = json.loads(pathlib.Path({manifest!r}).read_text())
    data.load_manifest = lambda *a, **k: manifest
    data.checkout_root = lambda: None
    real = data.unpack
    def slow(archive, members, digest):
        with open({count!r}, "a") as f:
            f.write(str(os.getpid()) + "\\n")
        time.sleep(0.5)
        return real(archive, members, digest)
    data.unpack = slow
    pathlib.Path({markers!r}, str(os.getpid())).touch()
    while len(list(pathlib.Path({markers!r}).iterdir())) < {processes}:
        time.sleep(0.01)
    print(data.locate("giamip/tree/a.nc", "original"))
""")


def test_processes_unpack_an_archive_in_place_once(tmp_path):
    """An archive present but not unpacked (after --no-unpack, or copied alone) is unpacked once."""
    root, markers = tmp_path / "root", tmp_path / "markers"
    markers.mkdir()
    archive_bytes = _zip({"top/a.nc": b"member"})
    entry = _original_entry("giamip/tree.zip", archive_bytes, members=[
        {"name": "top/a.nc", "bytes": 6, "sha256": _sum(b"member")}])
    archive = root / "original" / "giamip" / "tree.zip"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(archive_bytes)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"base_url": "https://unused.invalid/",
                                         "artifacts": {"giamip/tree.zip": entry}}))
    count = tmp_path / "count.txt"
    processes = 4
    script = UNPACKING.format(repository=str(REPOSITORY), manifest=str(manifest_path),
                              count=str(count), markers=str(markers), processes=processes)
    env = {**os.environ, "GIAMIP_DATA_ROOT": str(root)}
    runs = [subprocess.Popen([sys.executable, "-c", script], env=env, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            for _ in range(processes)]
    outputs = [run.communicate(timeout=60) for run in runs]
    for run, (out, err) in zip(runs, outputs):
        assert run.returncode == 0, err
    assert len({out for out, _ in outputs}) == 1
    assert len(count.read_text().split()) == 1
