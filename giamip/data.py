"""Find, fetch and verify every data file, and cache it at the first use.

No data file is in the package. The third-party originals (341 MB) and the
converted reference data derived from them (10 MB) live in the gadopt object
store under ``giamip/``, served at https://data.gadopt.org/giamip/. Every
file is checked against the SHA-256 sum in ``giamip/resources/manifest.json``,
so a file this package reads is known to be the file the manifest describes.

A reader asks for a file with `locate`, which returns a local path:

    >>> from giamip import data
    >>> data.locate("martinec2018/martinec2018-B.nc", "converted")   # doctest: +SKIP

`locate` looks in the checkout's ``data/`` (when the package runs from a
checkout) and in the download root, and downloads the file into the download
root when neither has it. The download root is ``$GIAMIP_DATA_ROOT`` when set,
else ``data/`` of the checkout, else ``~/.cache/giamip``. A machine with no
network, such as a compute node of Gadi, reads what an earlier run on a
machine with network put there; `locate` then raises `DataUnavailable` with
the command that fills the directory.

The same can be done in advance from the command line:

    python -m giamip.data                         # both benchmarks, originals
    python -m giamip.data --stage converted       # the converted reference data
    python -m giamip.data --collection giamip     # the GIAMIP inputs, 363 MB
    python -m giamip.data --collection spada2011  # one collection
    python -m giamip.data --all
    python -m giamip.data --list                  # what exists, what is local
    python -m giamip.data --verify                # re-check what is here
    python -m giamip.data --dest DIR              # or set GIAMIP_DATA_ROOT

A file is addressed by two independent axes. `--collection` is the body of
work it belongs to: `spada2011`, `martinec2018` or `giamip`, with `benchmarks`
as an alias for the first two. `--stage` is whether it is an untouched
third-party file (`original`, the default) or something this project derived
from one (`converted`).

Where a file lives under a root is `cache_path`: an original at
``<root>/original/<key>``, and a converted file at ``<root>/<object>``, where
``<object>`` is ``versions/<sha256>/<key>``. The sum in the path gives each
version of a converted file its own place, so two versions of giamip that
share one cache never overwrite each other's files.

Archives are unpacked next to themselves after verification, into a directory
named after the archive with its suffixes removed. If an archive holds exactly
one top-level directory, that directory is stripped, so
``giamip/ice-history.zip`` gives ``giamip/ice-history/iceHistory-ICE-6G_D.nc``
and not ``giamip/ice-history/Ice History/iceHistory-ICE-6G_D.nc``. Pass
``--no-unpack`` to keep the archives closed.

Only the standard library is used. This module runs on a bare CI runner before
anything is installed, and it imports nothing else from the package, so a
broken reader can never stop a fetch.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import pathlib
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile

# `fcntl` gives the file lock that lets several processes, for example the MPI
# ranks of one job, share one first download. It exists on Linux and macOS.
# Where it does not (Windows), there is no lock and the module still imports.
try:
    import fcntl
except ImportError:                                  # pragma: no cover
    fcntl = None

CHUNK = 1 << 20

# The manifest travels with the package, so an installed giamip can fetch
# without a checkout. Resolved relative to this file for the same reason.
MANIFEST_PATH = pathlib.Path(__file__).resolve().parent / "resources" / "manifest.json"

# Two levels above this module is the repository root when running from a
# checkout. `checkout_root` decides whether it is one.
ROOT = pathlib.Path(__file__).resolve().parent.parent

# Seconds a download may wait for the server before it fails. A compute node
# that drops outgoing connections would otherwise wait for the operating
# system's own timeout, about two minutes on Linux, on every rank.
TIMEOUT_S = 30

# Seconds during which a recorded failed download makes `locate` fail at once
# without trying again, so that N processes behind one lock do not each wait
# a full timeout.
FAILURE_MEMORY_S = 60

# `benchmarks` is the only alias, because asking for "the two published
# benchmarks" is the common case and naming both every time is tedious.
COLLECTION_ALIASES = {"benchmarks": ("spada2011", "martinec2018")}

# Stages are a fixed vocabulary, not a set derived from the manifest. A stage
# that holds no files yet is a real stage and must give an empty result, while
# a misspelled stage must be an error.
STAGES = ("original", "converted")


def load_manifest(path: pathlib.Path = MANIFEST_PATH) -> dict:
    """Read the manifest. Raises if it is absent, which means a broken install."""
    with open(path) as f:
        return json.load(f)


def sha256(path: pathlib.Path) -> str:
    """SHA-256 of a file, streamed, so a 300 MB archive is never read whole."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def file_versions(manifest: dict | None = None) -> dict[str, dict]:
    """Every file the manifest describes, with its current and previous SHA-256.

    An archive's members are listed under the path they unpack to, so
    ``giamip/ice-history.zip`` member ``Ice History/iceHistory-ICE-6G_D.nc`` is
    ``giamip/ice-history/iceHistory-ICE-6G_D.nc``, which is the path a reader
    opens and the path a correction names.

    Parameters
    ----------
    manifest
        The manifest. Default: the one in the package resources.

    Returns
    -------
    dict
        Relative path to ``{"sha256": current sum, "previous": [older sums]}``.
        The older sums come from ``previous_versions``, which
        scripts/build_manifest.py appends to whenever a file changes.
    """
    manifest = manifest or load_manifest()
    out: dict[str, dict] = {}
    for rel, entry in manifest["artifacts"].items():
        previous = entry.get("previous_versions", [])
        out[rel] = {"sha256": entry["sha256"], "previous": [v["sha256"] for v in previous]}
        members = entry.get("members")
        if not members:
            continue
        top = common_top([m["name"] for m in members])
        target = unpack_target(pathlib.PurePosixPath(rel))
        for m in members:
            path = f"{target}/{member_path(m['name'], top)}"
            older = [x["sha256"] for v in previous for x in v.get("members", [])
                     if member_path(x["name"], common_top([y["name"] for y in v["members"]])) == member_path(m["name"], top)]
            out[path] = {"sha256": m["sha256"], "previous": older}
    return out


def fetch(url: str, dest: pathlib.Path) -> None:
    """Download one file, writing to a temporary name first.

    An interrupted download must never leave a truncated file under the final
    name, because the next run would find it present and, without --verify,
    could treat it as complete. The temporary name carries the process id, so
    two processes never write the same file, and starts with ".", so
    scripts/build_manifest.py never takes a left-over one for an artifact.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.part-{os.getpid()}")
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_S) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            for chunk in iter(lambda: r.read(CHUNK), b""):
                f.write(chunk)
                done += len(chunk)
                if total and sys.stderr.isatty():
                    pct = 100 * done / total
                    print(f"\r  {dest.name}: {pct:5.1f}%", end="", file=sys.stderr)
            if total and sys.stderr.isatty():
                print("\r" + " " * 40 + "\r", end="", file=sys.stderr)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(dest)


def unpack_target(archive: pathlib.Path) -> pathlib.Path:
    """Directory an archive unpacks into: the archive name without suffixes."""
    name = archive.name
    for suffix in (".tar.gz", ".tgz", ".tar", ".zip"):
        if name.endswith(suffix):
            return archive.with_name(name[: -len(suffix)])
    return archive.with_name(name + ".d")


def common_top(names: list[str]) -> str | None:
    """The single top-level directory every member sits inside, if there is one.

    Returns None when members sit at the archive root, or under more than one
    top-level directory, in which case nothing is stripped. A member that *is*
    the candidate name, with nothing below it, means the candidate is a file
    and not a directory, so None is returned there too: stripping it would
    leave an empty path.
    """
    cleaned = [n[2:] if n.startswith("./") else n for n in names]
    tops = {n.split("/", 1)[0] for n in cleaned if n}
    if len(tops) != 1:
        return None
    top = tops.pop()
    if not all(n.startswith(top + "/") for n in cleaned):
        return None
    return top


def member_path(name: str, top: str | None) -> str:
    """Where one archive member lands, relative to the unpack target.

    This is the single rule that maps a member name in the manifest to a path
    on disk. A leading "./" is dropped, and the archive's one top-level
    directory is removed when it has one, so
    ``giamip/ice-history.zip`` member ``Ice History/iceHistory-ICE-6G_D.nc``
    becomes ``iceHistory-ICE-6G_D.nc`` while
    ``sbk_benchmark_data.tar.gz`` member ``l721j256/A_fig10_SBK.dat`` keeps
    its directory, because that archive has no single top level to strip.
    """
    cleaned = name[2:] if name.startswith("./") else name
    if top and cleaned.startswith(top + "/"):
        return cleaned[len(top) + 1:]
    return cleaned


def check_member_names(names: list[str]) -> None:
    """Reject absolute paths and parent references, before anything is stripped.

    Checked on the raw names. Stripping first would let "../../evil.txt" pass,
    because removing a leading "./" with str.lstrip removes characters and not
    a prefix, turning "../../evil.txt" into "evil.txt".
    """
    for name in names:
        p = pathlib.PurePosixPath(name)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError(f"archive member escapes the target directory: {name}")


# Written inside an unpack target, holding the SHA-256 of the archive it came
# from. Its presence with a matching sum means the tree is current, so a
# re-run neither re-extracts 363 MB nor destroys anything a user put there.
STAMP = ".unpacked-from"


def is_unpacked(archive: pathlib.Path, digest: str) -> bool:
    """True when the archive's target already holds the contents of this archive."""
    stamp = unpack_target(archive) / STAMP
    try:
        return stamp.read_text().strip() == digest
    except OSError:
        return False


def unpack(archive: pathlib.Path, members: list[str], digest: str) -> pathlib.Path:
    """Unpack an archive, replacing its target directory only once it succeeds.

    Extraction goes to a scratch directory beside the target, and the target
    is replaced only after every member is written. An archive that turns out
    to be corrupt half way through therefore leaves the previous tree intact,
    which matters because the previous tree may be the only copy.

    Parameters
    ----------
    archive
        The archive to unpack, already verified against its SHA-256.
    members
        Member names from the manifest. Used to decide the common top-level
        directory in one pass, so the tar and zip branches agree.
    digest
        The archive's SHA-256, written into the target as a stamp.
    """
    check_member_names(members)
    top = common_top(members)
    target = unpack_target(archive)
    # The scratch and the old tree start with ".", so a killed unpack in a
    # checkout leaves nothing that scripts/build_manifest.py reports.
    scratch = target.with_name(f".{target.name}.unpacking-{os.getpid()}")
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)

    def write(relative: str, source) -> None:
        out = scratch / relative
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as dst:
            shutil.copyfileobj(source, dst, CHUNK)

    try:
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as z:
                check_member_names([i.filename for i in z.infolist()])
                for info in z.infolist():
                    if info.is_dir():
                        continue
                    with z.open(info) as src:
                        write(member_path(info.filename, top), src)
        else:
            with tarfile.open(archive, "r|*") as tar:
                for info in tar:
                    if not info.isfile():
                        continue
                    check_member_names([info.name])
                    src = tar.extractfile(info)
                    if src is None:
                        continue
                    write(member_path(info.name, top), src)
        (scratch / STAMP).write_text(digest + "\n")
    except BaseException:
        shutil.rmtree(scratch, ignore_errors=True)
        raise

    # Swap. The old tree is moved aside first and removed afterwards, so a
    # failure between the two leaves something recoverable on disk.
    if target.exists():
        stale = target.with_name(f".{target.name}.stale-{os.getpid()}")
        target.rename(stale)
        scratch.rename(target)
        shutil.rmtree(stale, ignore_errors=True)
    else:
        scratch.rename(target)
    return target


# ---------------------------------------------------------------- locate


class DataUnavailable(FileNotFoundError):
    """A data file is not on this machine and cannot be downloaded now.

    The message names the file, the reason and the command that fills a
    directory with it on a machine that has network access.
    """


def checkout_root() -> pathlib.Path | None:
    """The repository root when the package runs from a checkout, else None.

    A checkout has ``pyproject.toml`` two levels above this module. An
    installed package sits in ``site-packages``, which has none.
    """
    return ROOT if (ROOT / "pyproject.toml").exists() else None


def download_root() -> pathlib.Path:
    """Where `locate` puts a file it downloads.

    ``$GIAMIP_DATA_ROOT`` when set, else ``data/`` of the checkout, else
    ``~/.cache/giamip``. ``XDG_CACHE_HOME`` is not read on purpose: the
    Firedrake module on Gadi points it at the job's own storage, which is
    deleted when the job ends, so a cache there would be empty in every job.
    """
    if "GIAMIP_DATA_ROOT" in os.environ:
        return pathlib.Path(os.environ["GIAMIP_DATA_ROOT"])
    checkout = checkout_root()
    if checkout is not None:
        return checkout / "data"
    return pathlib.Path.home() / ".cache" / "giamip"


def cache_path(root: pathlib.Path, key: str, entry: dict) -> pathlib.Path:
    """Where an artifact lives under a root.

    ``root / entry["object"]`` for an artifact with an ``object`` field (a
    converted file: ``versions/<sha256>/<key>``), else
    ``root / stage / key`` (an original, where the converters read its
    unpacked tree). In a checkout, ``data/versions/`` is ignored by the
    ``/data/*`` rule of .gitignore and lies outside the tracked
    ``data/converted/``.
    """
    if "object" in entry:
        return pathlib.Path(root) / entry["object"]
    return pathlib.Path(root) / entry["stage"] / key


# Sums of files already checked in this process, keyed by path, size and
# modification time, so a 62 MB ice history is hashed once per process however
# often it is located or opened.
_VERIFIED: dict[tuple[str, int, int], str] = {}


def verified_sha256(path: pathlib.Path) -> str:
    """The SHA-256 of a file, computed once per process for each version of it."""
    stat = pathlib.Path(path).stat()
    key = (str(pathlib.Path(path).resolve()), stat.st_size, stat.st_mtime_ns)
    if key not in _VERIFIED:
        _VERIFIED[key] = sha256(pathlib.Path(path))
    return _VERIFIED[key]


def _lookup(relative: str, manifest: dict) -> tuple[str, dict, dict | None, str | None]:
    """The artifact that holds a path, and the member when it is inside an archive.

    Returns ``(key, entry, member, member_relative)``. For an artifact the last
    two are None. For an archive member, ``member`` is its manifest record and
    ``member_relative`` its path inside the unpack target.
    """
    artifacts = manifest["artifacts"]
    if relative in artifacts:
        return relative, artifacts[relative], None, None
    for key, entry in artifacts.items():
        members = entry.get("members")
        if not members:
            continue
        target = str(unpack_target(pathlib.PurePosixPath(key)))
        prefix = target + "/"
        if not relative.startswith(prefix):
            continue
        top = common_top([m["name"] for m in members])
        inside = relative[len(prefix):]
        for m in members:
            if member_path(m["name"], top) == inside:
                return key, entry, m, inside
    raise KeyError(f"{relative!r} is neither an artifact of the manifest nor a member "
                   f"of one of its archives")


def _command(entry: dict, root: pathlib.Path | None = None) -> str:
    """The command that puts an artifact into a directory, for error messages."""
    dest = f" --dest {root}" if root is not None else ""
    env = "" if root is not None else "GIAMIP_DATA_ROOT=<a directory this machine can read> "
    return (f"{env}python -m giamip.data --stage {entry['stage']} "
            f"--collection {entry['collection']}{dest}")


@contextlib.contextmanager
def _locked(path: pathlib.Path):
    """Hold an exclusive lock for one artifact while it is downloaded and unpacked.

    The lock file is ``.<name>.lock`` beside the artifact. It starts with "."
    so that scripts/build_manifest.py never counts it. On Gadi, /g/data and
    /scratch are mounted with ``localflock``, so the lock excludes only
    processes on the same node; processes on two nodes may both download,
    and each still writes through its own temporary file.
    """
    # A download root that cannot be written, for example a shared read-only
    # directory that lacks this file, is reported as unavailable data with
    # the way out, not as a bare OSError.
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(path.with_name(f".{path.name}.lock"), "a")
    except OSError as exc:
        raise DataUnavailable(f"cannot write to {path.parent} ({exc}). Set "
                              f"GIAMIP_DATA_ROOT to a directory this process can write, "
                              f"or fill {path.parent} from an account that can.") from exc
    with handle:
        if fcntl is None:                            # pragma: no cover
            yield
            return
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _failure_marker(path: pathlib.Path) -> pathlib.Path:
    """The file that records a failed download of an artifact, with its time."""
    return path.with_name(f".{path.name}.failed")


def _download(key: str, entry: dict, path: pathlib.Path, manifest: dict) -> None:
    """Download one artifact to ``path``, check its sum, unpack an archive.

    Runs under `_locked`. A process that gets the lock after another one
    finished finds the file in place and does nothing. A process that finds
    a failure younger than `FAILURE_MEMORY_S` raises at once.
    """
    with _locked(path):
        if path.exists() and verified_sha256(path) == entry["sha256"]:
            _ensure_unpacked(path, entry)
            return
        marker = _failure_marker(path)
        if marker.exists():
            age = time.time() - marker.stat().st_mtime
            if age < FAILURE_MEMORY_S:
                raise DataUnavailable(
                    f"{key}: a download failed {age:.0f} s ago "
                    f"({marker.read_text().strip()}); not trying again for "
                    f"{FAILURE_MEMORY_S} s. On a machine with network access run: "
                    f"{_command(entry)}")
        base = os.environ.get("GIAMIP_BASE_URL", manifest["base_url"])
        if not base.endswith("/"):
            base += "/"
        url = base + entry.get("object", key)
        try:
            fetch(url, path)
        except (urllib.error.URLError, OSError) as exc:
            marker.write_text(f"{url}: {exc}\n")
            raise DataUnavailable(
                f"{key} is not in {path.parent} and cannot be downloaded from {url} "
                f"({exc}). On a machine with network access run: {_command(entry)}, "
                f"and set GIAMIP_DATA_ROOT to the same directory here.") from exc
        if verified_sha256(path) != entry["sha256"]:
            path.unlink()
            raise DataUnavailable(f"{key}: the file downloaded from {url} does not match "
                                  f"the manifest's SHA-256")
        marker.unlink(missing_ok=True)
        _ensure_unpacked(path, entry)


def _ensure_unpacked(path: pathlib.Path, entry: dict) -> None:
    """Unpack a verified archive once; do nothing for a file that is not one.

    The caller holds the lock of the archive (`_locked`), because two
    processes that unpack one archive at once remove each other's trees.
    """
    members = entry.get("members")
    if members and not is_unpacked(path, entry["sha256"]):
        unpack(path, [m["name"] for m in members], entry["sha256"])


def _unpack_locked(path: pathlib.Path, entry: dict) -> None:
    """`_ensure_unpacked` under the archive's lock, for an archive already in place.

    An archive can be present and not unpacked: after ``--no-unpack``, after
    an unpack that was killed, or when only the archive was copied. The ranks
    of a job that find it so must unpack it once, so the check is made again
    once the lock is held. The quick test before the lock keeps the common
    case, a tree already unpacked, free of lock files.
    """
    if not entry.get("members") or is_unpacked(path, entry["sha256"]):
        return
    with _locked(path):
        _ensure_unpacked(path, entry)


def locate(relative: str, stage: str) -> pathlib.Path:
    """The local path of a data file, downloaded and checked on first use.

    Parameters
    ----------
    relative
        A manifest key, for example ``"martinec2018/martinec2018-B.nc"``, or
        the path of an archive member after unpacking, for example
        ``"giamip/ice-history/iceHistory-ICE-6G_D.nc"``.
    stage
        ``"original"`` or ``"converted"``. It must be the artifact's stage.

    Returns
    -------
    pathlib.Path
        A file whose SHA-256 is the manifest's. For an archive, the archive,
        with its tree unpacked beside it.

    Raises
    ------
    KeyError
        For a path the manifest does not know, or one of another stage.
    ValueError
        For a file in place with a wrong sum that `locate` must not replace:
        a converted file in the checkout's ``data/converted/`` (run
        scripts/build_manifest.py), or an original anywhere (the message gives
        the command that replaces it; an original can be the only local copy
        of a previous version, which the store no longer has).
    PermissionError
        For an artifact whose ``permission`` is not ``granted``.
    DataUnavailable
        When the file is absent and cannot be downloaded.
    """
    manifest = load_manifest()
    key, entry, member, inside = _lookup(relative, manifest)
    if entry["stage"] != stage:
        raise KeyError(f"{relative!r} is a {entry['stage']} file, not a {stage} one")
    expected = member["sha256"] if member else entry["sha256"]
    checkout = checkout_root()

    # Where the artifact itself (an archive, for a member) can be.
    homes: list[tuple[pathlib.Path, pathlib.Path]] = []      # (root, artifact path)
    if checkout is not None:
        if stage == "converted":
            homes.append((checkout / "data", checkout / "data" / "converted" / key))
        else:
            homes.append((checkout / "data", cache_path(checkout / "data", key, entry)))
    root = download_root()
    download_path = cache_path(root, key, entry)
    if all(download_path != path for _, path in homes):
        homes.append((root, download_path))

    def target(path: pathlib.Path) -> pathlib.Path:
        return unpack_target(path) / inside if member else path

    # A file identified by its sum can be read from any home that has it. A
    # wrong sum stops the search: it is either a development state of the
    # checkout or an original that must not be replaced here.
    for home_root, path in homes:
        candidate = target(path)
        if not candidate.exists():
            continue
        if verified_sha256(candidate) == expected:
            if not member:
                _unpack_locked(candidate, entry)
            return candidate
        if stage == "converted" and path.is_relative_to(home_root / "converted"):
            raise ValueError(f"{candidate} does not match the manifest. It is a tracked "
                             f"converted file; run scripts/build_manifest.py after a "
                             f"conversion.")
        if stage == "original":
            extra = (f" Remove {unpack_target(path)} first, because the archive and its "
                     f"stamp match and the command does not unpack it again."
                     if member else "")
            raise ValueError(f"{candidate} does not match the manifest. It can be a "
                             f"previous version, which giamip does not replace by itself. "
                             f"To replace it run: {_command(entry, home_root)}.{extra}")
        # A converted file at its content-addressed path whose sum is wrong is
        # corrupt, because the path names the sum. It is downloaded again.
        candidate.unlink()

    # Nothing usable in place. For a member, an archive in place is used: one
    # that matches is unpacked, and one that does not stops the search as any
    # original with a wrong sum does, because downloading over it would
    # remove a previous version and its tree. Otherwise the artifact is
    # downloaded.
    found = None
    if member:
        for home_root, path in homes:
            if not path.exists():
                continue
            if verified_sha256(path) != entry["sha256"]:
                raise ValueError(f"{path} does not match the manifest, and its member "
                                 f"{inside} is not unpacked. It can be a previous version, "
                                 f"which giamip does not replace by itself. To replace it "
                                 f"run: {_command(entry, home_root)}.")
            _unpack_locked(path, entry)
            found = path
            break
    if found is None:
        if entry["permission"] != "granted":
            raise PermissionError(f"{key}: the supplier has not agreed to redistribution "
                                  f"(permission={entry['permission']})")
        found = download_path
        _download(key, entry, found, manifest)
    path = found
    candidate = target(path)
    if not candidate.exists() or verified_sha256(candidate) != expected:
        raise ValueError(f"{candidate} is missing or wrong after unpacking {path}")
    return candidate


def original_tree(key: str) -> pathlib.Path:
    """The ``original/`` directory that holds an archive's unpacked tree.

    `locate` finds, downloads and unpacks the archive as needed. The
    converters read the tree below the returned directory.
    """
    archive = locate(key, "original")
    return archive.parents[len(pathlib.PurePosixPath(key).parts) - 1]


def expand(names: list[str]) -> set[str]:
    """Resolve collection names, replacing any alias with what it stands for."""
    out: set[str] = set()
    for name in names:
        out |= set(COLLECTION_ALIASES.get(name, (name,)))
    return out


def select(manifest: dict, collections: set[str] | None, stages: set[str]) -> dict:
    """The artifacts matching the requested collections and stages.

    `collections` of None means every collection. An unknown name is an error
    and not an empty result, because a typo that silently fetches nothing is
    indistinguishable from a collection that has no files yet.
    """
    artifacts = manifest["artifacts"]
    known_collections = {a["collection"] for a in artifacts.values()}
    known_stages = set(STAGES)

    if collections is not None:
        unknown = collections - known_collections
        if unknown:
            raise SystemExit(
                f"unknown collection(s): {', '.join(sorted(unknown))}. Known: "
                f"{', '.join(sorted(known_collections))}, plus the alias "
                f"{', '.join(sorted(COLLECTION_ALIASES))}"
            )
    unknown = stages - known_stages
    if unknown:
        raise SystemExit(
            f"unknown stage(s): {', '.join(sorted(unknown))}. "
            f"Known: {', '.join(STAGES)}"
        )

    return {
        k: v for k, v in artifacts.items()
        if (collections is None or v["collection"] in collections)
        and v["stage"] in stages
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dest", type=pathlib.Path, default=None,
                    help="directory to fill; default $GIAMIP_DATA_ROOT, else "
                         "<repo>/data in a checkout, else ~/.cache/giamip. On a "
                         "shared machine point this at one directory so worktrees "
                         "do not each hold 341 MB.")
    ap.add_argument("--collection", action="append", default=None,
                    help="spada2011, martinec2018, giamip, or the alias "
                         "benchmarks; repeatable. Default: benchmarks")
    ap.add_argument("--stage", action="append", default=None,
                    help="original or converted; repeatable. Default: original")
    ap.add_argument("--all", action="store_true",
                    help="every collection, at the requested stages")
    ap.add_argument("--list", action="store_true",
                    help="list the artifacts and whether each is present locally")
    ap.add_argument("--verify", action="store_true",
                    help="check the sums of what is already present, download nothing")
    ap.add_argument("--no-unpack", action="store_true", help="leave archives closed")
    args = ap.parse_args(argv)

    manifest = load_manifest()
    collections = None if args.all else expand(args.collection or ["benchmarks"])
    stages = set(args.stage or ["original"])
    artifacts = select(manifest, collections, stages)

    # Each artifact goes to `cache_path` under the root: an original at
    # <root>/original/<key>, a converted file at <root>/versions/<sha256>/<key>.
    root = args.dest or download_root()
    checkout = checkout_root()
    # In a checkout filled by default, the tracked converted files are the
    # converted data. A converted artifact whose tracked copy matches the
    # manifest is reported and not downloaded again under versions/. A tracked
    # copy that does not match is left alone, never overwritten: the
    # download, if any, goes to versions/.
    use_tracked = checkout is not None and args.dest is None and "GIAMIP_DATA_ROOT" not in os.environ

    def local(rel: str, entry: dict) -> pathlib.Path:
        if use_tracked and entry["stage"] == "converted":
            tracked = checkout / "data" / "converted" / rel
            if tracked.exists() and verified_sha256(tracked) == entry["sha256"]:
                return tracked
        return cache_path(root, rel, entry)

    if args.list and not args.verify:
        if not artifacts:
            print("no artifacts match that collection and stage")
        for rel, entry in artifacts.items():
            path = local(rel, entry)
            state = "present" if path.exists() else "absent"
            print(f"{entry['collection']:12s} {entry['stage']:9s} {rel:52s} "
                  f"{entry['bytes'] / 1e6:8.1f} MB  {state}  "
                  f"permission={entry['permission']}")
        return 0

    # A base URL without a trailing slash would give a mangled path, because
    # the artifact key is joined to it directly.
    base = os.environ.get("GIAMIP_BASE_URL", manifest["base_url"])
    if not base.endswith("/"):
        base += "/"

    failures = 0
    # Only an artifact whose sum matched in this run is unpacked. Unpacking a
    # corrupt archive would destroy the previous tree before failing.
    verified: dict[str, dict] = {}
    for rel, entry in artifacts.items():
        path = local(rel, entry)
        if path.exists():
            if path.stat().st_size != entry["bytes"]:
                print(f"WRONG SIZE {rel}: {path.stat().st_size} bytes on disk, "
                      f"{entry['bytes']} expected", file=sys.stderr)
                if args.verify:
                    failures += 1
                    continue
            else:
                digest = sha256(path)
                if digest == entry["sha256"]:
                    print(f"ok       {rel}")
                    verified[rel] = entry
                    continue
                print(f"CORRUPT  {rel}: sha256 {digest[:16]} expected "
                      f"{entry['sha256'][:16]}", file=sys.stderr)
                if args.verify:
                    failures += 1
                    continue
        elif args.verify:
            print(f"MISSING  {rel}", file=sys.stderr)
            failures += 1
            continue

        if entry["permission"] != "granted":
            print(f"SKIPPED  {rel}: the supplier has not agreed to redistribution "
                  f"(permission={entry['permission']}). See docs/data/manifest.qmd.",
                  file=sys.stderr)
            failures += 1
            continue

        url = base + entry.get("object", rel)
        print(f"fetch    {rel} ({entry['bytes'] / 1e6:.1f} MB)")
        try:
            fetch(url, path)
        except urllib.error.URLError as exc:
            print(f"FAILED   {rel}: {exc}", file=sys.stderr)
            failures += 1
            continue
        digest = sha256(path)
        if digest != entry["sha256"]:
            print(f"CORRUPT  {rel} after download: sha256 {digest[:16]} expected "
                  f"{entry['sha256'][:16]}", file=sys.stderr)
            failures += 1
            continue
        # The command ignores a failure that `locate` recorded, because a
        # person who runs it wants the download tried, and it clears the
        # record once the download succeeded.
        _failure_marker(path).unlink(missing_ok=True)
        verified[rel] = entry

    if not args.no_unpack and not args.verify:
        for rel, entry in verified.items():
            if "members" not in entry:
                continue
            path = local(rel, entry)
            if is_unpacked(path, entry["sha256"]):
                continue
            names = [m["name"] for m in entry["members"]]
            try:
                target = unpack(path, names, entry["sha256"])
            except (ValueError, tarfile.TarError, zipfile.BadZipFile, OSError) as exc:
                print(f"UNPACK FAILED {rel}: {exc}", file=sys.stderr)
                failures += 1
                continue
            print(f"unpack   {rel} -> {target.relative_to(root)}/")

    if failures:
        print(f"\n{failures} artifact(s) unavailable or unverified.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
