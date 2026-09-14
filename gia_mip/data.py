"""Fetch the original data files and verify them against the manifest.

The third-party data this project depends on is 341 MB and belongs to the
people who produced it, so it is not in the repository. It lives in the gadopt
object store under ``gia-mip/``. This module downloads what is missing or
corrupt and checks every file against the SHA-256 sums in
``gia_mip/resources/manifest.json``, so a fetched tree is known to be the tree
the conversions were made from.

    python -m gia_mip.data                         # both benchmarks, 24 MB
    python -m gia_mip.data --collection giamip     # the GIAMIP inputs, 317 MB
    python -m gia_mip.data --collection spada2011  # one collection
    python -m gia_mip.data --stage converted       # what this project derived
    python -m gia_mip.data --all
    python -m gia_mip.data --list                  # what exists, what is local
    python -m gia_mip.data --verify                # re-check what is here
    python -m gia_mip.data --dest DIR              # or set GIA_MIP_DATA_ROOT

A file is addressed by two independent axes. `--collection` is the body of
work it belongs to: `spada2011`, `martinec2018` or `giamip`, with `benchmarks`
as an alias for the first two. `--stage` is whether it is an untouched
third-party file (`original`, the default) or something this project derived
from one (`converted`).

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
import hashlib
import json
import os
import pathlib
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile

CHUNK = 1 << 20

# The manifest travels with the package, so an installed gia-mip can fetch
# without a checkout. Resolved relative to this file for the same reason.
MANIFEST_PATH = pathlib.Path(__file__).resolve().parent / "resources" / "manifest.json"

# Two levels above this module is the repository root when running from a
# checkout. An installed package has no such root, so --dest is required there.
ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_DEST = ROOT / "data"

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
    could treat it as complete.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    try:
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
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
# re-run neither re-extracts 317 MB nor destroys anything a user put there.
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
    scratch = target.with_name(target.name + f".unpacking-{os.getpid()}")
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
        stale = target.with_name(target.name + f".stale-{os.getpid()}")
        target.rename(stale)
        scratch.rename(target)
        shutil.rmtree(stale, ignore_errors=True)
    else:
        scratch.rename(target)
    return target


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
                    help="directory to fill; default $GIA_MIP_DATA_ROOT, "
                         "else <repo>/data. On a shared machine point this at "
                         "one directory so worktrees do not each hold 341 MB.")
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

    # The destination is <root>/<stage>/<relative path>, so originals and
    # converted products never collide and one root holds both.
    root = args.dest or (
        pathlib.Path(os.environ["GIA_MIP_DATA_ROOT"])
        if "GIA_MIP_DATA_ROOT" in os.environ
        else DEFAULT_DEST
    )
    # An installed package sits in site-packages, where DEFAULT_DEST would
    # scatter 341 MB among the dependencies. Refuse instead.
    if args.dest is None and "GIA_MIP_DATA_ROOT" not in os.environ:
        if not (ROOT / "pyproject.toml").exists():
            raise SystemExit(
                "gia_mip is installed rather than checked out, so there is no "
                "repository to fill. Pass --dest DIR or set GIA_MIP_DATA_ROOT."
            )

    def local(rel: str, entry: dict) -> pathlib.Path:
        return root / entry["stage"] / rel

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
    base = os.environ.get("GIA_MIP_BASE_URL", manifest["base_url"])
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

        url = base + rel
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
