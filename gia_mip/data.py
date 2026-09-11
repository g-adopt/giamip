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


def _common_top(names: list[str]) -> str | None:
    """The single top-level directory shared by every member, if there is one.

    Returns None when members sit at the archive root or under more than one
    top-level directory, in which case nothing is stripped.
    """
    tops = {n.split("/", 1)[0] for n in names if n not in (".", "")}
    tops.discard(".")
    if len(tops) != 1:
        return None
    top = tops.pop()
    # Every member must be *inside* that directory, not be that directory's
    # sibling with a shared prefix.
    if all(n == top or n.startswith(top + "/") for n in names):
        return top
    return None


def unpack(archive: pathlib.Path) -> pathlib.Path:
    """Unpack an archive into its target directory, replacing what is there.

    Members are checked for paths that escape the target before anything is
    written. A downloaded archive is verified by SHA-256 before this runs, so
    the check guards against a manifest that was poisoned upstream, not
    against an accident.
    """
    target = unpack_target(archive)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            names = [i.filename for i in z.infolist() if not i.is_dir()]
            _check_members(names)
            top = _common_top([n.lstrip("./") for n in names])
            for info in z.infolist():
                if info.is_dir():
                    continue
                rel = info.filename.lstrip("./")
                if top:
                    rel = rel[len(top) + 1 :]
                out = target / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, open(out, "wb") as dst:
                    shutil.copyfileobj(src, dst, CHUNK)
    else:
        with tarfile.open(archive, "r|*") as t:
            # Streaming mode gives one pass, so the common-prefix decision has
            # to be made from the manifest rather than from a first pass over
            # the archive. Tar archives in this project are packed from inside
            # the directory, so their members start with "./" and there is no
            # top-level directory to strip.
            for info in t:
                if not info.isfile():
                    continue
                rel = info.name.lstrip("./")
                _check_members([rel])
                src = t.extractfile(info)
                if src is None:
                    continue
                out = target / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                with open(out, "wb") as dst:
                    shutil.copyfileobj(src, dst, CHUNK)
    return target


def _check_members(names: list[str]) -> None:
    """Reject absolute paths and parent references in archive member names."""
    for name in names:
        p = pathlib.PurePosixPath(name)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError(f"archive member escapes the target directory: {name}")


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
    root = args.dest or pathlib.Path(
        os.environ.get("GIA_MIP_DATA_ROOT", DEFAULT_DEST))

    def local(rel: str, entry: dict) -> pathlib.Path:
        return root / entry["stage"] / rel

    if args.list:
        if not artifacts:
            print("no artifacts match that collection and stage")
        for rel, entry in artifacts.items():
            path = local(rel, entry)
            state = "present" if path.exists() else "absent"
            print(f"{entry['collection']:12s} {entry['stage']:9s} {rel:52s} "
                  f"{entry['bytes'] / 1e6:8.1f} MB  {state}  "
                  f"permission={entry['permission']}")
        return 0

    base = os.environ.get("GIA_MIP_BASE_URL", manifest["base_url"])
    failures = 0
    for rel, entry in artifacts.items():
        path = local(rel, entry)
        if path.exists() and path.stat().st_size == entry["bytes"]:
            digest = sha256(path)
            if digest == entry["sha256"]:
                print(f"ok       {rel}")
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

    if not args.no_unpack and not args.verify:
        for rel, entry in artifacts.items():
            path = local(rel, entry)
            if path.exists() and "members" in entry:
                target = unpack(path)
                print(f"unpack   {rel} -> {target.relative_to(root)}/")

    if failures:
        print(f"\n{failures} artifact(s) unavailable or unverified.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
