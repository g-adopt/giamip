"""Generate `gia_mip/resources/manifest.json` from the files in `data/original/`.

The manifest is the checksum record for every third-party file this project
depends on. It is generated, never hand-edited: the human-written half is
`gia_mip/resources/provenance.json`, which says where each artifact came from,
who supplied it and whether they have agreed to redistribution. This script
joins that provenance to the size and SHA-256 sum measured on disk.

Each artifact also gets an inventory: for an archive, the name, size and
SHA-256 of every member. The inventory has two uses. It lets the data pages
list what a collection contains without unpacking 400 MB, and it lets a reader
verify one file out of an archive that was unpacked some time ago.

Usage::

    python3 scripts/build_manifest.py [--root data/original] [--output gia_mip/resources/manifest.json]

Run it after adding a file to `data/original/` and after adding the matching
entry to `provenance.json`. The script fails if the two disagree, so a file
added without provenance, or provenance written for a file that is not there,
is caught here and not at upload time.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import sys
import tarfile
import zipfile

# The unpack rule lives in gia_mip.data so there is one definition of where an
# archive lands. That module imports only the standard library, so importing
# it here costs nothing and cannot fail on a bare machine.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from gia_mip.data import (  # noqa: E402
    STAMP as UNPACK_STAMP, common_top, member_path, unpack_target,
)

# Read files in one-megabyte pieces so a 300 MB archive is never held whole in
# memory. The same chunk size is used by radopt.data, for no reason beyond
# consistency between the two fetchers.
CHUNK = 1 << 20


def sha256_path(path: pathlib.Path) -> str:
    """SHA-256 of a file on disk, streamed."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_stream(fileobj) -> tuple[str, int]:
    """SHA-256 and byte count of an open binary stream, read to the end.

    Used for archive members, which are decompressed on the fly and never
    written to disk.
    """
    h = hashlib.sha256()
    total = 0
    for chunk in iter(lambda: fileobj.read(CHUNK), b""):
        h.update(chunk)
        total += len(chunk)
    return h.hexdigest(), total


def inventory(path: pathlib.Path) -> list[dict] | None:
    """List the members of an archive with their sizes and sums.

    Returns None for a file that is not an archive. Directory entries are
    skipped: they carry no content and their presence differs between the tar
    and zip formats, which would make two equivalent archives look different.
    """
    suffixes = "".join(path.suffixes[-2:])

    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            members = []
            for info in sorted(z.infolist(), key=lambda i: i.filename):
                if info.is_dir():
                    continue
                with z.open(info) as f:
                    digest, size = sha256_stream(f)
                members.append({"name": info.filename, "bytes": size, "sha256": digest})
            return members

    if suffixes in (".tar.gz", ".tgz") or path.suffix == ".tar":
        # Streaming mode ("r|*") reads the archive once from front to back,
        # which matters for a gzip member: random access would decompress the
        # whole stream again for every file.
        members = []
        with tarfile.open(path, "r|*") as t:
            for info in t:
                if not info.isfile():
                    continue
                f = t.extractfile(info)
                if f is None:
                    continue
                digest, size = sha256_stream(f)
                members.append({"name": info.name, "bytes": size, "sha256": digest})
        members.sort(key=lambda m: m["name"])
        return members

    return None


def carry_history(old: dict | None, artifacts: dict[str, dict]) -> None:
    """Keep the record of every earlier version of a file, in place.

    A supplier can replace a file with a new version under the same name. The
    manifest describes the version on disk, and without this step the sum of
    the version that earlier results were computed from would be overwritten
    and lost. So when a file's sum differs from the previous manifest, the old
    sum, size and member sums are appended to ``previous_versions``, and a
    history already recorded is carried forward unchanged.

    Parameters
    ----------
    old
        The previous manifest, or None when there is none.
    artifacts
        The new manifest entries, which are updated.
    """
    if not old:
        return
    for rel, entry in artifacts.items():
        before = old.get("artifacts", {}).get(rel)
        if before is None:
            continue
        history = list(before.get("previous_versions", []))
        if before["sha256"] != entry["sha256"]:
            record = {"sha256": before["sha256"], "bytes": before["bytes"],
                      "recorded_in_manifest_of": old.get("generated")}
            if before.get("members"):
                record["members"] = [{"name": m["name"], "sha256": m["sha256"]}
                                     for m in before["members"]]
            history.append(record)
        if history:
            entry["previous_versions"] = history


def main() -> int:
    here = pathlib.Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=pathlib.Path, default=here / "data" / "original",
                    help="directory holding the original files")
    ap.add_argument("--output", type=pathlib.Path,
                    default=here / "gia_mip" / "resources" / "manifest.json")
    ap.add_argument("--provenance", type=pathlib.Path,
                    default=here / "gia_mip" / "resources" / "provenance.json")
    args = ap.parse_args()

    provenance = json.loads(args.provenance.read_text())
    declared = set(provenance["artifacts"])

    # One inventory pass. Reading an archive decompresses all of it, and the
    # ice history is 316 MB, so this is done once and the result serves both
    # the manifest entry and the stray check.
    inventories: dict[str, list[dict] | None] = {}
    for rel in sorted(declared):
        path = args.root / rel
        if path.exists():
            inventories[rel] = inventory(path)

    def on_disk_names(rel: str) -> set[str]:
        """Where this archive's members land, relative to its unpack target."""
        members = inventories.get(rel) or []
        top = common_top([m["name"] for m in members])
        return {member_path(m["name"], top) for m in members}

    # Every file under the root is an artifact, with two exclusions. Files
    # inside an unpack target are archive members, already inventoried inside
    # their archive's entry, so counting them again would make every fetched
    # tree look undocumented. Caches and hidden files are never artifacts.
    unpacked = {
        unpack_target(args.root / rel): rel
        for rel in declared
        if (args.root / rel).exists()
    }

    def containing_target(path: pathlib.Path) -> pathlib.Path | None:
        """The unpack target this path sits inside, if any."""
        for target in unpacked:
            if target in path.parents:
                return target
        return None

    def is_noise(path: pathlib.Path) -> bool:
        return (not path.is_file()
                or path.name.startswith(".")
                or "__pycache__" in path.parts)

    found = set()
    strays = []
    for path in sorted(args.root.rglob("*")):
        if is_noise(path):
            continue
        target = containing_target(path)
        if target is None:
            found.add(str(path.relative_to(args.root)))
            continue
        # Inside an unpack target. It is accounted for when it is a member of
        # the archive that target came from, and a stray otherwise. Excluding
        # the whole directory would hide a file someone dropped in there.
        archive_rel = unpacked[target]
        relative = str(path.relative_to(target))
        if relative == UNPACK_STAMP:
            continue
        if relative not in on_disk_names(archive_rel):
            strays.append(str(path.relative_to(args.root)))

    missing = sorted(declared - found)
    undeclared = sorted(found - declared)
    if missing or undeclared or strays:
        for path in missing:
            print(f"declared in provenance but not on disk: {path}", file=sys.stderr)
        for path in undeclared:
            print(f"on disk but not declared in provenance: {path}", file=sys.stderr)
        for path in strays:
            print(f"inside an unpack target but not a member of its archive: "
                  f"{path}", file=sys.stderr)
        return 1

    artifacts = {}
    for rel in sorted(declared):
        path = args.root / rel
        entry = dict(provenance["artifacts"][rel])
        entry["bytes"] = path.stat().st_size
        entry["sha256"] = sha256_path(path)
        members = inventories.get(rel)
        if members is not None:
            entry["members"] = members
            entry["member_count"] = len(members)
        artifacts[rel] = entry
        size_mb = entry["bytes"] / 1e6
        count = entry.get("member_count")
        suffix = f", {count} members" if count else ""
        print(f"{rel:52s} {size_mb:8.1f} MB  {entry['sha256'][:16]}{suffix}")

    carry_history(json.loads(args.output.read_text()) if args.output.exists() else None,
                  artifacts)

    manifest = {
        "generated": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "generated_by": "scripts/build_manifest.py",
        "base_url": provenance["base_url"],
        "artifacts": artifacts,
    }
    args.output.write_text(json.dumps(manifest, indent=1, sort_keys=False) + "\n")

    total = sum(a["bytes"] for a in artifacts.values())
    by_collection: dict[str, int] = {}
    for a in artifacts.values():
        key = f"{a['collection']}/{a['stage']}"
        by_collection[key] = by_collection.get(key, 0) + a["bytes"]
    print(f"\n{len(artifacts)} artifacts, {total / 1e6:.0f} MB total")
    for key, size in sorted(by_collection.items()):
        print(f"  {key}: {size / 1e6:.0f} MB")
    try:
        shown = args.output.relative_to(here)
    except ValueError:
        shown = args.output          # --output outside the repository
    print(f"written to {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
