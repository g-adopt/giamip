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
from gia_mip.data import unpack_target  # noqa: E402

# Read files in one-megabyte pieces so a 300 MB archive is never held whole in
# memory. The same chunk size is used by sl1d.data, for no reason beyond
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

    # Every file under the root is an artifact, with two exclusions. Files
    # inside an unpack target are archive members, already inventoried inside
    # their archive's entry, so counting them again would make every fetched
    # tree look undocumented. Caches and hidden files are never artifacts.
    unpacked = {
        unpack_target(args.root / rel)
        for rel in declared
        if (args.root / rel).exists()
    }

    def is_artifact(path: pathlib.Path) -> bool:
        if not path.is_file() or path.name.startswith("."):
            return False
        for part in path.parts:
            if part == "__pycache__":
                return False
        return not any(d in path.parents for d in unpacked)

    found = {
        str(p.relative_to(args.root))
        for p in sorted(args.root.rglob("*"))
        if is_artifact(p)
    }

    missing = sorted(declared - found)
    undeclared = sorted(found - declared)
    if missing or undeclared:
        for path in missing:
            print(f"declared in provenance but not on disk: {path}", file=sys.stderr)
        for path in undeclared:
            print(f"on disk but not declared in provenance: {path}", file=sys.stderr)
        return 1

    artifacts = {}
    for rel in sorted(declared):
        path = args.root / rel
        entry = dict(provenance["artifacts"][rel])
        entry["bytes"] = path.stat().st_size
        entry["sha256"] = sha256_path(path)
        members = inventory(path)
        if members is not None:
            entry["members"] = members
            entry["member_count"] = len(members)
        artifacts[rel] = entry
        size_mb = entry["bytes"] / 1e6
        count = entry.get("member_count")
        suffix = f", {count} members" if count else ""
        print(f"{rel:52s} {size_mb:8.1f} MB  {entry['sha256'][:16]}{suffix}")

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
    print(f"written to {args.output.relative_to(here)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
