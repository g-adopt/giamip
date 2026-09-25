"""Upload the data files of the manifest to the object store behind data.gadopt.org.

`gia_mip.data` fetches every artifact from
https://data.gadopt.org/gia-mip/<object>, where <object> is the artifact's
`object` field when it has one (the converted files:
`versions/<sha256>/<collection>/<file>`) and its manifest key otherwise (the
originals). That host serves the `gadopt` bucket of the G-ADOPT DigitalOcean
Space, so an artifact is uploaded to the key `gia-mip/<object>` with public
read access. A converted file's key contains its sum, so a new version is a
new object and the old one stays for the releases that name it. Nothing is
ever deleted.

Only an artifact whose `permission` in the manifest is `granted` is uploaded,
and only after its SHA-256 on disk matches the manifest, so the object store
never holds a file the manifest does not describe. An object that already
exists with the same size and the same SHA-256 in its metadata is left alone.

Credentials are read from an s3cmd configuration file, by default
~/.s3cfg-gadopt. They are never needed to download: the files are public, and
the continuous integration workflow fetches them over HTTPS with no secret.

Usage::

    python3 scripts/upload_artifacts.py [--stage original|converted] [--collection martinec2018]
                                        [--config ~/.s3cfg-gadopt] [--dry-run]
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY))
from gia_mip import data  # noqa: E402

BUCKET = "gadopt"
PREFIX = "gia-mip/"


def read_s3cmd_config(path: pathlib.Path) -> dict[str, str]:
    """The key = value pairs of an s3cmd configuration file.

    s3cmd writes the file with or without a `[default]` section header, so it
    is parsed line by line and not with configparser.
    """
    values = {}
    for line in path.read_text().splitlines():
        m = re.match(r"\s*(\w+)\s*=\s*(.*?)\s*$", line)
        if m:
            values[m.group(1)] = m.group(2)
    return values


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--collection", action="append",
                    help="collection to upload; repeatable. Default: every granted artifact")
    ap.add_argument("--stage", action="append", choices=data.STAGES,
                    help="stage to upload; repeatable. Default: both")
    ap.add_argument("--config", type=pathlib.Path,
                    default=pathlib.Path("~/.s3cfg-gadopt").expanduser())
    ap.add_argument("--root", type=pathlib.Path, default=REPOSITORY / "data",
                    help="the directory holding original/ and converted/")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import boto3
    from botocore.exceptions import ClientError

    cfg = read_s3cmd_config(args.config)
    s3 = boto3.client("s3", endpoint_url="https://" + cfg["host_base"],
                      aws_access_key_id=cfg["access_key"],
                      aws_secret_access_key=cfg["secret_key"],
                      region_name=cfg["host_base"].split(".")[0])

    manifest = data.load_manifest()
    failures = 0
    for rel, entry in manifest["artifacts"].items():
        if args.collection and entry["collection"] not in args.collection:
            continue
        if args.stage and entry["stage"] not in args.stage:
            continue
        if entry["permission"] != "granted":
            continue
        path = args.root / entry["stage"] / rel
        if not path.exists():
            print(f"MISSING  {rel}", file=sys.stderr)
            failures += 1
            continue
        digest = data.sha256(path)
        if digest != entry["sha256"]:
            print(f"CORRUPT  {rel}: the file on disk does not match the manifest", file=sys.stderr)
            failures += 1
            continue
        key = PREFIX + entry.get("object", rel)
        try:
            head = s3.head_object(Bucket=BUCKET, Key=key)
            if head["ContentLength"] == entry["bytes"] and head["Metadata"].get("sha256") == digest:
                print(f"present  {key}")
                continue
        except ClientError as exc:
            if exc.response["Error"]["Code"] not in ("404", "NoSuchKey", "NotFound"):
                raise
        if args.dry_run:
            print(f"would upload {key} ({entry['bytes'] / 1e6:.1f} MB)")
            continue
        s3.upload_file(str(path), BUCKET, key,
                       ExtraArgs={"ACL": "public-read", "Metadata": {"sha256": digest}})
        print(f"uploaded {key} ({entry['bytes'] / 1e6:.1f} MB)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
