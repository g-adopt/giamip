"""Known defects in the original data files, and the rules for reading round them.

The original files are never edited. Every defect that has been found is an
entry in ``giamip/resources/corrections.json``, and a reader consults this
module instead of carrying the knowledge in its own head. Three things follow:
a correction is visible on the website, it survives a re-fetch of the
originals, and a reader that forgets one is a test failure and not a silently
wrong number.

Entries have a ``kind``:

``metadata``
    An attribute states something untrue. The file's data is fine.
``trap``
    The file is correct, and a natural way of reading it gives a wrong answer.
    The ice files' fill value of zero is the expensive one.
``truncated``
    Content is missing.
``superseded``
    A better copy of the same submission exists somewhere else.
``naming``
    The same thing carries different names in different places.

and a ``status``: ``confirmed`` when the evidence is reproducible from what is
in this repository, ``unverified`` when it rests on something outside it.

Files that change under the same name
-------------------------------------
A supplier can replace a file with a fixed version that keeps its name. A
correction that describes the content of a file therefore names the versions
it was found in, by SHA-256, in ``sha256``. It applies to those versions and to
no other, so a repaired file is never "corrected" a second time. Versions that
someone checked and found free of the defect go in ``checked_clean_sha256``.
When a new version arrives, ``tests/test_corrections.py`` fails until every
pinned correction names it in one of the two lists, which is the moment a
person looks at the new file. A correction that describes a safe way of
reading, such as turning NetCDF masking off, has no pins and applies to every
version.
"""

from __future__ import annotations

import fnmatch
import functools
import json
import pathlib

RESOURCE = pathlib.Path(__file__).resolve().parent / "resources" / "corrections.json"

KINDS = ("metadata", "trap", "truncated", "superseded", "naming")
STATUSES = ("confirmed", "unverified")


@functools.cache
def load() -> list[dict]:
    """Every correction, in the order they are recorded.

    Cached: the file is small, it never changes while a process runs, and a
    reader may consult it once per file it opens.
    """
    return json.loads(RESOURCE.read_text())["corrections"]


def by_id(correction_id: str) -> dict:
    """One correction, by its id. Raises `KeyError` if there is no such entry.

    A reader that applies a correction names the id, so that removing an entry
    from the table breaks the reader instead of leaving dead code behind.
    """
    for entry in load():
        if entry["id"] == correction_id:
            return entry
    raise KeyError(
        f"no correction with id {correction_id!r}. Known ids: "
        f"{', '.join(e['id'] for e in load())}"
    )


def applying_to(relative_path: str, sha256: str | None = None) -> list[dict]:
    """Corrections whose `applies_to` patterns match a path.

    Parameters
    ----------
    relative_path
        A path relative to ``data/original/``, using forward slashes, for
        example ``giamip/ice-history/iceHistory-ICE-6G_D.nc``.
    sha256
        The SHA-256 of the file being read. Required when a matching
        correction is pinned to file versions, because the answer depends on
        which version it is; a pinned correction applies only when this sum is
        in its ``sha256`` list.

    Raises
    ------
    ValueError
        If a matching correction is pinned and no ``sha256`` is given.

    Notes
    -----
    Matching is by shell glob, so a pattern like ``giamip/ice-history/*.nc``
    covers all six ice files. ``fnmatch`` treats ``*`` as matching a slash as
    well, which is loose, and every pattern in the table names its directory
    explicitly so that looseness never comes into play.
    """
    found = []
    for entry in load():
        if not any(fnmatch.fnmatch(relative_path, pattern)
                   for pattern in entry["applies_to"]):
            continue
        pins = entry.get("sha256")
        if pins is not None:
            if sha256 is None:
                raise ValueError(
                    f"{relative_path}: correction {entry['id']} applies only to "
                    f"named versions of this file. Pass the file's sha256.")
            if sha256 not in pins:
                continue
        found.append(entry)
    return found


def fortran_exponents(text: str) -> str:
    """Rewrite Fortran D exponents as E exponents so a parser accepts them.

    Applies correction ``benchmarkA-fortran-exponent``. Only a ``D`` or ``d``
    sitting between a digit and a sign or digit is rewritten, so a ``D`` that
    is part of a word in a header line is left alone.
    """
    import re

    by_id("benchmarkA-fortran-exponent")  # fail loudly if the entry is removed
    return re.sub(r"(?<=[0-9.])[Dd](?=[+-]?[0-9])", "E", text)
