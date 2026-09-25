"""Look up a case, an earth model or a load definition.

This is the reference interface: it answers "what is benchmark A" and "what
are the layers of M3-L70-V01". It reads no data files and runs nothing. The
JSON it returns is the specification, and the documentation pages are
generated from the same JSON, so a page and a case cannot disagree.

    >>> from giamip import cases
    >>> a = cases.case("martinec2018-A")
    >>> a["specification"]["ice_height_model"]
    'L0'
    >>> cases.earth_model(a["earth_model"])["layers"][0]["thickness_km"]
    70

Cases live in `giamip/resources/cases/`, earth models in
`giamip/resources/models/` and the analytic load tables in
`giamip/resources/loads/`. They are inside the package so that an installed
giamip carries them: a solver on a compute node with no network reads a case
without a clone of the repository. They are small JSON files, and people read
them on the repository page at those paths. The data the cases refer
to is not in the package; `giamip.data` fetches it and caches it.
"""

from __future__ import annotations

import functools
import json
import pathlib

# Resolved from this file, so the lookup is the same in a checkout, in a
# worktree put first on PYTHONPATH, in an editable install and in a wheel.
RESOURCES = pathlib.Path(__file__).resolve().parent / "resources"
CASE_DIRS = (RESOURCES / "cases" / "benchmarks" / "spada2011",
             RESOURCES / "cases" / "benchmarks" / "martinec2018",
             RESOURCES / "cases" / "giamip")
MODEL_DIR = RESOURCES / "models"
LOAD_DIR = RESOURCES / "loads"


def _read(path: pathlib.Path) -> dict:
    with open(path) as f:
        return json.load(f)


@functools.cache
def _index() -> dict[str, pathlib.Path]:
    """Map every case id to the file that holds it.

    Built once. A duplicate id is an error here and not a silent overwrite,
    because two cases sharing an id would make `case()` return whichever was
    read last.
    """
    found: dict[str, pathlib.Path] = {}
    for directory in CASE_DIRS:
        for path in sorted(directory.glob("*.json")):
            case_id = _read(path)["id"]
            if case_id in found:
                raise ValueError(
                    f"two cases share the id {case_id!r}: {found[case_id]} "
                    f"and {path}")
            found[case_id] = path
    return found


def ids(collection: str | None = None) -> list[str]:
    """Every case id, or those of one collection.

    Parameters
    ----------
    collection
        ``spada2011``, ``martinec2018`` or ``giamip``. None gives all of them.
    """
    all_ids = sorted(_index())
    if collection is None:
        return all_ids
    return [i for i in all_ids if case(i)["collection"] == collection]


def case(case_id: str) -> dict:
    """One case by id, for example ``martinec2018-A`` or ``giamip-exp01``."""
    index = _index()
    if case_id not in index:
        raise KeyError(
            f"no case with id {case_id!r}. Known ids: {', '.join(sorted(index))}")
    return _read(index[case_id])


def cases(collection: str | None = None) -> list[dict]:
    """Every case, or every case of one collection, in id order."""
    return [case(i) for i in ids(collection)]


def earth_model(model_id: str) -> dict:
    """One earth model by id, for example ``m3-l70-v01`` or ``vm5a``."""
    path = MODEL_DIR / f"{model_id}.json"
    if not path.exists():
        known = sorted(p.stem for p in MODEL_DIR.glob("*.json"))
        raise KeyError(f"no earth model {model_id!r}. Known: {', '.join(known)}")
    return _read(path)


def earth_models() -> list[dict]:
    """Every earth model, in id order."""
    return [_read(p) for p in sorted(MODEL_DIR.glob("*.json"))]


def loads(collection: str) -> dict:
    """The analytic load table of one collection, ``spada2011`` or ``martinec2018``."""
    path = LOAD_DIR / f"{collection}.json"
    if not path.exists():
        known = sorted(p.stem for p in LOAD_DIR.glob("*.json"))
        raise KeyError(f"no load table for {collection!r}. Known: {', '.join(known)}")
    return _read(path)
