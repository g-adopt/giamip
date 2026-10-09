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


# The case fields that a G-ADOPT choice may fill. Topography lives in the
# `topography` block of a case and the other four in its `earth` block. Only a
# field that the protocol leaves to the group, that is, a field listed in the
# case's `open_choices`, may carry a G-ADOPT choice.
# tests/test_gadopt_choices.py checks this, so a G-ADOPT choice can never
# overwrite a value the protocol prescribes.
# Compressibility is open in every experiment (protocol v4.0 Section 3), and
# every G-ADOPT run chooses a compressible earth.
GADOPT_FIELDS = ("viscosity", "elastic_structure", "rheology", "topography",
                 "compressibility")


def gadopt_choices(case_or_id: str | dict) -> dict[str, str]:
    """The values the G-ADOPT team chose for the open fields of a GIAMIP case.

    The protocol leaves some fields of some experiments to each group, for
    example the rheology of Exp02. The G-ADOPT team decided these for its own
    runs. The decisions sit in a `gadopt` block of the case file, next to the
    protocol values, so the case file stays a transcription of the protocol and
    the decisions are recorded in one place only.

    Parameters
    ----------
    case_or_id
        A case id such as ``giamip-exp03``, or a case dict already read.

    Returns
    -------
    dict
        Field name to chosen value, for the fields in `GADOPT_FIELDS` that the
        G-ADOPT team decided. Empty for a case with no `gadopt` block, which
        includes every benchmark case.
    """
    c = case(case_or_id) if isinstance(case_or_id, str) else case_or_id
    block = c.get("gadopt", {})
    # Keys that start with "$" are comments, as elsewhere in the case files.
    return {k: v for k, v in block.items() if not k.startswith("$")}


def protocol_value(c: dict, field: str) -> str:
    """The value the protocol gives for one of `GADOPT_FIELDS` of a case.

    Topography is stored as ``topography.model``; the other four are stored
    in the ``earth`` block. This hides that difference from callers.
    """
    if field == "topography":
        return c["topography"]["model"]
    return c["earth"][field]


def with_gadopt_choices(case_id: str) -> dict:
    """A GIAMIP case with the G-ADOPT choices filled into the open fields.

    This is the case as a G-ADOPT run uses it. The protocol values of the open
    fields are replaced by the G-ADOPT values, and `open_choices` lists only
    the fields that remain open. For a new topography the file is taken from a
    GIAMIP case that the protocol gives that topography, because the file of a
    topography is the same in every experiment that uses it.

    Parameters
    ----------
    case_id
        A GIAMIP case id, for example ``giamip-exp09``.

    Returns
    -------
    dict
        A new dict. The case file itself is not changed.
    """
    c = case(case_id)
    chosen = gadopt_choices(c)
    for field, value in chosen.items():
        if field == "topography":
            c["topography"] = {**c["topography"], "model": value,
                               "file": topography_files()[value]}
        else:
            c["earth"][field] = value
    c["open_choices"] = [f for f in c["open_choices"] if f not in chosen]
    return c


@functools.cache
def topography_files() -> dict[str, str]:
    """Map each prescribed GIAMIP topography model to its input file.

    Collected from the GIAMIP cases that prescribe a topography. Two cases that
    name the same model with different files are an error, since a G-ADOPT
    choice of that model would then have no single file.
    """
    files: dict[str, str] = {}
    for c in cases("giamip"):
        t = c["topography"]
        if t.get("file") is None:
            continue
        if files.setdefault(t["model"], t["file"]) != t["file"]:
            raise ValueError(f"topography {t['model']!r} names two files")
    return files
