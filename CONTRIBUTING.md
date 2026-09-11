# Contributing

## Before you change anything

Read [AGENTS.md](AGENTS.md). It lists six rules, and breaking any of them
produces a wrong answer with no error message.

## Running the tests

```bash
pip install -e ".[test]"
pytest -q
```

The default run needs no data. Tests marked `data` need the third-party
originals in `data/original/` and are excluded until they are present:

```bash
python -m gia_mip.data --all
pytest -q -m data
```

## Adding a data file

Add the file to `data/original/`, add an entry to
`gia_mip/resources/provenance.json` saying where it came from, who supplied
it, when, and whether they have agreed to redistribution, then run:

```bash
python3 scripts/build_manifest.py
```

The script fails if a file has no provenance or if provenance names a file
that is not there, so the two cannot drift apart. Commit the regenerated
`manifest.json`.

Never edit a file in `data/original/`. If it has a defect, add an entry to
`gia_mip/resources/corrections.json` with the evidence, and apply it in the
reader.

## Adding a case

A case is one JSON file in `cases/` and one page in `docs/results/`. The JSON
is validated against a schema in `gia_mip/schemas/` by the test suite. State
the physical constants in the case file: they differ between the benchmarks
and GIAMIP, and a shared module would silently give one of them the other's
values.

## Building the site

```bash
pip install -e ".[docs]"
quarto render
```

Pages execute from the committed `_freeze/` cache. A page that needs the
third-party originals or a solver run is re-executed only on a machine that
has them:

```bash
quarto render docs/path/to/page.qmd --no-freeze
```

Commit the updated `_freeze/` with the page.

## Reporting a problem with the data

Defects in the original files are expected, and finding one is useful. Open an
issue with the file, the field, what the file says, what you believe is true,
and how you know. It becomes an entry in the corrections table.
