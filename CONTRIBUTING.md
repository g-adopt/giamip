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

A case is one JSON file in `cases/`, shown on a page in `docs/results/`.
`tests/test_cases.py` checks that it names an earth model and a load table that
exist. State the physical constants in the case file: they differ between the benchmarks
and GIAMIP, and a shared module would silently give one of them the other's
values.

## Building the site

```bash
pip install -e ".[docs]"
quarto render
```

A render executes the code of every page. A local render writes the execution
cache `_freeze/` and reuses it on the next render. `_freeze/` is gitignored, so
CI executes every page from the checked-out files, and a page must read only
tracked files. To execute every page again locally, ignore the cache:

```bash
quarto render --no-freeze
```

## Adding a page for a test

A page for a single test is a Python file in the percent format of jupytext, at
`docs/cases/<collection>/<case>.py`. A `# %% [markdown]` line starts a text
cell, and a `# %%` line starts a code cell. The first cell is a `# %% [raw]`
cell with the YAML front matter, which gives the title.

Quarto renders the file directly, as the HTML page and as an executed notebook
that the page links to. `docs/cases/_metadata.yml` holds these settings. The
pages execute on every render, on CI too, so a page reads only tracked files.

To add a page:

1. Copy `docs/cases/spada2011/t02-01.py` and change the case.
2. Add the file to the "Cases" menu in `_quarto.yml`.
3. Run `pytest -q tests/test_pages.py`. The test converts each page with
   jupytext and executes it in a Jupyter kernel.
4. To see the page, run `quarto preview docs/cases/<collection>/<case>.py`.

## Reporting a problem with the data

Defects in the original files are expected, and finding one is useful. Open an
issue with the file, the field, what the file says, what you believe is true,
and how you know. It becomes an entry in the corrections table.
