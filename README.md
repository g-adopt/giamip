# gia-mip

Case definitions, reference data and output conventions for glacial isostatic
adjustment (GIA) model comparison.

This repository specifies what to run and holds what to compare against. It
contains no solver. A solver reads a case file from `cases/`, produces output,
and uses this package to check that the output obeys the conventions and to
score it against the reference data.

Three collections are covered.

| Collection | What it is | Reference answer |
| --- | --- | --- |
| Spada et al. (2011) | Ten tests of GIA codes: Love numbers, relaxation spectra, polar motion, and surface deformation for analytic loads | Published, several codes per test |
| Martinec et al. (2018) | Five cases (A to E) of the sea level equation, from no ocean to a moving coastline | Published, the VEGA solution is the reference |
| GIAMIP | Twelve experiments with global ice histories and present-day topography | None. This is an open intercomparison |

## Install

```
pip install -e .
python -m gia_mip.data --collection benchmarks
```

The second command downloads the original data files into `data/original/`
and verifies every one against a SHA-256 index, so a fetched tree is the tree
the conversions were made from. Use `--collection giamip` for the GIAMIP
inputs, which are larger.

The files are held in the gadopt object store at data.gadopt.org. The
fetcher downloads only files whose supplier has agreed to redistribution,
which today is every file: the Martinec and Spada archives, Kachuck's giapy
files and the GIAMIP inputs.
`docs/data/fetching.qmd` describes the fetcher, and `docs/data/manifest.qmd` records the
state of every file.

## Layout

```
gia_mip/     the package: data.py (fetch and verify), grids.py (the
             Gauss-Legendre grid), corrections.py (known defects in the
             originals), benchmarks/ (Martinec case set-up, results, scores
             and conversion) and mip/ (GIAMIP input readers). Not written:
             Spada readers, the GIAMIP output writer and plotting/
cases/       one JSON file per case, in benchmarks/ and giamip/
models/      earth-model parameter tables referenced by the cases
docs/        the Quarto site: one page per collection, per reference table and
             per case, plus the data manifest and the list of corrections.
             docs/cases/ holds the pages for single tests as percent-format
             Python files, which the site executes and plots
scripts/     conversion and upload scripts, run by hand
submissions/ one directory per submission: its identity, its compliance
             report and its scores. The output fields themselves stay in
             outputs/
tests/       the test suite
data/        original/ holds third-party files byte for byte and is
             gitignored; converted/ holds the NetCDF4 built from them and is
             tracked, so the tests and the site need no network
outputs/     gitignored. The output fields a solver produced for a case
```

## How to run a case

Each case has a JSON file in `cases/` and a page in `docs/results/`. The page
gives the full specification: the earth model, the load, the time interval, the
grid, the physical constants, the variables to produce and the data to compare
against. The JSON file gives the same information in machine-readable form.

Start with `docs/reference/solver-requirements.qmd`, which lists what a solver
must be able to do, and `docs/reference/python-interface.qmd`, which shows how
a solver reads a case and writes a result that can be scored.

## Data

The original data files are not in this repository. They belong to the people
who produced them and they are large. `docs/data/manifest.qmd` records, for
every file, its size, its SHA-256 sum, where it came from, who supplied it and
under what terms.

Some of the original files contain defects. `docs/data/corrections.qmd` lists
every one that is known, with the evidence. The readers apply the corrections.
The original files are never edited.

## Licence

MIT for the code, the case definitions and the documentation. The third-party
data in `data/original/` keeps the terms of whoever supplied it. See
`LICENSE-DATA.md`.
