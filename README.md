# giamip

Case definitions, reference data and output conventions for glacial isostatic
adjustment (GIA) model comparison.

This repository specifies what to run and holds what to compare against. It
contains no solver. A solver reads a case with `giamip.case`, produces
output, and uses this package to check that the output obeys the conventions
and to score it against the reference data.

Three collections are covered.

| Collection | What it is | Reference answer |
| --- | --- | --- |
| Spada et al. (2011) | Ten tests of GIA codes: Love numbers, relaxation spectra, polar motion, and surface deformation for analytic loads | Published, several codes per test |
| Martinec et al. (2018) | Five cases (A to E) of the sea level equation, from no ocean to a moving coastline | Published, the VEGA solution is the reference |
| GIAMIP | Twelve experiments with global ice histories and present-day topography | None. This is an open intercomparison |

## Install

```
pip install giamip
```

The package holds the code and the small JSON files: the case files, the load
tables, the earth models, the manifest and the corrections. It holds no data
file. Each data file is downloaded from data.gadopt.org the first time a
reader needs it, checked against its SHA-256 sum, and cached. For
development, install a checkout with `pip install -e ".[test]"`.

## How to run a case

```python
import giamip
from giamip.benchmarks.results import BenchmarkResult

case = giamip.case("martinec2018-B")                    # the set-up of benchmark B
h = case.ice_thickness(colatitude, longitude, t_kyr)     # the load at your points
b = case.topography(colatitude, longitude)               # the ocean basin

run = BenchmarkResult("B", "my-code")                    # your curves
for profile in case.profiles:                            # what the case compares
    for quantity in profile.quantities:
        run.add_profile(profile.name, quantity, colatitude_deg,
                        my_curve(profile, quantity))
run.write("martinec2018-B.nc")                           # netCDF-4, through h5py

rows = case.score(run)                                   # eq. 31, on your own points
checks = case.envelope_check(run, band_half_width_deg=6.74)   # criterion 2
```

`giamip.case` gives a `MartinecCase`, a `SpadaCase` or a `GiamipExperiment`.
`tests/test_doc_examples.py` runs this example. For a solver's own benchmark
test, with both stages and the per-case list of what a result must hold, read
the section "Using giamip in a solver's benchmark test" of
`docs/reference/python-interface.qmd`.
The reference curves of a Martinec case are VEGA, and
`docs/protocol/martinec2018.qmd` says how each one is built.

Each case also has a page in `docs/results/`, which gives the full
specification: the earth model, the load, the time interval, the grid, the
physical constants, the variables to produce and the data to compare against.
Start with `docs/reference/solver-requirements.qmd`, which lists what a solver
must be able to do, and `docs/reference/python-interface.qmd`, which shows the
classes.

## Data

The data files are not in the package. The third-party originals (341 MB)
belong to the people who produced them. The converted reference data (10 MB)
is derived from them and is tracked in this repository under
`data/converted/`. Both are in the gadopt object store at data.gadopt.org.
`docs/data/manifest.qmd` records, for every file, its size, its SHA-256 sum,
where it came from, who supplied it and under what terms.

A reader finds a file with `giamip.data.locate`. It looks in the checkout
(when it runs from one) and in the download root, and downloads the file into
the download root when neither has it. The download root is
`$GIAMIP_DATA_ROOT` when it is set, else `data/` of the checkout, else
`~/.cache/giamip`. `docs/data/fetching.qmd` gives the details.

Some of the original files contain defects. `docs/data/corrections.qmd` lists
every one that is known, with the evidence. The readers apply the corrections.
The original files are never edited.

### On Gadi

Gadi compute nodes have no network, and in a job the Firedrake module points
`XDG_CACHE_HOME` at storage that is deleted when the job ends. So:

1. The maintainer of the Firedrake module installs giamip with its
   dependencies h5netcdf and `packaging` (h5py, numpy and scipy are there).
2. On a login node, fill a directory on `/g/data` once:
   `GIAMIP_DATA_ROOT=/g/data/<project>/giamip python -m giamip.data --stage converted --collection benchmarks`.
   For the GIAMIP inputs (363 MB), run the same command with
   `--collection giamip` and without `--stage converted`: they are originals.
3. In the job script, set `GIAMIP_DATA_ROOT` to the same directory.

A reader on a compute node that does not find a file raises
`DataUnavailable`, and the message gives the command of step 2.

## Layout

```
giamip/            the package: data.py (locate, fetch and verify),
                    grids.py (the Gauss-Legendre grid), corrections.py
                    (known defects in the originals), benchmarks/ (Martinec
                    and Spada case set-up, results, scores and conversion)
                    and mip/ (GIAMIP input readers). Not written: the GIAMIP
                    output writer
giamip/resources/  the JSON files the package carries: cases/ (one file per
                    case, in benchmarks/ and giamip/), loads/ (the analytic
                    load tables), models/ (the earth models), the manifest,
                    the provenance record, the corrections and the column
                    layouts
docs/               the Quarto site: one page per collection, per reference
                    table and per case, plus the data manifest and the list
                    of corrections. docs/cases/ holds the pages for single
                    tests as percent-format Python files, which the site
                    executes and plots
scripts/            conversion, manifest and upload scripts, run by hand, and
                    check_installed.py, which checks an installed wheel
submissions/        one directory per submission: its identity, its
                    compliance report and its scores. The output fields
                    themselves stay in outputs/
tests/              the test suite
data/               original/ holds third-party files byte for byte and is
                    gitignored; converted/ holds the netCDF-4 files built
                    from them and is tracked; versions/ holds downloaded
                    converted files by SHA-256 and is gitignored
outputs/            gitignored. The output fields a solver produced for a case
```

## Correspondence about the GIAMIP inputs

We raised two defects in the GIAMIP inputs, and the instability of PREM, with
the GIAMIP organisers. The correspondence is not in this repository, so this
section records the result.

**GLAC3 version names** (correction `glac3-version-naming`). The protocol
table gives the ice history of Exp03, Exp04 and Exp09 to Exp11 as GLAC3b. The
delivered files are `iceHistory-GLAC3c_profile1.nc` to
`iceHistory-GLAC3c_profile3.nc`. On 2026-09-12, Holly Han, the lead of the
protocol, confirmed that these files are the intended inputs. Each file
combines GLAC3 versions a, b and c, with different versions for different ice
sheets. The case files give the ice history as `GLAC3_p1`, `GLAC3_p2` or
`GLAC3_p3`, with no version letter.

**Topography source attribute** (correction `topography-source-attribute`).
In the first version of `bedtopo_BedMach3ANT_BedMach5GRL_GEBCO_GLQ256.nc`, the
`source` attribute names BedMachine Greenland twice. The first of the two
entries must name BedMachine Antarctica v3. We reported this defect to Jan
Swierczek-Jereczek on 2026-09-12. The organisers replaced the file on
2026-09-24 and announced it on 2026-10-07. The new version names BedMachine
Antarctica v3.4, and its topography is identical to the first version. The
correction lists the new sum as checked and clean.

**PREM** (protocol v4.0 Section 3.1). GIAMIP gave no PREM file, and PREM is
gravitationally unstable where the mantle flows. The G-ADOPT team proposed
PREM-stable, and the organisers adopted it in protocol v4.0 of 2026-10-05 as
the PREM of every experiment that uses PREM. The file is
`data/converted/giamip/PREM-stable.csv`, copied unchanged from Zenodo
(doi:10.5281/zenodo.23178246).

## Licence

MIT for the code, the case definitions and the documentation. The third-party
data in `data/original/` keeps the terms of whoever supplied it. See
`LICENSE-DATA.md`.
