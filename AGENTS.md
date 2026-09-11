# Working rules for this repository

## What this repository is

It specifies GIA model comparison cases and holds the data to compare against.
It contains no solver. If a change would make this repository compute a
viscoelastic response, the change belongs in `sl1d` or in G-ADOPT.

## The rules that do not bend

Each of these prevents a wrong answer that arrives with no error message.

**Original data is never edited.** `data/original/` holds third-party files
byte for byte. Every known defect is an entry in
`gia_mip/resources/corrections.json`, applied by a reader at read time and
rendered on the corrections page. A defect fixed with a text editor is
invisible, and it is lost at the next re-fetch.

**Every reader turns NetCDF masking off.** All six GIAMIP ice files declare
`_FillValue = 0.0` on `ice_thickness`. A default read masks every ice-free
cell and propagates NaN into loads and mass integrals. Call
`Dataset.set_auto_mask(False)`, and test that you did.

**Physical constants come from the case file, never from a module.** Spada and
Martinec use an ice density of 931 kg/m3 and a water density of 1000 kg/m3.
GIAMIP uses 917 and 1025. One shared constants module gives a wrong answer for
one of them and reports nothing.

**The GIAMIP grid has non-uniform latitudes.** It is a Gauss-Legendre
quadrature grid, 257 latitudes by 513 longitudes. The spacing runs from 0.693
degrees at the pole to 0.699 at the equator. No reader and no integral may
treat it as an equal-angle grid, and cell area is the quadrature weight, never
`cos(lat) * dlat * dlon`. A mass integral computed the wrong way is close
enough to look correct.

**Signs and datums are stated, never assumed.** Several benchmark contributors
store the geoid with the opposite sign to others, and the plotting scripts
negate some of them. Every quantity in every file has its sign and its datum
recorded in `docs/reference/conventions.qmd` and in the column layout table. A
solver that guesses gives a result that is wrong by a factor of minus one and
looks plausible.

**`gia_mip/data.py` imports only the standard library.** It runs on a bare CI
runner before anything is installed, and `tests/test_import.py` checks this by
reading the source.

## Naming

The package is `gia_mip`. The project is GIAMIP. The compliance checker
publishes scripts named `giamip_*`. Keep the underscore in the package name so
the three stay apart in a sentence and on the command line.

## Where authority lives

For GIAMIP output conventions, the authority is the compliance checker at
`https://github.com/JanJereczek/GIA_SimulationChecker`, pinned to a commit
recorded in `gia_mip/resources/checker.json`. Its
`GIAMIP_VARIABLES` table transcribes the data request spreadsheet, and its
`variables.md` documents it. Do not re-type the variable list from the
protocol prose: generate it from the checker and test that the two agree.

For the benchmark column layouts, the authority is the GMT script that plotted
the figure. GMT 4 no longer runs, so for several data files those scripts are
the only record of what the columns contain.

## Case identity

A case file states which published case it is. Where the published identity is
not established, the field says so. Kachuck's benchmark submissions use
internal codes whose mapping to the published letters A to E is confirmed for
two of the five. Do not guess the other three.

## Redistribution

`gia_mip/resources/provenance.json` records, per file, whether the supplier
has agreed to redistribution through the object store. Nothing whose
`permission` is not `granted` is uploaded, and `python -m gia_mip.data`
refuses to download it. Do not work around this by hard-coding a URL.

## Working notes

`NOTES/` is a private working record. It is never committed and never staged.
Tracked documents may cite a path inside it.
