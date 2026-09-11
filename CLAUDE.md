# Working rules for this repository

## What this repository is

It specifies GIA model comparison cases and holds the data to compare against.
It contains no solver. If a change would make this repository compute a
viscoelastic response, the change belongs in `sl1d` or in G-ADOPT.

## The rules that do not bend

**Original data is never edited.** `data/original/` holds third-party files
byte for byte. Every known defect is an entry in
`gia_mip/resources/corrections.json`, applied by a reader at read time and
rendered on the corrections page. A defect fixed with a text editor is
invisible and is lost at the next re-fetch.

**Every reader turns NetCDF masking off.** All six GIAMIP ice files declare
`_FillValue = 0.0` on `ice_thickness`. A default read masks every ice-free
cell and propagates NaN into loads and mass integrals. Call
`Dataset.set_auto_mask(False)`, and test that you did.

**Physical constants come from the case file, never from a module.** Spada and
Martinec use an ice density of 931 kg/m3 and a water density of 1000 kg/m3.
GIAMIP uses 917 and 1025. One shared constants module gives wrong answers for
one of them without any error.

**`gia_mip/data.py` imports only the standard library.** It runs on a bare CI
runner before anything is installed.

**NOTES/ is never committed and never staged.**

## Naming

The package is `gia_mip`. The project is GIAMIP. The checker publishes scripts
named `giamip_*`. Keep the underscore in the package name so the three stay
apart in a sentence and on the command line.

## Where authority lives

For GIAMIP output conventions, the authority is the compliance checker at
`https://github.com/JanJereczek/GIA_SimulationChecker`, pinned by commit in
`gia_mip/resources/checker.json`. Its `GIAMIP_VARIABLES` table transcribes the
data request spreadsheet. Do not re-type the variable list from the protocol
prose: generate it from the checker and test that the two agree.

For the benchmark column layouts, the authority is the GMT script that plotted
the figure. GMT 4 no longer runs, so those scripts are the only record of what
several data files contain.

## Case identity

A case file states which published case it is. Where the published identity is
not established, the field says so. Kachuck's benchmark submissions use
internal codes whose mapping to the published letters A to E is confirmed for
two of the five. Do not guess the other three.
