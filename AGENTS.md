# Working rules for this repository

## What this repository is

It specifies GIA model comparison cases and holds the data to compare against.
The case files, load tables and earth models are in `giamip/resources/`
(`cases/`, `loads/`, `models/`), so an installed package carries them.
It contains no solver. If a change would make this repository compute a
viscoelastic response, the change belongs in `lovejx` or in G-ADOPT.

## The rules that do not bend

Each of these prevents a wrong answer that arrives with no error message.

**Original data is never edited.** `data/original/` holds third-party files
byte for byte. Every known defect is an entry in
`giamip/resources/corrections.json`, applied by a reader at read time and
rendered on the corrections page. A defect fixed with a text editor is
invisible, and it is lost at the next re-fetch.

**Every reader in the package opens NetCDF through h5netcdf.** All eight GIAMIP
ice files declare `_FillValue = 0.0` on `ice_thickness`. A masking read hides
every ice-free cell and propagates NaN into loads and mass integrals. h5netcdf,
on h5py, applies no mask, scale or fill value. No module of the package
imports netCDF4 or xarray; `tests/test_netcdf_io.py` checks this and checks
that a value equal to `_FillValue` comes back unchanged through every reader.
Tests may use netCDF4 to check that netCDF-C reads the files.

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

**A correction about file content names the file versions.** The GIAMIP
inputs change under the same name. A correction that describes what a file
contains lists the SHA-256 of each version it was found in, and a reader passes
the sum of the file it opens. A new version fails the test suite until someone
has checked every pinned correction against it. See
`docs/data/corrections.qmd`.

**`giamip/data.py` imports only the standard library.** It runs on a bare CI
runner before anything is installed, and `tests/test_import.py` checks this by
reading the source.

**No data file goes into the package.** The wheel holds the code and the JSON
files under `giamip/resources/`. Every data file, original or converted, is
found through `giamip.data.locate`, which downloads it from data.gadopt.org
on first use and checks its SHA-256. A converted file that changes gets a new
object key, `versions/<sha256>/<key>`, and is uploaded with
`scripts/upload_artifacts.py`; the old object stays for the releases that name
it.

## Naming

The package is `giamip`, on PyPI and on import. The project is GIAMIP, and the
case collection of its experiments is `giamip` (`giamip.case("giamip-exp01")`).
The compliance checker publishes scripts named `giamip_*`. The repository is
`g-adopt/giamip`, and the object store prefix is
`https://data.gadopt.org/giamip/`. Use one name, `giamip`, for the package,
the repository and the store.

## Where authority lives

For GIAMIP output conventions, the authority is the compliance checker at
`https://github.com/JanJereczek/GIA_SimulationChecker`, pinned to a commit
recorded in `giamip/resources/checker.json`. Its
`GIAMIP_VARIABLES` table transcribes the data request spreadsheet, and its
`variables.md` documents it. Do not re-type the variable list from the
protocol prose: generate it from the checker and test that the two agree.

For the benchmark column layouts, the authority is the GMT script that plotted
the figure. GMT 4 no longer runs, so for several data files those scripts are
the only record of what the columns contain.

## Documents reproduced on the site

Some pages reproduce a supplied document, for example the GIAMIP protocol in
`docs/protocol/`. Such a page keeps the meaning of the document. It does not
have to keep the exact words.

If the text of such a page has a mistake that does not change its meaning,
correct it on the page. Examples are a typo, an exponent that lost its
superscript in the conversion from Word, and broken Markdown. Do not record
these corrections anywhere else.

A change of meaning is not a correction of the text. If a value, a name or an
instruction in the document is wrong, record it in
`giamip/resources/corrections.json`. A choice that the G-ADOPT team made goes
into the `gadopt` block of the case file. The page then shows the change from
that file, so the change stays visible and has one source.

This section applies to the pages only. The files in `data/original/` keep the
rule above and are never edited.

## Case identity

A case file states which published case it is, and records the evidence for
any internal code. The benchmark files use internal codes C2, D3, E2 and F1
for the published cases B, C, D and E. B rests on byte-identical files, C and
D on the titles of Martinec's own figure scripts, and E on Martinec's archive
`resultsE_2026.zip`, which holds only F1 files. Where a new identity is not
established in the same way, the field says so. Do not guess.

## Reference data of the Spada benchmark

The archive `spada2011/spada2011-cost-es0701.tar.gz` is the reference data of
Spada et al. (2011). It is the most complete record that could be obtained.
Sia Ghelichkhani decided this on 2026-09-15 and closed the gaps in it as they
stand. Each case file says so in `reference_data.reference_status`, and
`docs/collections/spada2011.qmd` lists the gaps.

Do not fill a gap from another source, for example by typing values from the
tables of the paper or by reading them off a figure. If a contributor supplies
new data, add it as a new original with its provenance and corrections. Then
update the case file and the collection page.

## Redistribution

`giamip/resources/provenance.json` records, per file, whether the supplier
has agreed to redistribution through the object store. Nothing whose
`permission` is not `granted` is uploaded, and neither `python -m giamip.data`
nor `giamip.data.locate` downloads it. Do not work around this by hard-coding a URL.

## Working notes

`NOTES/` is a private working record. It is never committed and never staged.
Tracked documents may cite a path inside it.
