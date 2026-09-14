"""Case definitions, reference data and output conventions for GIA model comparison.

This package describes and checks work. It never performs it. Nothing here
imports a solver, and nothing here computes a viscoelastic response. A solver
reads a case from ``cases/``, produces output, and uses this package to check
that the output obeys the conventions and to score it against the published
reference data.

Three collections are covered, and they are different in kind:

* Spada et al. (2011), Geophys. J. Int. 185, 106-132, doi:10.1111/j.1365-246X.2011.04952.x
  Ten tests of GIA codes on analytic loads. Published answers from several codes.
* Martinec et al. (2018), Geophys. J. Int. 215, 389-414, doi:10.1093/gji/ggy280
  Five cases of the sea level equation. The VEGA solution is the reference.
* GIAMIP, an open intercomparison with twelve experiments and no reference answer.

Submodules
----------
data
    Fetch the original files from the object store and verify them by SHA-256.
    Imports only the standard library, so it runs before anything is installed.
grids
    Planned. The Gauss-Legendre quadrature grid that GIAMIP prescribes, and a
    validator that checks a file's axes against it.
corrections
    Planned. Known defects in the original files, applied when a file is read.
    The original files themselves are never edited.
benchmarks
    Readers for the two benchmarks. ``benchmarks.martinec2018`` converts the
    Martinec (2018) archives to NetCDF. Spada (2011) readers are planned.
mip
    Planned. Readers, case schema, output writer and compliance wrapper for
    GIAMIP.
plotting
    Planned. Redraws of the published figures, and the comparison figures.
"""

__version__ = "0.1.0.dev0"
