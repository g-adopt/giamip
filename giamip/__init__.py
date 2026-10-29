"""Case definitions, reference data and output conventions for GIA model comparison.

This package describes and checks work. It never performs it. Nothing here
imports a solver, and nothing here computes a viscoelastic response. A solver
reads a case with `case`, for example ``giamip.case("martinec2018-B")``,
produces output, and uses this package to check
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
    The Gauss-Legendre quadrature grid that GIAMIP prescribes, and a validator
    that checks a file's axes against it.
corrections
    Known defects in the original files, applied when a file is read. The
    original files themselves are never edited.
cases
    Look up a case, an earth model or a load table.
earth
    The radial earth of a GIAMIP case: density, elastic moduli, gravity and
    viscosity at any radius, interpolated from the earth model files.
benchmarks
    Martinec (2018) and Spada (2011) case set-up, results, scores and
    conversion to NetCDF.
mip
    GIAMIP input readers. The output writer and compliance wrapper are not
    written.

Figures are drawn inline in the pages of ``docs/``, so the package holds no
plotting code.
"""

__version__ = "0.3.0"


def case(case_id: str):
    """One case as an object, by id: the entry point for a solver.

    ``martinec2018-*`` gives a `giamip.benchmarks.inputs.MartinecCase`,
    ``spada2011-*`` a `giamip.benchmarks.inputs.SpadaCase` and ``giamip-*`` a
    `giamip.mip.inputs.GiamipExperiment`. The classes are imported here and
    not at the top, so that ``import giamip`` stays light.

        >>> import giamip
        >>> giamip.case("martinec2018-B").terminal_time_kyr
        10.0

    Raises KeyError for an id that is not a case of the package.
    """
    from giamip import cases
    if case_id not in cases.ids():
        raise KeyError(f"no case with id {case_id!r}. Known ids: {', '.join(cases.ids())}")
    collection = case_id.split("-", 1)[0]
    if collection == "martinec2018":
        from giamip.benchmarks.inputs import MartinecCase
        return MartinecCase(case_id)
    if collection == "spada2011":
        from giamip.benchmarks.inputs import SpadaCase
        return SpadaCase(case_id)
    from giamip.mip.inputs import GiamipExperiment
    return GiamipExperiment(case_id)
