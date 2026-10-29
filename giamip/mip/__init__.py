"""Readers, case schema, output writer and compliance wrapper for GIAMIP.

GIAMIP is an open intercomparison with no reference answer, so this subpackage
does something different from `giamip.benchmarks`: instead of scoring output
against a published result, it specifies the twelve experiments, reads the
prescribed ice histories and topographies, and writes and validates output in
the form the project requires.

The authority on output format is the compliance checker the protocol
mandates, not the protocol prose. See AGENTS.md.

Modules
-------
inputs
    `GiamipExperiment`, `IceHistory` and `Topography`: the prescribed forcing,
    read with masking off, on the checked grid, with the corrections for the
    version of each file.
export
    `ExportFile`: the intermediate file of the G-ADOPT export (the surface
    fields at the Gauss-Legendre nodes and the scalars of the native mesh),
    checked against its layout.
outputs
    `write_experiment`: the GIAMIP NetCDF files of one experiment from an
    intermediate file, with the time interpolation of the schedule-D runs,
    the Stokes coefficients and every attribute the checker asks for.
"""
