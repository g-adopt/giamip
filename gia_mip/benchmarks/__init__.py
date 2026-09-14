"""Readers, case set-up, results and scoring for the two published GIA benchmarks.

Spada et al. (2011) tested GIA codes against each other on analytic cap, disc
and point loads, in the wavenumber domain (Love numbers, relaxation spectra,
the polar motion transfer function) and in the spatial domain (surface
displacement, its rate, polar motion and length of day). Martinec et al.
(2018) followed it with five cases of the sea level equation.

Both have published answers from several codes, so a solver is scored against
them. Neither uses the GIAMIP conventions: the ice density is 931 kg/m3 and
the water density is 1000 kg/m3, and the earth model is the incompressible
five-layer M3-L70-V01. Constants come from the case file for exactly this
reason.

Modules
-------
inputs
    `MartinecCase`: the earth model, load, topography and outputs of a case.
results
    `BenchmarkResult`: the curves of one code for one case, read and written
    in the layout of the converted reference data.
metrics
    RMS and maximum differences against the reference, as the paper defines.
martinec2018
    Conversion of Martinec's archives to NetCDF.
"""
