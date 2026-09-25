# Submissions

One directory per submission, named `<group>_<model>`, holding what identifies
it and what was found out about it. The output fields themselves are large and
live in `outputs/`, which is not tracked.

A submission directory holds:

- `identity.json`: the group name, the model name and version, the experiments
  and members covered, and the commit of the solver that produced it. These
  are the fields that go into every output filename, which GIAMIP requires to
  be `[variable]_[experiment]_[member]_[group]_[model].nc`.
- `report.md`: the model report the protocol asks for in Section 9, generated
  from the case files that were run so that it cannot drift from them.
- `compliance/`: the output of the GIAMIP compliance checker, one report per
  experiment. The protocol requires this to be submitted with the files.
- `scores/`: for a benchmark submission, the difference metrics against every
  contributor.

The GIAMIP identity is institute ANU and group G-ADOPT, with two models, `lovejx`
and `g-adopt`. The directories are therefore `G-ADOPT_lovejx` and `G-ADOPT_g-adopt`.
