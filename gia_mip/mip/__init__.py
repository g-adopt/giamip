"""Readers, case schema, output writer and compliance wrapper for GIAMIP.

GIAMIP is an open intercomparison with no reference answer, so this subpackage
does something different from `gia_mip.benchmarks`: instead of scoring output
against a published result, it specifies the twelve experiments, reads the
prescribed ice histories and topographies, and writes and validates output in
the form the project requires.

The authority on output format is the compliance checker the protocol
mandates, not the protocol prose. See AGENTS.md.
"""
