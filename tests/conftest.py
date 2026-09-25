"""Shared fixtures.

The only shared state the suite needs is where the third-party originals
live: `GIA_MIP_DATA_ROOT` if it is set, otherwise `data/` in the checkout. The
tests that convert the originals read the unpacked trees below
`<data_root>/original/`.

`gia_mip.data.locate`, which the package's readers use, searches the
checkout's `data/` and the download root (`GIA_MIP_DATA_ROOT`, else the
checkout's `data/`) both. That cannot give a test a different file from this
fixture: `locate` accepts a file only when its SHA-256 is the manifest's, so
two copies it could choose between are identical.
"""

import os
import pathlib

import pytest


@pytest.fixture(scope="session")
def data_root() -> pathlib.Path:
    """Root of the fetched data tree, holding `original/` and `converted/`."""
    root = os.environ.get("GIA_MIP_DATA_ROOT")
    if root:
        return pathlib.Path(root)
    return pathlib.Path(__file__).resolve().parent.parent / "data"
