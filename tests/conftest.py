"""Shared fixtures.

The only shared state the suite needs is where the third-party data lives,
which follows the same rule as `gia_mip.data`: `GIA_MIP_DATA_ROOT` if it is
set, otherwise `data/` in the checkout.
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
