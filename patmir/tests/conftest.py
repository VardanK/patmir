"""Test-suite isolation from the ambient environment.

`$PATMIR_PROFILE` is machine-scoped and normally exported from a shell profile —
`cuda` on the WSL workstation, unset on the Mac. That made the test suite
environment-dependent: `test_paths.py` builds minimal configs with no
`[pipeline.profiles]` table, so with the variable exported every one of them
failed on profile resolution rather than on what it was testing. Green on one
machine, five failures on another, same commit.

Any test that needs a profile sets it explicitly (see
`test_config_profiles.ProfileFixture`), so clearing it here costs nothing and
removes a whole class of "works on my machine".
"""
from __future__ import annotations

import os

import pytest

from patmir.pipeline.config import PROFILE_ENV

# At collection time, so it is gone before any module-level config load.
os.environ.pop(PROFILE_ENV, None)


@pytest.fixture(autouse=True)
def _no_ambient_profile():
    """Belt and braces: also clear it per test, and restore afterwards, so a
    test that sets it and dies mid-way cannot leak into its neighbours."""
    saved = os.environ.pop(PROFILE_ENV, None)
    try:
        yield
    finally:
        if saved is not None:
            os.environ[PROFILE_ENV] = saved
        else:
            os.environ.pop(PROFILE_ENV, None)
