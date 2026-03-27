"""Shared test fixtures for stellaris-advisor."""

import pytest

from tests.fixtures.create_test_save import create_test_save


@pytest.fixture(autouse=True, scope="session")
def _generate_test_save() -> None:
    """Auto-generate the synthetic .sav fixture before any tests run."""
    create_test_save()
