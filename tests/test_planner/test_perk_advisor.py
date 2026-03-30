# tests/test_planner/test_perk_advisor.py
"""Tests for perk advisor."""
from __future__ import annotations

import pytest

from stellaris_advisor.parser.models import EmpireState


@pytest.mark.unit
class TestPerkAdvisor:
    def test_returns_path(self, planner_empire: EmpireState) -> None:
        from stellaris_advisor.planner.perk_advisor import suggest_perk

        result = suggest_perk(planner_empire, game_graph=None, year=2275)
        assert result is None or isinstance(result, str)
