# tests/test_planner/test_tradition_advisor.py
"""Tests for tradition advisor."""
from __future__ import annotations

import pytest

from stellaris_advisor.parser.models import EmpireState


@pytest.mark.unit
class TestTraditionAdvisor:
    def test_returns_path(self, planner_empire: EmpireState) -> None:
        from stellaris_advisor.planner.models import TraditionPerkPath
        from stellaris_advisor.planner.tradition_advisor import suggest_tradition_path

        result = suggest_tradition_path(planner_empire, game_graph=None, year=2275)
        assert isinstance(result, TraditionPerkPath)
        assert result.traditions_completed == 0
        assert result.unity_income > 0
