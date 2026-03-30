# tests/test_planner/test_faction_advisor.py
"""Tests for faction advisor."""
from __future__ import annotations

import pytest

from stellaris_advisor.parser.models import EmpireState


@pytest.mark.unit
class TestFactionAdvisor:
    def test_returns_summary(self, planner_empire: EmpireState) -> None:
        from stellaris_advisor.planner.faction_advisor import analyze_factions
        from stellaris_advisor.planner.models import FactionSummary

        result = analyze_factions(planner_empire)
        assert isinstance(result, FactionSummary)
        assert result.influence_income >= 0
