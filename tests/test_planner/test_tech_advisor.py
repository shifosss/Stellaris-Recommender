# tests/test_planner/test_tech_advisor.py
"""Tests for tech path optimizer."""
from __future__ import annotations

import pytest

from stellaris_advisor.parser.models import EmpireState


@pytest.mark.unit
class TestTechAdvisor:
    def test_suggest_tech_path_returns_insight(self, planner_empire: EmpireState) -> None:
        from stellaris_advisor.planner.models import TechPathInsight
        from stellaris_advisor.planner.tech_advisor import suggest_tech_path

        result = suggest_tech_path(planner_empire, game_graph=None, bottleneck="fleet_power")
        assert result is None or isinstance(result, TechPathInsight)

    def test_resolves_keyword_goal(self, planner_empire: EmpireState) -> None:
        from stellaris_advisor.planner.tech_advisor import resolve_tech_goal

        tech_ids = resolve_tech_goal("battleships")
        assert "tech_battleships" in tech_ids
