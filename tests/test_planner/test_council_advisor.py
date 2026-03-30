# tests/test_planner/test_council_advisor.py
"""Tests for council advisor."""
from __future__ import annotations

import pytest

from stellaris_advisor.parser.models import EmpireState


@pytest.mark.unit
class TestCouncilAdvisor:
    def test_finds_hor(self, planner_empire: EmpireState) -> None:
        from stellaris_advisor.planner.council_advisor import analyze_council

        result = analyze_council(planner_empire)
        assert "hor_name" in result or "no_hor" in result
