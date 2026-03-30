# tests/test_planner/test_megastructure_advisor.py
"""Tests for megastructure advisor."""
from __future__ import annotations

import pytest

from stellaris_advisor.parser.models import EmpireState, GalacticState


@pytest.mark.unit
class TestMegastructureAdvisor:
    def test_not_ready_without_tech(
        self,
        planner_empire: EmpireState,
        planner_galactic: GalacticState,
    ) -> None:
        from stellaris_advisor.planner.megastructure_advisor import (
            check_megastructure_readiness,
        )

        ready, missing = check_megastructure_readiness(planner_empire, planner_galactic)
        assert not ready
        assert "tech_mega_engineering" in missing
