# tests/test_planner/test_integration.py
"""End-to-end planner integration tests."""
from __future__ import annotations

import pytest

from stellaris_advisor.config import PlannerConfig
from stellaris_advisor.parser.models import EmpireState, GalacticState


@pytest.mark.integration
class TestPlannerIntegration:
    def test_full_pipeline(
        self,
        planner_empire: EmpireState,
        planner_galactic: GalacticState,
    ) -> None:
        """End-to-end: analyze() produces a complete Report."""
        from stellaris_advisor.planner import Report, analyze

        config = PlannerConfig()
        report = analyze(planner_empire, planner_galactic, config)

        assert isinstance(report, Report)
        assert report.empire_name == "United Nations of Earth"
        assert report.year == 2275
        assert report.mode == "degraded"
        assert len(report.recommendations) > 0
        assert len(report.recommendations) <= config.top_n_recommendations
        assert report.recommendations[0].rank == 1
        assert report.empire_health.overall_percentile >= 0
        assert isinstance(report.alerts, tuple)

    def test_import_without_ml(self) -> None:
        """Planner module imports without ML dependencies."""
        from stellaris_advisor.planner import analyze  # noqa: F401

    def test_recommendations_are_ranked(
        self,
        planner_empire: EmpireState,
        planner_galactic: GalacticState,
    ) -> None:
        from stellaris_advisor.planner import analyze

        config = PlannerConfig()
        report = analyze(planner_empire, planner_galactic, config)
        for i in range(len(report.recommendations) - 1):
            assert (
                report.recommendations[i].final_score
                >= report.recommendations[i + 1].final_score
            )

    def test_all_model_types_frozen(self) -> None:
        """All planner data types are immutable."""
        from stellaris_advisor.planner import (
            Alert,
            AlertSeverity,  # noqa: F401
            BottleneckResult,
            EmpireHealth,
            Recommendation,
            Report,
            TechPathInsight,
            TechPathStep,
            TraditionPerkPath,
        )

        # Verify frozen by checking __dataclass_params__
        for cls in (
            Alert,
            BottleneckResult,
            EmpireHealth,
            Recommendation,
            Report,
            TechPathInsight,
            TechPathStep,
            TraditionPerkPath,
        ):
            assert cls.__dataclass_params__.frozen  # type: ignore[attr-defined]
