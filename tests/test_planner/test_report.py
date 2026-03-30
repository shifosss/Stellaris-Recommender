# tests/test_planner/test_report.py
"""Tests for report assembly — the top-level analyze() function."""
from __future__ import annotations

import pytest

from stellaris_advisor.config import PlannerConfig
from stellaris_advisor.parser.models import EmpireState, GalacticState


@pytest.mark.unit
class TestAnalyze:
    def test_returns_report(
        self,
        planner_empire: EmpireState,
        planner_galactic: GalacticState,
    ) -> None:
        from stellaris_advisor.planner.models import Report
        from stellaris_advisor.planner.report import analyze

        config = PlannerConfig()
        report = analyze(planner_empire, planner_galactic, config)
        assert isinstance(report, Report)

    def test_degraded_mode(
        self,
        planner_empire: EmpireState,
        planner_galactic: GalacticState,
    ) -> None:
        from stellaris_advisor.planner.report import analyze

        config = PlannerConfig()
        report = analyze(planner_empire, planner_galactic, config)
        assert report.mode == "degraded"

    def test_has_recommendations(
        self,
        planner_empire: EmpireState,
        planner_galactic: GalacticState,
    ) -> None:
        from stellaris_advisor.planner.report import analyze

        config = PlannerConfig()
        report = analyze(planner_empire, planner_galactic, config)
        assert len(report.recommendations) > 0
        assert report.recommendations[0].rank == 1

    def test_has_alerts(
        self,
        planner_empire: EmpireState,
        planner_galactic: GalacticState,
    ) -> None:
        from stellaris_advisor.planner.report import analyze

        config = PlannerConfig()
        report = analyze(planner_empire, planner_galactic, config)
        assert isinstance(report.alerts, tuple)

    def test_has_empire_health(
        self,
        planner_empire: EmpireState,
        planner_galactic: GalacticState,
    ) -> None:
        from stellaris_advisor.planner.models import EmpireHealth
        from stellaris_advisor.planner.report import analyze

        config = PlannerConfig()
        report = analyze(planner_empire, planner_galactic, config)
        assert isinstance(report.empire_health, EmpireHealth)
        assert 0 <= report.empire_health.overall_percentile <= 100

    def test_report_frozen(
        self,
        planner_empire: EmpireState,
        planner_galactic: GalacticState,
    ) -> None:
        from stellaris_advisor.planner.report import analyze

        config = PlannerConfig()
        report = analyze(planner_empire, planner_galactic, config)
        with pytest.raises(AttributeError):
            report.mode = "full"  # type: ignore[misc]

    def test_recommendations_capped_at_top_n(
        self,
        planner_empire: EmpireState,
        planner_galactic: GalacticState,
    ) -> None:
        from stellaris_advisor.planner.report import analyze

        config = PlannerConfig(top_n_recommendations=3)
        report = analyze(planner_empire, planner_galactic, config)
        assert len(report.recommendations) <= 3
