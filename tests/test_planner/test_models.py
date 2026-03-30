# tests/test_planner/test_models.py
"""Tests for planner data models — all must be frozen."""
from __future__ import annotations

import pytest


@pytest.mark.unit
class TestAlertSeverity:
    def test_values(self) -> None:
        from stellaris_advisor.planner.models import AlertSeverity

        assert AlertSeverity.HIGH == "high"
        assert AlertSeverity.MEDIUM == "medium"
        assert AlertSeverity.LOW == "low"
        assert AlertSeverity.INFO == "info"


@pytest.mark.unit
class TestAlert:
    def test_frozen(self) -> None:
        from stellaris_advisor.planner.models import Alert, AlertSeverity

        a = Alert(
            severity=AlertSeverity.HIGH,
            category="resource",
            message="Energy deficit",
            detail="12 months to depletion",
        )
        with pytest.raises(AttributeError):
            a.message = "changed"  # type: ignore[misc]


@pytest.mark.unit
class TestRecommendation:
    def test_frozen(self) -> None:
        from stellaris_advisor.planner.models import Recommendation

        r = Recommendation(
            rank=1,
            category="economy",
            action_type="build_building",
            description="Build Alloy Foundry on Capital",
            detail="Produces +8 alloys/month",
            rule_roi=0.8,
            rl_delta=0.0,
            final_score=0.8,
            action_params=(),
        )
        with pytest.raises(AttributeError):
            r.rank = 2  # type: ignore[misc]

    def test_default_rl_delta(self) -> None:
        from stellaris_advisor.planner.models import Recommendation

        r = Recommendation(
            rank=1,
            category="tech",
            action_type="research_tech",
            description="Research Destroyers",
            detail="",
            rule_roi=0.5,
            rl_delta=0.0,
            final_score=0.5,
            action_params=(),
        )
        assert r.rl_delta == 0.0


@pytest.mark.unit
class TestBottleneckResult:
    def test_frozen(self) -> None:
        from stellaris_advisor.planner.models import BottleneckResult

        b = BottleneckResult(
            dimension="alloy_income",
            gap=-0.3,
            percentile_rank=25,
            resolution_hint="Build alloy foundries or research alloy tech",
        )
        with pytest.raises(AttributeError):
            b.dimension = "tech"  # type: ignore[misc]


@pytest.mark.unit
class TestEmpireHealth:
    def test_frozen(self) -> None:
        from stellaris_advisor.planner.models import EmpireHealth

        h = EmpireHealth(
            overall_percentile=65,
            dimension_percentiles=(
                ("alloy_income", 45),
                ("research_total", 70),
                ("fleet_power", 55),
                ("tech_count", 60),
                ("system_count", 80),
                ("influence_income", 50),
                ("unity_income", 75),
                ("traditions_completed", 65),
            ),
            bottleneck=None,
        )
        assert h.overall_percentile == 65


@pytest.mark.unit
class TestTechPathInsight:
    def test_frozen(self) -> None:
        from stellaris_advisor.planner.models import TechPathInsight, TechPathStep

        step = TechPathStep(
            tech_id="tech_destroyers",
            tech_name="Destroyers",
            area="engineering",
            tier=2,
            cost=3000,
            estimated_months=15.0,
            unlocks=("Destroyer hull",),
        )
        insight = TechPathInsight(
            goal="battleships",
            steps=(step,),
            total_months=15.0,
            bottleneck_addressed="fleet_power",
        )
        with pytest.raises(AttributeError):
            insight.goal = "changed"  # type: ignore[misc]


@pytest.mark.unit
class TestReport:
    def test_frozen(self) -> None:
        from stellaris_advisor.planner.models import EmpireHealth, Report

        r = Report(
            empire_name="Test Empire",
            year=2250,
            mode="degraded",
            empire_health=EmpireHealth(
                overall_percentile=50,
                dimension_percentiles=(),
                bottleneck=None,
            ),
            recommendations=(),
            tech_path_insight=None,
            tradition_perk_path=None,
            faction_influence_summary=None,
            alerts=(),
        )
        assert r.mode == "degraded"
