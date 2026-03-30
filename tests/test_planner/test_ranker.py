# tests/test_planner/test_ranker.py
"""Tests for hybrid ranker."""
from __future__ import annotations

import pytest

from stellaris_advisor.config import PlannerConfig
from stellaris_advisor.planner.models import Recommendation


@pytest.mark.unit
class TestRankDegraded:
    def test_ranks_by_rule_roi(self) -> None:
        from stellaris_advisor.planner.ranker import rank_recommendations

        recs = (
            Recommendation(
                rank=0,
                category="economy",
                action_type="build_building",
                description="A",
                detail="",
                rule_roi=0.3,
                rl_delta=0.0,
                final_score=0.0,
                action_params=(),
            ),
            Recommendation(
                rank=0,
                category="military",
                action_type="build_ships",
                description="B",
                detail="",
                rule_roi=0.9,
                rl_delta=0.0,
                final_score=0.0,
                action_params=(),
            ),
            Recommendation(
                rank=0,
                category="tech",
                action_type="research_tech",
                description="C",
                detail="",
                rule_roi=0.6,
                rl_delta=0.0,
                final_score=0.0,
                action_params=(),
            ),
        )
        config = PlannerConfig(top_n_recommendations=3)
        ranked = rank_recommendations(recs, config, rl_predictor=None, empire_features=None)
        assert ranked[0].rank == 1
        assert ranked[0].description == "B"  # highest ROI
        assert ranked[1].description == "C"
        assert ranked[2].description == "A"

    def test_limits_to_top_n(self) -> None:
        from stellaris_advisor.planner.ranker import rank_recommendations

        recs = tuple(
            Recommendation(
                rank=0,
                category="economy",
                action_type="build_building",
                description=f"rec_{i}",
                detail="",
                rule_roi=i * 0.1,
                rl_delta=0.0,
                final_score=0.0,
                action_params=(),
            )
            for i in range(20)
        )
        config = PlannerConfig(top_n_recommendations=7)
        ranked = rank_recommendations(recs, config, rl_predictor=None, empire_features=None)
        assert len(ranked) == 7

    def test_final_score_equals_rule_roi_in_degraded(self) -> None:
        from stellaris_advisor.planner.ranker import rank_recommendations

        recs = (
            Recommendation(
                rank=0,
                category="tech",
                action_type="research_tech",
                description="X",
                detail="",
                rule_roi=0.75,
                rl_delta=0.0,
                final_score=0.0,
                action_params=(),
            ),
        )
        config = PlannerConfig()
        ranked = rank_recommendations(recs, config, rl_predictor=None, empire_features=None)
        assert ranked[0].final_score == pytest.approx(0.75)

    def test_empty_input_returns_empty(self) -> None:
        from stellaris_advisor.planner.ranker import rank_recommendations

        config = PlannerConfig()
        ranked = rank_recommendations((), config, rl_predictor=None, empire_features=None)
        assert ranked == ()
