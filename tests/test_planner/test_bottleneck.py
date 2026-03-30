# tests/test_planner/test_bottleneck.py
"""Tests for bottleneck detection."""
from __future__ import annotations

import pytest

from stellaris_advisor.colonization.benchmark import BenchmarkStore


@pytest.mark.unit
class TestDetectBottleneck:
    def test_returns_bottleneck_result(self) -> None:
        from stellaris_advisor.planner.bottleneck import detect_bottleneck
        from stellaris_advisor.planner.models import BottleneckResult

        features = {
            "resource_net_alloys": 5.0,  # very low
            "tech_physics_research_speed": 100.0,
            "military_fleet_power": 8000.0,
            "tech_total_count": 30,
            "expansion_system_count": 15,
            "resource_net_influence": 3.0,
            "tradition_unity_income": 20.0,
            "tradition_trees_completed": 0,
        }
        store = BenchmarkStore.hardcoded_defaults()
        result = detect_bottleneck(features, 2275, "captain", "standard", store)
        assert isinstance(result, BottleneckResult)
        assert result.dimension in (
            "alloy_income",
            "research_total",
            "fleet_power",
            "tech_count",
            "system_count",
            "influence_income",
            "unity_income",
            "traditions_completed",
        )
        assert 0 <= result.percentile_rank <= 100

    def test_low_alloys_detected(self) -> None:
        from stellaris_advisor.planner.bottleneck import detect_bottleneck

        features = {
            "resource_net_alloys": 5.0,  # very low for year 2275
            "tech_physics_research_speed": 500.0,
            "military_fleet_power": 30000.0,
            "tech_total_count": 80,
            "expansion_system_count": 30,
            "resource_net_influence": 5.0,
            "tradition_unity_income": 100.0,
            "tradition_trees_completed": 3,
        }
        store = BenchmarkStore.hardcoded_defaults()
        result = detect_bottleneck(features, 2275, "captain", "standard", store)
        assert result.dimension == "alloy_income"

    def test_empire_health(self) -> None:
        from stellaris_advisor.planner.bottleneck import compute_empire_health
        from stellaris_advisor.planner.models import EmpireHealth

        features = {
            "resource_net_alloys": 100.0,
            "tech_physics_research_speed": 500.0,
            "military_fleet_power": 30000.0,
            "tech_total_count": 80,
            "expansion_system_count": 30,
            "resource_net_influence": 5.0,
            "tradition_unity_income": 100.0,
            "tradition_trees_completed": 3,
        }
        store = BenchmarkStore.hardcoded_defaults()
        health = compute_empire_health(features, 2275, "captain", "standard", store)
        assert isinstance(health, EmpireHealth)
        assert 0 <= health.overall_percentile <= 100
        assert len(health.dimension_percentiles) == 8

    def test_missing_benchmark_returns_neutral(self) -> None:
        from stellaris_advisor.planner.bottleneck import compute_empire_health

        features = {
            "resource_net_alloys": 100.0,
            "tech_physics_research_speed": 500.0,
            "military_fleet_power": 30000.0,
            "tech_total_count": 80,
            "expansion_system_count": 30,
            "resource_net_influence": 5.0,
            "tradition_unity_income": 100.0,
            "tradition_trees_completed": 3,
        }
        store = BenchmarkStore.hardcoded_defaults()
        # Unknown difficulty → no data → neutral 50
        health = compute_empire_health(features, 2275, "impossible", "alien", store)
        assert health.overall_percentile == 50
