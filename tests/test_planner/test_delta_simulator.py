# tests/test_planner/test_delta_simulator.py
"""Tests for feature delta simulation."""
from __future__ import annotations

import pytest


@pytest.mark.unit
class TestSimulateDelta:
    def test_returns_new_dict(self) -> None:
        from stellaris_advisor.planner.delta_simulator import simulate_delta

        features = {"resource_net_alloys": 25.0, "tech_total_count": 30.0}
        result = simulate_delta(features, "research_tech", {"area": "engineering"})
        assert result is not features
        assert features["tech_total_count"] == 30.0  # original unchanged

    def test_research_tech_increments_count(self) -> None:
        from stellaris_advisor.planner.delta_simulator import simulate_delta

        features = {
            "tech_total_count": 30.0,
            "tech_engineering_count": 10.0,
            "tech_frontier_size": 5.0,
        }
        result = simulate_delta(features, "research_tech", {"area": "engineering"})
        assert result["tech_total_count"] == 31.0
        assert result["tech_engineering_count"] == 11.0

    def test_build_building_adjusts_resources(self) -> None:
        from stellaris_advisor.planner.delta_simulator import simulate_delta

        features = {"resource_net_alloys": 25.0, "expansion_building_utilization": 0.5}
        result = simulate_delta(features, "build_building", {"net_output": "8.0"})
        assert result["resource_net_alloys"] == 33.0

    def test_colonize_system_increments_counts(self) -> None:
        from stellaris_advisor.planner.delta_simulator import simulate_delta

        features = {"expansion_system_count": 15.0, "expansion_planet_count": 5.0}
        result = simulate_delta(features, "colonize_system", {})
        assert result["expansion_system_count"] == 16.0
        assert result["expansion_planet_count"] == 6.0

    def test_build_ships_adjusts_fleet(self) -> None:
        from stellaris_advisor.planner.delta_simulator import simulate_delta

        features = {
            "military_fleet_power": 8000.0,
            "military_naval_capacity_utilization": 0.4,
        }
        result = simulate_delta(features, "build_ships", {"fleet_power_gain": "2000"})
        assert result["military_fleet_power"] == 10000.0

    def test_declare_rival_adjusts_influence(self) -> None:
        from stellaris_advisor.planner.delta_simulator import simulate_delta

        features = {"resource_net_influence": 3.0}
        result = simulate_delta(features, "declare_rival", {})
        assert result["resource_net_influence"] == 3.5

    def test_unknown_action_returns_copy(self) -> None:
        from stellaris_advisor.planner.delta_simulator import simulate_delta

        features = {"resource_net_alloys": 25.0}
        result = simulate_delta(features, "unknown_action", {})
        assert result == features
        assert result is not features
