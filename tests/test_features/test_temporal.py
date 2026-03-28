"""Tests for temporal feature extraction.

Covers: extract_temporal_features and all 8 temporal delta features.
"""

from __future__ import annotations

import pytest

from stellaris_advisor.features.temporal import extract_temporal_features

TEMPORAL_NAMES = frozenset(
    {
        "temporal_delta_energy",
        "temporal_delta_minerals",
        "temporal_delta_alloys",
        "temporal_delta_research",
        "temporal_expansion_velocity",
        "temporal_tech_velocity",
        "temporal_delta_fleet_power",
        "temporal_stall_flag",
    }
)

# ---------------------------------------------------------------------------
# Minimal empire feature dicts used as fixtures
# ---------------------------------------------------------------------------

_BASE: dict[str, float] = {
    "resource_net_energy": 30.0,
    "resource_net_minerals": 30.0,
    "resource_net_alloys": 10.0,
    "tech_physics_research_speed": 1.0,
    "tech_society_research_speed": 1.0,
    "tech_engineering_research_speed": 1.0,
    "expansion_system_count": 5.0,
    "tech_total_count": 20.0,
    "military_fleet_power": 1000.0,
}

_CHANGED: dict[str, float] = {
    "resource_net_energy": 35.0,  # +5
    "resource_net_minerals": 32.0,  # +2
    "resource_net_alloys": 10.5,  # +0.5
    "tech_physics_research_speed": 1.1,
    "tech_society_research_speed": 1.0,
    "tech_engineering_research_speed": 1.0,  # research total: +0.1
    "expansion_system_count": 6.0,  # +1
    "tech_total_count": 22.0,  # +2
    "military_fleet_power": 1200.0,  # +200
}

# A state identical to _BASE (zero deltas → stall)
_STALL: dict[str, float] = dict(_BASE)


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


class TestExtractTemporalFeaturesStructure:
    @pytest.mark.unit
    def test_returns_dict(self) -> None:
        result = extract_temporal_features(_BASE, (_BASE,))
        assert isinstance(result, dict)

    @pytest.mark.unit
    def test_all_values_float(self) -> None:
        result = extract_temporal_features(_BASE, (_BASE,))
        assert all(isinstance(v, float) for v in result.values())

    @pytest.mark.unit
    def test_correct_keys(self) -> None:
        result = extract_temporal_features(_BASE, (_BASE,))
        assert set(result.keys()) == TEMPORAL_NAMES

    @pytest.mark.unit
    def test_keys_match_registry(self) -> None:
        from stellaris_advisor.features.registry import TEMPORAL_FEATURE_NAMES

        result = extract_temporal_features(_BASE, (_BASE,))
        assert set(result.keys()) == set(TEMPORAL_FEATURE_NAMES)

    @pytest.mark.unit
    def test_feature_count(self) -> None:
        result = extract_temporal_features(_BASE, (_BASE,))
        assert len(result) == 8


# ---------------------------------------------------------------------------
# Empty history → all zeros
# ---------------------------------------------------------------------------


class TestEmptyHistory:
    @pytest.mark.unit
    def test_all_zero_when_no_history(self) -> None:
        result = extract_temporal_features(_BASE, ())
        assert all(v == pytest.approx(0.0) for v in result.values())

    @pytest.mark.unit
    def test_returns_all_temporal_keys_when_empty(self) -> None:
        result = extract_temporal_features(_BASE, ())
        assert set(result.keys()) == TEMPORAL_NAMES


# ---------------------------------------------------------------------------
# Delta features
# ---------------------------------------------------------------------------


class TestDeltaFeatures:
    @pytest.mark.unit
    def test_delta_energy(self) -> None:
        result = extract_temporal_features(_CHANGED, (_BASE,))
        # 35.0 - 30.0 = 5.0
        assert result["temporal_delta_energy"] == pytest.approx(5.0)

    @pytest.mark.unit
    def test_delta_minerals(self) -> None:
        result = extract_temporal_features(_CHANGED, (_BASE,))
        # 32.0 - 30.0 = 2.0
        assert result["temporal_delta_minerals"] == pytest.approx(2.0)

    @pytest.mark.unit
    def test_delta_alloys(self) -> None:
        result = extract_temporal_features(_CHANGED, (_BASE,))
        # 10.5 - 10.0 = 0.5
        assert result["temporal_delta_alloys"] == pytest.approx(0.5)

    @pytest.mark.unit
    def test_delta_research_sums_three_speeds(self) -> None:
        # current: 1.1+1.0+1.0 = 3.1; prev: 1.0+1.0+1.0 = 3.0 → delta = 0.1
        result = extract_temporal_features(_CHANGED, (_BASE,))
        assert result["temporal_delta_research"] == pytest.approx(0.1)

    @pytest.mark.unit
    def test_expansion_velocity(self) -> None:
        result = extract_temporal_features(_CHANGED, (_BASE,))
        # 6.0 - 5.0 = 1.0
        assert result["temporal_expansion_velocity"] == pytest.approx(1.0)

    @pytest.mark.unit
    def test_tech_velocity(self) -> None:
        result = extract_temporal_features(_CHANGED, (_BASE,))
        # 22.0 - 20.0 = 2.0
        assert result["temporal_tech_velocity"] == pytest.approx(2.0)

    @pytest.mark.unit
    def test_delta_fleet_power(self) -> None:
        result = extract_temporal_features(_CHANGED, (_BASE,))
        # 1200.0 - 1000.0 = 200.0
        assert result["temporal_delta_fleet_power"] == pytest.approx(200.0)

    @pytest.mark.unit
    def test_negative_delta(self) -> None:
        # Fleet power declined
        current = dict(_BASE)
        current["military_fleet_power"] = 800.0
        result = extract_temporal_features(current, (_BASE,))
        assert result["temporal_delta_fleet_power"] == pytest.approx(-200.0)

    @pytest.mark.unit
    def test_uses_only_history_index_zero(self) -> None:
        # history[0] is _BASE, history[1] has very different values — only [0] used
        very_old: dict[str, float] = dict(_BASE)
        very_old["resource_net_energy"] = 0.0
        current = dict(_BASE)
        current["resource_net_energy"] = 35.0
        result = extract_temporal_features(current, (_BASE, very_old))
        # Should use _BASE (index 0), not very_old
        assert result["temporal_delta_energy"] == pytest.approx(5.0)

    @pytest.mark.unit
    def test_zero_delta_when_same_state(self) -> None:
        result = extract_temporal_features(_BASE, (_BASE,))
        assert result["temporal_delta_energy"] == pytest.approx(0.0)
        assert result["temporal_delta_fleet_power"] == pytest.approx(0.0)
        assert result["temporal_expansion_velocity"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Stall flag
# ---------------------------------------------------------------------------


class TestStallFlag:
    @pytest.mark.unit
    def test_stall_when_all_deltas_zero(self) -> None:
        result = extract_temporal_features(_STALL, (_BASE,))
        assert result["temporal_stall_flag"] == pytest.approx(1.0)

    @pytest.mark.unit
    def test_stall_when_all_deltas_below_threshold(self) -> None:
        # All changes are tiny — below every threshold
        current: dict[str, float] = {
            "resource_net_energy": 30.1,  # delta 0.1 < 0.5 ✓
            "resource_net_minerals": 30.1,  # delta 0.1 < 0.5 ✓
            "resource_net_alloys": 10.05,  # delta 0.05 < 0.1 ✓
            "tech_physics_research_speed": 1.0,
            "tech_society_research_speed": 1.0,
            "tech_engineering_research_speed": 1.04,  # research delta 0.04 < 0.1 ✓
            "expansion_system_count": 5.005,  # delta 0.005 < 0.01 ✓
            "tech_total_count": 20.005,  # delta 0.005 < 0.01 ✓
            "military_fleet_power": 1005.0,  # delta 5.0 < 10.0 ✓
        }
        result = extract_temporal_features(current, (_BASE,))
        assert result["temporal_stall_flag"] == pytest.approx(1.0)

    @pytest.mark.unit
    def test_no_stall_when_energy_above_threshold(self) -> None:
        current = dict(_BASE)
        current["resource_net_energy"] = 31.0  # delta = 1.0 ≥ 0.5
        result = extract_temporal_features(current, (_BASE,))
        assert result["temporal_stall_flag"] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_no_stall_when_fleet_power_above_threshold(self) -> None:
        current = dict(_BASE)
        current["military_fleet_power"] = 1015.0  # delta = 15.0 ≥ 10.0
        result = extract_temporal_features(current, (_BASE,))
        assert result["temporal_stall_flag"] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_no_stall_when_expansion_above_threshold(self) -> None:
        current = dict(_BASE)
        current["expansion_system_count"] = 6.0  # delta = 1.0 ≥ 0.01
        result = extract_temporal_features(current, (_BASE,))
        assert result["temporal_stall_flag"] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_stall_is_zero_when_history_empty(self) -> None:
        # Empty history → all zeros including stall flag
        result = extract_temporal_features(_BASE, ())
        assert result["temporal_stall_flag"] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_negative_delta_still_triggers_stall_check(self) -> None:
        # Small negative delta should still be considered "stalled"
        current = dict(_BASE)
        current["resource_net_energy"] = 29.9  # delta = -0.1, abs = 0.1 < 0.5 ✓
        current["military_fleet_power"] = 999.0  # delta = -1.0, abs = 1.0 < 10.0 ✓
        result = extract_temporal_features(current, (_BASE,))
        assert result["temporal_stall_flag"] == pytest.approx(1.0)
