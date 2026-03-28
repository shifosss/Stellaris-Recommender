"""Integration tests for the features package.

End-to-end pipeline: synthetic game state → all extractors → verify output shape.
Tests also validate public API exports from features/__init__.py.
"""

from __future__ import annotations

import pytest

from stellaris_advisor.features import (
    ALL_FEATURE_NAMES,
    EMPIRE_FEATURE_NAMES,
    SYSTEM_FEATURE_NAMES,
    TEMPORAL_FEATURE_NAMES,
    build_hyperlane_graph,
    chokepoint_score,
    compute_betweenness,
    distance_to_nearest,
    distance_to_system,
    extract_empire_features,
    extract_system_features,
    extract_temporal_features,
)

from .conftest import make_galactic


# ---------------------------------------------------------------------------
# Public API imports
# ---------------------------------------------------------------------------


class TestPublicAPI:
    @pytest.mark.unit
    def test_extract_empire_features_importable(self) -> None:
        assert callable(extract_empire_features)

    @pytest.mark.unit
    def test_extract_system_features_importable(self) -> None:
        assert callable(extract_system_features)

    @pytest.mark.unit
    def test_extract_temporal_features_importable(self) -> None:
        assert callable(extract_temporal_features)

    @pytest.mark.unit
    def test_hyperlane_functions_importable(self) -> None:
        assert callable(build_hyperlane_graph)
        assert callable(compute_betweenness)
        assert callable(chokepoint_score)
        assert callable(distance_to_system)
        assert callable(distance_to_nearest)

    @pytest.mark.unit
    def test_feature_registries_importable(self) -> None:
        assert EMPIRE_FEATURE_NAMES is not None
        assert SYSTEM_FEATURE_NAMES is not None
        assert TEMPORAL_FEATURE_NAMES is not None
        assert ALL_FEATURE_NAMES is not None


# ---------------------------------------------------------------------------
# Registry counts
# ---------------------------------------------------------------------------


class TestRegistryCounts:
    @pytest.mark.unit
    def test_empire_feature_count(self) -> None:
        assert len(EMPIRE_FEATURE_NAMES) == 109

    @pytest.mark.unit
    def test_system_feature_count(self) -> None:
        assert len(SYSTEM_FEATURE_NAMES) == 12

    @pytest.mark.unit
    def test_temporal_feature_count(self) -> None:
        assert len(TEMPORAL_FEATURE_NAMES) == 8

    @pytest.mark.unit
    def test_all_feature_count(self) -> None:
        # 109 + 12 + 8 = 129
        assert len(ALL_FEATURE_NAMES) == 129

    @pytest.mark.unit
    def test_all_feature_names_is_concatenation(self) -> None:
        assert ALL_FEATURE_NAMES == (
            EMPIRE_FEATURE_NAMES + SYSTEM_FEATURE_NAMES + TEMPORAL_FEATURE_NAMES
        )

    @pytest.mark.unit
    def test_no_duplicate_feature_names(self) -> None:
        assert len(set(ALL_FEATURE_NAMES)) == len(ALL_FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Empire features integration
# ---------------------------------------------------------------------------


class TestEmpireFeaturesIntegration:
    @pytest.mark.integration
    def test_returns_dict(self) -> None:
        galactic = make_galactic()
        result = extract_empire_features(galactic.player_empire, galactic)
        assert isinstance(result, dict)

    @pytest.mark.integration
    def test_returns_correct_key_count(self) -> None:
        galactic = make_galactic()
        result = extract_empire_features(galactic.player_empire, galactic)
        assert len(result) == len(EMPIRE_FEATURE_NAMES)

    @pytest.mark.integration
    def test_keys_match_registry(self) -> None:
        galactic = make_galactic()
        result = extract_empire_features(galactic.player_empire, galactic)
        assert set(result.keys()) == set(EMPIRE_FEATURE_NAMES)

    @pytest.mark.integration
    def test_all_values_are_float(self) -> None:
        galactic = make_galactic()
        result = extract_empire_features(galactic.player_empire, galactic)
        assert all(isinstance(v, float) for v in result.values())


# ---------------------------------------------------------------------------
# System features integration
# ---------------------------------------------------------------------------


class TestSystemFeaturesIntegration:
    @pytest.mark.integration
    def test_returns_dict(self) -> None:
        galactic = make_galactic()
        graph = build_hyperlane_graph(galactic)
        btw = compute_betweenness(graph)
        result = extract_system_features(
            galactic.systems[0], galactic.player_empire, galactic, graph, btw
        )
        assert isinstance(result, dict)

    @pytest.mark.integration
    def test_returns_correct_key_count(self) -> None:
        galactic = make_galactic()
        graph = build_hyperlane_graph(galactic)
        btw = compute_betweenness(graph)
        result = extract_system_features(
            galactic.systems[0], galactic.player_empire, galactic, graph, btw
        )
        assert len(result) == len(SYSTEM_FEATURE_NAMES)

    @pytest.mark.integration
    def test_keys_match_registry(self) -> None:
        galactic = make_galactic()
        graph = build_hyperlane_graph(galactic)
        btw = compute_betweenness(graph)
        result = extract_system_features(
            galactic.systems[0], galactic.player_empire, galactic, graph, btw
        )
        assert set(result.keys()) == set(SYSTEM_FEATURE_NAMES)

    @pytest.mark.integration
    def test_all_values_are_float(self) -> None:
        galactic = make_galactic()
        graph = build_hyperlane_graph(galactic)
        btw = compute_betweenness(graph)
        result = extract_system_features(
            galactic.systems[0], galactic.player_empire, galactic, graph, btw
        )
        assert all(isinstance(v, float) for v in result.values())


# ---------------------------------------------------------------------------
# Temporal features integration
# ---------------------------------------------------------------------------


class TestTemporalFeaturesIntegration:
    @pytest.mark.integration
    def test_returns_dict(self) -> None:
        galactic = make_galactic()
        current = extract_empire_features(galactic.player_empire, galactic)
        result = extract_temporal_features(current, (current,))
        assert isinstance(result, dict)

    @pytest.mark.integration
    def test_returns_correct_key_count(self) -> None:
        galactic = make_galactic()
        current = extract_empire_features(galactic.player_empire, galactic)
        result = extract_temporal_features(current, (current,))
        assert len(result) == len(TEMPORAL_FEATURE_NAMES)

    @pytest.mark.integration
    def test_keys_match_registry(self) -> None:
        galactic = make_galactic()
        current = extract_empire_features(galactic.player_empire, galactic)
        result = extract_temporal_features(current, (current,))
        assert set(result.keys()) == set(TEMPORAL_FEATURE_NAMES)

    @pytest.mark.integration
    def test_all_values_are_float(self) -> None:
        galactic = make_galactic()
        current = extract_empire_features(galactic.player_empire, galactic)
        result = extract_temporal_features(current, (current,))
        assert all(isinstance(v, float) for v in result.values())

    @pytest.mark.integration
    def test_zero_deltas_same_state_triggers_stall(self) -> None:
        galactic = make_galactic()
        current = extract_empire_features(galactic.player_empire, galactic)
        result = extract_temporal_features(current, (current,))
        assert result["temporal_delta_energy"] == pytest.approx(0.0)
        assert result["temporal_delta_fleet_power"] == pytest.approx(0.0)
        assert result["temporal_stall_flag"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Full pipeline integration
# ---------------------------------------------------------------------------


class TestFullPipeline:
    @pytest.mark.integration
    def test_all_extractors_produce_float_dicts(self) -> None:
        """All three extractors produce dict[str, float] from a synthetic game state."""
        galactic = make_galactic()
        empire = galactic.player_empire
        system = galactic.systems[0]
        graph = build_hyperlane_graph(galactic)
        btw = compute_betweenness(graph)

        empire_feats = extract_empire_features(empire, galactic)
        system_feats = extract_system_features(system, empire, galactic, graph, btw)
        temporal_feats = extract_temporal_features(empire_feats, (empire_feats,))

        for label, feats in [
            ("empire", empire_feats),
            ("system", system_feats),
            ("temporal", temporal_feats),
        ]:
            assert isinstance(feats, dict), f"{label} extractor must return dict"
            assert all(
                isinstance(v, float) for v in feats.values()
            ), f"{label} values must all be float"

    @pytest.mark.integration
    def test_total_feature_count_matches_registry(self) -> None:
        """Combined feature count matches 109 + 12 + 8 = 129."""
        galactic = make_galactic()
        empire = galactic.player_empire
        system = galactic.systems[0]
        graph = build_hyperlane_graph(galactic)
        btw = compute_betweenness(graph)

        empire_feats = extract_empire_features(empire, galactic)
        system_feats = extract_system_features(system, empire, galactic, graph, btw)
        temporal_feats = extract_temporal_features(empire_feats, ())

        assert len(empire_feats) == 109
        assert len(system_feats) == 12
        assert len(temporal_feats) == 8
        assert len(empire_feats) + len(system_feats) + len(temporal_feats) == 129

    @pytest.mark.integration
    def test_hyperlane_pipeline(self) -> None:
        """build_hyperlane_graph → compute_betweenness → chokepoint_score all work together."""
        galactic = make_galactic()
        graph = build_hyperlane_graph(galactic)
        btw = compute_betweenness(graph)

        for system in galactic.systems:
            score = chokepoint_score(btw, system.id)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    @pytest.mark.integration
    def test_all_systems_can_be_extracted(self) -> None:
        """extract_system_features works for every system in the galactic state."""
        galactic = make_galactic()
        empire = galactic.player_empire
        graph = build_hyperlane_graph(galactic)
        btw = compute_betweenness(graph)

        for system in galactic.systems:
            result = extract_system_features(system, empire, galactic, graph, btw)
            assert set(result.keys()) == set(SYSTEM_FEATURE_NAMES)
