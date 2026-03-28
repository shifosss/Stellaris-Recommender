"""Tests for system feature extraction.

Covers: extract_system_features and all 12 system-level features.
"""

from __future__ import annotations

from types import MappingProxyType

import networkx as nx
import pytest

from stellaris_advisor.features.hyperlane import (
    build_hyperlane_graph,
    compute_betweenness,
)
from stellaris_advisor.features.system import extract_system_features
from stellaris_advisor.parser.models import (
    EmpireState,
    GalacticState,
    PlanetSummary,
    SystemState,
)

from .conftest import make_empire, make_galactic, make_planet, make_system

SYSTEM_NAMES = frozenset(
    {
        "system_best_habitability",
        "system_best_planet_size",
        "system_district_slots_city",
        "system_district_slots_mining",
        "system_district_slots_farming",
        "system_district_slots_generator",
        "system_resource_deposit_value",
        "system_hyperlane_degree",
        "system_chokepoint_score",
        "system_distance_to_capital",
        "system_distance_to_hostile",
        "system_climate_match_score",
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_graph_and_betweenness(
    galactic: GalacticState,
) -> tuple[nx.Graph, dict[int, float]]:
    g = build_hyperlane_graph(galactic)
    return g, compute_betweenness(g)


def _call(
    system: SystemState,
    empire: EmpireState | None = None,
    galactic: GalacticState | None = None,
) -> dict[str, float]:
    """Convenience wrapper with sensible defaults."""
    if empire is None:
        empire = make_empire()
    if galactic is None:
        galactic = make_galactic()
    g, btw = _make_graph_and_betweenness(galactic)
    return extract_system_features(system, empire, galactic, g, btw)


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


class TestExtractSystemFeaturesStructure:
    @pytest.mark.unit
    def test_returns_dict(self) -> None:
        assert isinstance(_call(make_system()), dict)

    @pytest.mark.unit
    def test_all_values_float(self) -> None:
        result = _call(make_system())
        assert all(isinstance(v, float) for v in result.values())

    @pytest.mark.unit
    def test_correct_keys(self) -> None:
        assert set(_call(make_system()).keys()) == SYSTEM_NAMES

    @pytest.mark.unit
    def test_keys_match_registry(self) -> None:
        from stellaris_advisor.features.registry import SYSTEM_FEATURE_NAMES

        result = _call(make_system())
        assert set(result.keys()) == set(SYSTEM_FEATURE_NAMES)


# ---------------------------------------------------------------------------
# system_best_habitability
# ---------------------------------------------------------------------------


class TestSystemBestHabitability:
    @pytest.mark.unit
    def test_habitable_planet_scores_one(self) -> None:
        system = make_system(
            planets=(
                PlanetSummary(
                    id=1,
                    name="P",
                    planet_class="pc_continental",
                    size=18,
                    is_habitable=True,
                    is_colonized=False,
                ),
            )
        )
        assert _call(system)["system_best_habitability"] == pytest.approx(1.0)

    @pytest.mark.unit
    def test_no_habitable_planet_scores_zero(self) -> None:
        system = make_system(
            planets=(
                PlanetSummary(
                    id=1,
                    name="P",
                    planet_class="pc_gas_giant",
                    size=30,
                    is_habitable=False,
                    is_colonized=False,
                ),
            )
        )
        assert _call(system)["system_best_habitability"] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_no_planets_scores_zero(self) -> None:
        system = make_system(planets=())
        assert _call(system)["system_best_habitability"] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_mixed_planets_scores_one(self) -> None:
        system = make_system(
            planets=(
                PlanetSummary(
                    id=1,
                    name="Gas",
                    planet_class="pc_gas_giant",
                    size=30,
                    is_habitable=False,
                    is_colonized=False,
                ),
                PlanetSummary(
                    id=2,
                    name="Hab",
                    planet_class="pc_continental",
                    size=18,
                    is_habitable=True,
                    is_colonized=True,
                ),
            )
        )
        assert _call(system)["system_best_habitability"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# system_best_planet_size
# ---------------------------------------------------------------------------


class TestSystemBestPlanetSize:
    @pytest.mark.unit
    def test_returns_max_size(self) -> None:
        system = make_system(
            planets=(
                PlanetSummary(
                    id=1,
                    name="Small",
                    planet_class="pc_continental",
                    size=12,
                    is_habitable=True,
                    is_colonized=False,
                ),
                PlanetSummary(
                    id=2,
                    name="Large",
                    planet_class="pc_continental",
                    size=25,
                    is_habitable=True,
                    is_colonized=False,
                ),
            )
        )
        assert _call(system)["system_best_planet_size"] == pytest.approx(25.0)

    @pytest.mark.unit
    def test_no_planets_zero(self) -> None:
        system = make_system(planets=())
        assert _call(system)["system_best_planet_size"] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_single_planet(self) -> None:
        system = make_system(
            planets=(
                PlanetSummary(
                    id=1,
                    name="P",
                    planet_class="pc_continental",
                    size=20,
                    is_habitable=True,
                    is_colonized=False,
                ),
            )
        )
        assert _call(system)["system_best_planet_size"] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# system_district_slots_*
# ---------------------------------------------------------------------------


class TestSystemDistrictSlots:
    @pytest.mark.unit
    def test_slots_from_colonized_planet(self) -> None:
        planet = make_planet(
            id=10,
            district_slots=MappingProxyType(
                {"city": 3, "mining": 4, "farming": 2, "generator": 5}
            ),
        )
        empire = make_empire(owned_planets=(planet,))
        system = make_system(
            planets=(
                PlanetSummary(
                    id=10,
                    name="Home",
                    planet_class="pc_continental",
                    size=18,
                    is_habitable=True,
                    is_colonized=True,
                ),
            )
        )
        result = _call(system, empire)
        assert result["system_district_slots_city"] == pytest.approx(3.0)
        assert result["system_district_slots_mining"] == pytest.approx(4.0)
        assert result["system_district_slots_farming"] == pytest.approx(2.0)
        assert result["system_district_slots_generator"] == pytest.approx(5.0)

    @pytest.mark.unit
    def test_uncolonized_planet_slots_zero(self) -> None:
        # Planet in system but not in empire.owned_planets → no district data
        empire = make_empire(owned_planets=())
        system = make_system(
            planets=(
                PlanetSummary(
                    id=99,
                    name="Alien",
                    planet_class="pc_continental",
                    size=18,
                    is_habitable=True,
                    is_colonized=False,
                ),
            )
        )
        result = _call(system, empire)
        assert result["system_district_slots_city"] == pytest.approx(0.0)
        assert result["system_district_slots_mining"] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_two_planets_slots_summed(self) -> None:
        p1 = make_planet(
            id=1,
            district_slots=MappingProxyType(
                {"city": 2, "mining": 3, "farming": 1, "generator": 2}
            ),
        )
        p2 = make_planet(
            id=2,
            district_slots=MappingProxyType(
                {"city": 1, "mining": 2, "farming": 3, "generator": 1}
            ),
        )
        empire = make_empire(owned_planets=(p1, p2))
        system = make_system(
            planets=(
                PlanetSummary(
                    id=1,
                    name="P1",
                    planet_class="pc_continental",
                    size=18,
                    is_habitable=True,
                    is_colonized=True,
                ),
                PlanetSummary(
                    id=2,
                    name="P2",
                    planet_class="pc_ocean",
                    size=16,
                    is_habitable=True,
                    is_colonized=True,
                ),
            )
        )
        result = _call(system, empire)
        assert result["system_district_slots_city"] == pytest.approx(3.0)
        assert result["system_district_slots_mining"] == pytest.approx(5.0)
        assert result["system_district_slots_farming"] == pytest.approx(4.0)
        assert result["system_district_slots_generator"] == pytest.approx(3.0)

    @pytest.mark.unit
    def test_no_planets_all_zero(self) -> None:
        result = _call(make_system(planets=()))
        assert result["system_district_slots_city"] == pytest.approx(0.0)
        assert result["system_district_slots_mining"] == pytest.approx(0.0)
        assert result["system_district_slots_farming"] == pytest.approx(0.0)
        assert result["system_district_slots_generator"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# system_resource_deposit_value
# ---------------------------------------------------------------------------


class TestSystemResourceDepositValue:
    @pytest.mark.unit
    def test_sum_of_all_resources(self) -> None:
        system = make_system(
            resources=MappingProxyType({"minerals": 10.0, "energy": 5.0, "alloys": 3.0})
        )
        assert _call(system)["system_resource_deposit_value"] == pytest.approx(18.0)

    @pytest.mark.unit
    def test_empty_resources_zero(self) -> None:
        system = make_system(resources=MappingProxyType({}))
        assert _call(system)["system_resource_deposit_value"] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_single_resource(self) -> None:
        system = make_system(resources=MappingProxyType({"minerals": 7.0}))
        assert _call(system)["system_resource_deposit_value"] == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# system_hyperlane_degree
# ---------------------------------------------------------------------------


class TestSystemHyperlaneDegree:
    @pytest.mark.unit
    def test_degree_for_connected_system(self) -> None:
        s1 = make_system(id=1, hyperlane_connections=(2, 3))
        s2 = make_system(id=2, hyperlane_connections=(1,))
        s3 = make_system(id=3, hyperlane_connections=(1,))
        galactic = make_galactic(systems=(s1, s2, s3))
        g, btw = _make_graph_and_betweenness(galactic)
        result = extract_system_features(s1, make_empire(), galactic, g, btw)
        assert result["system_hyperlane_degree"] == pytest.approx(2.0)

    @pytest.mark.unit
    def test_degree_leaf_node(self) -> None:
        s1 = make_system(id=1, hyperlane_connections=(2,))
        s2 = make_system(id=2, hyperlane_connections=(1,))
        galactic = make_galactic(systems=(s1, s2))
        g, btw = _make_graph_and_betweenness(galactic)
        result = extract_system_features(s2, make_empire(), galactic, g, btw)
        assert result["system_hyperlane_degree"] == pytest.approx(1.0)

    @pytest.mark.unit
    def test_isolated_system_degree_zero(self) -> None:
        s1 = make_system(id=1, hyperlane_connections=())
        galactic = make_galactic(systems=(s1,))
        g, btw = _make_graph_and_betweenness(galactic)
        result = extract_system_features(s1, make_empire(), galactic, g, btw)
        assert result["system_hyperlane_degree"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# system_chokepoint_score
# ---------------------------------------------------------------------------


class TestSystemChokepointScore:
    @pytest.mark.unit
    def test_interior_node_nonzero(self) -> None:
        # Linear chain: 1-2-3-4 → system 2 and 3 are chokepoints
        s1 = make_system(id=1, hyperlane_connections=(2,))
        s2 = make_system(id=2, hyperlane_connections=(1, 3))
        s3 = make_system(id=3, hyperlane_connections=(2, 4))
        s4 = make_system(id=4, hyperlane_connections=(3,))
        galactic = make_galactic(systems=(s1, s2, s3, s4))
        g, btw = _make_graph_and_betweenness(galactic)
        result = extract_system_features(s2, make_empire(), galactic, g, btw)
        assert result["system_chokepoint_score"] > 0.0

    @pytest.mark.unit
    def test_endpoint_lower_than_interior(self) -> None:
        s1 = make_system(id=1, hyperlane_connections=(2,))
        s2 = make_system(id=2, hyperlane_connections=(1, 3))
        s3 = make_system(id=3, hyperlane_connections=(2,))
        galactic = make_galactic(systems=(s1, s2, s3))
        g, btw = _make_graph_and_betweenness(galactic)
        r1 = extract_system_features(s1, make_empire(), galactic, g, btw)
        r2 = extract_system_features(s2, make_empire(), galactic, g, btw)
        assert r2["system_chokepoint_score"] > r1["system_chokepoint_score"]

    @pytest.mark.unit
    def test_score_in_range(self) -> None:
        s1 = make_system(id=1, hyperlane_connections=(2,))
        s2 = make_system(id=2, hyperlane_connections=(1,))
        galactic = make_galactic(systems=(s1, s2))
        g, btw = _make_graph_and_betweenness(galactic)
        result = extract_system_features(s1, make_empire(), galactic, g, btw)
        assert 0.0 <= result["system_chokepoint_score"] <= 1.0


# ---------------------------------------------------------------------------
# system_distance_to_capital
# ---------------------------------------------------------------------------


class TestSystemDistanceToCapital:
    @pytest.mark.unit
    def test_adjacent_capital(self) -> None:
        # Capital planet (id=10) is on system 2; we're querying system 1
        capital_planet = make_planet(id=10, is_capital=True)
        empire = make_empire(id=1, owned_planets=(capital_planet,))
        s1 = make_system(id=1, hyperlane_connections=(2,), owner=1)
        s2 = make_system(
            id=2,
            hyperlane_connections=(1,),
            owner=1,
            planets=(
                PlanetSummary(
                    id=10,
                    name="Home",
                    planet_class="pc_continental",
                    size=18,
                    is_habitable=True,
                    is_colonized=True,
                ),
            ),
        )
        galactic = make_galactic(
            empires=(empire, make_empire(id=2)),
            player_empire_id=1,
            systems=(s1, s2),
        )
        g, btw = _make_graph_and_betweenness(galactic)
        result = extract_system_features(s1, empire, galactic, g, btw)
        assert result["system_distance_to_capital"] == pytest.approx(1.0)

    @pytest.mark.unit
    def test_same_system_as_capital(self) -> None:
        capital_planet = make_planet(id=10, is_capital=True)
        empire = make_empire(id=1, owned_planets=(capital_planet,))
        s1 = make_system(
            id=1,
            hyperlane_connections=(),
            owner=1,
            planets=(
                PlanetSummary(
                    id=10,
                    name="Home",
                    planet_class="pc_continental",
                    size=18,
                    is_habitable=True,
                    is_colonized=True,
                ),
            ),
        )
        galactic = make_galactic(
            empires=(empire, make_empire(id=2)),
            player_empire_id=1,
            systems=(s1,),
        )
        g, btw = _make_graph_and_betweenness(galactic)
        result = extract_system_features(s1, empire, galactic, g, btw)
        assert result["system_distance_to_capital"] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_two_hops_to_capital(self) -> None:
        capital_planet = make_planet(id=10, is_capital=True)
        empire = make_empire(id=1, owned_planets=(capital_planet,))
        s1 = make_system(id=1, hyperlane_connections=(2,), owner=1)
        s2 = make_system(id=2, hyperlane_connections=(1, 3), owner=1)
        s3 = make_system(
            id=3,
            hyperlane_connections=(2,),
            owner=1,
            planets=(
                PlanetSummary(
                    id=10,
                    name="Capital",
                    planet_class="pc_continental",
                    size=18,
                    is_habitable=True,
                    is_colonized=True,
                ),
            ),
        )
        galactic = make_galactic(
            empires=(empire, make_empire(id=2)),
            player_empire_id=1,
            systems=(s1, s2, s3),
        )
        g, btw = _make_graph_and_betweenness(galactic)
        result = extract_system_features(s1, empire, galactic, g, btw)
        assert result["system_distance_to_capital"] == pytest.approx(2.0)

    @pytest.mark.unit
    def test_no_capital_planet_returns_sentinel(self) -> None:
        empire = make_empire(owned_planets=())
        result = _call(make_system(id=1), empire)
        assert result["system_distance_to_capital"] == pytest.approx(999.0)


# ---------------------------------------------------------------------------
# system_distance_to_hostile
# ---------------------------------------------------------------------------


class TestSystemDistanceToHostile:
    @pytest.mark.unit
    def test_adjacent_rival_system(self) -> None:
        rival = make_empire(id=2, name="Rival")
        player = make_empire(id=1, name="Player", rivals=(2,))
        # system 1 (player) - system 2 (rival)
        s1 = make_system(id=1, hyperlane_connections=(2,), owner=1)
        s2 = make_system(id=2, hyperlane_connections=(1,), owner=2)
        galactic = make_galactic(
            empires=(player, rival),
            player_empire_id=1,
            systems=(s1, s2),
        )
        g, btw = _make_graph_and_betweenness(galactic)
        result = extract_system_features(s1, player, galactic, g, btw)
        assert result["system_distance_to_hostile"] == pytest.approx(1.0)

    @pytest.mark.unit
    def test_no_rivals_returns_sentinel(self) -> None:
        player = make_empire(id=1, rivals=())
        s1 = make_system(id=1, hyperlane_connections=(2,), owner=1)
        s2 = make_system(id=2, hyperlane_connections=(1,), owner=2)
        galactic = make_galactic(
            empires=(player, make_empire(id=2)),
            player_empire_id=1,
            systems=(s1, s2),
        )
        g, btw = _make_graph_and_betweenness(galactic)
        result = extract_system_features(s1, player, galactic, g, btw)
        assert result["system_distance_to_hostile"] == pytest.approx(999.0)

    @pytest.mark.unit
    def test_two_hops_to_hostile(self) -> None:
        rival = make_empire(id=2, name="Rival")
        player = make_empire(id=1, name="Player", rivals=(2,))
        s1 = make_system(id=1, hyperlane_connections=(2,), owner=1)
        s2 = make_system(id=2, hyperlane_connections=(1, 3), owner=1)
        s3 = make_system(id=3, hyperlane_connections=(2,), owner=2)
        galactic = make_galactic(
            empires=(player, rival),
            player_empire_id=1,
            systems=(s1, s2, s3),
        )
        g, btw = _make_graph_and_betweenness(galactic)
        result = extract_system_features(s1, player, galactic, g, btw)
        assert result["system_distance_to_hostile"] == pytest.approx(2.0)

    @pytest.mark.unit
    def test_unreachable_rival_returns_sentinel(self) -> None:
        rival = make_empire(id=2, name="Rival")
        player = make_empire(id=1, name="Player", rivals=(2,))
        # Disconnected components — rival exists but is unreachable
        s1 = make_system(id=1, hyperlane_connections=(), owner=1)
        s2 = make_system(id=2, hyperlane_connections=(), owner=2)
        galactic = make_galactic(
            empires=(player, rival),
            player_empire_id=1,
            systems=(s1, s2),
        )
        g, btw = _make_graph_and_betweenness(galactic)
        result = extract_system_features(s1, player, galactic, g, btw)
        assert result["system_distance_to_hostile"] == pytest.approx(999.0)


# ---------------------------------------------------------------------------
# system_climate_match_score
# ---------------------------------------------------------------------------


class TestSystemClimateMatchScore:
    @pytest.mark.unit
    def test_matching_planet_class(self) -> None:
        # Capital is pc_continental; system has a pc_continental habitable planet
        capital = make_planet(id=1, planet_class="pc_continental", is_capital=True)
        empire = make_empire(owned_planets=(capital,))
        system = make_system(
            id=10,
            planets=(
                PlanetSummary(
                    id=2,
                    name="Match",
                    planet_class="pc_continental",
                    size=18,
                    is_habitable=True,
                    is_colonized=False,
                ),
            ),
        )
        result = _call(system, empire)
        assert result["system_climate_match_score"] == pytest.approx(1.0)

    @pytest.mark.unit
    def test_no_matching_planet_class(self) -> None:
        capital = make_planet(id=1, planet_class="pc_continental", is_capital=True)
        empire = make_empire(owned_planets=(capital,))
        system = make_system(
            id=10,
            planets=(
                PlanetSummary(
                    id=2,
                    name="Desert",
                    planet_class="pc_desert",
                    size=18,
                    is_habitable=True,
                    is_colonized=False,
                ),
            ),
        )
        result = _call(system, empire)
        assert result["system_climate_match_score"] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_partial_match(self) -> None:
        capital = make_planet(id=1, planet_class="pc_continental", is_capital=True)
        empire = make_empire(owned_planets=(capital,))
        system = make_system(
            id=10,
            planets=(
                PlanetSummary(
                    id=2,
                    name="Match",
                    planet_class="pc_continental",
                    size=18,
                    is_habitable=True,
                    is_colonized=False,
                ),
                PlanetSummary(
                    id=3,
                    name="NoMatch",
                    planet_class="pc_desert",
                    size=16,
                    is_habitable=True,
                    is_colonized=False,
                ),
            ),
        )
        result = _call(system, empire)
        assert result["system_climate_match_score"] == pytest.approx(0.5)

    @pytest.mark.unit
    def test_no_habitable_planets_zero(self) -> None:
        capital = make_planet(id=1, planet_class="pc_continental", is_capital=True)
        empire = make_empire(owned_planets=(capital,))
        system = make_system(
            id=10,
            planets=(
                PlanetSummary(
                    id=2,
                    name="Gas",
                    planet_class="pc_gas_giant",
                    size=30,
                    is_habitable=False,
                    is_colonized=False,
                ),
            ),
        )
        result = _call(system, empire)
        assert result["system_climate_match_score"] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_no_capital_no_division_error(self) -> None:
        empire = make_empire(owned_planets=())
        system = make_system(
            id=10,
            planets=(
                PlanetSummary(
                    id=2,
                    name="P",
                    planet_class="pc_continental",
                    size=18,
                    is_habitable=True,
                    is_colonized=False,
                ),
            ),
        )
        result = _call(system, empire)
        assert isinstance(result["system_climate_match_score"], float)
