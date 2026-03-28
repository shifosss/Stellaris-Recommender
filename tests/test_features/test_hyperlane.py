"""Tests for hyperlane graph utilities.

Covers: build_hyperlane_graph, compute_betweenness,
        chokepoint_score, distance_to_system, distance_to_nearest.

All tests use synthetic galaxy graphs built directly from SystemState
fixtures to stay independent of the full save-file parser.
"""

from __future__ import annotations

import networkx as nx
import pytest

from stellaris_advisor.features.hyperlane import (
    build_hyperlane_graph,
    chokepoint_score,
    compute_betweenness,
    distance_to_nearest,
    distance_to_system,
)
from stellaris_advisor.parser.models import GalacticState

from .conftest import make_galactic, make_system

# ---------------------------------------------------------------------------
# Shared galaxy fixtures
# ---------------------------------------------------------------------------
#
# Linear chain:   1 -- 2 -- 3 -- 4
#
# Star topology:       2
#                      |
#                  3 - 1 - 4
#                      |
#                      5
#
# Disconnected:   1 -- 2    3 -- 4


@pytest.fixture
def linear_galactic() -> GalacticState:
    """Four systems in a chain: 1-2-3-4."""
    s1 = make_system(id=1, hyperlane_connections=(2,))
    s2 = make_system(id=2, hyperlane_connections=(1, 3))
    s3 = make_system(id=3, hyperlane_connections=(2, 4))
    s4 = make_system(id=4, hyperlane_connections=(3,))
    return make_galactic(systems=(s1, s2, s3, s4))


@pytest.fixture
def star_galactic() -> GalacticState:
    """Hub-and-spoke: node 1 connects to 2, 3, 4, 5."""
    hub = make_system(id=1, hyperlane_connections=(2, 3, 4, 5))
    s2 = make_system(id=2, hyperlane_connections=(1,))
    s3 = make_system(id=3, hyperlane_connections=(1,))
    s4 = make_system(id=4, hyperlane_connections=(1,))
    s5 = make_system(id=5, hyperlane_connections=(1,))
    return make_galactic(systems=(hub, s2, s3, s4, s5))


@pytest.fixture
def disconnected_galactic() -> GalacticState:
    """Two separate components: (1-2) and (3-4)."""
    s1 = make_system(id=1, hyperlane_connections=(2,))
    s2 = make_system(id=2, hyperlane_connections=(1,))
    s3 = make_system(id=3, hyperlane_connections=(4,))
    s4 = make_system(id=4, hyperlane_connections=(3,))
    return make_galactic(systems=(s1, s2, s3, s4))


# ---------------------------------------------------------------------------
# build_hyperlane_graph
# ---------------------------------------------------------------------------


class TestBuildHyperlaneGraph:
    @pytest.mark.unit
    def test_returns_networkx_graph(self, linear_galactic: GalacticState) -> None:
        g = build_hyperlane_graph(linear_galactic)
        assert isinstance(g, nx.Graph)

    @pytest.mark.unit
    def test_node_count_matches_systems(self, linear_galactic: GalacticState) -> None:
        g = build_hyperlane_graph(linear_galactic)
        assert g.number_of_nodes() == 4

    @pytest.mark.unit
    def test_edge_count_linear(self, linear_galactic: GalacticState) -> None:
        # 1-2, 2-3, 3-4 → 3 undirected edges
        g = build_hyperlane_graph(linear_galactic)
        assert g.number_of_edges() == 3

    @pytest.mark.unit
    def test_nodes_are_system_ids(self, linear_galactic: GalacticState) -> None:
        g = build_hyperlane_graph(linear_galactic)
        assert set(g.nodes()) == {1, 2, 3, 4}

    @pytest.mark.unit
    def test_edges_are_undirected(self, linear_galactic: GalacticState) -> None:
        g = build_hyperlane_graph(linear_galactic)
        assert g.has_edge(1, 2)
        assert g.has_edge(2, 1)  # undirected — same edge

    @pytest.mark.unit
    def test_star_topology_hub_degree(self, star_galactic: GalacticState) -> None:
        g = build_hyperlane_graph(star_galactic)
        assert g.degree(1) == 4

    @pytest.mark.unit
    def test_star_topology_spoke_degree(self, star_galactic: GalacticState) -> None:
        g = build_hyperlane_graph(star_galactic)
        for spoke in (2, 3, 4, 5):
            assert g.degree(spoke) == 1

    @pytest.mark.unit
    def test_no_self_loops(self, linear_galactic: GalacticState) -> None:
        g = build_hyperlane_graph(linear_galactic)
        assert nx.number_of_selfloops(g) == 0

    @pytest.mark.unit
    def test_no_duplicate_edges(self) -> None:
        # System 2 lists 1 in connections and system 1 lists 2 — should produce only one edge
        s1 = make_system(id=1, hyperlane_connections=(2,))
        s2 = make_system(id=2, hyperlane_connections=(1,))
        galactic = make_galactic(systems=(s1, s2))
        g = build_hyperlane_graph(galactic)
        assert g.number_of_edges() == 1

    @pytest.mark.unit
    def test_empty_systems(self) -> None:
        galactic = make_galactic(systems=())
        g = build_hyperlane_graph(galactic)
        assert g.number_of_nodes() == 0
        assert g.number_of_edges() == 0

    @pytest.mark.unit
    def test_isolated_node_no_connections(self) -> None:
        s1 = make_system(id=1, hyperlane_connections=())
        galactic = make_galactic(systems=(s1,))
        g = build_hyperlane_graph(galactic)
        assert g.number_of_nodes() == 1
        assert g.number_of_edges() == 0


# ---------------------------------------------------------------------------
# compute_betweenness
# ---------------------------------------------------------------------------


class TestComputeBetweenness:
    @pytest.mark.unit
    def test_returns_dict(self, linear_galactic: GalacticState) -> None:
        g = build_hyperlane_graph(linear_galactic)
        result = compute_betweenness(g)
        assert isinstance(result, dict)

    @pytest.mark.unit
    def test_keys_are_system_ids(self, linear_galactic: GalacticState) -> None:
        g = build_hyperlane_graph(linear_galactic)
        result = compute_betweenness(g)
        assert set(result.keys()) == {1, 2, 3, 4}

    @pytest.mark.unit
    def test_values_are_floats(self, linear_galactic: GalacticState) -> None:
        g = build_hyperlane_graph(linear_galactic)
        result = compute_betweenness(g)
        assert all(isinstance(v, float) for v in result.values())

    @pytest.mark.unit
    def test_values_in_range_zero_to_one(self, linear_galactic: GalacticState) -> None:
        g = build_hyperlane_graph(linear_galactic)
        result = compute_betweenness(g)
        assert all(0.0 <= v <= 1.0 for v in result.values())

    @pytest.mark.unit
    def test_linear_endpoints_lower_betweenness(
        self, linear_galactic: GalacticState
    ) -> None:
        # In a chain 1-2-3-4, interior nodes (2,3) have higher betweenness
        g = build_hyperlane_graph(linear_galactic)
        result = compute_betweenness(g)
        assert result[1] < result[2]
        assert result[4] < result[3]

    @pytest.mark.unit
    def test_star_hub_highest_betweenness(self, star_galactic: GalacticState) -> None:
        g = build_hyperlane_graph(star_galactic)
        result = compute_betweenness(g)
        hub_score = result[1]
        assert all(hub_score >= result[spoke] for spoke in (2, 3, 4, 5))

    @pytest.mark.unit
    def test_star_spokes_zero_betweenness(self, star_galactic: GalacticState) -> None:
        g = build_hyperlane_graph(star_galactic)
        result = compute_betweenness(g)
        for spoke in (2, 3, 4, 5):
            assert result[spoke] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_empty_graph_returns_empty_dict(self) -> None:
        g = nx.Graph()
        result = compute_betweenness(g)
        assert result == {}

    @pytest.mark.unit
    def test_single_node(self) -> None:
        g = nx.Graph()
        g.add_node(1)
        result = compute_betweenness(g)
        assert result == {1: 0.0}


# ---------------------------------------------------------------------------
# chokepoint_score
# ---------------------------------------------------------------------------


class TestChokepointScore:
    @pytest.mark.unit
    def test_returns_float(self, linear_galactic: GalacticState) -> None:
        g = build_hyperlane_graph(linear_galactic)
        btw = compute_betweenness(g)
        assert isinstance(chokepoint_score(btw, 2), float)

    @pytest.mark.unit
    def test_interior_node_higher_than_endpoint(
        self, linear_galactic: GalacticState
    ) -> None:
        g = build_hyperlane_graph(linear_galactic)
        btw = compute_betweenness(g)
        assert chokepoint_score(btw, 2) > chokepoint_score(btw, 1)
        assert chokepoint_score(btw, 3) > chokepoint_score(btw, 4)

    @pytest.mark.unit
    def test_star_hub_is_max_chokepoint(self, star_galactic: GalacticState) -> None:
        g = build_hyperlane_graph(star_galactic)
        btw = compute_betweenness(g)
        hub_score = chokepoint_score(btw, 1)
        for spoke in (2, 3, 4, 5):
            assert hub_score >= chokepoint_score(btw, spoke)

    @pytest.mark.unit
    def test_score_in_range(self, linear_galactic: GalacticState) -> None:
        g = build_hyperlane_graph(linear_galactic)
        btw = compute_betweenness(g)
        for node in g.nodes():
            assert 0.0 <= chokepoint_score(btw, node) <= 1.0

    @pytest.mark.unit
    def test_missing_node_returns_zero(self, linear_galactic: GalacticState) -> None:
        g = build_hyperlane_graph(linear_galactic)
        btw = compute_betweenness(g)
        assert chokepoint_score(btw, 999) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# distance_to_system
# ---------------------------------------------------------------------------


class TestDistanceToSystem:
    @pytest.mark.unit
    def test_returns_int_for_connected(self, linear_galactic: GalacticState) -> None:
        g = build_hyperlane_graph(linear_galactic)
        result = distance_to_system(g, 1, 4)
        assert isinstance(result, int)

    @pytest.mark.unit
    def test_distance_in_linear_chain(self, linear_galactic: GalacticState) -> None:
        g = build_hyperlane_graph(linear_galactic)
        assert distance_to_system(g, 1, 4) == 3
        assert distance_to_system(g, 1, 2) == 1
        assert distance_to_system(g, 2, 4) == 2

    @pytest.mark.unit
    def test_distance_to_self_is_zero(self, linear_galactic: GalacticState) -> None:
        g = build_hyperlane_graph(linear_galactic)
        assert distance_to_system(g, 1, 1) == 0

    @pytest.mark.unit
    def test_distance_is_symmetric(self, linear_galactic: GalacticState) -> None:
        g = build_hyperlane_graph(linear_galactic)
        assert distance_to_system(g, 1, 3) == distance_to_system(g, 3, 1)

    @pytest.mark.unit
    def test_returns_none_for_disconnected(
        self, disconnected_galactic: GalacticState
    ) -> None:
        g = build_hyperlane_graph(disconnected_galactic)
        assert distance_to_system(g, 1, 3) is None
        assert distance_to_system(g, 2, 4) is None

    @pytest.mark.unit
    def test_returns_none_for_missing_source(
        self, linear_galactic: GalacticState
    ) -> None:
        g = build_hyperlane_graph(linear_galactic)
        assert distance_to_system(g, 99, 1) is None

    @pytest.mark.unit
    def test_returns_none_for_missing_target(
        self, linear_galactic: GalacticState
    ) -> None:
        g = build_hyperlane_graph(linear_galactic)
        assert distance_to_system(g, 1, 99) is None

    @pytest.mark.unit
    def test_star_spoke_to_spoke_distance(self, star_galactic: GalacticState) -> None:
        g = build_hyperlane_graph(star_galactic)
        # All spokes route through hub: distance = 2
        assert distance_to_system(g, 2, 3) == 2
        assert distance_to_system(g, 2, 5) == 2


# ---------------------------------------------------------------------------
# distance_to_nearest
# ---------------------------------------------------------------------------


class TestDistanceToNearest:
    @pytest.mark.unit
    def test_returns_int_for_reachable_target(
        self, linear_galactic: GalacticState
    ) -> None:
        g = build_hyperlane_graph(linear_galactic)
        result = distance_to_nearest(g, 1, frozenset({3, 4}))
        assert isinstance(result, int)

    @pytest.mark.unit
    def test_nearest_of_multiple_targets(self, linear_galactic: GalacticState) -> None:
        g = build_hyperlane_graph(linear_galactic)
        # From 1: distance to 3 is 2, to 4 is 3 → nearest is 2
        assert distance_to_nearest(g, 1, frozenset({3, 4})) == 2

    @pytest.mark.unit
    def test_source_in_targets_is_zero(self, linear_galactic: GalacticState) -> None:
        g = build_hyperlane_graph(linear_galactic)
        assert distance_to_nearest(g, 1, frozenset({1, 4})) == 0

    @pytest.mark.unit
    def test_empty_targets_returns_none(self, linear_galactic: GalacticState) -> None:
        g = build_hyperlane_graph(linear_galactic)
        assert distance_to_nearest(g, 1, frozenset()) is None

    @pytest.mark.unit
    def test_all_targets_unreachable_returns_none(
        self, disconnected_galactic: GalacticState
    ) -> None:
        g = build_hyperlane_graph(disconnected_galactic)
        assert distance_to_nearest(g, 1, frozenset({3, 4})) is None

    @pytest.mark.unit
    def test_single_target(self, linear_galactic: GalacticState) -> None:
        g = build_hyperlane_graph(linear_galactic)
        assert distance_to_nearest(g, 1, frozenset({4})) == 3

    @pytest.mark.unit
    def test_missing_source_returns_none(self, linear_galactic: GalacticState) -> None:
        g = build_hyperlane_graph(linear_galactic)
        assert distance_to_nearest(g, 99, frozenset({1, 2})) is None
