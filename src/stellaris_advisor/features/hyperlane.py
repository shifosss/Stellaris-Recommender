"""Hyperlane graph utilities.

All functions are pure: no mutation, no side effects.

Provides graph construction from GalacticState and graph-theoretic
metrics used by the system feature extractor.
"""

from __future__ import annotations

import networkx as nx

from stellaris_advisor.parser.models import GalacticState


def build_hyperlane_graph(galactic: GalacticState) -> nx.Graph:
    """Build an undirected hyperlane graph from galactic system connections.

    Nodes are system IDs (int). Edges represent hyperlane connections.
    Duplicate edges (both endpoints listing the other) are collapsed
    automatically by nx.Graph.
    """
    graph: nx.Graph = nx.Graph()
    for system in galactic.systems:
        graph.add_node(system.id)
        for neighbour in system.hyperlane_connections:
            graph.add_edge(system.id, neighbour)
    return graph


def compute_betweenness(graph: nx.Graph) -> dict[int, float]:
    """Return normalised betweenness centrality for every node.

    Uses networkx's built-in implementation (normalised by default so
    values lie in [0, 1]). Returns an empty dict for an empty graph.
    """
    if graph.number_of_nodes() == 0:
        return {}
    return dict(nx.betweenness_centrality(graph, normalized=True))


def chokepoint_score(betweenness: dict[int, float], system_id: int) -> float:
    """Return the chokepoint score for a system from a precomputed betweenness dict.

    Returns 0.0 if the system is absent from the dict.
    """
    return betweenness.get(system_id, 0.0)


def distance_to_system(graph: nx.Graph, source: int, target: int) -> int | None:
    """Return the shortest hop-count between source and target systems.

    Returns None if either node is absent or the systems are not connected.
    """
    if source not in graph or target not in graph:
        return None
    try:
        return int(nx.shortest_path_length(graph, source, target))
    except nx.NetworkXNoPath:
        return None


def distance_to_nearest(
    graph: nx.Graph, source: int, targets: frozenset[int]
) -> int | None:
    """Return the shortest hop-count from source to the nearest target system.

    Returns None if targets is empty, source is absent, or no target is
    reachable. Returns 0 if source is itself in targets.
    """
    if not targets or source not in graph:
        return None
    best: int | None = None
    for target in targets:
        d = distance_to_system(graph, source, target)
        if d is not None and (best is None or d < best):
            best = d
    return best
