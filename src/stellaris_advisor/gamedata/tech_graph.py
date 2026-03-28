"""Tech DAG construction and query functions for the Stellaris gamedata layer."""

from __future__ import annotations

from typing import Any

import networkx as nx

from stellaris_advisor.gamedata.models import TechNode, WeightModifier
from stellaris_advisor.parser.extractor import (
    get_path,
    safe_bool,
    safe_float,
    safe_int,
    safe_str,
)

# Number of tier-N techs required in *researched* before tier-N+1 draws unlock.
_TIER_GATE_THRESHOLD: int = 6


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_prerequisites(data: dict[str, Any]) -> frozenset[str]:
    raw = data.get("prerequisites")
    if raw is None:
        return frozenset()
    if isinstance(raw, str):
        return frozenset({raw})
    if isinstance(raw, list):
        return frozenset(str(p) for p in raw)
    return frozenset()


def _extract_weight_modifiers(data: dict[str, Any]) -> tuple[WeightModifier, ...]:
    raw_wm = data.get("weight_modifier")
    if raw_wm is None:
        return ()
    wm_list: list[Any] = (
        [raw_wm]
        if isinstance(raw_wm, dict)
        else (raw_wm if isinstance(raw_wm, list) else [])
    )
    result: list[WeightModifier] = []
    for wm in wm_list:
        if not isinstance(wm, dict):
            continue
        factor = safe_float(get_path(wm, "factor"), default=1.0)
        nested = wm.get("modifier", {})
        conditions = (
            tuple(f"{k} = {v}" for k, v in nested.items() if k != "factor")
            if isinstance(nested, dict)
            else ()
        )
        result.append(WeightModifier(factor=factor, conditions=conditions))
    return tuple(result)


def _build_dag(tech_defs: dict[str, TechNode]) -> nx.DiGraph:
    dag: nx.DiGraph = nx.DiGraph()
    dag.add_nodes_from(tech_defs.keys())
    for tech_id, tech in tech_defs.items():
        for prereq_id in tech.prerequisites:
            dag.add_edge(prereq_id, tech_id)
    return dag


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_tech_defs(raw: dict[str, Any]) -> tuple[dict[str, TechNode], nx.DiGraph]:
    """Parse a raw Clausewitz tech dict into typed TechNode instances and a DAG.

    *raw* is the output of ``clausewitz.parse()`` applied to one or more
    technology definition files — a flat mapping of tech_id → attribute dict.

    Returns a ``(tech_defs, dag)`` pair where:
    - ``tech_defs`` maps tech IDs to frozen :class:`TechNode` dataclasses.
    - ``dag`` is a NetworkX DiGraph with an edge ``(prereq_id, tech_id)`` for
      every prerequisite relationship.
    """
    tech_defs: dict[str, TechNode] = {}

    for tech_id, data in raw.items():
        if not isinstance(data, dict):
            continue

        prerequisites = _extract_prerequisites(data)
        weight_modifiers = _extract_weight_modifiers(data)

        tech = TechNode(
            id=tech_id,
            name=safe_str(data.get("name"), default=tech_id),
            area=safe_str(data.get("area")),
            tier=safe_int(data.get("tier")),
            cost=safe_int(data.get("cost")),
            category=safe_str(data.get("category")),
            prerequisites=prerequisites,
            weight_modifiers=weight_modifiers,
            is_rare=safe_bool(data.get("is_rare")),
            is_dangerous=safe_bool(data.get("is_dangerous")),
            is_start_tech=safe_bool(data.get("start_tech")),
        )
        tech_defs[tech_id] = tech

    dag = _build_dag(tech_defs)
    return tech_defs, dag


def reachable_techs(
    tech_defs: dict[str, TechNode],
    tech_dag: nx.DiGraph,
    researched: frozenset[str],
    empire_type: str = "",
    ethics: frozenset[str] = frozenset(),
) -> frozenset[str]:
    """Return techs currently eligible for the research draw pool.

    A tech is reachable when ALL of the following hold:
    1. It is not already in *researched*.
    2. Every direct prerequisite is in *researched*.
    3. Tier gating: if ``tier > 0``, at least :data:`_TIER_GATE_THRESHOLD`
       techs from ``tier - 1`` must be in *researched*.

    *empire_type* and *ethics* are accepted for future filtering hooks but are
    not yet used in the base implementation.
    """
    tier_counts = tech_tier_counts(tech_defs, researched)

    result: set[str] = set()
    for tech_id, tech in tech_defs.items():
        if tech_id in researched:
            continue
        if not all(p in researched for p in tech.prerequisites):
            continue
        if tech.tier > 0 and tier_counts.get(tech.tier - 1, 0) < _TIER_GATE_THRESHOLD:
            continue
        result.add(tech_id)

    return frozenset(result)


def tech_path_to(
    tech_defs: dict[str, TechNode],
    tech_dag: nx.DiGraph,
    target: str,
    researched: frozenset[str],
) -> tuple[str, ...] | None:
    """Return an ordered sequence of techs to research to reach *target*.

    Uses topological ordering (BFS-consistent) over the subgraph of
    unresearched ancestors plus the target itself.

    Returns:
    - ``None`` if *target* is not in the DAG.
    - ``()`` if *target* is already in *researched*.
    - A topologically sorted ``tuple[str, ...]`` of unresearched techs
      (shallowest prerequisite first, target last).
    """
    if target not in tech_dag:
        return None
    if target in researched:
        return ()

    needed: set[str] = (nx.ancestors(tech_dag, target) - researched) | {target}
    subgraph = tech_dag.subgraph(needed)
    return tuple(nx.topological_sort(subgraph))


def tech_tier_counts(
    tech_defs: dict[str, TechNode],
    researched: frozenset[str],
    area: str = "",
) -> dict[int, int]:
    """Count researched techs per tier, optionally filtered by research *area*.

    Techs present in *researched* but absent from *tech_defs* are silently
    skipped. An empty *area* string means no area filter.
    """
    counts: dict[int, int] = {}
    for tech_id in researched:
        if tech_id not in tech_defs:
            continue
        tech = tech_defs[tech_id]
        if area and tech.area != area:
            continue
        counts[tech.tier] = counts.get(tech.tier, 0) + 1
    return counts


def validate_dag(tech_dag: nx.DiGraph) -> bool:
    """Return True if *tech_dag* is a valid directed acyclic graph."""
    return bool(nx.is_directed_acyclic_graph(tech_dag))


def downstream_unlocks(
    tech_defs: dict[str, TechNode],
    tech_dag: nx.DiGraph,
    tech_id: str,
) -> frozenset[str]:
    """Return all techs transitively unlocked by researching *tech_id*.

    Returns an empty frozenset if *tech_id* is not in the DAG.
    """
    if tech_id not in tech_dag:
        return frozenset()
    return frozenset(nx.descendants(tech_dag, tech_id))
