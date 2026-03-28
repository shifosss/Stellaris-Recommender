"""System feature extraction functions.

All functions are pure: no mutation, no side effects.
Output is always dict[str, float].

Feature names match SYSTEM_FEATURE_NAMES in registry.py exactly.
"""

from __future__ import annotations

from collections.abc import Mapping

import networkx as nx

from stellaris_advisor.features.hyperlane import chokepoint_score, distance_to_nearest
from stellaris_advisor.parser.models import EmpireState, GalacticState, SystemState


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _find_capital_system_id(empire: EmpireState, galactic: GalacticState) -> int | None:
    """Return the system ID containing the empire's capital planet, or None."""
    capital_ids = frozenset(p.id for p in empire.owned_planets if p.is_capital)
    if not capital_ids:
        return None
    for system in galactic.systems:
        for planet in system.planets:
            if planet.id in capital_ids:
                return system.id
    return None


def _hostile_system_ids(empire: EmpireState, galactic: GalacticState) -> frozenset[int]:
    """Return IDs of systems owned by rival empires."""
    rival_ids = frozenset(empire.rivals)
    if not rival_ids:
        return frozenset()
    return frozenset(
        s.id for s in galactic.systems if s.owner is not None and s.owner in rival_ids
    )


# ---------------------------------------------------------------------------
# Public extractor
# ---------------------------------------------------------------------------


def extract_system_features(
    system: SystemState,
    empire: EmpireState,
    galactic: GalacticState,
    hyperlane_graph: nx.Graph,
    betweenness: dict[int, float],
) -> dict[str, float]:
    """Extract all 12 system-level features for a single system.

    Parameters
    ----------
    system:
        The system being evaluated.
    empire:
        The player empire (used for district lookup, capital, rival detection).
    galactic:
        Full game snapshot (used for capital and hostile system lookups).
    hyperlane_graph:
        Prebuilt undirected graph; call build_hyperlane_graph once per save.
    betweenness:
        Precomputed betweenness centrality; call compute_betweenness once per save.
    """
    planets = system.planets

    # --- Habitability & size ------------------------------------------------
    best_habitability: float = max(
        (1.0 if p.is_habitable else 0.0 for p in planets), default=0.0
    )
    best_planet_size: float = float(max((p.size for p in planets), default=0))

    # --- District slots from colonized empire planets -----------------------
    planet_slots = {p.id: p.district_slots for p in empire.owned_planets}
    city = mining = farming = generator = 0
    for psummary in planets:
        slots: Mapping[str, int] = planet_slots.get(psummary.id, {})
        city += slots.get("city", 0)
        mining += slots.get("mining", 0)
        farming += slots.get("farming", 0)
        generator += slots.get("generator", 0)

    # --- Resource deposit value ---------------------------------------------
    resource_deposit_value: float = sum(system.resources.values())

    # --- Hyperlane topology -------------------------------------------------
    hyperlane_degree: float = float(
        hyperlane_graph.degree(system.id) if system.id in hyperlane_graph else 0
    )
    choke: float = chokepoint_score(betweenness, system.id)

    # --- Distance to capital ------------------------------------------------
    capital_system = _find_capital_system_id(empire, galactic)
    if capital_system is not None and system.id in hyperlane_graph:
        dist_capital = distance_to_nearest(
            hyperlane_graph, system.id, frozenset({capital_system})
        )
        distance_to_capital: float = (
            float(dist_capital) if dist_capital is not None else 999.0
        )
    else:
        distance_to_capital = 999.0

    # --- Distance to hostile border -----------------------------------------
    hostile_ids = _hostile_system_ids(empire, galactic)
    if hostile_ids and system.id in hyperlane_graph:
        dist_hostile = distance_to_nearest(hyperlane_graph, system.id, hostile_ids)
        distance_to_hostile: float = (
            float(dist_hostile) if dist_hostile is not None else 999.0
        )
    else:
        distance_to_hostile = 999.0

    # --- Climate match score ------------------------------------------------
    capital_planets = [p for p in empire.owned_planets if p.is_capital]
    preferred_class = capital_planets[0].planet_class if capital_planets else ""
    habitable = [p for p in planets if p.is_habitable]
    if habitable and preferred_class:
        matching = sum(1 for p in habitable if p.planet_class == preferred_class)
        climate_match: float = matching / len(habitable)
    else:
        climate_match = 0.0

    return {
        "system_best_habitability": best_habitability,
        "system_best_planet_size": best_planet_size,
        "system_district_slots_city": float(city),
        "system_district_slots_mining": float(mining),
        "system_district_slots_farming": float(farming),
        "system_district_slots_generator": float(generator),
        "system_resource_deposit_value": resource_deposit_value,
        "system_hyperlane_degree": hyperlane_degree,
        "system_chokepoint_score": choke,
        "system_distance_to_capital": distance_to_capital,
        "system_distance_to_hostile": distance_to_hostile,
        "system_climate_match_score": climate_match,
    }
