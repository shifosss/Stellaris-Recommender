"""Building loader and query functions for the Stellaris gamedata layer."""

from __future__ import annotations

from typing import Any

import networkx as nx

from stellaris_advisor.gamedata.models import BuildingDef, OutputModifier, TechNode
from stellaris_advisor.gamedata.tech_graph import tech_path_to
from stellaris_advisor.parser.extractor import safe_float, safe_str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_modifiers(raw_dict: Any) -> tuple[OutputModifier, ...]:
    """Convert a parsed resource dict (e.g. ``{"minerals": 150}``) to OutputModifiers."""
    if not isinstance(raw_dict, dict):
        return ()
    return tuple(
        OutputModifier(resource=str(k), amount=safe_float(v))
        for k, v in raw_dict.items()
    )


def _extract_jobs(planet_modifier: Any) -> tuple[str, ...]:
    """Extract job IDs from keys of the form ``job_<name>_add``."""
    if not isinstance(planet_modifier, dict):
        return ()
    jobs: list[str] = []
    for key in planet_modifier:
        if key.startswith("job_") and key.endswith("_add"):
            jobs.append(key[4:-4])  # strip "job_" prefix and "_add" suffix
    return tuple(sorted(jobs))


def _extract_prerequisites(data: dict[str, Any]) -> frozenset[str]:
    raw = data.get("prerequisites")
    if raw is None:
        return frozenset()
    if isinstance(raw, str):
        return frozenset({raw})
    if isinstance(raw, list):
        return frozenset(str(p) for p in raw)
    return frozenset()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_building_defs(raw: dict[str, Any]) -> dict[str, BuildingDef]:
    """Parse a raw Clausewitz building dict into typed BuildingDef instances.

    *raw* is the output of ``clausewitz.parse()`` applied to one or more
    building definition files — a flat mapping of building_id → attribute dict.

    Encoding rules:
    - ``can_build = no`` → ``max_per_planet = 0`` (never player-buildable)
    - ``unique = yes``   → ``max_per_planet = 1`` (at most one per planet)
    - Otherwise          → ``max_per_planet = -1`` (unlimited)
    - Jobs are extracted from ``planet_modifier.job_<name>_add`` keys.
    """
    result: dict[str, BuildingDef] = {}

    for bid, data in raw.items():
        if not isinstance(data, dict):
            continue

        can_build = data.get("can_build")
        unique = data.get("unique")
        if can_build is False or can_build == "no":
            max_per_planet = 0
        elif unique is True or unique == "yes":
            max_per_planet = 1
        else:
            max_per_planet = -1

        result[bid] = BuildingDef(
            id=bid,
            name=safe_str(data.get("name"), default=bid),
            category=safe_str(data.get("category")),
            base_cost=_extract_modifiers(data.get("cost")),
            upkeep=_extract_modifiers(data.get("upkeep")),
            jobs=_extract_jobs(data.get("planet_modifier")),
            prerequisites=_extract_prerequisites(data),
            planet_classes=frozenset(),  # complex potential logic not parsed
            empire_types=frozenset(),  # complex potential logic not parsed
            max_per_planet=max_per_planet,
            upgrades_from=safe_str(data.get("upgrades_from")),
        )

    return result


def available_buildings(
    building_defs: dict[str, BuildingDef],
    researched: frozenset[str],
    planet_class: str = "",
    empire_type: str = "",
    existing_buildings: frozenset[str] = frozenset(),
    capital_level: int = 0,
) -> frozenset[str]:
    """Return building IDs that can currently be constructed.

    A building is available when ALL of the following hold:
    1. ``max_per_planet != 0`` (not flagged as never-buildable).
    2. All tech prerequisites are in *researched*.
    3. If ``planet_classes`` is non-empty, *planet_class* must be a member.
    4. If ``empire_types`` is non-empty, *empire_type* must be a member.
    5. If ``max_per_planet == 1`` (unique), the building is not in *existing_buildings*.
    """
    result: set[str] = set()
    for bid, building in building_defs.items():
        if building.max_per_planet == 0:
            continue
        if not all(p in researched for p in building.prerequisites):
            continue
        if building.planet_classes and planet_class not in building.planet_classes:
            continue
        if building.empire_types and empire_type not in building.empire_types:
            continue
        if building.max_per_planet == 1 and bid in existing_buildings:
            continue
        result.add(bid)
    return frozenset(result)


def building_unlock_chain(
    building_defs: dict[str, BuildingDef],
    tech_defs: dict[str, TechNode],
    tech_dag: nx.DiGraph,
    target_building: str,
    researched: frozenset[str],
) -> tuple[str, ...] | None:
    """Return the ordered sequence of techs needed to unlock *target_building*.

    Returns:
    - ``None`` if *target_building* is not in *building_defs*.
    - ``()`` if all tech prerequisites are already in *researched*.
    - An ordered tuple of tech IDs to research (prerequisites first).
    """
    if target_building not in building_defs:
        return None

    building = building_defs[target_building]
    if not building.prerequisites:
        return ()

    # Accumulate unresearched tech nodes needed across all building prerequisites.
    all_needed: set[str] = set()
    for tech_prereq in building.prerequisites:
        path = tech_path_to(tech_defs, tech_dag, tech_prereq, researched)
        if path is None:
            return None  # prerequisite tech is not in the DAG
        all_needed.update(path)

    if not all_needed:
        return ()

    subgraph = tech_dag.subgraph(all_needed)
    return tuple(nx.topological_sort(subgraph))


def estimate_building_net_output(
    building_def: BuildingDef,
    empire_state: Any,
) -> dict[str, float]:
    """Estimate the net resource impact of a building (upkeep only, as negative).

    Buildings produce resources through their jobs rather than directly.
    This function reports the building's direct upkeep cost as negative values.
    *empire_state* is accepted for future modifier application but not yet used.
    """
    result: dict[str, float] = {}
    for modifier in building_def.upkeep:
        result[modifier.resource] = result.get(modifier.resource, 0.0) - modifier.amount
    return result
