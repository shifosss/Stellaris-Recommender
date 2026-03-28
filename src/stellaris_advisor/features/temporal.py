"""Temporal feature extraction functions.

Computes delta features between the current empire state snapshot
and the most recent historical snapshot.

All functions are pure: no mutation, no side effects.
Output is always dict[str, float].

Feature names match TEMPORAL_FEATURE_NAMES in registry.py exactly.
"""

from __future__ import annotations

from stellaris_advisor.features.registry import TEMPORAL_FEATURE_NAMES

# ---------------------------------------------------------------------------
# Stall-detection thresholds
# ---------------------------------------------------------------------------

_STALL_THRESHOLD_ENERGY: float = 0.5
_STALL_THRESHOLD_MINERALS: float = 0.5
_STALL_THRESHOLD_ALLOYS: float = 0.1
_STALL_THRESHOLD_RESEARCH: float = 0.1
_STALL_THRESHOLD_EXPANSION: float = 0.01
_STALL_THRESHOLD_TECH: float = 0.01
_STALL_THRESHOLD_FLEET: float = 10.0

_ZERO_STATE: dict[str, float] = {name: 0.0 for name in TEMPORAL_FEATURE_NAMES}


def _research_total(state: dict[str, float]) -> float:
    """Sum of all three research speed values."""
    return (
        state.get("tech_physics_research_speed", 0.0)
        + state.get("tech_society_research_speed", 0.0)
        + state.get("tech_engineering_research_speed", 0.0)
    )


def extract_temporal_features(
    current: dict[str, float],
    history: tuple[dict[str, float], ...],
) -> dict[str, float]:
    """Extract 8 temporal delta features.

    Parameters
    ----------
    current:
        Empire feature dict for the current save (from extract_empire_features).
    history:
        Tuple of prior empire feature dicts, most-recent-first. Only
        history[0] is used; older entries are ignored.

    Returns all 0.0 when history is empty (no prior save to diff against).
    """
    if not history:
        return dict(_ZERO_STATE)

    prev = history[0]

    delta_energy = current.get("resource_net_energy", 0.0) - prev.get(
        "resource_net_energy", 0.0
    )
    delta_minerals = current.get("resource_net_minerals", 0.0) - prev.get(
        "resource_net_minerals", 0.0
    )
    delta_alloys = current.get("resource_net_alloys", 0.0) - prev.get(
        "resource_net_alloys", 0.0
    )
    delta_research = _research_total(current) - _research_total(prev)
    expansion_velocity = current.get("expansion_system_count", 0.0) - prev.get(
        "expansion_system_count", 0.0
    )
    tech_velocity = current.get("tech_total_count", 0.0) - prev.get(
        "tech_total_count", 0.0
    )
    delta_fleet_power = current.get("military_fleet_power", 0.0) - prev.get(
        "military_fleet_power", 0.0
    )

    stall_flag = (
        1.0
        if (
            abs(delta_energy) < _STALL_THRESHOLD_ENERGY
            and abs(delta_minerals) < _STALL_THRESHOLD_MINERALS
            and abs(delta_alloys) < _STALL_THRESHOLD_ALLOYS
            and abs(delta_research) < _STALL_THRESHOLD_RESEARCH
            and abs(expansion_velocity) < _STALL_THRESHOLD_EXPANSION
            and abs(tech_velocity) < _STALL_THRESHOLD_TECH
            and abs(delta_fleet_power) < _STALL_THRESHOLD_FLEET
        )
        else 0.0
    )

    return {
        "temporal_delta_energy": delta_energy,
        "temporal_delta_minerals": delta_minerals,
        "temporal_delta_alloys": delta_alloys,
        "temporal_delta_research": delta_research,
        "temporal_expansion_velocity": expansion_velocity,
        "temporal_tech_velocity": tech_velocity,
        "temporal_delta_fleet_power": delta_fleet_power,
        "temporal_stall_flag": stall_flag,
    }
