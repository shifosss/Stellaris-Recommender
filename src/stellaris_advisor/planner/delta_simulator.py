# src/stellaris_advisor/planner/delta_simulator.py
"""Approximate post-action feature vectors.

Pure function: returns a new dict, never mutates the input.
Approximation only — does NOT model second-order effects.
"""
from __future__ import annotations


def simulate_delta(
    features: dict[str, float],
    action_type: str,
    params: dict[str, str],
) -> dict[str, float]:
    """Return a new feature dict with approximate post-action modifications.

    Args:
        features: Current empire features (from extract_empire_features).
        action_type: Category of action being simulated.
        params: Action-specific parameters (all string values).

    Returns:
        New dict with modified features. Original is never mutated.
    """
    result = dict(features)  # shallow copy — all values are floats

    if action_type == "research_tech":
        result["tech_total_count"] = result.get("tech_total_count", 0) + 1
        area = params.get("area", "")
        if area == "physics":
            result["tech_physics_count"] = result.get("tech_physics_count", 0) + 1
        elif area == "society":
            result["tech_society_count"] = result.get("tech_society_count", 0) + 1
        elif area == "engineering":
            result["tech_engineering_count"] = (
                result.get("tech_engineering_count", 0) + 1
            )

    elif action_type == "build_building":
        net_output = float(params.get("net_output", "0"))
        result["resource_net_alloys"] = (
            result.get("resource_net_alloys", 0) + net_output
        )

    elif action_type == "build_district":
        net_output = float(params.get("net_output", "0"))
        resource_key = params.get("resource_key", "resource_net_energy")
        result[resource_key] = result.get(resource_key, 0) + net_output

    elif action_type == "colonize_system":
        result["expansion_system_count"] = result.get("expansion_system_count", 0) + 1
        result["expansion_planet_count"] = result.get("expansion_planet_count", 0) + 1

    elif action_type == "build_ships":
        gain = float(params.get("fleet_power_gain", "0"))
        result["military_fleet_power"] = result.get("military_fleet_power", 0) + gain

    elif action_type == "open_tradition":
        result["tradition_total_unlocked"] = (
            result.get("tradition_total_unlocked", 0) + 1
        )
        result["tradition_trees_opened"] = result.get("tradition_trees_opened", 0) + 1

    elif action_type == "complete_tradition":
        result["tradition_trees_completed"] = (
            result.get("tradition_trees_completed", 0) + 1
        )
        result["tradition_ascension_perk_count"] = result.get(
            "tradition_ascension_perk_count", 0
        )
        result["tradition_ascension_slots_available"] = (
            result.get("tradition_ascension_slots_available", 0) + 1
        )

    elif action_type == "take_perk":
        result["tradition_ascension_perk_count"] = (
            result.get("tradition_ascension_perk_count", 0) + 1
        )
        result["tradition_ascension_slots_available"] = max(
            0,
            result.get("tradition_ascension_slots_available", 0) - 1,
        )
        perk_id = params.get("perk_id", "")
        if perk_id == "ap_technological_ascendancy":
            for key in (
                "tech_physics_research_speed",
                "tech_society_research_speed",
                "tech_engineering_research_speed",
            ):
                result[key] = result.get(key, 0) * 1.10

    elif action_type == "begin_megastructure":
        result["megastructure_under_construction_count"] = (
            result.get("megastructure_under_construction_count", 0) + 1
        )
        alloy_cost_monthly = float(params.get("alloy_cost_monthly", "0"))
        result["resource_net_alloys"] = (
            result.get("resource_net_alloys", 0) - alloy_cost_monthly
        )

    elif action_type == "declare_rival":
        result["resource_net_influence"] = result.get("resource_net_influence", 0) + 0.5

    elif action_type == "change_policy":
        influence_delta = float(params.get("influence_delta", "0"))
        result["resource_net_influence"] = (
            result.get("resource_net_influence", 0) + influence_delta
        )

    elif action_type == "reassign_hor":
        new_level = float(params.get("new_level", "0"))
        result["leader_hor_level"] = new_level

    elif action_type == "switch_agenda":
        result["leader_has_active_agenda"] = 1.0

    return result
