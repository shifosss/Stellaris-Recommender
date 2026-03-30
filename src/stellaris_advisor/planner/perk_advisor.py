# src/stellaris_advisor/planner/perk_advisor.py
"""Perk selection + ascension path recommendation."""
from __future__ import annotations

from typing import Any

from stellaris_advisor.parser.models import EmpireState


def suggest_perk(
    empire: EmpireState,
    game_graph: Any,
    year: int,
) -> str | None:
    """Suggest the best ascension perk to take, or None if no slot available."""
    if empire.ascension_perk_slots_available <= 0:
        return None

    taken = set(empire.ascension_perks)

    # Priority ordering for common perks
    priority_perks = [
        "ap_technological_ascendancy",
        "ap_interstellar_dominion",
        "ap_galactic_wonders",
        "ap_master_builders",
        "ap_arcology_project",
    ]

    for perk_id in priority_perks:
        if perk_id not in taken:
            if game_graph is not None:
                eligible = game_graph.eligible_perks(
                    frozenset(taken),
                    empire.researched_techs,
                    len(empire.tradition_trees_completed),
                    empire.authority,
                )
                if perk_id in eligible:
                    return perk_id
            else:
                return perk_id

    return None
