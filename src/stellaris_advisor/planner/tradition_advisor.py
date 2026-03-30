# src/stellaris_advisor/planner/tradition_advisor.py
"""Tradition ordering + perk slot timing + lifetime value."""
from __future__ import annotations

from typing import Any

from stellaris_advisor.parser.models import EmpireState
from stellaris_advisor.planner.models import TraditionPerkPath


def suggest_tradition_path(
    empire: EmpireState,
    game_graph: Any,
    year: int,
) -> TraditionPerkPath:
    """Generate tradition and perk path guidance."""
    unity_income = empire.resource_income.get(
        "unity", 0.0
    ) - empire.resource_expense.get("unity", 0.0)

    # Identify recommended tradition
    opened = set(empire.tradition_trees_opened)
    completed = set(empire.tradition_trees_completed)
    in_progress = opened - completed
    recommended_tradition = None
    rationale = "No specific recommendation"

    if in_progress:
        recommended_tradition = sorted(in_progress)[0]
        rationale = (
            f"Complete '{recommended_tradition}' to unlock an ascension perk slot"
        )
    elif not opened:
        recommended_tradition = "discovery"
        rationale = "Discovery tree provides strong early-game research bonuses"

    return TraditionPerkPath(
        traditions_completed=len(empire.tradition_trees_completed),
        traditions_total=14,
        perks_taken=len(empire.ascension_perks),
        perks_total=empire.ascension_perk_slots_total,
        unity_income=unity_income,
        ascension_path=empire.ascension_path,
        recommended_tradition=recommended_tradition,
        tradition_rationale=rationale,
        recommended_perk=None,
        perk_rationale="",
    )
