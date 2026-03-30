# src/stellaris_advisor/planner/megastructure_advisor.py
"""Megastructure readiness detection, prerequisite chain, site selection, ROI."""
from __future__ import annotations

from stellaris_advisor.parser.models import EmpireState, GalacticState


def check_megastructure_readiness(
    empire: EmpireState,
    galactic: GalacticState,
) -> tuple[bool, tuple[str, ...]]:
    """Check if empire is ready to build megastructures.

    Returns (is_ready, missing_prerequisites).
    """
    missing: list[str] = []

    if "tech_mega_engineering" not in empire.researched_techs:
        missing.append("tech_mega_engineering")

    if "ap_galactic_wonders" not in empire.ascension_perks:
        missing.append("ap_galactic_wonders")

    alloy_net = empire.resource_income.get("alloys", 0) - empire.resource_expense.get(
        "alloys", 0
    )
    if alloy_net < 100:
        missing.append(f"alloy_surplus (current: {alloy_net:.0f}, need: 100+)")

    if empire.megastructures_under_construction > 0:
        missing.append("construction_slot (one already under construction)")

    return (len(missing) == 0, tuple(missing))
