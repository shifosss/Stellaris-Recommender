# src/stellaris_advisor/planner/faction_advisor.py
"""Faction approval analysis and policy change impact."""
from __future__ import annotations

from stellaris_advisor.parser.models import EmpireState
from stellaris_advisor.planner.models import FactionSummary


def analyze_factions(empire: EmpireState) -> FactionSummary:
    """Analyze faction approval and influence generation."""
    influence_income = empire.resource_income.get(
        "influence", 0.0
    ) - empire.resource_expense.get("influence", 0.0)

    top_faction = ""
    top_approval = 0.0
    problem_faction = None
    problem_approval = None

    if empire.factions:
        sorted_factions = sorted(empire.factions, key=lambda f: f.support, reverse=True)
        top = sorted_factions[0]
        top_faction = top.ethic
        top_approval = top.approval

        worst = min(empire.factions, key=lambda f: f.approval)
        if worst.approval < 50:
            problem_faction = worst.ethic
            problem_approval = worst.approval

    recommendation = ""
    if problem_faction and problem_approval is not None and problem_approval < 20:
        recommendation = (
            f"Faction '{problem_faction}' needs immediate attention — review policies"
        )
    elif influence_income < 2.0:
        recommendation = (
            "Influence income is low"
            " — consider declaring rivals or improving faction approval"
        )
    else:
        recommendation = "Faction situation is stable"

    return FactionSummary(
        influence_income=influence_income,
        top_faction=top_faction,
        top_faction_approval=top_approval,
        problem_faction=problem_faction,
        problem_faction_approval=problem_approval,
        recommendation=recommendation,
    )
