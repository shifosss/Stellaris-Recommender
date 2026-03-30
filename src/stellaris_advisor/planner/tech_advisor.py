# src/stellaris_advisor/planner/tech_advisor.py
"""Tech path BFS + ROI scoring + step-by-step breakdown."""
from __future__ import annotations

from typing import Any

from stellaris_advisor.parser.models import EmpireState
from stellaris_advisor.planner.models import TechPathInsight, TechPathStep

GOAL_TECH_MAP: dict[str, tuple[str, ...]] = {
    "battleships": ("tech_battleships",),
    "cruisers": ("tech_cruisers",),
    "titans": ("tech_titans",),
    "megastructures": ("tech_mega_engineering",),
    "habitats": ("tech_habitat_1",),
    "colossus": ("tech_colossus",),
}


def resolve_tech_goal(keyword: str) -> tuple[str, ...]:
    """Resolve a keyword goal to tech IDs."""
    return GOAL_TECH_MAP.get(keyword.lower(), (keyword,))


def suggest_tech_path(
    empire: EmpireState,
    game_graph: Any,
    bottleneck: str = "",
) -> TechPathInsight | None:
    """Generate a tech path recommendation based on bottleneck.

    Returns None if no game_graph is available or no path found.
    """
    if game_graph is None:
        return None

    # Map bottleneck to tech goal
    bottleneck_goals: dict[str, str] = {
        "fleet_power": "battleships",
        "alloy_income": "tech_alloys_2",
        "research_total": "tech_physics_3",
    }
    goal = bottleneck_goals.get(bottleneck, "")
    if not goal:
        return None

    tech_ids = resolve_tech_goal(goal)
    target = tech_ids[0] if tech_ids else ""
    if not target or target in empire.researched_techs:
        return None

    # Use game_graph.tech_path_to if available
    path = game_graph.tech_path_to(target, empire.researched_techs)
    if path is None:
        return None

    steps: list[TechPathStep] = []
    total_months = 0.0
    for tech_id in path:
        tech_def = game_graph.tech_defs.get(tech_id)
        if tech_def is None:
            continue
        area = tech_def.area
        speed = empire.research_speed.get(area, 100.0)
        months = tech_def.cost / max(speed, 1.0)
        total_months += months
        steps.append(
            TechPathStep(
                tech_id=tech_id,
                tech_name=tech_def.name,
                area=area,
                tier=tech_def.tier,
                cost=tech_def.cost,
                estimated_months=months,
                unlocks=(*tech_def.unlocks_buildings, *tech_def.unlocks_districts),
            )
        )

    return TechPathInsight(
        goal=goal,
        steps=tuple(steps),
        total_months=total_months,
        bottleneck_addressed=bottleneck,
    )
