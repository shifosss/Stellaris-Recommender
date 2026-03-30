# src/stellaris_advisor/planner/council_advisor.py
"""Council analysis: HoR expertise alignment, agenda optimization."""
from __future__ import annotations

from stellaris_advisor.parser.models import EmpireState


def analyze_council(empire: EmpireState) -> dict[str, str]:
    """Analyze council composition and return advisory dict.

    Returns dict with keys like 'hor_name', 'hor_expertise', 'recommendation'.
    """
    result: dict[str, str] = {}

    hor_leaders = [
        leader for leader in empire.leaders if leader.assignment == "head_of_research"
    ]
    if hor_leaders:
        hor = hor_leaders[0]
        result["hor_name"] = hor.name
        result["hor_level"] = str(hor.level)
        result["hor_expertise"] = hor.expertise or "none"

        # Check if a higher-level scientist is available
        scientists = [
            leader
            for leader in empire.leaders
            if leader.leader_class == "scientist" and leader.id != hor.id
        ]
        better = [s for s in scientists if s.level > hor.level]
        if better:
            best = max(better, key=lambda s: s.level)
            result["recommendation"] = (
                f"Consider reassigning HoR to {best.name}"
                f" (level {best.level} vs current {hor.level})"
            )
        else:
            result["recommendation"] = "Current Head of Research is optimal"
    else:
        result["no_hor"] = "true"
        result["recommendation"] = (
            "Assign a scientist as Head of Research to boost research speed"
        )

    return result
