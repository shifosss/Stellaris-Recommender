# src/stellaris_advisor/planner/alerts.py
"""12 rule-based alert types across 4 severity levels."""
from __future__ import annotations

from stellaris_advisor.config import PlannerConfig
from stellaris_advisor.parser.models import EmpireState, GalacticState
from stellaris_advisor.planner.models import Alert, AlertSeverity


def generate_alerts(
    empire: EmpireState,
    galactic: GalacticState,
    config: PlannerConfig,
) -> tuple[Alert, ...]:
    """Generate all applicable alerts for the current game state.

    Returns tuple of Alert sorted by severity (HIGH first).
    """
    alerts: list[Alert] = []

    # --- HIGH SEVERITY ---

    # 1. Resource deficit imminent
    for resource in ("energy", "minerals", "alloys", "food", "consumer_goods"):
        net = empire.resource_income.get(resource, 0.0) - empire.resource_expense.get(
            resource, 0.0
        )
        if net < 0:
            stockpile = empire.resource_stockpile.get(resource, 0.0)
            months = stockpile / abs(net) if abs(net) > 0.01 else 999.0
            if months < 12:
                alerts.append(
                    Alert(
                        severity=AlertSeverity.HIGH,
                        category="resource",
                        message=(
                            f"{resource.replace('_', ' ').title()} deficit"
                            f" — {months:.0f} months to depletion"
                        ),
                        detail=f"Net: {net:+.1f}/mo, stockpile: {stockpile:.0f}",
                    )
                )

    # 2. Hostile neighbor threat (fires even with zero fleet — that's the worst case)
    for other in galactic.empires:
        if other.id != empire.id and other.id in set(empire.rivals):
            ratio = other.fleet_power / max(empire.fleet_power, 1.0)
            if ratio > config.threat_power_ratio:
                alerts.append(
                    Alert(
                        severity=AlertSeverity.HIGH,
                        category="military",
                        message=(
                            f"Hostile neighbor {other.name}"
                            f" has {ratio:.1f}x your fleet power"
                        ),
                        detail=(
                            f"Their fleet: {other.fleet_power:.0f},"
                            f" yours: {empire.fleet_power:.0f}"
                        ),
                    )
                )

    # 3. No ascension path chosen (warn 20 years before deadline)
    if (
        galactic.year > config.ascension_path_deadline_year - 20
        and empire.ascension_path is None
    ):
        alerts.append(
            Alert(
                severity=AlertSeverity.HIGH,
                category="ascension",
                message=(
                    f"No ascension path chosen"
                    f" — deadline year {config.ascension_path_deadline_year}"
                ),
                detail="Choose Synthetic, Cybernetic, Psionic, or Biomorphosis",
            )
        )

    # --- MEDIUM SEVERITY ---

    # 4. Faction crisis
    for faction in empire.factions:
        if (
            faction.approval < config.faction_approval_warning
            and faction.support > 0.15
        ):
            alerts.append(
                Alert(
                    severity=AlertSeverity.MEDIUM,
                    category="faction",
                    message=(
                        f"Faction '{faction.ethic}' approval critically low"
                        f" ({faction.approval:.0f}%)"
                    ),
                    detail=f"Support: {faction.support * 100:.0f}% of pops",
                )
            )

    # 5. Influence starved
    influence_net = empire.resource_income.get(
        "influence", 0.0
    ) - empire.resource_expense.get("influence", 0.0)
    if influence_net < config.influence_crisis_threshold:
        alerts.append(
            Alert(
                severity=AlertSeverity.MEDIUM,
                category="influence",
                message=f"Influence income low ({influence_net:.1f}/mo)",
                detail=(
                    "Consider declaring rivals, improving faction approval,"
                    " or adjusting policies"
                ),
            )
        )

    # 6. Perk slot unused
    if empire.ascension_perk_slots_available > 0:
        alerts.append(
            Alert(
                severity=AlertSeverity.MEDIUM,
                category="ascension",
                message=(
                    f"{empire.ascension_perk_slots_available}"
                    " ascension perk slot(s) available"
                ),
                detail="Select a perk to gain permanent empire-wide bonuses",
            )
        )

    # 7. HoR expertise mismatch
    hor_leaders = [
        leader for leader in empire.leaders if leader.assignment == "head_of_research"
    ]
    if hor_leaders:
        hor = hor_leaders[0]
        current_areas = set(empire.current_research.keys())
        if hor.expertise and not any(
            hor.expertise in empire.current_research.get(a, "") for a in current_areas
        ):
            alerts.append(
                Alert(
                    severity=AlertSeverity.MEDIUM,
                    category="council",
                    message=(
                        f"Head of Research expertise ({hor.expertise})"
                        " may not match research priorities"
                    ),
                    detail=(
                        "Consider reassigning for better tech draw weight alignment"
                    ),
                )
            )

    # --- LOW SEVERITY ---

    # 8. Naval cap underutilized
    if empire.naval_capacity_max > 0:
        utilization = empire.naval_capacity_used / empire.naval_capacity_max
        if utilization < 0.5:
            alerts.append(
                Alert(
                    severity=AlertSeverity.LOW,
                    category="military",
                    message=f"Naval capacity underutilized ({utilization:.0%})",
                    detail=(
                        f"Using {empire.naval_capacity_used}"
                        f"/{empire.naval_capacity_max}"
                    ),
                )
            )

    # 9. Building slots wasted
    if empire.owned_planets:
        total_slots = sum(p.building_slots_total for p in empire.owned_planets)
        used_slots = sum(p.building_slots_used for p in empire.owned_planets)
        if total_slots > 0 and used_slots / total_slots < 0.6:
            alerts.append(
                Alert(
                    severity=AlertSeverity.LOW,
                    category="economy",
                    message=f"Building slots underutilized ({used_slots}/{total_slots})",
                    detail="Fill empty building slots to maximize production",
                )
            )

    # 10. Megastructure readiness
    if galactic.year >= config.megastructure_readiness_year:
        has_mega_eng = "tech_mega_engineering" in empire.researched_techs
        has_gw = "ap_galactic_wonders" in empire.ascension_perks
        if (
            has_mega_eng
            and has_gw
            and empire.megastructures_under_construction == 0
            and not empire.megastructures
        ):
            alerts.append(
                Alert(
                    severity=AlertSeverity.LOW,
                    category="megastructure",
                    message="Megastructure construction ready but not started",
                    detail=(
                        "You have Mega-Engineering and Galactic Wonders — start building"
                    ),
                )
            )

    # --- INFO SEVERITY ---

    # 11. Tradition tree nearly complete
    for tree in empire.tradition_trees_opened:
        if tree not in empire.tradition_trees_completed:
            alerts.append(
                Alert(
                    severity=AlertSeverity.INFO,
                    category="tradition",
                    message=f"Tradition tree '{tree}' in progress",
                    detail="Completing it unlocks a perk slot",
                )
            )

    # 12. Council position empty
    if not empire.council_positions:
        alerts.append(
            Alert(
                severity=AlertSeverity.INFO,
                category="council",
                message="Council positions may be unfilled",
                detail=(
                    "Empty positions incur penalties"
                    " (e.g., -25% research speed for empty HoR)"
                ),
            )
        )

    # Sort: HIGH first, then MEDIUM, LOW, INFO
    severity_order = {
        AlertSeverity.HIGH: 0,
        AlertSeverity.MEDIUM: 1,
        AlertSeverity.LOW: 2,
        AlertSeverity.INFO: 3,
    }
    alerts.sort(key=lambda a: severity_order.get(a.severity, 4))

    return tuple(alerts)
