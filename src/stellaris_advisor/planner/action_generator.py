# src/stellaris_advisor/planner/action_generator.py
"""9 action generators producing legal candidate actions.

Each generator queries game state and optionally GameKnowledgeGraph
for available options, returning Recommendation objects with rule-based ROI.
All generators enforce game constraints — zero illegal actions.
"""
from __future__ import annotations

from typing import Any

from stellaris_advisor.config import PlannerConfig
from stellaris_advisor.parser.models import EmpireState, GalacticState
from stellaris_advisor.planner.models import Recommendation


def generate_economy_actions(
    empire: EmpireState,
    game_graph: Any,
) -> tuple[Recommendation, ...]:
    """Generate building and district construction candidates."""
    recs: list[Recommendation] = []

    for planet in empire.owned_planets:
        # Building slots available
        if planet.building_slots_used < planet.building_slots_total:
            available_count = planet.building_slots_total - planet.building_slots_used
            recs.append(
                Recommendation(
                    rank=0,
                    category="economy",
                    action_type="build_building",
                    description=f"Fill building slot on {planet.name}",
                    detail=f"{available_count} slots available on {planet.name}",
                    rule_roi=0.3 + 0.1 * available_count,
                    rl_delta=0.0,
                    final_score=0.0,
                    action_params=(
                        ("planet_id", str(planet.id)),
                        ("planet_name", planet.name),
                    ),
                )
            )

        # District slots available
        for dist_type, max_slots in planet.district_slots.items():
            current = planet.districts.get(dist_type, 0)
            if current < max_slots:
                recs.append(
                    Recommendation(
                        rank=0,
                        category="economy",
                        action_type="build_district",
                        description=f"Build {dist_type} district on {planet.name}",
                        detail=f"{current}/{max_slots} built",
                        rule_roi=0.2 + 0.05 * (max_slots - current),
                        rl_delta=0.0,
                        final_score=0.0,
                        action_params=(
                            ("planet_id", str(planet.id)),
                            ("district_type", dist_type),
                        ),
                    )
                )

    return tuple(recs)


def generate_tech_actions(
    empire: EmpireState,
    game_graph: Any,
) -> tuple[Recommendation, ...]:
    """Generate tech research prioritization candidates."""
    recs: list[Recommendation] = []

    for area, tech_id in empire.current_research.items():
        if tech_id:
            speed = empire.research_speed.get(area, 1.0)
            progress = empire.research_progress.get(area, 0.0)
            recs.append(
                Recommendation(
                    rank=0,
                    category="tech",
                    action_type="research_tech",
                    description=f"Continue researching {tech_id} ({area})",
                    detail=f"Progress: {progress:.0f}%, speed: {speed:.0f}/mo",
                    rule_roi=0.5 + 0.3 * (progress / 100.0),
                    rl_delta=0.0,
                    final_score=0.0,
                    action_params=(("tech_id", tech_id), ("area", area)),
                )
            )

    return tuple(recs)


def generate_expansion_actions(
    empire: EmpireState,
    galactic: GalacticState,
    config: PlannerConfig,
    colonization_scorer: Any = None,
) -> tuple[Recommendation, ...]:
    """Generate system colonization candidates."""
    recs: list[Recommendation] = []

    for system in galactic.systems:
        if system.owner is not None:
            continue
        habitable = [p for p in system.planets if p.is_habitable and not p.is_colonized]
        if not habitable:
            continue

        best_planet = max(habitable, key=lambda p: p.size)
        score = best_planet.size / 25.0  # normalize by max planet size

        recs.append(
            Recommendation(
                rank=0,
                category="expansion",
                action_type="colonize_system",
                description=(
                    f"Colonize {system.name}"
                    f" ({best_planet.name}, size {best_planet.size})"
                ),
                detail=f"Planet class: {best_planet.planet_class}",
                rule_roi=score,
                rl_delta=0.0,
                final_score=0.0,
                action_params=(
                    ("system_id", str(system.id)),
                    ("planet_id", str(best_planet.id)),
                ),
            )
        )

    recs.sort(key=lambda r: r.rule_roi, reverse=True)
    return tuple(recs[: config.top_n_colony_candidates])


def generate_military_actions(
    empire: EmpireState,
    galactic: GalacticState,
    config: PlannerConfig,
) -> tuple[Recommendation, ...]:
    """Generate ship building and military candidates."""
    recs: list[Recommendation] = []

    if empire.naval_capacity_max > 0:
        utilization = empire.naval_capacity_used / empire.naval_capacity_max
        if utilization < config.naval_utilization_threshold:
            gap = empire.naval_capacity_max - empire.naval_capacity_used
            recs.append(
                Recommendation(
                    rank=0,
                    category="military",
                    action_type="build_ships",
                    description=f"Build ships — {gap} naval capacity available",
                    detail=(
                        f"Utilization: {utilization:.0%},"
                        f" capacity: {empire.naval_capacity_used}"
                        f"/{empire.naval_capacity_max}"
                    ),
                    rule_roi=0.4 + 0.3 * (1.0 - utilization),
                    rl_delta=0.0,
                    final_score=0.0,
                    action_params=(("naval_gap", str(gap)),),
                )
            )

    for other in galactic.empires:
        if (
            other.id != empire.id
            and other.fleet_power > empire.fleet_power * config.threat_power_ratio
        ):
            recs.append(
                Recommendation(
                    rank=0,
                    category="military",
                    action_type="build_ships",
                    description=f"Urgent: military buildup against {other.name}",
                    detail=(
                        f"Threat ratio:"
                        f" {other.fleet_power / max(empire.fleet_power, 1):.1f}x"
                    ),
                    rule_roi=0.9,
                    rl_delta=0.0,
                    final_score=0.0,
                    action_params=(("threat_empire", other.name),),
                )
            )

    return tuple(recs)


def generate_tradition_actions(
    empire: EmpireState,
    game_graph: Any,
    year: int,
) -> tuple[Recommendation, ...]:
    """Generate tradition unlock and tree opening candidates."""
    recs: list[Recommendation] = []

    opened = set(empire.tradition_trees_opened)
    completed = set(empire.tradition_trees_completed)
    for tree in opened - completed:
        recs.append(
            Recommendation(
                rank=0,
                category="tradition",
                action_type="unlock_tradition",
                description=f"Continue tradition tree: {tree}",
                detail="Completing unlocks a perk slot",
                rule_roi=0.5,
                rl_delta=0.0,
                final_score=0.0,
                action_params=(("tree_id", tree),),
            )
        )

    return tuple(recs)


def generate_perk_actions(
    empire: EmpireState,
    game_graph: Any,
    year: int,
) -> tuple[Recommendation, ...]:
    """Generate ascension perk selection candidates."""
    recs: list[Recommendation] = []

    if empire.ascension_perk_slots_available > 0:
        recs.append(
            Recommendation(
                rank=0,
                category="perk",
                action_type="take_perk",
                description="Select an ascension perk",
                detail=(f"{empire.ascension_perk_slots_available} slot(s) available"),
                rule_roi=0.8,
                rl_delta=0.0,
                final_score=0.0,
                action_params=(),
            )
        )

    return tuple(recs)


def generate_megastructure_actions(
    empire: EmpireState,
    galactic: GalacticState,
    game_graph: Any,
) -> tuple[Recommendation, ...]:
    """Generate megastructure construction candidates."""
    recs: list[Recommendation] = []

    has_mega_eng = "tech_mega_engineering" in empire.researched_techs
    has_gw = "ap_galactic_wonders" in empire.ascension_perks

    if has_mega_eng and has_gw and empire.megastructures_under_construction == 0:
        alloy_income = empire.resource_income.get(
            "alloys", 0
        ) - empire.resource_expense.get("alloys", 0)
        if alloy_income > 100:
            recs.append(
                Recommendation(
                    rank=0,
                    category="megastructure",
                    action_type="begin_megastructure",
                    description="Begin megastructure construction",
                    detail=f"Alloy surplus: {alloy_income:.0f}/mo",
                    rule_roi=0.9,
                    rl_delta=0.0,
                    final_score=0.0,
                    action_params=(),
                )
            )

    return tuple(recs)


def generate_council_actions(
    empire: EmpireState,
    game_graph: Any,
) -> tuple[Recommendation, ...]:
    """Generate council and agenda optimization candidates."""
    recs: list[Recommendation] = []

    hor_leaders = [
        leader for leader in empire.leaders if leader.assignment == "head_of_research"
    ]
    if hor_leaders:
        hor = hor_leaders[0]
        scientists = [
            leader
            for leader in empire.leaders
            if leader.leader_class == "scientist" and leader.id != hor.id
        ]
        for sci in scientists:
            if sci.level > hor.level:
                recs.append(
                    Recommendation(
                        rank=0,
                        category="council",
                        action_type="reassign_hor",
                        description=(f"Reassign HoR to {sci.name} (level {sci.level})"),
                        detail=f"Current HoR: {hor.name} (level {hor.level})",
                        rule_roi=0.3 * (sci.level - hor.level),
                        rl_delta=0.0,
                        final_score=0.0,
                        action_params=(
                            ("leader_id", str(sci.id)),
                            ("new_level", str(sci.level)),
                        ),
                    )
                )

    return tuple(recs)


def generate_faction_actions(
    empire: EmpireState,
    game_graph: Any,
) -> tuple[Recommendation, ...]:
    """Generate faction management and rival declaration candidates."""
    recs: list[Recommendation] = []

    rival_count = len(empire.rivals)
    if rival_count < 3:
        recs.append(
            Recommendation(
                rank=0,
                category="faction",
                action_type="declare_rival",
                description=(
                    f"Declare rival (+0.5 influence/mo, current: {rival_count}/3)"
                ),
                detail=(
                    f"Current influence:"
                    f" {empire.resource_income.get('influence', 0):.1f}/mo"
                ),
                rule_roi=0.4,
                rl_delta=0.0,
                final_score=0.0,
                action_params=(("rival_count", str(rival_count)),),
            )
        )

    return tuple(recs)


def generate_all_actions(
    empire: EmpireState,
    galactic: GalacticState,
    config: PlannerConfig,
    game_graph: Any = None,
    colonization_scorer: Any = None,
) -> tuple[Recommendation, ...]:
    """Run all 9 generators and collect candidates."""
    all_recs: list[Recommendation] = []
    all_recs.extend(generate_economy_actions(empire, game_graph))
    all_recs.extend(generate_tech_actions(empire, game_graph))
    all_recs.extend(
        generate_expansion_actions(empire, galactic, config, colonization_scorer)
    )
    all_recs.extend(generate_military_actions(empire, galactic, config))
    all_recs.extend(generate_tradition_actions(empire, game_graph, galactic.year))
    all_recs.extend(generate_perk_actions(empire, game_graph, galactic.year))
    all_recs.extend(generate_megastructure_actions(empire, galactic, game_graph))
    all_recs.extend(generate_council_actions(empire, game_graph))
    all_recs.extend(generate_faction_actions(empire, game_graph))
    return tuple(all_recs)
