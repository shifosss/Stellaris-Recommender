"""Cross-linking and integrity validation for the Stellaris gamedata layer.

Two responsibilities:
1. Cross-link — populate derived fields on frozen dataclasses by returning
   new instances (immutable pattern with ``dataclasses.replace``).
2. Validate — run 11 integrity checks, each returning a :class:`ValidationResult`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import networkx as nx

from stellaris_advisor.gamedata.models import (
    AscensionPerkDef,
    BuildingDef,
    CouncilPositionDef,
    DistrictDef,
    FactionDef,
    MegastructureDef,
    PlanetClassDef,
    PolicyDef,
    ShipClassDef,
    TechNode,
    TraditionDef,
    TraditionTree,
)
from stellaris_advisor.gamedata.tech_graph import validate_dag

# Leader classes recognised by Stellaris as valid council position classes.
_VALID_LEADER_CLASSES: frozenset[str] = frozenset(
    {"commander", "official", "scientist"}
)


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationResult:
    """Result of a single integrity check."""

    check_name: str
    passed: bool
    errors: tuple[str, ...]


# ---------------------------------------------------------------------------
# Cross-linking
# ---------------------------------------------------------------------------


def cross_link_tech_unlocks(
    tech_defs: dict[str, TechNode],
    building_defs: dict[str, BuildingDef],
    district_defs: dict[str, DistrictDef],
    ship_class_defs: dict[str, ShipClassDef],
) -> dict[str, TechNode]:
    """Return new TechNode instances with unlocks_* fields populated.

    Scans building/district/ship defs for tech prerequisites and builds
    reverse lookups. Returns fresh (immutable) TechNode copies via
    ``dataclasses.replace``. The input *tech_defs* dict is not mutated.
    """
    unlocks_buildings: dict[str, list[str]] = {tid: [] for tid in tech_defs}
    unlocks_districts: dict[str, list[str]] = {tid: [] for tid in tech_defs}
    unlocks_ships: dict[str, list[str]] = {tid: [] for tid in tech_defs}

    for bid, building in building_defs.items():
        for prereq in building.prerequisites:
            if prereq in unlocks_buildings:
                unlocks_buildings[prereq].append(bid)

    for did, district in district_defs.items():
        for prereq in district.prerequisites:
            if prereq in unlocks_districts:
                unlocks_districts[prereq].append(did)

    for sid, ship in ship_class_defs.items():
        for prereq in ship.prerequisites:
            if prereq in unlocks_ships:
                unlocks_ships[prereq].append(sid)

    return {
        tid: dataclasses.replace(
            tech,
            unlocks_buildings=tuple(sorted(unlocks_buildings[tid])),
            unlocks_districts=tuple(sorted(unlocks_districts[tid])),
            unlocks_ship_sizes=tuple(sorted(unlocks_ships[tid])),
        )
        for tid, tech in tech_defs.items()
    }


# ---------------------------------------------------------------------------
# Individual integrity checks
# ---------------------------------------------------------------------------


def check_tech_dag_acyclicity(tech_dag: nx.DiGraph) -> ValidationResult:
    """Check 1: Tech DAG must be acyclic."""
    if validate_dag(tech_dag):
        return ValidationResult(
            check_name="tech_dag_acyclicity", passed=True, errors=()
        )
    cycles = list(nx.simple_cycles(tech_dag))
    error_msgs = tuple(f"Cycle detected: {' -> '.join(c)}" for c in cycles[:5])
    return ValidationResult(
        check_name="tech_dag_acyclicity", passed=False, errors=error_msgs
    )


def check_tech_prereq_refs(tech_defs: dict[str, TechNode]) -> ValidationResult:
    """Check 2: All tech prerequisites must reference known tech IDs."""
    errors: list[str] = []
    for tid, tech in tech_defs.items():
        for prereq in tech.prerequisites:
            if prereq not in tech_defs:
                errors.append(f"Tech '{tid}' has unknown prerequisite '{prereq}'")
    passed = len(errors) == 0
    return ValidationResult(
        check_name="tech_prereq_refs", passed=passed, errors=tuple(errors)
    )


def check_tech_tier_consistency(tech_defs: dict[str, TechNode]) -> ValidationResult:
    """Check 3: Each tech's prerequisites must have a strictly lower tier.

    Tier-0 techs are exempt — the constraint is vacuously satisfied.
    """
    errors: list[str] = []
    for tid, tech in tech_defs.items():
        if tech.tier == 0:
            continue
        for prereq_id in tech.prerequisites:
            if prereq_id not in tech_defs:
                continue  # caught by check_tech_prereq_refs
            prereq_tier = tech_defs[prereq_id].tier
            if prereq_tier >= tech.tier:
                errors.append(
                    f"Tech '{tid}' (tier {tech.tier}) has prerequisite "
                    f"'{prereq_id}' at tier {prereq_tier} (must be lower)"
                )
    passed = len(errors) == 0
    return ValidationResult(
        check_name="tech_tier_consistency", passed=passed, errors=tuple(errors)
    )


def check_building_tech_linkage(
    building_defs: dict[str, BuildingDef],
    tech_defs: dict[str, TechNode],
) -> ValidationResult:
    """Check 4: All building prerequisites must reference known tech IDs."""
    errors: list[str] = []
    for bid, building in building_defs.items():
        for prereq in building.prerequisites:
            if prereq not in tech_defs:
                errors.append(
                    f"Building '{bid}' has unknown tech prerequisite '{prereq}'"
                )
    passed = len(errors) == 0
    return ValidationResult(
        check_name="building_tech_linkage", passed=passed, errors=tuple(errors)
    )


def check_district_planet_class_validity(
    district_defs: dict[str, DistrictDef],
    planet_class_defs: dict[str, PlanetClassDef],
) -> ValidationResult:
    """Check 5: District planet_classes must reference known planet class IDs."""
    errors: list[str] = []
    for did, district in district_defs.items():
        for pc in district.planet_classes:
            if pc not in planet_class_defs:
                errors.append(
                    f"District '{did}' references unknown planet class '{pc}'"
                )
    passed = len(errors) == 0
    return ValidationResult(
        check_name="district_planet_class_validity",
        passed=passed,
        errors=tuple(errors),
    )


def check_tradition_tree_linkage(
    tradition_defs: dict[str, TraditionDef],
    tradition_tree_defs: dict[str, TraditionTree],
) -> ValidationResult:
    """Check 6: Tradition↔tree linkage must be consistent in both directions.

    Forward: each tradition's tree_id must reference a known TraditionTree.
    Reverse: each TraditionTree's tradition_ids must reference known TraditionDefs.
    """
    errors: list[str] = []
    for tid, tradition in tradition_defs.items():
        if tradition.tree_id not in tradition_tree_defs:
            errors.append(
                f"Tradition '{tid}' references unknown tree '{tradition.tree_id}'"
            )
    for tree_id, tree in tradition_tree_defs.items():
        for t_id in tree.tradition_ids:
            if t_id not in tradition_defs:
                errors.append(f"Tree '{tree_id}' lists unknown tradition '{t_id}'")
    passed = len(errors) == 0
    return ValidationResult(
        check_name="tradition_tree_linkage", passed=passed, errors=tuple(errors)
    )


def check_perk_prereq_consistency(
    ascension_perk_defs: dict[str, AscensionPerkDef],
) -> ValidationResult:
    """Check 7: All required_perks must reference known ascension perk IDs."""
    errors: list[str] = []
    for pid, perk in ascension_perk_defs.items():
        for req in perk.required_perks:
            if req not in ascension_perk_defs:
                errors.append(f"Perk '{pid}' requires unknown perk '{req}'")
    passed = len(errors) == 0
    return ValidationResult(
        check_name="perk_prereq_consistency", passed=passed, errors=tuple(errors)
    )


def check_perk_mutual_exclusion_symmetry(
    ascension_perk_defs: dict[str, AscensionPerkDef],
) -> ValidationResult:
    """Check 8: Perk mutual exclusions must be symmetric.

    If perk A lists perk B in excluded_perks, B must also list A.
    Also reports excluded perks that do not exist in the definitions.
    """
    errors: list[str] = []
    for pid, perk in ascension_perk_defs.items():
        for excl in perk.excluded_perks:
            if excl not in ascension_perk_defs:
                errors.append(f"Perk '{pid}' excludes unknown perk '{excl}'")
                continue
            if pid not in ascension_perk_defs[excl].excluded_perks:
                errors.append(
                    f"Perk '{pid}' excludes '{excl}' but '{excl}' "
                    f"does not reciprocally exclude '{pid}'"
                )
    passed = len(errors) == 0
    return ValidationResult(
        check_name="perk_mutual_exclusion_symmetry",
        passed=passed,
        errors=tuple(errors),
    )


def check_megastructure_linkage(
    megastructure_defs: dict[str, MegastructureDef],
    tech_defs: dict[str, TechNode],
    ascension_perk_defs: dict[str, AscensionPerkDef],
) -> ValidationResult:
    """Check 9: Megastructure tech prerequisites and required perks must be known."""
    errors: list[str] = []
    for mid, mega in megastructure_defs.items():
        for prereq in mega.prerequisites:
            if prereq not in tech_defs:
                errors.append(
                    f"Megastructure '{mid}' has unknown tech prerequisite '{prereq}'"
                )
        for perk in mega.required_perks:
            if perk not in ascension_perk_defs:
                errors.append(f"Megastructure '{mid}' requires unknown perk '{perk}'")
    passed = len(errors) == 0
    return ValidationResult(
        check_name="megastructure_linkage", passed=passed, errors=tuple(errors)
    )


def check_policy_faction_linkage(
    faction_defs: dict[str, FactionDef],
    policy_defs: dict[str, PolicyDef],
) -> ValidationResult:
    """Check 10: Each faction issue's desired_option must exist in policy_defs."""
    all_option_ids: frozenset[str] = frozenset(
        option.id for policy in policy_defs.values() for option in policy.options
    )
    errors: list[str] = []
    for fid, faction in faction_defs.items():
        for issue in faction.issues:
            if issue.desired_option and issue.desired_option not in all_option_ids:
                errors.append(
                    f"Faction '{fid}' issue '{issue.id}' references "
                    f"unknown policy option '{issue.desired_option}'"
                )
    passed = len(errors) == 0
    return ValidationResult(
        check_name="policy_faction_linkage", passed=passed, errors=tuple(errors)
    )


def check_council_leader_class(
    council_defs: dict[str, CouncilPositionDef],
) -> ValidationResult:
    """Check 11: Council position leader_class must be a recognised class."""
    errors: list[str] = []
    for pid, pos in council_defs.items():
        if pos.leader_class not in _VALID_LEADER_CLASSES:
            errors.append(
                f"Council position '{pid}' has invalid leader_class "
                f"'{pos.leader_class}' (expected one of "
                f"{sorted(_VALID_LEADER_CLASSES)})"
            )
    passed = len(errors) == 0
    return ValidationResult(
        check_name="council_leader_class", passed=passed, errors=tuple(errors)
    )


# ---------------------------------------------------------------------------
# Aggregate runner
# ---------------------------------------------------------------------------


def validate_game_data(
    tech_defs: dict[str, TechNode],
    tech_dag: nx.DiGraph,
    building_defs: dict[str, BuildingDef],
    district_defs: dict[str, DistrictDef],
    ship_class_defs: dict[str, ShipClassDef],
    tradition_trees: dict[str, TraditionTree],
    tradition_defs: dict[str, TraditionDef],
    perk_defs: dict[str, AscensionPerkDef],
    mega_defs: dict[str, MegastructureDef],
    policy_defs: dict[str, PolicyDef],
    faction_defs: dict[str, FactionDef],
    council_position_defs: dict[str, CouncilPositionDef],
    planet_class_defs: dict[str, PlanetClassDef] | None = None,
) -> tuple[ValidationResult, ...]:
    """Run all 11 integrity checks and return one ValidationResult per check."""
    return (
        check_tech_dag_acyclicity(tech_dag),
        check_tech_prereq_refs(tech_defs),
        check_tech_tier_consistency(tech_defs),
        check_building_tech_linkage(building_defs, tech_defs),
        check_district_planet_class_validity(district_defs, planet_class_defs or {}),
        check_tradition_tree_linkage(tradition_defs, tradition_trees),
        check_perk_prereq_consistency(perk_defs),
        check_perk_mutual_exclusion_symmetry(perk_defs),
        check_megastructure_linkage(mega_defs, tech_defs, perk_defs),
        check_policy_faction_linkage(faction_defs, policy_defs),
        check_council_leader_class(council_position_defs),
    )
