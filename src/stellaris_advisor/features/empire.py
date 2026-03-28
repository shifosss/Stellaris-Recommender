"""Empire feature extraction functions.

All functions are pure: no mutation, no side effects.
Output is always dict[str, float].

Feature names match the registry in registry.py exactly.
"""

from __future__ import annotations

import re

from stellaris_advisor.parser.models import EmpireState, GalacticState

# Months-until-depletion returned when net income is non-negative.
_LARGE_MONTHS: float = 999.0

# Ship types classified as support vessels (not counted as combat).
_SUPPORT_SHIP_TYPES: frozenset[str] = frozenset(
    {"science_ship", "constructor", "colonizer", "transport"}
)

# All 10 core resources tracked for net income.
_CORE_RESOURCES: tuple[str, ...] = (
    "energy",
    "minerals",
    "food",
    "consumer_goods",
    "alloys",
    "unity",
    "influence",
    "physics_research",
    "society_research",
    "engineering_research",
)

# Resources with finite stockpiles where depletion and utilisation matter.
_FINITE_RESOURCES: tuple[str, ...] = (
    "energy",
    "minerals",
    "food",
    "consumer_goods",
    "alloys",
)


def _resource_features(empire: EmpireState) -> dict[str, float]:
    """Compute 20 resource-rate features.

    Categories:
    - resource_net_*              (10): net income per resource
    - resource_months_until_*     (5):  months until stockpile hits zero
    - resource_stockpile_util_*   (5):  stockpile / cap
    """
    income = empire.resource_income
    expense = empire.resource_expense
    stockpile = empire.resource_stockpile
    cap = empire.resource_storage_cap

    result: dict[str, float] = {}

    for resource in _CORE_RESOURCES:
        net = income.get(resource, 0.0) - expense.get(resource, 0.0)
        result[f"resource_net_{resource}"] = net

    for resource in _FINITE_RESOURCES:
        net = result[f"resource_net_{resource}"]
        stock = stockpile.get(resource, 0.0)
        storage = cap.get(resource, 0.0)

        result[f"resource_months_until_depletion_{resource}"] = (
            stock / (-net) if net < 0.0 else _LARGE_MONTHS
        )
        result[f"resource_stockpile_utilization_{resource}"] = (
            stock / storage if storage > 0.0 else 0.0
        )

    return result


def _economic_efficiency_features(empire: EmpireState) -> dict[str, float]:
    """Compute 8 economic-efficiency features.

    All per-pop values use max(total_pops, 1) to avoid division by zero.
    Per-district values use max(district_count, 1).
    """
    pops = max(float(empire.total_pops), 1.0)
    income = empire.resource_income
    expense = empire.resource_expense

    generator_districts: int = sum(
        p.districts.get("district_generator", 0) for p in empire.owned_planets
    )
    mining_districts: int = sum(
        p.districts.get("district_mining", 0) for p in empire.owned_planets
    )

    total_research = (
        income.get("physics_research", 0.0)
        + income.get("society_research", 0.0)
        + income.get("engineering_research", 0.0)
    )
    energy_net = income.get("energy", 0.0) - expense.get("energy", 0.0)
    minerals_net = income.get("minerals", 0.0) - expense.get("minerals", 0.0)

    return {
        "economic_alloys_per_pop": income.get("alloys", 0.0) / pops,
        "economic_research_per_pop": total_research / pops,
        "economic_energy_per_pop": income.get("energy", 0.0) / pops,
        "economic_minerals_per_pop": income.get("minerals", 0.0) / pops,
        "economic_consumer_goods_per_pop": income.get("consumer_goods", 0.0) / pops,
        "economic_unity_per_pop": income.get("unity", 0.0) / pops,
        "economic_energy_per_generator_district": (
            energy_net / max(generator_districts, 1)
        ),
        "economic_minerals_per_mining_district": (
            minerals_net / max(mining_districts, 1)
        ),
    }


def _expansion_features(
    empire: EmpireState, galactic: GalacticState
) -> dict[str, float]:
    """Compute 8 expansion features."""
    planets = empire.owned_planets
    total_systems = max(len(galactic.systems), 1)

    total_district_cap: int = sum(p.total_district_cap for p in planets)
    districts_used: int = sum(p.districts_used for p in planets)
    total_building_slots: int = sum(p.building_slots_total for p in planets)
    building_slots_used: int = sum(p.building_slots_used for p in planets)
    avg_size: float = sum(p.size for p in planets) / len(planets) if planets else 0.0

    return {
        "expansion_system_count": float(len(empire.owned_systems)),
        "expansion_planet_count": float(len(planets)),
        "expansion_avg_planet_size": avg_size,
        "expansion_total_district_slots": float(total_district_cap),
        "expansion_district_utilization": (districts_used / max(total_district_cap, 1)),
        "expansion_total_building_slots": float(total_building_slots),
        "expansion_building_utilization": (
            building_slots_used / max(total_building_slots, 1)
        ),
        "expansion_border_ratio": len(empire.owned_systems) / total_systems,
    }


def _military_features(
    empire: EmpireState, galactic: GalacticState
) -> dict[str, float]:
    """Compute 8 military features."""
    total_fp: float = sum(e.fleet_power for e in galactic.empires)
    relative_fp: float = empire.fleet_power / max(total_fp, 1.0)
    naval_util: float = empire.naval_capacity_used / max(empire.naval_capacity_max, 1)

    combat_ships: int = 0
    support_ships: int = 0
    for fleet in empire.fleets:
        for ship_type, count in fleet.ships.items():
            if ship_type in _SUPPORT_SHIP_TYPES:
                support_ships += count
            else:
                combat_ships += count
    total_ships: int = combat_ships + support_ships

    return {
        "military_fleet_power": empire.fleet_power,
        "military_fleet_power_relative": relative_fp,
        "military_naval_capacity_utilization": naval_util,
        "military_naval_capacity_max": float(empire.naval_capacity_max),
        "military_fleet_count": float(len(empire.fleets)),
        "military_combat_ship_ratio": combat_ships / max(total_ships, 1),
        "military_support_ship_ratio": support_ships / max(total_ships, 1),
        "military_army_strength": empire.total_army_strength,
    }


# ---------------------------------------------------------------------------
# Tech helpers
# ---------------------------------------------------------------------------

_TECH_SUFFIX_RE: re.Pattern[str] = re.compile(r"_(\d+)$")

# Keyword sets for heuristic tech-area classification (substring match on
# lowercased tech ID). Engineering checked first to avoid ambiguity.
_ENGINEERING_KW: frozenset[str] = frozenset(
    {
        "alloys",
        "mining_network",
        "construction",
        "destroyers",
        "cruisers",
        "battleships",
        "titans",
        "armies",
        "infantry",
        "megastructure",
        "gateway",
        "dyson",
        "ring_world",
        "habitats",
        "hull",
        "hangar",
    }
)

_PHYSICS_KW: frozenset[str] = frozenset(
    {
        "power_plant",
        "sensor",
        "shields",
        "laser",
        "plasma",
        "kinetic",
        "quantum",
        "antimatter",
        "jump_drive",
        "wormhole",
        "dark_matter",
        "zero_point",
        "science_lab",
        "science_nexus",
        "particle",
    }
)


def _classify_tech(tech_id: str) -> str:
    """Return 'physics', 'engineering', or 'society' based on tech ID heuristics."""
    lower = tech_id.lower()
    for kw in _ENGINEERING_KW:
        if kw in lower:
            return "engineering"
    for kw in _PHYSICS_KW:
        if kw in lower:
            return "physics"
    return "society"


def _tech_tier(tech_id: str) -> int:
    """Return tier 1-3 inferred from trailing _N suffix (capped at 3)."""
    m = _TECH_SUFFIX_RE.search(tech_id)
    return min(int(m.group(1)), 3) if m else 1


# ---------------------------------------------------------------------------
# Tech features (12)
# ---------------------------------------------------------------------------


def _tech_features(empire: EmpireState) -> dict[str, float]:
    """Compute 12 technology features.

    Per-area and per-tier counts use keyword/suffix heuristics on tech IDs.
    Exact classification requires the GameKnowledgeGraph (future integration).
    """
    techs = empire.researched_techs
    speed = empire.research_speed

    physics_count: int = 0
    society_count: int = 0
    engineering_count: int = 0
    tier1: int = 0
    tier2: int = 0
    tier3: int = 0

    for tech_id in techs:
        area = _classify_tech(tech_id)
        if area == "physics":
            physics_count += 1
        elif area == "engineering":
            engineering_count += 1
        else:
            society_count += 1

        tier = _tech_tier(tech_id)
        if tier == 1:
            tier1 += 1
        elif tier == 2:
            tier2 += 1
        else:
            tier3 += 1

    return {
        "tech_total_count": float(len(techs)),
        "tech_physics_count": float(physics_count),
        "tech_society_count": float(society_count),
        "tech_engineering_count": float(engineering_count),
        "tech_tier1_count": float(tier1),
        "tech_tier2_count": float(tier2),
        "tech_tier3_count": float(tier3),
        "tech_physics_research_speed": speed.get("physics", 0.0),
        "tech_society_research_speed": speed.get("society", 0.0),
        "tech_engineering_research_speed": speed.get("engineering", 0.0),
        "tech_frontier_size": 0.0,  # Stub: requires KG integration
        "tech_repeatable_count": 0.0,  # Stub: requires KG (suffix heuristic misclassifies)
    }


# ---------------------------------------------------------------------------
# Planet portfolio helpers
# ---------------------------------------------------------------------------

_CONTINENTAL_CLASSES: frozenset[str] = frozenset(
    {"pc_continental", "pc_ocean", "pc_tropical"}
)
_ARID_CLASSES: frozenset[str] = frozenset({"pc_desert", "pc_arid", "pc_savanna"})
_ARCTIC_CLASSES: frozenset[str] = frozenset({"pc_arctic", "pc_tundra", "pc_alpine"})

_ECUMENOPOLIS_CLASS: str = "pc_city"
_RING_WORLD_CLASS: str = "pc_ringworld_habitable"
_HABITAT_CLASS: str = "pc_habitat"

_SPECIALIZED_DESIGNATIONS: frozenset[str] = frozenset(
    {
        "col_mining",
        "col_generator",
        "col_farming",
        "col_factory",
        "col_research",
        "col_fortress",
        "col_refinery",
    }
)


# ---------------------------------------------------------------------------
# Planet portfolio features (10)
# ---------------------------------------------------------------------------


def _planet_portfolio_features(empire: EmpireState) -> dict[str, float]:
    """Compute 10 planet portfolio features."""
    planets = empire.owned_planets
    total_planets = max(len(planets), 1)

    continental = sum(1 for p in planets if p.planet_class in _CONTINENTAL_CLASSES)
    arid = sum(1 for p in planets if p.planet_class in _ARID_CLASSES)
    arctic = sum(1 for p in planets if p.planet_class in _ARCTIC_CLASSES)

    # Average habitability: proxy via weighted pop happiness across all planets.
    total_weighted_happiness: float = sum(
        pg.happiness * pg.count for p in planets for pg in p.pop_groups
    )
    total_planet_pops: int = sum(pg.count for p in planets for pg in p.pop_groups)
    avg_habitability: float = total_weighted_happiness / max(total_planet_pops, 1)

    district_slots_remaining = sum(
        p.total_district_cap - p.districts_used for p in planets
    )
    building_slots_remaining = sum(
        p.building_slots_total - p.building_slots_used for p in planets
    )

    specialized = sum(1 for p in planets if p.designation in _SPECIALIZED_DESIGNATIONS)
    ecumenopolis = sum(1 for p in planets if p.planet_class == _ECUMENOPOLIS_CLASS)
    ring_world = sum(1 for p in planets if p.planet_class == _RING_WORLD_CLASS)
    habitat = sum(1 for p in planets if p.planet_class == _HABITAT_CLASS)

    return {
        "planet_climate_distribution_continental": continental / total_planets,
        "planet_climate_distribution_arid": arid / total_planets,
        "planet_climate_distribution_arctic": arctic / total_planets,
        "planet_avg_habitability": avg_habitability,
        "planet_total_district_slots_remaining": float(district_slots_remaining),
        "planet_total_building_slots_remaining": float(building_slots_remaining),
        "planet_specialized_world_count": float(specialized),
        "planet_ecumenopolis_count": float(ecumenopolis),
        "planet_ring_world_count": float(ring_world),
        "planet_habitat_count": float(habitat),
    }


# ---------------------------------------------------------------------------
# Pop features (6)
# ---------------------------------------------------------------------------


def _pop_features(empire: EmpireState) -> dict[str, float]:
    """Compute 6 population features."""
    pop_groups = empire.pop_groups

    total_weighted_happiness: float = sum(pg.happiness * pg.count for pg in pop_groups)
    total_counted: int = sum(pg.count for pg in pop_groups)
    avg_happiness: float = total_weighted_happiness / max(total_counted, 1)

    species_diversity: float = float(len({pg.species for pg in pop_groups}))

    factions = empire.factions
    if factions:
        max_support: float = max(f.support for f in factions)
        pop_faction_spread: float = 1.0 - max_support
    else:
        pop_faction_spread = 0.0

    return {
        "pop_total": float(empire.total_pops),
        "pop_avg_happiness": avg_happiness,
        "pop_unemployment_rate": 0.0,  # Requires job-slot data (future KG integration)
        "pop_species_diversity": species_diversity,
        "pop_faction_spread": pop_faction_spread,
        "pop_growth_rate": 0.0,  # Requires temporal state (computed in temporal extractor)
    }


# ---------------------------------------------------------------------------
# Diplomatic features (6)
# ---------------------------------------------------------------------------


def _diplomatic_features(empire: EmpireState) -> dict[str, float]:
    """Compute 6 diplomatic features."""
    return {
        "diplomatic_rival_count": float(len(empire.rivals)),
        "diplomatic_pact_count": float(len(empire.defensive_pacts)),
        "diplomatic_federation_member": (
            1.0 if empire.federation_id is not None else 0.0
        ),
        "diplomatic_subject_count": float(len(empire.subjects)),
        "diplomatic_is_overlord": 1.0 if empire.subjects else 0.0,
        "diplomatic_war_count": float(len(empire.active_wars)),
    }


# ---------------------------------------------------------------------------
# Archetype features (4)
# ---------------------------------------------------------------------------

_HIVE_AUTHORITY: str = "auth_hive_mind"
_MACHINE_AUTHORITY: str = "auth_machine_intelligence"
_MEGACORP_AUTHORITY: str = "auth_corporate"


def _archetype_features(empire: EmpireState) -> dict[str, float]:
    """Compute 4 one-hot archetype features based on empire authority."""
    authority = empire.authority
    return {
        "archetype_hive": 1.0 if authority == _HIVE_AUTHORITY else 0.0,
        "archetype_machine": 1.0 if authority == _MACHINE_AUTHORITY else 0.0,
        "archetype_megacorp": 1.0 if authority == _MEGACORP_AUTHORITY else 0.0,
        "archetype_standard": (
            0.0
            if authority in (_HIVE_AUTHORITY, _MACHINE_AUTHORITY, _MEGACORP_AUTHORITY)
            else 1.0
        ),
    }


# ---------------------------------------------------------------------------
# Tradition features (8)
# ---------------------------------------------------------------------------

_ASCENSION_PATH_ENCODING: dict[str, float] = {
    "psionic": 1.0,
    "synthetic": 2.0,
    "biological": 3.0,
}


def _tradition_features(empire: EmpireState) -> dict[str, float]:
    """Compute 8 tradition and ascension features."""
    trees_opened = max(len(empire.tradition_trees_opened), 1)
    total_unlocked = len(empire.traditions_unlocked)

    # cost_efficiency: avg traditions per opened tree (0.0 when no trees opened)
    cost_efficiency: float = (
        total_unlocked / trees_opened if empire.tradition_trees_opened else 0.0
    )

    path_type: float = _ASCENSION_PATH_ENCODING.get(empire.ascension_path or "", 0.0)

    return {
        "tradition_total_unlocked": float(total_unlocked),
        "tradition_trees_completed": float(len(empire.tradition_trees_completed)),
        "tradition_trees_opened": float(len(empire.tradition_trees_opened)),
        "tradition_ascension_perk_count": float(len(empire.ascension_perks)),
        "tradition_ascension_slots_available": float(
            empire.ascension_perk_slots_available
        ),
        "tradition_unity_income": empire.resource_income.get("unity", 0.0),
        "tradition_ascension_path_type": path_type,
        "tradition_cost_efficiency": cost_efficiency,
    }


# ---------------------------------------------------------------------------
# Leader features (5)
# ---------------------------------------------------------------------------

_RULER_ASSIGNMENT: str = "ruler"


def _leader_features(empire: EmpireState) -> dict[str, float]:
    """Compute 5 leader and council features."""
    # Head of Ruling Party level: first leader with assignment == "ruler"
    hor_level: float = 0.0
    for leader in empire.leaders:
        if leader.assignment == _RULER_ASSIGNMENT:
            hor_level = float(leader.level)
            break

    # Council average level
    positions = empire.council_positions
    council_avg: float = (
        sum(p.leader_level for p in positions) / len(positions) if positions else 0.0
    )

    has_agenda: float = 1.0 if empire.current_agenda is not None else 0.0

    return {
        "leader_hor_level": hor_level,
        "leader_council_avg_level": council_avg,
        "leader_council_legitimacy": empire.council_legitimacy,
        "leader_total_count": float(len(empire.leaders)),
        "leader_has_active_agenda": has_agenda,
    }


# ---------------------------------------------------------------------------
# Faction features (6)
# ---------------------------------------------------------------------------


def _faction_features(empire: EmpireState) -> dict[str, float]:
    """Compute 6 faction features."""
    factions = empire.factions
    if not factions:
        return {
            "faction_avg_approval": 0.0,
            "faction_min_approval": 0.0,
            "faction_max_approval": 0.0,
            "faction_count": 0.0,
            "faction_largest_size_ratio": 0.0,
            "faction_influence_per_approval_point": 0.0,
        }

    approvals = [f.approval for f in factions]
    total_support = sum(f.support for f in factions)
    avg_approval = sum(f.approval * f.support for f in factions) / max(
        total_support, 0.01
    )
    total_pops = max(float(empire.total_pops), 1.0)
    largest_size = max(f.size for f in factions)
    influence_income = empire.resource_income.get("influence", 0.0)
    # Guard: avg_approval may be 0.0
    influence_per_point = influence_income / max(avg_approval, 0.01)

    return {
        "faction_avg_approval": avg_approval,
        "faction_min_approval": min(approvals),
        "faction_max_approval": max(approvals),
        "faction_count": float(len(factions)),
        "faction_largest_size_ratio": largest_size / total_pops,
        "faction_influence_per_approval_point": influence_per_point,
    }


# ---------------------------------------------------------------------------
# Galactic Community features (2)
# ---------------------------------------------------------------------------


def _galcom_features(empire: EmpireState) -> dict[str, float]:
    """Compute 2 Galactic Community features."""
    return {
        "galcom_diplomatic_weight": empire.galcom_diplomatic_weight,
        "galcom_council_member": 1.0 if empire.galcom_council_member else 0.0,
    }


# ---------------------------------------------------------------------------
# Federation features (2)
# ---------------------------------------------------------------------------


def _federation_features(empire: EmpireState) -> dict[str, float]:
    """Compute 2 federation features (0.0 when not in a federation)."""
    fed = empire.federation
    if fed is None:
        return {"federation_level": 0.0, "federation_fleet_power": 0.0}
    return {
        "federation_level": float(fed.level),
        "federation_fleet_power": fed.fleet_power,
    }


# ---------------------------------------------------------------------------
# Megastructure features (4)
# ---------------------------------------------------------------------------

_DYSON_SPHERE_TYPE: str = "dyson_sphere"
_MATTER_DECOMPRESSOR_TYPE: str = "matter_decompressor"


def _megastructure_features(empire: EmpireState) -> dict[str, float]:
    """Compute 4 megastructure features."""
    megas = empire.megastructures
    types = {m.megastructure_type for m in megas}
    return {
        "megastructure_completed_count": float(len(megas)),
        "megastructure_under_construction_count": float(
            empire.megastructures_under_construction
        ),
        "megastructure_has_dyson_sphere": 1.0 if _DYSON_SPHERE_TYPE in types else 0.0,
        "megastructure_has_matter_decompressor": (
            1.0 if _MATTER_DECOMPRESSOR_TYPE in types else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# Top-level empire feature extractor
# ---------------------------------------------------------------------------


def extract_empire_features(
    empire: EmpireState, galactic: GalacticState
) -> dict[str, float]:
    """Extract all ~109 empire features.

    Returns a flat dict mapping every name in EMPIRE_FEATURE_NAMES to a float.
    All sub-functions are pure; this function has no side effects.
    Raises ValueError if produced keys do not match EMPIRE_FEATURE_NAMES exactly.
    """
    from stellaris_advisor.features.registry import EMPIRE_FEATURE_NAMES

    result: dict[str, float] = {}
    result.update(_resource_features(empire))
    result.update(_economic_efficiency_features(empire))
    result.update(_expansion_features(empire, galactic))
    result.update(_military_features(empire, galactic))
    result.update(_tech_features(empire))
    result.update(_planet_portfolio_features(empire))
    result.update(_pop_features(empire))
    result.update(_diplomatic_features(empire))
    result.update(_archetype_features(empire))
    result.update(_tradition_features(empire))
    result.update(_leader_features(empire))
    result.update(_faction_features(empire))
    result.update(_galcom_features(empire))
    result.update(_federation_features(empire))
    result.update(_megastructure_features(empire))

    expected = set(EMPIRE_FEATURE_NAMES)
    if result.keys() != expected:
        missing = expected - result.keys()
        extra = result.keys() - expected
        raise ValueError(f"Feature mismatch — missing: {missing}, extra: {extra}")

    return result
