"""Extraction helpers and typed extraction from raw parsed Clausewitz dicts."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Callable

from stellaris_advisor.parser.models import (
    AgendaState,
    ClaimState,
    CouncilPosition,
    EmpireState,
    FactionIssueState,
    FactionState,
    FederationState,
    FleetState,
    LeaderState,
    LeaderSummary,
    MegastructureState,
    PlanetState,
    PlanetSummary,
    PopGroupState,
    StarbaseState,
    SystemState,
    WarState,
)

_EMPTY_MAP: MappingProxyType[str, Any] = MappingProxyType({})


# ---------------------------------------------------------------------------
# Path navigation
# ---------------------------------------------------------------------------


def get_path(data: dict[str, Any], path: str, default: Any = None) -> Any:
    """Navigate nested dicts by dot-separated path.

    Returns *default* when any intermediate key is missing or a non-dict
    intermediate is encountered.
    """
    current: Any = data
    for key in path.split("."):
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


# ---------------------------------------------------------------------------
# Safe type conversions
# ---------------------------------------------------------------------------


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert to float, returning *default* on failure."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Convert to int, returning *default* on failure."""
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def safe_str(value: Any, default: str = "") -> str:
    """Convert to str, returning *default* for None."""
    if value is None:
        return default
    return str(value)


def safe_bool(value: Any, default: bool = False) -> bool:
    """Convert to bool, returning *default* for None."""
    if value is None:
        return default
    return bool(value)


# ---------------------------------------------------------------------------
# Collection converters
# ---------------------------------------------------------------------------


def to_str_tuple(value: Any) -> tuple[str, ...]:
    """Convert list, None, or single string → tuple[str, ...]."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    return (str(value),)


def to_int_tuple(value: Any) -> tuple[int, ...]:
    """Convert list, None, or single int → tuple[int, ...]."""
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    return (int(value),)


def to_frozen_map(
    value: Any,
    value_converter: Callable[[Any], Any] = float,
) -> MappingProxyType[str, Any]:
    """Convert dict or None → MappingProxyType with converted values."""
    if not value or not isinstance(value, dict):
        return MappingProxyType({})
    return MappingProxyType({str(k): value_converter(v) for k, v in value.items()})


# ---------------------------------------------------------------------------
# Typed extraction — Empire
# ---------------------------------------------------------------------------


def extract_empire(raw: dict[str, Any], empire_id: int, version: str) -> EmpireState:
    """Convert a raw parsed empire dict into a frozen EmpireState."""
    gov = raw.get("government") or raw.get("gov") or {}
    ethos = raw.get("ethos", {}) or {}
    budget_month = get_path(raw, "budget.current_month", default={}) or {}
    naval = raw.get("naval_capacity", {}) or {}

    return EmpireState(
        id=empire_id,
        name=safe_str(raw.get("name"), default=""),
        authority=safe_str(gov.get("authority"), default=""),
        ethics=to_str_tuple(ethos.get("ethic")),
        civics=to_str_tuple(gov.get("civics")),
        origin=safe_str(raw.get("origin"), default=""),
        government_type=safe_str(gov.get("type"), default=""),
        resource_income=to_frozen_map(budget_month.get("income")),
        resource_expense=to_frozen_map(budget_month.get("expenses")),
        resource_stockpile=to_frozen_map(raw.get("resource_stockpile")),
        resource_storage_cap=to_frozen_map(raw.get("resource_storage_cap")),
        researched_techs=_extract_researched_techs(raw),
        current_research=to_frozen_map(
            get_path(raw, "tech_status.current_research"), value_converter=str
        ),
        research_progress=to_frozen_map(get_path(raw, "tech_status.research_progress")),
        research_speed=to_frozen_map(raw.get("research_speed")),
        stored_research=to_frozen_map(raw.get("stored_research")),
        fleet_power=safe_float(raw.get("fleet_power")),
        naval_capacity_used=safe_int(naval.get("used")),
        naval_capacity_max=safe_int(naval.get("max")),
        fleets=_extract_fleets(raw),
        owned_systems=to_int_tuple(raw.get("owned_systems")),
        owned_planets=(),  # Linked by save_handler after full parse
        starbases=_extract_starbases(raw),
        total_pops=safe_int(raw.get("total_pops")),
        pop_groups=_extract_pop_groups(raw),
        rivals=to_int_tuple(raw.get("rivals")),
        defensive_pacts=to_int_tuple(raw.get("defensive_pacts")),
        federation_id=raw.get("federation"),
        federation_power=(
            safe_float(raw.get("federation_power"))
            if raw.get("federation_power") is not None
            else None
        ),
        subjects=to_int_tuple(raw.get("subjects")),
        overlord=raw.get("overlord"),
        traditions_unlocked=to_str_tuple(raw.get("traditions")),
        tradition_trees_opened=to_str_tuple(raw.get("tradition_trees_opened")),
        tradition_trees_completed=to_str_tuple(raw.get("tradition_trees_completed")),
        ascension_perks=to_str_tuple(raw.get("ascension_perks")),
        ascension_perk_slots_available=safe_int(
            raw.get("ascension_perk_slots_available")
        ),
        ascension_perk_slots_total=safe_int(raw.get("ascension_perk_slots_total")),
        ascension_path=raw.get("ascension_path"),
        ascension_situation_progress=(
            safe_float(raw["ascension_situation_progress"])
            if raw.get("ascension_situation_progress") is not None
            else None
        ),
        leaders=_extract_leaders(raw),
        council_positions=_extract_council(raw),
        current_agenda=_extract_agenda(raw),
        council_legitimacy=safe_float(raw.get("council_legitimacy")),
        factions=_extract_factions(raw),
        policies=to_frozen_map(raw.get("policies"), value_converter=str),
        galcom_member=safe_bool(raw.get("galcom_member")),
        galcom_diplomatic_weight=safe_float(raw.get("galcom_diplomatic_weight")),
        galcom_council_member=safe_bool(raw.get("galcom_council_member")),
        galcom_custodian=safe_bool(raw.get("galcom_custodian")),
        galcom_emperor=safe_bool(raw.get("galcom_emperor")),
        active_resolutions=to_str_tuple(raw.get("active_resolutions")),
        federation=_extract_federation(raw),
        megastructures=_extract_megastructures(raw),
        megastructures_under_construction=safe_int(
            raw.get("megastructures_under_construction")
        ),
        active_wars=_extract_wars(raw),
        claims_placed=_extract_claims(raw),
        total_army_strength=safe_float(raw.get("total_army_strength")),
        victory_score=safe_float(raw.get("victory_score")),
        victory_rank=safe_int(raw.get("victory_rank")),
    )


def _extract_researched_techs(raw: dict[str, Any]) -> frozenset[str]:
    """Extract set of researched technology IDs from tech_status."""
    tech_status = raw.get("tech_status", {}) or {}
    technology = tech_status.get("technology", {}) or {}
    if not isinstance(technology, dict):
        return frozenset()
    techs: set[str] = set()
    for entry in technology.values():
        if isinstance(entry, dict):
            tech_id = entry.get("technology")
            if tech_id:
                techs.add(str(tech_id))
    return frozenset(techs)


def _extract_leaders(raw: dict[str, Any]) -> tuple[LeaderState, ...]:
    """Extract leader entries."""
    leaders_raw = raw.get("leaders", {}) or {}
    if not isinstance(leaders_raw, dict):
        return ()
    result: list[LeaderState] = []
    for entry in leaders_raw.values():
        if isinstance(entry, dict):
            result.append(
                LeaderState(
                    id=safe_int(entry.get("id")),
                    name=safe_str(entry.get("name")),
                    leader_class=safe_str(entry.get("class")),
                    level=safe_int(entry.get("level")),
                    traits=to_str_tuple(entry.get("traits")),
                    expertise=entry.get("expertise"),
                    assignment=entry.get("assignment"),
                )
            )
    return tuple(result)


def _extract_council(raw: dict[str, Any]) -> tuple[CouncilPosition, ...]:
    """Extract council position entries."""
    council_raw = raw.get("council_positions", {}) or {}
    if not isinstance(council_raw, dict):
        if isinstance(council_raw, list):
            return ()
        return ()
    result: list[CouncilPosition] = []
    for entry in council_raw.values():
        if isinstance(entry, dict):
            result.append(
                CouncilPosition(
                    position_type=safe_str(entry.get("position_type")),
                    leader_id=entry.get("leader_id"),
                    leader_name=entry.get("leader_name"),
                    leader_level=safe_int(entry.get("leader_level")),
                )
            )
    return tuple(result)


def _extract_agenda(raw: dict[str, Any]) -> AgendaState | None:
    """Extract current agenda if present."""
    agenda_raw = raw.get("current_agenda")
    if not isinstance(agenda_raw, dict):
        return None
    return AgendaState(
        agenda_type=safe_str(agenda_raw.get("agenda_type")),
        progress=safe_float(agenda_raw.get("progress")),
        is_launched=safe_bool(agenda_raw.get("is_launched")),
    )


def _extract_factions(raw: dict[str, Any]) -> tuple[FactionState, ...]:
    """Extract faction entries."""
    factions_raw = raw.get("factions", {}) or {}
    if not isinstance(factions_raw, dict):
        return ()
    result: list[FactionState] = []
    for entry in factions_raw.values():
        if isinstance(entry, dict):
            issues_raw = entry.get("issues", {}) or {}
            issues: list[FactionIssueState] = []
            if isinstance(issues_raw, dict):
                for issue_id, fulfilled in issues_raw.items():
                    issues.append(
                        FactionIssueState(
                            issue_id=str(issue_id),
                            is_fulfilled=safe_bool(fulfilled),
                        )
                    )
            result.append(
                FactionState(
                    ethic=safe_str(entry.get("ethic")),
                    approval=safe_float(entry.get("approval")),
                    size=safe_int(entry.get("size")),
                    support=safe_float(entry.get("support")),
                    issues=tuple(issues),
                )
            )
    return tuple(result)


def _extract_federation(raw: dict[str, Any]) -> FederationState | None:
    """Extract federation details if present."""
    fed_raw = raw.get("federation_details")
    if not isinstance(fed_raw, dict):
        return None
    return FederationState(
        federation_type=safe_str(fed_raw.get("federation_type")),
        level=safe_int(fed_raw.get("level")),
        cohesion=safe_float(fed_raw.get("cohesion")),
        experience=safe_float(fed_raw.get("experience")),
        fleet_power=safe_float(fed_raw.get("fleet_power")),
    )


def _extract_megastructures(raw: dict[str, Any]) -> tuple[MegastructureState, ...]:
    """Extract megastructure entries."""
    mega_raw = raw.get("megastructures", {}) or {}
    if not isinstance(mega_raw, dict):
        return ()
    result: list[MegastructureState] = []
    for entry in mega_raw.values():
        if isinstance(entry, dict):
            result.append(
                MegastructureState(
                    megastructure_type=safe_str(entry.get("type")),
                    stage=safe_int(entry.get("stage")),
                    system_id=safe_int(entry.get("system_id")),
                )
            )
    return tuple(result)


def _extract_wars(raw: dict[str, Any]) -> tuple[WarState, ...]:
    """Extract active war entries."""
    wars_raw = raw.get("active_wars", {}) or {}
    if not isinstance(wars_raw, dict):
        return ()
    result: list[WarState] = []
    for entry in wars_raw.values():
        if isinstance(entry, dict):
            result.append(
                WarState(
                    war_id=safe_int(entry.get("war_id")),
                    name=safe_str(entry.get("name")),
                    war_goal=safe_str(entry.get("war_goal")),
                    exhaustion=safe_float(entry.get("exhaustion")),
                    is_attacker=safe_bool(entry.get("is_attacker")),
                )
            )
    return tuple(result)


def _extract_claims(raw: dict[str, Any]) -> tuple[ClaimState, ...]:
    """Extract territorial claim entries."""
    claims_raw = raw.get("claims_placed", {}) or {}
    if not isinstance(claims_raw, dict):
        return ()
    result: list[ClaimState] = []
    for entry in claims_raw.values():
        if isinstance(entry, dict):
            result.append(
                ClaimState(
                    system_id=safe_int(entry.get("system_id")),
                    influence_cost=safe_int(entry.get("influence_cost")),
                )
            )
    return tuple(result)


def _extract_fleets(raw: dict[str, Any]) -> tuple[FleetState, ...]:
    """Extract fleet entries."""
    fleets_raw = raw.get("fleets", {}) or {}
    if not isinstance(fleets_raw, dict):
        return ()
    result: list[FleetState] = []
    for entry in fleets_raw.values():
        if isinstance(entry, dict):
            commander_raw = entry.get("commander")
            commander = None
            if isinstance(commander_raw, dict):
                commander = LeaderSummary(
                    id=safe_int(commander_raw.get("id")),
                    name=safe_str(commander_raw.get("name")),
                    level=safe_int(commander_raw.get("level")),
                )
            result.append(
                FleetState(
                    id=safe_int(entry.get("id")),
                    name=safe_str(entry.get("name")),
                    fleet_power=safe_float(entry.get("fleet_power")),
                    ships=to_frozen_map(entry.get("ships"), value_converter=int),
                    naval_capacity_used=safe_int(entry.get("naval_capacity_used")),
                    location_system=safe_int(entry.get("location_system")),
                    commander=commander,
                )
            )
    return tuple(result)


def _extract_starbases(raw: dict[str, Any]) -> tuple[StarbaseState, ...]:
    """Extract starbase entries."""
    sb_raw = raw.get("starbases", {}) or {}
    if not isinstance(sb_raw, dict):
        return ()
    result: list[StarbaseState] = []
    for entry in sb_raw.values():
        if isinstance(entry, dict):
            result.append(
                StarbaseState(
                    system_id=safe_int(entry.get("system_id")),
                    level=safe_str(entry.get("level")),
                    modules=to_str_tuple(entry.get("modules")),
                    buildings=to_str_tuple(entry.get("buildings")),
                    defense_fleet_power=safe_float(entry.get("defense_fleet_power")),
                )
            )
    return tuple(result)


def _extract_pop_groups(raw: dict[str, Any]) -> tuple[PopGroupState, ...]:
    """Extract population group entries."""
    groups_raw = raw.get("pop_groups", {}) or {}
    if not isinstance(groups_raw, dict):
        return ()
    result: list[PopGroupState] = []
    for entry in groups_raw.values():
        if isinstance(entry, dict):
            result.append(
                PopGroupState(
                    species=safe_str(entry.get("species")),
                    count=safe_int(entry.get("count")),
                    happiness=safe_float(entry.get("happiness")),
                )
            )
    return tuple(result)


# ---------------------------------------------------------------------------
# Typed extraction — Planet
# ---------------------------------------------------------------------------


def extract_planet(raw: dict[str, Any], planet_id: int, version: str) -> PlanetState:
    """Convert a raw parsed planet dict into a frozen PlanetState."""
    districts = raw.get("districts", {}) or {}
    district_slots = raw.get("district_slots", {}) or {}

    return PlanetState(
        id=planet_id,
        name=safe_str(raw.get("name")),
        planet_class=safe_str(raw.get("planet_class")),
        size=safe_int(raw.get("planet_size")),
        districts=to_frozen_map(districts, value_converter=int),
        district_slots=to_frozen_map(district_slots, value_converter=int),
        total_district_cap=safe_int(raw.get("total_district_cap")),
        districts_used=(
            sum(int(v) for v in districts.values())
            if isinstance(districts, dict)
            else 0
        ),
        buildings=to_str_tuple(raw.get("buildings")),
        building_slots_total=safe_int(raw.get("building_slots_total")),
        building_slots_used=len(raw.get("buildings", []) or []),
        district_specializations=to_frozen_map(
            raw.get("district_specializations"), value_converter=str
        ),
        pops=safe_int(raw.get("pop")),
        pop_groups=_extract_planet_pop_groups(raw),
        stability=safe_float(raw.get("stability")),
        crime=safe_float(raw.get("crime")),
        amenities=safe_float(raw.get("amenities")),
        amenities_usage=safe_float(raw.get("amenities_usage")),
        housing=safe_float(raw.get("free_housing")),
        housing_usage=safe_float(raw.get("housing_usage")),
        designation=safe_str(raw.get("designation")),
        planetary_features=to_str_tuple(raw.get("planetary_features")),
        is_capital=safe_bool(raw.get("is_capital")),
    )


def _extract_planet_pop_groups(raw: dict[str, Any]) -> tuple[PopGroupState, ...]:
    """Extract pop groups from a planet dict."""
    groups_raw = raw.get("pop_groups", {}) or {}
    if not isinstance(groups_raw, dict):
        return ()
    result: list[PopGroupState] = []
    for entry in groups_raw.values():
        if isinstance(entry, dict):
            result.append(
                PopGroupState(
                    species=safe_str(entry.get("species")),
                    count=safe_int(entry.get("count")),
                    happiness=safe_float(entry.get("happiness")),
                )
            )
    return tuple(result)


# ---------------------------------------------------------------------------
# Typed extraction — System
# ---------------------------------------------------------------------------


def extract_system(raw: dict[str, Any], system_id: int, version: str) -> SystemState:
    """Convert a raw parsed system dict into a frozen SystemState."""
    hyperlanes = raw.get("hyperlane", []) or []
    connections: tuple[int, ...] = ()
    if isinstance(hyperlanes, list):
        connections = tuple(
            safe_int(h.get("to")) for h in hyperlanes if isinstance(h, dict)
        )

    starbase_raw = raw.get("starbase")
    starbase = None
    if isinstance(starbase_raw, dict):
        starbase = StarbaseState(
            system_id=system_id,
            level=safe_str(starbase_raw.get("level")),
            modules=to_str_tuple(starbase_raw.get("modules")),
            buildings=to_str_tuple(starbase_raw.get("buildings")),
            defense_fleet_power=safe_float(starbase_raw.get("defense_fleet_power")),
        )

    planets_raw = raw.get("planets", []) or []
    planets: tuple[PlanetSummary, ...] = ()
    if isinstance(planets_raw, list):
        planet_entries: list[PlanetSummary] = []
        for p in planets_raw:
            if isinstance(p, dict):
                planet_entries.append(
                    PlanetSummary(
                        id=safe_int(p.get("id")),
                        name=safe_str(p.get("name")),
                        planet_class=safe_str(p.get("planet_class")),
                        size=safe_int(p.get("size")),
                        is_habitable=safe_bool(p.get("is_habitable")),
                        is_colonized=safe_bool(p.get("is_colonized")),
                    )
                )
        planets = tuple(planet_entries)

    return SystemState(
        id=system_id,
        name=safe_str(raw.get("name")),
        star_class=safe_str(raw.get("star_class")),
        hyperlane_connections=connections,
        owner=raw.get("owner"),
        claimed_by=to_int_tuple(raw.get("claimed_by")),
        starbase=starbase,
        planets=planets,
        resources=to_frozen_map(raw.get("resources")),
        has_megastructure=safe_bool(raw.get("has_megastructure")),
        megastructure_type=raw.get("megastructure_type"),
    )
