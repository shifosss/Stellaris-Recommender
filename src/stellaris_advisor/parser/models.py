"""Frozen dataclass models representing parsed Stellaris save state.

Every collection is immutable: tuple (not list), frozenset (for sets),
MappingProxyType (for dicts). All dataclasses use frozen=True.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType


# ---------------------------------------------------------------------------
# Supporting types (small, referenced by the core types)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PopGroupState:
    """A population group with species and happiness."""

    species: str
    count: int
    happiness: float


@dataclass(frozen=True)
class PopGroupSummary:
    """Lightweight population group reference."""

    species: str
    count: int


@dataclass(frozen=True)
class LeaderState:
    """Full leader representation."""

    id: int
    name: str
    leader_class: str
    level: int
    traits: tuple[str, ...]
    expertise: str | None
    assignment: str | None


@dataclass(frozen=True)
class LeaderSummary:
    """Lightweight leader reference."""

    id: int
    name: str
    level: int


@dataclass(frozen=True)
class CouncilPosition:
    """A position on the empire's governing council."""

    position_type: str
    leader_id: int | None
    leader_name: str | None
    leader_level: int


@dataclass(frozen=True)
class AgendaState:
    """The current ruling agenda."""

    agenda_type: str
    progress: float
    is_launched: bool


@dataclass(frozen=True)
class FactionIssueState:
    """A single faction issue and its fulfillment status."""

    issue_id: str
    is_fulfilled: bool


@dataclass(frozen=True)
class FactionState:
    """A political faction within an empire."""

    ethic: str
    approval: float
    size: int
    support: float
    issues: tuple[FactionIssueState, ...]


@dataclass(frozen=True)
class FederationState:
    """Federation membership details."""

    federation_type: str
    level: int
    cohesion: float
    experience: float
    fleet_power: float


@dataclass(frozen=True)
class MegastructureState:
    """A megastructure owned by an empire."""

    megastructure_type: str
    stage: int
    system_id: int


@dataclass(frozen=True)
class WarState:
    """An active war involving the empire."""

    war_id: int
    name: str
    war_goal: str
    exhaustion: float
    is_attacker: bool


@dataclass(frozen=True)
class ClaimState:
    """A territorial claim placed by the empire."""

    system_id: int
    influence_cost: int


@dataclass(frozen=True)
class StarbaseState:
    """A starbase in a system."""

    system_id: int
    level: str
    modules: tuple[str, ...]
    buildings: tuple[str, ...]
    defense_fleet_power: float


@dataclass(frozen=True)
class FleetState:
    """A military fleet."""

    id: int
    name: str
    fleet_power: float
    ships: MappingProxyType[str, int]
    naval_capacity_used: int
    location_system: int
    commander: LeaderSummary | None


@dataclass(frozen=True)
class PlanetSummary:
    """Lightweight planet reference."""

    id: int
    name: str
    planet_class: str
    size: int
    is_habitable: bool
    is_colonized: bool


@dataclass(frozen=True)
class ColonyCandidate:
    """A planet evaluated for colonization."""

    system_id: int
    system_name: str
    planet_id: int
    planet_name: str
    planet_class: str
    size: int
    habitability: float
    district_slots: MappingProxyType[str, int]
    resource_deposits: MappingProxyType[str, float]
    distance_to_capital: int
    distance_to_hostile_border: int
    is_in_claimed_space: bool


@dataclass(frozen=True)
class EmpireSummary:
    """Lightweight empire reference for rankings and comparisons."""

    id: int
    name: str
    authority: str
    ethics: tuple[str, ...]
    fleet_power: float
    tech_count: int
    system_count: int
    victory_score: float


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlanetState:
    """Full state of a colonized planet."""

    id: int
    name: str
    planet_class: str
    size: int
    districts: MappingProxyType[str, int]
    district_slots: MappingProxyType[str, int]
    total_district_cap: int
    districts_used: int
    buildings: tuple[str, ...]
    building_slots_total: int
    building_slots_used: int
    district_specializations: MappingProxyType[str, str]
    pops: int
    pop_groups: tuple[PopGroupState, ...]
    stability: float
    crime: float
    amenities: float
    amenities_usage: float
    housing: float
    housing_usage: float
    designation: str
    planetary_features: tuple[str, ...]
    is_capital: bool


@dataclass(frozen=True)
class SystemState:
    """Full state of a galactic system."""

    id: int
    name: str
    star_class: str
    hyperlane_connections: tuple[int, ...]
    owner: int | None
    claimed_by: tuple[int, ...]
    starbase: StarbaseState | None
    planets: tuple[PlanetSummary, ...]
    resources: MappingProxyType[str, float]
    has_megastructure: bool
    megastructure_type: str | None


@dataclass(frozen=True)
class EmpireState:
    """Full state of an empire at a point in time."""

    # --- Identity ---
    id: int
    name: str
    authority: str
    ethics: tuple[str, ...]
    civics: tuple[str, ...]
    origin: str
    government_type: str

    # --- Resources ---
    resource_income: MappingProxyType[str, float]
    resource_expense: MappingProxyType[str, float]
    resource_stockpile: MappingProxyType[str, float]
    resource_storage_cap: MappingProxyType[str, float]

    # --- Technology ---
    researched_techs: frozenset[str]
    current_research: MappingProxyType[str, str]
    research_progress: MappingProxyType[str, float]
    research_speed: MappingProxyType[str, float]
    stored_research: MappingProxyType[str, float]

    # --- Military ---
    fleet_power: float
    naval_capacity_used: int
    naval_capacity_max: int
    fleets: tuple[FleetState, ...]

    # --- Territory ---
    owned_systems: tuple[int, ...]
    owned_planets: tuple[PlanetState, ...]
    starbases: tuple[StarbaseState, ...]

    # --- Population ---
    total_pops: int
    pop_groups: tuple[PopGroupState, ...]

    # --- Diplomacy ---
    rivals: tuple[int, ...]
    defensive_pacts: tuple[int, ...]
    federation_id: int | None
    federation_power: float | None
    subjects: tuple[int, ...]
    overlord: int | None

    # --- Traditions & Ascension ---
    traditions_unlocked: tuple[str, ...]
    tradition_trees_opened: tuple[str, ...]
    tradition_trees_completed: tuple[str, ...]
    ascension_perks: tuple[str, ...]
    ascension_perk_slots_available: int
    ascension_perk_slots_total: int
    ascension_path: str | None
    ascension_situation_progress: float | None

    # --- Leaders & Council ---
    leaders: tuple[LeaderState, ...]
    council_positions: tuple[CouncilPosition, ...]
    current_agenda: AgendaState | None
    council_legitimacy: float

    # --- Factions & Policies ---
    factions: tuple[FactionState, ...]
    policies: MappingProxyType[str, str]

    # --- Galactic Community ---
    galcom_member: bool
    galcom_diplomatic_weight: float
    galcom_council_member: bool
    galcom_custodian: bool
    galcom_emperor: bool
    active_resolutions: tuple[str, ...]

    # --- Federation ---
    federation: FederationState | None

    # --- Megastructures ---
    megastructures: tuple[MegastructureState, ...]
    megastructures_under_construction: int

    # --- Warfare ---
    active_wars: tuple[WarState, ...]
    claims_placed: tuple[ClaimState, ...]
    total_army_strength: float

    # --- Score ---
    victory_score: float
    victory_rank: int


@dataclass(frozen=True)
class GalacticState:
    """Top-level snapshot of the entire game state."""

    year: int
    month: int
    game_version: str
    galaxy_size: int
    difficulty: str
    crisis_active: bool
    crisis_type: str | None
    war_in_heaven: bool
    galactic_market_founded: bool
    empires: tuple[EmpireState, ...]
    player_empire_id: int
    systems: tuple[SystemState, ...]
    player_empire: EmpireState
