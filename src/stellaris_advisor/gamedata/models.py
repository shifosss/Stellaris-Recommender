"""Frozen dataclass models representing Stellaris game definition data.

All collections are immutable: tuple (ordered sequences), frozenset (unordered sets).
All dataclasses use frozen=True. No MappingProxyType is needed here as the game
definition data does not use arbitrary-keyed dicts.

Organised into four layers:
  Core    — TechNode, BuildingDef, DistrictDef, JobDef, Ships, PlanetClass
  Meta    — TraditionTree/Def, AscensionPerkDef, MegastructureDef
  Political — PolicyDef, FactionDef, CouncilPositionDef, AgendaDef, FederationTypeDef
  Supporting — CivicDef, TraitDef, EdictDef, SpecializationDef
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WeightModifier:
    """Technology research-weight modifier with trigger conditions."""

    factor: float
    conditions: tuple[str, ...] = ()


@dataclass(frozen=True)
class OutputModifier:
    """A single resource production or consumption entry."""

    resource: str
    amount: float


# ---------------------------------------------------------------------------
# Core Layer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TechNode:
    """A technology in the research tree."""

    id: str
    name: str
    area: str
    tier: int
    cost: int
    category: str
    prerequisites: frozenset[str] = field(default_factory=frozenset)
    weight_modifiers: tuple[WeightModifier, ...] = ()
    is_rare: bool = False
    is_dangerous: bool = False
    is_start_tech: bool = False
    # Populated by cross-linking after all loaders run
    unlocks_buildings: tuple[str, ...] = ()
    unlocks_districts: tuple[str, ...] = ()
    unlocks_ship_sizes: tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True)
class JobDef:
    """A population job definition."""

    id: str
    name: str
    category: str
    produces: tuple[OutputModifier, ...]
    upkeep: tuple[OutputModifier, ...]
    conditions: tuple[str, ...] = ()
    weight: float = 1.0


@dataclass(frozen=True)
class BuildingDef:
    """A planetary building definition."""

    id: str
    name: str
    category: str
    base_cost: tuple[OutputModifier, ...]
    upkeep: tuple[OutputModifier, ...]
    jobs: tuple[str, ...]
    prerequisites: frozenset[str]
    planet_classes: frozenset[str]
    empire_types: frozenset[str]
    max_per_planet: int = -1  # -1 means unlimited
    upgrades_from: str = ""
    description: str = ""


@dataclass(frozen=True)
class DistrictDef:
    """A planetary district definition."""

    id: str
    name: str
    base_cost: tuple[OutputModifier, ...]
    upkeep: tuple[OutputModifier, ...]
    jobs: tuple[str, ...]
    planet_classes: frozenset[str]
    prerequisites: frozenset[str]
    max_per_planet: int = -1  # -1 means unlimited
    description: str = ""


@dataclass(frozen=True)
class SpecializationDef:
    """A planetary designation / specialization."""

    id: str
    name: str
    bonuses: tuple[OutputModifier, ...]
    conditions: tuple[str, ...] = ()


@dataclass(frozen=True)
class ShipSectionDef:
    """A ship section template with weapon slots."""

    id: str
    name: str
    slot_types: tuple[str, ...]
    component_slots: int


@dataclass(frozen=True)
class ShipClassDef:
    """A ship class (hull) definition."""

    id: str
    name: str
    hull: int
    evasion: float
    speed: float
    sections: tuple[ShipSectionDef, ...]
    base_cost: tuple[OutputModifier, ...]
    upkeep: tuple[OutputModifier, ...]
    prerequisites: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class PlanetClassDef:
    """A planet class definition (habitability, districts, modifiers)."""

    id: str
    name: str
    habitability_category: str
    district_cap: int
    base_habitability: float
    modifiers: tuple[OutputModifier, ...]


# ---------------------------------------------------------------------------
# Meta Layer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TraditionTree:
    """A tradition adoption tree (e.g. Expansion, Domination)."""

    id: str
    name: str
    tradition_ids: tuple[str, ...]
    adopt_effect: str
    finish_effect: str
    prerequisites: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class TraditionDef:
    """An individual tradition within a tree."""

    id: str
    name: str
    tree_id: str
    effect: str
    prerequisites: frozenset[str] = field(default_factory=frozenset)
    cost: int = 0


@dataclass(frozen=True)
class AscensionPerkDef:
    """An ascension perk slot definition."""

    id: str
    name: str
    description: str
    effect: str
    prerequisites: frozenset[str] = field(default_factory=frozenset)
    required_perks: frozenset[str] = field(default_factory=frozenset)
    excluded_perks: frozenset[str] = field(default_factory=frozenset)
    required_traditions: int = 0


@dataclass(frozen=True)
class MegastructureStageDef:
    """A single construction stage of a megastructure."""

    stage: int
    cost: tuple[OutputModifier, ...]
    build_time: int


@dataclass(frozen=True)
class MegastructureDef:
    """A megastructure definition with all build stages."""

    id: str
    name: str
    stages: tuple[MegastructureStageDef, ...]
    prerequisites: frozenset[str]
    required_perks: frozenset[str]
    site_requirements: tuple[str, ...]
    description: str = ""


# ---------------------------------------------------------------------------
# Political Layer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyOptionDef:
    """A single selectable option within a policy."""

    id: str
    name: str
    effects: tuple[OutputModifier, ...]
    # Each entry is (faction_id, approval_delta)
    faction_effects: tuple[tuple[str, float], ...] = ()
    conditions: tuple[str, ...] = ()


@dataclass(frozen=True)
class PolicyDef:
    """A game policy with selectable options."""

    id: str
    name: str
    category: str
    options: tuple[PolicyOptionDef, ...]


@dataclass(frozen=True)
class FactionIssueDef:
    """An issue a faction cares about (maps to a policy preference)."""

    id: str
    name: str
    approval_effect: float
    policy_id: str
    desired_option: str


@dataclass(frozen=True)
class FactionDef:
    """A political faction definition."""

    id: str
    name: str
    ethic: str
    issues: tuple[FactionIssueDef, ...]
    happiness_effects: tuple[OutputModifier, ...] = ()


@dataclass(frozen=True)
class CouncilPositionDef:
    """A council position that can be filled by a leader."""

    id: str
    name: str
    leader_class: str
    effects: tuple[OutputModifier, ...]
    max_positions: int = 1


@dataclass(frozen=True)
class AgendaDef:
    """A ruling agenda definition."""

    id: str
    name: str
    effects: tuple[OutputModifier, ...]
    prerequisites: frozenset[str] = field(default_factory=frozenset)
    conditions: tuple[str, ...] = ()


@dataclass(frozen=True)
class FederationTypeDef:
    """A federation type with levelled perks."""

    id: str
    name: str
    perks: tuple[str, ...]
    max_level: int
    challenge_threshold: float


# ---------------------------------------------------------------------------
# Supporting Layer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CivicDef:
    """An empire civic definition."""

    id: str
    name: str
    description: str
    effects: tuple[OutputModifier, ...]
    prerequisites: frozenset[str] = field(default_factory=frozenset)
    excluded_civics: frozenset[str] = field(default_factory=frozenset)
    authority_types: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class TraitDef:
    """A species trait definition."""

    id: str
    name: str
    cost: int
    effects: tuple[OutputModifier, ...]
    category: str
    opposites: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class EdictDef:
    """An empire edict definition."""

    id: str
    name: str
    cost: tuple[OutputModifier, ...]
    upkeep: tuple[OutputModifier, ...]
    effects: tuple[OutputModifier, ...]
    prerequisites: frozenset[str] = field(default_factory=frozenset)
    duration: int = -1  # -1 means permanent
