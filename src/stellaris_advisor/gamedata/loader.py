"""GameKnowledgeGraph assembly and pickle caching for the Stellaris gamedata layer.

Responsibilities:
1. Orchestrate all domain loaders (tech, buildings, districts, …).
2. Apply cross-linking (populate derived fields on TechNode).
3. Run all 11 integrity validation checks.
4. Cache the assembled graph to disk via pickle, keyed on (version, content_hash).
5. Expose thin query delegation methods that forward to domain modules.
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import networkx as nx

import stellaris_advisor.gamedata.ascension_perks as _perks_mod
import stellaris_advisor.gamedata.buildings as _buildings_mod
import stellaris_advisor.gamedata.districts as _districts_mod
import stellaris_advisor.gamedata.tech_graph as _tech_mod
from stellaris_advisor.gamedata.ascension_perks import load_ascension_perk_defs
from stellaris_advisor.gamedata.buildings import load_building_defs
from stellaris_advisor.gamedata.council import load_agenda_defs, load_council_defs
from stellaris_advisor.gamedata.districts import load_district_defs
from stellaris_advisor.gamedata.empires import (
    load_civic_defs,
    load_edict_defs,
    load_trait_defs,
)
from stellaris_advisor.gamedata.factions import load_faction_defs
from stellaris_advisor.gamedata.federations import load_federation_type_defs
from stellaris_advisor.gamedata.jobs import load_job_defs
from stellaris_advisor.gamedata.megastructures import load_megastructure_defs
from stellaris_advisor.gamedata.models import (
    AgendaDef,
    AscensionPerkDef,
    BuildingDef,
    CivicDef,
    CouncilPositionDef,
    DistrictDef,
    EdictDef,
    FactionDef,
    FederationTypeDef,
    JobDef,
    MegastructureDef,
    PlanetClassDef,
    PolicyDef,
    ShipClassDef,
    TechNode,
    TraditionDef,
    TraditionTree,
    TraitDef,
)
from stellaris_advisor.gamedata.planet_classes import load_planet_class_defs
from stellaris_advisor.gamedata.policies import load_policy_defs
from stellaris_advisor.gamedata.ships import load_ship_class_defs
from stellaris_advisor.gamedata.tech_graph import load_tech_defs
from stellaris_advisor.gamedata.traditions import (
    load_tradition_defs,
    load_tradition_tree_defs,
)
from stellaris_advisor.gamedata.validation import (
    ValidationResult,
    cross_link_tech_unlocks,
    validate_game_data,
)
from stellaris_advisor.parser.clausewitz import parse


# ---------------------------------------------------------------------------
# GameKnowledgeGraph
# ---------------------------------------------------------------------------


@dataclass
class GameKnowledgeGraph:
    """Assembled, cross-linked, and validated Stellaris game knowledge graph.

    All dict attributes are wrapped in :class:`MappingProxyType` to prevent
    accidental mutation at call-sites. Use the query delegation methods for
    game-logic queries rather than accessing the dicts directly.
    """

    tech_defs: MappingProxyType[str, TechNode]
    tech_dag: nx.DiGraph
    building_defs: MappingProxyType[str, BuildingDef]
    district_defs: MappingProxyType[str, DistrictDef]
    job_defs: MappingProxyType[str, JobDef]
    ship_class_defs: MappingProxyType[str, ShipClassDef]
    planet_class_defs: MappingProxyType[str, PlanetClassDef]
    tradition_trees: MappingProxyType[str, TraditionTree]
    tradition_defs: MappingProxyType[str, TraditionDef]
    ascension_perk_defs: MappingProxyType[str, AscensionPerkDef]
    megastructure_defs: MappingProxyType[str, MegastructureDef]
    policy_defs: MappingProxyType[str, PolicyDef]
    faction_defs: MappingProxyType[str, FactionDef]
    council_position_defs: MappingProxyType[str, CouncilPositionDef]
    agenda_defs: MappingProxyType[str, AgendaDef]
    federation_type_defs: MappingProxyType[str, FederationTypeDef]
    civic_defs: MappingProxyType[str, CivicDef]
    trait_defs: MappingProxyType[str, TraitDef]
    edict_defs: MappingProxyType[str, EdictDef]
    validation_results: tuple[ValidationResult, ...]
    version: str

    # Fields that must be re-wrapped as MappingProxyType after unpickling.
    _PROXY_FIELDS: frozenset[str] = frozenset(
        {
            "tech_defs",
            "building_defs",
            "district_defs",
            "job_defs",
            "ship_class_defs",
            "planet_class_defs",
            "tradition_trees",
            "tradition_defs",
            "ascension_perk_defs",
            "megastructure_defs",
            "policy_defs",
            "faction_defs",
            "council_position_defs",
            "agenda_defs",
            "federation_type_defs",
            "civic_defs",
            "trait_defs",
            "edict_defs",
        }
    )

    def __getstate__(self) -> dict[str, Any]:
        """Serialize: convert MappingProxyType values to plain dicts."""
        return {
            k: dict(v) if isinstance(v, MappingProxyType) else v
            for k, v in self.__dict__.items()
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Deserialize: re-wrap plain dicts as MappingProxyType."""
        for key, val in state.items():
            if key in self._PROXY_FIELDS:
                self.__dict__[key] = MappingProxyType(val)
            else:
                self.__dict__[key] = val

    # -----------------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        stellaris_dir: Path,
        version: str,
        cache_dir: Path | None = None,
    ) -> GameKnowledgeGraph:
        """Load and assemble a :class:`GameKnowledgeGraph` from *stellaris_dir*.

        If *cache_dir* is provided, a pickle cache keyed on
        ``(version, content_hash)`` is read on cache-hit and written on miss.
        The content hash is a SHA-256 digest of the sorted ``*.txt`` file
        paths and their modification timestamps under ``common/``.
        Pass ``cache_dir=None`` to disable caching entirely.
        """
        if cache_dir is not None:
            content_hash = _content_hash(stellaris_dir)
            cache_file = cache_dir / f"gamedata-{version}-{content_hash[:12]}.pkl"
            if cache_file.exists():
                with cache_file.open("rb") as fh:
                    return pickle.load(fh)  # type: ignore[no-any-return]

            gkg = _build(stellaris_dir, version)
            cache_dir.mkdir(parents=True, exist_ok=True)
            with cache_file.open("wb") as fh:
                pickle.dump(gkg, fh)
            return gkg

        return _build(stellaris_dir, version)

    # -----------------------------------------------------------------------
    # Query delegation
    # -----------------------------------------------------------------------

    def reachable_techs(
        self,
        researched: frozenset[str],
        empire_type: str = "",
        ethics: frozenset[str] = frozenset(),
    ) -> frozenset[str]:
        """Return techs currently in the research draw pool."""
        return _tech_mod.reachable_techs(
            dict(self.tech_defs), self.tech_dag, researched, empire_type, ethics
        )

    def available_buildings(
        self,
        researched: frozenset[str],
        planet_class: str = "",
        empire_type: str = "",
        existing: frozenset[str] = frozenset(),
        capital_level: int = 0,
    ) -> frozenset[str]:
        """Return building IDs constructable right now."""
        return _buildings_mod.available_buildings(
            dict(self.building_defs),
            researched,
            planet_class,
            empire_type,
            existing,
            capital_level,
        )

    def available_districts(
        self,
        planet_class: str = "",
        researched: frozenset[str] = frozenset(),
    ) -> frozenset[str]:
        """Return district IDs buildable on *planet_class*."""
        return _districts_mod.available_districts(
            dict(self.district_defs), planet_class, researched
        )

    def eligible_perks(
        self,
        taken: frozenset[str],
        researched: frozenset[str],
        traditions_completed: int,
        empire_type: str = "",
    ) -> frozenset[str]:
        """Return ascension perk IDs currently selectable."""
        return _perks_mod.eligible_perks(
            dict(self.ascension_perk_defs),
            researched,
            traditions_completed,
            taken,
            empire_type,
        )

    def tech_path_to(
        self,
        target: str,
        researched: frozenset[str],
    ) -> tuple[str, ...] | None:
        """Return ordered prerequisite chain to reach *target*."""
        return _tech_mod.tech_path_to(
            dict(self.tech_defs), self.tech_dag, target, researched
        )

    def downstream_unlocks(self, tech_id: str) -> frozenset[str]:
        """Return all techs transitively unlocked by *tech_id*."""
        return _tech_mod.downstream_unlocks(
            dict(self.tech_defs), self.tech_dag, tech_id
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_dirs(stellaris_dir: Path, subdirs: list[str]) -> dict[str, Any]:
    """Parse all ``*.txt`` files in *subdirs* and merge into one dict.

    Missing directories are silently skipped so the loader degrades gracefully
    on minimal or partial Stellaris installations.
    """
    merged: dict[str, Any] = {}
    for subdir in subdirs:
        dirpath = stellaris_dir / subdir
        if not dirpath.is_dir():
            continue
        for txt_path in sorted(dirpath.glob("*.txt")):
            raw = parse(txt_path.read_text(encoding="utf-8"))
            merged.update(raw)
    return merged


def _content_hash(stellaris_dir: Path) -> str:
    """Return a SHA-256 hex digest of sorted ``*.txt`` file paths and mtimes."""
    common = stellaris_dir / "common"
    hasher = hashlib.sha256()
    for txt_path in sorted(common.rglob("*.txt")):
        hasher.update(str(txt_path).encode())
        hasher.update(str(txt_path.stat().st_mtime_ns).encode())
    return hasher.hexdigest()


def _build(stellaris_dir: Path, version: str) -> GameKnowledgeGraph:
    """Parse every domain, cross-link, validate, and return a fresh GKG."""
    # --- Tech ---
    tech_defs_raw, tech_dag = load_tech_defs(
        _parse_dirs(stellaris_dir, ["common/technology"])
    )

    # --- Buildings / Districts / Ships ---
    building_defs = load_building_defs(_parse_dirs(stellaris_dir, ["common/buildings"]))
    district_defs = load_district_defs(
        _parse_dirs(stellaris_dir, ["common/district_types"])
    )
    job_defs = load_job_defs(_parse_dirs(stellaris_dir, ["common/jobs"]))
    ship_class_defs = load_ship_class_defs(
        _parse_dirs(stellaris_dir, ["common/ship_sizes"])
    )

    # --- Planet classes ---
    planet_class_defs = load_planet_class_defs(
        _parse_dirs(stellaris_dir, ["common/planet_classes"])
    )

    # --- Traditions ---
    tradition_defs = load_tradition_defs(
        _parse_dirs(stellaris_dir, ["common/traditions"])
    )
    tradition_trees = load_tradition_tree_defs(
        _parse_dirs(stellaris_dir, ["common/tradition_categories"])
    )

    # --- Ascension perks ---
    ascension_perk_defs = load_ascension_perk_defs(
        _parse_dirs(stellaris_dir, ["common/ascension_perks"])
    )

    # --- Megastructures ---
    megastructure_defs = load_megastructure_defs(
        _parse_dirs(stellaris_dir, ["common/megastructures"])
    )

    # --- Policies / Factions ---
    policy_defs = load_policy_defs(_parse_dirs(stellaris_dir, ["common/policies"]))
    faction_defs = load_faction_defs(
        _parse_dirs(stellaris_dir, ["common/pop_faction_types"])
    )

    # --- Council / Agendas ---
    council_position_defs = load_council_defs(
        _parse_dirs(stellaris_dir, ["common/council_positions"])
    )
    agenda_defs = load_agenda_defs(_parse_dirs(stellaris_dir, ["common/agendas"]))

    # --- Federations ---
    federation_type_defs = load_federation_type_defs(
        _parse_dirs(stellaris_dir, ["common/federation_types"])
    )

    # --- Empires (civics / traits / edicts) ---
    civic_defs = load_civic_defs(
        _parse_dirs(stellaris_dir, ["common/governments/civics"])
    )
    trait_defs = load_trait_defs(_parse_dirs(stellaris_dir, ["common/traits"]))
    edict_defs = load_edict_defs(_parse_dirs(stellaris_dir, ["common/edicts"]))

    # --- Cross-link ---
    tech_defs = cross_link_tech_unlocks(
        tech_defs_raw, building_defs, district_defs, ship_class_defs
    )

    # --- Validate ---
    validation_results = validate_game_data(
        tech_defs=tech_defs,
        tech_dag=tech_dag,
        building_defs=building_defs,
        district_defs=district_defs,
        ship_class_defs=ship_class_defs,
        tradition_trees=tradition_trees,
        tradition_defs=tradition_defs,
        perk_defs=ascension_perk_defs,
        mega_defs=megastructure_defs,
        policy_defs=policy_defs,
        faction_defs=faction_defs,
        council_position_defs=council_position_defs,
        planet_class_defs=planet_class_defs,
    )

    return GameKnowledgeGraph(
        tech_defs=MappingProxyType(tech_defs),
        tech_dag=tech_dag,
        building_defs=MappingProxyType(building_defs),
        district_defs=MappingProxyType(district_defs),
        job_defs=MappingProxyType(job_defs),
        ship_class_defs=MappingProxyType(ship_class_defs),
        planet_class_defs=MappingProxyType(planet_class_defs),
        tradition_trees=MappingProxyType(tradition_trees),
        tradition_defs=MappingProxyType(tradition_defs),
        ascension_perk_defs=MappingProxyType(ascension_perk_defs),
        megastructure_defs=MappingProxyType(megastructure_defs),
        policy_defs=MappingProxyType(policy_defs),
        faction_defs=MappingProxyType(faction_defs),
        council_position_defs=MappingProxyType(council_position_defs),
        agenda_defs=MappingProxyType(agenda_defs),
        federation_type_defs=MappingProxyType(federation_type_defs),
        civic_defs=MappingProxyType(civic_defs),
        trait_defs=MappingProxyType(trait_defs),
        edict_defs=MappingProxyType(edict_defs),
        validation_results=validation_results,
        version=version,
    )
