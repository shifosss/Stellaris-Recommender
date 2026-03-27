"""Top-level save file and game data parsing entry points."""

from __future__ import annotations

import dataclasses
import zipfile
from pathlib import Path
from typing import Any

from stellaris_advisor.parser.clausewitz import parse
from stellaris_advisor.parser.extractor import (
    extract_empire,
    extract_planet,
    extract_system,
    safe_int,
    safe_str,
    to_int_tuple,
)
from stellaris_advisor.parser.models import (
    EmpireState,
    GalacticState,
    PlanetState,
    SystemState,
)
from stellaris_advisor.parser.version import detect_version


# ---------------------------------------------------------------------------
# Game data file categories to scan
# ---------------------------------------------------------------------------

_GAME_DATA_CATEGORIES: tuple[str, ...] = (
    "technology",
    "buildings",
    "districts",
    "pop_jobs",
    "ship_sizes",
    "component_templates",
    "planet_classes",
    "tradition_categories",
    "traditions",
    "ascension_perks",
    "megastructures",
    "policies",
    "factions",
    "council_positions",
    "agendas",
    "federation_types",
    "governments/civics",
    "governments/authorities",
    "traits",
    "edicts",
)


# ---------------------------------------------------------------------------
# parse_save
# ---------------------------------------------------------------------------


def parse_save(path: str) -> GalacticState:
    """Parse a Stellaris .sav file into a fully-linked GalacticState.

    The .sav is a ZIP containing 'meta' and 'gamestate' entries.

    Raises:
        FileNotFoundError: If the save file does not exist.
    """
    save_path = Path(path)
    if not save_path.exists():
        raise FileNotFoundError(f"Save file not found: {path}")

    with zipfile.ZipFile(save_path, "r") as zf:
        meta_text = zf.read("meta").decode("utf-8-sig")
        gamestate_text = zf.read("gamestate").decode("utf-8-sig")

    meta_raw = parse(meta_text)
    gs_raw = parse(gamestate_text)

    version = detect_version(meta_raw)
    year, month = _parse_date(gs_raw.get("date", "2200.01.01"))
    player_empire_id = _extract_player_id(gs_raw)

    # --- Parse planets ---
    planet_lookup = _build_planet_lookup(gs_raw, version)

    # --- Parse empires with linked planets ---
    empires = _build_empires(gs_raw, planet_lookup, version)

    # --- Parse systems ---
    systems = _build_systems(gs_raw, version)

    # --- Find player empire ---
    player_empire = next(
        (e for e in empires if e.id == player_empire_id),
        empires[0] if empires else _empty_empire(player_empire_id),
    )

    return GalacticState(
        year=year,
        month=month,
        game_version=version,
        galaxy_size=safe_int(
            gs_raw.get("galaxy", {}).get("star_count")
            if isinstance(gs_raw.get("galaxy"), dict)
            else None
        ),
        difficulty=safe_str(gs_raw.get("difficulty")),
        crisis_active=bool(gs_raw.get("crisis_active")),
        crisis_type=gs_raw.get("crisis_type"),
        war_in_heaven=bool(gs_raw.get("war_in_heaven")),
        galactic_market_founded=bool(gs_raw.get("galactic_market_founded")),
        empires=tuple(empires),
        player_empire_id=player_empire_id,
        systems=tuple(systems),
        player_empire=player_empire,
    )


# ---------------------------------------------------------------------------
# parse_game_files
# ---------------------------------------------------------------------------


def parse_game_files(stellaris_dir: str) -> dict[str, dict[str, Any]]:
    """Parse Stellaris game definition files from install directory.

    Scans `common/<category>/*.txt` for each known category, parses each
    file, and merges results by category.

    Returns a dict mapping category name → merged key-value dict.
    """
    base = Path(stellaris_dir) / "common"
    result: dict[str, dict[str, Any]] = {}

    for category in _GAME_DATA_CATEGORIES:
        category_dir = base / category
        if not category_dir.is_dir():
            continue
        merged: dict[str, Any] = {}
        for txt_file in sorted(category_dir.glob("*.txt")):
            try:
                text = txt_file.read_text(encoding="utf-8-sig")
                parsed = parse(text)
                merged.update(parsed)
            except Exception:
                continue
        if merged:
            # Use the leaf directory name as the category key
            result[category.split("/")[-1]] = merged

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_date(date_str: Any) -> tuple[int, int]:
    """Parse Clausewitz date string 'YYYY.MM.DD' → (year, month)."""
    parts = str(date_str).split(".")
    try:
        return int(parts[0]), int(parts[1]) if len(parts) > 1 else 1
    except (ValueError, IndexError):
        return 2200, 1


def _extract_player_id(gs_raw: dict[str, Any]) -> int:
    """Extract the player's empire ID from the gamestate."""
    player = gs_raw.get("player", {})
    if isinstance(player, dict):
        for entry in player.values():
            if isinstance(entry, dict) and "country" in entry:
                return safe_int(entry["country"])
    return 0


def _build_planet_lookup(
    gs_raw: dict[str, Any], version: str
) -> dict[int, PlanetState]:
    """Build a mapping of planet_id → PlanetState from the gamestate."""
    planets_section = gs_raw.get("planets", {})
    if not isinstance(planets_section, dict):
        return {}
    planet_entries = planets_section.get("planet", {})
    if not isinstance(planet_entries, dict):
        return {}

    lookup: dict[int, PlanetState] = {}
    for pid_str, planet_raw in planet_entries.items():
        if isinstance(planet_raw, dict):
            pid = safe_int(pid_str)
            lookup[pid] = extract_planet(planet_raw, planet_id=pid, version=version)
    return lookup


def _build_empires(
    gs_raw: dict[str, Any],
    planet_lookup: dict[int, PlanetState],
    version: str,
) -> list[EmpireState]:
    """Build empire list with linked planet objects."""
    country_section = gs_raw.get("country", {})
    if not isinstance(country_section, dict):
        return []

    empires: list[EmpireState] = []
    for eid_str, empire_raw in country_section.items():
        if not isinstance(empire_raw, dict):
            continue
        eid = safe_int(eid_str)
        empire = extract_empire(empire_raw, empire_id=eid, version=version)

        # Link planets
        raw_planet_ids = empire_raw.get("owned_planets")
        linked_planets: tuple[PlanetState, ...] = ()
        if raw_planet_ids is not None:
            ids = to_int_tuple(raw_planet_ids)
            linked_planets = tuple(
                planet_lookup[pid] for pid in ids if pid in planet_lookup
            )

        # Rebuild with linked planets
        empire = _replace_field(empire, "owned_planets", linked_planets)

        # Also set total_pops from raw if available
        pop_count = empire_raw.get("pop_count")
        if pop_count is not None:
            empire = _replace_field(empire, "total_pops", safe_int(pop_count))

        empires.append(empire)
    return empires


def _build_systems(gs_raw: dict[str, Any], version: str) -> list[SystemState]:
    """Build system list from galactic_object section."""
    objects = gs_raw.get("galactic_object", {})
    if not isinstance(objects, dict):
        return []

    systems = []
    for sid_str, sys_raw in objects.items():
        if isinstance(sys_raw, dict):
            sid = safe_int(sid_str)
            systems.append(extract_system(sys_raw, system_id=sid, version=version))
    return systems


def _replace_field(obj: Any, field_name: str, value: Any) -> Any:
    """Create a new frozen dataclass instance with one field replaced."""
    fields = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
    fields[field_name] = value
    return type(obj)(**fields)


def _empty_empire(empire_id: int) -> EmpireState:
    """Create an empty EmpireState placeholder."""
    return extract_empire({}, empire_id=empire_id, version="4.3")
