"""Ship class loader for the Stellaris gamedata layer."""

from __future__ import annotations

from typing import Any

from stellaris_advisor.gamedata.models import (
    OutputModifier,
    ShipClassDef,
    ShipSectionDef,
)
from stellaris_advisor.parser.extractor import get_path, safe_float, safe_int, safe_str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_modifiers(raw_dict: Any) -> tuple[OutputModifier, ...]:
    if not isinstance(raw_dict, dict):
        return ()
    return tuple(
        OutputModifier(resource=str(k), amount=safe_float(v))
        for k, v in raw_dict.items()
    )


def _extract_prerequisites(data: dict[str, Any]) -> frozenset[str]:
    raw = data.get("prerequisites")
    if raw is None:
        return frozenset()
    if isinstance(raw, str):
        return frozenset({raw})
    if isinstance(raw, list):
        return frozenset(str(p) for p in raw)
    return frozenset()


def _extract_sections(
    ship_id: str, raw_components: dict[str, Any]
) -> tuple[ShipSectionDef, ...]:
    """Build ShipSectionDef instances for sections that belong to *ship_id*.

    A section belongs to *ship_id* when its ``ship_size`` field matches.
    *component_slot* and *utility_slot* lists determine slot types.
    """
    sections: list[ShipSectionDef] = []
    for section_id, section_data in raw_components.items():
        if not isinstance(section_data, dict):
            continue
        if safe_str(section_data.get("ship_size")) != ship_id:
            continue

        slot_types: list[str] = []
        for slot_key in ("component_slot", "utility_slot"):
            raw_slots = section_data.get(slot_key, [])
            if isinstance(raw_slots, dict):
                raw_slots = [raw_slots]
            if isinstance(raw_slots, list):
                for slot in raw_slots:
                    if isinstance(slot, dict):
                        name = slot.get("name")
                        if name:
                            slot_types.append(str(name))

        component_slots = safe_int(section_data.get("slot_count"))

        sections.append(
            ShipSectionDef(
                id=section_id,
                name=safe_str(section_data.get("name"), default=section_id),
                slot_types=tuple(slot_types),
                component_slots=component_slots,
            )
        )
    return tuple(sections)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_ship_class_defs(
    raw_sizes: dict[str, Any],
    raw_components: dict[str, Any] | None = None,
) -> dict[str, ShipClassDef]:
    """Parse raw Clausewitz ship size data into typed ShipClassDef instances.

    *raw_sizes* is the output of ``clausewitz.parse()`` applied to a ship sizes
    definition file (e.g. ``ship_sizes.txt``).

    *raw_components* is the optional output of parsing ship section template
    files. When provided, sections are matched by their ``ship_size`` field.
    When ``None`` or empty, all ship classes have ``sections = ()``.
    """
    if raw_components is None:
        raw_components = {}

    result: dict[str, ShipClassDef] = {}

    for ship_id, data in raw_sizes.items():
        if not isinstance(data, dict):
            continue

        evasion = safe_float(get_path(data, "modifier.evasion_mult"))

        result[ship_id] = ShipClassDef(
            id=ship_id,
            name=safe_str(data.get("name"), default=ship_id),
            hull=safe_int(data.get("max_hitpoints")),
            evasion=evasion,
            speed=safe_float(data.get("max_speed")),
            sections=_extract_sections(ship_id, raw_components),
            base_cost=_extract_modifiers(data.get("cost")),
            upkeep=_extract_modifiers(data.get("upkeep")),
            prerequisites=_extract_prerequisites(data),
        )

    return result
