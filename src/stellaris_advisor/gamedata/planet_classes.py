"""Planet class loader for the Stellaris gamedata layer."""

from __future__ import annotations

from typing import Any

from stellaris_advisor.gamedata.models import OutputModifier, PlanetClassDef
from stellaris_advisor.parser.extractor import safe_float, safe_int, safe_str


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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_planet_class_defs(raw: dict[str, Any]) -> dict[str, PlanetClassDef]:
    """Parse a raw Clausewitz planet class dict into typed PlanetClassDef instances.

    *raw* is the output of ``clausewitz.parse()`` applied to a planet classes
    definition file (e.g. ``planet_classes.txt``).

    Field mapping:
    - ``climate``      → ``habitability_category``
    - ``habitability`` → ``base_habitability``
    - ``district_cap`` → ``district_cap``
    - ``modifier``     → ``modifiers`` (as OutputModifier tuples)
    """
    result: dict[str, PlanetClassDef] = {}

    for pc_id, data in raw.items():
        if not isinstance(data, dict):
            continue

        result[pc_id] = PlanetClassDef(
            id=pc_id,
            name=safe_str(data.get("name"), default=pc_id),
            habitability_category=safe_str(data.get("climate")),
            district_cap=safe_int(data.get("district_cap")),
            base_habitability=safe_float(data.get("habitability")),
            modifiers=_extract_modifiers(data.get("modifier")),
        )

    return result
