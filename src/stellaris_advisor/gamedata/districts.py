"""District loader and query functions for the Stellaris gamedata layer."""

from __future__ import annotations

from typing import Any

from stellaris_advisor.gamedata.models import DistrictDef, OutputModifier
from stellaris_advisor.parser.extractor import safe_float, safe_str


# ---------------------------------------------------------------------------
# Internal helpers (shared patterns with buildings module)
# ---------------------------------------------------------------------------


def _extract_modifiers(raw_dict: Any) -> tuple[OutputModifier, ...]:
    if not isinstance(raw_dict, dict):
        return ()
    return tuple(
        OutputModifier(resource=str(k), amount=safe_float(v))
        for k, v in raw_dict.items()
    )


def _extract_jobs(planet_modifier: Any) -> tuple[str, ...]:
    if not isinstance(planet_modifier, dict):
        return ()
    jobs: list[str] = []
    for key in planet_modifier:
        if key.startswith("job_") and key.endswith("_add"):
            jobs.append(key[4:-4])
    return tuple(sorted(jobs))


def _extract_prerequisites(data: dict[str, Any]) -> frozenset[str]:
    raw = data.get("prerequisites")
    if raw is None:
        return frozenset()
    if isinstance(raw, str):
        return frozenset({raw})
    if isinstance(raw, list):
        return frozenset(str(p) for p in raw)
    return frozenset()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_district_defs(raw: dict[str, Any]) -> dict[str, DistrictDef]:
    """Parse a raw Clausewitz district dict into typed DistrictDef instances.

    *raw* is the output of ``clausewitz.parse()`` applied to one or more
    district definition files — a flat mapping of district_id → attribute dict.
    """
    result: dict[str, DistrictDef] = {}

    for did, data in raw.items():
        if not isinstance(data, dict):
            continue

        result[did] = DistrictDef(
            id=did,
            name=safe_str(data.get("name"), default=did),
            base_cost=_extract_modifiers(data.get("cost")),
            upkeep=_extract_modifiers(data.get("upkeep")),
            jobs=_extract_jobs(data.get("planet_modifier")),
            planet_classes=frozenset(),  # complex potential logic not parsed
            prerequisites=_extract_prerequisites(data),
        )

    return result


def available_districts(
    district_defs: dict[str, DistrictDef],
    planet_class: str = "",
    researched: frozenset[str] = frozenset(),
) -> frozenset[str]:
    """Return district IDs that can currently be built.

    A district is available when ALL of the following hold:
    1. All tech prerequisites are in *researched*.
    2. If ``planet_classes`` is non-empty, *planet_class* must be a member.
    """
    result: set[str] = set()
    for did, district in district_defs.items():
        if not all(p in researched for p in district.prerequisites):
            continue
        if district.planet_classes and planet_class not in district.planet_classes:
            continue
        result.add(did)
    return frozenset(result)
