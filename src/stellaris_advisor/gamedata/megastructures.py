"""Megastructure loader and query functions for the Stellaris gamedata layer."""

from __future__ import annotations

from typing import Any

from stellaris_advisor.gamedata.models import (
    MegastructureDef,
    MegastructureStageDef,
    OutputModifier,
)
from stellaris_advisor.parser.extractor import safe_float, safe_int, safe_str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_frozenset(raw: Any) -> frozenset[str]:
    """Convert None, a string, or a list of strings to a frozenset."""
    if raw is None:
        return frozenset()
    if isinstance(raw, str):
        return frozenset({raw})
    if isinstance(raw, list):
        return frozenset(str(p) for p in raw)
    return frozenset()


def _extract_modifiers(raw_dict: Any) -> tuple[OutputModifier, ...]:
    if not isinstance(raw_dict, dict):
        return ()
    return tuple(
        OutputModifier(resource=str(k), amount=safe_float(v))
        for k, v in raw_dict.items()
    )


def _extract_stages(data: dict[str, Any]) -> tuple[MegastructureStageDef, ...]:
    """Extract numeric-keyed stage sub-dicts into MegastructureStageDef tuples.

    Stage keys are stored as strings ("0", "1", …) by the Clausewitz parser.
    """
    stages: list[MegastructureStageDef] = []
    for key, stage_data in data.items():
        try:
            stage_num = int(key)
        except (ValueError, TypeError):
            continue
        if not isinstance(stage_data, dict):
            continue
        resources = stage_data.get("resources")
        cost_raw = resources.get("cost") if isinstance(resources, dict) else None
        cost = _extract_modifiers(cost_raw)
        build_time = safe_int(stage_data.get("construction_days"))
        stages.append(
            MegastructureStageDef(stage=stage_num, cost=cost, build_time=build_time)
        )
    return tuple(sorted(stages, key=lambda s: s.stage))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_megastructure_defs(raw: dict[str, Any]) -> dict[str, MegastructureDef]:
    """Parse a raw Clausewitz megastructure dict into typed MegastructureDef instances.

    *raw* is the output of ``clausewitz.parse()`` applied to a megastructures
    definition file (e.g. ``megastructures.txt``).

    Numeric-keyed sub-dicts (``0 = { … }``) are parsed as construction stages.
    """
    result: dict[str, MegastructureDef] = {}

    for mid, data in raw.items():
        if not isinstance(data, dict):
            continue

        result[mid] = MegastructureDef(
            id=mid,
            name=safe_str(data.get("name"), default=mid),
            stages=_extract_stages(data),
            prerequisites=_extract_frozenset(data.get("prerequisites")),
            required_perks=_extract_frozenset(data.get("required_perks")),
            site_requirements=(),
            description="",
        )

    return result


def available_megastructures(
    mega_defs: dict[str, MegastructureDef],
    researched: frozenset[str],
    existing_perks: frozenset[str],
) -> frozenset[str]:
    """Return megastructure IDs that can currently be built.

    A megastructure is available when ALL of:
    1. All tech ``prerequisites`` are in *researched*.
    2. All ``required_perks`` are in *existing_perks*.
    """
    result: set[str] = set()
    for mid, mega in mega_defs.items():
        if not all(p in researched for p in mega.prerequisites):
            continue
        if not all(p in existing_perks for p in mega.required_perks):
            continue
        result.add(mid)
    return frozenset(result)


def megastructure_sites(mega_def: MegastructureDef) -> tuple[str, ...]:
    """Return the site requirements for building *mega_def*."""
    return mega_def.site_requirements


def megastructure_total_cost(mega_def: MegastructureDef) -> dict[str, float]:
    """Sum construction costs across all stages and return as a resource dict."""
    result: dict[str, float] = {}
    for stage in mega_def.stages:
        for modifier in stage.cost:
            result[modifier.resource] = (
                result.get(modifier.resource, 0.0) + modifier.amount
            )
    return result
