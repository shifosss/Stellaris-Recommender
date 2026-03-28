"""Ascension perk loader and query functions for the Stellaris gamedata layer."""

from __future__ import annotations

from typing import Any

from stellaris_advisor.gamedata.models import AscensionPerkDef
from stellaris_advisor.parser.extractor import safe_int, safe_str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_frozenset(raw: Any) -> frozenset[str]:
    """Convert a None, string, or list of strings to a frozenset."""
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


def load_ascension_perk_defs(raw: dict[str, Any]) -> dict[str, AscensionPerkDef]:
    """Parse a raw Clausewitz ascension perk dict into typed AscensionPerkDef instances.

    *raw* is the output of ``clausewitz.parse()`` applied to an ascension perks
    definition file (e.g. ``ascension_perks.txt``).
    """
    result: dict[str, AscensionPerkDef] = {}

    for pid, data in raw.items():
        if not isinstance(data, dict):
            continue

        result[pid] = AscensionPerkDef(
            id=pid,
            name=safe_str(data.get("name"), default=pid),
            description="",
            effect="",
            prerequisites=_extract_frozenset(data.get("prerequisites")),
            required_perks=_extract_frozenset(data.get("required_perks")),
            excluded_perks=_extract_frozenset(data.get("excluded_perks")),
            required_traditions=safe_int(data.get("required_traditions")),
        )

    return result


def eligible_perks(
    ascension_perk_defs: dict[str, AscensionPerkDef],
    researched: frozenset[str],
    traditions_count: int,
    existing_perks: frozenset[str],
    empire_type: str = "",
) -> frozenset[str]:
    """Return perk IDs that can currently be selected.

    A perk is eligible when ALL of:
    1. It is not already in *existing_perks*.
    2. All ``required_perks`` are in *existing_perks*.
    3. *traditions_count* >= ``required_traditions``.
    4. None of the perk's ``excluded_perks`` are in *existing_perks*.
    5. All tech ``prerequisites`` are in *researched*.

    *empire_type* is reserved for future gestalt/organic perk filtering.
    # TODO: filter by empire_type (some perks are gestalt-only or blocked for gestalt)
    """
    result: set[str] = set()
    for pid, perk in ascension_perk_defs.items():
        if pid in existing_perks:
            continue
        if not all(p in existing_perks for p in perk.required_perks):
            continue
        if traditions_count < perk.required_traditions:
            continue
        if perk.excluded_perks & existing_perks:
            continue
        if not all(p in researched for p in perk.prerequisites):
            continue
        result.add(pid)
    return frozenset(result)


def perk_prerequisites_met(
    ascension_perk_defs: dict[str, AscensionPerkDef],
    perk_id: str,
    existing_perks: frozenset[str],
    traditions_count: int,
) -> bool:
    """Return True if *perk_id* has its structural prerequisites satisfied.

    Checks ``required_perks`` and ``required_traditions`` only.
    Returns False if *perk_id* is not in *ascension_perk_defs*.
    """
    if perk_id not in ascension_perk_defs:
        return False
    perk = ascension_perk_defs[perk_id]
    if not all(p in existing_perks for p in perk.required_perks):
        return False
    if traditions_count < perk.required_traditions:
        return False
    return True
