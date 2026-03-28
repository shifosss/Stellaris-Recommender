"""Civic, trait, and edict loaders for the Stellaris gamedata layer."""

from __future__ import annotations

from typing import Any

from stellaris_advisor.gamedata.models import (
    CivicDef,
    EdictDef,
    OutputModifier,
    TraitDef,
)
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


def _extract_frozenset(raw: Any) -> frozenset[str]:
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


def load_civic_defs(raw: dict[str, Any]) -> dict[str, CivicDef]:
    """Parse a raw Clausewitz civic dict into typed CivicDef instances.

    *raw* is the output of ``clausewitz.parse()`` applied to a civics
    definition file (e.g. ``governments/civics/00_civics.txt``).

    ``potential`` and ``authority`` blocks contain complex runtime conditions;
    ``prerequisites``, ``excluded_civics``, and ``authority_types`` are stored
    as empty frozensets (to be populated by cross-linking in Task 8).
    """
    result: dict[str, CivicDef] = {}

    for cid, data in raw.items():
        if not isinstance(data, dict):
            continue

        result[cid] = CivicDef(
            id=cid,
            name=safe_str(data.get("name"), default=cid),
            description=safe_str(data.get("description")),
            effects=_extract_modifiers(data.get("modifier")),
            prerequisites=frozenset(),
            excluded_civics=frozenset(),
            authority_types=frozenset(),
        )

    return result


def load_trait_defs(raw: dict[str, Any]) -> dict[str, TraitDef]:
    """Parse a raw Clausewitz trait dict into typed TraitDef instances.

    *raw* is the output of ``clausewitz.parse()`` applied to a species traits
    definition file (e.g. ``traits/00_species_traits.txt``).
    """
    result: dict[str, TraitDef] = {}

    for tid, data in raw.items():
        if not isinstance(data, dict):
            continue

        result[tid] = TraitDef(
            id=tid,
            name=safe_str(data.get("name"), default=tid),
            cost=safe_int(data.get("cost")),
            effects=_extract_modifiers(data.get("modifier")),
            category=safe_str(data.get("category")),
            opposites=_extract_frozenset(data.get("opposites")),
        )

    return result


def load_edict_defs(raw: dict[str, Any]) -> dict[str, EdictDef]:
    """Parse a raw Clausewitz edict dict into typed EdictDef instances.

    *raw* is the output of ``clausewitz.parse()`` applied to an edicts
    definition file (e.g. ``edicts/00_edicts.txt``).

    ``duration`` maps to the ``length`` field (-1 means permanent).
    """
    result: dict[str, EdictDef] = {}

    for eid, data in raw.items():
        if not isinstance(data, dict):
            continue

        result[eid] = EdictDef(
            id=eid,
            name=safe_str(data.get("name"), default=eid),
            cost=_extract_modifiers(data.get("cost")),
            upkeep=_extract_modifiers(data.get("upkeep")),
            effects=_extract_modifiers(data.get("modifier")),
            prerequisites=_extract_frozenset(data.get("prerequisites")),
            duration=safe_int(data.get("length"), default=-1),
        )

    return result
