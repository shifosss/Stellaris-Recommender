"""Federation type loader for the Stellaris gamedata layer."""

from __future__ import annotations

from typing import Any

from stellaris_advisor.gamedata.models import FederationTypeDef
from stellaris_advisor.parser.extractor import safe_float, safe_int, safe_str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_perk_names(raw: Any) -> tuple[str, ...]:
    """Extract perk name strings from a single perk dict or list of perk dicts."""
    if raw is None:
        return ()
    if isinstance(raw, dict):
        name = safe_str(raw.get("name"))
        return (name,) if name else ()
    if isinstance(raw, list):
        names: list[str] = []
        for item in raw:
            if isinstance(item, dict):
                name = safe_str(item.get("name"))
                if name:
                    names.append(name)
        return tuple(names)
    return ()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_federation_type_defs(raw: dict[str, Any]) -> dict[str, FederationTypeDef]:
    """Parse a raw Clausewitz federation type dict into typed FederationTypeDef instances.

    *raw* is the output of ``clausewitz.parse()`` applied to a federation
    types definition file (e.g. ``federation_types/00_federation_types.txt``).
    """
    result: dict[str, FederationTypeDef] = {}

    for fid, data in raw.items():
        if not isinstance(data, dict):
            continue

        result[fid] = FederationTypeDef(
            id=fid,
            name=safe_str(data.get("name"), default=fid),
            perks=_extract_perk_names(data.get("perk")),
            max_level=safe_int(data.get("max_level")),
            challenge_threshold=safe_float(data.get("challenge_threshold")),
        )

    return result
