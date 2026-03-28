"""Faction loader for the Stellaris gamedata layer."""

from __future__ import annotations

from typing import Any

from stellaris_advisor.gamedata.models import FactionDef, FactionIssueDef
from stellaris_advisor.parser.extractor import safe_float, safe_str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_issues(data: dict[str, Any]) -> tuple[FactionIssueDef, ...]:
    """Extract political issues from the raw faction dict."""
    issues_raw = data.get("issues")
    if not isinstance(issues_raw, dict):
        return ()

    raw_items = issues_raw.get("political_issue")
    if raw_items is None:
        return ()
    # Parser emits list for duplicate keys, dict for a single entry
    if isinstance(raw_items, dict):
        raw_items = [raw_items]

    issues: list[FactionIssueDef] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        title = safe_str(item.get("title"))
        desired_option = safe_str(item.get("option"))
        approval_effect = safe_float(
            (item.get("effect") or {}).get("faction_approval")
            if isinstance(item.get("effect"), dict)
            else None
        )
        issues.append(
            FactionIssueDef(
                id=title or desired_option,
                name=title or desired_option,
                approval_effect=approval_effect,
                policy_id="",
                desired_option=desired_option,
            )
        )
    return tuple(issues)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_faction_defs(raw: dict[str, Any]) -> dict[str, FactionDef]:
    """Parse a raw Clausewitz faction dict into typed FactionDef instances.

    *raw* is the output of ``clausewitz.parse()`` applied to a faction types
    definition file (e.g. ``pop_faction_types/00_factions.txt``).
    """
    result: dict[str, FactionDef] = {}

    for fid, data in raw.items():
        if not isinstance(data, dict):
            continue

        result[fid] = FactionDef(
            id=fid,
            name=safe_str(data.get("name"), default=fid),
            ethic=safe_str(data.get("guiding_ethic")),
            issues=_extract_issues(data),
            happiness_effects=(),
        )

    return result
