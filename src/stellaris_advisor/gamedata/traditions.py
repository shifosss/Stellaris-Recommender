"""Tradition loader and query functions for the Stellaris gamedata layer."""

from __future__ import annotations

from typing import Any

from stellaris_advisor.gamedata.models import TraditionDef, TraditionTree
from stellaris_advisor.parser.extractor import safe_int, safe_str, to_str_tuple


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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


def load_tradition_defs(raw: dict[str, Any]) -> dict[str, TraditionDef]:
    """Parse a flat Clausewitz tradition dict into typed TraditionDef instances.

    Individual traditions are identified by the presence of a ``tree`` key.
    Tree-level entries (which have a ``traditions`` key instead) are skipped.
    """
    result: dict[str, TraditionDef] = {}

    for tid, data in raw.items():
        if not isinstance(data, dict):
            continue
        # Tree entries have "traditions" (list) or no "tree" key
        if "tree" not in data:
            continue

        result[tid] = TraditionDef(
            id=tid,
            name=safe_str(data.get("name"), default=tid),
            tree_id=safe_str(data.get("tree")),
            cost=safe_int(data.get("cost")),
            effect="",
            prerequisites=_extract_prerequisites(data),
        )

    return result


def load_tradition_tree_defs(raw: dict[str, Any]) -> dict[str, TraditionTree]:
    """Parse a flat Clausewitz tradition dict into typed TraditionTree instances.

    Tree entries are identified by the presence of a ``traditions`` key.
    Individual tradition entries (which have a ``tree`` key) are skipped.
    """
    result: dict[str, TraditionTree] = {}

    for tid, data in raw.items():
        if not isinstance(data, dict):
            continue
        # Skip individual traditions — they have a "tree" key
        if "tree" in data:
            continue
        # Must have "traditions" list to be a valid tree
        if "traditions" not in data:
            continue

        result[tid] = TraditionTree(
            id=tid,
            name=safe_str(data.get("name"), default=tid),
            tradition_ids=to_str_tuple(data.get("traditions")),
            adopt_effect=safe_str(data.get("adopt_effect")),
            finish_effect=safe_str(data.get("finish_effect")),
            prerequisites=_extract_prerequisites(data),
        )

    return result


def available_traditions(
    tradition_defs: dict[str, TraditionDef],
    opened_trees: frozenset[str],
    unlocked_traditions: frozenset[str],
    empire_type: str = "",
) -> frozenset[str]:
    """Return tradition IDs that can currently be picked.

    A tradition is available when ALL of:
    1. Its tree is in *opened_trees*.
    2. It is not already in *unlocked_traditions*.
    3. All its prerequisites are in *unlocked_traditions*.

    *empire_type* is reserved for future gestalt/organic filtering.
    # TODO: filter by empire_type (gestalt empires cannot unlock organic traditions)
    """
    result: set[str] = set()
    for tid, trad in tradition_defs.items():
        if tid in unlocked_traditions:
            continue
        if trad.tree_id not in opened_trees:
            continue
        if not all(p in unlocked_traditions for p in trad.prerequisites):
            continue
        result.add(tid)
    return frozenset(result)


def tradition_tree_progress(
    tradition_tree: TraditionTree,
    unlocked: frozenset[str],
) -> tuple[int, int]:
    """Return (completed, total) tradition counts for *tradition_tree*.

    *completed* is the number of the tree's tradition IDs present in *unlocked*.
    *total* is the number of traditions listed in the tree.
    """
    completed = sum(1 for t_id in tradition_tree.tradition_ids if t_id in unlocked)
    return (completed, len(tradition_tree.tradition_ids))
