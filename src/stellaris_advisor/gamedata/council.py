"""Council position and agenda loaders for the Stellaris gamedata layer."""

from __future__ import annotations

from typing import Any

from stellaris_advisor.gamedata.models import (
    AgendaDef,
    CouncilPositionDef,
    OutputModifier,
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


def load_council_defs(raw: dict[str, Any]) -> dict[str, CouncilPositionDef]:
    """Parse a raw Clausewitz council position dict into typed CouncilPositionDef instances.

    *raw* is the output of ``clausewitz.parse()`` applied to a council
    positions definition file (e.g. ``council_positions/00_council_positions.txt``).
    """
    result: dict[str, CouncilPositionDef] = {}

    for pid, data in raw.items():
        if not isinstance(data, dict):
            continue

        result[pid] = CouncilPositionDef(
            id=pid,
            name=safe_str(data.get("name"), default=pid),
            leader_class=safe_str(data.get("leader_class")),
            effects=_extract_modifiers(data.get("modifier")),
            max_positions=safe_int(data.get("max_council_positions"), default=1),
        )

    return result


def load_agenda_defs(raw: dict[str, Any]) -> dict[str, AgendaDef]:
    """Parse a raw Clausewitz agenda dict into typed AgendaDef instances.

    *raw* is the output of ``clausewitz.parse()`` applied to an agendas
    definition file (e.g. ``agendas/00_agendas.txt``).
    """
    result: dict[str, AgendaDef] = {}

    for aid, data in raw.items():
        if not isinstance(data, dict):
            continue

        result[aid] = AgendaDef(
            id=aid,
            name=safe_str(data.get("name"), default=aid),
            effects=_extract_modifiers(data.get("effect")),
            prerequisites=_extract_prerequisites(data),
        )

    return result


def optimal_council_assignment(
    council_defs: dict[str, CouncilPositionDef],
    target_effects: frozenset[str] = frozenset(),
) -> tuple[str, ...]:
    """Return council position IDs ordered by relevance and effect magnitude.

    When *target_effects* is non-empty, positions with effects on any of the
    target resources are ranked first.  Within each group (relevant / other),
    positions are sorted by the sum of absolute effect values descending.
    """

    def _total_magnitude(pos: CouncilPositionDef) -> float:
        return sum(abs(m.amount) for m in pos.effects)

    def _is_relevant(pos: CouncilPositionDef) -> bool:
        return bool(target_effects) and any(
            m.resource in target_effects for m in pos.effects
        )

    return tuple(
        sorted(
            council_defs,
            key=lambda pid: (
                _is_relevant(council_defs[pid]),
                _total_magnitude(council_defs[pid]),
            ),
            reverse=True,
        )
    )


def agenda_prerequisites_met(
    agenda_def: AgendaDef,
    researched: frozenset[str],
) -> bool:
    """Return True if all tech prerequisites for *agenda_def* are in *researched*."""
    return all(p in researched for p in agenda_def.prerequisites)
