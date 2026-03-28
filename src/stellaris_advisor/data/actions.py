"""Action extraction from consecutive EmpireState diffs.

Diffs two consecutive EmpireState snapshots to infer the macro-actions
taken by the player between save files.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from stellaris_advisor.parser.models import EmpireState

# ---------------------------------------------------------------------------
# Action type string constants
# ---------------------------------------------------------------------------

ACTION_TECH_RESEARCHED = "tech_researched"
ACTION_BUILDING_CONSTRUCTED = "building_constructed"
ACTION_DISTRICT_BUILT = "district_built"
ACTION_SYSTEM_CLAIMED = "system_claimed"
ACTION_SHIPS_BUILT = "ships_built"
ACTION_TRADITION_UNLOCKED = "tradition_unlocked"
ACTION_PERK_TAKEN = "perk_taken"
ACTION_MEGASTRUCTURE_BUILT = "megastructure_built"
ACTION_POLICY_CHANGED = "policy_changed"
ACTION_COUNCIL_REASSIGNED = "council_reassigned"


# ---------------------------------------------------------------------------
# MacroAction model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MacroAction:
    """An inferred macro-action extracted from a consecutive empire state diff."""

    action_type: str
    params: MappingProxyType[str, str]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_actions(
    current: EmpireState, previous: EmpireState
) -> tuple[MacroAction, ...]:
    """Return all macro-actions inferred by diffing current vs previous state."""
    actions: list[MacroAction] = []
    actions.extend(_tech_actions(current, previous))
    actions.extend(_building_actions(current, previous))
    actions.extend(_district_actions(current, previous))
    actions.extend(_system_actions(current, previous))
    actions.extend(_ship_actions(current, previous))
    actions.extend(_tradition_actions(current, previous))
    actions.extend(_perk_actions(current, previous))
    actions.extend(_megastructure_actions(current, previous))
    actions.extend(_policy_actions(current, previous))
    actions.extend(_council_actions(current, previous))
    return tuple(actions)


# ---------------------------------------------------------------------------
# Per-category extractors (each returns list[MacroAction])
# ---------------------------------------------------------------------------


def _tech_actions(curr: EmpireState, prev: EmpireState) -> list[MacroAction]:
    return [
        MacroAction(ACTION_TECH_RESEARCHED, MappingProxyType({"tech_id": tech}))
        for tech in curr.researched_techs - prev.researched_techs
    ]


def _building_actions(curr: EmpireState, prev: EmpireState) -> list[MacroAction]:
    prev_buildings: dict[int, frozenset[str]] = {
        p.id: frozenset(p.buildings) for p in prev.owned_planets
    }
    actions: list[MacroAction] = []
    for planet in curr.owned_planets:
        prev_set = prev_buildings.get(planet.id, frozenset())
        for bldg in frozenset(planet.buildings) - prev_set:
            actions.append(
                MacroAction(
                    ACTION_BUILDING_CONSTRUCTED,
                    MappingProxyType(
                        {"planet_id": str(planet.id), "building_type": bldg}
                    ),
                )
            )
    return actions


def _district_actions(curr: EmpireState, prev: EmpireState) -> list[MacroAction]:
    prev_districts: dict[int, dict[str, int]] = {
        p.id: dict(p.districts) for p in prev.owned_planets
    }
    actions: list[MacroAction] = []
    for planet in curr.owned_planets:
        prev_d = prev_districts.get(planet.id, {})
        for dtype, count in planet.districts.items():
            delta = count - prev_d.get(dtype, 0)
            if delta > 0:
                actions.append(
                    MacroAction(
                        ACTION_DISTRICT_BUILT,
                        MappingProxyType(
                            {
                                "planet_id": str(planet.id),
                                "district_type": dtype,
                                "count": str(delta),
                            }
                        ),
                    )
                )
    return actions


def _system_actions(curr: EmpireState, prev: EmpireState) -> list[MacroAction]:
    return [
        MacroAction(ACTION_SYSTEM_CLAIMED, MappingProxyType({"system_id": str(sid)}))
        for sid in set(curr.owned_systems) - set(prev.owned_systems)
    ]


def _ship_actions(curr: EmpireState, prev: EmpireState) -> list[MacroAction]:
    prev_fleets: dict[int, dict[str, int]] = {
        f.id: dict(f.ships) for f in prev.fleets
    }
    actions: list[MacroAction] = []
    for fleet in curr.fleets:
        prev_ships = prev_fleets.get(fleet.id, {})
        for ship_type, count in fleet.ships.items():
            delta = count - prev_ships.get(ship_type, 0)
            if delta > 0:
                actions.append(
                    MacroAction(
                        ACTION_SHIPS_BUILT,
                        MappingProxyType(
                            {
                                "fleet_id": str(fleet.id),
                                "ship_type": ship_type,
                                "count": str(delta),
                            }
                        ),
                    )
                )
    return actions


def _tradition_actions(curr: EmpireState, prev: EmpireState) -> list[MacroAction]:
    return [
        MacroAction(ACTION_TRADITION_UNLOCKED, MappingProxyType({"tradition_id": trad}))
        for trad in set(curr.traditions_unlocked) - set(prev.traditions_unlocked)
    ]


def _perk_actions(curr: EmpireState, prev: EmpireState) -> list[MacroAction]:
    return [
        MacroAction(ACTION_PERK_TAKEN, MappingProxyType({"perk_id": perk}))
        for perk in set(curr.ascension_perks) - set(prev.ascension_perks)
    ]


def _megastructure_actions(curr: EmpireState, prev: EmpireState) -> list[MacroAction]:
    prev_mega: dict[int, int] = {m.system_id: m.stage for m in prev.megastructures}
    actions: list[MacroAction] = []
    for m in curr.megastructures:
        prev_stage = prev_mega.get(m.system_id)
        if prev_stage is None or m.stage > prev_stage:
            actions.append(
                MacroAction(
                    ACTION_MEGASTRUCTURE_BUILT,
                    MappingProxyType(
                        {
                            "megastructure_type": m.megastructure_type,
                            "stage": str(m.stage),
                        }
                    ),
                )
            )
    return actions


def _policy_actions(curr: EmpireState, prev: EmpireState) -> list[MacroAction]:
    actions: list[MacroAction] = []
    for policy_category, new_val in curr.policies.items():
        old_val = prev.policies.get(policy_category, "")
        if old_val != new_val:
            actions.append(
                MacroAction(
                    ACTION_POLICY_CHANGED,
                    MappingProxyType(
                        {
                            "policy_category": policy_category,
                            "new_value": new_val,
                            "old_value": old_val,
                        }
                    ),
                )
            )
    return actions


def _council_actions(curr: EmpireState, prev: EmpireState) -> list[MacroAction]:
    prev_council: dict[str, int | None] = {
        p.position_type: p.leader_id for p in prev.council_positions
    }
    actions: list[MacroAction] = []
    for pos in curr.council_positions:
        if pos.leader_id is None:
            continue
        old_id = prev_council.get(pos.position_type)
        if old_id != pos.leader_id:
            actions.append(
                MacroAction(
                    ACTION_COUNCIL_REASSIGNED,
                    MappingProxyType(
                        {
                            "position_type": pos.position_type,
                            "new_leader_id": str(pos.leader_id),
                            "old_leader_id": str(old_id) if old_id is not None else "",
                        }
                    ),
                )
            )
    return actions
