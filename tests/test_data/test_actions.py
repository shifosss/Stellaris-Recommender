"""Tests for action extraction from consecutive EmpireState diffs."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from stellaris_advisor.parser.models import (
    CouncilPosition,
    FleetState,
    MegastructureState,
)
from stellaris_advisor.data.actions import (
    ACTION_BUILDING_CONSTRUCTED,
    ACTION_COUNCIL_REASSIGNED,
    ACTION_DISTRICT_BUILT,
    ACTION_MEGASTRUCTURE_BUILT,
    ACTION_PERK_TAKEN,
    ACTION_POLICY_CHANGED,
    ACTION_SHIPS_BUILT,
    ACTION_SYSTEM_CLAIMED,
    ACTION_TECH_RESEARCHED,
    ACTION_TRADITION_UNLOCKED,
    MacroAction,
    extract_actions,
)
from tests.test_features.conftest import make_empire, make_planet


# ---------------------------------------------------------------------------
# Local mini-factories for action-specific models
# ---------------------------------------------------------------------------


def _fleet(fid: int, ships: dict[str, int]) -> FleetState:
    return FleetState(
        id=fid,
        name=f"Fleet {fid}",
        fleet_power=0.0,
        ships=MappingProxyType(ships),
        naval_capacity_used=0,
        location_system=0,
        commander=None,
    )


def _mega(mtype: str, system_id: int, stage: int) -> MegastructureState:
    return MegastructureState(
        megastructure_type=mtype, stage=stage, system_id=system_id
    )


def _council_pos(pos_type: str, leader_id: int | None) -> CouncilPosition:
    return CouncilPosition(
        position_type=pos_type,
        leader_id=leader_id,
        leader_name=None,
        leader_level=0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNoChange:
    def test_identical_states_return_empty_tuple(self) -> None:
        emp = make_empire()
        assert extract_actions(emp, emp) == ()

    def test_return_type_is_tuple_of_macro_actions(self) -> None:
        emp = make_empire()
        result = extract_actions(emp, emp)
        assert isinstance(result, tuple)
        assert all(isinstance(a, MacroAction) for a in result)

    def test_all_param_values_are_strings(self) -> None:
        base = make_empire(owned_systems=(1, 2))
        curr = make_empire(
            researched_techs=frozenset({"tech_a", "tech_b"}),
            owned_systems=(1, 2, 3),
        )
        actions = extract_actions(curr, base)
        for action in actions:
            for v in action.params.values():
                assert isinstance(v, str), f"Non-string param value in {action}"


@pytest.mark.unit
class TestTechResearched:
    def test_single_new_tech_detected(self) -> None:
        prev = make_empire(researched_techs=frozenset({"tech_a"}))
        curr = make_empire(researched_techs=frozenset({"tech_a", "tech_b"}))
        actions = [
            a
            for a in extract_actions(curr, prev)
            if a.action_type == ACTION_TECH_RESEARCHED
        ]
        assert len(actions) == 1
        assert actions[0].params["tech_id"] == "tech_b"

    def test_multiple_new_techs(self) -> None:
        prev = make_empire(researched_techs=frozenset())
        curr = make_empire(researched_techs=frozenset({"tech_a", "tech_b"}))
        actions = [
            a
            for a in extract_actions(curr, prev)
            if a.action_type == ACTION_TECH_RESEARCHED
        ]
        assert len(actions) == 2
        assert {a.params["tech_id"] for a in actions} == {"tech_a", "tech_b"}

    def test_no_tech_change_produces_no_action(self) -> None:
        emp = make_empire(researched_techs=frozenset({"tech_a"}))
        actions = [
            a
            for a in extract_actions(emp, emp)
            if a.action_type == ACTION_TECH_RESEARCHED
        ]
        assert actions == []


@pytest.mark.unit
class TestBuildingConstructed:
    def test_new_building_on_existing_planet(self) -> None:
        p_prev = make_planet(id=1, buildings=("building_capital",))
        p_curr = make_planet(id=1, buildings=("building_capital", "building_factory_1"))
        prev = make_empire(owned_planets=(p_prev,))
        curr = make_empire(owned_planets=(p_curr,))
        actions = [
            a
            for a in extract_actions(curr, prev)
            if a.action_type == ACTION_BUILDING_CONSTRUCTED
        ]
        assert len(actions) == 1
        assert actions[0].params["planet_id"] == "1"
        assert actions[0].params["building_type"] == "building_factory_1"

    def test_buildings_on_newly_colonized_planet_all_detected(self) -> None:
        p_new = make_planet(id=2, buildings=("building_capital",))
        prev = make_empire(owned_planets=())
        curr = make_empire(owned_planets=(p_new,))
        actions = [
            a
            for a in extract_actions(curr, prev)
            if a.action_type == ACTION_BUILDING_CONSTRUCTED
        ]
        assert len(actions) == 1
        assert actions[0].params["planet_id"] == "2"

    def test_no_building_change_produces_no_action(self) -> None:
        p = make_planet(id=1, buildings=("building_capital",))
        emp = make_empire(owned_planets=(p,))
        actions = [
            a
            for a in extract_actions(emp, emp)
            if a.action_type == ACTION_BUILDING_CONSTRUCTED
        ]
        assert actions == []


@pytest.mark.unit
class TestDistrictBuilt:
    def test_district_count_increase_detected(self) -> None:
        p_prev = make_planet(id=1, districts={"district_city": 1})
        p_curr = make_planet(id=1, districts={"district_city": 2})
        prev = make_empire(owned_planets=(p_prev,))
        curr = make_empire(owned_planets=(p_curr,))
        actions = [
            a
            for a in extract_actions(curr, prev)
            if a.action_type == ACTION_DISTRICT_BUILT
        ]
        assert len(actions) == 1
        assert actions[0].params["planet_id"] == "1"
        assert actions[0].params["district_type"] == "district_city"
        assert actions[0].params["count"] == "1"

    def test_no_district_change_produces_no_action(self) -> None:
        p = make_planet(id=1, districts={"district_city": 2})
        emp = make_empire(owned_planets=(p,))
        actions = [
            a
            for a in extract_actions(emp, emp)
            if a.action_type == ACTION_DISTRICT_BUILT
        ]
        assert actions == []


@pytest.mark.unit
class TestSystemClaimed:
    def test_new_system_detected(self) -> None:
        prev = make_empire(owned_systems=(1, 2))
        curr = make_empire(owned_systems=(1, 2, 3))
        actions = [
            a
            for a in extract_actions(curr, prev)
            if a.action_type == ACTION_SYSTEM_CLAIMED
        ]
        assert len(actions) == 1
        assert actions[0].params["system_id"] == "3"

    def test_no_system_change_produces_no_action(self) -> None:
        emp = make_empire(owned_systems=(1, 2))
        actions = [
            a
            for a in extract_actions(emp, emp)
            if a.action_type == ACTION_SYSTEM_CLAIMED
        ]
        assert actions == []


@pytest.mark.unit
class TestShipsBuilt:
    def test_count_increase_in_existing_fleet(self) -> None:
        prev = make_empire(fleets=(_fleet(1, {"corvette": 2}),))
        curr = make_empire(fleets=(_fleet(1, {"corvette": 5}),))
        actions = [
            a
            for a in extract_actions(curr, prev)
            if a.action_type == ACTION_SHIPS_BUILT
        ]
        assert len(actions) == 1
        assert actions[0].params["fleet_id"] == "1"
        assert actions[0].params["ship_type"] == "corvette"
        assert actions[0].params["count"] == "3"

    def test_ships_in_new_fleet_detected(self) -> None:
        prev = make_empire(fleets=())
        curr = make_empire(fleets=(_fleet(2, {"destroyer": 2}),))
        actions = [
            a
            for a in extract_actions(curr, prev)
            if a.action_type == ACTION_SHIPS_BUILT
        ]
        assert len(actions) == 1
        assert actions[0].params["fleet_id"] == "2"
        assert actions[0].params["ship_type"] == "destroyer"
        assert actions[0].params["count"] == "2"

    def test_no_ship_change_produces_no_action(self) -> None:
        emp = make_empire(fleets=(_fleet(1, {"corvette": 3}),))
        actions = [
            a for a in extract_actions(emp, emp) if a.action_type == ACTION_SHIPS_BUILT
        ]
        assert actions == []


@pytest.mark.unit
class TestTraditionUnlocked:
    def test_new_tradition_detected(self) -> None:
        prev = make_empire(traditions_unlocked=("tr_expansion_adopting",))
        curr = make_empire(
            traditions_unlocked=("tr_expansion_adopting", "tr_expansion_finish")
        )
        actions = [
            a
            for a in extract_actions(curr, prev)
            if a.action_type == ACTION_TRADITION_UNLOCKED
        ]
        assert len(actions) == 1
        assert actions[0].params["tradition_id"] == "tr_expansion_finish"

    def test_no_tradition_change_produces_no_action(self) -> None:
        emp = make_empire(traditions_unlocked=("tr_expansion_adopting",))
        actions = [
            a
            for a in extract_actions(emp, emp)
            if a.action_type == ACTION_TRADITION_UNLOCKED
        ]
        assert actions == []


@pytest.mark.unit
class TestPerkTaken:
    def test_new_perk_detected(self) -> None:
        prev = make_empire(ascension_perks=())
        curr = make_empire(ascension_perks=("ap_transcendence",))
        actions = [
            a for a in extract_actions(curr, prev) if a.action_type == ACTION_PERK_TAKEN
        ]
        assert len(actions) == 1
        assert actions[0].params["perk_id"] == "ap_transcendence"

    def test_no_perk_change_produces_no_action(self) -> None:
        emp = make_empire(ascension_perks=("ap_transcendence",))
        actions = [
            a for a in extract_actions(emp, emp) if a.action_type == ACTION_PERK_TAKEN
        ]
        assert actions == []


@pytest.mark.unit
class TestMegastructureBuilt:
    def test_new_megastructure_detected(self) -> None:
        prev = make_empire(megastructures=())
        curr = make_empire(megastructures=(_mega("megastructure_dyson_sphere", 10, 1),))
        actions = [
            a
            for a in extract_actions(curr, prev)
            if a.action_type == ACTION_MEGASTRUCTURE_BUILT
        ]
        assert len(actions) == 1
        assert actions[0].params["megastructure_type"] == "megastructure_dyson_sphere"
        assert actions[0].params["stage"] == "1"

    def test_stage_advance_detected(self) -> None:
        prev = make_empire(megastructures=(_mega("megastructure_dyson_sphere", 10, 1),))
        curr = make_empire(megastructures=(_mega("megastructure_dyson_sphere", 10, 2),))
        actions = [
            a
            for a in extract_actions(curr, prev)
            if a.action_type == ACTION_MEGASTRUCTURE_BUILT
        ]
        assert len(actions) == 1
        assert actions[0].params["stage"] == "2"

    def test_no_megastructure_change_produces_no_action(self) -> None:
        emp = make_empire(megastructures=(_mega("megastructure_dyson_sphere", 10, 1),))
        actions = [
            a
            for a in extract_actions(emp, emp)
            if a.action_type == ACTION_MEGASTRUCTURE_BUILT
        ]
        assert actions == []


@pytest.mark.unit
class TestPolicyChanged:
    def test_changed_policy_detected(self) -> None:
        prev = make_empire(policies=MappingProxyType({"economic": "socialism"}))
        curr = make_empire(policies=MappingProxyType({"economic": "capitalism"}))
        actions = [
            a
            for a in extract_actions(curr, prev)
            if a.action_type == ACTION_POLICY_CHANGED
        ]
        assert len(actions) == 1
        assert actions[0].params["policy_category"] == "economic"
        assert actions[0].params["new_value"] == "capitalism"
        assert actions[0].params["old_value"] == "socialism"

    def test_no_policy_change_produces_no_action(self) -> None:
        emp = make_empire(policies=MappingProxyType({"economic": "socialism"}))
        actions = [
            a
            for a in extract_actions(emp, emp)
            if a.action_type == ACTION_POLICY_CHANGED
        ]
        assert actions == []


@pytest.mark.unit
class TestCouncilReassigned:
    def test_leader_change_detected(self) -> None:
        prev = make_empire(council_positions=(_council_pos("chancellor", 1),))
        curr = make_empire(council_positions=(_council_pos("chancellor", 2),))
        actions = [
            a
            for a in extract_actions(curr, prev)
            if a.action_type == ACTION_COUNCIL_REASSIGNED
        ]
        assert len(actions) == 1
        assert actions[0].params["position_type"] == "chancellor"
        assert actions[0].params["new_leader_id"] == "2"
        assert actions[0].params["old_leader_id"] == "1"

    def test_new_position_filled(self) -> None:
        prev = make_empire(council_positions=())
        curr = make_empire(council_positions=(_council_pos("chancellor", 1),))
        actions = [
            a
            for a in extract_actions(curr, prev)
            if a.action_type == ACTION_COUNCIL_REASSIGNED
        ]
        assert len(actions) == 1
        assert actions[0].params["new_leader_id"] == "1"
        assert actions[0].params["old_leader_id"] == ""

    def test_no_council_change_produces_no_action(self) -> None:
        emp = make_empire(council_positions=(_council_pos("chancellor", 1),))
        actions = [
            a
            for a in extract_actions(emp, emp)
            if a.action_type == ACTION_COUNCIL_REASSIGNED
        ]
        assert actions == []
