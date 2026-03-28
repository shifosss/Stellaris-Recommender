"""Tests for gamedata/ships.py and gamedata/planet_classes.py — TDD: tests written first.

Covers:
  ships:         load_ship_class_defs(raw_sizes, raw_components)
  planet_classes: load_planet_class_defs(raw)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "game_data"


# ---------------------------------------------------------------------------
# Raw-dict helpers
# ---------------------------------------------------------------------------


def _raw_ship(
    sid: str,
    max_hitpoints: int = 200,
    max_speed: float = 1.0,
    evasion_mult: float = 0.3,
    cost: dict[str, Any] | None = None,
    upkeep: dict[str, Any] | None = None,
    prerequisites: str | list[str] | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "name": sid,
        "ship_class": "military",
        "max_hitpoints": max_hitpoints,
        "max_speed": max_speed,
        "modifier": {"evasion_mult": evasion_mult},
        "cost": cost or {"alloys": 40},
        "upkeep": upkeep or {"energy": 1},
    }
    if prerequisites is not None:
        data["prerequisites"] = prerequisites
    return data


def _raw_planet_class(
    pc_id: str,
    climate: str = "wet",
    habitability: float = 1.0,
    district_cap: int = 20,
    modifier: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "climate": climate,
        "habitability": habitability,
        "district_cap": district_cap,
        "colonizable": True,
        "modifier": modifier or {},
    }


# ===========================================================================
# load_ship_class_defs
# ===========================================================================


@pytest.mark.unit
class TestLoadShipClassDefs:
    def test_basic_fields(self):
        from stellaris_advisor.gamedata.ships import load_ship_class_defs

        raw = {"corvette": _raw_ship("corvette", max_hitpoints=200, max_speed=1.2)}
        defs = load_ship_class_defs(raw)
        assert "corvette" in defs
        assert defs["corvette"].id == "corvette"
        assert defs["corvette"].hull == 200

    def test_speed_extracted(self):
        from stellaris_advisor.gamedata.ships import load_ship_class_defs

        raw = {"corvette": _raw_ship("corvette", max_speed=1.2)}
        defs = load_ship_class_defs(raw)
        assert defs["corvette"].speed == pytest.approx(1.2)

    def test_evasion_from_modifier(self):
        from stellaris_advisor.gamedata.ships import load_ship_class_defs

        raw = {"corvette": _raw_ship("corvette", evasion_mult=0.60)}
        defs = load_ship_class_defs(raw)
        assert defs["corvette"].evasion == pytest.approx(0.60)

    def test_cost_extraction(self):
        from stellaris_advisor.gamedata.ships import load_ship_class_defs

        raw = {"corvette": _raw_ship("corvette", cost={"alloys": 40})}
        defs = load_ship_class_defs(raw)
        costs = {m.resource: m.amount for m in defs["corvette"].base_cost}
        assert costs["alloys"] == 40.0

    def test_upkeep_extraction(self):
        from stellaris_advisor.gamedata.ships import load_ship_class_defs

        raw = {"corvette": _raw_ship("corvette", upkeep={"energy": 1})}
        defs = load_ship_class_defs(raw)
        upkeep = {m.resource: m.amount for m in defs["corvette"].upkeep}
        assert upkeep["energy"] == 1.0

    def test_prerequisite_as_string(self):
        from stellaris_advisor.gamedata.ships import load_ship_class_defs

        raw = {"corvette": _raw_ship("corvette", prerequisites="tech_corvettes")}
        defs = load_ship_class_defs(raw)
        assert defs["corvette"].prerequisites == frozenset({"tech_corvettes"})

    def test_prerequisite_as_list(self):
        from stellaris_advisor.gamedata.ships import load_ship_class_defs

        raw = {"battleship": _raw_ship("battleship", prerequisites=["tech_a", "tech_b"])}
        defs = load_ship_class_defs(raw)
        assert defs["battleship"].prerequisites == frozenset({"tech_a", "tech_b"})

    def test_no_prerequisites(self):
        from stellaris_advisor.gamedata.ships import load_ship_class_defs

        raw = {"probe": _raw_ship("probe")}
        defs = load_ship_class_defs(raw)
        assert defs["probe"].prerequisites == frozenset()

    def test_raw_components_none_gives_empty_sections(self):
        from stellaris_advisor.gamedata.ships import load_ship_class_defs

        raw = {"corvette": _raw_ship("corvette")}
        defs = load_ship_class_defs(raw)
        assert defs["corvette"].sections == ()

    def test_raw_components_empty_gives_empty_sections(self):
        from stellaris_advisor.gamedata.ships import load_ship_class_defs

        raw = {"corvette": _raw_ship("corvette")}
        defs = load_ship_class_defs(raw, raw_components={})
        assert defs["corvette"].sections == ()

    def test_section_extracted_from_components(self):
        from stellaris_advisor.gamedata.ships import load_ship_class_defs

        raw_sizes = {"corvette": _raw_ship("corvette")}
        raw_components = {
            "corvette_mid": {
                "ship_size": "corvette",
                "slot_count": 3,
                "component_slot": [{"name": "SMALL_GUN_01"}, {"name": "SMALL_GUN_02"}],
                "utility_slot": [{"name": "SHIELD_01"}],
            }
        }
        defs = load_ship_class_defs(raw_sizes, raw_components=raw_components)
        assert len(defs["corvette"].sections) == 1
        assert defs["corvette"].sections[0].id == "corvette_mid"

    def test_returns_frozen_dataclasses(self):
        from stellaris_advisor.gamedata.ships import load_ship_class_defs

        defs = load_ship_class_defs({"corvette": _raw_ship("corvette")})
        with pytest.raises(AttributeError):
            defs["corvette"].hull = 999  # type: ignore[misc]

    def test_empty_raw_returns_empty(self):
        from stellaris_advisor.gamedata.ships import load_ship_class_defs

        assert load_ship_class_defs({}) == {}


# ---------------------------------------------------------------------------
# Integration — real ship_sizes fixture
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestShipClassDefsFromFixtures:
    @pytest.fixture
    def ship_raw(self):
        from stellaris_advisor.parser.clausewitz import parse

        return parse(_FIXTURE_DIR.joinpath("ship_sizes.txt").read_text())

    def test_loads_all_ship_sizes(self, ship_raw):
        from stellaris_advisor.gamedata.ships import load_ship_class_defs

        defs = load_ship_class_defs(ship_raw)
        for sid in ("corvette", "destroyer", "cruiser"):
            assert sid in defs

    def test_corvette_hull(self, ship_raw):
        from stellaris_advisor.gamedata.ships import load_ship_class_defs

        defs = load_ship_class_defs(ship_raw)
        assert defs["corvette"].hull == 200

    def test_destroyer_hull(self, ship_raw):
        from stellaris_advisor.gamedata.ships import load_ship_class_defs

        defs = load_ship_class_defs(ship_raw)
        assert defs["destroyer"].hull == 500

    def test_corvette_evasion(self, ship_raw):
        from stellaris_advisor.gamedata.ships import load_ship_class_defs

        defs = load_ship_class_defs(ship_raw)
        assert defs["corvette"].evasion == pytest.approx(0.60)

    def test_corvette_prerequisites(self, ship_raw):
        from stellaris_advisor.gamedata.ships import load_ship_class_defs

        defs = load_ship_class_defs(ship_raw)
        assert "tech_corvettes" in defs["corvette"].prerequisites

    def test_cruiser_cost(self, ship_raw):
        from stellaris_advisor.gamedata.ships import load_ship_class_defs

        defs = load_ship_class_defs(ship_raw)
        costs = {m.resource: m.amount for m in defs["cruiser"].base_cost}
        assert costs["alloys"] == 200.0


# ===========================================================================
# load_planet_class_defs
# ===========================================================================


@pytest.mark.unit
class TestLoadPlanetClassDefs:
    def test_basic_fields(self):
        from stellaris_advisor.gamedata.planet_classes import load_planet_class_defs

        raw = {"pc_continental": _raw_planet_class("pc_continental", climate="wet", habitability=1.0)}
        defs = load_planet_class_defs(raw)
        assert "pc_continental" in defs
        assert defs["pc_continental"].id == "pc_continental"
        assert defs["pc_continental"].habitability_category == "wet"

    def test_base_habitability(self):
        from stellaris_advisor.gamedata.planet_classes import load_planet_class_defs

        raw = {"pc_arid": _raw_planet_class("pc_arid", habitability=0.60)}
        defs = load_planet_class_defs(raw)
        assert defs["pc_arid"].base_habitability == pytest.approx(0.60)

    def test_district_cap(self):
        from stellaris_advisor.gamedata.planet_classes import load_planet_class_defs

        raw = {"pc_continental": _raw_planet_class("pc_continental", district_cap=20)}
        defs = load_planet_class_defs(raw)
        assert defs["pc_continental"].district_cap == 20

    def test_modifiers_extracted(self):
        from stellaris_advisor.gamedata.planet_classes import load_planet_class_defs

        raw = {
            "pc_arid": _raw_planet_class(
                "pc_arid", modifier={"planet_max_districts_add": -2, "pop_growth_speed": -0.05}
            )
        }
        defs = load_planet_class_defs(raw)
        modifiers = {m.resource: m.amount for m in defs["pc_arid"].modifiers}
        assert modifiers["planet_max_districts_add"] == pytest.approx(-2.0)
        assert modifiers["pop_growth_speed"] == pytest.approx(-0.05)

    def test_empty_modifier_block(self):
        from stellaris_advisor.gamedata.planet_classes import load_planet_class_defs

        raw = {"pc_continental": _raw_planet_class("pc_continental", modifier={})}
        defs = load_planet_class_defs(raw)
        assert defs["pc_continental"].modifiers == ()

    def test_climate_none_for_habitat(self):
        from stellaris_advisor.gamedata.planet_classes import load_planet_class_defs

        raw = {"pc_habitat": _raw_planet_class("pc_habitat", climate="none")}
        defs = load_planet_class_defs(raw)
        assert defs["pc_habitat"].habitability_category == "none"

    def test_returns_frozen_dataclasses(self):
        from stellaris_advisor.gamedata.planet_classes import load_planet_class_defs

        defs = load_planet_class_defs({"pc_x": _raw_planet_class("pc_x")})
        with pytest.raises(AttributeError):
            defs["pc_x"].district_cap = 99  # type: ignore[misc]

    def test_empty_raw_returns_empty(self):
        from stellaris_advisor.gamedata.planet_classes import load_planet_class_defs

        assert load_planet_class_defs({}) == {}


# ---------------------------------------------------------------------------
# Integration — real planet_classes fixture
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestPlanetClassDefsFromFixtures:
    @pytest.fixture
    def pc_raw(self):
        from stellaris_advisor.parser.clausewitz import parse

        return parse(_FIXTURE_DIR.joinpath("planet_classes.txt").read_text())

    def test_loads_all_planet_classes(self, pc_raw):
        from stellaris_advisor.gamedata.planet_classes import load_planet_class_defs

        defs = load_planet_class_defs(pc_raw)
        for pc_id in ("pc_continental", "pc_arid", "pc_tomb", "pc_habitat"):
            assert pc_id in defs

    def test_continental_habitability(self, pc_raw):
        from stellaris_advisor.gamedata.planet_classes import load_planet_class_defs

        defs = load_planet_class_defs(pc_raw)
        assert defs["pc_continental"].base_habitability == pytest.approx(1.0)
        assert defs["pc_continental"].habitability_category == "wet"

    def test_arid_district_cap(self, pc_raw):
        from stellaris_advisor.gamedata.planet_classes import load_planet_class_defs

        defs = load_planet_class_defs(pc_raw)
        assert defs["pc_arid"].district_cap == 18

    def test_tomb_has_modifiers(self, pc_raw):
        from stellaris_advisor.gamedata.planet_classes import load_planet_class_defs

        defs = load_planet_class_defs(pc_raw)
        modifiers = {m.resource: m.amount for m in defs["pc_tomb"].modifiers}
        assert "pop_happiness" in modifiers
        assert modifiers["pop_happiness"] == pytest.approx(-0.10)

    def test_habitat_climate_none(self, pc_raw):
        from stellaris_advisor.gamedata.planet_classes import load_planet_class_defs

        defs = load_planet_class_defs(pc_raw)
        assert defs["pc_habitat"].habitability_category == "none"
        assert defs["pc_habitat"].district_cap == 8
