"""Tests for extraction helpers and typed extraction functions."""

import pytest
from types import MappingProxyType


# ---------------------------------------------------------------------------
# Path navigation + safe conversions
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetPath:
    def test_simple_path(self):
        from stellaris_advisor.parser.extractor import get_path

        data = {"a": {"b": {"c": 42}}}
        assert get_path(data, "a.b.c") == 42

    def test_missing_path_returns_default(self):
        from stellaris_advisor.parser.extractor import get_path

        assert get_path({"a": 1}, "a.b.c") is None

    def test_missing_path_returns_custom_default(self):
        from stellaris_advisor.parser.extractor import get_path

        assert get_path({}, "x.y", default="fallback") == "fallback"

    def test_single_key(self):
        from stellaris_advisor.parser.extractor import get_path

        assert get_path({"key": "val"}, "key") == "val"

    def test_non_dict_intermediate(self):
        from stellaris_advisor.parser.extractor import get_path

        assert get_path({"a": 5}, "a.b") is None


@pytest.mark.unit
class TestSafeConversions:
    def test_safe_float_valid(self):
        from stellaris_advisor.parser.extractor import safe_float

        assert safe_float(3.14) == pytest.approx(3.14)

    def test_safe_float_from_int(self):
        from stellaris_advisor.parser.extractor import safe_float

        assert safe_float(5) == 5.0

    def test_safe_float_none_returns_default(self):
        from stellaris_advisor.parser.extractor import safe_float

        assert safe_float(None) == 0.0

    def test_safe_float_string_returns_default(self):
        from stellaris_advisor.parser.extractor import safe_float

        assert safe_float("not_a_number", default=-1.0) == -1.0

    def test_safe_int_valid(self):
        from stellaris_advisor.parser.extractor import safe_int

        assert safe_int(42) == 42

    def test_safe_int_none_returns_default(self):
        from stellaris_advisor.parser.extractor import safe_int

        assert safe_int(None) == 0

    def test_safe_str_valid(self):
        from stellaris_advisor.parser.extractor import safe_str

        assert safe_str("hello") == "hello"

    def test_safe_str_none_returns_default(self):
        from stellaris_advisor.parser.extractor import safe_str

        assert safe_str(None) == ""

    def test_safe_bool_valid(self):
        from stellaris_advisor.parser.extractor import safe_bool

        assert safe_bool(True) is True

    def test_safe_bool_none_returns_default(self):
        from stellaris_advisor.parser.extractor import safe_bool

        assert safe_bool(None) is False


@pytest.mark.unit
class TestTupleConverters:
    def test_to_str_tuple_from_list(self):
        from stellaris_advisor.parser.extractor import to_str_tuple

        assert to_str_tuple(["a", "b", "c"]) == ("a", "b", "c")

    def test_to_str_tuple_from_none(self):
        from stellaris_advisor.parser.extractor import to_str_tuple

        assert to_str_tuple(None) == ()

    def test_to_str_tuple_from_single_string(self):
        from stellaris_advisor.parser.extractor import to_str_tuple

        assert to_str_tuple("solo") == ("solo",)

    def test_to_int_tuple_from_list(self):
        from stellaris_advisor.parser.extractor import to_int_tuple

        assert to_int_tuple([1, 2, 3]) == (1, 2, 3)

    def test_to_int_tuple_from_none(self):
        from stellaris_advisor.parser.extractor import to_int_tuple

        assert to_int_tuple(None) == ()

    def test_to_int_tuple_from_single_int(self):
        from stellaris_advisor.parser.extractor import to_int_tuple

        assert to_int_tuple(7) == (7,)


@pytest.mark.unit
class TestToFrozenMap:
    def test_from_dict(self):
        from stellaris_advisor.parser.extractor import to_frozen_map

        result = to_frozen_map({"energy": 100, "minerals": 200})
        assert isinstance(result, MappingProxyType)
        assert result["energy"] == 100.0
        assert result["minerals"] == 200.0

    def test_from_none(self):
        from stellaris_advisor.parser.extractor import to_frozen_map

        result = to_frozen_map(None)
        assert isinstance(result, MappingProxyType)
        assert len(result) == 0

    def test_with_int_converter(self):
        from stellaris_advisor.parser.extractor import to_frozen_map

        result = to_frozen_map({"ships": 5.0}, value_converter=int)
        assert result["ships"] == 5
        assert isinstance(result["ships"], int)


# ---------------------------------------------------------------------------
# Typed extraction
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExtractEmpire:
    def test_extract_empire_basic_fields(self):
        from stellaris_advisor.parser.extractor import extract_empire

        raw = {
            "name": "United Nations of Earth",
            "gov": {
                "authority": "auth_democratic",
                "type": "gov_democratic",
                "civics": ["civic_beacon_of_liberty", "civic_idealistic_foundation"],
            },
            "ethos": {"ethic": ["ethic_fanatic_egalitarian", "ethic_xenophile"]},
            "origin": "origin_default",
            "budget": {
                "current_month": {
                    "income": {"energy": 150.0, "minerals": 200.0},
                    "expenses": {"energy": 100.0},
                },
            },
            "modules": {
                "standard_economy_module": {
                    "resources": {
                        "energy": {"balance": 50.0},
                    },
                },
            },
            "tech_status": {
                "technology": {
                    "0": {"technology": "tech_corvettes", "level": 1},
                    "1": {"technology": "tech_destroyers", "level": 1},
                },
            },
            "fleet_power": 5000.0,
            "naval_capacity": {"used": 30, "max": 80},
            "traditions": ["tradition_expansion_adopt"],
            "ascension_perks": ["ap_technological_ascendancy"],
        }
        empire = extract_empire(raw, empire_id=0, version="4.3")
        assert empire.name == "United Nations of Earth"
        assert empire.authority == "auth_democratic"
        assert empire.ethics == ("ethic_fanatic_egalitarian", "ethic_xenophile")
        assert empire.civics == (
            "civic_beacon_of_liberty",
            "civic_idealistic_foundation",
        )
        assert empire.origin == "origin_default"
        assert empire.resource_income["energy"] == pytest.approx(150.0)
        assert "tech_corvettes" in empire.researched_techs
        assert "tech_destroyers" in empire.researched_techs
        assert empire.fleet_power == pytest.approx(5000.0)
        assert empire.naval_capacity_used == 30
        assert empire.naval_capacity_max == 80
        assert empire.traditions_unlocked == ("tradition_expansion_adopt",)
        assert empire.ascension_perks == ("ap_technological_ascendancy",)

    def test_extract_empire_missing_optional_fields(self):
        from stellaris_advisor.parser.extractor import extract_empire

        raw = {"name": "Minimal Empire"}
        empire = extract_empire(raw, empire_id=1, version="4.3")
        assert empire.id == 1
        assert empire.name == "Minimal Empire"
        assert empire.authority == ""
        assert empire.ethics == ()
        assert empire.researched_techs == frozenset()
        assert empire.fleet_power == 0.0
        assert empire.federation is None
        assert empire.current_agenda is None
        assert empire.factions == ()
        assert empire.leaders == ()


@pytest.mark.unit
class TestExtractPlanet:
    def test_extract_planet_basic(self):
        from stellaris_advisor.parser.extractor import extract_planet

        raw = {
            "name": "Earth",
            "planet_class": "pc_continental",
            "planet_size": 25,
            "districts": {"district_city": 3, "district_generator": 2},
            "district_slots": {"district_city": 10, "district_generator": 8},
            "buildings": ["building_capital_2", "building_alloy_foundry"],
            "pop": 30,
            "stability": 55.0,
            "crime": 5.0,
            "amenities": 10.0,
            "amenities_usage": 8.0,
            "free_housing": 10.0,
            "housing_usage": 30.0,
            "designation": "col_default",
            "is_capital": True,
        }
        planet = extract_planet(raw, planet_id=0, version="4.3")
        assert planet.id == 0
        assert planet.name == "Earth"
        assert planet.planet_class == "pc_continental"
        assert planet.size == 25
        assert planet.districts["district_city"] == 3
        assert planet.buildings == ("building_capital_2", "building_alloy_foundry")
        assert planet.pops == 30
        assert planet.stability == pytest.approx(55.0)
        assert planet.is_capital is True

    def test_extract_planet_minimal(self):
        from stellaris_advisor.parser.extractor import extract_planet

        raw = {"name": "Barren World", "planet_class": "pc_barren", "planet_size": 10}
        planet = extract_planet(raw, planet_id=5, version="4.3")
        assert planet.id == 5
        assert planet.name == "Barren World"
        assert planet.pops == 0
        assert planet.buildings == ()
        assert planet.is_capital is False


@pytest.mark.unit
class TestExtractEmpireHelpers:
    """Cover private extraction helpers via extract_empire with rich raw data."""

    def test_extract_leaders(self):
        from stellaris_advisor.parser.extractor import extract_empire

        raw = {
            "name": "Empire",
            "leaders": {
                "0": {
                    "id": 1,
                    "name": "Leader One",
                    "class": "admiral",
                    "level": 3,
                    "traits": ["trait_aggressive"],
                    "expertise": "military",
                    "assignment": "fleet_1",
                },
            },
        }
        empire = extract_empire(raw, empire_id=0, version="4.3")
        assert len(empire.leaders) == 1
        assert empire.leaders[0].name == "Leader One"
        assert empire.leaders[0].leader_class == "admiral"
        assert empire.leaders[0].traits == ("trait_aggressive",)

    def test_extract_factions(self):
        from stellaris_advisor.parser.extractor import extract_empire

        raw = {
            "name": "Empire",
            "factions": {
                "0": {
                    "ethic": "ethic_egalitarian",
                    "approval": 65.0,
                    "size": 12,
                    "support": 0.3,
                    "issues": {"faction_demand_democracy": True},
                },
            },
        }
        empire = extract_empire(raw, empire_id=0, version="4.3")
        assert len(empire.factions) == 1
        assert empire.factions[0].ethic == "ethic_egalitarian"
        assert empire.factions[0].approval == pytest.approx(65.0)
        assert len(empire.factions[0].issues) == 1
        assert empire.factions[0].issues[0].is_fulfilled is True

    def test_extract_federation(self):
        from stellaris_advisor.parser.extractor import extract_empire

        raw = {
            "name": "Empire",
            "federation_details": {
                "federation_type": "default_federation",
                "level": 3,
                "cohesion": 80.0,
                "experience": 500.0,
                "fleet_power": 12000.0,
            },
        }
        empire = extract_empire(raw, empire_id=0, version="4.3")
        assert empire.federation is not None
        assert empire.federation.federation_type == "default_federation"
        assert empire.federation.level == 3

    def test_extract_agenda(self):
        from stellaris_advisor.parser.extractor import extract_empire

        raw = {
            "name": "Empire",
            "current_agenda": {
                "agenda_type": "agenda_military",
                "progress": 0.5,
                "is_launched": True,
            },
        }
        empire = extract_empire(raw, empire_id=0, version="4.3")
        assert empire.current_agenda is not None
        assert empire.current_agenda.agenda_type == "agenda_military"
        assert empire.current_agenda.is_launched is True

    def test_extract_council_positions(self):
        from stellaris_advisor.parser.extractor import extract_empire

        raw = {
            "name": "Empire",
            "council_positions": {
                "0": {
                    "position_type": "ruler",
                    "leader_id": 1,
                    "leader_name": "President",
                    "leader_level": 5,
                },
            },
        }
        empire = extract_empire(raw, empire_id=0, version="4.3")
        assert len(empire.council_positions) == 1
        assert empire.council_positions[0].position_type == "ruler"

    def test_extract_megastructures(self):
        from stellaris_advisor.parser.extractor import extract_empire

        raw = {
            "name": "Empire",
            "megastructures": {
                "0": {"type": "dyson_sphere", "stage": 3, "system_id": 42},
            },
        }
        empire = extract_empire(raw, empire_id=0, version="4.3")
        assert len(empire.megastructures) == 1
        assert empire.megastructures[0].megastructure_type == "dyson_sphere"
        assert empire.megastructures[0].stage == 3

    def test_extract_wars(self):
        from stellaris_advisor.parser.extractor import extract_empire

        raw = {
            "name": "Empire",
            "active_wars": {
                "0": {
                    "war_id": 1,
                    "name": "First Contact War",
                    "war_goal": "wg_conquest",
                    "exhaustion": 25.0,
                    "is_attacker": True,
                },
            },
        }
        empire = extract_empire(raw, empire_id=0, version="4.3")
        assert len(empire.active_wars) == 1
        assert empire.active_wars[0].name == "First Contact War"

    def test_extract_claims(self):
        from stellaris_advisor.parser.extractor import extract_empire

        raw = {
            "name": "Empire",
            "claims_placed": {
                "0": {"system_id": 10, "influence_cost": 50},
            },
        }
        empire = extract_empire(raw, empire_id=0, version="4.3")
        assert len(empire.claims_placed) == 1
        assert empire.claims_placed[0].system_id == 10

    def test_extract_fleets(self):
        from stellaris_advisor.parser.extractor import extract_empire

        raw = {
            "name": "Empire",
            "fleets": {
                "0": {
                    "id": 1,
                    "name": "1st Fleet",
                    "fleet_power": 3000.0,
                    "ships": {"corvette": 20},
                    "naval_capacity_used": 20,
                    "location_system": 0,
                    "commander": {"id": 5, "name": "Admiral", "level": 3},
                },
            },
        }
        empire = extract_empire(raw, empire_id=0, version="4.3")
        assert len(empire.fleets) == 1
        assert empire.fleets[0].name == "1st Fleet"
        assert empire.fleets[0].commander is not None
        assert empire.fleets[0].commander.name == "Admiral"

    def test_extract_starbases(self):
        from stellaris_advisor.parser.extractor import extract_empire

        raw = {
            "name": "Empire",
            "starbases": {
                "0": {
                    "system_id": 0,
                    "level": "starbase_starport",
                    "modules": ["shipyard"],
                    "buildings": ["listening_post"],
                    "defense_fleet_power": 500.0,
                },
            },
        }
        empire = extract_empire(raw, empire_id=0, version="4.3")
        assert len(empire.starbases) == 1
        assert empire.starbases[0].level == "starbase_starport"

    def test_extract_pop_groups(self):
        from stellaris_advisor.parser.extractor import extract_empire

        raw = {
            "name": "Empire",
            "pop_groups": {
                "0": {"species": "human", "count": 30, "happiness": 0.7},
            },
        }
        empire = extract_empire(raw, empire_id=0, version="4.3")
        assert len(empire.pop_groups) == 1
        assert empire.pop_groups[0].species == "human"


@pytest.mark.unit
class TestExtractSystem:
    def test_extract_system_basic(self):
        from stellaris_advisor.parser.extractor import extract_system

        raw = {
            "name": "Sol",
            "star_class": "sc_g",
            "hyperlane": [{"to": 1}, {"to": 2}],
            "owner": 0,
        }
        system = extract_system(raw, system_id=0, version="4.3")
        assert system.id == 0
        assert system.name == "Sol"
        assert system.star_class == "sc_g"
        assert system.hyperlane_connections == (1, 2)
        assert system.owner == 0

    def test_extract_system_minimal(self):
        from stellaris_advisor.parser.extractor import extract_system

        raw = {"name": "Unknown", "star_class": "sc_m"}
        system = extract_system(raw, system_id=99, version="4.3")
        assert system.id == 99
        assert system.owner is None
        assert system.hyperlane_connections == ()
        assert system.has_megastructure is False
