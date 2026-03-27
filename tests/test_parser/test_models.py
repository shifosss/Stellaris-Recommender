"""Tests for parser data models (frozen dataclasses)."""

import pytest
from types import MappingProxyType


@pytest.mark.unit
class TestEmpireState:
    def test_empire_state_is_frozen(self):

        empire = _make_test_empire(name="Test Empire")
        assert empire.name == "Test Empire"
        with pytest.raises(AttributeError):
            empire.name = "Modified"

    def test_empire_techs_are_frozenset(self):

        empire = _make_test_empire(researched_techs=frozenset({"tech_a", "tech_b"}))
        assert isinstance(empire.researched_techs, frozenset)
        assert "tech_a" in empire.researched_techs

    def test_empire_resources_are_immutable(self):

        empire = _make_test_empire(resource_income=MappingProxyType({"energy": 100.0}))
        with pytest.raises(TypeError):
            empire.resource_income["energy"] = 200.0


@pytest.mark.unit
class TestPlanetState:
    def test_planet_state_is_frozen(self):
        from stellaris_advisor.parser.models import PlanetState

        planet = PlanetState(
            id=0,
            name="Earth",
            planet_class="pc_continental",
            size=25,
            districts=MappingProxyType({"district_city": 3}),
            district_slots=MappingProxyType({"district_city": 10}),
            total_district_cap=16,
            districts_used=5,
            buildings=("building_capital_2", "building_alloy_foundry"),
            building_slots_total=8,
            building_slots_used=2,
            district_specializations=MappingProxyType({}),
            pops=30,
            pop_groups=(),
            stability=55.0,
            crime=5.0,
            amenities=10.0,
            amenities_usage=8.0,
            housing=40.0,
            housing_usage=30.0,
            designation="col_default",
            planetary_features=(),
            is_capital=True,
        )
        assert planet.name == "Earth"
        assert planet.is_capital is True
        with pytest.raises(AttributeError):
            planet.name = "Mars"


@pytest.mark.unit
class TestSystemState:
    def test_system_state_is_frozen(self):
        from stellaris_advisor.parser.models import SystemState

        system = SystemState(
            id=0,
            name="Sol",
            star_class="sc_g",
            hyperlane_connections=(1, 2, 3),
            owner=0,
            claimed_by=(),
            starbase=None,
            planets=(),
            resources=MappingProxyType({}),
            has_megastructure=False,
            megastructure_type=None,
        )
        assert system.name == "Sol"


@pytest.mark.unit
class TestGalacticState:
    def test_galactic_state_is_frozen(self):
        from stellaris_advisor.parser.models import GalacticState

        empire = _make_test_empire()
        state = GalacticState(
            year=2250,
            month=1,
            game_version="4.3",
            galaxy_size=400,
            difficulty="grand_admiral",
            crisis_active=False,
            crisis_type=None,
            war_in_heaven=False,
            galactic_market_founded=False,
            empires=(),
            player_empire_id=0,
            systems=(),
            player_empire=empire,
        )
        assert state.year == 2250
        assert state.game_version == "4.3"


def _make_test_empire(**overrides):
    """Factory for EmpireState with sensible defaults."""
    from stellaris_advisor.parser.models import EmpireState

    defaults = dict(
        id=0,
        name="Test",
        authority="auth_democratic",
        ethics=(),
        civics=(),
        origin="",
        government_type="",
        resource_income=MappingProxyType({}),
        resource_expense=MappingProxyType({}),
        resource_stockpile=MappingProxyType({}),
        resource_storage_cap=MappingProxyType({}),
        researched_techs=frozenset(),
        current_research=MappingProxyType({}),
        research_progress=MappingProxyType({}),
        research_speed=MappingProxyType({}),
        stored_research=MappingProxyType({}),
        fleet_power=0.0,
        naval_capacity_used=0,
        naval_capacity_max=0,
        fleets=(),
        owned_systems=(),
        owned_planets=(),
        starbases=(),
        total_pops=0,
        pop_groups=(),
        rivals=(),
        defensive_pacts=(),
        federation_id=None,
        federation_power=None,
        subjects=(),
        overlord=None,
        traditions_unlocked=(),
        tradition_trees_opened=(),
        tradition_trees_completed=(),
        ascension_perks=(),
        ascension_perk_slots_available=0,
        ascension_perk_slots_total=0,
        ascension_path=None,
        ascension_situation_progress=None,
        leaders=(),
        council_positions=(),
        current_agenda=None,
        council_legitimacy=0.0,
        factions=(),
        policies=MappingProxyType({}),
        galcom_member=False,
        galcom_diplomatic_weight=0.0,
        galcom_council_member=False,
        galcom_custodian=False,
        galcom_emperor=False,
        active_resolutions=(),
        federation=None,
        megastructures=(),
        megastructures_under_construction=0,
        active_wars=(),
        claims_placed=(),
        total_army_strength=0.0,
        victory_score=0.0,
        victory_rank=0,
    )
    defaults.update(overrides)
    return EmpireState(**defaults)
