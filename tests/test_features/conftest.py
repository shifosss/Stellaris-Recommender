"""Test helper factory functions for features tests.

Each factory creates a minimal valid instance with sensible defaults.
Pass keyword overrides to customize specific fields.

Example:
    empire = make_empire(fleet_power=5000.0, rivals=(2, 3))
    planet = make_planet(pops=30, stability=90.0, is_capital=False)
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

import pytest

from stellaris_advisor.parser.models import (
    EmpireState,
    GalacticState,
    PlanetState,
    PlanetSummary,
    PopGroupState,
    SystemState,
)


# ---------------------------------------------------------------------------
# Resource defaults shared across factories
# ---------------------------------------------------------------------------

_DEFAULT_INCOME: MappingProxyType[str, float] = MappingProxyType(
    {
        "energy": 50.0,
        "minerals": 40.0,
        "food": 30.0,
        "consumer_goods": 20.0,
        "alloys": 15.0,
        "unity": 10.0,
        "influence": 5.0,
        "physics_research": 30.0,
        "society_research": 25.0,
        "engineering_research": 25.0,
    }
)

_DEFAULT_EXPENSE: MappingProxyType[str, float] = MappingProxyType(
    {
        "energy": 20.0,
        "minerals": 10.0,
        "food": 15.0,
        "consumer_goods": 10.0,
        "alloys": 5.0,
        "unity": 5.0,
        "influence": 2.0,
        "physics_research": 0.0,
        "society_research": 0.0,
        "engineering_research": 0.0,
    }
)

_DEFAULT_STOCKPILE: MappingProxyType[str, float] = MappingProxyType(
    {
        "energy": 500.0,
        "minerals": 400.0,
        "food": 300.0,
        "consumer_goods": 200.0,
        "alloys": 150.0,
        "unity": 100.0,
        "influence": 50.0,
    }
)

_DEFAULT_STORAGE_CAP: MappingProxyType[str, float] = MappingProxyType(
    {
        "energy": 1000.0,
        "minerals": 1000.0,
        "food": 1000.0,
        "consumer_goods": 1000.0,
        "alloys": 1000.0,
        "unity": 1000.0,
        "influence": 1000.0,
    }
)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def make_empire(**overrides: Any) -> EmpireState:
    """Create a minimal valid EmpireState with sensible defaults."""
    defaults: dict[str, Any] = dict(
        id=1,
        name="Test Empire",
        authority="auth_democratic",
        ethics=("ethic_egalitarian",),
        civics=("civic_parliamentary_system",),
        origin="origin_default",
        government_type="gov_democracy",
        resource_income=_DEFAULT_INCOME,
        resource_expense=_DEFAULT_EXPENSE,
        resource_stockpile=_DEFAULT_STOCKPILE,
        resource_storage_cap=_DEFAULT_STORAGE_CAP,
        researched_techs=frozenset({"tech_power_plant_1", "tech_mining_network_1"}),
        current_research=MappingProxyType(
            {
                "physics": "tech_power_plant_2",
                "society": "tech_gene_tailoring",
                "engineering": "tech_alloys_1",
            }
        ),
        research_progress=MappingProxyType(
            {"physics": 0.5, "society": 0.3, "engineering": 0.7}
        ),
        research_speed=MappingProxyType(
            {"physics": 1.0, "society": 1.0, "engineering": 1.0}
        ),
        stored_research=MappingProxyType(
            {"physics": 30.0, "society": 25.0, "engineering": 25.0}
        ),
        fleet_power=1000.0,
        naval_capacity_used=20,
        naval_capacity_max=40,
        fleets=(),
        owned_systems=(1, 2, 3),
        owned_planets=(),
        starbases=(),
        total_pops=50,
        pop_groups=(PopGroupState(species="Human", count=50, happiness=0.7),),
        rivals=(),
        defensive_pacts=(),
        federation_id=None,
        federation_power=None,
        subjects=(),
        overlord=None,
        traditions_unlocked=("tr_expansion_adopting",),
        tradition_trees_opened=("tr_expansion",),
        tradition_trees_completed=(),
        ascension_perks=(),
        ascension_perk_slots_available=0,
        ascension_perk_slots_total=3,
        ascension_path=None,
        ascension_situation_progress=None,
        leaders=(),
        council_positions=(),
        current_agenda=None,
        council_legitimacy=0.8,
        factions=(),
        policies=MappingProxyType({}),
        galcom_member=False,
        galcom_diplomatic_weight=100.0,
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
        victory_score=100.0,
        victory_rank=1,
    )
    defaults.update(overrides)
    return EmpireState(**defaults)


def make_planet(**overrides: Any) -> PlanetState:
    """Create a minimal valid PlanetState with sensible defaults."""
    defaults: dict[str, Any] = dict(
        id=1,
        name="Test Planet",
        planet_class="pc_continental",
        size=18,
        districts=MappingProxyType({"district_city": 2, "district_mining": 1}),
        district_slots=MappingProxyType(
            {"city": 5, "mining": 4, "farming": 3, "generator": 3}
        ),
        total_district_cap=15,
        districts_used=3,
        buildings=("building_capital",),
        building_slots_total=12,
        building_slots_used=1,
        district_specializations=MappingProxyType({}),
        pops=10,
        pop_groups=(PopGroupState(species="Human", count=10, happiness=0.7),),
        stability=70.0,
        crime=5.0,
        amenities=15.0,
        amenities_usage=10.0,
        housing=20.0,
        housing_usage=10.0,
        designation="col_default",
        planetary_features=("feature_tidal_locked",),
        is_capital=True,
    )
    defaults.update(overrides)
    return PlanetState(**defaults)


def make_system(**overrides: Any) -> SystemState:
    """Create a minimal valid SystemState with sensible defaults."""
    defaults: dict[str, Any] = dict(
        id=1,
        name="Sol",
        star_class="sc_g",
        hyperlane_connections=(2, 3),
        owner=1,
        claimed_by=(),
        starbase=None,
        planets=(
            PlanetSummary(
                id=1,
                name="Earth",
                planet_class="pc_continental",
                size=18,
                is_habitable=True,
                is_colonized=True,
            ),
        ),
        resources=MappingProxyType({"minerals": 10.0, "energy": 5.0}),
        has_megastructure=False,
        megastructure_type=None,
    )
    defaults.update(overrides)
    return SystemState(**defaults)


def make_galactic(**overrides: Any) -> GalacticState:
    """Create a minimal valid GalacticState with 2 empires and 3 systems."""
    empire1 = make_empire(id=1, name="Empire One")
    empire2 = make_empire(id=2, name="Empire Two")
    system1 = make_system(id=1, name="Sol", owner=1)
    system2 = make_system(id=2, name="Alpha Centauri", owner=1)
    system3 = make_system(id=3, name="Sirius", owner=2)
    defaults: dict[str, Any] = dict(
        year=2200,
        month=1,
        game_version="3.11.0",
        galaxy_size=400,
        difficulty="normal",
        crisis_active=False,
        crisis_type=None,
        war_in_heaven=False,
        galactic_market_founded=False,
        empires=(empire1, empire2),
        player_empire_id=1,
        systems=(system1, system2, system3),
        player_empire=empire1,
    )
    defaults.update(overrides)
    # Sync player_empire to match player_empire_id unless explicitly overridden
    if "player_empire" not in overrides:
        emp_id = defaults["player_empire_id"]
        empire_map = {e.id: e for e in defaults["empires"]}
        if emp_id in empire_map:
            defaults["player_empire"] = empire_map[emp_id]
    return GalacticState(**defaults)


# ---------------------------------------------------------------------------
# pytest fixtures wrapping the factory functions
# ---------------------------------------------------------------------------


@pytest.fixture
def empire() -> EmpireState:
    return make_empire()


@pytest.fixture
def planet() -> PlanetState:
    return make_planet()


@pytest.fixture
def system() -> SystemState:
    return make_system()


@pytest.fixture
def galactic() -> GalacticState:
    return make_galactic()
