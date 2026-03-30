# tests/test_planner/conftest.py
"""Shared fixtures for planner tests."""
from __future__ import annotations

from types import MappingProxyType

import pytest

from stellaris_advisor.parser.models import (
    EmpireState,
    FactionState,
    GalacticState,
    LeaderState,
    PlanetState,
    SystemState,
)


def _proxy(d: dict) -> MappingProxyType:  # type: ignore[type-arg]
    return MappingProxyType(d)


@pytest.fixture()
def planner_empire() -> EmpireState:
    """Mid-game empire at year ~2275 with reasonable state for planner testing."""
    planet = PlanetState(
        id=0,
        name="Earth",
        planet_class="continental",
        size=20,
        districts=_proxy({"city": 3, "mining": 2, "generator": 2, "farming": 1}),
        district_slots=_proxy({"city": 6, "mining": 4, "generator": 4, "farming": 3}),
        total_district_cap=20,
        districts_used=8,
        buildings=("building_capital_2", "building_research_lab_1"),
        building_slots_total=8,
        building_slots_used=2,
        district_specializations=_proxy({}),
        pops=45,
        pop_groups=(),
        stability=55.0,
        crime=5.0,
        amenities=12.0,
        amenities_usage=10.0,
        housing=30.0,
        housing_usage=25.0,
        designation="capital",
        planetary_features=(),
        is_capital=True,
    )
    leader = LeaderState(
        id=0,
        name="Dr. Smith",
        leader_class="scientist",
        level=4,
        traits=("trait_expertise_voidcraft",),
        expertise="voidcraft",
        assignment="head_of_research",
    )
    faction = FactionState(
        ethic="materialist",
        approval=65.0,
        support=0.3,
        size=40,
        issues=(),
    )
    return EmpireState(
        id=0,
        name="United Nations of Earth",
        authority="democratic",
        ethics=("egalitarian", "materialist"),
        civics=("beacon_of_liberty",),
        origin="prosperous_unification",
        government_type="democracy",
        resource_income=_proxy(
            {
                "energy": 80.0,
                "minerals": 60.0,
                "alloys": 25.0,
                "food": 40.0,
                "consumer_goods": 15.0,
                "unity": 20.0,
                "influence": 3.0,
                "physics_research": 100.0,
                "society_research": 90.0,
                "engineering_research": 95.0,
            }
        ),
        resource_expense=_proxy(
            {
                "energy": 65.0,
                "minerals": 40.0,
                "alloys": 10.0,
                "food": 35.0,
                "consumer_goods": 12.0,
            }
        ),
        resource_stockpile=_proxy(
            {"energy": 5000.0, "minerals": 3000.0, "alloys": 1500.0}
        ),
        resource_storage_cap=_proxy(
            {"energy": 10000.0, "minerals": 10000.0, "alloys": 10000.0}
        ),
        researched_techs=frozenset(
            {
                "tech_corvettes",
                "tech_destroyers",
                "tech_cruisers",
                "tech_fusion_power",
                "tech_powered_exoskeletons",
                "tech_alloys_1",
                "tech_mining_1",
            }
        ),
        current_research=_proxy(
            {
                "physics": "tech_shields_2",
                "society": "tech_gene_clinics",
                "engineering": "tech_battleships",
            }
        ),
        research_progress=_proxy(
            {"physics": 40.0, "society": 25.0, "engineering": 60.0}
        ),
        research_speed=_proxy({"physics": 100.0, "society": 90.0, "engineering": 95.0}),
        stored_research=_proxy({}),
        fleet_power=8000.0,
        naval_capacity_used=35,
        naval_capacity_max=80,
        fleets=(),
        owned_systems=tuple(range(15)),
        owned_planets=(planet,),
        starbases=(),
        total_pops=120,
        pop_groups=(),
        rivals=(3,),
        defensive_pacts=(1,),
        federation_id=None,
        federation_power=None,
        subjects=(),
        overlord=None,
        traditions_unlocked=("tr_discovery_adopt", "tr_discovery_science_division"),
        tradition_trees_opened=("discovery",),
        tradition_trees_completed=(),
        ascension_perks=(),
        ascension_perk_slots_available=0,
        ascension_perk_slots_total=2,
        ascension_path=None,
        ascension_situation_progress=None,
        leaders=(leader,),
        council_positions=(),
        current_agenda=None,
        council_legitimacy=60.0,
        factions=(faction,),
        policies=_proxy(
            {
                "economic_policy": "civilian_economy",
                "war_philosophy": "liberation_wars",
            }
        ),
        galcom_member=True,
        galcom_diplomatic_weight=200.0,
        galcom_council_member=False,
        galcom_custodian=False,
        galcom_emperor=False,
        active_resolutions=(),
        federation=None,
        megastructures=(),
        megastructures_under_construction=0,
        active_wars=(),
        claims_placed=(),
        total_army_strength=500.0,
        victory_score=1500.0,
        victory_rank=3,
    )


@pytest.fixture()
def planner_galactic(planner_empire: EmpireState) -> GalacticState:
    """Galaxy with 3 systems, 1 owned, 2 unowned."""
    from stellaris_advisor.parser.models import PlanetSummary

    systems = (
        SystemState(
            id=0,
            name="Sol",
            star_class="G",
            hyperlane_connections=(1, 2),
            owner=0,
            claimed_by=(),
            starbase=None,
            planets=(PlanetSummary(0, "Earth", "continental", 20, True, True),),
            resources=_proxy({}),
            has_megastructure=False,
            megastructure_type=None,
        ),
        SystemState(
            id=1,
            name="Alpha Centauri",
            star_class="K",
            hyperlane_connections=(0, 2),
            owner=None,
            claimed_by=(),
            starbase=None,
            planets=(PlanetSummary(1, "AC-1", "ocean", 18, True, False),),
            resources=_proxy({"volatile_motes": 2.0}),
            has_megastructure=False,
            megastructure_type=None,
        ),
        SystemState(
            id=2,
            name="Hostile Border",
            star_class="M",
            hyperlane_connections=(0, 1),
            owner=3,
            claimed_by=(),
            starbase=None,
            planets=(),
            resources=_proxy({}),
            has_megastructure=False,
            megastructure_type=None,
        ),
    )
    hostile_empire = EmpireState(
        id=3,
        name="Hostile Empire",
        authority="imperial",
        ethics=("militarist",),
        civics=(),
        origin="",
        government_type="despotism",
        resource_income=_proxy({}),
        resource_expense=_proxy({}),
        resource_stockpile=_proxy({}),
        resource_storage_cap=_proxy({}),
        researched_techs=frozenset(),
        current_research=_proxy({}),
        research_progress=_proxy({}),
        research_speed=_proxy({}),
        stored_research=_proxy({}),
        fleet_power=15000.0,
        naval_capacity_used=60,
        naval_capacity_max=100,
        fleets=(),
        owned_systems=(2,),
        owned_planets=(),
        starbases=(),
        total_pops=80,
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
        policies=_proxy({}),
        galcom_member=False,
        galcom_diplomatic_weight=0,
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
        victory_score=2000.0,
        victory_rank=1,
    )
    return GalacticState(
        year=2275,
        month=6,
        game_version="4.3",
        galaxy_size=600,
        difficulty="captain",
        crisis_active=False,
        crisis_type=None,
        war_in_heaven=False,
        galactic_market_founded=True,
        empires=(planner_empire, hostile_empire),
        player_empire_id=0,
        systems=systems,
        player_empire=planner_empire,
    )
