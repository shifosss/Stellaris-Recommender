"""Tests for empire feature extraction — Tasks 2, 3, & 4.

Covers: _resource_features, _economic_efficiency_features,
        _expansion_features, _military_features,
        _tech_features, _planet_portfolio_features,
        _pop_features, _diplomatic_features,
        _archetype_features, _tradition_features,
        _leader_features, _faction_features,
        _galcom_features, _federation_features,
        _megastructure_features, extract_empire_features.
"""

from __future__ import annotations

from types import MappingProxyType

import pytest

from stellaris_advisor.features.empire import (
    _archetype_features,
    _diplomatic_features,
    _economic_efficiency_features,
    _expansion_features,
    _faction_features,
    _federation_features,
    _galcom_features,
    _leader_features,
    _megastructure_features,
    _military_features,
    _planet_portfolio_features,
    _pop_features,
    _resource_features,
    _tech_features,
    _tradition_features,
    extract_empire_features,
)
from stellaris_advisor.parser.models import (
    AgendaState,
    CouncilPosition,
    FactionState,
    FederationState,
    FleetState,
    LeaderState,
    MegastructureState,
)

from .conftest import make_empire, make_galactic, make_planet

# ---------------------------------------------------------------------------
# Ground-truth feature name sets
# ---------------------------------------------------------------------------

RESOURCE_NAMES = frozenset(
    {
        "resource_net_energy",
        "resource_net_minerals",
        "resource_net_food",
        "resource_net_consumer_goods",
        "resource_net_alloys",
        "resource_net_unity",
        "resource_net_influence",
        "resource_net_physics_research",
        "resource_net_society_research",
        "resource_net_engineering_research",
        "resource_months_until_depletion_energy",
        "resource_months_until_depletion_minerals",
        "resource_months_until_depletion_food",
        "resource_months_until_depletion_consumer_goods",
        "resource_months_until_depletion_alloys",
        "resource_stockpile_utilization_energy",
        "resource_stockpile_utilization_minerals",
        "resource_stockpile_utilization_food",
        "resource_stockpile_utilization_consumer_goods",
        "resource_stockpile_utilization_alloys",
    }
)

ECONOMIC_NAMES = frozenset(
    {
        "economic_alloys_per_pop",
        "economic_research_per_pop",
        "economic_energy_per_pop",
        "economic_minerals_per_pop",
        "economic_consumer_goods_per_pop",
        "economic_unity_per_pop",
        "economic_energy_per_generator_district",
        "economic_minerals_per_mining_district",
    }
)

EXPANSION_NAMES = frozenset(
    {
        "expansion_system_count",
        "expansion_planet_count",
        "expansion_avg_planet_size",
        "expansion_total_district_slots",
        "expansion_district_utilization",
        "expansion_total_building_slots",
        "expansion_building_utilization",
        "expansion_border_ratio",
    }
)

MILITARY_NAMES = frozenset(
    {
        "military_fleet_power",
        "military_fleet_power_relative",
        "military_naval_capacity_utilization",
        "military_naval_capacity_max",
        "military_fleet_count",
        "military_combat_ship_ratio",
        "military_support_ship_ratio",
        "military_army_strength",
    }
)


# ---------------------------------------------------------------------------
# _resource_features
# ---------------------------------------------------------------------------


class TestResourceFeatures:
    @pytest.mark.unit
    def test_returns_dict(self) -> None:
        result = _resource_features(make_empire())
        assert isinstance(result, dict)

    @pytest.mark.unit
    def test_correct_keys(self) -> None:
        result = _resource_features(make_empire())
        assert set(result.keys()) == RESOURCE_NAMES

    @pytest.mark.unit
    def test_all_values_float(self) -> None:
        result = _resource_features(make_empire())
        assert all(isinstance(v, float) for v in result.values())

    @pytest.mark.unit
    def test_net_income_positive(self) -> None:
        empire = make_empire(
            resource_income=MappingProxyType({"energy": 50.0}),
            resource_expense=MappingProxyType({"energy": 20.0}),
        )
        result = _resource_features(empire)
        assert result["resource_net_energy"] == pytest.approx(30.0)

    @pytest.mark.unit
    def test_net_income_negative(self) -> None:
        empire = make_empire(
            resource_income=MappingProxyType({"energy": 10.0}),
            resource_expense=MappingProxyType({"energy": 30.0}),
        )
        result = _resource_features(empire)
        assert result["resource_net_energy"] == pytest.approx(-20.0)

    @pytest.mark.unit
    def test_months_depletion_zero_net_is_large(self) -> None:
        empire = make_empire(
            resource_income=MappingProxyType({"energy": 20.0}),
            resource_expense=MappingProxyType({"energy": 20.0}),
        )
        result = _resource_features(empire)
        assert result["resource_months_until_depletion_energy"] >= 999.0

    @pytest.mark.unit
    def test_months_depletion_positive_net_is_large(self) -> None:
        empire = make_empire(
            resource_income=MappingProxyType({"energy": 50.0}),
            resource_expense=MappingProxyType({"energy": 20.0}),
        )
        result = _resource_features(empire)
        assert result["resource_months_until_depletion_energy"] >= 999.0

    @pytest.mark.unit
    def test_months_depletion_negative_net(self) -> None:
        # net = 10 - 30 = -20; stockpile = 100; depletion = 100/20 = 5 months
        empire = make_empire(
            resource_income=MappingProxyType({"energy": 10.0}),
            resource_expense=MappingProxyType({"energy": 30.0}),
            resource_stockpile=MappingProxyType({"energy": 100.0}),
        )
        result = _resource_features(empire)
        assert result["resource_months_until_depletion_energy"] == pytest.approx(5.0)

    @pytest.mark.unit
    def test_stockpile_utilization_correct_value(self) -> None:
        empire = make_empire(
            resource_stockpile=MappingProxyType({"energy": 500.0}),
            resource_storage_cap=MappingProxyType({"energy": 1000.0}),
        )
        result = _resource_features(empire)
        assert result["resource_stockpile_utilization_energy"] == pytest.approx(0.5)

    @pytest.mark.unit
    def test_stockpile_utilization_zero_cap(self) -> None:
        empire = make_empire(
            resource_stockpile=MappingProxyType({"energy": 100.0}),
            resource_storage_cap=MappingProxyType({}),
        )
        result = _resource_features(empire)
        assert result["resource_stockpile_utilization_energy"] == 0.0

    @pytest.mark.unit
    def test_missing_resource_defaults_zero_net(self) -> None:
        empire = make_empire(
            resource_income=MappingProxyType({}),
            resource_expense=MappingProxyType({}),
        )
        result = _resource_features(empire)
        assert result["resource_net_energy"] == 0.0
        assert result["resource_net_alloys"] == 0.0

    @pytest.mark.unit
    def test_stockpile_utilization_in_range(self) -> None:
        result = _resource_features(make_empire())
        for key in (
            "resource_stockpile_utilization_energy",
            "resource_stockpile_utilization_minerals",
        ):
            assert 0.0 <= result[key] <= 1.0


# ---------------------------------------------------------------------------
# _economic_efficiency_features
# ---------------------------------------------------------------------------


class TestEconomicEfficiencyFeatures:
    @pytest.mark.unit
    def test_returns_dict(self) -> None:
        result = _economic_efficiency_features(make_empire())
        assert isinstance(result, dict)

    @pytest.mark.unit
    def test_correct_keys(self) -> None:
        result = _economic_efficiency_features(make_empire())
        assert set(result.keys()) == ECONOMIC_NAMES

    @pytest.mark.unit
    def test_all_values_float(self) -> None:
        result = _economic_efficiency_features(make_empire())
        assert all(isinstance(v, float) for v in result.values())

    @pytest.mark.unit
    def test_alloys_per_pop(self) -> None:
        empire = make_empire(
            resource_income=MappingProxyType({"alloys": 30.0}),
            resource_expense=MappingProxyType({}),
            total_pops=10,
        )
        result = _economic_efficiency_features(empire)
        assert result["economic_alloys_per_pop"] == pytest.approx(3.0)

    @pytest.mark.unit
    def test_zero_pops_no_division_error(self) -> None:
        result = _economic_efficiency_features(make_empire(total_pops=0))
        assert all(isinstance(v, float) for v in result.values())

    @pytest.mark.unit
    def test_research_per_pop_sums_three_areas(self) -> None:
        # 10 + 10 + 10 = 30 total research; 10 pops → 3.0 per pop
        empire = make_empire(
            resource_income=MappingProxyType(
                {
                    "physics_research": 10.0,
                    "society_research": 10.0,
                    "engineering_research": 10.0,
                }
            ),
            resource_expense=MappingProxyType({}),
            total_pops=10,
        )
        result = _economic_efficiency_features(empire)
        assert result["economic_research_per_pop"] == pytest.approx(3.0)

    @pytest.mark.unit
    def test_energy_per_generator_district(self) -> None:
        # net energy = 80 - 40 = 40; 4 generator districts → 10.0 per district
        planet = make_planet(
            districts=MappingProxyType({"district_generator": 4}),
        )
        empire = make_empire(
            resource_income=MappingProxyType({"energy": 80.0}),
            resource_expense=MappingProxyType({"energy": 40.0}),
            total_pops=10,
            owned_planets=(planet,),
        )
        result = _economic_efficiency_features(empire)
        assert result["economic_energy_per_generator_district"] == pytest.approx(10.0)

    @pytest.mark.unit
    def test_minerals_per_mining_district(self) -> None:
        # net minerals = 60 - 20 = 40; 5 mining districts → 8.0 per district
        planet = make_planet(
            districts=MappingProxyType({"district_mining": 5}),
        )
        empire = make_empire(
            resource_income=MappingProxyType({"minerals": 60.0}),
            resource_expense=MappingProxyType({"minerals": 20.0}),
            total_pops=10,
            owned_planets=(planet,),
        )
        result = _economic_efficiency_features(empire)
        assert result["economic_minerals_per_mining_district"] == pytest.approx(8.0)

    @pytest.mark.unit
    def test_no_planets_no_division_error(self) -> None:
        result = _economic_efficiency_features(make_empire(owned_planets=()))
        assert isinstance(result["economic_energy_per_generator_district"], float)
        assert isinstance(result["economic_minerals_per_mining_district"], float)


# ---------------------------------------------------------------------------
# _expansion_features
# ---------------------------------------------------------------------------


class TestExpansionFeatures:
    @pytest.mark.unit
    def test_returns_dict(self) -> None:
        result = _expansion_features(make_empire(), make_galactic())
        assert isinstance(result, dict)

    @pytest.mark.unit
    def test_correct_keys(self) -> None:
        result = _expansion_features(make_empire(), make_galactic())
        assert set(result.keys()) == EXPANSION_NAMES

    @pytest.mark.unit
    def test_all_values_float(self) -> None:
        result = _expansion_features(make_empire(), make_galactic())
        assert all(isinstance(v, float) for v in result.values())

    @pytest.mark.unit
    def test_system_count(self) -> None:
        empire = make_empire(owned_systems=(1, 2, 3, 4, 5))
        result = _expansion_features(empire, make_galactic())
        assert result["expansion_system_count"] == pytest.approx(5.0)

    @pytest.mark.unit
    def test_planet_count(self) -> None:
        planets = (make_planet(id=1), make_planet(id=2))
        empire = make_empire(owned_planets=planets)
        result = _expansion_features(empire, make_galactic())
        assert result["expansion_planet_count"] == pytest.approx(2.0)

    @pytest.mark.unit
    def test_avg_planet_size(self) -> None:
        planets = (make_planet(id=1, size=20), make_planet(id=2, size=10))
        empire = make_empire(owned_planets=planets)
        result = _expansion_features(empire, make_galactic())
        assert result["expansion_avg_planet_size"] == pytest.approx(15.0)

    @pytest.mark.unit
    def test_avg_planet_size_no_planets(self) -> None:
        result = _expansion_features(make_empire(owned_planets=()), make_galactic())
        assert result["expansion_avg_planet_size"] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_district_utilization(self) -> None:
        planet = make_planet(total_district_cap=10, districts_used=5)
        empire = make_empire(owned_planets=(planet,))
        result = _expansion_features(empire, make_galactic())
        assert result["expansion_district_utilization"] == pytest.approx(0.5)

    @pytest.mark.unit
    def test_building_utilization(self) -> None:
        planet = make_planet(building_slots_total=12, building_slots_used=3)
        empire = make_empire(owned_planets=(planet,))
        result = _expansion_features(empire, make_galactic())
        assert result["expansion_building_utilization"] == pytest.approx(0.25)

    @pytest.mark.unit
    def test_border_ratio(self) -> None:
        # empire owns 2 of the 3 default galactic systems
        empire = make_empire(owned_systems=(1, 2))
        result = _expansion_features(empire, make_galactic())
        assert result["expansion_border_ratio"] == pytest.approx(2.0 / 3.0)

    @pytest.mark.unit
    def test_no_planets_no_division_error(self) -> None:
        result = _expansion_features(make_empire(owned_planets=()), make_galactic())
        assert result["expansion_district_utilization"] == pytest.approx(0.0)
        assert result["expansion_building_utilization"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _military_features
# ---------------------------------------------------------------------------


class TestMilitaryFeatures:
    @pytest.mark.unit
    def test_returns_dict(self) -> None:
        result = _military_features(make_empire(), make_galactic())
        assert isinstance(result, dict)

    @pytest.mark.unit
    def test_correct_keys(self) -> None:
        result = _military_features(make_empire(), make_galactic())
        assert set(result.keys()) == MILITARY_NAMES

    @pytest.mark.unit
    def test_all_values_float(self) -> None:
        result = _military_features(make_empire(), make_galactic())
        assert all(isinstance(v, float) for v in result.values())

    @pytest.mark.unit
    def test_fleet_power_absolute(self) -> None:
        empire = make_empire(fleet_power=5000.0)
        result = _military_features(empire, make_galactic())
        assert result["military_fleet_power"] == pytest.approx(5000.0)

    @pytest.mark.unit
    def test_fleet_power_relative_sole_empire(self) -> None:
        empire = make_empire(id=1, fleet_power=1000.0)
        galactic = make_galactic(
            empires=(empire,), player_empire_id=1, player_empire=empire
        )
        result = _military_features(empire, galactic)
        assert result["military_fleet_power_relative"] == pytest.approx(1.0)

    @pytest.mark.unit
    def test_fleet_power_relative_two_empires(self) -> None:
        e1 = make_empire(id=1, fleet_power=1000.0)
        e2 = make_empire(id=2, fleet_power=3000.0)
        galactic = make_galactic(empires=(e1, e2), player_empire_id=1, player_empire=e1)
        result = _military_features(e1, galactic)
        # 1000 / (1000 + 3000) = 0.25
        assert result["military_fleet_power_relative"] == pytest.approx(0.25)

    @pytest.mark.unit
    def test_naval_capacity_utilization(self) -> None:
        empire = make_empire(naval_capacity_used=30, naval_capacity_max=40)
        result = _military_features(empire, make_galactic())
        assert result["military_naval_capacity_utilization"] == pytest.approx(0.75)

    @pytest.mark.unit
    def test_zero_naval_capacity_no_division_error(self) -> None:
        empire = make_empire(naval_capacity_used=0, naval_capacity_max=0)
        result = _military_features(empire, make_galactic())
        assert isinstance(result["military_naval_capacity_utilization"], float)

    @pytest.mark.unit
    def test_fleet_count(self) -> None:
        fleet = FleetState(
            id=1,
            name="First Fleet",
            fleet_power=500.0,
            ships=MappingProxyType({"corvette": 5}),
            naval_capacity_used=5,
            location_system=1,
            commander=None,
        )
        empire = make_empire(fleets=(fleet,))
        result = _military_features(empire, make_galactic())
        assert result["military_fleet_count"] == pytest.approx(1.0)

    @pytest.mark.unit
    def test_combat_and_support_ship_ratio(self) -> None:
        fleet = FleetState(
            id=1,
            name="Mixed Fleet",
            fleet_power=1000.0,
            ships=MappingProxyType({"corvette": 8, "science_ship": 2}),
            naval_capacity_used=10,
            location_system=1,
            commander=None,
        )
        empire = make_empire(fleets=(fleet,))
        result = _military_features(empire, make_galactic())
        assert result["military_combat_ship_ratio"] == pytest.approx(0.8)
        assert result["military_support_ship_ratio"] == pytest.approx(0.2)

    @pytest.mark.unit
    def test_no_fleets_ship_ratios_are_zero(self) -> None:
        result = _military_features(make_empire(fleets=()), make_galactic())
        assert result["military_combat_ship_ratio"] == pytest.approx(0.0)
        assert result["military_support_ship_ratio"] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_army_strength(self) -> None:
        empire = make_empire(total_army_strength=250.0)
        result = _military_features(empire, make_galactic())
        assert result["military_army_strength"] == pytest.approx(250.0)


# ---------------------------------------------------------------------------
# _tech_features
# ---------------------------------------------------------------------------

TECH_NAMES = frozenset(
    {
        "tech_total_count",
        "tech_physics_count",
        "tech_society_count",
        "tech_engineering_count",
        "tech_tier1_count",
        "tech_tier2_count",
        "tech_tier3_count",
        "tech_physics_research_speed",
        "tech_society_research_speed",
        "tech_engineering_research_speed",
        "tech_frontier_size",
        "tech_repeatable_count",
    }
)


class TestTechFeatures:
    @pytest.mark.unit
    def test_returns_dict(self) -> None:
        assert isinstance(_tech_features(make_empire()), dict)

    @pytest.mark.unit
    def test_correct_keys(self) -> None:
        assert set(_tech_features(make_empire()).keys()) == TECH_NAMES

    @pytest.mark.unit
    def test_all_values_float(self) -> None:
        assert all(isinstance(v, float) for v in _tech_features(make_empire()).values())

    @pytest.mark.unit
    def test_total_count_matches_techs(self) -> None:
        empire = make_empire(researched_techs=frozenset({"tech_a", "tech_b", "tech_c"}))
        result = _tech_features(empire)
        assert result["tech_total_count"] == pytest.approx(3.0)

    @pytest.mark.unit
    def test_empty_techs_all_counts_zero(self) -> None:
        empire = make_empire(researched_techs=frozenset())
        result = _tech_features(empire)
        assert result["tech_total_count"] == pytest.approx(0.0)
        assert result["tech_physics_count"] == pytest.approx(0.0)
        assert result["tech_engineering_count"] == pytest.approx(0.0)
        assert result["tech_society_count"] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_physics_tech_classified_correctly(self) -> None:
        empire = make_empire(researched_techs=frozenset({"tech_power_plant_1"}))
        result = _tech_features(empire)
        assert result["tech_physics_count"] == pytest.approx(1.0)
        assert result["tech_engineering_count"] == pytest.approx(0.0)
        assert result["tech_society_count"] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_engineering_tech_classified_correctly(self) -> None:
        empire = make_empire(researched_techs=frozenset({"tech_mining_network_1"}))
        result = _tech_features(empire)
        assert result["tech_engineering_count"] == pytest.approx(1.0)
        assert result["tech_physics_count"] == pytest.approx(0.0)
        assert result["tech_society_count"] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_society_tech_classified_correctly(self) -> None:
        empire = make_empire(researched_techs=frozenset({"tech_gene_tailoring"}))
        result = _tech_features(empire)
        assert result["tech_society_count"] == pytest.approx(1.0)
        assert result["tech_physics_count"] == pytest.approx(0.0)
        assert result["tech_engineering_count"] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_per_area_counts_sum_to_total(self) -> None:
        empire = make_empire(
            researched_techs=frozenset(
                {"tech_power_plant_1", "tech_mining_network_1", "tech_gene_tailoring"}
            )
        )
        result = _tech_features(empire)
        area_sum = (
            result["tech_physics_count"]
            + result["tech_society_count"]
            + result["tech_engineering_count"]
        )
        assert area_sum == pytest.approx(result["tech_total_count"])

    @pytest.mark.unit
    def test_tier2_count_for_suffix_2_tech(self) -> None:
        empire = make_empire(researched_techs=frozenset({"tech_alloys_2"}))
        result = _tech_features(empire)
        assert result["tech_tier2_count"] == pytest.approx(1.0)
        assert result["tech_tier1_count"] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_tier1_count_for_no_suffix_tech(self) -> None:
        empire = make_empire(researched_techs=frozenset({"tech_gene_tailoring"}))
        result = _tech_features(empire)
        assert result["tech_tier1_count"] == pytest.approx(1.0)
        assert result["tech_tier2_count"] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_repeatable_count_is_stubbed_zero(self) -> None:
        # Stub until KG integration: suffix heuristic misclassifies regular techs
        empire = make_empire(
            researched_techs=frozenset(
                {"tech_alloys_1", "tech_alloys_2", "tech_alloys_3"}
            )
        )
        assert _tech_features(empire)["tech_repeatable_count"] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_research_speed_features(self) -> None:
        empire = make_empire(
            research_speed=MappingProxyType(
                {"physics": 1.2, "society": 0.9, "engineering": 1.1}
            )
        )
        result = _tech_features(empire)
        assert result["tech_physics_research_speed"] == pytest.approx(1.2)
        assert result["tech_society_research_speed"] == pytest.approx(0.9)
        assert result["tech_engineering_research_speed"] == pytest.approx(1.1)

    @pytest.mark.unit
    def test_frontier_size_is_stubbed_zero(self) -> None:
        # Stub until KG integration: always 3 without KG (not useful signal)
        assert _tech_features(make_empire())["tech_frontier_size"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _planet_portfolio_features
# ---------------------------------------------------------------------------

PLANET_PORTFOLIO_NAMES = frozenset(
    {
        "planet_climate_distribution_continental",
        "planet_climate_distribution_arid",
        "planet_climate_distribution_arctic",
        "planet_avg_habitability",
        "planet_total_district_slots_remaining",
        "planet_total_building_slots_remaining",
        "planet_specialized_world_count",
        "planet_ecumenopolis_count",
        "planet_ring_world_count",
        "planet_habitat_count",
    }
)


class TestPlanetPortfolioFeatures:
    @pytest.mark.unit
    def test_returns_dict(self) -> None:
        assert isinstance(_planet_portfolio_features(make_empire()), dict)

    @pytest.mark.unit
    def test_correct_keys(self) -> None:
        assert (
            set(_planet_portfolio_features(make_empire()).keys())
            == PLANET_PORTFOLIO_NAMES
        )

    @pytest.mark.unit
    def test_all_values_float(self) -> None:
        assert all(
            isinstance(v, float)
            for v in _planet_portfolio_features(make_empire()).values()
        )

    @pytest.mark.unit
    def test_no_planets_all_zeros(self) -> None:
        result = _planet_portfolio_features(make_empire(owned_planets=()))
        assert result["planet_climate_distribution_continental"] == pytest.approx(0.0)
        assert result["planet_climate_distribution_arid"] == pytest.approx(0.0)
        assert result["planet_climate_distribution_arctic"] == pytest.approx(0.0)
        assert result["planet_total_district_slots_remaining"] == pytest.approx(0.0)
        assert result["planet_ecumenopolis_count"] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_continental_planet_climate(self) -> None:
        planet = make_planet(planet_class="pc_continental")
        empire = make_empire(owned_planets=(planet,))
        result = _planet_portfolio_features(empire)
        assert result["planet_climate_distribution_continental"] == pytest.approx(1.0)
        assert result["planet_climate_distribution_arid"] == pytest.approx(0.0)
        assert result["planet_climate_distribution_arctic"] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_arid_planet_climate(self) -> None:
        planet = make_planet(planet_class="pc_desert")
        empire = make_empire(owned_planets=(planet,))
        result = _planet_portfolio_features(empire)
        assert result["planet_climate_distribution_arid"] == pytest.approx(1.0)
        assert result["planet_climate_distribution_continental"] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_arctic_planet_climate(self) -> None:
        planet = make_planet(planet_class="pc_arctic")
        empire = make_empire(owned_planets=(planet,))
        result = _planet_portfolio_features(empire)
        assert result["planet_climate_distribution_arctic"] == pytest.approx(1.0)

    @pytest.mark.unit
    def test_climate_distribution_is_ratio(self) -> None:
        p1 = make_planet(id=1, planet_class="pc_continental")
        p2 = make_planet(id=2, planet_class="pc_desert")
        empire = make_empire(owned_planets=(p1, p2))
        result = _planet_portfolio_features(empire)
        assert result["planet_climate_distribution_continental"] == pytest.approx(0.5)
        assert result["planet_climate_distribution_arid"] == pytest.approx(0.5)

    @pytest.mark.unit
    def test_district_slots_remaining(self) -> None:
        planet = make_planet(total_district_cap=10, districts_used=3)
        empire = make_empire(owned_planets=(planet,))
        result = _planet_portfolio_features(empire)
        assert result["planet_total_district_slots_remaining"] == pytest.approx(7.0)

    @pytest.mark.unit
    def test_building_slots_remaining(self) -> None:
        planet = make_planet(building_slots_total=12, building_slots_used=4)
        empire = make_empire(owned_planets=(planet,))
        result = _planet_portfolio_features(empire)
        assert result["planet_total_building_slots_remaining"] == pytest.approx(8.0)

    @pytest.mark.unit
    def test_ecumenopolis_count(self) -> None:
        planet = make_planet(planet_class="pc_city")
        empire = make_empire(owned_planets=(planet,))
        result = _planet_portfolio_features(empire)
        assert result["planet_ecumenopolis_count"] == pytest.approx(1.0)

    @pytest.mark.unit
    def test_ring_world_count(self) -> None:
        planet = make_planet(planet_class="pc_ringworld_habitable")
        empire = make_empire(owned_planets=(planet,))
        result = _planet_portfolio_features(empire)
        assert result["planet_ring_world_count"] == pytest.approx(1.0)

    @pytest.mark.unit
    def test_habitat_count(self) -> None:
        planet = make_planet(planet_class="pc_habitat")
        empire = make_empire(owned_planets=(planet,))
        result = _planet_portfolio_features(empire)
        assert result["planet_habitat_count"] == pytest.approx(1.0)

    @pytest.mark.unit
    def test_avg_habitability_in_range(self) -> None:
        result = _planet_portfolio_features(make_empire())
        assert 0.0 <= result["planet_avg_habitability"] <= 1.0


# ---------------------------------------------------------------------------
# _pop_features
# ---------------------------------------------------------------------------

POP_NAMES = frozenset(
    {
        "pop_total",
        "pop_avg_happiness",
        "pop_unemployment_rate",
        "pop_species_diversity",
        "pop_faction_spread",
        "pop_growth_rate",
    }
)


class TestPopFeatures:
    @pytest.mark.unit
    def test_returns_dict(self) -> None:
        assert isinstance(_pop_features(make_empire()), dict)

    @pytest.mark.unit
    def test_correct_keys(self) -> None:
        assert set(_pop_features(make_empire()).keys()) == POP_NAMES

    @pytest.mark.unit
    def test_all_values_float(self) -> None:
        assert all(isinstance(v, float) for v in _pop_features(make_empire()).values())

    @pytest.mark.unit
    def test_pop_total(self) -> None:
        empire = make_empire(total_pops=75)
        assert _pop_features(empire)["pop_total"] == pytest.approx(75.0)

    @pytest.mark.unit
    def test_avg_happiness_weighted(self) -> None:
        from stellaris_advisor.parser.models import PopGroupState

        groups = (
            PopGroupState(species="Human", count=80, happiness=0.8),
            PopGroupState(species="Alien", count=20, happiness=0.6),
        )
        empire = make_empire(pop_groups=groups, total_pops=100)
        result = _pop_features(empire)
        # (80*0.8 + 20*0.6) / 100 = (64 + 12) / 100 = 0.76
        assert result["pop_avg_happiness"] == pytest.approx(0.76)

    @pytest.mark.unit
    def test_avg_happiness_no_pops(self) -> None:
        empire = make_empire(pop_groups=(), total_pops=0)
        result = _pop_features(empire)
        assert isinstance(result["pop_avg_happiness"], float)

    @pytest.mark.unit
    def test_species_diversity_single(self) -> None:
        from stellaris_advisor.parser.models import PopGroupState

        groups = (PopGroupState(species="Human", count=100, happiness=0.7),)
        empire = make_empire(pop_groups=groups)
        assert _pop_features(empire)["pop_species_diversity"] == pytest.approx(1.0)

    @pytest.mark.unit
    def test_species_diversity_multiple(self) -> None:
        from stellaris_advisor.parser.models import PopGroupState

        groups = (
            PopGroupState(species="Human", count=60, happiness=0.7),
            PopGroupState(species="Alien", count=30, happiness=0.6),
            PopGroupState(species="Robot", count=10, happiness=0.9),
        )
        empire = make_empire(pop_groups=groups)
        assert _pop_features(empire)["pop_species_diversity"] == pytest.approx(3.0)

    @pytest.mark.unit
    def test_faction_spread_no_factions(self) -> None:
        empire = make_empire(factions=())
        assert _pop_features(empire)["pop_faction_spread"] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_faction_spread_dominated(self) -> None:
        # One faction with support 1.0 → spread = 0.0
        factions = (
            FactionState(
                ethic="ethic_egalitarian",
                approval=0.8,
                size=100,
                support=1.0,
                issues=(),
            ),
        )
        empire = make_empire(factions=factions)
        assert _pop_features(empire)["pop_faction_spread"] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_faction_spread_two_factions(self) -> None:
        # Dominant faction support=0.6 → spread = 1 - 0.6 = 0.4
        factions = (
            FactionState(
                ethic="ethic_egalitarian",
                approval=0.7,
                size=60,
                support=0.6,
                issues=(),
            ),
            FactionState(
                ethic="ethic_authoritarian",
                approval=0.5,
                size=40,
                support=0.4,
                issues=(),
            ),
        )
        empire = make_empire(factions=factions)
        assert _pop_features(empire)["pop_faction_spread"] == pytest.approx(0.4)

    @pytest.mark.unit
    def test_unemployment_rate_is_float(self) -> None:
        assert isinstance(_pop_features(make_empire())["pop_unemployment_rate"], float)

    @pytest.mark.unit
    def test_growth_rate_is_float(self) -> None:
        assert isinstance(_pop_features(make_empire())["pop_growth_rate"], float)


# ---------------------------------------------------------------------------
# _diplomatic_features
# ---------------------------------------------------------------------------

DIPLOMATIC_NAMES = frozenset(
    {
        "diplomatic_rival_count",
        "diplomatic_pact_count",
        "diplomatic_federation_member",
        "diplomatic_subject_count",
        "diplomatic_is_overlord",
        "diplomatic_war_count",
    }
)


class TestDiplomaticFeatures:
    @pytest.mark.unit
    def test_returns_dict(self) -> None:
        assert isinstance(_diplomatic_features(make_empire()), dict)

    @pytest.mark.unit
    def test_correct_keys(self) -> None:
        assert set(_diplomatic_features(make_empire()).keys()) == DIPLOMATIC_NAMES

    @pytest.mark.unit
    def test_all_values_float(self) -> None:
        assert all(
            isinstance(v, float) for v in _diplomatic_features(make_empire()).values()
        )

    @pytest.mark.unit
    def test_rival_count(self) -> None:
        empire = make_empire(rivals=(2, 3, 4))
        assert _diplomatic_features(empire)["diplomatic_rival_count"] == pytest.approx(
            3.0
        )

    @pytest.mark.unit
    def test_pact_count(self) -> None:
        empire = make_empire(defensive_pacts=(2, 3))
        assert _diplomatic_features(empire)["diplomatic_pact_count"] == pytest.approx(
            2.0
        )

    @pytest.mark.unit
    def test_federation_member_true(self) -> None:
        empire = make_empire(federation_id=1)
        assert _diplomatic_features(empire)[
            "diplomatic_federation_member"
        ] == pytest.approx(1.0)

    @pytest.mark.unit
    def test_federation_member_false(self) -> None:
        empire = make_empire(federation_id=None)
        assert _diplomatic_features(empire)[
            "diplomatic_federation_member"
        ] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_subject_count(self) -> None:
        empire = make_empire(subjects=(3, 4, 5))
        assert _diplomatic_features(empire)[
            "diplomatic_subject_count"
        ] == pytest.approx(3.0)

    @pytest.mark.unit
    def test_is_overlord_with_subjects(self) -> None:
        empire = make_empire(subjects=(3,))
        assert _diplomatic_features(empire)["diplomatic_is_overlord"] == pytest.approx(
            1.0
        )

    @pytest.mark.unit
    def test_is_not_overlord_without_subjects(self) -> None:
        empire = make_empire(subjects=())
        assert _diplomatic_features(empire)["diplomatic_is_overlord"] == pytest.approx(
            0.0
        )

    @pytest.mark.unit
    def test_war_count(self) -> None:
        from stellaris_advisor.parser.models import WarState

        wars = (
            WarState(
                war_id=1,
                name="War 1",
                war_goal="wg_domination",
                exhaustion=0.2,
                is_attacker=True,
            ),
        )
        empire = make_empire(active_wars=wars)
        assert _diplomatic_features(empire)["diplomatic_war_count"] == pytest.approx(
            1.0
        )

    @pytest.mark.unit
    def test_no_diplomacy_all_zeros(self) -> None:
        empire = make_empire(
            rivals=(),
            defensive_pacts=(),
            federation_id=None,
            subjects=(),
            active_wars=(),
        )
        result = _diplomatic_features(empire)
        assert result["diplomatic_rival_count"] == pytest.approx(0.0)
        assert result["diplomatic_federation_member"] == pytest.approx(0.0)
        assert result["diplomatic_is_overlord"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _archetype_features
# ---------------------------------------------------------------------------

ARCHETYPE_NAMES = frozenset(
    {
        "archetype_hive",
        "archetype_machine",
        "archetype_megacorp",
        "archetype_standard",
    }
)


class TestArchetypeFeatures:
    @pytest.mark.unit
    def test_returns_dict(self) -> None:
        assert isinstance(_archetype_features(make_empire()), dict)

    @pytest.mark.unit
    def test_correct_keys(self) -> None:
        assert set(_archetype_features(make_empire()).keys()) == ARCHETYPE_NAMES

    @pytest.mark.unit
    def test_all_values_float(self) -> None:
        assert all(
            isinstance(v, float) for v in _archetype_features(make_empire()).values()
        )

    @pytest.mark.unit
    def test_standard_authority(self) -> None:
        empire = make_empire(authority="auth_democratic")
        result = _archetype_features(empire)
        assert result["archetype_standard"] == pytest.approx(1.0)
        assert result["archetype_hive"] == pytest.approx(0.0)
        assert result["archetype_machine"] == pytest.approx(0.0)
        assert result["archetype_megacorp"] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_hive_authority(self) -> None:
        empire = make_empire(authority="auth_hive_mind")
        result = _archetype_features(empire)
        assert result["archetype_hive"] == pytest.approx(1.0)
        assert result["archetype_standard"] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_machine_authority(self) -> None:
        empire = make_empire(authority="auth_machine_intelligence")
        result = _archetype_features(empire)
        assert result["archetype_machine"] == pytest.approx(1.0)
        assert result["archetype_standard"] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_megacorp_authority(self) -> None:
        empire = make_empire(authority="auth_corporate")
        result = _archetype_features(empire)
        assert result["archetype_megacorp"] == pytest.approx(1.0)
        assert result["archetype_standard"] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_exactly_one_hot(self) -> None:
        for authority in (
            "auth_democratic",
            "auth_hive_mind",
            "auth_machine_intelligence",
            "auth_corporate",
        ):
            empire = make_empire(authority=authority)
            result = _archetype_features(empire)
            assert sum(result.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _tradition_features
# ---------------------------------------------------------------------------

TRADITION_NAMES = frozenset(
    {
        "tradition_total_unlocked",
        "tradition_trees_completed",
        "tradition_trees_opened",
        "tradition_ascension_perk_count",
        "tradition_ascension_slots_available",
        "tradition_unity_income",
        "tradition_ascension_path_type",
        "tradition_cost_efficiency",
    }
)


class TestTraditionFeatures:
    @pytest.mark.unit
    def test_returns_dict(self) -> None:
        assert isinstance(_tradition_features(make_empire()), dict)

    @pytest.mark.unit
    def test_correct_keys(self) -> None:
        assert set(_tradition_features(make_empire()).keys()) == TRADITION_NAMES

    @pytest.mark.unit
    def test_all_values_float(self) -> None:
        assert all(
            isinstance(v, float) for v in _tradition_features(make_empire()).values()
        )

    @pytest.mark.unit
    def test_total_unlocked(self) -> None:
        empire = make_empire(traditions_unlocked=("tr_a", "tr_b", "tr_c"))
        assert _tradition_features(empire)["tradition_total_unlocked"] == pytest.approx(
            3.0
        )

    @pytest.mark.unit
    def test_trees_completed(self) -> None:
        empire = make_empire(tradition_trees_completed=("tr_harmony", "tr_expansion"))
        assert _tradition_features(empire)[
            "tradition_trees_completed"
        ] == pytest.approx(2.0)

    @pytest.mark.unit
    def test_trees_opened(self) -> None:
        empire = make_empire(tradition_trees_opened=("tr_diplomacy",))
        assert _tradition_features(empire)["tradition_trees_opened"] == pytest.approx(
            1.0
        )

    @pytest.mark.unit
    def test_ascension_perk_count(self) -> None:
        empire = make_empire(
            ascension_perks=("ap_mind_over_matter", "ap_transcendence")
        )
        assert _tradition_features(empire)[
            "tradition_ascension_perk_count"
        ] == pytest.approx(2.0)

    @pytest.mark.unit
    def test_ascension_slots_available(self) -> None:
        empire = make_empire(ascension_perk_slots_available=2)
        assert _tradition_features(empire)[
            "tradition_ascension_slots_available"
        ] == pytest.approx(2.0)

    @pytest.mark.unit
    def test_unity_income(self) -> None:
        empire = make_empire(resource_income=MappingProxyType({"unity": 25.0}))
        assert _tradition_features(empire)["tradition_unity_income"] == pytest.approx(
            25.0
        )

    @pytest.mark.unit
    def test_ascension_path_none_is_zero(self) -> None:
        empire = make_empire(ascension_path=None)
        assert _tradition_features(empire)[
            "tradition_ascension_path_type"
        ] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_ascension_path_psionic(self) -> None:
        empire = make_empire(ascension_path="psionic")
        result = _tradition_features(empire)
        assert result["tradition_ascension_path_type"] != pytest.approx(0.0)

    @pytest.mark.unit
    def test_cost_efficiency_no_trees(self) -> None:
        # 0 traditions unlocked, 0 trees opened → cost_efficiency 0.0 (no division error)
        empire = make_empire(
            traditions_unlocked=(),
            tradition_trees_opened=(),
        )
        result = _tradition_features(empire)
        assert isinstance(result["tradition_cost_efficiency"], float)


# ---------------------------------------------------------------------------
# _leader_features
# ---------------------------------------------------------------------------

LEADER_NAMES = frozenset(
    {
        "leader_hor_level",
        "leader_council_avg_level",
        "leader_council_legitimacy",
        "leader_total_count",
        "leader_has_active_agenda",
    }
)


class TestLeaderFeatures:
    @pytest.mark.unit
    def test_returns_dict(self) -> None:
        assert isinstance(_leader_features(make_empire()), dict)

    @pytest.mark.unit
    def test_correct_keys(self) -> None:
        assert set(_leader_features(make_empire()).keys()) == LEADER_NAMES

    @pytest.mark.unit
    def test_all_values_float(self) -> None:
        assert all(
            isinstance(v, float) for v in _leader_features(make_empire()).values()
        )

    @pytest.mark.unit
    def test_no_leaders(self) -> None:
        empire = make_empire(leaders=(), council_positions=(), current_agenda=None)
        result = _leader_features(empire)
        assert result["leader_total_count"] == pytest.approx(0.0)
        assert result["leader_hor_level"] == pytest.approx(0.0)
        assert result["leader_council_avg_level"] == pytest.approx(0.0)
        assert result["leader_has_active_agenda"] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_hor_level_from_ruler_assignment(self) -> None:
        ruler = LeaderState(
            id=1,
            name="Chancellor",
            leader_class="official",
            level=5,
            traits=(),
            expertise=None,
            assignment="ruler",
        )
        other = LeaderState(
            id=2,
            name="Admiral",
            leader_class="admiral",
            level=3,
            traits=(),
            expertise=None,
            assignment="fleet",
        )
        empire = make_empire(leaders=(ruler, other))
        result = _leader_features(empire)
        assert result["leader_hor_level"] == pytest.approx(5.0)

    @pytest.mark.unit
    def test_hor_level_zero_when_no_ruler(self) -> None:
        admiral = LeaderState(
            id=1,
            name="Admiral",
            leader_class="admiral",
            level=4,
            traits=(),
            expertise=None,
            assignment="fleet",
        )
        empire = make_empire(leaders=(admiral,))
        result = _leader_features(empire)
        assert result["leader_hor_level"] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_council_avg_level(self) -> None:
        positions = (
            CouncilPosition(
                position_type="councillor", leader_id=1, leader_name="A", leader_level=4
            ),
            CouncilPosition(
                position_type="councillor", leader_id=2, leader_name="B", leader_level=6
            ),
        )
        empire = make_empire(council_positions=positions)
        result = _leader_features(empire)
        assert result["leader_council_avg_level"] == pytest.approx(5.0)

    @pytest.mark.unit
    def test_council_avg_level_empty(self) -> None:
        empire = make_empire(council_positions=())
        result = _leader_features(empire)
        assert result["leader_council_avg_level"] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_council_legitimacy(self) -> None:
        empire = make_empire(council_legitimacy=0.75)
        assert _leader_features(empire)["leader_council_legitimacy"] == pytest.approx(
            0.75
        )

    @pytest.mark.unit
    def test_total_count(self) -> None:
        leaders = (
            LeaderState(
                id=1,
                name="A",
                leader_class="admiral",
                level=3,
                traits=(),
                expertise=None,
                assignment=None,
            ),
            LeaderState(
                id=2,
                name="B",
                leader_class="scientist",
                level=2,
                traits=(),
                expertise=None,
                assignment=None,
            ),
        )
        empire = make_empire(leaders=leaders)
        assert _leader_features(empire)["leader_total_count"] == pytest.approx(2.0)

    @pytest.mark.unit
    def test_has_active_agenda_true(self) -> None:
        agenda = AgendaState(
            agenda_type="agenda_expand", progress=0.5, is_launched=True
        )
        empire = make_empire(current_agenda=agenda)
        assert _leader_features(empire)["leader_has_active_agenda"] == pytest.approx(
            1.0
        )

    @pytest.mark.unit
    def test_has_active_agenda_false(self) -> None:
        empire = make_empire(current_agenda=None)
        assert _leader_features(empire)["leader_has_active_agenda"] == pytest.approx(
            0.0
        )


# ---------------------------------------------------------------------------
# _faction_features
# ---------------------------------------------------------------------------

FACTION_NAMES = frozenset(
    {
        "faction_avg_approval",
        "faction_min_approval",
        "faction_max_approval",
        "faction_count",
        "faction_largest_size_ratio",
        "faction_influence_per_approval_point",
    }
)


class TestFactionFeatures:
    @pytest.mark.unit
    def test_returns_dict(self) -> None:
        assert isinstance(_faction_features(make_empire()), dict)

    @pytest.mark.unit
    def test_correct_keys(self) -> None:
        assert set(_faction_features(make_empire()).keys()) == FACTION_NAMES

    @pytest.mark.unit
    def test_all_values_float(self) -> None:
        assert all(
            isinstance(v, float) for v in _faction_features(make_empire()).values()
        )

    @pytest.mark.unit
    def test_no_factions_all_zeros(self) -> None:
        empire = make_empire(factions=())
        result = _faction_features(empire)
        assert result["faction_count"] == pytest.approx(0.0)
        assert result["faction_avg_approval"] == pytest.approx(0.0)
        assert result["faction_min_approval"] == pytest.approx(0.0)
        assert result["faction_max_approval"] == pytest.approx(0.0)
        assert result["faction_largest_size_ratio"] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_faction_count(self) -> None:
        factions = (
            FactionState(ethic="e_a", approval=0.7, size=50, support=0.5, issues=()),
            FactionState(ethic="e_b", approval=0.4, size=50, support=0.5, issues=()),
        )
        empire = make_empire(factions=factions)
        assert _faction_features(empire)["faction_count"] == pytest.approx(2.0)

    @pytest.mark.unit
    def test_avg_approval(self) -> None:
        factions = (
            FactionState(ethic="e_a", approval=0.8, size=60, support=0.6, issues=()),
            FactionState(ethic="e_b", approval=0.4, size=40, support=0.4, issues=()),
        )
        empire = make_empire(factions=factions)
        # support-weighted: (0.8*0.6 + 0.4*0.4) / (0.6+0.4) = (0.48+0.16) / 1.0 = 0.64
        assert _faction_features(empire)["faction_avg_approval"] == pytest.approx(0.64)

    @pytest.mark.unit
    def test_min_max_approval(self) -> None:
        factions = (
            FactionState(ethic="e_a", approval=0.9, size=50, support=0.5, issues=()),
            FactionState(ethic="e_b", approval=0.3, size=50, support=0.5, issues=()),
        )
        empire = make_empire(factions=factions)
        result = _faction_features(empire)
        assert result["faction_min_approval"] == pytest.approx(0.3)
        assert result["faction_max_approval"] == pytest.approx(0.9)

    @pytest.mark.unit
    def test_largest_size_ratio(self) -> None:
        factions = (
            FactionState(ethic="e_a", approval=0.7, size=70, support=0.7, issues=()),
            FactionState(ethic="e_b", approval=0.5, size=30, support=0.3, issues=()),
        )
        empire = make_empire(factions=factions, total_pops=100)
        result = _faction_features(empire)
        # largest faction = 70, total pops = 100 → 0.7
        assert result["faction_largest_size_ratio"] == pytest.approx(0.7)

    @pytest.mark.unit
    def test_influence_per_approval_point_no_division_error(self) -> None:
        factions = (
            FactionState(ethic="e_a", approval=0.0, size=50, support=0.5, issues=()),
        )
        empire = make_empire(
            factions=factions, resource_income=MappingProxyType({"influence": 5.0})
        )
        result = _faction_features(empire)
        assert isinstance(result["faction_influence_per_approval_point"], float)


# ---------------------------------------------------------------------------
# _galcom_features
# ---------------------------------------------------------------------------

GALCOM_NAMES = frozenset({"galcom_diplomatic_weight", "galcom_council_member"})


class TestGalcomFeatures:
    @pytest.mark.unit
    def test_returns_dict(self) -> None:
        assert isinstance(_galcom_features(make_empire()), dict)

    @pytest.mark.unit
    def test_correct_keys(self) -> None:
        assert set(_galcom_features(make_empire()).keys()) == GALCOM_NAMES

    @pytest.mark.unit
    def test_all_values_float(self) -> None:
        assert all(
            isinstance(v, float) for v in _galcom_features(make_empire()).values()
        )

    @pytest.mark.unit
    def test_diplomatic_weight(self) -> None:
        empire = make_empire(galcom_diplomatic_weight=250.0)
        assert _galcom_features(empire)["galcom_diplomatic_weight"] == pytest.approx(
            250.0
        )

    @pytest.mark.unit
    def test_council_member_true(self) -> None:
        empire = make_empire(galcom_council_member=True)
        assert _galcom_features(empire)["galcom_council_member"] == pytest.approx(1.0)

    @pytest.mark.unit
    def test_council_member_false(self) -> None:
        empire = make_empire(galcom_council_member=False)
        assert _galcom_features(empire)["galcom_council_member"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _federation_features
# ---------------------------------------------------------------------------

FEDERATION_NAMES = frozenset({"federation_level", "federation_fleet_power"})


class TestFederationFeatures:
    @pytest.mark.unit
    def test_returns_dict(self) -> None:
        assert isinstance(_federation_features(make_empire()), dict)

    @pytest.mark.unit
    def test_correct_keys(self) -> None:
        assert set(_federation_features(make_empire()).keys()) == FEDERATION_NAMES

    @pytest.mark.unit
    def test_all_values_float(self) -> None:
        assert all(
            isinstance(v, float) for v in _federation_features(make_empire()).values()
        )

    @pytest.mark.unit
    def test_no_federation_zeros(self) -> None:
        empire = make_empire(federation=None)
        result = _federation_features(empire)
        assert result["federation_level"] == pytest.approx(0.0)
        assert result["federation_fleet_power"] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_with_federation(self) -> None:
        fed = FederationState(
            federation_type="fed_research",
            level=3,
            cohesion=0.8,
            experience=250.0,
            fleet_power=5000.0,
        )
        empire = make_empire(federation=fed)
        result = _federation_features(empire)
        assert result["federation_level"] == pytest.approx(3.0)
        assert result["federation_fleet_power"] == pytest.approx(5000.0)


# ---------------------------------------------------------------------------
# _megastructure_features
# ---------------------------------------------------------------------------

MEGASTRUCTURE_NAMES = frozenset(
    {
        "megastructure_completed_count",
        "megastructure_under_construction_count",
        "megastructure_has_dyson_sphere",
        "megastructure_has_matter_decompressor",
    }
)


class TestMegastructureFeatures:
    @pytest.mark.unit
    def test_returns_dict(self) -> None:
        assert isinstance(_megastructure_features(make_empire()), dict)

    @pytest.mark.unit
    def test_correct_keys(self) -> None:
        assert set(_megastructure_features(make_empire()).keys()) == MEGASTRUCTURE_NAMES

    @pytest.mark.unit
    def test_all_values_float(self) -> None:
        assert all(
            isinstance(v, float)
            for v in _megastructure_features(make_empire()).values()
        )

    @pytest.mark.unit
    def test_no_megastructures(self) -> None:
        empire = make_empire(megastructures=(), megastructures_under_construction=0)
        result = _megastructure_features(empire)
        assert result["megastructure_completed_count"] == pytest.approx(0.0)
        assert result["megastructure_under_construction_count"] == pytest.approx(0.0)
        assert result["megastructure_has_dyson_sphere"] == pytest.approx(0.0)
        assert result["megastructure_has_matter_decompressor"] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_completed_count(self) -> None:
        megas = (
            MegastructureState(megastructure_type="gateway", stage=4, system_id=1),
            MegastructureState(megastructure_type="gateway", stage=4, system_id=2),
        )
        empire = make_empire(megastructures=megas)
        assert _megastructure_features(empire)[
            "megastructure_completed_count"
        ] == pytest.approx(2.0)

    @pytest.mark.unit
    def test_under_construction_count(self) -> None:
        empire = make_empire(megastructures_under_construction=3)
        assert _megastructure_features(empire)[
            "megastructure_under_construction_count"
        ] == pytest.approx(3.0)

    @pytest.mark.unit
    def test_has_dyson_sphere(self) -> None:
        megas = (
            MegastructureState(megastructure_type="dyson_sphere", stage=5, system_id=1),
        )
        empire = make_empire(megastructures=megas)
        assert _megastructure_features(empire)[
            "megastructure_has_dyson_sphere"
        ] == pytest.approx(1.0)

    @pytest.mark.unit
    def test_has_matter_decompressor(self) -> None:
        megas = (
            MegastructureState(
                megastructure_type="matter_decompressor", stage=5, system_id=1
            ),
        )
        empire = make_empire(megastructures=megas)
        assert _megastructure_features(empire)[
            "megastructure_has_matter_decompressor"
        ] == pytest.approx(1.0)

    @pytest.mark.unit
    def test_no_dyson_when_other_mega(self) -> None:
        megas = (
            MegastructureState(megastructure_type="gateway", stage=4, system_id=1),
        )
        empire = make_empire(megastructures=megas)
        assert _megastructure_features(empire)[
            "megastructure_has_dyson_sphere"
        ] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# extract_empire_features (top-level integration)
# ---------------------------------------------------------------------------


class TestExtractEmpireFeatures:
    @pytest.mark.unit
    def test_returns_dict(self) -> None:
        assert isinstance(extract_empire_features(make_empire(), make_galactic()), dict)

    @pytest.mark.unit
    def test_all_values_float(self) -> None:
        result = extract_empire_features(make_empire(), make_galactic())
        assert all(isinstance(v, float) for v in result.values())

    @pytest.mark.unit
    def test_key_count_matches_registry(self) -> None:
        from stellaris_advisor.features.registry import EMPIRE_FEATURE_NAMES

        result = extract_empire_features(make_empire(), make_galactic())
        assert set(result.keys()) == set(EMPIRE_FEATURE_NAMES)

    @pytest.mark.unit
    def test_no_missing_keys(self) -> None:
        from stellaris_advisor.features.registry import EMPIRE_FEATURE_NAMES

        result = extract_empire_features(make_empire(), make_galactic())
        missing = set(EMPIRE_FEATURE_NAMES) - set(result.keys())
        assert not missing, f"Missing keys: {missing}"

    @pytest.mark.unit
    def test_no_extra_keys(self) -> None:
        from stellaris_advisor.features.registry import EMPIRE_FEATURE_NAMES

        result = extract_empire_features(make_empire(), make_galactic())
        extra = set(result.keys()) - set(EMPIRE_FEATURE_NAMES)
        assert not extra, f"Extra keys: {extra}"
