"""Feature name registries for all three extractors.

All names are snake_case with a category prefix.
Order is stable and defines tensor column order for RL training.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Empire feature name segments (grouped by category)
# ---------------------------------------------------------------------------

_RESOURCE: tuple[str, ...] = (
    # Net income for 10 core resources
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
    # Months until depletion for finite stockpiles
    "resource_months_until_depletion_energy",
    "resource_months_until_depletion_minerals",
    "resource_months_until_depletion_food",
    "resource_months_until_depletion_consumer_goods",
    "resource_months_until_depletion_alloys",
    # Stockpile utilisation (current / cap) for finite resources
    "resource_stockpile_utilization_energy",
    "resource_stockpile_utilization_minerals",
    "resource_stockpile_utilization_food",
    "resource_stockpile_utilization_consumer_goods",
    "resource_stockpile_utilization_alloys",
)  # 20

_ECONOMIC: tuple[str, ...] = (
    "economic_alloys_per_pop",
    "economic_research_per_pop",
    "economic_energy_per_pop",
    "economic_minerals_per_pop",
    "economic_consumer_goods_per_pop",
    "economic_unity_per_pop",
    "economic_energy_per_generator_district",
    "economic_minerals_per_mining_district",
)  # 8

_EXPANSION: tuple[str, ...] = (
    "expansion_system_count",
    "expansion_planet_count",
    "expansion_avg_planet_size",
    "expansion_total_district_slots",
    "expansion_district_utilization",
    "expansion_total_building_slots",
    "expansion_building_utilization",
    "expansion_border_ratio",
)  # 8

_MILITARY: tuple[str, ...] = (
    "military_fleet_power",
    "military_fleet_power_relative",
    "military_naval_capacity_utilization",
    "military_naval_capacity_max",
    "military_fleet_count",
    "military_combat_ship_ratio",
    "military_support_ship_ratio",
    "military_army_strength",
)  # 8

_TECH: tuple[str, ...] = (
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
)  # 12

_PLANET: tuple[str, ...] = (
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
)  # 10

_POP: tuple[str, ...] = (
    "pop_total",
    "pop_avg_happiness",
    "pop_unemployment_rate",
    "pop_species_diversity",
    "pop_faction_spread",
    "pop_growth_rate",
)  # 6

_DIPLOMATIC: tuple[str, ...] = (
    "diplomatic_rival_count",
    "diplomatic_pact_count",
    "diplomatic_federation_member",
    "diplomatic_subject_count",
    "diplomatic_is_overlord",
    "diplomatic_war_count",
)  # 6

_ARCHETYPE: tuple[str, ...] = (
    "archetype_hive",
    "archetype_machine",
    "archetype_megacorp",
    "archetype_standard",
)  # 4

_TRADITION: tuple[str, ...] = (
    "tradition_total_unlocked",
    "tradition_trees_completed",
    "tradition_trees_opened",
    "tradition_ascension_perk_count",
    "tradition_ascension_slots_available",
    "tradition_unity_income",
    "tradition_ascension_path_type",
    "tradition_cost_efficiency",
)  # 8

_LEADER: tuple[str, ...] = (
    "leader_hor_level",
    "leader_council_avg_level",
    "leader_council_legitimacy",
    "leader_total_count",
    "leader_has_active_agenda",
)  # 5

_FACTION: tuple[str, ...] = (
    "faction_avg_approval",
    "faction_min_approval",
    "faction_max_approval",
    "faction_count",
    "faction_largest_size_ratio",
    "faction_influence_per_approval_point",
)  # 6

_GALCOM: tuple[str, ...] = (
    "galcom_diplomatic_weight",
    "galcom_council_member",
)  # 2

_FEDERATION: tuple[str, ...] = (
    "federation_level",
    "federation_fleet_power",
)  # 2

_MEGASTRUCTURE: tuple[str, ...] = (
    "megastructure_completed_count",
    "megastructure_under_construction_count",
    "megastructure_has_dyson_sphere",
    "megastructure_has_matter_decompressor",
)  # 4

# ---------------------------------------------------------------------------
# Public registries
# ---------------------------------------------------------------------------

EMPIRE_FEATURE_NAMES: tuple[str, ...] = (
    _RESOURCE
    + _ECONOMIC
    + _EXPANSION
    + _MILITARY
    + _TECH
    + _PLANET
    + _POP
    + _DIPLOMATIC
    + _ARCHETYPE
    + _TRADITION
    + _LEADER
    + _FACTION
    + _GALCOM
    + _FEDERATION
    + _MEGASTRUCTURE
)
# 20+8+8+8+12+10+6+6+4+8+5+6+2+2+4 = 109

SYSTEM_FEATURE_NAMES: tuple[str, ...] = (
    "system_best_habitability",
    "system_best_planet_size",
    "system_district_slots_city",
    "system_district_slots_mining",
    "system_district_slots_farming",
    "system_district_slots_generator",
    "system_resource_deposit_value",
    "system_hyperlane_degree",
    "system_chokepoint_score",
    "system_distance_to_capital",
    "system_distance_to_hostile",
    "system_climate_match_score",
)  # 12

TEMPORAL_FEATURE_NAMES: tuple[str, ...] = (
    "temporal_delta_energy",
    "temporal_delta_minerals",
    "temporal_delta_alloys",
    "temporal_delta_research",
    "temporal_expansion_velocity",
    "temporal_tech_velocity",
    "temporal_delta_fleet_power",
    "temporal_stall_flag",
)  # 8

ALL_FEATURE_NAMES: tuple[str, ...] = (
    EMPIRE_FEATURE_NAMES + SYSTEM_FEATURE_NAMES + TEMPORAL_FEATURE_NAMES
)
