"""Integration tests for GameKnowledgeGraph queries and full-stack coverage.

Covers:
  End-to-end load → query pipelines using synthetic fixtures
  All 11 validation checks on known-clean synthetic data
  Query delegation method correctness
  Edge-case branches in loaders not hit by unit tests
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import pytest

# ---------------------------------------------------------------------------
# Rich synthetic fixture content
# ---------------------------------------------------------------------------

# Two tier-0 techs + two tier-1 techs (one with single-string prereq, one with
# list prereq) — exercises both branches of _extract_prerequisites.
_TECH_TXT = """\
tech_physics_1 = {
    tier = 0
    area = physics
    cost = 100
    category = particles
}
tech_society_1 = {
    tier = 0
    area = society
    cost = 100
    category = biology
}
tech_physics_2 = {
    tier = 1
    area = physics
    cost = 200
    category = particles
    prerequisites = "tech_physics_1"
}
tech_advanced = {
    tier = 1
    area = physics
    cost = 300
    category = particles
    prerequisites = { "tech_physics_1" "tech_society_1" }
}
"""

# Building locked behind tier-0 and tier-1 tech
_BUILDING_TXT = """\
building_lab = {
    prerequisites = tech_physics_1
    cost = { minerals = 150 }
    upkeep = { energy = 2 }
}
building_advanced_lab = {
    prerequisites = tech_physics_2
    cost = { minerals = 300 }
    upkeep = { energy = 4 }
}
"""

# District without prerequisites (any planet)
_DISTRICT_TXT = """\
district_city = {
    cost = { minerals = 300 }
    upkeep = { energy = 5 }
}
district_industrial = {
    cost = { minerals = 400 }
    upkeep = { energy = 6 }
    prerequisites = "tech_physics_1"
}
"""

# Job with a plain numeric weight (not a block)
_JOB_TXT = """\
job_researcher = {
    category = specialist
    produces = { research = 5 }
    weight = 3
}
"""

# Policy with option used by faction
_POLICY_TXT = """\
policy_development = {
    potential = { always = yes }
    option = {
        name = policy_development_1
    }
    option = {
        name = policy_development_2
    }
}
"""


def _make_rich_stellaris_dir(tmp_path: Path) -> Path:
    """Write a richer Stellaris common/ directory for integration tests."""
    stellaris = tmp_path / "stellaris"
    common = stellaris / "common"

    (common / "technology").mkdir(parents=True)
    (common / "technology" / "00_tech.txt").write_text(_TECH_TXT, encoding="utf-8")

    (common / "buildings").mkdir(parents=True)
    (common / "buildings" / "00_buildings.txt").write_text(
        _BUILDING_TXT, encoding="utf-8"
    )

    (common / "district_types").mkdir(parents=True)
    (common / "district_types" / "00_districts.txt").write_text(
        _DISTRICT_TXT, encoding="utf-8"
    )

    (common / "jobs").mkdir(parents=True)
    (common / "jobs" / "00_jobs.txt").write_text(_JOB_TXT, encoding="utf-8")

    (common / "policies").mkdir(parents=True)
    (common / "policies" / "00_policies.txt").write_text(_POLICY_TXT, encoding="utf-8")

    for subdir in (
        "ship_sizes",
        "planet_classes",
        "traditions",
        "tradition_categories",
        "ascension_perks",
        "megastructures",
        "pop_faction_types",
        "council_positions",
        "agendas",
        "federation_types",
        "traits",
        "edicts",
    ):
        (common / subdir).mkdir(parents=True)

    (common / "governments" / "civics").mkdir(parents=True)
    return stellaris


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rich_gkg(tmp_path_factory: pytest.TempPathFactory):
    """Load a GameKnowledgeGraph from the rich synthetic fixture once per module."""
    from stellaris_advisor.gamedata.loader import GameKnowledgeGraph

    tmp = tmp_path_factory.mktemp("rich_gkg")
    stellaris = _make_rich_stellaris_dir(tmp)
    return GameKnowledgeGraph.load(stellaris, "3.14")


# ===========================================================================
# All 11 validation checks pass on clean synthetic data
# ===========================================================================


@pytest.mark.integration
class TestValidationChecks:
    def test_exactly_11_validation_results(self, rich_gkg):
        assert len(rich_gkg.validation_results) == 11

    def test_check_names(self, rich_gkg):
        names = {r.check_name for r in rich_gkg.validation_results}
        expected = {
            "tech_dag_acyclicity",
            "tech_prereq_refs",
            "tech_tier_consistency",
            "building_tech_linkage",
            "district_planet_class_validity",
            "tradition_tree_linkage",
            "perk_prereq_consistency",
            "perk_mutual_exclusion_symmetry",
            "megastructure_linkage",
            "policy_faction_linkage",
            "council_leader_class",
        }
        assert names == expected

    def test_all_checks_pass(self, rich_gkg):
        failed = [r for r in rich_gkg.validation_results if not r.passed]
        assert failed == [], f"Failing checks: {[r.check_name for r in failed]}"

    def test_tech_dag_acyclicity_passes(self, rich_gkg):
        result = next(
            r
            for r in rich_gkg.validation_results
            if r.check_name == "tech_dag_acyclicity"
        )
        assert result.passed
        assert result.errors == ()

    def test_building_tech_linkage_passes(self, rich_gkg):
        result = next(
            r
            for r in rich_gkg.validation_results
            if r.check_name == "building_tech_linkage"
        )
        assert result.passed


# ===========================================================================
# Query delegation methods
# ===========================================================================


@pytest.mark.integration
class TestReachableTechs:
    def test_tier0_reachable_with_nothing_researched(self, rich_gkg):
        result = rich_gkg.reachable_techs(frozenset())
        assert "tech_physics_1" in result
        assert "tech_society_1" in result

    def test_tier1_not_reachable_with_nothing_researched(self, rich_gkg):
        # tier-1 requires 6 tier-0 researched (gate threshold) — not met here
        result = rich_gkg.reachable_techs(frozenset())
        assert "tech_physics_2" not in result

    def test_already_researched_not_returned(self, rich_gkg):
        researched = frozenset({"tech_physics_1"})
        result = rich_gkg.reachable_techs(researched)
        assert "tech_physics_1" not in result

    def test_returns_frozenset(self, rich_gkg):
        assert isinstance(rich_gkg.reachable_techs(frozenset()), frozenset)


@pytest.mark.integration
class TestAvailableBuildings:
    def test_no_buildings_without_prereq(self, rich_gkg):
        # building_lab requires tech_physics_1
        result = rich_gkg.available_buildings(frozenset())
        assert "building_lab" not in result

    def test_building_available_when_prereq_researched(self, rich_gkg):
        result = rich_gkg.available_buildings(frozenset({"tech_physics_1"}))
        assert "building_lab" in result

    def test_advanced_building_requires_tier1_tech(self, rich_gkg):
        result = rich_gkg.available_buildings(frozenset({"tech_physics_1"}))
        assert "building_advanced_lab" not in result
        result2 = rich_gkg.available_buildings(
            frozenset({"tech_physics_1", "tech_physics_2"})
        )
        assert "building_advanced_lab" in result2

    def test_returns_frozenset(self, rich_gkg):
        assert isinstance(rich_gkg.available_buildings(frozenset()), frozenset)


@pytest.mark.integration
class TestAvailableDistricts:
    def test_district_without_prereq_always_available(self, rich_gkg):
        result = rich_gkg.available_districts()
        assert "district_city" in result

    def test_district_with_prereq_requires_research(self, rich_gkg):
        no_research = rich_gkg.available_districts(researched=frozenset())
        assert "district_industrial" not in no_research
        with_research = rich_gkg.available_districts(
            researched=frozenset({"tech_physics_1"})
        )
        assert "district_industrial" in with_research

    def test_returns_frozenset(self, rich_gkg):
        assert isinstance(rich_gkg.available_districts(), frozenset)


@pytest.mark.integration
class TestTechPathTo:
    def test_unknown_target_returns_none(self, rich_gkg):
        assert rich_gkg.tech_path_to("nonexistent", frozenset()) is None

    def test_already_researched_returns_empty_tuple(self, rich_gkg):
        result = rich_gkg.tech_path_to("tech_physics_1", frozenset({"tech_physics_1"}))
        assert result == ()

    def test_path_to_tier0_is_just_itself(self, rich_gkg):
        result = rich_gkg.tech_path_to("tech_physics_1", frozenset())
        assert result == ("tech_physics_1",)

    def test_path_to_tier1_includes_prereq(self, rich_gkg):
        result = rich_gkg.tech_path_to("tech_physics_2", frozenset())
        assert result is not None
        assert "tech_physics_1" in result
        assert "tech_physics_2" in result
        assert result.index("tech_physics_1") < result.index("tech_physics_2")

    def test_returns_tuple(self, rich_gkg):
        result = rich_gkg.tech_path_to("tech_physics_1", frozenset())
        assert isinstance(result, tuple)


@pytest.mark.integration
class TestDownstreamUnlocks:
    def test_unknown_tech_returns_empty(self, rich_gkg):
        assert rich_gkg.downstream_unlocks("nonexistent") == frozenset()

    def test_tier0_downstream_includes_tier1(self, rich_gkg):
        result = rich_gkg.downstream_unlocks("tech_physics_1")
        assert "tech_physics_2" in result

    def test_leaf_tech_has_no_downstream(self, rich_gkg):
        # tech_physics_2 has no successors in our fixture
        result = rich_gkg.downstream_unlocks("tech_physics_2")
        assert result == frozenset()

    def test_returns_frozenset(self, rich_gkg):
        assert isinstance(rich_gkg.downstream_unlocks("tech_physics_1"), frozenset)


@pytest.mark.integration
class TestEligiblePerks:
    def test_returns_frozenset(self, rich_gkg):
        assert isinstance(
            rich_gkg.eligible_perks(frozenset(), frozenset(), 0), frozenset
        )

    def test_no_perks_in_empty_fixture(self, rich_gkg):
        # No ascension perks defined in our fixture
        result = rich_gkg.eligible_perks(frozenset(), frozenset(), 10)
        assert result == frozenset()


# ===========================================================================
# Cross-linking: tech.unlocks_buildings populated
# ===========================================================================


@pytest.mark.integration
class TestCrossLinking:
    def test_tech_physics_1_unlocks_lab(self, rich_gkg):
        tech = rich_gkg.tech_defs["tech_physics_1"]
        assert "building_lab" in tech.unlocks_buildings

    def test_tech_physics_2_unlocks_advanced_lab(self, rich_gkg):
        tech = rich_gkg.tech_defs["tech_physics_2"]
        assert "building_advanced_lab" in tech.unlocks_buildings

    def test_tier1_tech_populated(self, rich_gkg):
        assert "tech_physics_2" in rich_gkg.tech_defs
        assert rich_gkg.tech_defs["tech_physics_2"].tier == 1


# ===========================================================================
# Edge-case branches in loader functions (direct unit tests)
# ===========================================================================


@pytest.mark.unit
class TestLoaderBranches:
    """Target specific defensive branches not reachable through file-based load."""

    def test_extract_prerequisites_unknown_type_returns_empty(self):
        """tech_graph._extract_prerequisites: unknown type → frozenset()."""
        from stellaris_advisor.gamedata.tech_graph import load_tech_defs

        # Pass an integer as prerequisites — should gracefully produce empty frozenset
        raw = {
            "tech_x": {"tier": 0, "area": "physics", "cost": 100, "prerequisites": 42}
        }
        defs, dag = load_tech_defs(raw)
        assert defs["tech_x"].prerequisites == frozenset()

    def test_load_tech_defs_skips_non_dict_entry(self):
        """tech_graph.load_tech_defs: non-dict value → entry skipped."""
        from stellaris_advisor.gamedata.tech_graph import load_tech_defs

        raw = {
            "tech_valid": {"tier": 0, "area": "physics", "cost": 100},
            "bad_entry": "not_a_dict",
        }
        defs, _ = load_tech_defs(raw)
        assert "tech_valid" in defs
        assert "bad_entry" not in defs

    def test_load_tech_defs_weight_modifier_non_dict_entry(self):
        """tech_graph: non-dict item inside weight_modifier list → skipped."""
        from stellaris_advisor.gamedata.tech_graph import load_tech_defs

        raw = {
            "tech_wm": {
                "tier": 0,
                "area": "physics",
                "cost": 100,
                "weight_modifier": ["not_a_dict", {"factor": 2.0}],
            }
        }
        defs, _ = load_tech_defs(raw)
        # Only the valid dict entry is included; "not_a_dict" is skipped
        assert len(defs["tech_wm"].weight_modifiers) == 1
        assert defs["tech_wm"].weight_modifiers[0].factor == 2.0

    def test_load_building_defs_skips_non_dict_entry(self):
        """buildings.load_building_defs: non-dict value → skipped."""
        from stellaris_advisor.gamedata.buildings import load_building_defs

        raw = {
            "building_valid": {"cost": {"minerals": 100}},
            "bad_entry": 99,
        }
        defs = load_building_defs(raw)
        assert "building_valid" in defs
        assert "bad_entry" not in defs

    def test_load_building_prerequisites_unknown_type(self):
        """buildings._extract_prerequisites: unknown type → empty frozenset."""
        from stellaris_advisor.gamedata.buildings import load_building_defs

        raw = {"building_x": {"cost": {"minerals": 100}, "prerequisites": 123}}
        defs = load_building_defs(raw)
        assert defs["building_x"].prerequisites == frozenset()

    def test_load_district_prerequisites_list(self):
        """districts._extract_prerequisites: list value → frozenset."""
        from stellaris_advisor.gamedata.districts import load_district_defs

        raw = {
            "district_x": {
                "cost": {"minerals": 200},
                "prerequisites": ["tech_a", "tech_b"],
            }
        }
        defs = load_district_defs(raw)
        assert defs["district_x"].prerequisites == frozenset({"tech_a", "tech_b"})

    def test_load_district_prerequisites_unknown_type(self):
        """districts._extract_prerequisites: unknown type → empty frozenset."""
        from stellaris_advisor.gamedata.districts import load_district_defs

        raw = {"district_x": {"cost": {"minerals": 200}, "prerequisites": 999}}
        defs = load_district_defs(raw)
        assert defs["district_x"].prerequisites == frozenset()

    def test_load_district_defs_skips_non_dict_entry(self):
        """districts.load_district_defs: non-dict value → skipped."""
        from stellaris_advisor.gamedata.districts import load_district_defs

        raw = {"district_valid": {"cost": {"minerals": 200}}, "bad": "oops"}
        defs = load_district_defs(raw)
        assert "district_valid" in defs
        assert "bad" not in defs

    def test_load_job_with_plain_weight(self):
        """jobs.load_job_defs: plain numeric weight (not a block dict)."""
        from stellaris_advisor.gamedata.jobs import load_job_defs

        raw = {"job_worker": {"category": "worker", "weight": 5}}
        defs = load_job_defs(raw)
        assert defs["job_worker"].weight == pytest.approx(5.0)

    def test_load_job_defs_skips_non_dict_entry(self):
        """jobs.load_job_defs: non-dict value → skipped."""
        from stellaris_advisor.gamedata.jobs import load_job_defs

        raw = {"job_valid": {"category": "worker"}, "bad": "skip_me"}
        defs = load_job_defs(raw)
        assert "job_valid" in defs
        assert "bad" not in defs

    def test_ascension_perk_single_string_prereq(self):
        """ascension_perks._extract_frozenset: single string → frozenset of 1."""
        from stellaris_advisor.gamedata.ascension_perks import load_ascension_perk_defs

        raw = {"perk_x": {"prerequisites": "tech_a", "required_traditions": 3}}
        defs = load_ascension_perk_defs(raw)
        assert defs["perk_x"].prerequisites == frozenset({"tech_a"})

    def test_ascension_perk_unknown_type_prereq(self):
        """ascension_perks._extract_frozenset: unknown type → empty frozenset."""
        from stellaris_advisor.gamedata.ascension_perks import load_ascension_perk_defs

        raw = {"perk_x": {"prerequisites": 42, "required_traditions": 0}}
        defs = load_ascension_perk_defs(raw)
        assert defs["perk_x"].prerequisites == frozenset()

    def test_ascension_perk_defs_skips_non_dict_entry(self):
        """ascension_perks.load_ascension_perk_defs: non-dict → skipped."""
        from stellaris_advisor.gamedata.ascension_perks import load_ascension_perk_defs

        raw = {"perk_valid": {}, "bad": "not_dict"}
        defs = load_ascension_perk_defs(raw)
        assert "perk_valid" in defs
        assert "bad" not in defs

    def test_building_unlock_chain_prereq_not_in_dag(self):
        """buildings.building_unlock_chain: prereq tech absent from DAG → None."""
        from stellaris_advisor.gamedata.buildings import (
            building_unlock_chain,
            load_building_defs,
        )
        from stellaris_advisor.gamedata.tech_graph import load_tech_defs

        # Building requires "tech_missing" which doesn't exist in tech_defs
        building_raw = {"building_x": {"prerequisites": "tech_missing"}}
        tech_raw = {"tech_present": {"tier": 0, "area": "physics", "cost": 100}}
        building_defs = load_building_defs(building_raw)
        tech_defs, tech_dag = load_tech_defs(tech_raw)

        result = building_unlock_chain(
            building_defs, tech_defs, tech_dag, "building_x", frozenset()
        )
        assert result is None

    def test_eligible_perks_skips_excluded(self):
        """ascension_perks.eligible_perks: perk excluded by taken perk."""
        from stellaris_advisor.gamedata.ascension_perks import (
            eligible_perks,
            load_ascension_perk_defs,
        )

        raw = {
            "perk_a": {"excluded_perks": "perk_b"},
            "perk_b": {"excluded_perks": "perk_a"},
        }
        defs = load_ascension_perk_defs(raw)
        # With perk_a taken, perk_b should be excluded
        result = eligible_perks(defs, frozenset(), 0, frozenset({"perk_a"}))
        assert "perk_b" not in result
        assert "perk_a" not in result  # already taken

    def test_validation_tier_consistency_skips_unknown_prereq(self):
        """validation.check_tech_tier_consistency: unknown prereq is skipped (line 141)."""
        from stellaris_advisor.gamedata.validation import check_tech_tier_consistency
        from stellaris_advisor.gamedata.tech_graph import load_tech_defs

        # tech_b has unknown prereq "nonexistent" AND tier violation with "tech_a"
        raw = {
            "tech_a": {"tier": 0, "area": "physics", "cost": 100},
            "tech_b": {
                "tier": 1,
                "area": "physics",
                "cost": 200,
                "prerequisites": ["nonexistent", "tech_a"],
            },
        }
        tech_defs, _ = load_tech_defs(raw)
        result = check_tech_tier_consistency(tech_defs)
        # "nonexistent" is not in tech_defs → skipped (line 141), not reported here
        errors_about_nonexistent = [e for e in result.errors if "nonexistent" in e]
        assert errors_about_nonexistent == []


# ===========================================================================
# GKG attribute completeness
# ===========================================================================


@pytest.mark.integration
class TestGKGAttributes:
    def test_all_proxy_attrs_are_mapping_proxy(self, rich_gkg):
        proxy_attrs = [
            "tech_defs",
            "building_defs",
            "district_defs",
            "job_defs",
            "ship_class_defs",
            "planet_class_defs",
            "tradition_trees",
            "tradition_defs",
            "ascension_perk_defs",
            "megastructure_defs",
            "policy_defs",
            "faction_defs",
            "council_position_defs",
            "agenda_defs",
            "federation_type_defs",
            "civic_defs",
            "trait_defs",
            "edict_defs",
        ]
        for attr in proxy_attrs:
            assert isinstance(getattr(rich_gkg, attr), MappingProxyType), attr

    def test_tech_defs_populated(self, rich_gkg):
        assert len(rich_gkg.tech_defs) == 4  # 4 techs in fixture

    def test_building_defs_populated(self, rich_gkg):
        assert len(rich_gkg.building_defs) == 2

    def test_district_defs_populated(self, rich_gkg):
        assert len(rich_gkg.district_defs) == 2

    def test_job_defs_populated(self, rich_gkg):
        assert len(rich_gkg.job_defs) == 1

    def test_policy_defs_populated(self, rich_gkg):
        assert len(rich_gkg.policy_defs) == 1

    def test_version_set_correctly(self, rich_gkg):
        assert rich_gkg.version == "3.14"
