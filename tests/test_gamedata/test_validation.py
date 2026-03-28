"""Tests for gamedata/validation.py — TDD: tests written first.

Covers:
  cross_link_tech_unlocks
  11 ValidationResult checks + validate_game_data
"""

from __future__ import annotations

import pytest
import networkx as nx

from stellaris_advisor.gamedata.models import (
    AscensionPerkDef,
    BuildingDef,
    CouncilPositionDef,
    DistrictDef,
    FactionDef,
    FactionIssueDef,
    MegastructureDef,
    PlanetClassDef,
    PolicyDef,
    PolicyOptionDef,
    ShipClassDef,
    TechNode,
    TraditionDef,
    TraditionTree,
)


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------


def _tech(
    tid: str,
    tier: int = 0,
    prerequisites: frozenset[str] = frozenset(),
) -> TechNode:
    return TechNode(
        id=tid,
        name=tid,
        area="physics",
        tier=tier,
        cost=1000,
        category="particles",
        prerequisites=prerequisites,
    )


def _building(
    bid: str,
    prereqs: frozenset[str] = frozenset(),
    planet_classes: frozenset[str] = frozenset(),
) -> BuildingDef:
    return BuildingDef(
        id=bid,
        name=bid,
        category="research",
        base_cost=(),
        upkeep=(),
        jobs=(),
        prerequisites=prereqs,
        planet_classes=planet_classes,
        empire_types=frozenset(),
    )


def _district(
    did: str,
    prereqs: frozenset[str] = frozenset(),
    planet_classes: frozenset[str] = frozenset(),
) -> DistrictDef:
    return DistrictDef(
        id=did,
        name=did,
        base_cost=(),
        upkeep=(),
        jobs=(),
        planet_classes=planet_classes,
        prerequisites=prereqs,
    )


def _ship(sid: str, prereqs: frozenset[str] = frozenset()) -> ShipClassDef:
    return ShipClassDef(
        id=sid,
        name=sid,
        hull=100,
        evasion=0.0,
        speed=0.0,
        sections=(),
        base_cost=(),
        upkeep=(),
        prerequisites=prereqs,
    )


def _planet_class(pcid: str) -> PlanetClassDef:
    return PlanetClassDef(
        id=pcid,
        name=pcid,
        habitability_category="continental",
        district_cap=20,
        base_habitability=1.0,
        modifiers=(),
    )


def _tradition(tid: str, tree_id: str = "tr_test") -> TraditionDef:
    return TraditionDef(id=tid, name=tid, tree_id=tree_id, effect="")


def _tradition_tree(trid: str) -> TraditionTree:
    return TraditionTree(
        id=trid, name=trid, tradition_ids=(), adopt_effect="", finish_effect=""
    )


def _perk(
    pid: str,
    required_perks: frozenset[str] = frozenset(),
    excluded_perks: frozenset[str] = frozenset(),
    prerequisites: frozenset[str] = frozenset(),
) -> AscensionPerkDef:
    return AscensionPerkDef(
        id=pid,
        name=pid,
        description="",
        effect="",
        prerequisites=prerequisites,
        required_perks=required_perks,
        excluded_perks=excluded_perks,
    )


def _mega(
    mid: str,
    prereqs: frozenset[str] = frozenset(),
    required_perks: frozenset[str] = frozenset(),
) -> MegastructureDef:
    return MegastructureDef(
        id=mid,
        name=mid,
        stages=(),
        prerequisites=prereqs,
        required_perks=required_perks,
        site_requirements=(),
    )


def _policy(pid: str, option_ids: list[str]) -> PolicyDef:
    return PolicyDef(
        id=pid,
        name=pid,
        category="",
        options=tuple(
            PolicyOptionDef(id=oid, name=oid, effects=()) for oid in option_ids
        ),
    )


def _issue(iid: str, desired_option: str) -> FactionIssueDef:
    return FactionIssueDef(
        id=iid,
        name=iid,
        approval_effect=0.1,
        policy_id="",
        desired_option=desired_option,
    )


def _faction(fid: str, issues: list[FactionIssueDef] | None = None) -> FactionDef:
    return FactionDef(id=fid, name=fid, ethic="", issues=tuple(issues or []))


def _council_pos(pid: str, leader_class: str = "scientist") -> CouncilPositionDef:
    return CouncilPositionDef(id=pid, name=pid, leader_class=leader_class, effects=())


# ===========================================================================
# cross_link_tech_unlocks
# ===========================================================================


@pytest.mark.unit
class TestCrossLinkTechUnlocks:
    def test_unlocks_buildings_populated(self):
        from stellaris_advisor.gamedata.validation import cross_link_tech_unlocks

        tech_defs = {"tech_a": _tech("tech_a")}
        building_defs = {"b1": _building("b1", prereqs=frozenset({"tech_a"}))}
        result = cross_link_tech_unlocks(tech_defs, building_defs, {}, {})
        assert "b1" in result["tech_a"].unlocks_buildings

    def test_unlocks_buildings_empty_when_no_prereq_match(self):
        from stellaris_advisor.gamedata.validation import cross_link_tech_unlocks

        tech_defs = {"tech_a": _tech("tech_a")}
        building_defs = {"b1": _building("b1", prereqs=frozenset({"tech_b"}))}
        result = cross_link_tech_unlocks(tech_defs, building_defs, {}, {})
        assert result["tech_a"].unlocks_buildings == ()

    def test_unlocks_districts_populated(self):
        from stellaris_advisor.gamedata.validation import cross_link_tech_unlocks

        tech_defs = {"tech_a": _tech("tech_a")}
        district_defs = {"d1": _district("d1", prereqs=frozenset({"tech_a"}))}
        result = cross_link_tech_unlocks(tech_defs, {}, district_defs, {})
        assert "d1" in result["tech_a"].unlocks_districts

    def test_unlocks_ship_sizes_populated(self):
        from stellaris_advisor.gamedata.validation import cross_link_tech_unlocks

        tech_defs = {"tech_a": _tech("tech_a")}
        ship_defs = {"ship_a": _ship("ship_a", prereqs=frozenset({"tech_a"}))}
        result = cross_link_tech_unlocks(tech_defs, {}, {}, ship_defs)
        assert "ship_a" in result["tech_a"].unlocks_ship_sizes

    def test_original_not_mutated(self):
        from stellaris_advisor.gamedata.validation import cross_link_tech_unlocks

        tech_defs = {"tech_a": _tech("tech_a")}
        building_defs = {"b1": _building("b1", prereqs=frozenset({"tech_a"}))}
        original_tech = tech_defs["tech_a"]
        cross_link_tech_unlocks(tech_defs, building_defs, {}, {})
        assert original_tech.unlocks_buildings == ()

    def test_multiple_buildings_for_same_tech(self):
        from stellaris_advisor.gamedata.validation import cross_link_tech_unlocks

        tech_defs = {"tech_a": _tech("tech_a")}
        building_defs = {
            "b1": _building("b1", prereqs=frozenset({"tech_a"})),
            "b2": _building("b2", prereqs=frozenset({"tech_a"})),
        }
        result = cross_link_tech_unlocks(tech_defs, building_defs, {}, {})
        assert set(result["tech_a"].unlocks_buildings) == {"b1", "b2"}


# ===========================================================================
# ValidationResult
# ===========================================================================


@pytest.mark.unit
class TestValidationResult:
    def test_frozen_dataclass(self):
        from stellaris_advisor.gamedata.validation import ValidationResult

        vr = ValidationResult(check_name="test_check", passed=True, errors=())
        with pytest.raises(AttributeError):
            vr.passed = False  # type: ignore[misc]

    def test_fields(self):
        from stellaris_advisor.gamedata.validation import ValidationResult

        vr = ValidationResult(
            check_name="my_check", passed=False, errors=("error one", "error two")
        )
        assert vr.check_name == "my_check"
        assert vr.passed is False
        assert len(vr.errors) == 2


# ===========================================================================
# check_tech_dag_acyclicity
# ===========================================================================


@pytest.mark.unit
class TestCheckTechDagAcyclicity:
    def test_acyclic_dag_passes(self):
        from stellaris_advisor.gamedata.validation import check_tech_dag_acyclicity

        dag = nx.DiGraph()
        dag.add_edges_from([("tech_a", "tech_b"), ("tech_b", "tech_c")])
        result = check_tech_dag_acyclicity(dag)
        assert result.passed is True
        assert result.errors == ()
        assert result.check_name == "tech_dag_acyclicity"

    def test_cyclic_dag_fails(self):
        from stellaris_advisor.gamedata.validation import check_tech_dag_acyclicity

        dag = nx.DiGraph()
        dag.add_edges_from([("tech_a", "tech_b"), ("tech_b", "tech_a")])
        result = check_tech_dag_acyclicity(dag)
        assert result.passed is False
        assert len(result.errors) >= 1

    def test_empty_dag_passes(self):
        from stellaris_advisor.gamedata.validation import check_tech_dag_acyclicity

        result = check_tech_dag_acyclicity(nx.DiGraph())
        assert result.passed is True


# ===========================================================================
# check_tech_prereq_refs
# ===========================================================================


@pytest.mark.unit
class TestCheckTechPrereqRefs:
    def test_valid_refs_passes(self):
        from stellaris_advisor.gamedata.validation import check_tech_prereq_refs

        tech_defs = {
            "tech_a": _tech("tech_a"),
            "tech_b": _tech("tech_b", prerequisites=frozenset({"tech_a"})),
        }
        result = check_tech_prereq_refs(tech_defs)
        assert result.passed is True
        assert result.check_name == "tech_prereq_refs"

    def test_dangling_prereq_fails(self):
        from stellaris_advisor.gamedata.validation import check_tech_prereq_refs

        tech_defs = {
            "tech_b": _tech("tech_b", prerequisites=frozenset({"tech_missing"})),
        }
        result = check_tech_prereq_refs(tech_defs)
        assert result.passed is False
        assert any("tech_missing" in e for e in result.errors)

    def test_empty_defs_passes(self):
        from stellaris_advisor.gamedata.validation import check_tech_prereq_refs

        result = check_tech_prereq_refs({})
        assert result.passed is True


# ===========================================================================
# check_tech_tier_consistency
# ===========================================================================


@pytest.mark.unit
class TestCheckTechTierConsistency:
    def test_consistent_tiers_passes(self):
        from stellaris_advisor.gamedata.validation import check_tech_tier_consistency

        tech_defs = {
            "tech_a": _tech("tech_a", tier=0),
            "tech_b": _tech("tech_b", tier=1, prerequisites=frozenset({"tech_a"})),
        }
        result = check_tech_tier_consistency(tech_defs)
        assert result.passed is True
        assert result.check_name == "tech_tier_consistency"

    def test_prereq_same_tier_fails(self):
        from stellaris_advisor.gamedata.validation import check_tech_tier_consistency

        tech_defs = {
            "tech_a": _tech("tech_a", tier=1),
            "tech_b": _tech("tech_b", tier=1, prerequisites=frozenset({"tech_a"})),
        }
        result = check_tech_tier_consistency(tech_defs)
        assert result.passed is False

    def test_prereq_higher_tier_fails(self):
        from stellaris_advisor.gamedata.validation import check_tech_tier_consistency

        tech_defs = {
            "tech_a": _tech("tech_a", tier=2),
            "tech_b": _tech("tech_b", tier=1, prerequisites=frozenset({"tech_a"})),
        }
        result = check_tech_tier_consistency(tech_defs)
        assert result.passed is False

    def test_tier_zero_no_constraint(self):
        from stellaris_advisor.gamedata.validation import check_tech_tier_consistency

        tech_defs = {
            "tech_a": _tech("tech_a", tier=0),
            "tech_b": _tech("tech_b", tier=0, prerequisites=frozenset({"tech_a"})),
        }
        result = check_tech_tier_consistency(tech_defs)
        assert result.passed is True


# ===========================================================================
# check_building_tech_linkage
# ===========================================================================


@pytest.mark.unit
class TestCheckBuildingTechLinkage:
    def test_valid_prereqs_passes(self):
        from stellaris_advisor.gamedata.validation import check_building_tech_linkage

        tech_defs = {"tech_a": _tech("tech_a")}
        building_defs = {"b1": _building("b1", prereqs=frozenset({"tech_a"}))}
        result = check_building_tech_linkage(building_defs, tech_defs)
        assert result.passed is True
        assert result.check_name == "building_tech_linkage"

    def test_missing_tech_fails(self):
        from stellaris_advisor.gamedata.validation import check_building_tech_linkage

        building_defs = {"b1": _building("b1", prereqs=frozenset({"tech_missing"}))}
        result = check_building_tech_linkage(building_defs, {})
        assert result.passed is False
        assert any("tech_missing" in e for e in result.errors)

    def test_no_prereqs_passes(self):
        from stellaris_advisor.gamedata.validation import check_building_tech_linkage

        result = check_building_tech_linkage({"b1": _building("b1")}, {})
        assert result.passed is True


# ===========================================================================
# check_district_planet_class_validity
# ===========================================================================


@pytest.mark.unit
class TestCheckDistrictPlanetClassValidity:
    def test_valid_planet_classes_passes(self):
        from stellaris_advisor.gamedata.validation import (
            check_district_planet_class_validity,
        )

        pc_defs = {"pc_continental": _planet_class("pc_continental")}
        district_defs = {
            "d1": _district("d1", planet_classes=frozenset({"pc_continental"}))
        }
        result = check_district_planet_class_validity(district_defs, pc_defs)
        assert result.passed is True
        assert result.check_name == "district_planet_class_validity"

    def test_missing_planet_class_fails(self):
        from stellaris_advisor.gamedata.validation import (
            check_district_planet_class_validity,
        )

        district_defs = {"d1": _district("d1", planet_classes=frozenset({"pc_missing"}))}
        result = check_district_planet_class_validity(district_defs, {})
        assert result.passed is False
        assert any("pc_missing" in e for e in result.errors)

    def test_empty_planet_classes_passes(self):
        from stellaris_advisor.gamedata.validation import (
            check_district_planet_class_validity,
        )

        result = check_district_planet_class_validity({"d1": _district("d1")}, {})
        assert result.passed is True


# ===========================================================================
# check_tradition_tree_linkage
# ===========================================================================


@pytest.mark.unit
class TestCheckTraditionTreeLinkage:
    def test_valid_tree_id_passes(self):
        from stellaris_advisor.gamedata.validation import check_tradition_tree_linkage

        tree_defs = {"tr_discovery": _tradition_tree("tr_discovery")}
        trad_defs = {"tr_a": _tradition("tr_a", tree_id="tr_discovery")}
        result = check_tradition_tree_linkage(trad_defs, tree_defs)
        assert result.passed is True
        assert result.check_name == "tradition_tree_linkage"

    def test_missing_tree_fails(self):
        from stellaris_advisor.gamedata.validation import check_tradition_tree_linkage

        trad_defs = {"tr_a": _tradition("tr_a", tree_id="tr_missing")}
        result = check_tradition_tree_linkage(trad_defs, {})
        assert result.passed is False
        assert any("tr_missing" in e for e in result.errors)

    def test_empty_tradition_defs_passes(self):
        from stellaris_advisor.gamedata.validation import check_tradition_tree_linkage

        result = check_tradition_tree_linkage({}, {})
        assert result.passed is True

    def test_tree_lists_unknown_tradition_fails(self):
        from stellaris_advisor.gamedata.validation import check_tradition_tree_linkage

        tree_defs = {
            "tr_discovery": TraditionTree(
                id="tr_discovery",
                name="Discovery",
                tradition_ids=("unknown_trad",),
                adopt_effect="",
                finish_effect="",
            )
        }
        result = check_tradition_tree_linkage({}, tree_defs)
        assert result.passed is False
        assert any("unknown_trad" in e for e in result.errors)


# ===========================================================================
# check_perk_prereq_consistency
# ===========================================================================


@pytest.mark.unit
class TestCheckPerkPrereqConsistency:
    def test_valid_required_perks_passes(self):
        from stellaris_advisor.gamedata.validation import check_perk_prereq_consistency

        perk_defs = {
            "ap_a": _perk("ap_a"),
            "ap_b": _perk("ap_b", required_perks=frozenset({"ap_a"})),
        }
        result = check_perk_prereq_consistency(perk_defs)
        assert result.passed is True
        assert result.check_name == "perk_prereq_consistency"

    def test_missing_required_perk_fails(self):
        from stellaris_advisor.gamedata.validation import check_perk_prereq_consistency

        perk_defs = {"ap_b": _perk("ap_b", required_perks=frozenset({"ap_missing"}))}
        result = check_perk_prereq_consistency(perk_defs)
        assert result.passed is False
        assert any("ap_missing" in e for e in result.errors)

    def test_empty_perk_defs_passes(self):
        from stellaris_advisor.gamedata.validation import check_perk_prereq_consistency

        result = check_perk_prereq_consistency({})
        assert result.passed is True


# ===========================================================================
# check_perk_mutual_exclusion_symmetry
# ===========================================================================


@pytest.mark.unit
class TestCheckPerkMutualExclusionSymmetry:
    def test_symmetric_exclusion_passes(self):
        from stellaris_advisor.gamedata.validation import (
            check_perk_mutual_exclusion_symmetry,
        )

        perk_defs = {
            "ap_a": _perk("ap_a", excluded_perks=frozenset({"ap_b"})),
            "ap_b": _perk("ap_b", excluded_perks=frozenset({"ap_a"})),
        }
        result = check_perk_mutual_exclusion_symmetry(perk_defs)
        assert result.passed is True
        assert result.check_name == "perk_mutual_exclusion_symmetry"

    def test_asymmetric_exclusion_fails(self):
        from stellaris_advisor.gamedata.validation import (
            check_perk_mutual_exclusion_symmetry,
        )

        perk_defs = {
            "ap_a": _perk("ap_a", excluded_perks=frozenset({"ap_b"})),
            "ap_b": _perk("ap_b"),
        }
        result = check_perk_mutual_exclusion_symmetry(perk_defs)
        assert result.passed is False

    def test_excluded_perk_not_in_defs_fails(self):
        from stellaris_advisor.gamedata.validation import (
            check_perk_mutual_exclusion_symmetry,
        )

        perk_defs = {"ap_a": _perk("ap_a", excluded_perks=frozenset({"ap_missing"}))}
        result = check_perk_mutual_exclusion_symmetry(perk_defs)
        assert result.passed is False


# ===========================================================================
# check_megastructure_linkage
# ===========================================================================


@pytest.mark.unit
class TestCheckMegastructureLinkage:
    def test_valid_linkage_passes(self):
        from stellaris_advisor.gamedata.validation import check_megastructure_linkage

        tech_defs = {"tech_a": _tech("tech_a")}
        perk_defs = {"ap_a": _perk("ap_a")}
        mega_defs = {
            "mega_a": _mega(
                "mega_a",
                prereqs=frozenset({"tech_a"}),
                required_perks=frozenset({"ap_a"}),
            )
        }
        result = check_megastructure_linkage(mega_defs, tech_defs, perk_defs)
        assert result.passed is True
        assert result.check_name == "megastructure_linkage"

    def test_missing_tech_prereq_fails(self):
        from stellaris_advisor.gamedata.validation import check_megastructure_linkage

        mega_defs = {"mega_a": _mega("mega_a", prereqs=frozenset({"tech_missing"}))}
        result = check_megastructure_linkage(mega_defs, {}, {})
        assert result.passed is False
        assert any("tech_missing" in e for e in result.errors)

    def test_missing_perk_prereq_fails(self):
        from stellaris_advisor.gamedata.validation import check_megastructure_linkage

        mega_defs = {
            "mega_a": _mega("mega_a", required_perks=frozenset({"ap_missing"}))
        }
        result = check_megastructure_linkage(mega_defs, {}, {})
        assert result.passed is False
        assert any("ap_missing" in e for e in result.errors)

    def test_no_prereqs_passes(self):
        from stellaris_advisor.gamedata.validation import check_megastructure_linkage

        result = check_megastructure_linkage({"mega_a": _mega("mega_a")}, {}, {})
        assert result.passed is True


# ===========================================================================
# check_policy_faction_linkage
# ===========================================================================


@pytest.mark.unit
class TestCheckPolicyFactionLinkage:
    def test_valid_option_passes(self):
        from stellaris_advisor.gamedata.validation import check_policy_faction_linkage

        policy_defs = {"pol_a": _policy("pol_a", ["opt_open"])}
        faction_defs = {
            "faction_a": _faction("faction_a", [_issue("issue_1", "opt_open")])
        }
        result = check_policy_faction_linkage(faction_defs, policy_defs)
        assert result.passed is True
        assert result.check_name == "policy_faction_linkage"

    def test_dangling_desired_option_fails(self):
        from stellaris_advisor.gamedata.validation import check_policy_faction_linkage

        policy_defs = {"pol_a": _policy("pol_a", ["opt_open"])}
        faction_defs = {
            "faction_a": _faction("faction_a", [_issue("issue_1", "opt_missing")])
        }
        result = check_policy_faction_linkage(faction_defs, policy_defs)
        assert result.passed is False
        assert any("opt_missing" in e for e in result.errors)

    def test_faction_with_no_issues_passes(self):
        from stellaris_advisor.gamedata.validation import check_policy_faction_linkage

        result = check_policy_faction_linkage({"faction_a": _faction("faction_a")}, {})
        assert result.passed is True


# ===========================================================================
# check_council_leader_class
# ===========================================================================


@pytest.mark.unit
class TestCheckCouncilLeaderClass:
    def test_valid_leader_classes_passes(self):
        from stellaris_advisor.gamedata.validation import check_council_leader_class

        council_defs = {
            "pos_s": _council_pos("pos_s", "scientist"),
            "pos_o": _council_pos("pos_o", "official"),
            "pos_c": _council_pos("pos_c", "commander"),
        }
        result = check_council_leader_class(council_defs)
        assert result.passed is True
        assert result.check_name == "council_leader_class"

    def test_empty_leader_class_fails(self):
        from stellaris_advisor.gamedata.validation import check_council_leader_class

        result = check_council_leader_class({"pos_a": _council_pos("pos_a", "")})
        assert result.passed is False

    def test_invalid_leader_class_fails(self):
        from stellaris_advisor.gamedata.validation import check_council_leader_class

        result = check_council_leader_class({"pos_a": _council_pos("pos_a", "wizard")})
        assert result.passed is False
        assert any("wizard" in e for e in result.errors)

    def test_empty_council_defs_passes(self):
        from stellaris_advisor.gamedata.validation import check_council_leader_class

        result = check_council_leader_class({})
        assert result.passed is True


# ===========================================================================
# validate_game_data
# ===========================================================================


@pytest.mark.unit
class TestValidateGameData:
    @pytest.fixture
    def clean_args(self):
        tech_defs = {"tech_a": _tech("tech_a", tier=0)}
        dag = nx.DiGraph()
        dag.add_node("tech_a")
        building_defs = {"b1": _building("b1", prereqs=frozenset({"tech_a"}))}
        district_defs = {"d1": _district("d1")}
        ship_class_defs: dict[str, ShipClassDef] = {}
        tradition_trees = {"tr_test": _tradition_tree("tr_test")}
        tradition_defs = {"tr_a": _tradition("tr_a", tree_id="tr_test")}
        perk_defs = {"ap_a": _perk("ap_a")}
        mega_defs = {"mega_a": _mega("mega_a")}
        policy_defs = {"pol_a": _policy("pol_a", ["opt_open"])}
        faction_defs = {
            "faction_a": _faction("faction_a", [_issue("issue_1", "opt_open")])
        }
        council_position_defs = {"pos_a": _council_pos("pos_a", "scientist")}
        return (
            tech_defs,
            dag,
            building_defs,
            district_defs,
            ship_class_defs,
            tradition_trees,
            tradition_defs,
            perk_defs,
            mega_defs,
            policy_defs,
            faction_defs,
            council_position_defs,
        )

    def test_clean_data_all_pass(self, clean_args):
        from stellaris_advisor.gamedata.validation import validate_game_data

        results = validate_game_data(*clean_args)
        assert all(r.passed for r in results)

    def test_returns_11_results(self, clean_args):
        from stellaris_advisor.gamedata.validation import validate_game_data

        results = validate_game_data(*clean_args)
        assert len(results) == 11

    def test_returns_tuple_of_validation_results(self, clean_args):
        from stellaris_advisor.gamedata.validation import (
            ValidationResult,
            validate_game_data,
        )

        results = validate_game_data(*clean_args)
        assert isinstance(results, tuple)
        for r in results:
            assert isinstance(r, ValidationResult)

    def test_cyclic_dag_produces_failed_result(self):
        from stellaris_advisor.gamedata.validation import validate_game_data

        dag = nx.DiGraph()
        dag.add_edges_from([("tech_a", "tech_b"), ("tech_b", "tech_a")])
        results = validate_game_data(
            {}, dag, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
        )
        failed = [r for r in results if not r.passed]
        assert any(r.check_name == "tech_dag_acyclicity" for r in failed)

    def test_missing_tech_prereq_produces_failed_result(self):
        from stellaris_advisor.gamedata.validation import validate_game_data

        tech_defs = {"tech_b": _tech("tech_b", prerequisites=frozenset({"tech_missing"}))}
        dag = nx.DiGraph()
        dag.add_node("tech_b")
        results = validate_game_data(
            tech_defs, dag, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
        )
        failed_names = {r.check_name for r in results if not r.passed}
        assert "tech_prereq_refs" in failed_names
