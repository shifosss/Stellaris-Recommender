"""Tests for gamedata frozen dataclass models (TDD: RED phase runs before implementation)."""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_output_modifier(resource: str = "energy", amount: float = 1.0):
    from stellaris_advisor.gamedata.models import OutputModifier

    return OutputModifier(resource=resource, amount=amount)


def _make_weight_modifier(factor: float = 1.5, conditions: tuple[str, ...] = ()):
    from stellaris_advisor.gamedata.models import WeightModifier

    return WeightModifier(factor=factor, conditions=conditions)


def _make_tech_node(**overrides):
    from stellaris_advisor.gamedata.models import TechNode

    defaults = dict(
        id="tech_lasers_1",
        name="Red Lasers",
        area="physics",
        tier=1,
        cost=1000,
        category="field_manipulation",
    )
    defaults.update(overrides)
    return TechNode(**defaults)


def _make_building_def(**overrides):
    from stellaris_advisor.gamedata.models import BuildingDef

    defaults = dict(
        id="building_city_1",
        name="City",
        category="general",
        base_cost=(_make_output_modifier("minerals", 50.0),),
        upkeep=(_make_output_modifier("energy", 1.0),),
        jobs=("clerk",),
        prerequisites=frozenset(),
        planet_classes=frozenset(),
        empire_types=frozenset(),
    )
    defaults.update(overrides)
    return BuildingDef(**defaults)


def _make_district_def(**overrides):
    from stellaris_advisor.gamedata.models import DistrictDef

    defaults = dict(
        id="district_city",
        name="City District",
        base_cost=(_make_output_modifier("minerals", 100.0),),
        upkeep=(_make_output_modifier("energy", 2.0),),
        jobs=("clerk",),
        planet_classes=frozenset({"pc_continental"}),
        prerequisites=frozenset(),
    )
    defaults.update(overrides)
    return DistrictDef(**defaults)


# ---------------------------------------------------------------------------
# Core Layer: WeightModifier
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestWeightModifier:
    def test_frozen(self):
        wm = _make_weight_modifier(factor=1.5)
        with pytest.raises(AttributeError):
            wm.factor = 2.0  # type: ignore[misc]

    def test_default_conditions(self):
        wm = _make_weight_modifier()
        assert wm.conditions == ()

    def test_conditions_is_tuple(self):
        wm = _make_weight_modifier(conditions=("has_tradition",))
        assert isinstance(wm.conditions, tuple)
        assert wm.conditions == ("has_tradition",)


# ---------------------------------------------------------------------------
# Core Layer: OutputModifier
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOutputModifier:
    def test_frozen(self):
        om = _make_output_modifier()
        with pytest.raises(AttributeError):
            om.resource = "minerals"  # type: ignore[misc]

    def test_fields(self):
        om = _make_output_modifier("alloys", 5.0)
        assert om.resource == "alloys"
        assert om.amount == 5.0


# ---------------------------------------------------------------------------
# Core Layer: TechNode
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTechNode:
    def test_frozen(self):
        node = _make_tech_node()
        with pytest.raises(AttributeError):
            node.name = "Blue Lasers"  # type: ignore[misc]

    def test_defaults(self):
        node = _make_tech_node()
        assert node.is_rare is False
        assert node.is_dangerous is False
        assert node.is_start_tech is False
        assert node.prerequisites == frozenset()
        assert node.weight_modifiers == ()
        assert node.unlocks_buildings == ()
        assert node.unlocks_districts == ()
        assert node.unlocks_ship_sizes == ()
        assert node.description == ""

    def test_prerequisites_is_frozenset(self):
        node = _make_tech_node(prerequisites=frozenset({"tech_a", "tech_b"}))
        assert isinstance(node.prerequisites, frozenset)

    def test_weight_modifiers_is_tuple(self):
        wm = _make_weight_modifier(factor=2.0)
        node = _make_tech_node(weight_modifiers=(wm,))
        assert isinstance(node.weight_modifiers, tuple)
        assert node.weight_modifiers[0].factor == 2.0

    def test_unlocks_fields_are_tuples(self):
        node = _make_tech_node(
            unlocks_buildings=("building_research_lab_1",),
            unlocks_districts=("district_research_complex",),
            unlocks_ship_sizes=("destroyer",),
        )
        assert isinstance(node.unlocks_buildings, tuple)
        assert isinstance(node.unlocks_districts, tuple)
        assert isinstance(node.unlocks_ship_sizes, tuple)


# ---------------------------------------------------------------------------
# Core Layer: JobDef
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestJobDef:
    def test_frozen(self):
        from stellaris_advisor.gamedata.models import JobDef

        job = JobDef(
            id="researcher",
            name="Researcher",
            category="specialist",
            produces=(_make_output_modifier("physics_research", 5.0),),
            upkeep=(_make_output_modifier("consumer_goods", 1.0),),
        )
        with pytest.raises(AttributeError):
            job.name = "Modified"  # type: ignore[misc]

    def test_defaults(self):
        from stellaris_advisor.gamedata.models import JobDef

        job = JobDef(
            id="researcher",
            name="Researcher",
            category="specialist",
            produces=(),
            upkeep=(),
        )
        assert job.conditions == ()
        assert job.weight == 1.0

    def test_produces_is_tuple(self):
        from stellaris_advisor.gamedata.models import JobDef

        job = JobDef(
            id="researcher",
            name="Researcher",
            category="specialist",
            produces=(_make_output_modifier("physics_research", 5.0),),
            upkeep=(),
        )
        assert isinstance(job.produces, tuple)


# ---------------------------------------------------------------------------
# Core Layer: BuildingDef
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildingDef:
    def test_frozen(self):
        b = _make_building_def()
        with pytest.raises(AttributeError):
            b.name = "Palace"  # type: ignore[misc]

    def test_defaults(self):
        b = _make_building_def()
        assert b.max_per_planet == -1
        assert b.upgrades_from == ""
        assert b.description == ""

    def test_jobs_is_tuple(self):
        b = _make_building_def(jobs=("clerk", "merchant"))
        assert isinstance(b.jobs, tuple)

    def test_prerequisites_is_frozenset(self):
        b = _make_building_def(prerequisites=frozenset({"tech_lasers_1"}))
        assert isinstance(b.prerequisites, frozenset)


# ---------------------------------------------------------------------------
# Core Layer: DistrictDef
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDistrictDef:
    def test_frozen(self):
        d = _make_district_def()
        with pytest.raises(AttributeError):
            d.name = "Modified"  # type: ignore[misc]

    def test_defaults(self):
        d = _make_district_def()
        assert d.max_per_planet == -1
        assert d.description == ""

    def test_planet_classes_is_frozenset(self):
        d = _make_district_def(planet_classes=frozenset({"pc_continental", "pc_ocean"}))
        assert isinstance(d.planet_classes, frozenset)
        assert "pc_continental" in d.planet_classes


# ---------------------------------------------------------------------------
# Core Layer: SpecializationDef
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSpecializationDef:
    def test_frozen(self):
        from stellaris_advisor.gamedata.models import SpecializationDef

        spec = SpecializationDef(
            id="col_farming",
            name="Farming World",
            bonuses=(_make_output_modifier("food", 0.1),),
        )
        with pytest.raises(AttributeError):
            spec.name = "Modified"  # type: ignore[misc]

    def test_defaults(self):
        from stellaris_advisor.gamedata.models import SpecializationDef

        spec = SpecializationDef(
            id="col_farming",
            name="Farming World",
            bonuses=(),
        )
        assert spec.conditions == ()

    def test_bonuses_is_tuple(self):
        from stellaris_advisor.gamedata.models import SpecializationDef

        spec = SpecializationDef(
            id="col_farming",
            name="Farming World",
            bonuses=(_make_output_modifier("food", 0.1),),
        )
        assert isinstance(spec.bonuses, tuple)


# ---------------------------------------------------------------------------
# Core Layer: ShipSectionDef
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestShipSectionDef:
    def test_frozen(self):
        from stellaris_advisor.gamedata.models import ShipSectionDef

        section = ShipSectionDef(
            id="corvette_bow",
            name="Corvette Bow",
            slot_types=("S", "S"),
            component_slots=2,
        )
        with pytest.raises(AttributeError):
            section.name = "Modified"  # type: ignore[misc]

    def test_slot_types_is_tuple(self):
        from stellaris_advisor.gamedata.models import ShipSectionDef

        section = ShipSectionDef(
            id="corvette_bow",
            name="Corvette Bow",
            slot_types=("S", "M", "L"),
            component_slots=3,
        )
        assert isinstance(section.slot_types, tuple)


# ---------------------------------------------------------------------------
# Core Layer: ShipClassDef
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestShipClassDef:
    def test_frozen(self):
        from stellaris_advisor.gamedata.models import ShipClassDef

        ship = ShipClassDef(
            id="corvette",
            name="Corvette",
            hull=100,
            evasion=0.6,
            speed=2.0,
            sections=(),
            base_cost=(_make_output_modifier("alloys", 40.0),),
            upkeep=(_make_output_modifier("energy", 1.0),),
        )
        with pytest.raises(AttributeError):
            ship.name = "Modified"  # type: ignore[misc]

    def test_defaults(self):
        from stellaris_advisor.gamedata.models import ShipClassDef

        ship = ShipClassDef(
            id="corvette",
            name="Corvette",
            hull=100,
            evasion=0.6,
            speed=2.0,
            sections=(),
            base_cost=(),
            upkeep=(),
        )
        assert ship.prerequisites == frozenset()

    def test_prerequisites_is_frozenset(self):
        from stellaris_advisor.gamedata.models import ShipClassDef

        ship = ShipClassDef(
            id="destroyer",
            name="Destroyer",
            hull=200,
            evasion=0.3,
            speed=1.5,
            sections=(),
            base_cost=(),
            upkeep=(),
            prerequisites=frozenset({"tech_destroyers"}),
        )
        assert isinstance(ship.prerequisites, frozenset)


# ---------------------------------------------------------------------------
# Core Layer: PlanetClassDef
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPlanetClassDef:
    def test_frozen(self):
        from stellaris_advisor.gamedata.models import PlanetClassDef

        pc = PlanetClassDef(
            id="pc_continental",
            name="Continental",
            habitability_category="continental",
            district_cap=20,
            base_habitability=1.0,
            modifiers=(),
        )
        with pytest.raises(AttributeError):
            pc.name = "Modified"  # type: ignore[misc]

    def test_modifiers_is_tuple(self):
        from stellaris_advisor.gamedata.models import PlanetClassDef

        pc = PlanetClassDef(
            id="pc_arid",
            name="Arid",
            habitability_category="arid",
            district_cap=18,
            base_habitability=0.8,
            modifiers=(_make_output_modifier("food", -0.1),),
        )
        assert isinstance(pc.modifiers, tuple)


# ---------------------------------------------------------------------------
# Meta Layer: TraditionTree
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTraditionTree:
    def test_frozen(self):
        from stellaris_advisor.gamedata.models import TraditionTree

        tree = TraditionTree(
            id="tr_expansion",
            name="Expansion",
            tradition_ids=("tr_expansion_adopt", "tr_expansion_finish"),
            adopt_effect="",
            finish_effect="",
        )
        with pytest.raises(AttributeError):
            tree.name = "Modified"  # type: ignore[misc]

    def test_defaults(self):
        from stellaris_advisor.gamedata.models import TraditionTree

        tree = TraditionTree(
            id="tr_expansion",
            name="Expansion",
            tradition_ids=(),
            adopt_effect="",
            finish_effect="",
        )
        assert tree.prerequisites == frozenset()

    def test_tradition_ids_is_tuple(self):
        from stellaris_advisor.gamedata.models import TraditionTree

        tree = TraditionTree(
            id="tr_expansion",
            name="Expansion",
            tradition_ids=("tr_expansion_adopt",),
            adopt_effect="",
            finish_effect="",
        )
        assert isinstance(tree.tradition_ids, tuple)


# ---------------------------------------------------------------------------
# Meta Layer: TraditionDef
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTraditionDef:
    def test_frozen(self):
        from stellaris_advisor.gamedata.models import TraditionDef

        trad = TraditionDef(
            id="tr_expansion_adopt",
            name="Adopt Expansion",
            tree_id="tr_expansion",
            effect="",
        )
        with pytest.raises(AttributeError):
            trad.name = "Modified"  # type: ignore[misc]

    def test_defaults(self):
        from stellaris_advisor.gamedata.models import TraditionDef

        trad = TraditionDef(
            id="tr_expansion_adopt",
            name="Adopt Expansion",
            tree_id="tr_expansion",
            effect="",
        )
        assert trad.prerequisites == frozenset()
        assert trad.cost == 0

    def test_prerequisites_is_frozenset(self):
        from stellaris_advisor.gamedata.models import TraditionDef

        trad = TraditionDef(
            id="tr_expansion_finish",
            name="Finish Expansion",
            tree_id="tr_expansion",
            prerequisites=frozenset({"tr_expansion_adopt"}),
            effect="",
        )
        assert isinstance(trad.prerequisites, frozenset)


# ---------------------------------------------------------------------------
# Meta Layer: AscensionPerkDef
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAscensionPerkDef:
    def test_frozen(self):
        from stellaris_advisor.gamedata.models import AscensionPerkDef

        ap = AscensionPerkDef(
            id="ap_mind_over_matter",
            name="Mind Over Matter",
            description="",
            effect="",
        )
        with pytest.raises(AttributeError):
            ap.name = "Modified"  # type: ignore[misc]

    def test_defaults(self):
        from stellaris_advisor.gamedata.models import AscensionPerkDef

        ap = AscensionPerkDef(
            id="ap_mind_over_matter",
            name="Mind Over Matter",
            description="",
            effect="",
        )
        assert ap.prerequisites == frozenset()
        assert ap.required_perks == frozenset()
        assert ap.excluded_perks == frozenset()
        assert ap.required_traditions == 0

    def test_excluded_perks_is_frozenset(self):
        from stellaris_advisor.gamedata.models import AscensionPerkDef

        ap = AscensionPerkDef(
            id="ap_synthetic_age",
            name="Synthetic Age",
            description="",
            excluded_perks=frozenset({"ap_flesh_is_weak", "ap_mind_over_matter"}),
            effect="",
        )
        assert isinstance(ap.excluded_perks, frozenset)
        assert "ap_flesh_is_weak" in ap.excluded_perks


# ---------------------------------------------------------------------------
# Meta Layer: MegastructureStageDef
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMegastructureStageDef:
    def test_frozen(self):
        from stellaris_advisor.gamedata.models import MegastructureStageDef

        stage = MegastructureStageDef(
            stage=1,
            cost=(_make_output_modifier("alloys", 1000.0),),
            build_time=360,
        )
        with pytest.raises(AttributeError):
            stage.stage = 2  # type: ignore[misc]

    def test_cost_is_tuple(self):
        from stellaris_advisor.gamedata.models import MegastructureStageDef

        stage = MegastructureStageDef(
            stage=0,
            cost=(
                _make_output_modifier("alloys", 500.0),
                _make_output_modifier("sr_dark_matter", 10.0),
            ),
            build_time=180,
        )
        assert isinstance(stage.cost, tuple)
        assert len(stage.cost) == 2


# ---------------------------------------------------------------------------
# Meta Layer: MegastructureDef
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMegastructureDef:
    def test_frozen(self):
        from stellaris_advisor.gamedata.models import (
            MegastructureDef,
            MegastructureStageDef,
        )

        stage = MegastructureStageDef(stage=1, cost=(), build_time=360)
        mega = MegastructureDef(
            id="megastructure_dyson_sphere",
            name="Dyson Sphere",
            stages=(stage,),
            prerequisites=frozenset({"tech_dyson_sphere"}),
            required_perks=frozenset({"ap_galactic_wonders"}),
            site_requirements=(),
        )
        with pytest.raises(AttributeError):
            mega.name = "Modified"  # type: ignore[misc]

    def test_defaults(self):
        from stellaris_advisor.gamedata.models import MegastructureDef

        mega = MegastructureDef(
            id="megastructure_dyson_sphere",
            name="Dyson Sphere",
            stages=(),
            prerequisites=frozenset(),
            required_perks=frozenset(),
            site_requirements=(),
        )
        assert mega.description == ""

    def test_prerequisites_is_frozenset(self):
        from stellaris_advisor.gamedata.models import MegastructureDef

        mega = MegastructureDef(
            id="mega_ring_world",
            name="Ring World",
            stages=(),
            prerequisites=frozenset({"tech_mega_engineering"}),
            required_perks=frozenset({"ap_galactic_wonders"}),
            site_requirements=(),
        )
        assert isinstance(mega.prerequisites, frozenset)


# ---------------------------------------------------------------------------
# Political Layer: PolicyOptionDef
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPolicyOptionDef:
    def test_frozen(self):
        from stellaris_advisor.gamedata.models import PolicyOptionDef

        opt = PolicyOptionDef(
            id="pop_controls_allowed",
            name="Population Controls Allowed",
            effects=(_make_output_modifier("pop_growth_speed", -0.1),),
        )
        with pytest.raises(AttributeError):
            opt.name = "Modified"  # type: ignore[misc]

    def test_defaults(self):
        from stellaris_advisor.gamedata.models import PolicyOptionDef

        opt = PolicyOptionDef(
            id="pop_controls_allowed",
            name="Population Controls Allowed",
            effects=(),
        )
        assert opt.faction_effects == ()
        assert opt.conditions == ()

    def test_faction_effects_nested_tuple(self):
        from stellaris_advisor.gamedata.models import PolicyOptionDef

        opt = PolicyOptionDef(
            id="pop_controls_allowed",
            name="Population Controls Allowed",
            effects=(),
            faction_effects=(("supremacist", -10.0), ("egalitarian", 5.0)),
        )
        assert isinstance(opt.faction_effects, tuple)
        assert opt.faction_effects[0] == ("supremacist", -10.0)


# ---------------------------------------------------------------------------
# Political Layer: PolicyDef
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPolicyDef:
    def test_frozen(self):
        from stellaris_advisor.gamedata.models import PolicyDef, PolicyOptionDef

        policy = PolicyDef(
            id="policy_pop_controls",
            name="Population Controls",
            category="population",
            options=(
                PolicyOptionDef(id="pop_controls_allowed", name="Allowed", effects=()),
            ),
        )
        with pytest.raises(AttributeError):
            policy.name = "Modified"  # type: ignore[misc]

    def test_options_is_tuple(self):
        from stellaris_advisor.gamedata.models import PolicyDef, PolicyOptionDef

        policy = PolicyDef(
            id="policy_pop_controls",
            name="Population Controls",
            category="population",
            options=(
                PolicyOptionDef(id="opt_a", name="A", effects=()),
                PolicyOptionDef(id="opt_b", name="B", effects=()),
            ),
        )
        assert isinstance(policy.options, tuple)
        assert len(policy.options) == 2


# ---------------------------------------------------------------------------
# Political Layer: FactionIssueDef
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFactionIssueDef:
    def test_frozen(self):
        from stellaris_advisor.gamedata.models import FactionIssueDef

        issue = FactionIssueDef(
            id="issue_migration_controls",
            name="Migration Controls",
            approval_effect=10.0,
            policy_id="policy_migration_policy",
            desired_option="migration_controls_allowed",
        )
        with pytest.raises(AttributeError):
            issue.name = "Modified"  # type: ignore[misc]

    def test_fields(self):
        from stellaris_advisor.gamedata.models import FactionIssueDef

        issue = FactionIssueDef(
            id="issue_migration_controls",
            name="Migration Controls",
            approval_effect=10.0,
            policy_id="policy_migration_policy",
            desired_option="migration_controls_allowed",
        )
        assert issue.approval_effect == 10.0
        assert issue.policy_id == "policy_migration_policy"


# ---------------------------------------------------------------------------
# Political Layer: FactionDef
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFactionDef:
    def test_frozen(self):
        from stellaris_advisor.gamedata.models import FactionDef

        faction = FactionDef(
            id="faction_egalitarian",
            name="Egalitarian",
            ethic="ethic_egalitarian",
            issues=(),
        )
        with pytest.raises(AttributeError):
            faction.name = "Modified"  # type: ignore[misc]

    def test_defaults(self):
        from stellaris_advisor.gamedata.models import FactionDef

        faction = FactionDef(
            id="faction_egalitarian",
            name="Egalitarian",
            ethic="ethic_egalitarian",
            issues=(),
        )
        assert faction.happiness_effects == ()

    def test_issues_is_tuple(self):
        from stellaris_advisor.gamedata.models import FactionDef, FactionIssueDef

        issue = FactionIssueDef(
            id="issue_a",
            name="Issue A",
            approval_effect=5.0,
            policy_id="policy_a",
            desired_option="option_a",
        )
        faction = FactionDef(
            id="faction_egalitarian",
            name="Egalitarian",
            ethic="ethic_egalitarian",
            issues=(issue,),
        )
        assert isinstance(faction.issues, tuple)


# ---------------------------------------------------------------------------
# Political Layer: CouncilPositionDef
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCouncilPositionDef:
    def test_frozen(self):
        from stellaris_advisor.gamedata.models import CouncilPositionDef

        cp = CouncilPositionDef(
            id="council_position_economic",
            name="Economic Advisor",
            leader_class="official",
            effects=(),
        )
        with pytest.raises(AttributeError):
            cp.name = "Modified"  # type: ignore[misc]

    def test_defaults(self):
        from stellaris_advisor.gamedata.models import CouncilPositionDef

        cp = CouncilPositionDef(
            id="council_position_economic",
            name="Economic Advisor",
            leader_class="official",
            effects=(),
        )
        assert cp.max_positions == 1

    def test_effects_is_tuple(self):
        from stellaris_advisor.gamedata.models import CouncilPositionDef

        cp = CouncilPositionDef(
            id="council_position_economic",
            name="Economic Advisor",
            leader_class="official",
            effects=(_make_output_modifier("energy", 0.1),),
        )
        assert isinstance(cp.effects, tuple)


# ---------------------------------------------------------------------------
# Political Layer: AgendaDef
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAgendaDef:
    def test_frozen(self):
        from stellaris_advisor.gamedata.models import AgendaDef

        agenda = AgendaDef(
            id="agenda_economic_expansion",
            name="Economic Expansion",
            effects=(_make_output_modifier("energy", 0.1),),
        )
        with pytest.raises(AttributeError):
            agenda.name = "Modified"  # type: ignore[misc]

    def test_defaults(self):
        from stellaris_advisor.gamedata.models import AgendaDef

        agenda = AgendaDef(
            id="agenda_economic_expansion",
            name="Economic Expansion",
            effects=(),
        )
        assert agenda.prerequisites == frozenset()
        assert agenda.conditions == ()

    def test_prerequisites_is_frozenset(self):
        from stellaris_advisor.gamedata.models import AgendaDef

        agenda = AgendaDef(
            id="agenda_economic_expansion",
            name="Economic Expansion",
            effects=(),
            prerequisites=frozenset({"tech_lasers_1"}),
        )
        assert isinstance(agenda.prerequisites, frozenset)


# ---------------------------------------------------------------------------
# Political Layer: FederationTypeDef
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFederationTypeDef:
    def test_frozen(self):
        from stellaris_advisor.gamedata.models import FederationTypeDef

        fed_type = FederationTypeDef(
            id="federation_default",
            name="Federation",
            perks=("fp_free_migration",),
            max_level=10,
            challenge_threshold=0.5,
        )
        with pytest.raises(AttributeError):
            fed_type.name = "Modified"  # type: ignore[misc]

    def test_perks_is_tuple(self):
        from stellaris_advisor.gamedata.models import FederationTypeDef

        fed_type = FederationTypeDef(
            id="federation_default",
            name="Federation",
            perks=("fp_free_migration", "fp_increase_cohesion_speed"),
            max_level=10,
            challenge_threshold=0.5,
        )
        assert isinstance(fed_type.perks, tuple)


# ---------------------------------------------------------------------------
# Supporting Layer: CivicDef
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCivicDef:
    def test_frozen(self):
        from stellaris_advisor.gamedata.models import CivicDef

        civic = CivicDef(
            id="civic_agrarian_idyll",
            name="Agrarian Idyll",
            description="",
            effects=(),
        )
        with pytest.raises(AttributeError):
            civic.name = "Modified"  # type: ignore[misc]

    def test_defaults(self):
        from stellaris_advisor.gamedata.models import CivicDef

        civic = CivicDef(
            id="civic_agrarian_idyll",
            name="Agrarian Idyll",
            description="",
            effects=(),
        )
        assert civic.prerequisites == frozenset()
        assert civic.excluded_civics == frozenset()
        assert civic.authority_types == frozenset()

    def test_excluded_civics_is_frozenset(self):
        from stellaris_advisor.gamedata.models import CivicDef

        civic = CivicDef(
            id="civic_agrarian_idyll",
            name="Agrarian Idyll",
            description="",
            effects=(),
            excluded_civics=frozenset({"civic_industrial_world"}),
        )
        assert isinstance(civic.excluded_civics, frozenset)


# ---------------------------------------------------------------------------
# Supporting Layer: TraitDef
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTraitDef:
    def test_frozen(self):
        from stellaris_advisor.gamedata.models import TraitDef

        trait = TraitDef(
            id="trait_intelligent",
            name="Intelligent",
            cost=2,
            effects=(_make_output_modifier("research_speed", 0.1),),
            category="normal",
        )
        with pytest.raises(AttributeError):
            trait.name = "Modified"  # type: ignore[misc]

    def test_defaults(self):
        from stellaris_advisor.gamedata.models import TraitDef

        trait = TraitDef(
            id="trait_intelligent",
            name="Intelligent",
            cost=2,
            effects=(),
            category="normal",
        )
        assert trait.opposites == frozenset()

    def test_opposites_is_frozenset(self):
        from stellaris_advisor.gamedata.models import TraitDef

        trait = TraitDef(
            id="trait_intelligent",
            name="Intelligent",
            cost=2,
            effects=(),
            opposites=frozenset({"trait_simple_minded"}),
            category="normal",
        )
        assert isinstance(trait.opposites, frozenset)


# ---------------------------------------------------------------------------
# Supporting Layer: EdictDef
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEdictDef:
    def test_frozen(self):
        from stellaris_advisor.gamedata.models import EdictDef

        edict = EdictDef(
            id="edict_industrial_development",
            name="Industrial Development",
            cost=(_make_output_modifier("energy", 100.0),),
            upkeep=(_make_output_modifier("energy", 5.0),),
            effects=(_make_output_modifier("mineral_production", 0.1),),
        )
        with pytest.raises(AttributeError):
            edict.name = "Modified"  # type: ignore[misc]

    def test_defaults(self):
        from stellaris_advisor.gamedata.models import EdictDef

        edict = EdictDef(
            id="edict_industrial_development",
            name="Industrial Development",
            cost=(),
            upkeep=(),
            effects=(),
        )
        assert edict.prerequisites == frozenset()
        assert edict.duration == -1

    def test_cost_is_tuple(self):
        from stellaris_advisor.gamedata.models import EdictDef

        edict = EdictDef(
            id="edict_industrial_development",
            name="Industrial Development",
            cost=(_make_output_modifier("energy", 100.0),),
            upkeep=(),
            effects=(),
        )
        assert isinstance(edict.cost, tuple)
