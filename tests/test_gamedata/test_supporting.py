"""Tests for council.py, federations.py, empires.py — TDD.

Covers:
  council:     load_council_defs, load_agenda_defs,
               optimal_council_assignment, agenda_prerequisites_met
  federations: load_federation_type_defs
  empires:     load_civic_defs, load_trait_defs, load_edict_defs
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "game_data"
_COMMON_DIR = _FIXTURE_DIR / "common"


# ---------------------------------------------------------------------------
# Raw-dict helpers
# ---------------------------------------------------------------------------


def _raw_council_pos(
    pid: str,
    leader_class: str = "scientist",
    modifier: dict[str, Any] | None = None,
    max_positions: int = 1,
) -> dict[str, Any]:
    return {
        "name": pid,
        "leader_class": leader_class,
        "modifier": modifier or {},
        "max_council_positions": max_positions,
    }


def _raw_agenda(
    aid: str,
    effect: dict[str, Any] | None = None,
    prerequisites: list[str] | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "name": aid,
        "effect": effect or {},
        "monthly_progress": {"base": 2},
    }
    if prerequisites:
        data["prerequisites"] = prerequisites
    return data


def _raw_fed_type(
    fid: str,
    perk_names: list[str] | None = None,
    max_level: int = 5,
    challenge_threshold: float = 0.5,
) -> dict[str, Any]:
    perks = perk_names or []
    perk_dicts: Any
    if len(perks) == 0:
        perk_dicts = None
    elif len(perks) == 1:
        perk_dicts = {"name": perks[0], "level": 1, "modifier": {}}
    else:
        perk_dicts = [{"name": p, "level": i + 1, "modifier": {}} for i, p in enumerate(perks)]
    data: dict[str, Any] = {
        "name": fid,
        "max_level": max_level,
        "challenge_threshold": challenge_threshold,
    }
    if perk_dicts is not None:
        data["perk"] = perk_dicts
    return data


def _raw_civic(
    cid: str,
    modifier: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": cid,
        "description": f"{cid}_desc",
        "playable": {"always": True},
        "modifier": modifier or {},
    }


def _raw_trait(
    tid: str,
    cost: int = 1,
    category: str = "normal",
    modifier: dict[str, Any] | None = None,
    opposites: list[str] | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "cost": cost,
        "category": category,
        "modifier": modifier or {},
    }
    if opposites is not None:
        data["opposites"] = opposites if len(opposites) > 1 else opposites[0]
    return data


def _raw_edict(
    eid: str,
    cost: dict[str, Any] | None = None,
    upkeep: dict[str, Any] | None = None,
    modifier: dict[str, Any] | None = None,
    length: int = 3600,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "name": eid,
        "length": length,
        "edict_type": "country",
        "potential": {"always": True},
        "allow": {"always": True},
        "modifier": modifier or {},
    }
    if cost:
        data["cost"] = cost
    if upkeep:
        data["upkeep"] = upkeep
    return data


# ===========================================================================
# load_council_defs
# ===========================================================================


@pytest.mark.unit
class TestLoadCouncilDefs:
    def test_basic_fields(self):
        from stellaris_advisor.gamedata.council import load_council_defs

        raw = {"council_position_scientist": _raw_council_pos("council_position_scientist")}
        defs = load_council_defs(raw)
        assert "council_position_scientist" in defs
        assert defs["council_position_scientist"].id == "council_position_scientist"
        assert defs["council_position_scientist"].leader_class == "scientist"

    def test_effects_from_modifier(self):
        from stellaris_advisor.gamedata.council import load_council_defs

        raw = {
            "pos": _raw_council_pos(
                "pos", modifier={"all_technology_research_speed": 0.05}
            )
        }
        defs = load_council_defs(raw)
        effects = {m.resource: m.amount for m in defs["pos"].effects}
        assert effects["all_technology_research_speed"] == pytest.approx(0.05)

    def test_max_positions(self):
        from stellaris_advisor.gamedata.council import load_council_defs

        raw = {"pos": _raw_council_pos("pos", max_positions=2)}
        defs = load_council_defs(raw)
        assert defs["pos"].max_positions == 2

    def test_max_positions_defaults_to_one(self):
        from stellaris_advisor.gamedata.council import load_council_defs

        raw = {"pos": {"name": "pos", "leader_class": "admiral"}}
        defs = load_council_defs(raw)
        assert defs["pos"].max_positions == 1

    def test_frozen_dataclass(self):
        from stellaris_advisor.gamedata.council import load_council_defs

        defs = load_council_defs({"p": _raw_council_pos("p")})
        with pytest.raises(AttributeError):
            defs["p"].leader_class = "changed"  # type: ignore[misc]

    def test_empty_raw_returns_empty(self):
        from stellaris_advisor.gamedata.council import load_council_defs

        assert load_council_defs({}) == {}


# ===========================================================================
# load_agenda_defs
# ===========================================================================


@pytest.mark.unit
class TestLoadAgendaDefs:
    def test_basic_fields(self):
        from stellaris_advisor.gamedata.council import load_agenda_defs

        raw = {"agenda_x": _raw_agenda("agenda_x")}
        defs = load_agenda_defs(raw)
        assert "agenda_x" in defs
        assert defs["agenda_x"].id == "agenda_x"

    def test_effects_from_effect_block(self):
        from stellaris_advisor.gamedata.council import load_agenda_defs

        raw = {
            "agenda_x": _raw_agenda(
                "agenda_x", effect={"all_technology_research_speed": 0.10}
            )
        }
        defs = load_agenda_defs(raw)
        effects = {m.resource: m.amount for m in defs["agenda_x"].effects}
        assert effects["all_technology_research_speed"] == pytest.approx(0.10)

    def test_prerequisites_extracted(self):
        from stellaris_advisor.gamedata.council import load_agenda_defs

        raw = {"agenda_x": _raw_agenda("agenda_x", prerequisites=["tech_a"])}
        defs = load_agenda_defs(raw)
        assert defs["agenda_x"].prerequisites == frozenset({"tech_a"})

    def test_no_prerequisites_defaults_empty(self):
        from stellaris_advisor.gamedata.council import load_agenda_defs

        raw = {"agenda_x": _raw_agenda("agenda_x")}
        defs = load_agenda_defs(raw)
        assert defs["agenda_x"].prerequisites == frozenset()

    def test_frozen_dataclass(self):
        from stellaris_advisor.gamedata.council import load_agenda_defs

        defs = load_agenda_defs({"a": _raw_agenda("a")})
        with pytest.raises(AttributeError):
            defs["a"].name = "changed"  # type: ignore[misc]

    def test_empty_raw_returns_empty(self):
        from stellaris_advisor.gamedata.council import load_agenda_defs

        assert load_agenda_defs({}) == {}


# ===========================================================================
# optimal_council_assignment
# ===========================================================================


@pytest.mark.unit
class TestOptimalCouncilAssignment:
    def _make_defs(self):
        from stellaris_advisor.gamedata.council import load_council_defs

        return load_council_defs(
            {
                "pos_high": _raw_council_pos(
                    "pos_high", modifier={"all_technology_research_speed": 0.10}
                ),
                "pos_low": _raw_council_pos(
                    "pos_low", modifier={"country_border_friction_mult": 0.01}
                ),
                "pos_mid": _raw_council_pos(
                    "pos_mid", modifier={"ship_fire_rate_mult": 0.05}
                ),
            }
        )

    def test_returns_tuple_of_position_ids(self):
        from stellaris_advisor.gamedata.council import optimal_council_assignment

        defs = self._make_defs()
        result = optimal_council_assignment(defs)
        assert isinstance(result, tuple)
        assert set(result) == {"pos_high", "pos_low", "pos_mid"}

    def test_highest_magnitude_first_with_no_target(self):
        from stellaris_advisor.gamedata.council import optimal_council_assignment

        defs = self._make_defs()
        result = optimal_council_assignment(defs)
        assert result[0] == "pos_high"

    def test_target_effects_prioritizes_relevant_positions(self):
        from stellaris_advisor.gamedata.council import optimal_council_assignment

        defs = self._make_defs()
        result = optimal_council_assignment(
            defs, target_effects=frozenset({"ship_fire_rate_mult"})
        )
        assert result[0] == "pos_mid"


# ===========================================================================
# agenda_prerequisites_met
# ===========================================================================


@pytest.mark.unit
class TestAgendaPrerequisitesMet:
    def test_no_prerequisites_always_true(self):
        from stellaris_advisor.gamedata.council import (
            agenda_prerequisites_met,
            load_agenda_defs,
        )

        defs = load_agenda_defs({"a": _raw_agenda("a")})
        assert agenda_prerequisites_met(defs["a"], frozenset()) is True

    def test_missing_prerequisite_false(self):
        from stellaris_advisor.gamedata.council import (
            agenda_prerequisites_met,
            load_agenda_defs,
        )

        defs = load_agenda_defs({"a": _raw_agenda("a", prerequisites=["tech_required"])})
        assert agenda_prerequisites_met(defs["a"], frozenset()) is False

    def test_all_prerequisites_met_true(self):
        from stellaris_advisor.gamedata.council import (
            agenda_prerequisites_met,
            load_agenda_defs,
        )

        defs = load_agenda_defs({"a": _raw_agenda("a", prerequisites=["tech_a", "tech_b"])})
        assert agenda_prerequisites_met(defs["a"], frozenset({"tech_a", "tech_b"})) is True


# ===========================================================================
# load_federation_type_defs
# ===========================================================================


@pytest.mark.unit
class TestLoadFederationTypeDefs:
    def test_basic_fields(self):
        from stellaris_advisor.gamedata.federations import load_federation_type_defs

        raw = {
            "default_federation": _raw_fed_type(
                "fed_default", perk_names=["fp_a", "fp_b"], max_level=10
            )
        }
        defs = load_federation_type_defs(raw)
        assert "default_federation" in defs
        assert defs["default_federation"].id == "default_federation"

    def test_max_level_extracted(self):
        from stellaris_advisor.gamedata.federations import load_federation_type_defs

        raw = {"fed_x": _raw_fed_type("fed_x", max_level=7)}
        defs = load_federation_type_defs(raw)
        assert defs["fed_x"].max_level == 7

    def test_challenge_threshold_extracted(self):
        from stellaris_advisor.gamedata.federations import load_federation_type_defs

        raw = {"fed_x": _raw_fed_type("fed_x", challenge_threshold=0.40)}
        defs = load_federation_type_defs(raw)
        assert defs["fed_x"].challenge_threshold == pytest.approx(0.40)

    def test_perks_extracted_from_list(self):
        from stellaris_advisor.gamedata.federations import load_federation_type_defs

        raw = {"fed_x": _raw_fed_type("fed_x", perk_names=["fp_a", "fp_b"])}
        defs = load_federation_type_defs(raw)
        assert "fp_a" in defs["fed_x"].perks
        assert "fp_b" in defs["fed_x"].perks

    def test_single_perk_as_dict_handled(self):
        from stellaris_advisor.gamedata.federations import load_federation_type_defs

        raw = {"fed_x": _raw_fed_type("fed_x", perk_names=["only_perk"])}
        defs = load_federation_type_defs(raw)
        assert "only_perk" in defs["fed_x"].perks

    def test_frozen_dataclass(self):
        from stellaris_advisor.gamedata.federations import load_federation_type_defs

        defs = load_federation_type_defs({"f": _raw_fed_type("f")})
        with pytest.raises(AttributeError):
            defs["f"].max_level = 99  # type: ignore[misc]

    def test_empty_raw_returns_empty(self):
        from stellaris_advisor.gamedata.federations import load_federation_type_defs

        assert load_federation_type_defs({}) == {}


# ===========================================================================
# load_civic_defs
# ===========================================================================


@pytest.mark.unit
class TestLoadCivicDefs:
    def test_basic_fields(self):
        from stellaris_advisor.gamedata.empires import load_civic_defs

        raw = {"civic_warrior": _raw_civic("civic_warrior")}
        defs = load_civic_defs(raw)
        assert "civic_warrior" in defs
        assert defs["civic_warrior"].id == "civic_warrior"
        assert defs["civic_warrior"].description == "civic_warrior_desc"

    def test_effects_from_modifier(self):
        from stellaris_advisor.gamedata.empires import load_civic_defs

        raw = {"civic_x": _raw_civic("civic_x", modifier={"ship_fire_rate_mult": 0.05})}
        defs = load_civic_defs(raw)
        effects = {m.resource: m.amount for m in defs["civic_x"].effects}
        assert effects["ship_fire_rate_mult"] == pytest.approx(0.05)

    def test_prerequisites_default_empty(self):
        from stellaris_advisor.gamedata.empires import load_civic_defs

        defs = load_civic_defs({"c": _raw_civic("c")})
        assert defs["c"].prerequisites == frozenset()

    def test_frozen_dataclass(self):
        from stellaris_advisor.gamedata.empires import load_civic_defs

        defs = load_civic_defs({"c": _raw_civic("c")})
        with pytest.raises(AttributeError):
            defs["c"].name = "changed"  # type: ignore[misc]

    def test_empty_raw_returns_empty(self):
        from stellaris_advisor.gamedata.empires import load_civic_defs

        assert load_civic_defs({}) == {}


# ===========================================================================
# load_trait_defs
# ===========================================================================


@pytest.mark.unit
class TestLoadTraitDefs:
    def test_basic_fields(self):
        from stellaris_advisor.gamedata.empires import load_trait_defs

        raw = {"trait_intelligent": _raw_trait("trait_intelligent", cost=2, category="normal")}
        defs = load_trait_defs(raw)
        assert "trait_intelligent" in defs
        assert defs["trait_intelligent"].id == "trait_intelligent"
        assert defs["trait_intelligent"].cost == 2
        assert defs["trait_intelligent"].category == "normal"

    def test_effects_from_modifier(self):
        from stellaris_advisor.gamedata.empires import load_trait_defs

        raw = {"t": _raw_trait("t", modifier={"research_speed_factor": 0.10})}
        defs = load_trait_defs(raw)
        effects = {m.resource: m.amount for m in defs["t"].effects}
        assert effects["research_speed_factor"] == pytest.approx(0.10)

    def test_opposites_as_single_string(self):
        from stellaris_advisor.gamedata.empires import load_trait_defs

        raw = {"t_a": _raw_trait("t_a", opposites=["t_b"])}
        defs = load_trait_defs(raw)
        assert defs["t_a"].opposites == frozenset({"t_b"})

    def test_opposites_as_list(self):
        from stellaris_advisor.gamedata.empires import load_trait_defs

        raw = {"t_a": _raw_trait("t_a", opposites=["t_b", "t_c"])}
        defs = load_trait_defs(raw)
        assert defs["t_a"].opposites == frozenset({"t_b", "t_c"})

    def test_negative_cost(self):
        from stellaris_advisor.gamedata.empires import load_trait_defs

        raw = {"t": _raw_trait("t", cost=-2)}
        defs = load_trait_defs(raw)
        assert defs["t"].cost == -2

    def test_frozen_dataclass(self):
        from stellaris_advisor.gamedata.empires import load_trait_defs

        defs = load_trait_defs({"t": _raw_trait("t")})
        with pytest.raises(AttributeError):
            defs["t"].cost = 99  # type: ignore[misc]

    def test_empty_raw_returns_empty(self):
        from stellaris_advisor.gamedata.empires import load_trait_defs

        assert load_trait_defs({}) == {}


# ===========================================================================
# load_edict_defs
# ===========================================================================


@pytest.mark.unit
class TestLoadEdictDefs:
    def test_basic_fields(self):
        from stellaris_advisor.gamedata.empires import load_edict_defs

        raw = {"edict_dev": _raw_edict("edict_dev")}
        defs = load_edict_defs(raw)
        assert "edict_dev" in defs
        assert defs["edict_dev"].id == "edict_dev"

    def test_duration_from_length(self):
        from stellaris_advisor.gamedata.empires import load_edict_defs

        raw = {"edict_x": _raw_edict("edict_x", length=3600)}
        defs = load_edict_defs(raw)
        assert defs["edict_x"].duration == 3600

    def test_permanent_duration(self):
        from stellaris_advisor.gamedata.empires import load_edict_defs

        raw = {"edict_x": _raw_edict("edict_x", length=-1)}
        defs = load_edict_defs(raw)
        assert defs["edict_x"].duration == -1

    def test_cost_extracted(self):
        from stellaris_advisor.gamedata.empires import load_edict_defs

        raw = {"edict_x": _raw_edict("edict_x", cost={"energy": 500})}
        defs = load_edict_defs(raw)
        costs = {m.resource: m.amount for m in defs["edict_x"].cost}
        assert costs["energy"] == pytest.approx(500.0)

    def test_upkeep_extracted(self):
        from stellaris_advisor.gamedata.empires import load_edict_defs

        raw = {"edict_x": _raw_edict("edict_x", upkeep={"energy": 5})}
        defs = load_edict_defs(raw)
        upkeep = {m.resource: m.amount for m in defs["edict_x"].upkeep}
        assert upkeep["energy"] == pytest.approx(5.0)

    def test_effects_from_modifier(self):
        from stellaris_advisor.gamedata.empires import load_edict_defs

        raw = {"edict_x": _raw_edict("edict_x", modifier={"pop_growth_speed": 0.15})}
        defs = load_edict_defs(raw)
        effects = {m.resource: m.amount for m in defs["edict_x"].effects}
        assert effects["pop_growth_speed"] == pytest.approx(0.15)

    def test_frozen_dataclass(self):
        from stellaris_advisor.gamedata.empires import load_edict_defs

        defs = load_edict_defs({"e": _raw_edict("e")})
        with pytest.raises(AttributeError):
            defs["e"].duration = 99  # type: ignore[misc]

    def test_empty_raw_returns_empty(self):
        from stellaris_advisor.gamedata.empires import load_edict_defs

        assert load_edict_defs({}) == {}


# ===========================================================================
# Integration — council positions + agendas
# ===========================================================================


@pytest.mark.integration
class TestCouncilDefsFromFixtures:
    @pytest.fixture
    def council_raw(self):
        from stellaris_advisor.parser.clausewitz import parse

        return parse(
            (_COMMON_DIR / "council_positions" / "00_council_positions.txt").read_text()
        )

    @pytest.fixture
    def agenda_raw(self):
        from stellaris_advisor.parser.clausewitz import parse

        return parse((_COMMON_DIR / "agendas" / "00_agendas.txt").read_text())

    def test_loads_all_council_positions(self, council_raw):
        from stellaris_advisor.gamedata.council import load_council_defs

        defs = load_council_defs(council_raw)
        for pid in (
            "council_position_scientist",
            "council_position_admiral",
            "council_position_diplomat",
        ):
            assert pid in defs

    def test_scientist_leader_class(self, council_raw):
        from stellaris_advisor.gamedata.council import load_council_defs

        defs = load_council_defs(council_raw)
        assert defs["council_position_scientist"].leader_class == "scientist"

    def test_scientist_research_effect(self, council_raw):
        from stellaris_advisor.gamedata.council import load_council_defs

        defs = load_council_defs(council_raw)
        effects = {m.resource: m.amount for m in defs["council_position_scientist"].effects}
        assert effects["all_technology_research_speed"] == pytest.approx(0.05)

    def test_loads_all_agendas(self, agenda_raw):
        from stellaris_advisor.gamedata.council import load_agenda_defs

        defs = load_agenda_defs(agenda_raw)
        for aid in (
            "agenda_diplomatic_corps",
            "agenda_scientific_revolution",
            "agenda_militarization",
        ):
            assert aid in defs

    def test_agenda_effects(self, agenda_raw):
        from stellaris_advisor.gamedata.council import load_agenda_defs

        defs = load_agenda_defs(agenda_raw)
        effects = {m.resource: m.amount for m in defs["agenda_scientific_revolution"].effects}
        assert effects["all_technology_research_speed"] == pytest.approx(0.10)


# ===========================================================================
# Integration — federation types
# ===========================================================================


@pytest.mark.integration
class TestFederationTypeDefsFromFixtures:
    @pytest.fixture
    def fed_raw(self):
        from stellaris_advisor.parser.clausewitz import parse

        return parse(
            (_COMMON_DIR / "federation_types" / "00_federation_types.txt").read_text()
        )

    def test_loads_federation_types(self, fed_raw):
        from stellaris_advisor.gamedata.federations import load_federation_type_defs

        defs = load_federation_type_defs(fed_raw)
        assert "default_federation" in defs
        assert "research_cooperative" in defs

    def test_default_federation_max_level(self, fed_raw):
        from stellaris_advisor.gamedata.federations import load_federation_type_defs

        defs = load_federation_type_defs(fed_raw)
        assert defs["default_federation"].max_level == 10

    def test_default_federation_perks(self, fed_raw):
        from stellaris_advisor.gamedata.federations import load_federation_type_defs

        defs = load_federation_type_defs(fed_raw)
        assert "fp_free_migration" in defs["default_federation"].perks
        assert "fp_increase_cohesion_speed" in defs["default_federation"].perks

    def test_challenge_threshold(self, fed_raw):
        from stellaris_advisor.gamedata.federations import load_federation_type_defs

        defs = load_federation_type_defs(fed_raw)
        assert defs["research_cooperative"].challenge_threshold == pytest.approx(0.40)


# ===========================================================================
# Integration — civics + traits + edicts
# ===========================================================================


@pytest.mark.integration
class TestCivicTraitEdictFromFixtures:
    @pytest.fixture
    def civic_raw(self):
        from stellaris_advisor.parser.clausewitz import parse

        return parse(_FIXTURE_DIR.joinpath("civics.txt").read_text())

    @pytest.fixture
    def trait_raw(self):
        from stellaris_advisor.parser.clausewitz import parse

        return parse(_FIXTURE_DIR.joinpath("traits.txt").read_text())

    @pytest.fixture
    def edict_raw(self):
        from stellaris_advisor.parser.clausewitz import parse

        return parse((_COMMON_DIR / "edicts" / "00_edicts.txt").read_text())

    def test_loads_all_civics(self, civic_raw):
        from stellaris_advisor.gamedata.empires import load_civic_defs

        defs = load_civic_defs(civic_raw)
        for cid in (
            "civic_warrior_culture",
            "civic_scientific_directorate",
            "civic_agrarian_idyll",
        ):
            assert cid in defs

    def test_civic_effects(self, civic_raw):
        from stellaris_advisor.gamedata.empires import load_civic_defs

        defs = load_civic_defs(civic_raw)
        effects = {m.resource: m.amount for m in defs["civic_warrior_culture"].effects}
        assert effects["army_damage_mult"] == pytest.approx(0.15)

    def test_loads_all_traits(self, trait_raw):
        from stellaris_advisor.gamedata.empires import load_trait_defs

        defs = load_trait_defs(trait_raw)
        for tid in ("trait_intelligent", "trait_simple_minded", "trait_nomadic"):
            assert tid in defs

    def test_trait_cost_and_opposites(self, trait_raw):
        from stellaris_advisor.gamedata.empires import load_trait_defs

        defs = load_trait_defs(trait_raw)
        assert defs["trait_intelligent"].cost == 2
        assert "trait_simple_minded" in defs["trait_intelligent"].opposites

    def test_trait_negative_cost(self, trait_raw):
        from stellaris_advisor.gamedata.empires import load_trait_defs

        defs = load_trait_defs(trait_raw)
        assert defs["trait_simple_minded"].cost == -2

    def test_loads_all_edicts(self, edict_raw):
        from stellaris_advisor.gamedata.empires import load_edict_defs

        defs = load_edict_defs(edict_raw)
        for eid in (
            "edict_industrial_development",
            "edict_nutritional_plentitude",
            "edict_capacity_subsidies",
        ):
            assert eid in defs

    def test_edict_cost(self, edict_raw):
        from stellaris_advisor.gamedata.empires import load_edict_defs

        defs = load_edict_defs(edict_raw)
        costs = {m.resource: m.amount for m in defs["edict_industrial_development"].cost}
        assert costs["energy"] == pytest.approx(500.0)

    def test_permanent_edict_duration(self, edict_raw):
        from stellaris_advisor.gamedata.empires import load_edict_defs

        defs = load_edict_defs(edict_raw)
        assert defs["edict_capacity_subsidies"].duration == -1

    def test_edict_upkeep(self, edict_raw):
        from stellaris_advisor.gamedata.empires import load_edict_defs

        defs = load_edict_defs(edict_raw)
        upkeep = {m.resource: m.amount for m in defs["edict_capacity_subsidies"].upkeep}
        assert upkeep["energy"] == pytest.approx(5.0)
