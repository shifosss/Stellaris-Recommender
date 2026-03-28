"""Tests for gamedata/ascension_perks.py and gamedata/megastructures.py — TDD.

Covers:
  ascension_perks: load_ascension_perk_defs, eligible_perks, perk_prerequisites_met
  megastructures:  load_megastructure_defs, available_megastructures,
                   megastructure_sites, megastructure_total_cost
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "game_data"


# ---------------------------------------------------------------------------
# Raw-dict helpers
# ---------------------------------------------------------------------------


def _raw_perk(
    pid: str,
    tier: int = 0,
    required_traditions: int = 0,
    required_perks: list[str] | None = None,
    excluded_perks: list[str] | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "tier": tier,
        "modifier": {},
        "possible": {"always": True},
        "on_enabled": {},
        "ai_weight": {"weight": 1},
    }
    if required_traditions:
        data["required_traditions"] = required_traditions
    if required_perks is not None:
        data["required_perks"] = required_perks
    if excluded_perks is not None:
        data["excluded_perks"] = excluded_perks
    return data


def _raw_mega(
    mid: str,
    prerequisites: str | list[str] | None = None,
    required_perks: list[str] | None = None,
    stages: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a raw megastructure dict.

    *stages* keys must be strings ("0", "1", ...) matching parser output.
    When *stages* is None, a single default stage-0 is added.
    """
    data: dict[str, Any] = {"name": mid, "build_outside_gravity_well": False}
    if prerequisites is not None:
        data["prerequisites"] = prerequisites
    if required_perks is not None:
        data["required_perks"] = required_perks
    if stages is not None:
        data.update(stages)
    else:
        data["0"] = {
            "name": f"{mid}_0",
            "construction_days": 1800,
            "resources": {"cost": {"alloys": 2000}},
        }
    return data


# ===========================================================================
# load_ascension_perk_defs
# ===========================================================================


@pytest.mark.unit
class TestLoadAscensionPerkDefs:
    def test_basic_fields(self):
        from stellaris_advisor.gamedata.ascension_perks import load_ascension_perk_defs

        raw = {"ap_one_vision": _raw_perk("ap_one_vision", tier=0)}
        defs = load_ascension_perk_defs(raw)
        assert "ap_one_vision" in defs
        assert defs["ap_one_vision"].id == "ap_one_vision"

    def test_required_traditions_extracted(self):
        from stellaris_advisor.gamedata.ascension_perks import load_ascension_perk_defs

        raw = {"ap_perk": _raw_perk("ap_perk", required_traditions=3)}
        defs = load_ascension_perk_defs(raw)
        assert defs["ap_perk"].required_traditions == 3

    def test_required_traditions_default_zero(self):
        from stellaris_advisor.gamedata.ascension_perks import load_ascension_perk_defs

        raw = {"ap_perk": _raw_perk("ap_perk")}
        defs = load_ascension_perk_defs(raw)
        assert defs["ap_perk"].required_traditions == 0

    def test_required_perks_as_list(self):
        from stellaris_advisor.gamedata.ascension_perks import load_ascension_perk_defs

        raw = {"ap_b": _raw_perk("ap_b", required_perks=["ap_a"])}
        defs = load_ascension_perk_defs(raw)
        assert defs["ap_b"].required_perks == frozenset({"ap_a"})

    def test_excluded_perks_as_list(self):
        from stellaris_advisor.gamedata.ascension_perks import load_ascension_perk_defs

        raw = {"ap_a": _raw_perk("ap_a", excluded_perks=["ap_b", "ap_c"])}
        defs = load_ascension_perk_defs(raw)
        assert defs["ap_a"].excluded_perks == frozenset({"ap_b", "ap_c"})

    def test_no_required_perks_defaults_empty(self):
        from stellaris_advisor.gamedata.ascension_perks import load_ascension_perk_defs

        raw = {"ap_perk": _raw_perk("ap_perk")}
        defs = load_ascension_perk_defs(raw)
        assert defs["ap_perk"].required_perks == frozenset()
        assert defs["ap_perk"].excluded_perks == frozenset()

    def test_frozen_dataclass(self):
        from stellaris_advisor.gamedata.ascension_perks import load_ascension_perk_defs

        defs = load_ascension_perk_defs({"ap_x": _raw_perk("ap_x")})
        with pytest.raises(AttributeError):
            defs["ap_x"].required_traditions = 99  # type: ignore[misc]

    def test_empty_raw_returns_empty(self):
        from stellaris_advisor.gamedata.ascension_perks import load_ascension_perk_defs

        assert load_ascension_perk_defs({}) == {}


# ===========================================================================
# eligible_perks
# ===========================================================================


@pytest.mark.unit
class TestEligiblePerks:
    @pytest.fixture
    def defs(self):
        from stellaris_advisor.gamedata.ascension_perks import load_ascension_perk_defs

        return load_ascension_perk_defs(
            {
                "ap_free": _raw_perk("ap_free"),
                "ap_needs_perk": _raw_perk("ap_needs_perk", required_perks=["ap_free"]),
                "ap_needs_trad": _raw_perk("ap_needs_trad", required_traditions=3),
                "ap_excludes_free": _raw_perk(
                    "ap_excludes_free", excluded_perks=["ap_free"]
                ),
            }
        )

    def test_no_requirements_eligible(self, defs):
        from stellaris_advisor.gamedata.ascension_perks import eligible_perks

        result = eligible_perks(defs, frozenset(), 0, frozenset())
        assert "ap_free" in result

    def test_already_owned_not_eligible(self, defs):
        from stellaris_advisor.gamedata.ascension_perks import eligible_perks

        result = eligible_perks(defs, frozenset(), 0, frozenset({"ap_free"}))
        assert "ap_free" not in result

    def test_missing_required_perk_not_eligible(self, defs):
        from stellaris_advisor.gamedata.ascension_perks import eligible_perks

        result = eligible_perks(defs, frozenset(), 0, frozenset())
        assert "ap_needs_perk" not in result

    def test_required_perk_met_eligible(self, defs):
        from stellaris_advisor.gamedata.ascension_perks import eligible_perks

        result = eligible_perks(defs, frozenset(), 0, frozenset({"ap_free"}))
        assert "ap_needs_perk" in result

    def test_insufficient_traditions_not_eligible(self, defs):
        from stellaris_advisor.gamedata.ascension_perks import eligible_perks

        result = eligible_perks(defs, frozenset(), 2, frozenset())
        assert "ap_needs_trad" not in result

    def test_sufficient_traditions_eligible(self, defs):
        from stellaris_advisor.gamedata.ascension_perks import eligible_perks

        result = eligible_perks(defs, frozenset(), 3, frozenset())
        assert "ap_needs_trad" in result

    def test_excluded_perk_in_existing_not_eligible(self, defs):
        """ap_excludes_free has excluded_perks={ap_free}; owning ap_free blocks it."""
        from stellaris_advisor.gamedata.ascension_perks import eligible_perks

        result = eligible_perks(defs, frozenset(), 0, frozenset({"ap_free"}))
        assert "ap_excludes_free" not in result


# ===========================================================================
# perk_prerequisites_met
# ===========================================================================


@pytest.mark.unit
class TestPerkPrerequisitesMet:
    @pytest.fixture
    def defs(self):
        from stellaris_advisor.gamedata.ascension_perks import load_ascension_perk_defs

        return load_ascension_perk_defs(
            {
                "ap_base": _raw_perk("ap_base"),
                "ap_tier1": _raw_perk(
                    "ap_tier1", required_perks=["ap_base"], required_traditions=2
                ),
            }
        )

    def test_unknown_perk_returns_false(self, defs):
        from stellaris_advisor.gamedata.ascension_perks import perk_prerequisites_met

        assert perk_prerequisites_met(defs, "ap_nonexistent", frozenset(), 0) is False

    def test_no_requirements_returns_true(self, defs):
        from stellaris_advisor.gamedata.ascension_perks import perk_prerequisites_met

        assert perk_prerequisites_met(defs, "ap_base", frozenset(), 0) is True

    def test_missing_required_perk_returns_false(self, defs):
        from stellaris_advisor.gamedata.ascension_perks import perk_prerequisites_met

        assert perk_prerequisites_met(defs, "ap_tier1", frozenset(), 5) is False

    def test_insufficient_traditions_returns_false(self, defs):
        from stellaris_advisor.gamedata.ascension_perks import perk_prerequisites_met

        assert perk_prerequisites_met(defs, "ap_tier1", frozenset({"ap_base"}), 1) is False

    def test_all_requirements_met_returns_true(self, defs):
        from stellaris_advisor.gamedata.ascension_perks import perk_prerequisites_met

        assert perk_prerequisites_met(defs, "ap_tier1", frozenset({"ap_base"}), 2) is True


# ===========================================================================
# load_megastructure_defs
# ===========================================================================


@pytest.mark.unit
class TestLoadMegastructureDefs:
    def test_basic_fields(self):
        from stellaris_advisor.gamedata.megastructures import load_megastructure_defs

        raw = {"mega_x": _raw_mega("mega_x")}
        defs = load_megastructure_defs(raw)
        assert "mega_x" in defs
        assert defs["mega_x"].id == "mega_x"

    def test_prerequisites_as_string(self):
        from stellaris_advisor.gamedata.megastructures import load_megastructure_defs

        raw = {"mega_x": _raw_mega("mega_x", prerequisites="tech_a")}
        defs = load_megastructure_defs(raw)
        assert defs["mega_x"].prerequisites == frozenset({"tech_a"})

    def test_prerequisites_as_list(self):
        from stellaris_advisor.gamedata.megastructures import load_megastructure_defs

        raw = {"mega_x": _raw_mega("mega_x", prerequisites=["tech_a", "tech_b"])}
        defs = load_megastructure_defs(raw)
        assert defs["mega_x"].prerequisites == frozenset({"tech_a", "tech_b"})

    def test_required_perks_extracted(self):
        from stellaris_advisor.gamedata.megastructures import load_megastructure_defs

        raw = {"mega_x": _raw_mega("mega_x", required_perks=["ap_galactic_wonders"])}
        defs = load_megastructure_defs(raw)
        assert defs["mega_x"].required_perks == frozenset({"ap_galactic_wonders"})

    def test_stages_extracted(self):
        from stellaris_advisor.gamedata.megastructures import load_megastructure_defs

        raw = {
            "mega_x": _raw_mega(
                "mega_x",
                stages={
                    "0": {
                        "name": "s0",
                        "construction_days": 1800,
                        "resources": {"cost": {"alloys": 2000}},
                    },
                    "1": {
                        "name": "s1",
                        "construction_days": 3600,
                        "resources": {"cost": {"alloys": 10000}},
                    },
                },
            )
        }
        defs = load_megastructure_defs(raw)
        assert len(defs["mega_x"].stages) == 2

    def test_stage_cost_extracted(self):
        from stellaris_advisor.gamedata.megastructures import load_megastructure_defs

        raw = {
            "mega_x": _raw_mega(
                "mega_x",
                stages={
                    "0": {
                        "name": "s0",
                        "construction_days": 1800,
                        "resources": {"cost": {"alloys": 2000}},
                    }
                },
            )
        }
        defs = load_megastructure_defs(raw)
        costs = {m.resource: m.amount for m in defs["mega_x"].stages[0].cost}
        assert costs["alloys"] == pytest.approx(2000.0)

    def test_stage_build_time_extracted(self):
        from stellaris_advisor.gamedata.megastructures import load_megastructure_defs

        raw = {
            "mega_x": _raw_mega(
                "mega_x",
                stages={
                    "0": {
                        "name": "s0",
                        "construction_days": 1800,
                        "resources": {"cost": {"alloys": 2000}},
                    }
                },
            )
        }
        defs = load_megastructure_defs(raw)
        assert defs["mega_x"].stages[0].build_time == 1800

    def test_stages_ordered_by_stage_number(self):
        from stellaris_advisor.gamedata.megastructures import load_megastructure_defs

        # Provide stages out of order to confirm they are sorted
        raw = {
            "mega_x": _raw_mega(
                "mega_x",
                stages={
                    "2": {
                        "name": "s2",
                        "construction_days": 5400,
                        "resources": {"cost": {"alloys": 30000}},
                    },
                    "0": {
                        "name": "s0",
                        "construction_days": 1800,
                        "resources": {"cost": {"alloys": 2000}},
                    },
                    "1": {
                        "name": "s1",
                        "construction_days": 3600,
                        "resources": {"cost": {"alloys": 10000}},
                    },
                },
            )
        }
        defs = load_megastructure_defs(raw)
        stages = defs["mega_x"].stages
        assert stages[0].stage == 0
        assert stages[1].stage == 1
        assert stages[2].stage == 2

    def test_frozen_dataclass(self):
        from stellaris_advisor.gamedata.megastructures import load_megastructure_defs

        defs = load_megastructure_defs({"mega_x": _raw_mega("mega_x")})
        with pytest.raises(AttributeError):
            defs["mega_x"].name = "changed"  # type: ignore[misc]

    def test_empty_raw_returns_empty(self):
        from stellaris_advisor.gamedata.megastructures import load_megastructure_defs

        assert load_megastructure_defs({}) == {}


# ===========================================================================
# available_megastructures
# ===========================================================================


@pytest.mark.unit
class TestAvailableMegastructures:
    @pytest.fixture
    def defs(self):
        from stellaris_advisor.gamedata.megastructures import load_megastructure_defs

        return load_megastructure_defs(
            {
                "mega_free": _raw_mega("mega_free"),
                "mega_tech": _raw_mega("mega_tech", prerequisites="tech_required"),
                "mega_perk": _raw_mega("mega_perk", required_perks=["ap_required"]),
            }
        )

    def test_no_requirements_always_available(self, defs):
        from stellaris_advisor.gamedata.megastructures import available_megastructures

        result = available_megastructures(defs, frozenset(), frozenset())
        assert "mega_free" in result

    def test_missing_tech_not_available(self, defs):
        from stellaris_advisor.gamedata.megastructures import available_megastructures

        result = available_megastructures(defs, frozenset(), frozenset())
        assert "mega_tech" not in result

    def test_tech_met_makes_available(self, defs):
        from stellaris_advisor.gamedata.megastructures import available_megastructures

        result = available_megastructures(defs, frozenset({"tech_required"}), frozenset())
        assert "mega_tech" in result

    def test_missing_perk_not_available(self, defs):
        from stellaris_advisor.gamedata.megastructures import available_megastructures

        result = available_megastructures(defs, frozenset(), frozenset())
        assert "mega_perk" not in result

    def test_perk_met_makes_available(self, defs):
        from stellaris_advisor.gamedata.megastructures import available_megastructures

        result = available_megastructures(defs, frozenset(), frozenset({"ap_required"}))
        assert "mega_perk" in result


# ===========================================================================
# megastructure_sites
# ===========================================================================


@pytest.mark.unit
class TestMegastructureSites:
    def test_returns_site_requirements(self):
        from stellaris_advisor.gamedata.megastructures import (
            load_megastructure_defs,
            megastructure_sites,
        )

        defs = load_megastructure_defs({"mega_x": _raw_mega("mega_x")})
        # site_requirements is () since fixture has no explicit site_requirements
        assert megastructure_sites(defs["mega_x"]) == ()


# ===========================================================================
# megastructure_total_cost
# ===========================================================================


@pytest.mark.unit
class TestMegastructureTotalCost:
    def test_single_stage_cost(self):
        from stellaris_advisor.gamedata.megastructures import (
            load_megastructure_defs,
            megastructure_total_cost,
        )

        raw = {
            "mega_x": _raw_mega(
                "mega_x",
                stages={
                    "0": {
                        "name": "s0",
                        "construction_days": 1800,
                        "resources": {"cost": {"alloys": 2000}},
                    }
                },
            )
        }
        defs = load_megastructure_defs(raw)
        cost = megastructure_total_cost(defs["mega_x"])
        assert cost["alloys"] == pytest.approx(2000.0)

    def test_multi_stage_summed(self):
        from stellaris_advisor.gamedata.megastructures import (
            load_megastructure_defs,
            megastructure_total_cost,
        )

        raw = {
            "mega_x": _raw_mega(
                "mega_x",
                stages={
                    "0": {
                        "name": "s0",
                        "construction_days": 1800,
                        "resources": {"cost": {"alloys": 2000}},
                    },
                    "1": {
                        "name": "s1",
                        "construction_days": 3600,
                        "resources": {"cost": {"alloys": 10000}},
                    },
                },
            )
        }
        defs = load_megastructure_defs(raw)
        cost = megastructure_total_cost(defs["mega_x"])
        assert cost["alloys"] == pytest.approx(12000.0)

    def test_multiple_resources_summed(self):
        from stellaris_advisor.gamedata.megastructures import (
            load_megastructure_defs,
            megastructure_total_cost,
        )

        raw = {
            "mega_x": _raw_mega(
                "mega_x",
                stages={
                    "0": {
                        "name": "s0",
                        "construction_days": 1800,
                        "resources": {"cost": {"alloys": 2000, "unity": 500}},
                    }
                },
            )
        }
        defs = load_megastructure_defs(raw)
        cost = megastructure_total_cost(defs["mega_x"])
        assert cost["alloys"] == pytest.approx(2000.0)
        assert cost["unity"] == pytest.approx(500.0)

    def test_no_stages_returns_empty(self):
        from stellaris_advisor.gamedata.megastructures import (
            load_megastructure_defs,
            megastructure_total_cost,
        )

        raw = {"mega_x": {"name": "mega_x"}}
        defs = load_megastructure_defs(raw)
        assert megastructure_total_cost(defs["mega_x"]) == {}


# ===========================================================================
# Integration — real ascension perks fixture
# ===========================================================================


@pytest.mark.integration
class TestAscensionPerkDefsFromFixtures:
    @pytest.fixture
    def perk_raw(self):
        from stellaris_advisor.parser.clausewitz import parse

        return parse(_FIXTURE_DIR.joinpath("ascension_perks.txt").read_text())

    def test_loads_all_perks(self, perk_raw):
        from stellaris_advisor.gamedata.ascension_perks import load_ascension_perk_defs

        defs = load_ascension_perk_defs(perk_raw)
        for pid in (
            "ap_one_vision",
            "ap_executive_vigor",
            "ap_mastery_of_nature",
            "ap_evolutionary_mastery",
        ):
            assert pid in defs

    def test_one_vision_no_requirements(self, perk_raw):
        from stellaris_advisor.gamedata.ascension_perks import load_ascension_perk_defs

        defs = load_ascension_perk_defs(perk_raw)
        assert defs["ap_one_vision"].required_perks == frozenset()
        assert defs["ap_one_vision"].required_traditions == 0

    def test_mastery_of_nature_requires_perk(self, perk_raw):
        from stellaris_advisor.gamedata.ascension_perks import load_ascension_perk_defs

        defs = load_ascension_perk_defs(perk_raw)
        assert "ap_one_vision" in defs["ap_mastery_of_nature"].required_perks

    def test_evolutionary_mastery_required_traditions(self, perk_raw):
        from stellaris_advisor.gamedata.ascension_perks import load_ascension_perk_defs

        defs = load_ascension_perk_defs(perk_raw)
        assert defs["ap_evolutionary_mastery"].required_traditions == 3

    def test_excluded_perks_populated(self, perk_raw):
        from stellaris_advisor.gamedata.ascension_perks import load_ascension_perk_defs

        defs = load_ascension_perk_defs(perk_raw)
        assert "ap_synthetic_evolution" in defs["ap_evolutionary_mastery"].excluded_perks


# ===========================================================================
# Integration — real megastructures fixture
# ===========================================================================


@pytest.mark.integration
class TestMegastructureDefsFromFixtures:
    @pytest.fixture
    def mega_raw(self):
        from stellaris_advisor.parser.clausewitz import parse

        return parse(_FIXTURE_DIR.joinpath("megastructures.txt").read_text())

    def test_loads_all_megastructures(self, mega_raw):
        from stellaris_advisor.gamedata.megastructures import load_megastructure_defs

        defs = load_megastructure_defs(mega_raw)
        for mid in (
            "megastructure_dyson_sphere",
            "megastructure_ring_world",
            "megastructure_science_nexus",
        ):
            assert mid in defs

    def test_dyson_sphere_prerequisites(self, mega_raw):
        from stellaris_advisor.gamedata.megastructures import load_megastructure_defs

        defs = load_megastructure_defs(mega_raw)
        assert "tech_cruisers" in defs["megastructure_dyson_sphere"].prerequisites

    def test_dyson_sphere_required_perks(self, mega_raw):
        from stellaris_advisor.gamedata.megastructures import load_megastructure_defs

        defs = load_megastructure_defs(mega_raw)
        assert "ap_one_vision" in defs["megastructure_dyson_sphere"].required_perks

    def test_dyson_sphere_has_three_stages(self, mega_raw):
        from stellaris_advisor.gamedata.megastructures import load_megastructure_defs

        defs = load_megastructure_defs(mega_raw)
        assert len(defs["megastructure_dyson_sphere"].stages) == 3

    def test_dyson_sphere_stage0_cost(self, mega_raw):
        from stellaris_advisor.gamedata.megastructures import load_megastructure_defs

        defs = load_megastructure_defs(mega_raw)
        stage0 = defs["megastructure_dyson_sphere"].stages[0]
        costs = {m.resource: m.amount for m in stage0.cost}
        assert costs["alloys"] == pytest.approx(2000.0)

    def test_dyson_sphere_total_cost(self, mega_raw):
        from stellaris_advisor.gamedata.megastructures import (
            load_megastructure_defs,
            megastructure_total_cost,
        )

        defs = load_megastructure_defs(mega_raw)
        cost = megastructure_total_cost(defs["megastructure_dyson_sphere"])
        # Stage costs: 2000 + 10000 + 20000 = 32000 alloys
        assert cost["alloys"] == pytest.approx(32000.0)

    def test_ring_world_has_three_stages(self, mega_raw):
        from stellaris_advisor.gamedata.megastructures import load_megastructure_defs

        defs = load_megastructure_defs(mega_raw)
        assert len(defs["megastructure_ring_world"].stages) == 3
