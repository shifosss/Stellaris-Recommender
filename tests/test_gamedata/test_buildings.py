"""Tests for gamedata/buildings.py, districts.py, jobs.py — TDD: tests written first.

Covers:
  buildings: load_building_defs, available_buildings, building_unlock_chain,
             estimate_building_net_output
  districts: load_district_defs, available_districts
  jobs:      load_job_defs
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "game_data"
_TECH_DIR = _FIXTURE_DIR / "technology"


# ---------------------------------------------------------------------------
# Raw-dict helpers
# ---------------------------------------------------------------------------


def _raw_building(
    bid: str,
    category: str = "research",
    cost: dict[str, Any] | None = None,
    upkeep: dict[str, Any] | None = None,
    planet_modifier: dict[str, Any] | None = None,
    prerequisites: str | list[str] | None = None,
    upgrades_from: str | None = None,
    unique: bool = False,
    can_build: bool = True,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "category": category,
        "cost": {"minerals": 100} if cost is None else cost,
        "upkeep": {"energy": 1} if upkeep is None else upkeep,
        "planet_modifier": planet_modifier or {},
    }
    if prerequisites is not None:
        data["prerequisites"] = prerequisites
    if upgrades_from is not None:
        data["upgrades_from"] = upgrades_from
    if unique:
        data["unique"] = True
    if not can_build:
        data["can_build"] = False
    return data


def _raw_district(
    did: str,
    cost: dict[str, Any] | None = None,
    upkeep: dict[str, Any] | None = None,
    planet_modifier: dict[str, Any] | None = None,
    prerequisites: str | list[str] | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "cost": cost or {"minerals": 300},
        "upkeep": upkeep or {"energy": 2},
        "planet_modifier": planet_modifier or {},
    }
    if prerequisites is not None:
        data["prerequisites"] = prerequisites
    return data


def _raw_job(
    jid: str,
    category: str = "worker",
    produces: dict[str, Any] | None = None,
    upkeep: dict[str, Any] | None = None,
    weight: float = 5.0,
) -> dict[str, Any]:
    return {
        "category": category,
        "resources": {
            "produces": produces or {},
            "upkeep": upkeep or {},
        },
        "weight": {"weight": weight},
    }


# ---------------------------------------------------------------------------
# load_building_defs
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLoadBuildingDefs:
    def test_basic_fields(self):
        from stellaris_advisor.gamedata.buildings import load_building_defs

        raw = {"building_lab": _raw_building("building_lab", category="research")}
        defs = load_building_defs(raw)
        assert "building_lab" in defs
        b = defs["building_lab"]
        assert b.id == "building_lab"
        assert b.category == "research"

    def test_cost_extraction(self):
        from stellaris_advisor.gamedata.buildings import load_building_defs

        raw = {"b": _raw_building("b", cost={"minerals": 200, "energy": 50})}
        defs = load_building_defs(raw)
        costs = {m.resource: m.amount for m in defs["b"].base_cost}
        assert costs["minerals"] == 200.0
        assert costs["energy"] == 50.0

    def test_upkeep_extraction(self):
        from stellaris_advisor.gamedata.buildings import load_building_defs

        raw = {"b": _raw_building("b", upkeep={"energy": 3})}
        defs = load_building_defs(raw)
        upkeep = {m.resource: m.amount for m in defs["b"].upkeep}
        assert upkeep["energy"] == 3.0

    def test_job_extraction_from_planet_modifier(self):
        from stellaris_advisor.gamedata.buildings import load_building_defs

        raw = {"b": _raw_building("b", planet_modifier={"job_researcher_add": 2, "job_clerk_add": 1})}
        defs = load_building_defs(raw)
        assert "researcher" in defs["b"].jobs
        assert "clerk" in defs["b"].jobs

    def test_prerequisites_as_string(self):
        from stellaris_advisor.gamedata.buildings import load_building_defs

        raw = {"b": _raw_building("b", prerequisites="tech_lasers")}
        defs = load_building_defs(raw)
        assert defs["b"].prerequisites == frozenset({"tech_lasers"})

    def test_prerequisites_as_list(self):
        from stellaris_advisor.gamedata.buildings import load_building_defs

        raw = {"b": _raw_building("b", prerequisites=["tech_a", "tech_b"])}
        defs = load_building_defs(raw)
        assert defs["b"].prerequisites == frozenset({"tech_a", "tech_b"})

    def test_no_prerequisites(self):
        from stellaris_advisor.gamedata.buildings import load_building_defs

        raw = {"b": _raw_building("b")}
        defs = load_building_defs(raw)
        assert defs["b"].prerequisites == frozenset()

    def test_upgrades_from(self):
        from stellaris_advisor.gamedata.buildings import load_building_defs

        raw = {"b_adv": _raw_building("b_adv", upgrades_from="b_base")}
        defs = load_building_defs(raw)
        assert defs["b_adv"].upgrades_from == "b_base"

    def test_unique_sets_max_per_planet_1(self):
        from stellaris_advisor.gamedata.buildings import load_building_defs

        raw = {"b": _raw_building("b", unique=True)}
        defs = load_building_defs(raw)
        assert defs["b"].max_per_planet == 1

    def test_can_build_false_sets_max_per_planet_0(self):
        from stellaris_advisor.gamedata.buildings import load_building_defs

        raw = {"b": _raw_building("b", can_build=False)}
        defs = load_building_defs(raw)
        assert defs["b"].max_per_planet == 0

    def test_default_max_per_planet_is_unlimited(self):
        from stellaris_advisor.gamedata.buildings import load_building_defs

        raw = {"b": _raw_building("b")}
        defs = load_building_defs(raw)
        assert defs["b"].max_per_planet == -1

    def test_returns_frozen_dataclasses(self):
        from stellaris_advisor.gamedata.buildings import load_building_defs

        defs = load_building_defs({"b": _raw_building("b")})
        with pytest.raises(AttributeError):
            defs["b"].category = "other"  # type: ignore[misc]

    def test_empty_raw_returns_empty(self):
        from stellaris_advisor.gamedata.buildings import load_building_defs

        assert load_building_defs({}) == {}


# ---------------------------------------------------------------------------
# available_buildings
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAvailableBuildings:
    def _two_buildings(self):
        from stellaris_advisor.gamedata.buildings import load_building_defs

        return load_building_defs(
            {
                "b_free": _raw_building("b_free"),
                "b_prereq": _raw_building("b_prereq", prerequisites="tech_lasers"),
                "b_unique": _raw_building("b_unique", unique=True),
                "b_never": _raw_building("b_never", can_build=False),
            }
        )

    def test_no_prereq_building_always_available(self):
        from stellaris_advisor.gamedata.buildings import available_buildings

        defs = self._two_buildings()
        result = available_buildings(defs, frozenset())
        assert "b_free" in result

    def test_prereq_not_met_blocks_building(self):
        from stellaris_advisor.gamedata.buildings import available_buildings

        defs = self._two_buildings()
        result = available_buildings(defs, frozenset())
        assert "b_prereq" not in result

    def test_prereq_met_unlocks_building(self):
        from stellaris_advisor.gamedata.buildings import available_buildings

        defs = self._two_buildings()
        result = available_buildings(defs, frozenset({"tech_lasers"}))
        assert "b_prereq" in result

    def test_can_build_false_never_available(self):
        from stellaris_advisor.gamedata.buildings import available_buildings

        defs = self._two_buildings()
        result = available_buildings(defs, frozenset({"tech_lasers"}))
        assert "b_never" not in result

    def test_unique_blocked_if_already_built(self):
        from stellaris_advisor.gamedata.buildings import available_buildings

        defs = self._two_buildings()
        result = available_buildings(defs, frozenset(), existing_buildings=frozenset({"b_unique"}))
        assert "b_unique" not in result

    def test_unique_available_if_not_yet_built(self):
        from stellaris_advisor.gamedata.buildings import available_buildings

        defs = self._two_buildings()
        result = available_buildings(defs, frozenset(), existing_buildings=frozenset())
        assert "b_unique" in result

    def test_planet_class_filter(self):
        from stellaris_advisor.gamedata.buildings import load_building_defs, available_buildings
        from stellaris_advisor.gamedata.models import BuildingDef, OutputModifier

        # Building restricted to pc_continental only
        restricted = BuildingDef(
            id="b_restricted",
            name="b_restricted",
            category="research",
            base_cost=(OutputModifier(resource="minerals", amount=100.0),),
            upkeep=(OutputModifier(resource="energy", amount=1.0),),
            jobs=(),
            prerequisites=frozenset(),
            planet_classes=frozenset({"pc_continental"}),
            empire_types=frozenset(),
            max_per_planet=-1,
        )
        free = BuildingDef(
            id="b_free",
            name="b_free",
            category="research",
            base_cost=(OutputModifier(resource="minerals", amount=100.0),),
            upkeep=(OutputModifier(resource="energy", amount=1.0),),
            jobs=(),
            prerequisites=frozenset(),
            planet_classes=frozenset(),
            empire_types=frozenset(),
            max_per_planet=-1,
        )
        defs = {"b_restricted": restricted, "b_free": free}
        result = available_buildings(defs, frozenset(), planet_class="pc_arid")
        assert "b_restricted" not in result
        assert "b_free" in result

        result2 = available_buildings(defs, frozenset(), planet_class="pc_continental")
        assert "b_restricted" in result2


# ---------------------------------------------------------------------------
# building_unlock_chain
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildingUnlockChain:
    def _setup(self):
        from stellaris_advisor.gamedata.buildings import load_building_defs
        from stellaris_advisor.gamedata.tech_graph import load_tech_defs

        building_raw = {
            "b_free": _raw_building("b_free"),
            "b_requires_lasers": _raw_building("b_requires_lasers", prerequisites="tech_lasers"),
            "b_requires_advanced": _raw_building("b_requires_advanced", prerequisites="tech_advanced_lasers"),
        }
        tech_raw = {
            "tech_lasers": {"area": "physics", "tier": 0, "cost": 0, "category": "x"},
            "tech_advanced_lasers": {
                "area": "physics",
                "tier": 1,
                "cost": 1000,
                "category": "x",
                "prerequisites": "tech_lasers",
            },
        }
        building_defs = load_building_defs(building_raw)
        tech_defs, tech_dag = load_tech_defs(tech_raw)
        return building_defs, tech_defs, tech_dag

    def test_unknown_building_returns_none(self):
        from stellaris_advisor.gamedata.buildings import building_unlock_chain

        building_defs, tech_defs, tech_dag = self._setup()
        assert building_unlock_chain(building_defs, tech_defs, tech_dag, "b_missing", frozenset()) is None

    def test_no_prereq_building_returns_empty(self):
        from stellaris_advisor.gamedata.buildings import building_unlock_chain

        building_defs, tech_defs, tech_dag = self._setup()
        result = building_unlock_chain(building_defs, tech_defs, tech_dag, "b_free", frozenset())
        assert result == ()

    def test_single_prereq_chain(self):
        from stellaris_advisor.gamedata.buildings import building_unlock_chain

        building_defs, tech_defs, tech_dag = self._setup()
        path = building_unlock_chain(building_defs, tech_defs, tech_dag, "b_requires_lasers", frozenset())
        assert path == ("tech_lasers",)

    def test_chain_through_prerequisite_chain(self):
        from stellaris_advisor.gamedata.buildings import building_unlock_chain

        building_defs, tech_defs, tech_dag = self._setup()
        path = building_unlock_chain(
            building_defs, tech_defs, tech_dag, "b_requires_advanced", frozenset()
        )
        assert path is not None
        assert "tech_lasers" in path
        assert "tech_advanced_lasers" in path
        assert path.index("tech_lasers") < path.index("tech_advanced_lasers")

    def test_already_researched_prereqs_returns_empty(self):
        from stellaris_advisor.gamedata.buildings import building_unlock_chain

        building_defs, tech_defs, tech_dag = self._setup()
        path = building_unlock_chain(
            building_defs, tech_defs, tech_dag, "b_requires_lasers", frozenset({"tech_lasers"})
        )
        assert path == ()


# ---------------------------------------------------------------------------
# estimate_building_net_output
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEstimateBuildingNetOutput:
    def test_upkeep_reported_as_negative(self):
        from stellaris_advisor.gamedata.buildings import estimate_building_net_output, load_building_defs

        defs = load_building_defs({"b": _raw_building("b", upkeep={"energy": 2})})
        result = estimate_building_net_output(defs["b"], empire_state=None)
        assert result["energy"] == -2.0

    def test_multiple_upkeep_resources(self):
        from stellaris_advisor.gamedata.buildings import estimate_building_net_output, load_building_defs

        defs = load_building_defs({"b": _raw_building("b", upkeep={"energy": 5, "minerals": 10})})
        result = estimate_building_net_output(defs["b"], empire_state=None)
        assert result["energy"] == -5.0
        assert result["minerals"] == -10.0

    def test_no_upkeep_returns_empty(self):
        from stellaris_advisor.gamedata.buildings import estimate_building_net_output, load_building_defs

        defs = load_building_defs({"b": _raw_building("b", upkeep={})})
        result = estimate_building_net_output(defs["b"], empire_state=None)
        assert result == {}


# ---------------------------------------------------------------------------
# load_district_defs
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLoadDistrictDefs:
    def test_basic_fields(self):
        from stellaris_advisor.gamedata.districts import load_district_defs

        raw = {"district_city": _raw_district("district_city")}
        defs = load_district_defs(raw)
        assert "district_city" in defs
        assert defs["district_city"].id == "district_city"

    def test_cost_extraction(self):
        from stellaris_advisor.gamedata.districts import load_district_defs

        raw = {"d": _raw_district("d", cost={"minerals": 400})}
        defs = load_district_defs(raw)
        costs = {m.resource: m.amount for m in defs["d"].base_cost}
        assert costs["minerals"] == 400.0

    def test_jobs_from_planet_modifier(self):
        from stellaris_advisor.gamedata.districts import load_district_defs

        raw = {"d": _raw_district("d", planet_modifier={"job_miner_add": 2, "job_farmer_add": 1})}
        defs = load_district_defs(raw)
        assert "miner" in defs["d"].jobs
        assert "farmer" in defs["d"].jobs

    def test_prerequisites_extracted(self):
        from stellaris_advisor.gamedata.districts import load_district_defs

        raw = {"d": _raw_district("d", prerequisites="tech_alloys")}
        defs = load_district_defs(raw)
        assert defs["d"].prerequisites == frozenset({"tech_alloys"})

    def test_no_prerequisites(self):
        from stellaris_advisor.gamedata.districts import load_district_defs

        raw = {"d": _raw_district("d")}
        defs = load_district_defs(raw)
        assert defs["d"].prerequisites == frozenset()

    def test_returns_frozen_dataclasses(self):
        from stellaris_advisor.gamedata.districts import load_district_defs

        defs = load_district_defs({"d": _raw_district("d")})
        with pytest.raises(AttributeError):
            defs["d"].id = "other"  # type: ignore[misc]

    def test_empty_raw_returns_empty(self):
        from stellaris_advisor.gamedata.districts import load_district_defs

        assert load_district_defs({}) == {}


# ---------------------------------------------------------------------------
# available_districts
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAvailableDistricts:
    def _setup(self):
        from stellaris_advisor.gamedata.districts import load_district_defs

        return load_district_defs(
            {
                "d_free": _raw_district("d_free"),
                "d_prereq": _raw_district("d_prereq", prerequisites="tech_alloys"),
            }
        )

    def test_no_prereq_always_available(self):
        from stellaris_advisor.gamedata.districts import available_districts

        defs = self._setup()
        result = available_districts(defs, researched=frozenset())
        assert "d_free" in result

    def test_prereq_blocks_district(self):
        from stellaris_advisor.gamedata.districts import available_districts

        defs = self._setup()
        result = available_districts(defs, researched=frozenset())
        assert "d_prereq" not in result

    def test_prereq_met_unlocks_district(self):
        from stellaris_advisor.gamedata.districts import available_districts

        defs = self._setup()
        result = available_districts(defs, researched=frozenset({"tech_alloys"}))
        assert "d_prereq" in result

    def test_planet_class_filter(self):
        from stellaris_advisor.gamedata.districts import load_district_defs, available_districts
        from stellaris_advisor.gamedata.models import DistrictDef, OutputModifier

        restricted = DistrictDef(
            id="d_restricted",
            name="d_restricted",
            base_cost=(OutputModifier(resource="minerals", amount=300.0),),
            upkeep=(OutputModifier(resource="energy", amount=2.0),),
            jobs=(),
            planet_classes=frozenset({"pc_continental"}),
            prerequisites=frozenset(),
        )
        free = DistrictDef(
            id="d_free",
            name="d_free",
            base_cost=(OutputModifier(resource="minerals", amount=300.0),),
            upkeep=(OutputModifier(resource="energy", amount=2.0),),
            jobs=(),
            planet_classes=frozenset(),
            prerequisites=frozenset(),
        )
        defs = {"d_restricted": restricted, "d_free": free}

        result = available_districts(defs, planet_class="pc_arid", researched=frozenset())
        assert "d_restricted" not in result
        assert "d_free" in result

        result2 = available_districts(defs, planet_class="pc_continental", researched=frozenset())
        assert "d_restricted" in result2


# ---------------------------------------------------------------------------
# load_job_defs
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLoadJobDefs:
    def test_basic_fields(self):
        from stellaris_advisor.gamedata.jobs import load_job_defs

        raw = {"researcher": _raw_job("researcher", category="specialist")}
        defs = load_job_defs(raw)
        assert "researcher" in defs
        assert defs["researcher"].id == "researcher"
        assert defs["researcher"].category == "specialist"

    def test_produces_extraction(self):
        from stellaris_advisor.gamedata.jobs import load_job_defs

        raw = {"researcher": _raw_job("researcher", produces={"physics_research": 4, "society_research": 2})}
        defs = load_job_defs(raw)
        produces = {m.resource: m.amount for m in defs["researcher"].produces}
        assert produces["physics_research"] == 4.0
        assert produces["society_research"] == 2.0

    def test_upkeep_extraction(self):
        from stellaris_advisor.gamedata.jobs import load_job_defs

        raw = {"researcher": _raw_job("researcher", upkeep={"consumer_goods": 1})}
        defs = load_job_defs(raw)
        upkeep = {m.resource: m.amount for m in defs["researcher"].upkeep}
        assert upkeep["consumer_goods"] == 1.0

    def test_weight_extraction(self):
        from stellaris_advisor.gamedata.jobs import load_job_defs

        raw = {"miner": _raw_job("miner", weight=5.0)}
        defs = load_job_defs(raw)
        assert defs["miner"].weight == 5.0

    def test_empty_produces_and_upkeep(self):
        from stellaris_advisor.gamedata.jobs import load_job_defs

        raw = {"soldier": _raw_job("soldier", produces={}, upkeep={})}
        defs = load_job_defs(raw)
        assert defs["soldier"].produces == ()
        assert defs["soldier"].upkeep == ()

    def test_returns_frozen_dataclasses(self):
        from stellaris_advisor.gamedata.jobs import load_job_defs

        defs = load_job_defs({"j": _raw_job("j")})
        with pytest.raises(AttributeError):
            defs["j"].category = "other"  # type: ignore[misc]

    def test_empty_raw_returns_empty(self):
        from stellaris_advisor.gamedata.jobs import load_job_defs

        assert load_job_defs({}) == {}


# ---------------------------------------------------------------------------
# Integration — real fixture files
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestBuildingDefsFromFixtures:
    @pytest.fixture
    def building_raw(self):
        from stellaris_advisor.parser.clausewitz import parse

        return parse(_FIXTURE_DIR.joinpath("buildings.txt").read_text())

    @pytest.fixture
    def all_tech_raw(self):
        from stellaris_advisor.parser.clausewitz import parse

        combined: dict[str, Any] = {}
        for fname in ("physics.txt", "society.txt", "engineering.txt"):
            combined.update(parse(_TECH_DIR.joinpath(fname).read_text()))
        return combined

    def test_loads_all_buildings(self, building_raw):
        from stellaris_advisor.gamedata.buildings import load_building_defs

        defs = load_building_defs(building_raw)
        for bid in (
            "building_capital",
            "building_research_lab",
            "building_advanced_research_lab",
            "building_alloy_foundry",
            "building_fortress",
            "building_temple",
            "building_hydroponics_farm",
            "building_gene_clinic",
        ):
            assert bid in defs, f"Missing {bid}"

    def test_capital_not_buildable(self, building_raw):
        from stellaris_advisor.gamedata.buildings import load_building_defs, available_buildings

        defs = load_building_defs(building_raw)
        assert defs["building_capital"].max_per_planet == 0
        result = available_buildings(defs, frozenset())
        assert "building_capital" not in result

    def test_research_lab_requires_tech_basic_science(self, building_raw):
        from stellaris_advisor.gamedata.buildings import load_building_defs

        defs = load_building_defs(building_raw)
        assert "tech_basic_science" in defs["building_research_lab"].prerequisites

    def test_advanced_lab_upgrades_from_lab(self, building_raw):
        from stellaris_advisor.gamedata.buildings import load_building_defs

        defs = load_building_defs(building_raw)
        assert defs["building_advanced_research_lab"].upgrades_from == "building_research_lab"

    def test_building_unlock_chain_for_advanced_lab(self, building_raw, all_tech_raw):
        from stellaris_advisor.gamedata.buildings import building_unlock_chain, load_building_defs
        from stellaris_advisor.gamedata.tech_graph import load_tech_defs

        building_defs = load_building_defs(building_raw)
        tech_defs, tech_dag = load_tech_defs(all_tech_raw)
        path = building_unlock_chain(
            building_defs, tech_defs, tech_dag, "building_advanced_research_lab", frozenset()
        )
        assert path is not None
        assert "tech_neural_implants" in path


@pytest.mark.integration
class TestDistrictDefsFromFixtures:
    @pytest.fixture
    def district_raw(self):
        from stellaris_advisor.parser.clausewitz import parse

        return parse(_FIXTURE_DIR.joinpath("districts.txt").read_text())

    def test_loads_all_districts(self, district_raw):
        from stellaris_advisor.gamedata.districts import load_district_defs

        defs = load_district_defs(district_raw)
        for did in ("district_city", "district_mining", "district_farming", "district_generator", "district_industrial"):
            assert did in defs

    def test_industrial_requires_tech_alloys(self, district_raw):
        from stellaris_advisor.gamedata.districts import load_district_defs

        defs = load_district_defs(district_raw)
        assert "tech_alloys" in defs["district_industrial"].prerequisites

    def test_industrial_not_available_without_tech(self, district_raw):
        from stellaris_advisor.gamedata.districts import load_district_defs, available_districts

        defs = load_district_defs(district_raw)
        result = available_districts(defs, researched=frozenset())
        assert "district_industrial" not in result

    def test_city_always_available(self, district_raw):
        from stellaris_advisor.gamedata.districts import load_district_defs, available_districts

        defs = load_district_defs(district_raw)
        result = available_districts(defs, researched=frozenset())
        assert "district_city" in result


@pytest.mark.integration
class TestJobDefsFromFixtures:
    @pytest.fixture
    def job_raw(self):
        from stellaris_advisor.parser.clausewitz import parse

        return parse(_FIXTURE_DIR.joinpath("jobs.txt").read_text())

    def test_loads_all_jobs(self, job_raw):
        from stellaris_advisor.gamedata.jobs import load_job_defs

        defs = load_job_defs(job_raw)
        for jid in ("researcher", "miner", "farmer", "soldier", "clerk"):
            assert jid in defs

    def test_researcher_category(self, job_raw):
        from stellaris_advisor.gamedata.jobs import load_job_defs

        defs = load_job_defs(job_raw)
        assert defs["researcher"].category == "specialist"

    def test_researcher_produces_research(self, job_raw):
        from stellaris_advisor.gamedata.jobs import load_job_defs

        defs = load_job_defs(job_raw)
        produces = {m.resource: m.amount for m in defs["researcher"].produces}
        assert "physics_research" in produces
        assert produces["physics_research"] == 4.0

    def test_miner_produces_minerals(self, job_raw):
        from stellaris_advisor.gamedata.jobs import load_job_defs

        defs = load_job_defs(job_raw)
        produces = {m.resource: m.amount for m in defs["miner"].produces}
        assert produces.get("minerals") == 6.0
