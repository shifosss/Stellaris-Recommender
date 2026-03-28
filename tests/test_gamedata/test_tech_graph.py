"""Tests for gamedata/tech_graph.py — TDD: written before implementation.

Fixture tech layout (flat fixtures, updated tiers per team-lead spec):

    Physics:
        tech_basic_science (T0, start)
        tech_lasers        (T0) ──► tech_advanced_lasers (T1) ──► tech_plasma (T2, rare)
        tech_shields       (T0)

    Society:
        tech_colonization      (T0, start) ──► tech_advanced_colonization (T1)
        tech_alien_life_studies (T0)       ──► tech_genome_mapping (T1) ──► tech_neural_implants (T2, dangerous)

    Engineering:
        tech_basic_alloys (T0, start) ──► tech_alloys (T1)
        tech_corvettes    (T0, start) ──► tech_destroyers (T1) ──► tech_cruisers (T2)

Total: 7 T0 techs, 5 T1 techs, 3 T2 techs.
Tier gating threshold: ≥6 tier-N techs in researched to unlock tier-N+1 draws.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "game_data" / "technology"


def _raw_tech(
    tech_id: str,
    area: str = "physics",
    tier: int = 0,
    cost: int = 0,
    category: str = "x",
    prerequisites: str | list[str] | None = None,
    is_rare: bool = False,
    is_dangerous: bool = False,
    start_tech: bool = False,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "area": area,
        "tier": tier,
        "cost": cost,
        "category": category,
    }
    if prerequisites is not None:
        data["prerequisites"] = prerequisites
    if is_rare:
        data["is_rare"] = True
    if is_dangerous:
        data["is_dangerous"] = True
    if start_tech:
        data["start_tech"] = True
    return data


def _chain_raw() -> dict[str, Any]:
    """Minimal 3-tech chain: root(T0) → mid(T1) → leaf(T2)."""
    return {
        "tech_root": _raw_tech("tech_root", tier=0, start_tech=True),
        "tech_mid": _raw_tech("tech_mid", tier=1, cost=1000, prerequisites="tech_root"),
        "tech_leaf": _raw_tech("tech_leaf", tier=2, cost=3000, prerequisites="tech_mid"),
    }


def _tier_gate_raw() -> dict[str, Any]:
    """7 T0 + 2 T1 techs to exercise tier gating (need ≥6 T0 for T1 access)."""
    raw: dict[str, Any] = {}
    t0_ids = [f"tech_t0_{i}" for i in range(7)]
    for tid in t0_ids:
        raw[tid] = _raw_tech(tid, tier=0)
    # T1 techs: tech_t0_0 and tech_t0_3 are prerequisites respectively
    raw["tech_t1_a"] = _raw_tech("tech_t1_a", tier=1, cost=1000, prerequisites="tech_t0_0")
    raw["tech_t1_b"] = _raw_tech("tech_t1_b", tier=1, cost=1000, prerequisites="tech_t0_3")
    return raw


# ---------------------------------------------------------------------------
# load_tech_defs
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLoadTechDefs:
    def test_basic_fields(self):
        from stellaris_advisor.gamedata.tech_graph import load_tech_defs

        raw = {"tech_alpha": _raw_tech("tech_alpha", area="physics", tier=0, cost=0, category="computing")}
        tech_defs, _ = load_tech_defs(raw)
        assert "tech_alpha" in tech_defs
        t = tech_defs["tech_alpha"]
        assert t.id == "tech_alpha"
        assert t.area == "physics"
        assert t.tier == 0
        assert t.cost == 0
        assert t.category == "computing"

    def test_start_tech_flag(self):
        from stellaris_advisor.gamedata.tech_graph import load_tech_defs

        raw = {"tech_s": _raw_tech("tech_s", start_tech=True)}
        tech_defs, _ = load_tech_defs(raw)
        assert tech_defs["tech_s"].is_start_tech is True

    def test_is_rare_flag(self):
        from stellaris_advisor.gamedata.tech_graph import load_tech_defs

        raw = {"tech_r": _raw_tech("tech_r", tier=2, is_rare=True)}
        tech_defs, _ = load_tech_defs(raw)
        assert tech_defs["tech_r"].is_rare is True

    def test_is_dangerous_flag(self):
        from stellaris_advisor.gamedata.tech_graph import load_tech_defs

        raw = {"tech_d": _raw_tech("tech_d", tier=2, is_dangerous=True)}
        tech_defs, _ = load_tech_defs(raw)
        assert tech_defs["tech_d"].is_dangerous is True

    def test_prerequisite_as_string(self):
        from stellaris_advisor.gamedata.tech_graph import load_tech_defs

        raw = {
            "tech_a": _raw_tech("tech_a", tier=0),
            "tech_b": _raw_tech("tech_b", tier=1, cost=1000, prerequisites="tech_a"),
        }
        tech_defs, dag = load_tech_defs(raw)
        assert tech_defs["tech_b"].prerequisites == frozenset({"tech_a"})
        assert dag.has_edge("tech_a", "tech_b")

    def test_prerequisite_as_list(self):
        from stellaris_advisor.gamedata.tech_graph import load_tech_defs

        raw = {
            "tech_a": _raw_tech("tech_a", tier=0),
            "tech_b": _raw_tech("tech_b", tier=0),
            "tech_c": _raw_tech("tech_c", tier=1, cost=1000, prerequisites=["tech_a", "tech_b"]),
        }
        tech_defs, dag = load_tech_defs(raw)
        assert tech_defs["tech_c"].prerequisites == frozenset({"tech_a", "tech_b"})
        assert dag.has_edge("tech_a", "tech_c")
        assert dag.has_edge("tech_b", "tech_c")

    def test_no_prerequisites(self):
        from stellaris_advisor.gamedata.tech_graph import load_tech_defs

        raw = {"tech_x": _raw_tech("tech_x")}
        tech_defs, dag = load_tech_defs(raw)
        assert tech_defs["tech_x"].prerequisites == frozenset()
        assert list(dag.predecessors("tech_x")) == []

    def test_dag_is_directed_acyclic(self):
        from stellaris_advisor.gamedata.tech_graph import load_tech_defs

        _, dag = load_tech_defs(_chain_raw())
        assert nx.is_directed_acyclic_graph(dag)

    def test_all_nodes_in_dag(self):
        from stellaris_advisor.gamedata.tech_graph import load_tech_defs

        raw = _chain_raw()
        _, dag = load_tech_defs(raw)
        assert set(dag.nodes) >= set(raw.keys())

    def test_empty_raw_returns_empty(self):
        from stellaris_advisor.gamedata.tech_graph import load_tech_defs

        tech_defs, dag = load_tech_defs({})
        assert tech_defs == {}
        assert len(dag.nodes) == 0

    def test_returns_immutable_tech_nodes(self):
        from stellaris_advisor.gamedata.tech_graph import load_tech_defs

        tech_defs, _ = load_tech_defs({"tech_x": _raw_tech("tech_x")})
        with pytest.raises(AttributeError):
            tech_defs["tech_x"].tier = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# reachable_techs
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestReachableTechs:
    def test_no_research_returns_tier0_no_prereqs(self):
        from stellaris_advisor.gamedata.tech_graph import load_tech_defs, reachable_techs

        tech_defs, dag = load_tech_defs(_tier_gate_raw())
        result = reachable_techs(tech_defs, dag, frozenset())
        # All 7 T0 techs have no prerequisites → immediately reachable
        t0_ids = {k for k, v in tech_defs.items() if v.tier == 0}
        assert result == t0_ids
        # T1 techs are NOT reachable (tier gate: need ≥6 T0 and 0 researched)
        assert "tech_t1_a" not in result
        assert "tech_t1_b" not in result

    def test_six_tier0_unlocks_tier1_with_prereqs_met(self):
        from stellaris_advisor.gamedata.tech_graph import load_tech_defs, reachable_techs

        tech_defs, dag = load_tech_defs(_tier_gate_raw())
        # Research 6 T0 techs — includes tech_t0_0 (prereq for tech_t1_a) and tech_t0_3 (prereq for tech_t1_b)
        six_t0 = frozenset({"tech_t0_0", "tech_t0_1", "tech_t0_2", "tech_t0_3", "tech_t0_4", "tech_t0_5"})
        result = reachable_techs(tech_defs, dag, six_t0)
        assert "tech_t1_a" in result
        assert "tech_t1_b" in result

    def test_five_tier0_does_not_unlock_tier1(self):
        from stellaris_advisor.gamedata.tech_graph import load_tech_defs, reachable_techs

        tech_defs, dag = load_tech_defs(_tier_gate_raw())
        # Only 5 T0 researched (prereqs met but tier gate blocked)
        five_t0 = frozenset({"tech_t0_0", "tech_t0_1", "tech_t0_2", "tech_t0_3", "tech_t0_4"})
        result = reachable_techs(tech_defs, dag, five_t0)
        assert "tech_t1_a" not in result
        assert "tech_t1_b" not in result

    def test_prereq_not_met_blocks_tier1(self):
        from stellaris_advisor.gamedata.tech_graph import load_tech_defs, reachable_techs

        tech_defs, dag = load_tech_defs(_tier_gate_raw())
        # 6 T0 researched but tech_t0_3 (prereq of tech_t1_b) is excluded
        six_without_t0_3 = frozenset({"tech_t0_0", "tech_t0_1", "tech_t0_2", "tech_t0_4", "tech_t0_5", "tech_t0_6"})
        result = reachable_techs(tech_defs, dag, six_without_t0_3)
        assert "tech_t1_a" in result  # prereq tech_t0_0 is met
        assert "tech_t1_b" not in result  # prereq tech_t0_3 is NOT met

    def test_already_researched_excluded(self):
        from stellaris_advisor.gamedata.tech_graph import load_tech_defs, reachable_techs

        tech_defs, dag = load_tech_defs(_tier_gate_raw())
        all_t0 = frozenset(k for k, v in tech_defs.items() if v.tier == 0)
        result = reachable_techs(tech_defs, dag, all_t0)
        for tid in all_t0:
            assert tid not in result

    def test_default_empire_and_ethics_params_accepted(self):
        from stellaris_advisor.gamedata.tech_graph import load_tech_defs, reachable_techs

        tech_defs, dag = load_tech_defs(_tier_gate_raw())
        # Should not raise — empire_type and ethics are optional
        result = reachable_techs(
            tech_defs, dag, frozenset(), empire_type="democracy", ethics=frozenset({"ethic_egalitarian"})
        )
        assert isinstance(result, frozenset)


# ---------------------------------------------------------------------------
# tech_path_to
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTechPathTo:
    def _is_valid_order(
        self, path: tuple[str, ...], tech_defs: dict, researched: frozenset[str]
    ) -> bool:
        seen = set(researched)
        for tid in path:
            for prereq in tech_defs[tid].prerequisites:
                if prereq not in seen:
                    return False
            seen.add(tid)
        return True

    def test_unknown_target_returns_none(self):
        from stellaris_advisor.gamedata.tech_graph import load_tech_defs, tech_path_to

        tech_defs, dag = load_tech_defs(_chain_raw())
        assert tech_path_to(tech_defs, dag, "tech_nonexistent", frozenset()) is None

    def test_already_researched_returns_empty_tuple(self):
        from stellaris_advisor.gamedata.tech_graph import load_tech_defs, tech_path_to

        tech_defs, dag = load_tech_defs(_chain_raw())
        result = tech_path_to(tech_defs, dag, "tech_root", frozenset({"tech_root"}))
        assert result == ()

    def test_root_with_no_research(self):
        from stellaris_advisor.gamedata.tech_graph import load_tech_defs, tech_path_to

        tech_defs, dag = load_tech_defs(_chain_raw())
        path = tech_path_to(tech_defs, dag, "tech_root", frozenset())
        assert path == ("tech_root",)

    def test_chain_from_nothing(self):
        from stellaris_advisor.gamedata.tech_graph import load_tech_defs, tech_path_to

        tech_defs, dag = load_tech_defs(_chain_raw())
        path = tech_path_to(tech_defs, dag, "tech_leaf", frozenset())
        assert path is not None
        assert set(path) == {"tech_root", "tech_mid", "tech_leaf"}
        assert path.index("tech_root") < path.index("tech_mid") < path.index("tech_leaf")
        assert self._is_valid_order(path, tech_defs, frozenset())

    def test_chain_with_partial_research(self):
        from stellaris_advisor.gamedata.tech_graph import load_tech_defs, tech_path_to

        tech_defs, dag = load_tech_defs(_chain_raw())
        path = tech_path_to(tech_defs, dag, "tech_leaf", frozenset({"tech_root"}))
        assert path is not None
        assert "tech_root" not in path  # already researched
        assert "tech_mid" in path
        assert "tech_leaf" in path
        assert self._is_valid_order(path, tech_defs, frozenset({"tech_root"}))

    def test_path_excludes_all_researched_ancestors(self):
        from stellaris_advisor.gamedata.tech_graph import load_tech_defs, tech_path_to

        tech_defs, dag = load_tech_defs(_chain_raw())
        path = tech_path_to(tech_defs, dag, "tech_leaf", frozenset({"tech_root", "tech_mid"}))
        assert path == ("tech_leaf",)


# ---------------------------------------------------------------------------
# tech_tier_counts
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTechTierCounts:
    def _fixture(self) -> dict:
        from stellaris_advisor.gamedata.tech_graph import load_tech_defs

        raw = {
            "tech_p0": _raw_tech("tech_p0", area="physics", tier=0),
            "tech_p1": _raw_tech("tech_p1", area="physics", tier=1, cost=1000),
            "tech_s0": _raw_tech("tech_s0", area="society", tier=0),
            "tech_s1": _raw_tech("tech_s1", area="society", tier=1, cost=1000),
        }
        return load_tech_defs(raw)[0]

    def test_empty_researched(self):
        from stellaris_advisor.gamedata.tech_graph import tech_tier_counts

        assert tech_tier_counts(self._fixture(), frozenset()) == {}

    def test_counts_by_tier(self):
        from stellaris_advisor.gamedata.tech_graph import tech_tier_counts

        counts = tech_tier_counts(self._fixture(), frozenset({"tech_p0", "tech_p1", "tech_s0"}))
        assert counts[0] == 2  # tech_p0, tech_s0
        assert counts[1] == 1  # tech_p1

    def test_filter_by_area(self):
        from stellaris_advisor.gamedata.tech_graph import tech_tier_counts

        counts = tech_tier_counts(
            self._fixture(), frozenset({"tech_p0", "tech_s0", "tech_s1"}), area="society"
        )
        assert counts.get(0) == 1  # tech_s0 only (tech_p0 is physics)
        assert counts.get(1) == 1  # tech_s1

    def test_unknown_tech_in_researched_skipped(self):
        from stellaris_advisor.gamedata.tech_graph import tech_tier_counts

        counts = tech_tier_counts(self._fixture(), frozenset({"tech_nonexistent"}))
        assert counts == {}

    def test_empty_area_string_means_no_filter(self):
        from stellaris_advisor.gamedata.tech_graph import tech_tier_counts

        counts = tech_tier_counts(self._fixture(), frozenset({"tech_p0", "tech_s0"}), area="")
        assert counts[0] == 2


# ---------------------------------------------------------------------------
# downstream_unlocks
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDownstreamUnlocks:
    def _fixture(self):
        from stellaris_advisor.gamedata.tech_graph import load_tech_defs

        raw = {
            "tech_root": _raw_tech("tech_root", tier=0),
            "tech_mid": _raw_tech("tech_mid", tier=1, cost=1000, prerequisites="tech_root"),
            "tech_leaf": _raw_tech("tech_leaf", tier=2, cost=3000, prerequisites="tech_mid"),
            "tech_branch": _raw_tech("tech_branch", tier=1, cost=1000, prerequisites="tech_root"),
        }
        return load_tech_defs(raw)

    def test_root_unlocks_all_descendants(self):
        from stellaris_advisor.gamedata.tech_graph import downstream_unlocks

        tech_defs, dag = self._fixture()
        result = downstream_unlocks(tech_defs, dag, "tech_root")
        assert result == frozenset({"tech_mid", "tech_leaf", "tech_branch"})

    def test_mid_unlocks_leaf_only(self):
        from stellaris_advisor.gamedata.tech_graph import downstream_unlocks

        tech_defs, dag = self._fixture()
        result = downstream_unlocks(tech_defs, dag, "tech_mid")
        assert result == frozenset({"tech_leaf"})

    def test_leaf_has_no_unlocks(self):
        from stellaris_advisor.gamedata.tech_graph import downstream_unlocks

        tech_defs, dag = self._fixture()
        assert downstream_unlocks(tech_defs, dag, "tech_leaf") == frozenset()

    def test_unknown_tech_returns_empty(self):
        from stellaris_advisor.gamedata.tech_graph import downstream_unlocks

        tech_defs, dag = self._fixture()
        assert downstream_unlocks(tech_defs, dag, "tech_nonexistent") == frozenset()

    def test_returns_frozenset(self):
        from stellaris_advisor.gamedata.tech_graph import downstream_unlocks

        tech_defs, dag = self._fixture()
        result = downstream_unlocks(tech_defs, dag, "tech_root")
        assert isinstance(result, frozenset)


# ---------------------------------------------------------------------------
# Integration — real fixture files
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestLoadTechDefsFromFixtures:
    @pytest.fixture
    def physics_raw(self):
        from stellaris_advisor.parser.clausewitz import parse

        return parse(_FIXTURE_DIR.joinpath("physics.txt").read_text())

    @pytest.fixture
    def all_tech_raw(self):
        from stellaris_advisor.parser.clausewitz import parse

        combined: dict[str, Any] = {}
        for fname in ("physics.txt", "society.txt", "engineering.txt"):
            combined.update(parse(_FIXTURE_DIR.joinpath(fname).read_text()))
        return combined

    def test_loads_all_physics_techs(self, physics_raw):
        from stellaris_advisor.gamedata.tech_graph import load_tech_defs

        tech_defs, _ = load_tech_defs(physics_raw)
        for tid in ("tech_basic_science", "tech_lasers", "tech_advanced_lasers", "tech_plasma", "tech_shields"):
            assert tid in tech_defs

    def test_physics_tiers_and_flags(self, physics_raw):
        from stellaris_advisor.gamedata.tech_graph import load_tech_defs

        tech_defs, _ = load_tech_defs(physics_raw)
        assert tech_defs["tech_basic_science"].tier == 0
        assert tech_defs["tech_basic_science"].is_start_tech is True
        assert tech_defs["tech_lasers"].tier == 0
        assert tech_defs["tech_advanced_lasers"].tier == 1
        assert tech_defs["tech_plasma"].tier == 2
        assert tech_defs["tech_plasma"].is_rare is True
        assert tech_defs["tech_shields"].tier == 0

    def test_prerequisite_edges_in_dag(self, physics_raw):
        from stellaris_advisor.gamedata.tech_graph import load_tech_defs

        _, dag = load_tech_defs(physics_raw)
        assert dag.has_edge("tech_lasers", "tech_advanced_lasers")
        assert dag.has_edge("tech_advanced_lasers", "tech_plasma")

    def test_path_to_plasma(self, physics_raw):
        from stellaris_advisor.gamedata.tech_graph import load_tech_defs, tech_path_to

        tech_defs, dag = load_tech_defs(physics_raw)
        path = tech_path_to(tech_defs, dag, "tech_plasma", frozenset())
        assert path is not None
        assert {"tech_lasers", "tech_advanced_lasers", "tech_plasma"} <= set(path)
        assert path.index("tech_lasers") < path.index("tech_advanced_lasers") < path.index("tech_plasma")

    def test_downstream_of_lasers(self, physics_raw):
        from stellaris_advisor.gamedata.tech_graph import downstream_unlocks, load_tech_defs

        tech_defs, dag = load_tech_defs(physics_raw)
        result = downstream_unlocks(tech_defs, dag, "tech_lasers")
        assert "tech_advanced_lasers" in result
        assert "tech_plasma" in result

    def test_tier_gating_all15_techs(self, all_tech_raw):
        from stellaris_advisor.gamedata.tech_graph import load_tech_defs, reachable_techs

        tech_defs, dag = load_tech_defs(all_tech_raw)
        tier0_ids = frozenset(k for k, v in tech_defs.items() if v.tier == 0)
        assert len(tier0_ids) == 7  # 7 T0 techs across all areas

        # No research → only T0 techs reachable (no prerequisites)
        result = reachable_techs(tech_defs, dag, frozenset())
        assert result == tier0_ids

        # Research all 7 T0 → T1 techs with prerequisites met become reachable
        result_with_t0 = reachable_techs(tech_defs, dag, tier0_ids)
        tier1_ids = frozenset(k for k, v in tech_defs.items() if v.tier == 1)
        for tid in tier1_ids:
            assert tid in result_with_t0, f"{tid} should be reachable after all T0 researched"

    def test_society_dangerous_flag(self, all_tech_raw):
        from stellaris_advisor.gamedata.tech_graph import load_tech_defs

        tech_defs, _ = load_tech_defs(all_tech_raw)
        assert tech_defs["tech_neural_implants"].is_dangerous is True

    def test_engineering_prerequisite_chain(self, all_tech_raw):
        from stellaris_advisor.gamedata.tech_graph import load_tech_defs, tech_path_to

        tech_defs, dag = load_tech_defs(all_tech_raw)
        path = tech_path_to(tech_defs, dag, "tech_cruisers", frozenset())
        assert path is not None
        assert path.index("tech_corvettes") < path.index("tech_destroyers") < path.index("tech_cruisers")
