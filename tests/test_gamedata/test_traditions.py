"""Tests for gamedata/traditions.py — TDD: tests written first.

Covers:
  traditions: load_tradition_defs, load_tradition_tree_defs,
              available_traditions, tradition_tree_progress
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "game_data"


# ---------------------------------------------------------------------------
# Raw-dict helpers
# ---------------------------------------------------------------------------


def _raw_tree(
    tid: str,
    name: str = "",
    traditions: list[str] | None = None,
    adopt_effect: str = "adopt bonus",
    finish_effect: str = "finish bonus",
) -> dict[str, Any]:
    return {
        "name": name or tid,
        "traditions": traditions or [],
        "adopt_effect": adopt_effect,
        "finish_effect": finish_effect,
    }


def _raw_tradition(
    tid: str,
    tree: str = "tr_test",
    cost: int = 100,
    prerequisites: list[str] | dict | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "tree": tree,
        "cost": cost,
        "modifier": {},
    }
    if prerequisites is not None:
        data["prerequisites"] = prerequisites
    return data


# ===========================================================================
# load_tradition_defs
# ===========================================================================


@pytest.mark.unit
class TestLoadTraditionDefs:
    def test_basic_fields(self):
        from stellaris_advisor.gamedata.traditions import load_tradition_defs

        raw = {"tr_frugal": _raw_tradition("tr_frugal", tree="tr_expansion")}
        defs = load_tradition_defs(raw)
        assert "tr_frugal" in defs
        assert defs["tr_frugal"].id == "tr_frugal"
        assert defs["tr_frugal"].tree_id == "tr_expansion"

    def test_cost_extracted(self):
        from stellaris_advisor.gamedata.traditions import load_tradition_defs

        raw = {"tr_test": _raw_tradition("tr_test", cost=150)}
        defs = load_tradition_defs(raw)
        assert defs["tr_test"].cost == 150

    def test_cost_zero_when_absent(self):
        from stellaris_advisor.gamedata.traditions import load_tradition_defs

        raw = {"tr_adopt": {"tree": "tr_discovery", "modifier": {}}}
        defs = load_tradition_defs(raw)
        assert defs["tr_adopt"].cost == 0

    def test_prerequisites_as_list(self):
        from stellaris_advisor.gamedata.traditions import load_tradition_defs

        raw = {"tr_b": _raw_tradition("tr_b", prerequisites=["tr_a"])}
        defs = load_tradition_defs(raw)
        assert defs["tr_b"].prerequisites == frozenset({"tr_a"})

    def test_prerequisites_empty_dict(self):
        from stellaris_advisor.gamedata.traditions import load_tradition_defs

        raw = {"tr_a": _raw_tradition("tr_a", prerequisites={})}
        defs = load_tradition_defs(raw)
        assert defs["tr_a"].prerequisites == frozenset()

    def test_prerequisites_absent(self):
        from stellaris_advisor.gamedata.traditions import load_tradition_defs

        raw = {"tr_a": _raw_tradition("tr_a")}
        defs = load_tradition_defs(raw)
        assert defs["tr_a"].prerequisites == frozenset()

    def test_tree_entry_is_skipped(self):
        """Entries with no 'tree' key (i.e. tree defs) must not appear in tradition defs."""
        from stellaris_advisor.gamedata.traditions import load_tradition_defs

        raw = {
            "tr_expansion": _raw_tree("tr_expansion"),
            "tr_expansion_frugal": _raw_tradition("tr_expansion_frugal"),
        }
        defs = load_tradition_defs(raw)
        assert "tr_expansion" not in defs
        assert "tr_expansion_frugal" in defs

    def test_frozen_dataclass(self):
        from stellaris_advisor.gamedata.traditions import load_tradition_defs

        defs = load_tradition_defs({"tr_a": _raw_tradition("tr_a")})
        with pytest.raises(AttributeError):
            defs["tr_a"].cost = 999  # type: ignore[misc]

    def test_empty_raw_returns_empty(self):
        from stellaris_advisor.gamedata.traditions import load_tradition_defs

        assert load_tradition_defs({}) == {}


# ===========================================================================
# load_tradition_tree_defs
# ===========================================================================


@pytest.mark.unit
class TestLoadTraditionTreeDefs:
    def test_basic_fields(self):
        from stellaris_advisor.gamedata.traditions import load_tradition_tree_defs

        raw = {
            "tr_expansion": _raw_tree(
                "tr_expansion",
                name="expansion",
                adopt_effect="adopt it",
                finish_effect="finish it",
            )
        }
        defs = load_tradition_tree_defs(raw)
        assert "tr_expansion" in defs
        assert defs["tr_expansion"].id == "tr_expansion"
        assert defs["tr_expansion"].name == "expansion"

    def test_adopt_finish_effects(self):
        from stellaris_advisor.gamedata.traditions import load_tradition_tree_defs

        raw = {
            "tr_discovery": _raw_tree(
                "tr_discovery", adopt_effect="speed up", finish_effect="anomaly"
            )
        }
        defs = load_tradition_tree_defs(raw)
        assert defs["tr_discovery"].adopt_effect == "speed up"
        assert defs["tr_discovery"].finish_effect == "anomaly"

    def test_tradition_ids_extracted(self):
        from stellaris_advisor.gamedata.traditions import load_tradition_tree_defs

        raw = {
            "tr_expansion": _raw_tree(
                "tr_expansion",
                traditions=["tr_expansion_frugal", "tr_expansion_world_shaper"],
            )
        }
        defs = load_tradition_tree_defs(raw)
        assert "tr_expansion_frugal" in defs["tr_expansion"].tradition_ids
        assert "tr_expansion_world_shaper" in defs["tr_expansion"].tradition_ids

    def test_individual_tradition_is_skipped(self):
        """Entries with a 'tree' key must not appear in tree defs."""
        from stellaris_advisor.gamedata.traditions import load_tradition_tree_defs

        raw = {
            "tr_expansion": _raw_tree("tr_expansion"),
            "tr_expansion_frugal": _raw_tradition("tr_expansion_frugal"),
        }
        defs = load_tradition_tree_defs(raw)
        assert "tr_expansion" in defs
        assert "tr_expansion_frugal" not in defs

    def test_frozen_dataclass(self):
        from stellaris_advisor.gamedata.traditions import load_tradition_tree_defs

        defs = load_tradition_tree_defs({"tr_x": _raw_tree("tr_x")})
        with pytest.raises(AttributeError):
            defs["tr_x"].name = "changed"  # type: ignore[misc]

    def test_empty_raw_returns_empty(self):
        from stellaris_advisor.gamedata.traditions import load_tradition_tree_defs

        assert load_tradition_tree_defs({}) == {}


# ===========================================================================
# available_traditions
# ===========================================================================


@pytest.mark.unit
class TestAvailableTraditions:
    @pytest.fixture
    def defs(self):
        from stellaris_advisor.gamedata.traditions import load_tradition_defs

        return load_tradition_defs(
            {
                "tr_a": {"tree": "tr_tree", "cost": 100, "modifier": {}},
                "tr_b": {
                    "tree": "tr_tree",
                    "cost": 100,
                    "modifier": {},
                    "prerequisites": ["tr_a"],
                },
                "tr_c": {"tree": "tr_other", "cost": 100, "modifier": {}},
            }
        )

    def test_tradition_in_opened_tree_no_prereqs_available(self, defs):
        from stellaris_advisor.gamedata.traditions import available_traditions

        result = available_traditions(defs, frozenset({"tr_tree"}), frozenset())
        assert "tr_a" in result

    def test_tradition_not_in_opened_tree_not_available(self, defs):
        from stellaris_advisor.gamedata.traditions import available_traditions

        result = available_traditions(defs, frozenset({"tr_tree"}), frozenset())
        assert "tr_c" not in result

    def test_already_unlocked_not_available(self, defs):
        from stellaris_advisor.gamedata.traditions import available_traditions

        result = available_traditions(defs, frozenset({"tr_tree"}), frozenset({"tr_a"}))
        assert "tr_a" not in result

    def test_unmet_prereq_not_available(self, defs):
        from stellaris_advisor.gamedata.traditions import available_traditions

        result = available_traditions(defs, frozenset({"tr_tree"}), frozenset())
        assert "tr_b" not in result

    def test_met_prereq_available(self, defs):
        from stellaris_advisor.gamedata.traditions import available_traditions

        result = available_traditions(defs, frozenset({"tr_tree"}), frozenset({"tr_a"}))
        assert "tr_b" in result


# ===========================================================================
# tradition_tree_progress
# ===========================================================================


@pytest.mark.unit
class TestTraditionTreeProgress:
    @pytest.fixture
    def tree(self):
        from stellaris_advisor.gamedata.traditions import load_tradition_tree_defs

        raw = {
            "tr_tree": _raw_tree("tr_tree", traditions=["tr_a", "tr_b", "tr_c", "tr_d"]),
            "tr_a": _raw_tradition("tr_a"),
            "tr_b": _raw_tradition("tr_b"),
            "tr_c": _raw_tradition("tr_c"),
            "tr_d": _raw_tradition("tr_d"),
        }
        return load_tradition_tree_defs(raw)["tr_tree"]

    def test_progress_zero_for_none_unlocked(self, tree):
        from stellaris_advisor.gamedata.traditions import tradition_tree_progress

        assert tradition_tree_progress(tree, frozenset()) == (0, 4)

    def test_progress_full_for_all_unlocked(self, tree):
        from stellaris_advisor.gamedata.traditions import tradition_tree_progress

        assert tradition_tree_progress(tree, frozenset({"tr_a", "tr_b", "tr_c", "tr_d"})) == (4, 4)

    def test_progress_half_for_half_unlocked(self, tree):
        from stellaris_advisor.gamedata.traditions import tradition_tree_progress

        assert tradition_tree_progress(tree, frozenset({"tr_a", "tr_b"})) == (2, 4)

    def test_empty_tree_returns_zero_total(self):
        from stellaris_advisor.gamedata.traditions import (
            load_tradition_tree_defs,
            tradition_tree_progress,
        )

        raw = {"tr_empty": _raw_tree("tr_empty", traditions=[])}
        empty_tree = load_tradition_tree_defs(raw)["tr_empty"]
        assert tradition_tree_progress(empty_tree, frozenset()) == (0, 0)

    def test_progress_independent_per_tree(self):
        from stellaris_advisor.gamedata.traditions import (
            load_tradition_tree_defs,
            tradition_tree_progress,
        )

        raw = {
            "tr_t1": _raw_tree("tr_t1", traditions=["tr_a"]),
            "tr_t2": _raw_tree("tr_t2", traditions=["tr_b"]),
            "tr_a": _raw_tradition("tr_a", tree="tr_t1"),
            "tr_b": _raw_tradition("tr_b", tree="tr_t2"),
        }
        tree_defs = load_tradition_tree_defs(raw)
        assert tradition_tree_progress(tree_defs["tr_t1"], frozenset({"tr_a"})) == (1, 1)
        assert tradition_tree_progress(tree_defs["tr_t2"], frozenset({"tr_a"})) == (0, 1)


# ===========================================================================
# Integration — real traditions fixture
# ===========================================================================


@pytest.mark.integration
class TestTraditionDefsFromFixtures:
    @pytest.fixture
    def raw(self):
        from stellaris_advisor.parser.clausewitz import parse

        return parse(_FIXTURE_DIR.joinpath("traditions.txt").read_text())

    def test_loads_tradition_trees(self, raw):
        from stellaris_advisor.gamedata.traditions import load_tradition_tree_defs

        tree_defs = load_tradition_tree_defs(raw)
        assert "tr_discovery" in tree_defs
        assert "tr_expansion" in tree_defs

    def test_tree_adopt_finish_effects_non_empty(self, raw):
        from stellaris_advisor.gamedata.traditions import load_tradition_tree_defs

        tree_defs = load_tradition_tree_defs(raw)
        assert tree_defs["tr_discovery"].adopt_effect != ""
        assert tree_defs["tr_discovery"].finish_effect != ""

    def test_loads_individual_traditions(self, raw):
        from stellaris_advisor.gamedata.traditions import load_tradition_defs

        trad_defs = load_tradition_defs(raw)
        assert "tr_discovery_exploration" in trad_defs
        assert "tr_expansion_frugal_expansion" in trad_defs

    def test_tradition_has_correct_tree_id(self, raw):
        from stellaris_advisor.gamedata.traditions import load_tradition_defs

        trad_defs = load_tradition_defs(raw)
        assert trad_defs["tr_discovery_exploration"].tree_id == "tr_discovery"

    def test_tradition_with_prerequisites(self, raw):
        from stellaris_advisor.gamedata.traditions import load_tradition_defs

        trad_defs = load_tradition_defs(raw)
        assert (
            "tr_discovery_exploration"
            in trad_defs["tr_discovery_scientific_method"].prerequisites
        )

    def test_tradition_cost(self, raw):
        from stellaris_advisor.gamedata.traditions import load_tradition_defs

        trad_defs = load_tradition_defs(raw)
        assert trad_defs["tr_expansion_frugal_expansion"].cost == 100

    def test_tree_tradition_ids_populated(self, raw):
        from stellaris_advisor.gamedata.traditions import load_tradition_tree_defs

        tree_defs = load_tradition_tree_defs(raw)
        assert "tr_discovery_exploration" in tree_defs["tr_discovery"].tradition_ids

    def test_trees_not_in_tradition_defs(self, raw):
        """Tree-level entries must not appear in tradition defs."""
        from stellaris_advisor.gamedata.traditions import load_tradition_defs

        trad_defs = load_tradition_defs(raw)
        assert "tr_discovery" not in trad_defs
        assert "tr_expansion" not in trad_defs
