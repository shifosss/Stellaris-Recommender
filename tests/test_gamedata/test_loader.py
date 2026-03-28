"""Tests for gamedata/loader.py — TDD: tests written first.

Covers:
  GameKnowledgeGraph attributes and construction
  GameKnowledgeGraph.load() orchestration
  Pickle caching behaviour
  Query delegation methods
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import networkx as nx
import pytest

# ---------------------------------------------------------------------------
# Minimal Clausewitz-format fixture content
# ---------------------------------------------------------------------------

_TECH_TXT = """\
tech_basic_science = {
    tier = 0
    area = physics
    cost = 100
    category = particles
}
"""

_BUILDING_TXT = """\
building_research_lab = {
    prerequisites = tech_basic_science
    cost = { minerals = 150 }
    upkeep = { energy = 2 }
}
"""

_DISTRICT_TXT = """\
district_city = {
    cost = { minerals = 300 }
    upkeep = { energy = 5 }
}
"""


def _make_stellaris_dir(tmp_path: Path) -> Path:
    """Write a minimal Stellaris common/ directory for integration tests."""
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

    # Empty dirs for all other data types (loaders handle missing gracefully)
    for subdir in (
        "jobs",
        "ship_sizes",
        "planet_classes",
        "traditions",
        "tradition_categories",
        "ascension_perks",
        "megastructures",
        "policies",
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


# ===========================================================================
# GameKnowledgeGraph — construction and attribute tests
# ===========================================================================


@pytest.mark.unit
class TestGameKnowledgeGraph:
    @pytest.fixture
    def empty_gkg(self):
        from stellaris_advisor.gamedata.loader import GameKnowledgeGraph

        return GameKnowledgeGraph(
            tech_defs=MappingProxyType({}),
            tech_dag=nx.DiGraph(),
            building_defs=MappingProxyType({}),
            district_defs=MappingProxyType({}),
            job_defs=MappingProxyType({}),
            ship_class_defs=MappingProxyType({}),
            planet_class_defs=MappingProxyType({}),
            tradition_trees=MappingProxyType({}),
            tradition_defs=MappingProxyType({}),
            ascension_perk_defs=MappingProxyType({}),
            megastructure_defs=MappingProxyType({}),
            policy_defs=MappingProxyType({}),
            faction_defs=MappingProxyType({}),
            council_position_defs=MappingProxyType({}),
            agenda_defs=MappingProxyType({}),
            federation_type_defs=MappingProxyType({}),
            civic_defs=MappingProxyType({}),
            trait_defs=MappingProxyType({}),
            edict_defs=MappingProxyType({}),
            validation_results=(),
            version="1.0",
        )

    def test_version_attr(self, empty_gkg):
        assert empty_gkg.version == "1.0"

    def test_validation_results_is_tuple(self, empty_gkg):
        assert isinstance(empty_gkg.validation_results, tuple)

    def test_tech_dag_is_digraph(self, empty_gkg):
        assert isinstance(empty_gkg.tech_dag, nx.DiGraph)

    def test_dict_attrs_are_mapping(self, empty_gkg):
        for attr in (
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
        ):
            assert isinstance(getattr(empty_gkg, attr), Mapping), attr

    def test_dict_attrs_are_mapping_proxy(self, empty_gkg):
        assert isinstance(empty_gkg.tech_defs, MappingProxyType)
        assert isinstance(empty_gkg.building_defs, MappingProxyType)

    def test_mapping_proxy_is_read_only(self, empty_gkg):
        with pytest.raises(TypeError):
            empty_gkg.tech_defs["new_tech"] = None  # type: ignore[index]


# ===========================================================================
# GameKnowledgeGraph.load() — orchestration
# ===========================================================================


@pytest.mark.integration
class TestLoad:
    def test_load_returns_game_knowledge_graph(self, tmp_path):
        from stellaris_advisor.gamedata.loader import GameKnowledgeGraph

        stellaris = _make_stellaris_dir(tmp_path)
        gkg = GameKnowledgeGraph.load(
            stellaris, "1.0", cache_dir=tmp_path / "cache"
        )
        assert isinstance(gkg, GameKnowledgeGraph)

    def test_load_sets_version(self, tmp_path):
        from stellaris_advisor.gamedata.loader import GameKnowledgeGraph

        stellaris = _make_stellaris_dir(tmp_path)
        gkg = GameKnowledgeGraph.load(
            stellaris, "3.14", cache_dir=tmp_path / "cache"
        )
        assert gkg.version == "3.14"

    def test_load_populates_tech_defs(self, tmp_path):
        from stellaris_advisor.gamedata.loader import GameKnowledgeGraph

        stellaris = _make_stellaris_dir(tmp_path)
        gkg = GameKnowledgeGraph.load(
            stellaris, "1.0", cache_dir=tmp_path / "cache"
        )
        assert "tech_basic_science" in gkg.tech_defs

    def test_load_populates_building_defs(self, tmp_path):
        from stellaris_advisor.gamedata.loader import GameKnowledgeGraph

        stellaris = _make_stellaris_dir(tmp_path)
        gkg = GameKnowledgeGraph.load(
            stellaris, "1.0", cache_dir=tmp_path / "cache"
        )
        assert "building_research_lab" in gkg.building_defs

    def test_load_applies_cross_linking(self, tmp_path):
        """TechNode.unlocks_buildings must be populated after load."""
        from stellaris_advisor.gamedata.loader import GameKnowledgeGraph

        stellaris = _make_stellaris_dir(tmp_path)
        gkg = GameKnowledgeGraph.load(
            stellaris, "1.0", cache_dir=tmp_path / "cache"
        )
        tech = gkg.tech_defs.get("tech_basic_science")
        assert tech is not None
        assert "building_research_lab" in tech.unlocks_buildings

    def test_load_produces_validation_results(self, tmp_path):
        from stellaris_advisor.gamedata.loader import GameKnowledgeGraph

        stellaris = _make_stellaris_dir(tmp_path)
        gkg = GameKnowledgeGraph.load(
            stellaris, "1.0", cache_dir=tmp_path / "cache"
        )
        assert len(gkg.validation_results) == 11

    def test_load_missing_dirs_returns_empty_defs(self, tmp_path):
        """An empty (but valid) stellaris_dir returns empty defs without error."""
        from stellaris_advisor.gamedata.loader import GameKnowledgeGraph

        stellaris = tmp_path / "empty_stellaris"
        stellaris.mkdir()
        gkg = GameKnowledgeGraph.load(
            stellaris, "0.0", cache_dir=tmp_path / "cache"
        )
        assert len(gkg.tech_defs) == 0
        assert len(gkg.building_defs) == 0

    def test_load_dict_attrs_are_mapping_proxy(self, tmp_path):
        from stellaris_advisor.gamedata.loader import GameKnowledgeGraph

        stellaris = _make_stellaris_dir(tmp_path)
        gkg = GameKnowledgeGraph.load(
            stellaris, "1.0", cache_dir=tmp_path / "cache"
        )
        assert isinstance(gkg.tech_defs, MappingProxyType)
        assert isinstance(gkg.building_defs, MappingProxyType)


# ===========================================================================
# Pickle caching
# ===========================================================================


@pytest.mark.integration
class TestCaching:
    @pytest.fixture
    def stellaris(self, tmp_path):
        return _make_stellaris_dir(tmp_path)

    def test_cache_file_created_after_load(self, stellaris, tmp_path):
        from stellaris_advisor.gamedata.loader import GameKnowledgeGraph

        cache_dir = tmp_path / "cache"
        GameKnowledgeGraph.load(stellaris, "1.0", cache_dir=cache_dir)
        pkl_files = list(cache_dir.glob("*.pkl"))
        assert len(pkl_files) == 1

    def test_second_load_reads_from_cache(self, stellaris, tmp_path):
        """Second load with same files returns cached result (no new pkl written)."""
        from stellaris_advisor.gamedata.loader import GameKnowledgeGraph

        cache_dir = tmp_path / "cache"
        GameKnowledgeGraph.load(stellaris, "1.0", cache_dir=cache_dir)
        # Second call: same dir, same version, same mtimes → cache hit
        GameKnowledgeGraph.load(stellaris, "1.0", cache_dir=cache_dir)
        pkl_files = list(cache_dir.glob("*.pkl"))
        assert len(pkl_files) == 1

    def test_file_change_creates_new_cache_file(self, stellaris, tmp_path):
        """Modifying a file changes the content hash → new cache entry."""
        from stellaris_advisor.gamedata.loader import GameKnowledgeGraph

        cache_dir = tmp_path / "cache"
        GameKnowledgeGraph.load(stellaris, "1.0", cache_dir=cache_dir)
        # Overwrite a file: mtime changes → new hash → new pkl
        (stellaris / "common" / "technology" / "00_tech.txt").write_text(
            "tech_extra = { tier = 0 area = physics cost = 500 category = particles }\n",
            encoding="utf-8",
        )
        GameKnowledgeGraph.load(stellaris, "1.0", cache_dir=cache_dir)
        pkl_files = list(cache_dir.glob("*.pkl"))
        assert len(pkl_files) == 2

    def test_different_version_creates_separate_cache_file(self, stellaris, tmp_path):
        from stellaris_advisor.gamedata.loader import GameKnowledgeGraph

        cache_dir = tmp_path / "cache"
        GameKnowledgeGraph.load(stellaris, "1.0", cache_dir=cache_dir)
        GameKnowledgeGraph.load(stellaris, "2.0", cache_dir=cache_dir)
        pkl_files = list(cache_dir.glob("*.pkl"))
        assert len(pkl_files) == 2

    def test_no_cache_when_cache_dir_is_none(self, stellaris, tmp_path):
        from stellaris_advisor.gamedata.loader import GameKnowledgeGraph

        default_cache = tmp_path / "no_cache"
        GameKnowledgeGraph.load(stellaris, "1.0", cache_dir=None)
        # Nothing written to any directory we control
        assert not default_cache.exists()

    def test_cache_dir_created_if_missing(self, stellaris, tmp_path):
        from stellaris_advisor.gamedata.loader import GameKnowledgeGraph

        cache_dir = tmp_path / "new" / "nested" / "cache"
        assert not cache_dir.exists()
        GameKnowledgeGraph.load(stellaris, "1.0", cache_dir=cache_dir)
        assert cache_dir.exists()

    def test_cache_filename_contains_version_and_hash(self, stellaris, tmp_path):
        from stellaris_advisor.gamedata.loader import GameKnowledgeGraph

        cache_dir = tmp_path / "cache"
        GameKnowledgeGraph.load(stellaris, "3.99", cache_dir=cache_dir)
        pkl_files = list(cache_dir.glob("*.pkl"))
        assert len(pkl_files) == 1
        assert pkl_files[0].name.startswith("gamedata-3.99-")


# ===========================================================================
# Query delegation methods
# ===========================================================================


@pytest.mark.integration
class TestQueryDelegation:
    @pytest.fixture
    def gkg(self, tmp_path):
        from stellaris_advisor.gamedata.loader import GameKnowledgeGraph

        return GameKnowledgeGraph.load(_make_stellaris_dir(tmp_path), "1.0")

    def test_reachable_techs_returns_frozenset(self, gkg):
        result = gkg.reachable_techs(frozenset())
        assert isinstance(result, frozenset)

    def test_reachable_techs_includes_tier0(self, gkg):
        # tech_basic_science is tier-0; reachable with no prereqs
        result = gkg.reachable_techs(frozenset())
        assert "tech_basic_science" in result

    def test_available_buildings_returns_frozenset(self, gkg):
        result = gkg.available_buildings(frozenset())
        assert isinstance(result, frozenset)

    def test_available_buildings_with_prereq_researched(self, gkg):
        # building_research_lab requires tech_basic_science
        result = gkg.available_buildings(frozenset({"tech_basic_science"}))
        assert "building_research_lab" in result

    def test_available_districts_returns_frozenset(self, gkg):
        result = gkg.available_districts()
        assert isinstance(result, frozenset)

    def test_eligible_perks_returns_frozenset(self, gkg):
        result = gkg.eligible_perks(frozenset(), frozenset(), 0)
        assert isinstance(result, frozenset)

    def test_tech_path_to_unknown_returns_none(self, gkg):
        result = gkg.tech_path_to("nonexistent_tech", frozenset())
        assert result is None

    def test_tech_path_to_known_returns_tuple(self, gkg):
        result = gkg.tech_path_to("tech_basic_science", frozenset())
        assert isinstance(result, tuple)

    def test_downstream_unlocks_returns_frozenset(self, gkg):
        result = gkg.downstream_unlocks("tech_basic_science")
        assert isinstance(result, frozenset)

    def test_downstream_unlocks_unknown_returns_empty(self, gkg):
        result = gkg.downstream_unlocks("nonexistent_tech")
        assert result == frozenset()


# ===========================================================================
# __init__ exports
# ===========================================================================


@pytest.mark.unit
def test_game_knowledge_graph_importable_from_gamedata():
    from stellaris_advisor.gamedata import GameKnowledgeGraph  # noqa: F401
