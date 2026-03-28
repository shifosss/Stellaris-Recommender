"""Tests for Task 5 helper functions: _extract_game_metadata, _sort_saves_by_date, _generate_game_id."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from stellaris_advisor.data.ingest import (
    _extract_game_metadata,
    _generate_game_id,
    _sort_saves_by_date,
)
from stellaris_advisor.parser import parse_save

SAMPLE_SAVE = Path(__file__).parent.parent / "fixtures" / "sample_save_small.sav"


# ---------------------------------------------------------------------------
# Tests — _extract_game_metadata
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestExtractGameMetadata:
    def test_returns_dict(self) -> None:
        galactic = parse_save(str(SAMPLE_SAVE))
        result = _extract_game_metadata(galactic, "g1")
        assert isinstance(result, dict)

    def test_game_id_matches_argument(self) -> None:
        galactic = parse_save(str(SAMPLE_SAVE))
        result = _extract_game_metadata(galactic, "my_game")
        assert result["game_id"] == "my_game"

    def test_year_matches_galactic_year(self) -> None:
        galactic = parse_save(str(SAMPLE_SAVE))
        result = _extract_game_metadata(galactic, "g1")
        assert result["year"] == galactic.year

    def test_num_empires_matches_galactic_empires(self) -> None:
        galactic = parse_save(str(SAMPLE_SAVE))
        result = _extract_game_metadata(galactic, "g1")
        assert result["num_empires"] == len(galactic.empires)

    def test_player_empire_id_matches_galactic(self) -> None:
        galactic = parse_save(str(SAMPLE_SAVE))
        result = _extract_game_metadata(galactic, "g1")
        assert result["player_empire_id"] == galactic.player_empire_id

    def test_crisis_type_is_empty_string_when_none(self) -> None:
        galactic = parse_save(str(SAMPLE_SAVE))
        result = _extract_game_metadata(galactic, "g1")
        # crisis_type is None in the sample save → should be ""
        if galactic.crisis_type is None:
            assert result["crisis_type"] == ""

    def test_contains_all_expected_keys(self) -> None:
        galactic = parse_save(str(SAMPLE_SAVE))
        result = _extract_game_metadata(galactic, "g1")
        expected_keys = {
            "game_id",
            "game_version",
            "galaxy_size",
            "difficulty",
            "crisis_active",
            "crisis_type",
            "war_in_heaven",
            "galactic_market_founded",
            "player_empire_id",
            "num_empires",
            "year",
        }
        assert expected_keys.issubset(result.keys())


# ---------------------------------------------------------------------------
# Tests — _generate_game_id
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGenerateGameId:
    def test_returns_nonempty_string(self, tmp_path: Path) -> None:
        result = _generate_game_id(tmp_path)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_deterministic_same_path(self, tmp_path: Path) -> None:
        assert _generate_game_id(tmp_path) == _generate_game_id(tmp_path)

    def test_different_paths_produce_different_ids(self, tmp_path: Path) -> None:
        id_a = _generate_game_id(tmp_path / "folder_a")
        id_b = _generate_game_id(tmp_path / "folder_b")
        assert id_a != id_b

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        result_path = _generate_game_id(tmp_path)
        result_str = _generate_game_id(str(tmp_path))
        assert result_path == result_str

    def test_result_is_hex_string(self, tmp_path: Path) -> None:
        result = _generate_game_id(tmp_path)
        assert all(c in "0123456789abcdef" for c in result)


# ---------------------------------------------------------------------------
# Tests — _sort_saves_by_date
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSortSavesByDate:
    def test_empty_folder_returns_empty_list(self, tmp_path: Path) -> None:
        assert _sort_saves_by_date(tmp_path) == []

    def test_returns_list_of_tuples(self, tmp_path: Path) -> None:
        shutil.copy(SAMPLE_SAVE, tmp_path / "save_01.sav")
        result = _sort_saves_by_date(tmp_path)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_tuple_has_path_year_month(self, tmp_path: Path) -> None:
        shutil.copy(SAMPLE_SAVE, tmp_path / "save_01.sav")
        path, year, month = _sort_saves_by_date(tmp_path)[0]
        assert isinstance(path, Path)
        assert isinstance(year, int)
        assert isinstance(month, int)

    def test_year_matches_parsed_save(self, tmp_path: Path) -> None:
        shutil.copy(SAMPLE_SAVE, tmp_path / "save_01.sav")
        galactic = parse_save(str(SAMPLE_SAVE))
        _, year, month = _sort_saves_by_date(tmp_path)[0]
        assert year == galactic.year
        assert month == galactic.month

    def test_corrupted_save_is_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "bad.sav").write_bytes(b"not a zip file")
        shutil.copy(SAMPLE_SAVE, tmp_path / "valid.sav")
        result = _sort_saves_by_date(tmp_path)
        assert len(result) == 1

    def test_two_identical_saves_returns_two_tuples(self, tmp_path: Path) -> None:
        shutil.copy(SAMPLE_SAVE, tmp_path / "save_01.sav")
        shutil.copy(SAMPLE_SAVE, tmp_path / "save_02.sav")
        result = _sort_saves_by_date(tmp_path)
        assert len(result) == 2

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        shutil.copy(SAMPLE_SAVE, tmp_path / "save_01.sav")
        result = _sort_saves_by_date(str(tmp_path))
        assert len(result) == 1

    def test_result_sorted_by_year_then_month(self, tmp_path: Path) -> None:
        shutil.copy(SAMPLE_SAVE, tmp_path / "a.sav")
        shutil.copy(SAMPLE_SAVE, tmp_path / "b.sav")
        result = _sort_saves_by_date(tmp_path)
        years_months = [(r[1], r[2]) for r in result]
        assert years_months == sorted(years_months)
