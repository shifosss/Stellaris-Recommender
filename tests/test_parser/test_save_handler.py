"""Tests for save file handler."""

import pytest
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.mark.integration
class TestParseSave:
    def test_parse_synthetic_save(self):
        from stellaris_advisor.parser import parse_save

        save_path = FIXTURES_DIR / "sample_save_small.sav"
        state = parse_save(str(save_path))
        assert state.year == 2250
        assert state.game_version == "4.3"
        assert state.player_empire_id == 0
        assert state.player_empire.name == "United Nations of Earth"
        assert state.player_empire.authority == "auth_democratic"
        assert "tech_corvettes" in state.player_empire.researched_techs
        assert state.player_empire.resource_income["energy"] == pytest.approx(142.5)
        assert len(state.empires) == 2
        assert len(state.systems) == 2

    def test_parse_save_invalid_path_raises(self):
        from stellaris_advisor.parser import parse_save

        with pytest.raises(FileNotFoundError):
            parse_save("/nonexistent/save.sav")

    def test_parse_save_extracts_planets(self):
        from stellaris_advisor.parser import parse_save

        save_path = FIXTURES_DIR / "sample_save_small.sav"
        state = parse_save(str(save_path))
        empire = state.player_empire
        assert len(empire.owned_planets) >= 1
        earth = next((p for p in empire.owned_planets if p.name == "Earth"), None)
        assert earth is not None
        assert earth.planet_class == "pc_continental"
        assert earth.size == 25
        assert earth.is_capital is True


@pytest.mark.integration
class TestParseGameFiles:
    def test_parse_game_files(self, tmp_path: Path):
        from stellaris_advisor.parser import parse_game_files

        tech_dir = tmp_path / "common" / "technology"
        tech_dir.mkdir(parents=True)
        (tech_dir / "00_physics.txt").write_text(
            "tech_lasers_1={ area=physics tier=0 cost=100 weight=100 "
            "prerequisites={} }\n"
            "tech_lasers_2={ area=physics tier=1 cost=500 weight=80 "
            'prerequisites={ "tech_lasers_1" } }\n'
        )
        result = parse_game_files(str(tmp_path))
        assert "technology" in result
        assert "tech_lasers_1" in result["technology"]
        assert result["technology"]["tech_lasers_2"]["cost"] == 500
