"""Tests for synthetic data generation: SyntheticConfig + generator functions."""

from __future__ import annotations

from pathlib import Path

import pytest

from stellaris_advisor.data.synthetic import (
    SyntheticConfig,
    generate_launch_script,
    generate_orchestrator_config,
    generate_random_config,
)


# ---------------------------------------------------------------------------
# Tests — SyntheticConfig dataclass
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSyntheticConfig:
    def test_is_frozen(self) -> None:
        cfg = SyntheticConfig()
        with pytest.raises(Exception):
            cfg.galaxy_size = 800  # type: ignore[misc]

    def test_default_galaxy_size(self) -> None:
        assert SyntheticConfig().galaxy_size == 600

    def test_default_num_empires(self) -> None:
        assert SyntheticConfig().num_empires == 15

    def test_default_difficulty(self) -> None:
        assert SyntheticConfig().difficulty == "captain"

    def test_default_crisis_strength(self) -> None:
        assert SyntheticConfig().crisis_strength == pytest.approx(1.0)

    def test_default_hyperlane_density(self) -> None:
        assert SyntheticConfig().hyperlane_density == pytest.approx(1.0)

    def test_default_mid_game_start(self) -> None:
        assert SyntheticConfig().mid_game_start == 2275

    def test_default_end_game_start(self) -> None:
        assert SyntheticConfig().end_game_start == 2375

    def test_default_autosave_interval(self) -> None:
        assert SyntheticConfig().autosave_interval == 12

    def test_custom_values_accepted(self) -> None:
        cfg = SyntheticConfig(galaxy_size=400, num_empires=8, difficulty="ensign")
        assert cfg.galaxy_size == 400
        assert cfg.num_empires == 8
        assert cfg.difficulty == "ensign"


# ---------------------------------------------------------------------------
# Tests — generate_random_config
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGenerateRandomConfig:
    def test_returns_synthetic_config(self) -> None:
        assert isinstance(generate_random_config(), SyntheticConfig)

    def test_galaxy_size_in_range(self) -> None:
        for _ in range(20):
            cfg = generate_random_config()
            assert 400 <= cfg.galaxy_size <= 1000

    def test_num_empires_in_range(self) -> None:
        for _ in range(20):
            cfg = generate_random_config()
            assert 8 <= cfg.num_empires <= 30

    def test_crisis_strength_in_range(self) -> None:
        for _ in range(20):
            cfg = generate_random_config()
            assert 1.0 <= cfg.crisis_strength <= 5.0

    def test_hyperlane_density_positive(self) -> None:
        for _ in range(20):
            cfg = generate_random_config()
            assert cfg.hyperlane_density > 0.0

    def test_difficulty_is_valid_string(self) -> None:
        valid = {
            "civilian",
            "ensign",
            "captain",
            "commodore",
            "admiral",
            "grand_admiral",
        }
        for _ in range(20):
            cfg = generate_random_config()
            assert cfg.difficulty in valid

    def test_produces_varied_configs(self) -> None:
        # Extremely unlikely all 10 configs are identical
        configs = [generate_random_config() for _ in range(10)]
        unique = {(c.galaxy_size, c.num_empires, c.difficulty) for c in configs}
        assert len(unique) > 1

    def test_mid_game_start_in_range(self) -> None:
        for _ in range(20):
            cfg = generate_random_config()
            assert 2250 <= cfg.mid_game_start <= 2300

    def test_end_game_start_in_range(self) -> None:
        for _ in range(20):
            cfg = generate_random_config()
            assert 2350 <= cfg.end_game_start <= 2400

    def test_end_game_start_after_mid_game_start(self) -> None:
        for _ in range(20):
            cfg = generate_random_config()
            assert cfg.end_game_start > cfg.mid_game_start

    def test_result_is_frozen(self) -> None:
        cfg = generate_random_config()
        with pytest.raises(Exception):
            cfg.galaxy_size = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Tests — generate_launch_script
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGenerateLaunchScript:
    def test_returns_string(self) -> None:
        assert isinstance(generate_launch_script(SyntheticConfig()), str)

    def test_nonempty(self) -> None:
        assert len(generate_launch_script(SyntheticConfig())) > 0

    def test_contains_observer_flag(self) -> None:
        script = generate_launch_script(SyntheticConfig())
        assert "observer" in script.lower()

    def test_contains_galaxy_size(self) -> None:
        cfg = SyntheticConfig(galaxy_size=800)
        assert "800" in generate_launch_script(cfg)

    def test_contains_num_empires(self) -> None:
        cfg = SyntheticConfig(num_empires=20)
        assert "20" in generate_launch_script(cfg)

    def test_contains_difficulty(self) -> None:
        cfg = SyntheticConfig(difficulty="admiral")
        assert "admiral" in generate_launch_script(cfg)

    def test_contains_autosave_interval(self) -> None:
        cfg = SyntheticConfig(autosave_interval=6)
        assert "6" in generate_launch_script(cfg)

    def test_is_bash_script(self) -> None:
        script = generate_launch_script(SyntheticConfig())
        assert script.startswith("#!")


# ---------------------------------------------------------------------------
# Tests — generate_orchestrator_config
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGenerateOrchestratorConfig:
    def test_returns_dict(self, tmp_path: Path) -> None:
        result = generate_orchestrator_config(3, tmp_path)
        assert isinstance(result, dict)

    def test_has_num_runs_key(self, tmp_path: Path) -> None:
        result = generate_orchestrator_config(5, tmp_path)
        assert result["num_runs"] == 5

    def test_has_output_dir_key(self, tmp_path: Path) -> None:
        result = generate_orchestrator_config(2, tmp_path)
        assert "output_dir" in result
        assert str(tmp_path) in str(result["output_dir"])

    def test_has_configs_key(self, tmp_path: Path) -> None:
        result = generate_orchestrator_config(3, tmp_path)
        assert "configs" in result

    def test_configs_length_matches_num_runs(self, tmp_path: Path) -> None:
        result = generate_orchestrator_config(4, tmp_path)
        assert len(result["configs"]) == 4

    def test_each_config_is_dict(self, tmp_path: Path) -> None:
        result = generate_orchestrator_config(2, tmp_path)
        for cfg in result["configs"]:
            assert isinstance(cfg, dict)

    def test_each_config_has_galaxy_size(self, tmp_path: Path) -> None:
        result = generate_orchestrator_config(2, tmp_path)
        for cfg in result["configs"]:
            assert "galaxy_size" in cfg

    def test_zero_runs_returns_empty_configs(self, tmp_path: Path) -> None:
        result = generate_orchestrator_config(0, tmp_path)
        assert result["num_runs"] == 0
        assert result["configs"] == []

    def test_output_dir_accepts_path_object(self, tmp_path: Path) -> None:
        result = generate_orchestrator_config(1, tmp_path)
        assert result is not None
