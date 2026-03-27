"""Tests for configuration module."""

import pytest
from pathlib import Path


@pytest.mark.unit
class TestConfig:
    def test_default_config_creates_successfully(self):
        from stellaris_advisor.config import Config

        config = Config.default()
        assert config.stellaris.supported_versions == ("4.3",)
        assert config.parser.cache_parsed is True
        assert config.planner.alpha == 0.6
        assert config.planner.beta == 0.4
        assert config.rl.algorithm == "cql"

    def test_config_is_immutable(self):
        from stellaris_advisor.config import Config

        config = Config.default()
        with pytest.raises(Exception):
            config.planner.alpha = 0.9

    def test_config_from_yaml(self, tmp_path: Path):
        from stellaris_advisor.config import Config

        yaml_content = """
stellaris:
  supported_versions: ["4.3"]
planner:
  alpha: 0.7
  beta: 0.3
  top_n_recommendations: 10
"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content)
        config = Config.from_yaml(yaml_file)
        assert config.planner.alpha == 0.7
        assert config.planner.beta == 0.3
        assert config.planner.top_n_recommendations == 10
        assert config.parser.cache_parsed is True  # defaults still apply

    def test_config_from_yaml_missing_file_raises(self):
        from stellaris_advisor.config import Config

        with pytest.raises(FileNotFoundError):
            Config.from_yaml(Path("/nonexistent/config.yaml"))

    def test_reward_weights_accessible(self):
        from stellaris_advisor.config import Config

        config = Config.default()
        assert config.reward.fleet_power_weight == 0.25
        assert config.reward.tech_count_weight == 0.25
        assert config.reward.total_income_weight == 0.30
        assert config.reward.pop_count_weight == 0.20
