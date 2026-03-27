"""Pydantic v2 frozen configuration for Stellaris AI Advisor."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, field_validator


class StellarisDirConfig(BaseModel, frozen=True):
    """Paths and version info for the Stellaris installation."""

    install_dir: Path | None = None
    supported_versions: tuple[str, ...] = ("4.3",)
    save_dir: Path | None = None


class ParserConfig(BaseModel, frozen=True):
    """Clausewitz parser settings."""

    cache_parsed: bool = True
    cache_dir: Path = Path("~/.stellaris-advisor/cache/")
    empire_only: bool = False

    @field_validator("cache_dir", mode="after")
    @classmethod
    def _expand_cache_dir(cls, v: Path) -> Path:
        return v.expanduser()


class DataConfig(BaseModel, frozen=True):
    """Data pipeline directories and settings."""

    processed_dir: Path = Path("~/.stellaris-advisor/data/processed/")
    raw_dir: Path = Path("~/.stellaris-advisor/data/raw/")
    parquet_compression: str = "snappy"
    temporal_lookback: int = 5

    @field_validator("processed_dir", "raw_dir", mode="after")
    @classmethod
    def _expand_data_dirs(cls, v: Path) -> Path:
        return v.expanduser()


class RewardConfig(BaseModel, frozen=True):
    """Reward signal weights for RL training."""

    fleet_power_weight: float = 0.25
    tech_count_weight: float = 0.25
    total_income_weight: float = 0.30
    pop_count_weight: float = 0.20
    lookahead_years: int = 50


class RLConfig(BaseModel, frozen=True):
    """Offline RL training hyperparameters."""

    algorithm: str = "cql"
    learning_rate: float = 3e-4
    batch_size: int = 256
    n_epochs: int = 100
    gamma: float = 0.99
    cql_alpha: float = 1.0
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    holdout_galaxy_configs: tuple[str, ...] = ("1000_grand_admiral",)


class PlannerConfig(BaseModel, frozen=True):
    """Strategy planner tuning knobs."""

    alpha: float = 0.6
    beta: float = 0.4
    top_n_recommendations: int = 7
    top_n_colony_candidates: int = 5
    naval_utilization_threshold: float = 0.7
    threat_power_ratio: float = 1.5
    discount_rate: float = 0.05
    ascension_path_deadline_year: int = 2300
    megastructure_readiness_year: int = 2300
    megastructure_alloy_threshold: float = 200.0
    influence_crisis_threshold: float = 2.0
    faction_approval_warning: float = 20.0
    agenda_switch_progress_threshold: float = 0.7


class BenchmarkConfig(BaseModel, frozen=True):
    """Benchmark comparison settings."""

    year_bucket_size: int = 25
    percentiles: tuple[int, ...] = (25, 50, 75, 90)


class Config(BaseModel, frozen=True):
    """Top-level configuration composing all sub-configs."""

    stellaris: StellarisDirConfig = StellarisDirConfig()
    parser: ParserConfig = ParserConfig()
    data: DataConfig = DataConfig()
    reward: RewardConfig = RewardConfig()
    rl: RLConfig = RLConfig()
    planner: PlannerConfig = PlannerConfig()
    benchmark: BenchmarkConfig = BenchmarkConfig()

    @classmethod
    def default(cls) -> Config:
        """Return a Config with all default values."""
        return cls()

    @classmethod
    def from_yaml(cls, path: Path) -> Config:
        """Load configuration from a YAML file, applying defaults for missing fields.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        raw = yaml.safe_load(path.read_text()) or {}
        return cls.model_validate(raw)
