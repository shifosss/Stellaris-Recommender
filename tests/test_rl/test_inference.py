"""Tests for RL model inference."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.mark.unit
def test_is_available_without_numpy(tmp_path: Path) -> None:
    """is_available() must not import numpy and must not crash when numpy is absent."""
    # Ensure the module is not cached from a previous import that had numpy present.
    sys.modules.pop("stellaris_advisor.rl.inference", None)

    with patch.dict(sys.modules, {"numpy": None}):
        from stellaris_advisor.rl.inference import RLPredictor

        result = RLPredictor.is_available(tmp_path / "nonexistent")

    assert result is False

    # Clean up so later tests get a fresh import.
    sys.modules.pop("stellaris_advisor.rl.inference", None)


@pytest.mark.unit
class TestIsAvailable:
    def test_missing_dir_returns_false(self, tmp_path: Path) -> None:
        from stellaris_advisor.rl.inference import RLPredictor

        assert not RLPredictor.is_available(tmp_path / "nonexistent")

    def test_empty_dir_returns_false(self, tmp_path: Path) -> None:
        from stellaris_advisor.rl.inference import RLPredictor

        assert not RLPredictor.is_available(tmp_path)

    def test_valid_dir_returns_true(self, tmp_path: Path) -> None:
        from stellaris_advisor.rl.inference import RLPredictor

        (tmp_path / "model.d3").touch()
        (tmp_path / "encoder_stats.json").write_text(
            '{"state_dim":3,"mean":[0,0,0],"std":[1,1,1],"var":[1,1,1],"count":1}'
        )
        assert RLPredictor.is_available(tmp_path)


@pytest.mark.slow
class TestRLPredictor:
    def test_predict_value_returns_float(
        self,
        tiny_transitions_parquet: Path,
        tmp_path: Path,
    ) -> None:
        pytest.importorskip("d3rlpy")
        from stellaris_advisor.config import RLConfig
        from stellaris_advisor.features.registry import (
            EMPIRE_FEATURE_NAMES,
            TEMPORAL_FEATURE_NAMES,
        )
        from stellaris_advisor.rl.dataset import load_transitions
        from stellaris_advisor.rl.inference import RLPredictor
        from stellaris_advisor.rl.trainer import train

        config = RLConfig(algorithm="cql", n_epochs=1, batch_size=4)
        arrays = load_transitions(tiny_transitions_parquet, ["game_0"])
        model_dir = tmp_path / "model"
        train(
            config=config,
            train_arrays=arrays,
            val_arrays=arrays,
            output_dir=model_dir,
            device="cpu:0",
            n_steps=3,
        )

        predictor = RLPredictor(model_dir)
        state_dim = len(EMPIRE_FEATURE_NAMES) + len(TEMPORAL_FEATURE_NAMES)
        v = predictor.predict_value([0.0] * state_dim)
        assert isinstance(v, float)

    def test_rank_actions_returns_sorted(
        self,
        tiny_transitions_parquet: Path,
        tmp_path: Path,
    ) -> None:
        pytest.importorskip("d3rlpy")
        from stellaris_advisor.config import RLConfig
        from stellaris_advisor.features.registry import (
            EMPIRE_FEATURE_NAMES,
            TEMPORAL_FEATURE_NAMES,
        )
        from stellaris_advisor.rl.dataset import load_transitions
        from stellaris_advisor.rl.inference import RLPredictor
        from stellaris_advisor.rl.trainer import train

        config = RLConfig(algorithm="cql", n_epochs=1, batch_size=4)
        arrays = load_transitions(tiny_transitions_parquet, ["game_0"])
        model_dir = tmp_path / "model"
        train(
            config=config,
            train_arrays=arrays,
            val_arrays=arrays,
            output_dir=model_dir,
            device="cpu:0",
            n_steps=3,
        )

        predictor = RLPredictor(model_dir)
        state_dim = len(EMPIRE_FEATURE_NAMES) + len(TEMPORAL_FEATURE_NAMES)
        state = [0.0] * state_dim
        actions = [
            ("tech_researched", '{"tech_id": "tech_a"}'),
            ("building_constructed", '{"building_type": "b1"}'),
        ]
        ranked = predictor.rank_actions(state, actions)
        assert len(ranked) == 2
        # Sorted descending by Q-value
        assert ranked[0][0] >= ranked[1][0]
