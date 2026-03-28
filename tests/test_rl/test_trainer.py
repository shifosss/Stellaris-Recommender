"""Tests for d3rlpy training orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.slow
class TestTrainer:
    def test_create_cql_algorithm(self) -> None:
        pytest.importorskip("d3rlpy")
        from stellaris_advisor.config import RLConfig
        from stellaris_advisor.rl.trainer import create_algorithm

        config = RLConfig(algorithm="cql", learning_rate=1e-3)
        algo = create_algorithm(config, state_dim=10, action_dim=5, device="cpu:0")
        assert algo is not None

    def test_create_iql_algorithm(self) -> None:
        pytest.importorskip("d3rlpy")
        from stellaris_advisor.config import RLConfig
        from stellaris_advisor.rl.trainer import create_algorithm

        config = RLConfig(algorithm="iql", learning_rate=1e-3)
        algo = create_algorithm(config, state_dim=10, action_dim=5, device="cpu:0")
        assert algo is not None

    def test_train_produces_files(
        self, tiny_transitions_parquet: Path, tmp_path: Path
    ) -> None:
        pytest.importorskip("d3rlpy")
        from stellaris_advisor.config import RLConfig
        from stellaris_advisor.rl.dataset import load_transitions
        from stellaris_advisor.rl.trainer import train

        config = RLConfig(algorithm="cql", n_epochs=1, batch_size=4, learning_rate=1e-3)
        arrays = load_transitions(tiny_transitions_parquet, ["game_0", "game_1"])
        result = train(
            config=config,
            train_arrays=arrays,
            val_arrays=arrays,
            output_dir=tmp_path / "model",
            device="cpu:0",
            n_steps=5,
        )
        assert result.model_path.exists()
        assert result.encoder_path.exists()
        assert result.n_steps == 5

    def test_train_result_frozen(
        self, tiny_transitions_parquet: Path, tmp_path: Path
    ) -> None:
        pytest.importorskip("d3rlpy")
        from stellaris_advisor.config import RLConfig
        from stellaris_advisor.rl.dataset import load_transitions
        from stellaris_advisor.rl.trainer import train

        config = RLConfig(algorithm="cql", n_epochs=1, batch_size=4)
        arrays = load_transitions(tiny_transitions_parquet, ["game_0"])
        result = train(
            config=config,
            train_arrays=arrays,
            val_arrays=arrays,
            output_dir=tmp_path / "m",
            device="cpu:0",
            n_steps=3,
        )
        with pytest.raises(AttributeError):
            result.n_steps = 999  # type: ignore[misc]

    def test_invalid_algorithm_raises(self) -> None:
        pytest.importorskip("d3rlpy")
        from stellaris_advisor.config import RLConfig
        from stellaris_advisor.rl.trainer import create_algorithm

        config = RLConfig(algorithm="invalid")
        with pytest.raises(ValueError, match="algorithm"):
            create_algorithm(config, state_dim=10, action_dim=5, device="cpu:0")
