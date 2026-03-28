"""Tests for Optuna hyperparameter tuning."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.slow
class TestTuning:
    def test_tune_returns_result(
        self, tiny_transitions_parquet: Path, tmp_path: Path
    ) -> None:
        pytest.importorskip("d3rlpy")
        pytest.importorskip("optuna")
        from stellaris_advisor.rl.dataset import load_transitions
        from stellaris_advisor.rl.tuning import TuningResult, tune

        arrays = load_transitions(tiny_transitions_parquet, ["game_0", "game_1"])
        result = tune(
            train_arrays=arrays,
            val_arrays=arrays,
            output_dir=tmp_path,
            n_trials=2,
            n_steps_per_trial=3,
            device="cpu:0",
        )
        assert isinstance(result, TuningResult)
        assert result.n_trials == 2
        assert "learning_rate" in result.best_params

    def test_tuning_result_frozen(
        self, tiny_transitions_parquet: Path, tmp_path: Path
    ) -> None:
        pytest.importorskip("d3rlpy")
        pytest.importorskip("optuna")
        from stellaris_advisor.rl.dataset import load_transitions
        from stellaris_advisor.rl.tuning import tune

        arrays = load_transitions(tiny_transitions_parquet, ["game_0"])
        result = tune(
            train_arrays=arrays,
            val_arrays=arrays,
            output_dir=tmp_path,
            n_trials=1,
            n_steps_per_trial=2,
            device="cpu:0",
        )
        with pytest.raises(AttributeError):
            result.n_trials = 99  # type: ignore[misc]
