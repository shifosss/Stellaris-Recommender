"""Tests for Optuna hyperparameter tuning."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _make_fake_d3rlpy() -> ModuleType:
    """Build a minimal fake d3rlpy module for unit tests that run without the real package."""
    d3rlpy = ModuleType("d3rlpy")
    models = ModuleType("d3rlpy.models")
    algos = ModuleType("d3rlpy.algos")

    # VectorEncoderFactory stub
    models.VectorEncoderFactory = MagicMock(return_value=MagicMock())  # type: ignore[attr-defined]

    # CQLConfig stub — .create() returns a mock algo by default;
    # individual tests may override via patch
    mock_algo = MagicMock()
    mock_algo.predict.return_value = np.zeros((4, 2), dtype=np.float32)
    mock_algo.predict_value.return_value = np.zeros(4, dtype=np.float32)
    algos.CQLConfig = MagicMock(  # type: ignore[attr-defined]
        return_value=MagicMock(create=MagicMock(return_value=mock_algo))
    )

    d3rlpy.models = models  # type: ignore[attr-defined]
    d3rlpy.algos = algos  # type: ignore[attr-defined]
    return d3rlpy


@pytest.mark.unit
class TestObjectiveNaNGuard:
    """Unit tests for the NaN guard inside the tuning objective."""

    def test_objective_returns_minus_one_when_rho_is_nan(self) -> None:
        """Objective must return -1.0 (not NaN) when evaluate_value_predictions yields NaN.

        This covers the scenario where constant predicted values (or any degenerate
        distribution) cause spearmanr to return NaN, which would previously cause
        study.best_trial to raise ValueError.
        """
        from stellaris_advisor.rl.dataset import RLArrays

        obs = np.zeros((4, 3), dtype=np.float32)
        acts = np.zeros((4, 2), dtype=np.float32)
        rewards = np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32)
        terminals = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        val_arrays = RLArrays(
            observations=obs,
            actions=acts,
            rewards=rewards,
            terminals=terminals,
        )
        val_returns = np.zeros(4, dtype=np.float32)

        mock_trial = MagicMock()
        mock_trial.suggest_float.return_value = 1e-3
        mock_trial.suggest_categorical.return_value = 64

        fake_d3rlpy = _make_fake_d3rlpy()

        with patch.dict(sys.modules, {"d3rlpy": fake_d3rlpy, "d3rlpy.models": fake_d3rlpy.models, "d3rlpy.algos": fake_d3rlpy.algos}):  # type: ignore[attr-defined]
            with patch(
                "stellaris_advisor.rl.tuning.evaluate_value_predictions",
                return_value=(float("nan"), 1.0),
            ):
                from stellaris_advisor.rl.tuning import _make_objective

                objective = _make_objective(
                    train_dataset=object(),
                    val_arrays=val_arrays,
                    val_returns=val_returns,
                    gamma=0.99,
                    n_steps=1,
                    device="cpu:0",
                )
                result = objective(mock_trial)

        assert not math.isnan(result), "objective must not propagate NaN to Optuna"
        assert result == pytest.approx(-1.0)

    def test_objective_returns_rho_when_finite(self) -> None:
        """Objective must pass through a finite rho unchanged."""
        from stellaris_advisor.rl.dataset import RLArrays

        obs = np.zeros((4, 3), dtype=np.float32)
        acts = np.zeros((4, 2), dtype=np.float32)
        rewards = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        terminals = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        val_arrays = RLArrays(
            observations=obs,
            actions=acts,
            rewards=rewards,
            terminals=terminals,
        )
        val_returns = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

        mock_trial = MagicMock()
        mock_trial.suggest_float.return_value = 1e-3
        mock_trial.suggest_categorical.return_value = 64

        expected_rho = 0.75
        fake_d3rlpy = _make_fake_d3rlpy()

        with patch.dict(sys.modules, {"d3rlpy": fake_d3rlpy, "d3rlpy.models": fake_d3rlpy.models, "d3rlpy.algos": fake_d3rlpy.algos}):  # type: ignore[attr-defined]
            with patch(
                "stellaris_advisor.rl.tuning.evaluate_value_predictions",
                return_value=(expected_rho, 0.05),
            ):
                from stellaris_advisor.rl.tuning import _make_objective

                objective = _make_objective(
                    train_dataset=object(),
                    val_arrays=val_arrays,
                    val_returns=val_returns,
                    gamma=0.99,
                    n_steps=1,
                    device="cpu:0",
                )
                result = objective(mock_trial)

        assert result == pytest.approx(expected_rho)


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
