"""Optuna hyperparameter search for offline RL."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from stellaris_advisor.rl.dataset import RLArrays, build_mdp_dataset
from stellaris_advisor.rl.evaluation import (
    compute_discounted_returns,
    evaluate_value_predictions,
)


@dataclass(frozen=True)
class TuningResult:
    """Immutable tuning outcome."""

    best_params: MappingProxyType[str, Any]
    best_value: float
    n_trials: int
    study_name: str


def tune(
    train_arrays: RLArrays,
    val_arrays: RLArrays,
    output_dir: Path,
    n_trials: int = 20,
    n_steps_per_trial: int | None = None,
    device: str = "cpu:0",
    gamma: float = 0.99,
) -> TuningResult:
    """Run Optuna hyperparameter search.

    Search space:
    - learning_rate: [1e-5, 1e-2] log-uniform
    - batch_size: {64, 128, 256, 512}
    - cql_alpha: [0.1, 10.0] uniform

    Objective: Spearman ρ on validation set.
    """
    import d3rlpy
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    train_dataset = build_mdp_dataset(train_arrays)
    val_returns = compute_discounted_returns(
        val_arrays.rewards, val_arrays.terminals, gamma
    )

    if n_steps_per_trial is None:
        n_steps_per_trial = max(10, len(train_arrays.observations) // 4)

    def objective(trial: Any) -> float:
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
        cql_alpha = trial.suggest_float("cql_alpha", 0.1, 10.0)

        encoder_factory = d3rlpy.models.VectorEncoderFactory(
            hidden_units=[256, 128, 64],
            use_batch_norm=True,
        )

        algo = d3rlpy.algos.CQLConfig(
            batch_size=batch_size,
            gamma=gamma,
            actor_learning_rate=lr,
            critic_learning_rate=lr,
            conservative_weight=cql_alpha,
            actor_encoder_factory=encoder_factory,
            critic_encoder_factory=encoder_factory,
        ).create(device=device)

        algo.fit(train_dataset, n_steps=n_steps_per_trial, show_progress=False)

        actions = algo.predict(val_arrays.observations.copy())
        pred_values = algo.predict_value(
            val_arrays.observations.copy(), actions
        ).flatten()
        rho, _ = evaluate_value_predictions(pred_values, val_returns)

        return rho

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name="stellaris_rl_tune",
    )
    study.optimize(objective, n_trials=n_trials)

    return TuningResult(
        best_params=MappingProxyType(study.best_trial.params),
        best_value=study.best_trial.value,
        n_trials=n_trials,
        study_name=study.study_name,
    )
