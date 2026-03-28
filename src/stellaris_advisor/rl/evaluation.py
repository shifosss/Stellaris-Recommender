"""Evaluation metrics for offline RL models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EvalResult:
    """Immutable evaluation outcome."""

    spearman_rho: float
    spearman_p_value: float
    baseline_lag_rho: float
    baseline_random_rho: float
    n_test_samples: int


def compute_discounted_returns(
    rewards: Any,
    terminals: Any,
    gamma: float = 0.99,
) -> Any:
    """Backward-walk discounted returns: G_t = r_t + gamma * G_{t+1}.

    Resets at terminal states.
    """
    n = len(rewards)
    returns = np.zeros(n, dtype=np.float32)
    g = 0.0
    for t in range(n - 1, -1, -1):
        if terminals[t] > 0.5:
            g = 0.0
        g = float(rewards[t]) + gamma * g
        returns[t] = g
    return returns


def evaluate_value_predictions(
    predicted_values: Any,
    actual_returns: Any,
) -> tuple[float, float]:
    """Compute Spearman rank correlation between predicted V(s) and actual returns."""
    from scipy.stats import spearmanr

    rho, p_value = spearmanr(predicted_values, actual_returns)
    return (
        float(rho) if not math.isnan(rho) else 0.0,
        float(p_value) if not math.isnan(p_value) else 1.0,
    )


def evaluate(
    model: Any,
    test_observations: Any,
    test_rewards: Any,
    test_terminals: Any,
    gamma: float = 0.99,
) -> EvalResult:
    """Full evaluation: Spearman ρ + baselines on test data.

    Args:
        model: Trained d3rlpy model with predict() and predict_value().
        test_observations: (N, state_dim) numpy array.
        test_rewards: (N,) numpy array.
        test_terminals: (N,) numpy array.
        gamma: Discount factor.

    Returns:
        Frozen EvalResult.
    """
    from scipy.stats import spearmanr

    actual_returns = compute_discounted_returns(test_rewards, test_terminals, gamma)

    # Model predictions: V(s) ≈ Q(s, π(s))
    actions = model.predict(test_observations)
    predicted_values = model.predict_value(test_observations, actions).flatten()

    rho, p_value = evaluate_value_predictions(predicted_values, actual_returns)

    # Baseline: lag (predict 0 for everything)
    lag_rho, _ = spearmanr(np.zeros_like(actual_returns), actual_returns)

    # Baseline: random
    rng = np.random.RandomState(42)
    random_preds = rng.randn(len(actual_returns))
    random_rho, _ = spearmanr(random_preds, actual_returns)

    return EvalResult(
        spearman_rho=rho,
        spearman_p_value=p_value,
        baseline_lag_rho=float(lag_rho) if not np.isnan(lag_rho) else 0.0,
        baseline_random_rho=float(random_rho) if not np.isnan(random_rho) else 0.0,
        n_test_samples=len(actual_returns),
    )
