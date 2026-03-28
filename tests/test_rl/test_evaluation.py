"""Tests for RL evaluation metrics."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")


@pytest.mark.unit
class TestDiscountedReturns:
    def test_single_terminal(self) -> None:
        from stellaris_advisor.rl.evaluation import compute_discounted_returns

        rewards = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        terminals = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        returns = compute_discounted_returns(rewards, terminals, gamma=0.99)
        assert returns.shape == (3,)
        # G_2 = 3.0, G_1 = 2 + 0.99*3 = 4.97, G_0 = 1 + 0.99*4.97 = 5.9203
        assert abs(returns[2] - 3.0) < 1e-4
        assert abs(returns[1] - 4.97) < 1e-4
        assert abs(returns[0] - 5.9203) < 1e-3

    def test_multiple_episodes(self) -> None:
        from stellaris_advisor.rl.evaluation import compute_discounted_returns

        rewards = np.array([1.0, 2.0, 1.0, 3.0], dtype=np.float32)
        terminals = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        returns = compute_discounted_returns(rewards, terminals, gamma=0.99)
        # Episode 1: G_1=2.0, G_0=1+0.99*2=2.98
        # Episode 2: G_3=3.0, G_2=1+0.99*3=3.97
        assert abs(returns[1] - 2.0) < 1e-4
        assert abs(returns[3] - 3.0) < 1e-4


@pytest.mark.unit
class TestEvalResult:
    def test_frozen(self) -> None:
        from stellaris_advisor.rl.evaluation import EvalResult

        result = EvalResult(
            spearman_rho=0.35,
            spearman_p_value=0.01,
            baseline_lag_rho=0.0,
            baseline_random_rho=0.05,
            n_test_samples=100,
        )
        with pytest.raises(AttributeError):
            result.spearman_rho = 0.5  # type: ignore[misc]
