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


@pytest.mark.unit
class TestEvaluateValuePredictions:
    def test_constant_predictions_returns_zero_rho(self) -> None:
        """Constant predictions produce undefined Spearman — must return 0.0, not NaN."""
        from stellaris_advisor.rl.evaluation import evaluate_value_predictions

        predicted = np.ones(10, dtype=np.float32)
        actual = np.arange(10, dtype=np.float32)
        rho, p_value = evaluate_value_predictions(predicted, actual)
        assert not np.isnan(rho), "rho must not be NaN"
        assert rho == 0.0
        assert not np.isnan(p_value), "p_value must not be NaN"
        assert p_value == 1.0

    def test_constant_returns_returns_zero_rho(self) -> None:
        """Constant actual returns also produce undefined Spearman — must return 0.0, not NaN."""
        from stellaris_advisor.rl.evaluation import evaluate_value_predictions

        predicted = np.arange(10, dtype=np.float32)
        actual = np.ones(10, dtype=np.float32)
        rho, p_value = evaluate_value_predictions(predicted, actual)
        assert not np.isnan(rho)
        assert rho == 0.0
        assert not np.isnan(p_value)
        assert p_value == 1.0

    def test_normal_predictions_returns_float(self) -> None:
        """Non-degenerate inputs must return plain floats."""
        from stellaris_advisor.rl.evaluation import evaluate_value_predictions

        rng = np.random.RandomState(0)
        predicted = rng.randn(50).astype(np.float32)
        actual = rng.randn(50).astype(np.float32)
        rho, p_value = evaluate_value_predictions(predicted, actual)
        assert isinstance(rho, float)
        assert isinstance(p_value, float)
        assert not np.isnan(rho)
        assert not np.isnan(p_value)


@pytest.mark.unit
class TestBaselineNaNSanitization:
    def test_lag_baseline_is_float_not_nan(self) -> None:
        """baseline_lag_rho must be a finite float, never NaN."""
        from unittest.mock import MagicMock

        from stellaris_advisor.rl.evaluation import evaluate

        # Constant rewards → returns are constant → lag baseline would be NaN
        obs = np.zeros((5, 3), dtype=np.float32)
        rewards = np.ones(5, dtype=np.float32)
        terminals = np.array([0, 0, 0, 0, 1], dtype=np.float32)

        model = MagicMock()
        model.predict.return_value = np.zeros((5, 2), dtype=np.float32)
        model.predict_value.return_value = np.ones((5, 1), dtype=np.float32)

        result = evaluate(model, obs, rewards, terminals)
        assert isinstance(result.baseline_lag_rho, float)
        assert not np.isnan(result.baseline_lag_rho)

    def test_random_baseline_is_float_not_nan(self) -> None:
        """baseline_random_rho must be a finite float, never NaN."""
        from unittest.mock import MagicMock

        from stellaris_advisor.rl.evaluation import evaluate

        # Constant rewards → constant returns → random baseline would be NaN
        obs = np.zeros((5, 3), dtype=np.float32)
        rewards = np.ones(5, dtype=np.float32)
        terminals = np.array([0, 0, 0, 0, 1], dtype=np.float32)

        model = MagicMock()
        model.predict.return_value = np.zeros((5, 2), dtype=np.float32)
        model.predict_value.return_value = np.ones((5, 1), dtype=np.float32)

        result = evaluate(model, obs, rewards, terminals)
        assert isinstance(result.baseline_random_rho, float)
        assert not np.isnan(result.baseline_random_rho)
