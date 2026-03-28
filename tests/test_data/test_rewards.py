"""Tests for reward computation (composite scoring + delta reward + terminal bonus)."""
from __future__ import annotations

from types import MappingProxyType

import pytest

from stellaris_advisor.data.rewards import (
    DEFAULT_WEIGHTS,
    TERMINAL_BONUS,
    RewardWeights,
    composite_score,
    compute_reward,
)
from tests.test_features.conftest import make_empire


# ---------------------------------------------------------------------------
# Helper: empire with controlled raw stats
# ---------------------------------------------------------------------------


def _empire(
    fleet: float,
    tech_count: int,
    income: float,
    pop: int,
    rank: int = 1,
) -> object:
    return make_empire(
        fleet_power=fleet,
        researched_techs=frozenset(f"tech_{i}" for i in range(tech_count)),
        resource_income=MappingProxyType({"energy": income}),
        total_pops=pop,
        victory_rank=rank,
    )


# ---------------------------------------------------------------------------
# Tests — RewardWeights
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRewardWeights:
    def test_default_values(self) -> None:
        w = RewardWeights()
        assert w.fleet == pytest.approx(0.25)
        assert w.tech == pytest.approx(0.25)
        assert w.income == pytest.approx(0.30)
        assert w.pop == pytest.approx(0.20)

    def test_is_frozen(self) -> None:
        w = RewardWeights()
        with pytest.raises(Exception):
            w.fleet = 0.5  # type: ignore[misc]

    def test_default_weights_sum_to_one(self) -> None:
        w = DEFAULT_WEIGHTS
        assert w.fleet + w.tech + w.income + w.pop == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Tests — composite_score
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCompositeScore:
    def test_minimum_empire_scores_zero(self) -> None:
        lo = _empire(fleet=100, tech_count=5, income=50, pop=10)
        hi = _empire(fleet=200, tech_count=10, income=100, pop=20)
        assert composite_score(lo, (lo, hi)) == pytest.approx(0.0)

    def test_maximum_empire_scores_one(self) -> None:
        lo = _empire(fleet=100, tech_count=5, income=50, pop=10)
        hi = _empire(fleet=200, tech_count=10, income=100, pop=20)
        assert composite_score(hi, (lo, hi)) == pytest.approx(1.0)

    def test_single_empire_scores_half(self) -> None:
        emp = _empire(fleet=1000, tech_count=20, income=500, pop=100)
        assert composite_score(emp, (emp,)) == pytest.approx(0.5)

    def test_identical_empires_score_half(self) -> None:
        emp = _empire(fleet=500, tech_count=10, income=100, pop=50)
        same = _empire(fleet=500, tech_count=10, income=100, pop=50)
        assert composite_score(emp, (emp, same)) == pytest.approx(0.5)

    def test_middle_empire_scores_half(self) -> None:
        lo = _empire(fleet=0, tech_count=0, income=0, pop=0)
        mid = _empire(fleet=50, tech_count=5, income=50, pop=50)
        hi = _empire(fleet=100, tech_count=10, income=100, pop=100)
        assert composite_score(mid, (lo, mid, hi)) == pytest.approx(0.5)

    def test_custom_weights_isolate_fleet(self) -> None:
        lo = _empire(fleet=0, tech_count=0, income=0, pop=0)
        hi = _empire(fleet=100, tech_count=10, income=100, pop=100)
        weights = RewardWeights(fleet=1.0, tech=0.0, income=0.0, pop=0.0)
        assert composite_score(lo, (lo, hi), weights) == pytest.approx(0.0)
        assert composite_score(hi, (lo, hi), weights) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Tests — compute_reward
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeReward:
    def test_positive_delta(self) -> None:
        assert compute_reward(0.8, 0.5) == pytest.approx(0.3)

    def test_negative_delta(self) -> None:
        assert compute_reward(0.3, 0.7) == pytest.approx(-0.4)

    def test_zero_delta(self) -> None:
        assert compute_reward(0.5, 0.5) == pytest.approx(0.0)

    def test_non_terminal_ignores_rank_args(self) -> None:
        base = compute_reward(0.6, 0.4)
        with_rank = compute_reward(0.6, 0.4, is_terminal=False, final_rank=1, num_empires=4)
        assert base == pytest.approx(with_rank)

    def test_terminal_bonus_rank_1_of_4(self) -> None:
        # bonus = TERMINAL_BONUS * (4 - 1) / (4 - 1) = 1.0
        base = 0.1
        result = compute_reward(base + 0.0, 0.0, is_terminal=True, final_rank=1, num_empires=4)
        expected = base + TERMINAL_BONUS * 1.0
        assert result == pytest.approx(expected)

    def test_terminal_bonus_last_rank(self) -> None:
        # bonus = TERMINAL_BONUS * (4 - 4) / (4 - 1) = 0.0
        result = compute_reward(0.0, 0.0, is_terminal=True, final_rank=4, num_empires=4)
        assert result == pytest.approx(0.0)

    def test_terminal_bonus_middle_rank(self) -> None:
        # rank 2 of 4: bonus = 1.0 * (4-2)/3 = 2/3
        result = compute_reward(0.0, 0.0, is_terminal=True, final_rank=2, num_empires=4)
        assert result == pytest.approx(2 / 3)

    def test_terminal_bonus_single_empire(self) -> None:
        # num_empires=1 → max(0,1)=1 → bonus = 1.0*(1-1)/1 = 0.0
        result = compute_reward(0.0, 0.0, is_terminal=True, final_rank=1, num_empires=1)
        assert result == pytest.approx(0.0)

    def test_terminal_constant_value(self) -> None:
        assert TERMINAL_BONUS == pytest.approx(1.0)
