"""Reward computation for the RL training pipeline.

composite_score = w1*norm(fleet_power) + w2*norm(tech_count)
               + w3*norm(total_income) + w4*norm(pop_count)

Normalisation is min-max across all empires at the same time step.
When min == max for a component, 0.5 (midpoint) is used.
reward = Δ(composite_score) + optional rank-based terminal bonus.
"""
from __future__ import annotations

from dataclasses import dataclass

from stellaris_advisor.parser.models import EmpireState

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TERMINAL_BONUS: float = 1.0


# ---------------------------------------------------------------------------
# Weight configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RewardWeights:
    """Component weights for the composite score. Should sum to 1.0."""

    fleet: float = 0.25
    tech: float = 0.25
    income: float = 0.30
    pop: float = 0.20


DEFAULT_WEIGHTS: RewardWeights = RewardWeights()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _minmax_norm(value: float, values: tuple[float, ...]) -> float:
    """Min-max normalise value. Returns 0.5 (midpoint) when all equal."""
    min_v = min(values)
    max_v = max(values)
    if max_v == min_v:
        return 0.5
    return (value - min_v) / (max_v - min_v)


def _raw_stats(empire: EmpireState) -> tuple[float, float, float, float]:
    """Return (fleet_power, tech_count, total_income, pop_count) as floats."""
    return (
        empire.fleet_power,
        float(len(empire.researched_techs)),
        sum(empire.resource_income.values()),
        float(empire.total_pops),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def composite_score(
    empire: EmpireState,
    all_empires: tuple[EmpireState, ...],
    weights: RewardWeights = DEFAULT_WEIGHTS,
) -> float:
    """Compute empire's normalised composite score relative to all_empires."""
    raw = [_raw_stats(e) for e in all_empires]
    fleet_vals = tuple(r[0] for r in raw)
    tech_vals = tuple(r[1] for r in raw)
    income_vals = tuple(r[2] for r in raw)
    pop_vals = tuple(r[3] for r in raw)

    my = _raw_stats(empire)
    return (
        weights.fleet * _minmax_norm(my[0], fleet_vals)
        + weights.tech * _minmax_norm(my[1], tech_vals)
        + weights.income * _minmax_norm(my[2], income_vals)
        + weights.pop * _minmax_norm(my[3], pop_vals)
    )


def compute_reward(
    current_score: float,
    previous_score: float,
    is_terminal: bool = False,
    final_rank: int = 0,
    num_empires: int = 1,
) -> float:
    """Return base reward plus optional terminal bonus.

    Base reward: current_score - previous_score.
    Terminal bonus: TERMINAL_BONUS * (num_empires - final_rank) / max(num_empires - 1, 1).
    """
    base = current_score - previous_score
    if not is_terminal:
        return base
    bonus = TERMINAL_BONUS * (num_empires - final_rank) / max(num_empires - 1, 1)
    return base + bonus
