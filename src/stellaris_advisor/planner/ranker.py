# src/stellaris_advisor/planner/ranker.py
"""Hybrid ranker: degraded (rule ROI) or full (α·RL_delta + β·rule_ROI)."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from stellaris_advisor.config import PlannerConfig
from stellaris_advisor.planner.models import Recommendation


def _normalize_roi(recs: tuple[Recommendation, ...]) -> tuple[Recommendation, ...]:
    """Normalize rule_roi to [0, 1] range across all candidates."""
    if not recs:
        return recs
    max_roi = max(r.rule_roi for r in recs)
    min_roi = min(r.rule_roi for r in recs)
    span = max_roi - min_roi
    if span < 1e-8:
        return recs
    return tuple(replace(r, rule_roi=(r.rule_roi - min_roi) / span) for r in recs)


def rank_recommendations(
    candidates: tuple[Recommendation, ...],
    config: PlannerConfig,
    rl_predictor: Any = None,
    empire_features: dict[str, float] | None = None,
) -> tuple[Recommendation, ...]:
    """Score and rank all candidate recommendations.

    Degraded mode (no rl_predictor): final_score = rule_roi
    Full mode: final_score = alpha * rl_delta + beta * normalized_rule_roi
    """
    if not candidates:
        return ()

    normalized = _normalize_roi(candidates)

    scored: list[Recommendation] = []
    for rec in normalized:
        if rl_predictor is not None and empire_features is not None:
            # Full mode — RL delta scoring would go here
            # For now, use degraded scoring (RL integration deferred to actual model availability)
            final = config.alpha * rec.rl_delta + config.beta * rec.rule_roi
        else:
            # Degraded mode
            final = rec.rule_roi

        scored.append(replace(rec, final_score=final))

    # Sort descending by final_score
    scored.sort(key=lambda r: r.final_score, reverse=True)

    # Assign ranks (1-indexed) and limit to top_n
    top_n = scored[: config.top_n_recommendations]
    ranked = tuple(replace(r, rank=i + 1) for i, r in enumerate(top_n))

    return ranked
