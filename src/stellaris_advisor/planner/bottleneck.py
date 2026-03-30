# src/stellaris_advisor/planner/bottleneck.py
"""Bottleneck detection: 8 dimensions compared against benchmark percentiles."""
from __future__ import annotations

from stellaris_advisor.colonization.benchmark import BENCHMARK_METRICS, BenchmarkStore
from stellaris_advisor.planner.models import BottleneckResult, EmpireHealth

# Map benchmark metric names to empire feature keys
_METRIC_TO_FEATURE: dict[str, str] = {
    "alloy_income": "resource_net_alloys",
    "research_total": "tech_physics_research_speed",
    "fleet_power": "military_fleet_power",
    "tech_count": "tech_total_count",
    "system_count": "expansion_system_count",
    "influence_income": "resource_net_influence",
    "unity_income": "tradition_unity_income",
    "traditions_completed": "tradition_trees_completed",
}

_RESOLUTION_HINTS: dict[str, str] = {
    "alloy_income": "Build alloy foundries or research alloy production tech",
    "research_total": "Check Head of Research expertise; consider research buildings or council agenda",
    "fleet_power": "Build ships to fill naval capacity; research higher ship classes",
    "tech_count": "Increase research speed; align scientist expertise with tech goals",
    "system_count": "Colonize habitable systems; build starbases for expansion",
    "influence_income": "Improve faction approval; declare rivals; check policy alignment",
    "unity_income": "Build unity buildings; open new tradition trees for adoption bonuses",
    "traditions_completed": "Focus unity on completing tradition trees to unlock perk slots",
}


def detect_bottleneck(
    empire_features: dict[str, float],
    year: int,
    difficulty: str,
    empire_type: str,
    benchmark: BenchmarkStore,
) -> BottleneckResult:
    """Find the empire's worst-performing dimension relative to benchmarks.

    Returns the dimension with the lowest percentile rank.
    """
    worst_metric = ""
    worst_rank = 101
    worst_gap = 0.0

    for metric in BENCHMARK_METRICS:
        feature_key = _METRIC_TO_FEATURE[metric]
        value = empire_features.get(feature_key, 0.0)
        rank = benchmark.percentile_rank(year, difficulty, empire_type, metric, value)
        if rank < worst_rank:
            worst_rank = rank
            worst_metric = metric
            entry = benchmark.lookup(year, difficulty, empire_type, metric)
            median = entry.p50 if entry else 0.0
            worst_gap = (value - median) / max(abs(median), 1.0)

    if not worst_metric:
        worst_metric = "alloy_income"
        worst_rank = 50

    return BottleneckResult(
        dimension=worst_metric,
        gap=worst_gap,
        percentile_rank=worst_rank,
        resolution_hint=_RESOLUTION_HINTS.get(worst_metric, ""),
    )


def compute_empire_health(
    empire_features: dict[str, float],
    year: int,
    difficulty: str,
    empire_type: str,
    benchmark: BenchmarkStore,
) -> EmpireHealth:
    """Compute percentile ranks across all 8 dimensions."""
    dim_ranks: list[tuple[str, int]] = []

    for metric in BENCHMARK_METRICS:
        feature_key = _METRIC_TO_FEATURE[metric]
        value = empire_features.get(feature_key, 0.0)
        rank = benchmark.percentile_rank(year, difficulty, empire_type, metric, value)
        dim_ranks.append((metric, rank))

    overall = sum(r for _, r in dim_ranks) // max(len(dim_ranks), 1)
    bottleneck = detect_bottleneck(
        empire_features, year, difficulty, empire_type, benchmark
    )
    bn = bottleneck if bottleneck.percentile_rank < 50 else None

    return EmpireHealth(
        overall_percentile=overall,
        dimension_percentiles=tuple(dim_ranks),
        bottleneck=bn,
    )
