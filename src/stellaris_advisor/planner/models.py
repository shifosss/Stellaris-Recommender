# src/stellaris_advisor/planner/models.py
"""Planner data models — all frozen dataclasses."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass(frozen=True)
class Alert:
    """Single planner alert."""

    severity: AlertSeverity
    category: str  # e.g., "resource", "military", "ascension", "faction"
    message: str  # short summary
    detail: str  # explanation


@dataclass(frozen=True)
class Recommendation:
    """Single ranked action recommendation."""

    rank: int
    category: str  # economy, tech, expansion, military, tradition, perk, megastructure, council, faction
    action_type: str  # specific action (build_building, research_tech, etc.)
    description: str  # human-readable summary
    detail: str  # additional context
    rule_roi: float  # rule-based ROI score (normalized 0-1)
    rl_delta: float  # RL value delta (0.0 in degraded mode)
    final_score: float  # alpha * rl_delta + beta * rule_roi
    action_params: tuple[tuple[str, str], ...]  # key-value params for the action


@dataclass(frozen=True)
class BottleneckResult:
    """Detected bottleneck dimension."""

    dimension: str  # one of 8 benchmark metrics
    gap: float  # normalized gap (negative = below median)
    percentile_rank: int  # 0-100
    resolution_hint: str  # actionable guidance


@dataclass(frozen=True)
class EmpireHealth:
    """Empire health dashboard comparing state against benchmarks."""

    overall_percentile: int  # average across dimensions
    dimension_percentiles: tuple[tuple[str, int], ...]  # (metric, percentile) pairs
    bottleneck: BottleneckResult | None  # worst dimension, if below median


@dataclass(frozen=True)
class TechPathStep:
    """Single step in a recommended tech path."""

    tech_id: str
    tech_name: str
    area: str  # physics, society, engineering
    tier: int
    cost: int
    estimated_months: float
    unlocks: tuple[str, ...]  # building/district/component names unlocked


@dataclass(frozen=True)
class TechPathInsight:
    """Tech path recommendation from bottleneck to resolution."""

    goal: str  # keyword or tech ID
    steps: tuple[TechPathStep, ...]
    total_months: float
    bottleneck_addressed: str  # which bottleneck this resolves


@dataclass(frozen=True)
class TraditionPerkPath:
    """Tradition and ascension perk guidance."""

    traditions_completed: int
    traditions_total: int
    perks_taken: int
    perks_total: int
    unity_income: float
    ascension_path: str | None  # current path or None
    recommended_tradition: str | None  # tree or tradition ID
    tradition_rationale: str
    recommended_perk: str | None  # perk ID
    perk_rationale: str


@dataclass(frozen=True)
class FactionSummary:
    """Faction and influence analysis."""

    influence_income: float
    top_faction: str  # name of highest-support faction
    top_faction_approval: float
    problem_faction: str | None  # lowest approval faction
    problem_faction_approval: float | None
    recommendation: str  # actionable guidance


@dataclass(frozen=True)
class Report:
    """Top-level planner output. Pure data — rendering handled by CLI."""

    empire_name: str
    year: int
    mode: str  # "full" or "degraded"
    empire_health: EmpireHealth
    recommendations: tuple[Recommendation, ...]  # top N, ranked by final_score
    tech_path_insight: TechPathInsight | None
    tradition_perk_path: TraditionPerkPath | None
    faction_influence_summary: FactionSummary | None
    alerts: tuple[Alert, ...]
