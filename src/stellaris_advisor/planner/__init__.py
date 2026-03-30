# src/stellaris_advisor/planner/__init__.py
"""Strategic planner package.

Works in degraded mode (rule-based ROI only) without ML dependencies.
Full mode (RL value + rule ROI) requires trained models from rl/ package.
"""
from __future__ import annotations

from stellaris_advisor.planner.models import (
    Alert,
    AlertSeverity,
    BottleneckResult,
    EmpireHealth,
    FactionSummary,
    Recommendation,
    Report,
    TechPathInsight,
    TechPathStep,
    TraditionPerkPath,
)
from stellaris_advisor.planner.report import analyze

__all__ = [
    "analyze",
    "Alert",
    "AlertSeverity",
    "BottleneckResult",
    "EmpireHealth",
    "FactionSummary",
    "Recommendation",
    "Report",
    "TechPathInsight",
    "TechPathStep",
    "TraditionPerkPath",
]
