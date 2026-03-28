"""Colonization scoring and benchmark package.

BenchmarkStore.hardcoded_defaults() works without ML imports.
ColonizationScorer requires lightgbm.
"""

from __future__ import annotations

from typing import Any

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "ColonizationScorer": (
        "stellaris_advisor.colonization.scorer",
        "ColonizationScorer",
    ),
    "ColonizationTrainResult": (
        "stellaris_advisor.colonization.training",
        "ColonizationTrainResult",
    ),
    "train_colonization": (
        "stellaris_advisor.colonization.training",
        "train_colonization",
    ),
    "BenchmarkStore": ("stellaris_advisor.colonization.benchmark", "BenchmarkStore"),
    "PercentileEntry": ("stellaris_advisor.colonization.benchmark", "PercentileEntry"),
    "BENCHMARK_METRICS": (
        "stellaris_advisor.colonization.benchmark",
        "BENCHMARK_METRICS",
    ),
}

__all__ = list(_LAZY_IMPORTS.keys())


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_path)
        obj = getattr(module, attr_name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
