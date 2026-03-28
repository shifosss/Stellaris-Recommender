"""LightGBM LambdaMART colonization ranking model."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class ColonizationScorer:
    """Wraps a trained LightGBM ranker for colonization system scoring."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def predict(self, system_features: list[float]) -> float:
        """Score a single system. Higher = better colonization target."""
        import numpy as np

        X = np.array([system_features], dtype=np.float32)
        return float(self._model.predict(X)[0])

    def predict_batch(self, features: list[list[float]]) -> list[float]:
        """Score multiple systems."""
        import numpy as np

        X = np.array(features, dtype=np.float32)
        return [float(v) for v in self._model.predict(X)]

    def rank_systems(
        self,
        system_features_list: list[list[float]],
    ) -> list[tuple[float, int]]:
        """Rank systems by score, descending. Returns (score, original_index)."""
        scores = self.predict_batch(system_features_list)
        indexed = [(score, i) for i, score in enumerate(scores)]
        return sorted(indexed, key=lambda x: x[0], reverse=True)

    def save(self, path: Path) -> None:
        """Save model to file."""
        self._model.booster_.save_model(str(path))

    @classmethod
    def load(cls, path: Path) -> ColonizationScorer:
        """Load model from file."""
        import lightgbm as lgb

        booster = lgb.Booster(model_file=str(path))
        # Wrap in a predict-compatible object
        return cls(_BoosterWrapper(booster))

    @staticmethod
    def is_available(model_path: Path) -> bool:
        """Check if model file exists (no ML imports needed)."""
        return model_path.exists() and model_path.stat().st_size > 0


class _BoosterWrapper:
    """Thin wrapper to give Booster the same predict() interface as LGBMRanker."""

    def __init__(self, booster: Any) -> None:
        self._booster = booster

    def predict(self, X: Any) -> Any:
        return self._booster.predict(X)
