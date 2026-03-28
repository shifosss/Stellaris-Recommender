"""State encoder: feature vector → normalized tensor with running mean/std."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

_EPSILON = 1e-8


@dataclass(frozen=True)
class NormStats:
    """Immutable snapshot of normalisation statistics."""

    mean: tuple[float, ...]
    std: tuple[float, ...]
    count: int


class StateEncoder:
    """Encodes raw feature vectors into normalised tensors.

    Uses Welford's online algorithm for numerically stable running mean/std.
    Serialises stats as JSON (portable, no pickle/torch dependency).
    """

    def __init__(self, state_dim: int | None = None) -> None:
        if state_dim is None:
            from stellaris_advisor.features.registry import (
                EMPIRE_FEATURE_NAMES,
                TEMPORAL_FEATURE_NAMES,
            )

            state_dim = len(EMPIRE_FEATURE_NAMES) + len(TEMPORAL_FEATURE_NAMES)
        self._state_dim = state_dim
        self._mean: list[float] = [0.0] * state_dim
        self._var: list[float] = [0.0] * state_dim
        self._count: int = 0

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def stats(self) -> NormStats:
        return NormStats(
            mean=tuple(self._mean),
            std=tuple(math.sqrt(v) for v in self._var),
            count=self._count,
        )

    def fit(self, features: list[list[float]]) -> None:
        """Compute mean/std from a batch (overwrites prior stats)."""
        if not features:
            return
        n = len(features)
        dim = self._state_dim
        mean = [0.0] * dim
        for row in features:
            for i in range(dim):
                mean[i] += row[i]
        for i in range(dim):
            mean[i] /= n
        var = [0.0] * dim
        for row in features:
            for i in range(dim):
                diff = row[i] - mean[i]
                var[i] += diff * diff
        for i in range(dim):
            var[i] /= max(n - 1, 1)
        self._mean = mean
        self._var = var
        self._count = n

    def update(self, features: list[list[float]]) -> None:
        """Welford online update with a new batch."""
        for row in features:
            self._count += 1
            for i in range(self._state_dim):
                delta = row[i] - self._mean[i]
                self._mean[i] += delta / self._count
                delta2 = row[i] - self._mean[i]
                self._var[i] += (delta * delta2 - self._var[i]) / self._count

    def encode(self, features: list[float]) -> torch.Tensor:
        """Single vector → normalised float32 tensor."""
        import torch

        if self._count == 0:
            msg = "StateEncoder has not been fit — call fit() or load() first"
            raise RuntimeError(msg)
        normed = [
            (features[i] - self._mean[i]) / max(math.sqrt(self._var[i]), _EPSILON)
            for i in range(self._state_dim)
        ]
        return torch.tensor(normed, dtype=torch.float32)

    def encode_batch(self, features: list[list[float]]) -> torch.Tensor:
        """Batch of vectors → (N, state_dim) float32 tensor."""
        import torch

        if self._count == 0:
            msg = "StateEncoder has not been fit — call fit() or load() first"
            raise RuntimeError(msg)
        rows: list[list[float]] = []
        for row in features:
            normed = [
                (row[i] - self._mean[i]) / max(math.sqrt(self._var[i]), _EPSILON)
                for i in range(self._state_dim)
            ]
            rows.append(normed)
        return torch.tensor(rows, dtype=torch.float32)

    def save(self, path: Path) -> None:
        """Persist stats as JSON."""
        data = {
            "state_dim": self._state_dim,
            "mean": self._mean,
            "std": [math.sqrt(v) for v in self._var],
            "var": self._var,
            "count": self._count,
        }
        path.write_text(json.dumps(data, indent=2))

    def load(self, path: Path) -> None:
        """Restore stats from JSON."""
        data = json.loads(path.read_text())
        self._state_dim = data["state_dim"]
        self._mean = data["mean"]
        self._var = data["var"]
        self._count = data["count"]
