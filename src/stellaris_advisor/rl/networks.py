"""Standalone PyTorch V(s) and Q(s,a) network architectures.

Used for direct supervised V(s) training and future MCTS.
d3rlpy CQL/IQL training uses VectorEncoderFactory instead.

Requires torch — all classes defined inside functions to support lazy import.
"""

from __future__ import annotations

from typing import Any

_HIDDEN_UNITS: tuple[int, ...] = (256, 128, 64)
_DROPOUT: float = 0.1


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        msg = "torch is required for RL networks — install with: pip install stellaris-advisor[ml]"
        raise ImportError(msg) from exc
    return torch


def _build_mlp(torch: Any, input_dim: int) -> Any:
    """Build a 3-layer MLP with BatchNorm, ReLU, and Dropout."""
    nn = torch.nn
    layers: list[Any] = []
    prev = input_dim
    for units in _HIDDEN_UNITS:
        layers.extend(
            [
                nn.Linear(prev, units),
                nn.BatchNorm1d(units),
                nn.ReLU(),
                nn.Dropout(_DROPOUT),
            ]
        )
        prev = units
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


class ValueNetwork:
    """V(s): state_dim → 256 → 128 → 64 → 1 with BatchNorm + ReLU + Dropout(0.1)."""

    def __new__(cls, state_dim: int = 117, **kwargs: Any) -> Any:
        torch = _require_torch()

        class _ValueNetwork(torch.nn.Module):  # type: ignore[name-defined,misc]
            def __init__(self, state_dim: int) -> None:
                super().__init__()
                self.net = _build_mlp(torch, state_dim)

            def forward(self, x: Any) -> Any:
                return self.net(x)

        return _ValueNetwork(state_dim)


class QNetwork:
    """Q(s,a): concat(state, action) → 256 → 128 → 64 → 1."""

    def __new__(cls, state_dim: int = 117, action_dim: int = 20, **kwargs: Any) -> Any:
        torch = _require_torch()

        class _QNetwork(torch.nn.Module):  # type: ignore[name-defined,misc]
            def __init__(self, state_dim: int, action_dim: int) -> None:
                super().__init__()
                self.net = _build_mlp(torch, state_dim + action_dim)

            def forward(self, state: Any, action: Any) -> Any:
                x = torch.cat([state, action], dim=-1)
                return self.net(x)

        return _QNetwork(state_dim, action_dim)
