"""Load trained RL model and provide prediction methods for the planner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from stellaris_advisor.rl.action_encoder import ActionEncoder
from stellaris_advisor.rl.state_encoder import StateEncoder


class RLPredictor:
    """Wraps a trained d3rlpy model for planner consumption.

    Provides predict_value(state) and rank_actions(state, candidates).
    """

    def __init__(self, model_dir: Path) -> None:
        import d3rlpy

        self._encoder = StateEncoder()
        self._encoder.load(model_dir / "encoder_stats.json")
        self._action_encoder = ActionEncoder()
        self._model = d3rlpy.load_learnable(str(model_dir / "model.d3"))

    def predict_value(self, state_features: list[float]) -> float:
        """Approximate V(s) by querying Q(s, π(s))."""
        obs = np.array([state_features], dtype=np.float32)
        action = self._model.predict(obs)
        q = self._model.predict_value(obs, action)
        return float(q[0])

    def predict_q_value(
        self,
        state_features: list[float],
        action_type: str,
        action_params_json: str,
    ) -> float:
        """Q(s, a) for a specific state-action pair."""
        obs = np.array([state_features], dtype=np.float32)
        action_vec = np.array(
            [self._action_encoder.encode(action_type, action_params_json)],
            dtype=np.float32,
        )
        q = self._model.predict_value(obs, action_vec)
        return float(q[0])

    def rank_actions(
        self,
        state_features: list[float],
        actions: list[tuple[str, str]],
    ) -> list[tuple[float, str, str]]:
        """Rank candidate actions by Q-value, descending.

        Args:
            state_features: 117-element feature vector.
            actions: List of (action_type, action_params_json) tuples.

        Returns:
            List of (q_value, action_type, action_params_json) sorted desc.
        """
        if not actions:
            return []
        obs = np.tile(
            np.array(state_features, dtype=np.float32),
            (len(actions), 1),
        )
        action_vecs = np.array(
            [self._action_encoder.encode(at, ap) for at, ap in actions],
            dtype=np.float32,
        )
        q_values = self._model.predict_value(obs, action_vecs)
        ranked = [
            (float(q_values[i]), actions[i][0], actions[i][1])
            for i in range(len(actions))
        ]
        return sorted(ranked, key=lambda x: x[0], reverse=True)

    @staticmethod
    def is_available(model_dir: Path) -> bool:
        """Check if trained model artifacts exist (no ML imports needed)."""
        return (
            model_dir.exists()
            and (model_dir / "model.d3").exists()
            and (model_dir / "encoder_stats.json").exists()
        )


# Suppress unused import warning — numpy is used at runtime via np alias
_: Any = np
