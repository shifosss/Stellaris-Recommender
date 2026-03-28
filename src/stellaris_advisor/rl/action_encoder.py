"""Action encoder: MacroAction → fixed-size 20-float vector."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from stellaris_advisor.data.actions import (
    ACTION_BUILDING_CONSTRUCTED,
    ACTION_COUNCIL_REASSIGNED,
    ACTION_DISTRICT_BUILT,
    ACTION_MEGASTRUCTURE_BUILT,
    ACTION_PERK_TAKEN,
    ACTION_POLICY_CHANGED,
    ACTION_SHIPS_BUILT,
    ACTION_SYSTEM_CLAIMED,
    ACTION_TECH_RESEARCHED,
    ACTION_TRADITION_UNLOCKED,
    MacroAction,
)

if TYPE_CHECKING:
    import torch

ACTION_TYPES: tuple[str, ...] = (
    ACTION_TECH_RESEARCHED,
    ACTION_BUILDING_CONSTRUCTED,
    ACTION_DISTRICT_BUILT,
    ACTION_SYSTEM_CLAIMED,
    ACTION_SHIPS_BUILT,
    ACTION_TRADITION_UNLOCKED,
    ACTION_PERK_TAKEN,
    ACTION_MEGASTRUCTURE_BUILT,
    ACTION_POLICY_CHANGED,
    ACTION_COUNCIL_REASSIGNED,
)

ACTION_DIM: int = 20
_NUM_TYPES: int = 10
_NUM_PARAMS: int = 10


def _hash_param(value: str) -> float:
    """Deterministic hash of a string param to [0, 1]."""
    return (hash(value) % 10_000) / 10_000.0


def _parse_params(action_params_json: str) -> dict[str, str]:
    """Parse JSON action params, returning empty dict on failure."""
    if not action_params_json or action_params_json == "{}":
        return {}
    try:
        return json.loads(action_params_json)
    except (json.JSONDecodeError, TypeError):
        return {}


def _encode_params(action_type: str, params: dict[str, str]) -> list[float]:
    """Encode action-specific params into a 10-element float list."""
    result = [0.0] * _NUM_PARAMS
    if not params:
        return result

    if action_type == ACTION_TECH_RESEARCHED:
        result[0] = _hash_param(params.get("tech_id", ""))
    elif action_type == ACTION_BUILDING_CONSTRUCTED:
        result[0] = _hash_param(params.get("building_type", ""))
        result[1] = _hash_param(params.get("planet_id", ""))
    elif action_type == ACTION_DISTRICT_BUILT:
        result[0] = _hash_param(params.get("district_type", ""))
        result[1] = _hash_param(params.get("planet_id", ""))
        count_str = params.get("count", "1")
        result[2] = min(1.0, float(count_str) / 10.0) if count_str.isdigit() else 0.0
    elif action_type == ACTION_SYSTEM_CLAIMED:
        result[0] = _hash_param(params.get("system_id", ""))
    elif action_type == ACTION_SHIPS_BUILT:
        result[0] = _hash_param(params.get("ship_type", ""))
        count_str = params.get("count", "1")
        result[1] = min(1.0, float(count_str) / 20.0) if count_str.isdigit() else 0.0
    elif action_type == ACTION_TRADITION_UNLOCKED:
        result[0] = _hash_param(params.get("tradition_id", ""))
    elif action_type == ACTION_PERK_TAKEN:
        result[0] = _hash_param(params.get("perk_id", ""))
    elif action_type == ACTION_MEGASTRUCTURE_BUILT:
        result[0] = _hash_param(params.get("megastructure_type", ""))
        stage_str = params.get("stage", "0")
        result[1] = min(1.0, float(stage_str) / 5.0) if stage_str.isdigit() else 0.0
    elif action_type == ACTION_POLICY_CHANGED:
        result[0] = _hash_param(params.get("policy_category", ""))
        result[1] = _hash_param(params.get("new_value", ""))
        result[2] = _hash_param(params.get("old_value", ""))
    elif action_type == ACTION_COUNCIL_REASSIGNED:
        result[0] = _hash_param(params.get("position_type", ""))
        result[1] = _hash_param(params.get("new_leader_id", ""))

    return result


class ActionEncoder:
    """Encodes macro-actions into fixed-size float vectors.

    Encoding: [10 one-hot type bits | 10 continuous param features] = 20 floats.
    All values in [0, 1]. Empty/unknown actions produce all zeros.
    """

    def encode(self, action_type: str, action_params_json: str) -> list[float]:
        """Encode from raw strings (as stored in Parquet)."""
        one_hot = [0.0] * _NUM_TYPES
        if action_type in ACTION_TYPES:
            one_hot[ACTION_TYPES.index(action_type)] = 1.0
        params = _parse_params(action_params_json)
        param_vec = _encode_params(action_type, params)
        return one_hot + param_vec

    def encode_macro_action(self, action: MacroAction) -> list[float]:
        """Encode from a MacroAction instance."""
        return self.encode(action.action_type, json.dumps(dict(action.params)))

    def encode_batch(
        self,
        action_types: list[str],
        action_params_jsons: list[str],
    ) -> list[list[float]]:
        """Encode a batch of actions."""
        return [
            self.encode(at, ap) for at, ap in zip(action_types, action_params_jsons)
        ]

    def to_tensor(self, action_type: str, action_params_json: str) -> torch.Tensor:
        """Encode and return as torch float32 tensor."""
        import torch

        return torch.tensor(
            self.encode(action_type, action_params_json), dtype=torch.float32
        )

    def to_tensor_batch(
        self,
        action_types: list[str],
        action_params_jsons: list[str],
    ) -> torch.Tensor:
        """Encode batch and return as (N, 20) torch tensor."""
        import torch

        vecs = self.encode_batch(action_types, action_params_jsons)
        return torch.tensor(vecs, dtype=torch.float32)
