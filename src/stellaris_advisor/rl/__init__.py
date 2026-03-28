"""Offline RL training and inference package.

All ML imports are lazy — importing this module succeeds without torch/d3rlpy.
"""

from __future__ import annotations

from typing import Any

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "StateEncoder": ("stellaris_advisor.rl.state_encoder", "StateEncoder"),
    "NormStats": ("stellaris_advisor.rl.state_encoder", "NormStats"),
    "ActionEncoder": ("stellaris_advisor.rl.action_encoder", "ActionEncoder"),
    "ACTION_TYPES": ("stellaris_advisor.rl.action_encoder", "ACTION_TYPES"),
    "ACTION_DIM": ("stellaris_advisor.rl.action_encoder", "ACTION_DIM"),
    "ValueNetwork": ("stellaris_advisor.rl.networks", "ValueNetwork"),
    "QNetwork": ("stellaris_advisor.rl.networks", "QNetwork"),
    "DataSplit": ("stellaris_advisor.rl.dataset", "DataSplit"),
    "RLArrays": ("stellaris_advisor.rl.dataset", "RLArrays"),
    "split_by_game": ("stellaris_advisor.rl.dataset", "split_by_game"),
    "load_transitions": ("stellaris_advisor.rl.dataset", "load_transitions"),
    "build_mdp_dataset": ("stellaris_advisor.rl.dataset", "build_mdp_dataset"),
    "load_and_split": ("stellaris_advisor.rl.dataset", "load_and_split"),
    "TrainResult": ("stellaris_advisor.rl.trainer", "TrainResult"),
    "create_algorithm": ("stellaris_advisor.rl.trainer", "create_algorithm"),
    "train": ("stellaris_advisor.rl.trainer", "train"),
    "RLPredictor": ("stellaris_advisor.rl.inference", "RLPredictor"),
    "EvalResult": ("stellaris_advisor.rl.evaluation", "EvalResult"),
    "evaluate": ("stellaris_advisor.rl.evaluation", "evaluate"),
    "compute_discounted_returns": (
        "stellaris_advisor.rl.evaluation",
        "compute_discounted_returns",
    ),
    "TuningResult": ("stellaris_advisor.rl.tuning", "TuningResult"),
    "tune": ("stellaris_advisor.rl.tuning", "tune"),
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
