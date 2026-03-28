"""Public API for the stellaris_advisor.data subpackage.

Schema constants (EMPIRE_FEATURES_SCHEMA etc.) are re-exported lazily via
__getattr__ so that importing this module never fails without pyarrow installed.
"""
from __future__ import annotations

from typing import Any

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
    extract_actions,
)
from stellaris_advisor.data.ingest import IngestResult, ingest_saves
from stellaris_advisor.data.rewards import (
    DEFAULT_WEIGHTS,
    TERMINAL_BONUS,
    RewardWeights,
    composite_score,
    compute_reward,
)
from stellaris_advisor.data.synthetic import (
    SyntheticConfig,
    generate_launch_script,
    generate_orchestrator_config,
    generate_random_config,
)

_SCHEMA_NAMES = frozenset(
    {
        "EMPIRE_FEATURES_SCHEMA",
        "SYSTEM_FEATURES_SCHEMA",
        "TRANSITIONS_SCHEMA",
        "GAME_METADATA_SCHEMA",
    }
)


def __getattr__(name: str) -> Any:
    if name in _SCHEMA_NAMES:
        from stellaris_advisor.data import schemas

        return getattr(schemas, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # schemas (lazy — require pyarrow)
    "EMPIRE_FEATURES_SCHEMA",
    "SYSTEM_FEATURES_SCHEMA",
    "TRANSITIONS_SCHEMA",
    "GAME_METADATA_SCHEMA",
    # actions
    "ACTION_BUILDING_CONSTRUCTED",
    "ACTION_COUNCIL_REASSIGNED",
    "ACTION_DISTRICT_BUILT",
    "ACTION_MEGASTRUCTURE_BUILT",
    "ACTION_PERK_TAKEN",
    "ACTION_POLICY_CHANGED",
    "ACTION_SHIPS_BUILT",
    "ACTION_SYSTEM_CLAIMED",
    "ACTION_TECH_RESEARCHED",
    "ACTION_TRADITION_UNLOCKED",
    "MacroAction",
    "extract_actions",
    # ingest
    "IngestResult",
    "ingest_saves",
    # rewards
    "DEFAULT_WEIGHTS",
    "TERMINAL_BONUS",
    "RewardWeights",
    "composite_score",
    "compute_reward",
    # synthetic
    "SyntheticConfig",
    "generate_launch_script",
    "generate_orchestrator_config",
    "generate_random_config",
]
