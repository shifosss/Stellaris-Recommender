"""Parquet schema constants for the data pipeline.

pyarrow is an optional [ml] dependency. Schema constants are constructed
lazily on first access via module __getattr__ so that importing this module
never fails without pyarrow installed.
"""
from __future__ import annotations

from typing import Any

from stellaris_advisor.features.registry import (
    EMPIRE_FEATURE_NAMES,
    SYSTEM_FEATURE_NAMES,
)


def _empire_features_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("game_id", pa.string()),
            pa.field("year", pa.int32()),
            pa.field("empire_id", pa.int32()),
            *(pa.field(name, pa.float64()) for name in EMPIRE_FEATURE_NAMES),
        ]
    )


def _system_features_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("game_id", pa.string()),
            pa.field("year", pa.int32()),
            pa.field("system_id", pa.int32()),
            *(pa.field(name, pa.float64()) for name in SYSTEM_FEATURE_NAMES),
        ]
    )


def _transitions_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("game_id", pa.string()),
            pa.field("year", pa.int32()),
            pa.field("empire_id", pa.int32()),
            pa.field("state_features", pa.list_(pa.float64())),
            pa.field("action_type", pa.string()),
            pa.field("action_params", pa.string()),
            pa.field("reward", pa.float64()),
            pa.field("next_state_features", pa.list_(pa.float64())),
            pa.field("terminal", pa.bool_()),
            pa.field("final_rank", pa.int32()),
        ]
    )


def _game_metadata_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("game_id", pa.string()),
            pa.field("game_version", pa.string()),
            pa.field("galaxy_size", pa.int32()),
            pa.field("difficulty", pa.string()),
            pa.field("crisis_active", pa.bool_()),
            pa.field("crisis_type", pa.string()),
            pa.field("war_in_heaven", pa.bool_()),
            pa.field("galactic_market_founded", pa.bool_()),
            pa.field("player_empire_id", pa.int32()),
            pa.field("num_empires", pa.int32()),
            pa.field("start_year", pa.int32()),
            pa.field("end_year", pa.int32()),
            pa.field("total_saves", pa.int32()),
        ]
    )


_SCHEMA_BUILDERS: dict[str, Any] = {
    "EMPIRE_FEATURES_SCHEMA": _empire_features_schema,
    "SYSTEM_FEATURES_SCHEMA": _system_features_schema,
    "TRANSITIONS_SCHEMA": _transitions_schema,
    "GAME_METADATA_SCHEMA": _game_metadata_schema,
}


def __getattr__(name: str) -> Any:
    if name in _SCHEMA_BUILDERS:
        schema = _SCHEMA_BUILDERS[name]()
        globals()[name] = schema  # cache for subsequent access
        return schema
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
