"""Shared fixtures for RL tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture()
def tiny_transitions_parquet(tmp_path: Path) -> Path:
    """Write a minimal transitions.parquet with 12 rows across 3 game_ids."""
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    from stellaris_advisor.data.schemas import TRANSITIONS_SCHEMA
    from stellaris_advisor.features.registry import (
        EMPIRE_FEATURE_NAMES,
        TEMPORAL_FEATURE_NAMES,
    )

    state_dim = len(EMPIRE_FEATURE_NAMES) + len(TEMPORAL_FEATURE_NAMES)
    rows: list[dict] = []
    for game_idx in range(3):
        game_id = f"game_{game_idx}"
        for step in range(4):
            year = 2200 + step * 25
            is_terminal = step == 3
            rows.append(
                {
                    "game_id": game_id,
                    "year": year,
                    "empire_id": 0,
                    "state_features": [float(i + step) for i in range(state_dim)],
                    "action_type": (
                        "tech_researched" if step % 2 == 0 else "building_constructed"
                    ),
                    "action_params": json.dumps({"tech_id": f"tech_{step}"}),
                    "reward": 0.1 * (step + 1),
                    "next_state_features": [
                        float(i + step + 1) for i in range(state_dim)
                    ],
                    "terminal": is_terminal,
                    "final_rank": 1 if is_terminal else 0,
                }
            )

    table = pa.table(
        {col: [r[col] for r in rows] for col in TRANSITIONS_SCHEMA.names},
        schema=TRANSITIONS_SCHEMA,
    )
    path = tmp_path / "transitions.parquet"
    pq.write_table(table, path)
    return path


@pytest.fixture()
def tiny_system_features_parquet(tmp_path: Path) -> Path:
    """Write minimal system_features.parquet for colonization tests."""
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    from stellaris_advisor.data.schemas import SYSTEM_FEATURES_SCHEMA
    from stellaris_advisor.features.registry import SYSTEM_FEATURE_NAMES

    rows: list[dict] = []
    for game_idx in range(2):
        game_id = f"game_{game_idx}"
        for sys_id in range(5):
            rows.append(
                {
                    "game_id": game_id,
                    "year": 2225,
                    "system_id": sys_id,
                    **{name: float(sys_id + 1) * 0.1 for name in SYSTEM_FEATURE_NAMES},
                }
            )

    table = pa.table(
        {col: [r[col] for r in rows] for col in SYSTEM_FEATURES_SCHEMA.names},
        schema=SYSTEM_FEATURES_SCHEMA,
    )
    path = tmp_path / "system_features.parquet"
    pq.write_table(table, path)
    return path
