"""Tests for colonization model training."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.mark.integration
class TestLabelCreation:
    def test_creates_labels(
        self,
        tiny_system_features_parquet: Path,
        tiny_transitions_parquet: Path,
    ) -> None:
        pytest.importorskip("pyarrow")
        from stellaris_advisor.colonization.training import create_colonization_labels

        X, y, groups = create_colonization_labels(
            system_features_path=tiny_system_features_parquet,
            transitions_path=tiny_transitions_parquet,
        )
        assert X.shape[1] == 12  # SYSTEM_DIM
        assert len(y) == len(X)
        assert sum(groups) == len(X)


@pytest.mark.unit
class TestLabelBugsN6N7N8:
    """Regression tests for N6 (future-year label leak), N7 (AI empire claims),
    and N8 (empty X raises ValueError)."""

    def _write_system_features(
        self,
        tmp_path: Path,
        rows: list[dict],
    ) -> Path:
        """Write a system_features.parquet from explicit row dicts."""
        pa = pytest.importorskip("pyarrow")
        import pyarrow.parquet as pq

        from stellaris_advisor.data.schemas import SYSTEM_FEATURES_SCHEMA
        from stellaris_advisor.features.registry import SYSTEM_FEATURE_NAMES

        # Fill in any missing feature columns with 0.5
        full_rows = [
            {
                **{name: 0.5 for name in SYSTEM_FEATURE_NAMES},
                **row,
            }
            for row in rows
        ]
        table = pa.table(
            {col: [r[col] for r in full_rows] for col in SYSTEM_FEATURES_SCHEMA.names},
            schema=SYSTEM_FEATURES_SCHEMA,
        )
        path = tmp_path / "system_features.parquet"
        pq.write_table(table, path)
        return path

    def _write_transitions(
        self,
        tmp_path: Path,
        rows: list[dict],
    ) -> Path:
        """Write a transitions.parquet from explicit row dicts."""
        pa = pytest.importorskip("pyarrow")
        import pyarrow.parquet as pq

        from stellaris_advisor.data.schemas import TRANSITIONS_SCHEMA
        from stellaris_advisor.features.registry import (
            EMPIRE_FEATURE_NAMES,
            TEMPORAL_FEATURE_NAMES,
        )

        state_dim = len(EMPIRE_FEATURE_NAMES) + len(TEMPORAL_FEATURE_NAMES)
        defaults = {
            "state_features": [0.0] * state_dim,
            "next_state_features": [0.0] * state_dim,
            "reward": 0.0,
            "terminal": False,
            "final_rank": 0,
        }
        full_rows = [{**defaults, **row} for row in rows]
        table = pa.table(
            {col: [r[col] for r in full_rows] for col in TRANSITIONS_SCHEMA.names},
            schema=TRANSITIONS_SCHEMA,
        )
        path = tmp_path / "transitions.parquet"
        pq.write_table(table, path)
        return path

    def test_n6_future_year_does_not_leak_into_earlier_snapshot(
        self, tmp_path: Path
    ) -> None:
        """System 3 claimed at year=2275; snapshot at year=2225 must be label=0."""
        pytest.importorskip("pyarrow")
        from stellaris_advisor.colonization.training import create_colonization_labels

        sys_path = self._write_system_features(
            tmp_path,
            [
                # year=2225 snapshot — system 3 not yet claimed
                {"game_id": "g0", "year": 2225, "system_id": 3},
                {"game_id": "g0", "year": 2225, "system_id": 4},
                # year=2275 snapshot — system 3 claimed in this year
                {"game_id": "g0", "year": 2275, "system_id": 3},
                {"game_id": "g0", "year": 2275, "system_id": 4},
            ],
        )
        trans_path = self._write_transitions(
            tmp_path,
            [
                {
                    "game_id": "g0",
                    "year": 2275,
                    "empire_id": 0,
                    "action_type": "system_claimed",
                    "action_params": json.dumps({"system_id": 3}),
                },
            ],
        )

        X, y, groups = create_colonization_labels(
            sys_path, trans_path, player_empire_id=0
        )

        # Rows 0-1: year=2225 group; rows 2-3: year=2275 group
        assert y[0] == 0, "system_id=3 at year=2225 must be 0 (not yet claimed)"
        assert y[1] == 0, "system_id=4 at year=2225 must be 0"
        assert y[2] == 1, "system_id=3 at year=2275 must be 1 (claimed this year)"
        assert y[3] == 0, "system_id=4 at year=2275 must be 0"

    def test_n7_ai_empire_claims_do_not_produce_positive_labels(
        self, tmp_path: Path
    ) -> None:
        """Only player empire_id=0 claims should yield label=1."""
        pytest.importorskip("pyarrow")
        from stellaris_advisor.colonization.training import create_colonization_labels

        sys_path = self._write_system_features(
            tmp_path,
            [
                {"game_id": "g0", "year": 2225, "system_id": 10},
                {"game_id": "g0", "year": 2225, "system_id": 11},
            ],
        )
        trans_path = self._write_transitions(
            tmp_path,
            [
                # Player empire claims system 10
                {
                    "game_id": "g0",
                    "year": 2225,
                    "empire_id": 0,
                    "action_type": "system_claimed",
                    "action_params": json.dumps({"system_id": 10}),
                },
                # AI empire claims system 11 — must NOT become label=1
                {
                    "game_id": "g0",
                    "year": 2225,
                    "empire_id": 1,
                    "action_type": "system_claimed",
                    "action_params": json.dumps({"system_id": 11}),
                },
            ],
        )

        X, y, groups = create_colonization_labels(
            sys_path, trans_path, player_empire_id=0
        )

        assert y[0] == 1, "system_id=10 claimed by player (empire_id=0) must be 1"
        assert y[1] == 0, "system_id=11 claimed by AI (empire_id=1) must be 0"

    def test_n8_empty_system_features_raises_value_error(self, tmp_path: Path) -> None:
        """train_colonization with no system rows must raise ValueError immediately."""
        pytest.importorskip("pyarrow")
        from stellaris_advisor.colonization.training import train_colonization

        # Empty system features file
        sys_path = self._write_system_features(tmp_path, [])
        # Transitions file with no system_claimed rows either
        trans_path = self._write_transitions(
            tmp_path,
            [
                {
                    "game_id": "g0",
                    "year": 2200,
                    "empire_id": 0,
                    "action_type": "tech_researched",
                    "action_params": json.dumps({"tech_id": "t1"}),
                }
            ],
        )

        with pytest.raises(ValueError, match="No system features found"):
            train_colonization(
                system_features_path=sys_path,
                transitions_path=trans_path,
                output_path=tmp_path / "col_model.txt",
            )


@pytest.mark.slow
class TestColonizationTraining:
    def test_train_returns_result(
        self,
        tiny_system_features_parquet: Path,
        tiny_transitions_parquet: Path,
        tmp_path: Path,
    ) -> None:
        pytest.importorskip("lightgbm")
        from stellaris_advisor.colonization.training import (
            ColonizationTrainResult,
            train_colonization,
        )

        result = train_colonization(
            system_features_path=tiny_system_features_parquet,
            transitions_path=tiny_transitions_parquet,
            output_path=tmp_path / "col_model.txt",
            n_estimators=5,
        )
        assert isinstance(result, ColonizationTrainResult)
        assert result.model_path.exists()

    def test_result_frozen(
        self,
        tiny_system_features_parquet: Path,
        tiny_transitions_parquet: Path,
        tmp_path: Path,
    ) -> None:
        pytest.importorskip("lightgbm")
        from stellaris_advisor.colonization.training import train_colonization

        result = train_colonization(
            system_features_path=tiny_system_features_parquet,
            transitions_path=tiny_transitions_parquet,
            output_path=tmp_path / "col_model.txt",
            n_estimators=5,
        )
        with pytest.raises(AttributeError):
            result.ndcg_at_10 = 0.99  # type: ignore[misc]
