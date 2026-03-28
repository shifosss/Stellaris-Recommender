"""Tests for Parquet schema definitions.

Requires pyarrow (optional [ml] dependency). All tests are marked
@pytest.mark.integration and skipped if pyarrow is not installed.
"""
from __future__ import annotations

import pytest

pyarrow = pytest.importorskip("pyarrow")

from stellaris_advisor.features.registry import (  # noqa: E402
    EMPIRE_FEATURE_NAMES,
    SYSTEM_FEATURE_NAMES,
)


@pytest.mark.integration
class TestEmpireFeaturesSchema:
    def test_is_valid_schema(self) -> None:
        from stellaris_advisor.data.schemas import EMPIRE_FEATURES_SCHEMA

        assert isinstance(EMPIRE_FEATURES_SCHEMA, pyarrow.Schema)

    def test_has_id_columns_with_correct_types(self) -> None:
        from stellaris_advisor.data.schemas import EMPIRE_FEATURES_SCHEMA

        assert EMPIRE_FEATURES_SCHEMA.field("game_id").type == pyarrow.string()
        assert EMPIRE_FEATURES_SCHEMA.field("year").type == pyarrow.int32()
        assert EMPIRE_FEATURES_SCHEMA.field("empire_id").type == pyarrow.int32()

    def test_has_all_feature_columns(self) -> None:
        from stellaris_advisor.data.schemas import EMPIRE_FEATURES_SCHEMA

        schema_names = set(EMPIRE_FEATURES_SCHEMA.names)
        for name in EMPIRE_FEATURE_NAMES:
            assert name in schema_names, f"Missing feature column: {name}"

    def test_feature_columns_are_float64(self) -> None:
        from stellaris_advisor.data.schemas import EMPIRE_FEATURES_SCHEMA

        for name in EMPIRE_FEATURE_NAMES:
            field = EMPIRE_FEATURES_SCHEMA.field(name)
            assert field.type == pyarrow.float64(), (
                f"Column '{name}' should be float64, got {field.type}"
            )

    def test_column_count(self) -> None:
        from stellaris_advisor.data.schemas import EMPIRE_FEATURES_SCHEMA

        # 3 id columns (game_id, year, empire_id) + 109 feature columns
        expected = 3 + len(EMPIRE_FEATURE_NAMES)
        assert len(EMPIRE_FEATURES_SCHEMA.names) == expected


@pytest.mark.integration
class TestSystemFeaturesSchema:
    def test_is_valid_schema(self) -> None:
        from stellaris_advisor.data.schemas import SYSTEM_FEATURES_SCHEMA

        assert isinstance(SYSTEM_FEATURES_SCHEMA, pyarrow.Schema)

    def test_has_id_columns_with_correct_types(self) -> None:
        from stellaris_advisor.data.schemas import SYSTEM_FEATURES_SCHEMA

        assert SYSTEM_FEATURES_SCHEMA.field("game_id").type == pyarrow.string()
        assert SYSTEM_FEATURES_SCHEMA.field("year").type == pyarrow.int32()
        assert SYSTEM_FEATURES_SCHEMA.field("system_id").type == pyarrow.int32()

    def test_has_all_feature_columns(self) -> None:
        from stellaris_advisor.data.schemas import SYSTEM_FEATURES_SCHEMA

        schema_names = set(SYSTEM_FEATURES_SCHEMA.names)
        for name in SYSTEM_FEATURE_NAMES:
            assert name in schema_names, f"Missing feature column: {name}"

    def test_feature_columns_are_float64(self) -> None:
        from stellaris_advisor.data.schemas import SYSTEM_FEATURES_SCHEMA

        for name in SYSTEM_FEATURE_NAMES:
            field = SYSTEM_FEATURES_SCHEMA.field(name)
            assert field.type == pyarrow.float64(), (
                f"Column '{name}' should be float64, got {field.type}"
            )

    def test_column_count(self) -> None:
        from stellaris_advisor.data.schemas import SYSTEM_FEATURES_SCHEMA

        # 3 id columns (game_id, year, system_id) + 12 feature columns
        expected = 3 + len(SYSTEM_FEATURE_NAMES)
        assert len(SYSTEM_FEATURES_SCHEMA.names) == expected


@pytest.mark.integration
class TestTransitionsSchema:
    def test_is_valid_schema(self) -> None:
        from stellaris_advisor.data.schemas import TRANSITIONS_SCHEMA

        assert isinstance(TRANSITIONS_SCHEMA, pyarrow.Schema)

    def test_has_required_scalar_columns(self) -> None:
        from stellaris_advisor.data.schemas import TRANSITIONS_SCHEMA

        expected_types: dict[str, pyarrow.DataType] = {
            "game_id": pyarrow.string(),
            "year": pyarrow.int32(),
            "empire_id": pyarrow.int32(),
            "action_type": pyarrow.string(),
            "action_params": pyarrow.string(),
            "reward": pyarrow.float64(),
            "terminal": pyarrow.bool_(),
            "final_rank": pyarrow.int32(),
        }
        for col, expected_type in expected_types.items():
            field = TRANSITIONS_SCHEMA.field(col)
            assert field.type == expected_type, (
                f"Column '{col}' should be {expected_type}, got {field.type}"
            )

    def test_has_state_feature_list_columns(self) -> None:
        from stellaris_advisor.data.schemas import TRANSITIONS_SCHEMA

        list_float64 = pyarrow.list_(pyarrow.float64())
        assert TRANSITIONS_SCHEMA.field("state_features").type == list_float64
        assert TRANSITIONS_SCHEMA.field("next_state_features").type == list_float64


@pytest.mark.integration
class TestGameMetadataSchema:
    def test_is_valid_schema(self) -> None:
        from stellaris_advisor.data.schemas import GAME_METADATA_SCHEMA

        assert isinstance(GAME_METADATA_SCHEMA, pyarrow.Schema)

    def test_has_game_id(self) -> None:
        from stellaris_advisor.data.schemas import GAME_METADATA_SCHEMA

        assert GAME_METADATA_SCHEMA.field("game_id").type == pyarrow.string()

    def test_has_galaxy_config_fields_with_correct_types(self) -> None:
        from stellaris_advisor.data.schemas import GAME_METADATA_SCHEMA

        expected_types: dict[str, pyarrow.DataType] = {
            "game_version": pyarrow.string(),
            "galaxy_size": pyarrow.int32(),
            "difficulty": pyarrow.string(),
            "crisis_active": pyarrow.bool_(),
            "war_in_heaven": pyarrow.bool_(),
            "galactic_market_founded": pyarrow.bool_(),
            "player_empire_id": pyarrow.int32(),
            "num_empires": pyarrow.int32(),
            "start_year": pyarrow.int32(),
            "end_year": pyarrow.int32(),
            "total_saves": pyarrow.int32(),
        }
        for col, expected_type in expected_types.items():
            field = GAME_METADATA_SCHEMA.field(col)
            assert field.type == expected_type, (
                f"Column '{col}' should be {expected_type}, got {field.type}"
            )
