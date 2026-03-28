"""Tests for colonization model training."""

from __future__ import annotations

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
