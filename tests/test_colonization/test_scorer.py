"""Tests for LightGBM colonization scorer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from stellaris_advisor.features.registry import SYSTEM_FEATURE_NAMES

SYSTEM_DIM = len(SYSTEM_FEATURE_NAMES)  # 12


@pytest.mark.unit
class TestBoosterWrapper:
    def test_booster_property_returns_inner_booster(self) -> None:
        from stellaris_advisor.colonization.scorer import _BoosterWrapper

        inner = MagicMock()
        wrapper = _BoosterWrapper(inner)
        assert wrapper.booster_ is inner

    def test_load_then_save_does_not_crash(self, tmp_path: Path) -> None:
        """ColonizationScorer.load(path).save(new_path) must not crash (bug N5)."""
        from stellaris_advisor.colonization.scorer import (
            ColonizationScorer,
            _BoosterWrapper,
        )

        # Build a scorer backed by a _BoosterWrapper (the load() code path)
        inner_booster = MagicMock()
        inner_booster.save_model = MagicMock()
        wrapper = _BoosterWrapper(inner_booster)
        scorer = ColonizationScorer(wrapper)

        new_path = tmp_path / "resaved_model.txt"
        # This must not raise AttributeError: '_BoosterWrapper has no attribute booster_'
        scorer.save(new_path)

        inner_booster.save_model.assert_called_once_with(str(new_path))


@pytest.mark.unit
class TestIsAvailable:
    def test_missing_returns_false(self, tmp_path: Path) -> None:
        from stellaris_advisor.colonization.scorer import ColonizationScorer

        assert not ColonizationScorer.is_available(tmp_path / "nope.txt")

    def test_existing_returns_true(self, tmp_path: Path) -> None:
        from stellaris_advisor.colonization.scorer import ColonizationScorer

        model_path = tmp_path / "col_model.txt"
        model_path.write_text("dummy")
        assert ColonizationScorer.is_available(model_path)


@pytest.mark.integration
class TestColonizationScorer:
    def test_predict_returns_float(self) -> None:
        lgb = pytest.importorskip("lightgbm")
        np = pytest.importorskip("numpy")
        from stellaris_advisor.colonization.scorer import ColonizationScorer

        # Train a tiny model for testing
        X = np.random.rand(20, SYSTEM_DIM).astype(np.float32)
        y = np.random.randint(0, 2, size=20)
        groups = [10, 10]
        ranker = lgb.LGBMRanker(n_estimators=5, num_leaves=4)
        ranker.fit(X, y, group=groups)

        scorer = ColonizationScorer(ranker)
        score = scorer.predict([0.5] * SYSTEM_DIM)
        assert isinstance(score, float)

    def test_rank_systems_sorted(self) -> None:
        lgb = pytest.importorskip("lightgbm")
        np = pytest.importorskip("numpy")
        from stellaris_advisor.colonization.scorer import ColonizationScorer

        X = np.random.rand(20, SYSTEM_DIM).astype(np.float32)
        y = np.random.randint(0, 2, size=20)
        ranker = lgb.LGBMRanker(n_estimators=5, num_leaves=4)
        ranker.fit(X, y, group=[10, 10])

        scorer = ColonizationScorer(ranker)
        systems = [[0.1] * SYSTEM_DIM, [0.9] * SYSTEM_DIM, [0.5] * SYSTEM_DIM]
        ranked = scorer.rank_systems(systems)
        assert len(ranked) == 3
        # Sorted descending
        for i in range(len(ranked) - 1):
            assert ranked[i][0] >= ranked[i + 1][0]

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        lgb = pytest.importorskip("lightgbm")
        np = pytest.importorskip("numpy")
        from stellaris_advisor.colonization.scorer import ColonizationScorer

        X = np.random.rand(20, SYSTEM_DIM).astype(np.float32)
        y = np.random.randint(0, 2, size=20)
        ranker = lgb.LGBMRanker(n_estimators=5, num_leaves=4)
        ranker.fit(X, y, group=[10, 10])

        scorer = ColonizationScorer(ranker)
        path = tmp_path / "model.txt"
        scorer.save(path)

        loaded = ColonizationScorer.load(path)
        orig_score = scorer.predict([0.5] * SYSTEM_DIM)
        loaded_score = loaded.predict([0.5] * SYSTEM_DIM)
        assert abs(orig_score - loaded_score) < 1e-6
