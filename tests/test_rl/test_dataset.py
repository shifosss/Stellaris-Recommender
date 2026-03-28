"""Tests for dataset loading and game-level splitting."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.unit
class TestSplitByGame:
    def test_no_overlap(self) -> None:
        from stellaris_advisor.rl.dataset import split_by_game

        split = split_by_game(
            game_ids=["g0", "g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9"],
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )
        all_ids = (
            set(split.train_game_ids)
            | set(split.val_game_ids)
            | set(split.test_game_ids)
        )
        assert len(all_ids) == 10
        assert len(set(split.train_game_ids) & set(split.val_game_ids)) == 0
        assert len(set(split.train_game_ids) & set(split.test_game_ids)) == 0

    def test_holdout_goes_to_test(self) -> None:
        from stellaris_advisor.rl.dataset import split_by_game

        split = split_by_game(
            game_ids=["g0", "g1", "g2", "g3", "1000_grand_admiral_run1"],
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            holdout_patterns=("1000_grand_admiral",),
        )
        assert "1000_grand_admiral_run1" in split.test_game_ids

    def test_deterministic(self) -> None:
        from stellaris_advisor.rl.dataset import split_by_game

        s1 = split_by_game(game_ids=["a", "b", "c", "d", "e"], seed=42)
        s2 = split_by_game(game_ids=["a", "b", "c", "d", "e"], seed=42)
        assert s1 == s2

    def test_at_least_one_per_split(self) -> None:
        from stellaris_advisor.rl.dataset import split_by_game

        split = split_by_game(game_ids=["g0", "g1", "g2"])
        assert len(split.train_game_ids) >= 1
        assert len(split.val_game_ids) >= 1
        assert len(split.test_game_ids) >= 1


@pytest.mark.integration
class TestLoadTransitions:
    def test_load_shapes(self, tiny_transitions_parquet: Path) -> None:
        pytest.importorskip("pyarrow")
        from stellaris_advisor.features.registry import (
            EMPIRE_FEATURE_NAMES,
            TEMPORAL_FEATURE_NAMES,
        )
        from stellaris_advisor.rl.action_encoder import ACTION_DIM
        from stellaris_advisor.rl.dataset import load_transitions

        state_dim = len(EMPIRE_FEATURE_NAMES) + len(TEMPORAL_FEATURE_NAMES)
        arrays = load_transitions(
            tiny_transitions_parquet, game_ids=["game_0", "game_1"]
        )
        # 2 games × 4 rows each = 8 rows
        assert arrays.observations.shape == (8, state_dim)
        assert arrays.actions.shape == (8, ACTION_DIM)
        assert arrays.rewards.shape == (8,)
        assert arrays.terminals.shape == (8,)

    def test_arrays_not_writable(self, tiny_transitions_parquet: Path) -> None:
        pytest.importorskip("pyarrow")
        from stellaris_advisor.rl.dataset import load_transitions

        arrays = load_transitions(tiny_transitions_parquet, game_ids=["game_0"])
        assert not arrays.observations.flags.writeable
