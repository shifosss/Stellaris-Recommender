"""Tests for action encoding to fixed-size float vectors."""

from __future__ import annotations

from types import MappingProxyType

import pytest

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


@pytest.mark.unit
class TestActionEncoder:
    def test_action_types_match_data_module(self) -> None:
        from stellaris_advisor.rl.action_encoder import ACTION_TYPES

        expected = (
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
        assert ACTION_TYPES == expected

    def test_action_dim_is_20(self) -> None:
        from stellaris_advisor.rl.action_encoder import ACTION_DIM

        assert ACTION_DIM == 20

    def test_encode_returns_correct_length(self) -> None:
        from stellaris_advisor.rl.action_encoder import ActionEncoder

        enc = ActionEncoder()
        vec = enc.encode("tech_researched", '{"tech_id": "tech_corvettes"}')
        assert len(vec) == 20

    def test_encode_values_in_range(self) -> None:
        from stellaris_advisor.rl.action_encoder import ActionEncoder

        enc = ActionEncoder()
        vec = enc.encode(
            "building_constructed", '{"building_type": "building_alloy_foundry"}'
        )
        assert all(0.0 <= v <= 1.0 for v in vec)

    def test_one_hot_position_correct(self) -> None:
        from stellaris_advisor.rl.action_encoder import ACTION_TYPES, ActionEncoder

        enc = ActionEncoder()
        for i, atype in enumerate(ACTION_TYPES):
            vec = enc.encode(atype, "{}")
            assert vec[i] == 1.0
            # All other one-hot slots are 0
            for j in range(10):
                if j != i:
                    assert vec[j] == 0.0

    def test_encode_deterministic(self) -> None:
        from stellaris_advisor.rl.action_encoder import ActionEncoder

        enc = ActionEncoder()
        v1 = enc.encode("tech_researched", '{"tech_id": "tech_destroyers"}')
        v2 = enc.encode("tech_researched", '{"tech_id": "tech_destroyers"}')
        assert v1 == v2

    def test_encode_empty_action(self) -> None:
        from stellaris_advisor.rl.action_encoder import ActionEncoder

        enc = ActionEncoder()
        vec = enc.encode("", "{}")
        assert len(vec) == 20
        assert all(v == 0.0 for v in vec)

    def test_encode_macro_action(self) -> None:
        from stellaris_advisor.rl.action_encoder import ActionEncoder

        enc = ActionEncoder()
        action = MacroAction(
            action_type="tech_researched",
            params=MappingProxyType({"tech_id": "tech_corvettes"}),
        )
        vec = enc.encode_macro_action(action)
        assert len(vec) == 20
        assert vec[0] == 1.0  # tech_researched is index 0

    def test_encode_batch(self) -> None:
        from stellaris_advisor.rl.action_encoder import ActionEncoder

        enc = ActionEncoder()
        batch = enc.encode_batch(
            ["tech_researched", "ships_built"],
            ['{"tech_id": "tech_corvettes"}', '{"ship_type": "corvette"}'],
        )
        assert len(batch) == 2
        assert all(len(v) == 20 for v in batch)

    def test_to_tensor_shape(self) -> None:
        torch = pytest.importorskip("torch")
        from stellaris_advisor.rl.action_encoder import ActionEncoder

        enc = ActionEncoder()
        t = enc.to_tensor("tech_researched", '{"tech_id": "t1"}')
        assert isinstance(t, torch.Tensor)
        assert t.shape == (20,)

    def test_to_tensor_batch_shape(self) -> None:
        pytest.importorskip("torch")
        from stellaris_advisor.rl.action_encoder import ActionEncoder

        enc = ActionEncoder()
        t = enc.to_tensor_batch(
            ["tech_researched", "ships_built"],
            ['{"tech_id": "t1"}', '{"ship_type": "corvette"}'],
        )
        assert t.shape == (2, 20)
