"""Tests for standalone V(s) and Q(s,a) network architectures."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from stellaris_advisor.features.registry import (
    EMPIRE_FEATURE_NAMES,
    TEMPORAL_FEATURE_NAMES,
)  # noqa: E402
from stellaris_advisor.rl.action_encoder import ACTION_DIM  # noqa: E402

STATE_DIM = len(EMPIRE_FEATURE_NAMES) + len(TEMPORAL_FEATURE_NAMES)


@pytest.mark.unit
class TestValueNetwork:
    def test_output_shape_single(self) -> None:
        from stellaris_advisor.rl.networks import ValueNetwork

        net = ValueNetwork(state_dim=STATE_DIM).eval()
        x = torch.randn(1, STATE_DIM)
        out = net(x)
        assert out.shape == (1, 1)

    def test_output_shape_batch(self) -> None:
        from stellaris_advisor.rl.networks import ValueNetwork

        net = ValueNetwork(state_dim=STATE_DIM)
        x = torch.randn(32, STATE_DIM)
        out = net(x)
        assert out.shape == (32, 1)

    def test_custom_dim(self) -> None:
        from stellaris_advisor.rl.networks import ValueNetwork

        net = ValueNetwork(state_dim=10)
        x = torch.randn(4, 10)
        out = net(x)
        assert out.shape == (4, 1)

    def test_has_parameters(self) -> None:
        from stellaris_advisor.rl.networks import ValueNetwork

        net = ValueNetwork(state_dim=STATE_DIM)
        params = list(net.parameters())
        assert len(params) > 0
        total = sum(p.numel() for p in params)
        assert total > 0

    def test_is_nn_module(self) -> None:
        from stellaris_advisor.rl.networks import ValueNetwork

        net = ValueNetwork(state_dim=STATE_DIM)
        assert isinstance(net, torch.nn.Module)


@pytest.mark.unit
class TestQNetwork:
    def test_output_shape(self) -> None:
        from stellaris_advisor.rl.networks import QNetwork

        net = QNetwork(state_dim=STATE_DIM, action_dim=ACTION_DIM)
        s = torch.randn(8, STATE_DIM)
        a = torch.randn(8, ACTION_DIM)
        out = net(s, a)
        assert out.shape == (8, 1)

    def test_custom_dims(self) -> None:
        from stellaris_advisor.rl.networks import QNetwork

        net = QNetwork(state_dim=10, action_dim=5)
        s = torch.randn(4, 10)
        a = torch.randn(4, 5)
        out = net(s, a)
        assert out.shape == (4, 1)

    def test_gradient_flows(self) -> None:
        from stellaris_advisor.rl.networks import QNetwork

        net = QNetwork(state_dim=10, action_dim=5)
        s = torch.randn(4, 10, requires_grad=True)
        a = torch.randn(4, 5, requires_grad=True)
        out = net(s, a)
        loss = out.sum()
        loss.backward()
        assert s.grad is not None
