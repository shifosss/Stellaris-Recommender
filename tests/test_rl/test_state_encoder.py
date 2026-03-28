"""Tests for state encoder with running mean/std normalization."""

from __future__ import annotations

import json
import math

import pytest

from stellaris_advisor.features.registry import (
    EMPIRE_FEATURE_NAMES,
    TEMPORAL_FEATURE_NAMES,
)

STATE_DIM = len(EMPIRE_FEATURE_NAMES) + len(TEMPORAL_FEATURE_NAMES)  # 117


@pytest.mark.unit
class TestNormStats:
    def test_frozen(self) -> None:
        from stellaris_advisor.rl.state_encoder import NormStats

        stats = NormStats(
            mean=(0.0,) * STATE_DIM,
            std=(1.0,) * STATE_DIM,
            count=0,
        )
        with pytest.raises(AttributeError):
            stats.count = 5  # type: ignore[misc]

    def test_dimension(self) -> None:
        from stellaris_advisor.rl.state_encoder import NormStats

        stats = NormStats(
            mean=(1.0, 2.0),
            std=(0.5, 0.5),
            count=10,
        )
        assert len(stats.mean) == 2
        assert len(stats.std) == 2


@pytest.mark.unit
class TestStateEncoder:
    def test_default_dim_from_registry(self) -> None:
        from stellaris_advisor.rl.state_encoder import StateEncoder

        enc = StateEncoder()
        assert enc.state_dim == STATE_DIM

    def test_custom_dim(self) -> None:
        from stellaris_advisor.rl.state_encoder import StateEncoder

        enc = StateEncoder(state_dim=10)
        assert enc.state_dim == 10

    def test_fit_updates_stats(self) -> None:
        from stellaris_advisor.rl.state_encoder import StateEncoder

        enc = StateEncoder(state_dim=3)
        data = [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0]]
        enc.fit(data)
        stats = enc.stats
        assert stats.count == 3
        assert math.isclose(stats.mean[0], 3.0, abs_tol=1e-6)
        assert math.isclose(stats.mean[1], 4.0, abs_tol=1e-6)

    def test_update_welford(self) -> None:
        from stellaris_advisor.rl.state_encoder import StateEncoder

        enc = StateEncoder(state_dim=2)
        enc.fit([[1.0, 2.0], [3.0, 4.0]])
        enc.update([[5.0, 6.0], [7.0, 8.0]])
        stats = enc.stats
        assert stats.count == 4
        assert math.isclose(stats.mean[0], 4.0, abs_tol=1e-6)

    def test_encode_shape(self) -> None:
        torch = pytest.importorskip("torch")
        from stellaris_advisor.rl.state_encoder import StateEncoder

        enc = StateEncoder(state_dim=3)
        enc.fit([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
        t = enc.encode([2.0, 3.0, 4.0])
        assert isinstance(t, torch.Tensor)
        assert t.shape == (3,)
        assert t.dtype == torch.float32

    def test_encode_batch_shape(self) -> None:
        pytest.importorskip("torch")
        from stellaris_advisor.rl.state_encoder import StateEncoder

        enc = StateEncoder(state_dim=3)
        enc.fit([[1.0, 2.0, 3.0]])
        t = enc.encode_batch([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert t.shape == (2, 3)

    def test_zero_std_epsilon(self) -> None:
        torch = pytest.importorskip("torch")
        from stellaris_advisor.rl.state_encoder import StateEncoder

        enc = StateEncoder(state_dim=2)
        enc.fit([[5.0, 5.0], [5.0, 5.0]])  # zero variance
        t = enc.encode([5.0, 5.0])
        assert not torch.any(torch.isnan(t))
        assert not torch.any(torch.isinf(t))

    def test_save_load_roundtrip(self, tmp_path: object) -> None:
        from pathlib import Path

        from stellaris_advisor.rl.state_encoder import StateEncoder

        p = Path(str(tmp_path)) / "stats.json"
        enc = StateEncoder(state_dim=3)
        enc.fit([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        enc.save(p)

        enc2 = StateEncoder(state_dim=3)
        enc2.load(p)
        assert enc.stats == enc2.stats

    def test_save_is_json_readable(self, tmp_path: object) -> None:
        from pathlib import Path

        from stellaris_advisor.rl.state_encoder import StateEncoder

        p = Path(str(tmp_path)) / "stats.json"
        enc = StateEncoder(state_dim=2)
        enc.fit([[1.0, 2.0]])
        enc.save(p)
        data = json.loads(p.read_text())
        assert "mean" in data
        assert "std" in data
        assert "count" in data

    def test_encode_before_fit_raises(self) -> None:
        pytest.importorskip("torch")
        from stellaris_advisor.rl.state_encoder import StateEncoder

        enc = StateEncoder(state_dim=3)
        with pytest.raises(RuntimeError, match="fit"):
            enc.encode([1.0, 2.0, 3.0])
