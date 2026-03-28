"""Tests for benchmark percentile store."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.mark.unit
class TestPercentileEntry:
    def test_frozen(self) -> None:
        from stellaris_advisor.colonization.benchmark import PercentileEntry

        entry = PercentileEntry(p25=10.0, p50=20.0, p75=30.0, p90=40.0)
        with pytest.raises(AttributeError):
            entry.p50 = 99.0  # type: ignore[misc]


@pytest.mark.unit
class TestBenchmarkStore:
    def test_hardcoded_defaults_no_ml(self) -> None:
        from stellaris_advisor.colonization.benchmark import BenchmarkStore

        store = BenchmarkStore.hardcoded_defaults()
        entry = store.lookup(2225, "captain", "standard", "alloy_income")
        assert entry is not None
        assert entry.p25 < entry.p50 < entry.p75 < entry.p90

    def test_lookup_missing_returns_none(self) -> None:
        from stellaris_advisor.colonization.benchmark import BenchmarkStore

        store = BenchmarkStore.hardcoded_defaults()
        result = store.lookup(9999, "impossible", "alien", "alloy_income")
        assert result is None

    def test_percentile_rank(self) -> None:
        from stellaris_advisor.colonization.benchmark import BenchmarkStore

        store = BenchmarkStore.hardcoded_defaults()
        rank = store.percentile_rank(2225, "captain", "standard", "alloy_income", 0.0)
        assert 0 <= rank <= 100

    def test_percentile_rank_high_value(self) -> None:
        from stellaris_advisor.colonization.benchmark import BenchmarkStore

        store = BenchmarkStore.hardcoded_defaults()
        rank = store.percentile_rank(2225, "captain", "standard", "alloy_income", 99999.0)
        assert rank >= 90

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        from stellaris_advisor.colonization.benchmark import BenchmarkStore

        store = BenchmarkStore.hardcoded_defaults()
        path = tmp_path / "benchmark.json"
        store.save(path)

        loaded = BenchmarkStore.load(path)
        entry_orig = store.lookup(2225, "captain", "standard", "alloy_income")
        entry_loaded = loaded.lookup(2225, "captain", "standard", "alloy_income")
        assert entry_orig == entry_loaded

    def test_save_is_valid_json(self, tmp_path: Path) -> None:
        from stellaris_advisor.colonization.benchmark import BenchmarkStore

        store = BenchmarkStore.hardcoded_defaults()
        path = tmp_path / "benchmark.json"
        store.save(path)
        data = json.loads(path.read_text())
        assert isinstance(data, dict)

    def test_year_bucketing(self) -> None:
        from stellaris_advisor.colonization.benchmark import BenchmarkStore

        store = BenchmarkStore.hardcoded_defaults()
        # Year 2237 should bucket to 2225
        entry = store.lookup(2237, "captain", "standard", "alloy_income")
        assert entry is not None
