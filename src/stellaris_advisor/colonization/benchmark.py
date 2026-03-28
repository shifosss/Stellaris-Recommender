"""Percentile data store for contextualising empire performance.

Works without ML imports — hardcoded_defaults() provides degraded mode.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

BENCHMARK_METRICS: tuple[str, ...] = (
    "alloy_income",
    "research_total",
    "fleet_power",
    "tech_count",
    "system_count",
    "influence_income",
    "unity_income",
    "traditions_completed",
)

# Map benchmark metric names to feature names in EMPIRE_FEATURE_NAMES
_METRIC_TO_FEATURE: dict[str, str] = {
    "alloy_income": "resource_net_alloys",
    "research_total": "tech_physics_research_speed",  # proxy: sum of 3 research speeds
    "fleet_power": "military_fleet_power",
    "tech_count": "tech_total_count",
    "system_count": "expansion_system_count",
    "influence_income": "resource_net_influence",
    "unity_income": "tradition_unity_income",
    "traditions_completed": "tradition_trees_completed",
}


@dataclass(frozen=True)
class PercentileEntry:
    """Immutable percentile distribution for one metric."""

    p25: float
    p50: float
    p75: float
    p90: float


def _bucket_year(year: int, bucket_size: int = 25) -> int:
    """Round year down to nearest bucket."""
    return (year // bucket_size) * bucket_size


class BenchmarkStore:
    """Stores pre-computed percentile distributions per (year_bucket, difficulty, empire_type, metric).

    Degraded mode: hardcoded_defaults() returns hand-tuned values.
    Full mode: from_empire_features() computes from training Parquet.
    """

    def __init__(self, data: dict[str, PercentileEntry]) -> None:
        self._data = data  # key: "year_bucket:difficulty:empire_type:metric"

    @staticmethod
    def _key(year_bucket: int, difficulty: str, empire_type: str, metric: str) -> str:
        return f"{year_bucket}:{difficulty}:{empire_type}:{metric}"

    def lookup(
        self,
        year: int,
        difficulty: str,
        empire_type: str,
        metric: str,
    ) -> PercentileEntry | None:
        """Look up percentiles for given context. Returns None if not found."""
        bucket = _bucket_year(year)
        key = self._key(bucket, difficulty, empire_type, metric)
        return self._data.get(key)

    def percentile_rank(
        self,
        year: int,
        difficulty: str,
        empire_type: str,
        metric: str,
        value: float,
    ) -> int:
        """Estimate percentile rank (0-100) for a value."""
        entry = self.lookup(year, difficulty, empire_type, metric)
        if entry is None:
            return 50  # neutral default
        if value <= entry.p25:
            return int(25 * value / max(entry.p25, 1e-8))
        if value <= entry.p50:
            return 25 + int(25 * (value - entry.p25) / max(entry.p50 - entry.p25, 1e-8))
        if value <= entry.p75:
            return 50 + int(25 * (value - entry.p50) / max(entry.p75 - entry.p50, 1e-8))
        if value <= entry.p90:
            return 75 + int(15 * (value - entry.p75) / max(entry.p90 - entry.p75, 1e-8))
        return min(100, 90 + int(10 * (value - entry.p90) / max(entry.p90, 1e-8)))

    def save(self, path: Path) -> None:
        """Persist as JSON."""
        serialized = {
            k: {"p25": v.p25, "p50": v.p50, "p75": v.p75, "p90": v.p90}
            for k, v in self._data.items()
        }
        path.write_text(json.dumps(serialized, indent=2))

    @classmethod
    def load(cls, path: Path) -> BenchmarkStore:
        """Load from JSON."""
        raw = json.loads(path.read_text())
        data = {
            k: PercentileEntry(p25=v["p25"], p50=v["p50"], p75=v["p75"], p90=v["p90"])
            for k, v in raw.items()
        }
        return cls(data)

    @classmethod
    def hardcoded_defaults(cls) -> BenchmarkStore:
        """Hand-tuned percentiles for captain difficulty, standard empire type.

        Provides degraded mode — no ML imports required.
        Year buckets: 2200-2500 in 25-year increments.
        """
        data: dict[str, PercentileEntry] = {}

        # Hardcoded by year bucket for captain/standard
        # Format: (year_bucket, metric) -> (p25, p50, p75, p90)
        defaults: dict[tuple[int, str], tuple[float, float, float, float]] = {
            # alloy_income grows ~linearly with year
            (2200, "alloy_income"): (5, 10, 20, 35),
            (2225, "alloy_income"): (15, 30, 50, 80),
            (2250, "alloy_income"): (30, 60, 100, 150),
            (2275, "alloy_income"): (50, 100, 170, 250),
            (2300, "alloy_income"): (80, 150, 250, 400),
            (2350, "alloy_income"): (120, 250, 400, 600),
            (2400, "alloy_income"): (200, 400, 600, 1000),
            (2450, "alloy_income"): (300, 500, 800, 1500),
            (2500, "alloy_income"): (400, 700, 1200, 2000),
            # research_total (sum of 3 areas)
            (2200, "research_total"): (50, 80, 120, 180),
            (2225, "research_total"): (80, 150, 250, 400),
            (2250, "research_total"): (150, 300, 500, 800),
            (2275, "research_total"): (250, 500, 800, 1200),
            (2300, "research_total"): (400, 800, 1200, 2000),
            (2350, "research_total"): (600, 1200, 2000, 3500),
            (2400, "research_total"): (1000, 2000, 3500, 6000),
            (2450, "research_total"): (1500, 3000, 5000, 8000),
            (2500, "research_total"): (2000, 4000, 7000, 12000),
            # fleet_power
            (2200, "fleet_power"): (500, 1000, 2000, 4000),
            (2225, "fleet_power"): (1500, 3000, 6000, 10000),
            (2250, "fleet_power"): (3000, 8000, 15000, 25000),
            (2275, "fleet_power"): (8000, 15000, 30000, 50000),
            (2300, "fleet_power"): (15000, 30000, 60000, 100000),
            (2350, "fleet_power"): (30000, 60000, 120000, 200000),
            (2400, "fleet_power"): (50000, 100000, 200000, 400000),
            (2450, "fleet_power"): (80000, 150000, 300000, 600000),
            (2500, "fleet_power"): (100000, 200000, 500000, 1000000),
            # tech_count
            (2200, "tech_count"): (5, 10, 15, 25),
            (2225, "tech_count"): (15, 25, 40, 60),
            (2250, "tech_count"): (30, 50, 70, 100),
            (2275, "tech_count"): (50, 80, 110, 150),
            (2300, "tech_count"): (70, 110, 150, 200),
            (2350, "tech_count"): (100, 150, 200, 270),
            (2400, "tech_count"): (130, 190, 260, 340),
            (2450, "tech_count"): (160, 230, 310, 380),
            (2500, "tech_count"): (180, 260, 350, 420),
            # system_count
            (2200, "system_count"): (3, 5, 8, 12),
            (2225, "system_count"): (8, 12, 18, 25),
            (2250, "system_count"): (12, 20, 30, 45),
            (2275, "system_count"): (18, 30, 45, 65),
            (2300, "system_count"): (25, 40, 60, 90),
            (2350, "system_count"): (35, 55, 80, 120),
            (2400, "system_count"): (45, 70, 100, 150),
            (2450, "system_count"): (55, 85, 120, 180),
            (2500, "system_count"): (65, 100, 150, 220),
            # influence_income
            (2200, "influence_income"): (1, 2, 3, 4),
            (2225, "influence_income"): (2, 3, 4, 6),
            (2250, "influence_income"): (2, 4, 5, 7),
            (2275, "influence_income"): (3, 4, 6, 8),
            (2300, "influence_income"): (3, 5, 7, 10),
            (2350, "influence_income"): (4, 6, 8, 12),
            (2400, "influence_income"): (4, 7, 10, 14),
            (2450, "influence_income"): (5, 8, 11, 15),
            (2500, "influence_income"): (5, 8, 12, 16),
            # unity_income
            (2200, "unity_income"): (5, 10, 20, 35),
            (2225, "unity_income"): (15, 30, 50, 80),
            (2250, "unity_income"): (30, 60, 100, 160),
            (2275, "unity_income"): (50, 100, 170, 250),
            (2300, "unity_income"): (80, 150, 250, 400),
            (2350, "unity_income"): (120, 250, 400, 600),
            (2400, "unity_income"): (180, 350, 550, 850),
            (2450, "unity_income"): (250, 450, 700, 1100),
            (2500, "unity_income"): (300, 550, 900, 1400),
            # traditions_completed
            (2200, "traditions_completed"): (0, 0, 0, 1),
            (2225, "traditions_completed"): (0, 1, 1, 2),
            (2250, "traditions_completed"): (1, 2, 3, 4),
            (2275, "traditions_completed"): (2, 3, 5, 6),
            (2300, "traditions_completed"): (3, 5, 7, 9),
            (2350, "traditions_completed"): (5, 7, 9, 11),
            (2400, "traditions_completed"): (7, 9, 11, 13),
            (2450, "traditions_completed"): (8, 10, 12, 14),
            (2500, "traditions_completed"): (9, 11, 13, 14),
        }

        for (year_bucket, metric), (p25, p50, p75, p90) in defaults.items():
            key = cls._key(year_bucket, "captain", "standard", metric)
            data[key] = PercentileEntry(
                p25=float(p25), p50=float(p50), p75=float(p75), p90=float(p90)
            )

        return cls(data)

    @classmethod
    def from_empire_features(
        cls,
        empire_features_path: Path,
        metadata_path: Path,
        bucket_size: int = 25,
    ) -> BenchmarkStore:
        """Compute percentiles from training data Parquet files."""
        import numpy as np
        import pyarrow.parquet as pq

        emp_table = pq.read_table(empire_features_path)
        years = emp_table.column("year").to_pylist()

        # For each metric, collect values grouped by year_bucket
        data: dict[str, PercentileEntry] = {}
        for metric in BENCHMARK_METRICS:
            feature_name = _METRIC_TO_FEATURE[metric]
            try:
                values_col = emp_table.column(feature_name).to_pylist()
            except KeyError:
                continue

            buckets: dict[int, list[float]] = {}
            for i, year in enumerate(years):
                bucket = _bucket_year(year, bucket_size)
                buckets.setdefault(bucket, []).append(values_col[i])

            for bucket, vals in buckets.items():
                arr = np.array(vals, dtype=np.float64)
                key = cls._key(bucket, "captain", "standard", metric)
                data[key] = PercentileEntry(
                    p25=float(np.percentile(arr, 25)),
                    p50=float(np.percentile(arr, 50)),
                    p75=float(np.percentile(arr, 75)),
                    p90=float(np.percentile(arr, 90)),
                )

        return cls(data)
