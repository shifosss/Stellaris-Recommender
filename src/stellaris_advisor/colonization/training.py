"""Train LightGBM LambdaMART colonization scorer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from stellaris_advisor.colonization.scorer import ColonizationScorer
from stellaris_advisor.features.registry import SYSTEM_FEATURE_NAMES


@dataclass(frozen=True)
class ColonizationTrainResult:
    """Immutable colonization training outcome."""

    model_path: Path
    ndcg_at_10: float
    baseline_ndcg_at_10: float
    improvement_pct: float
    n_samples: int


def create_colonization_labels(
    system_features_path: Path,
    transitions_path: Path,
) -> tuple[Any, Any, list[int]]:
    """Create (X, y, groups) for LambdaMART training.

    Labels: 1 if the system was colonised in the next period, 0 otherwise.
    Groups: (game_id, year) pairs define query groups.
    """
    import pyarrow.parquet as pq

    # Load system features
    sys_table = pq.read_table(system_features_path)
    sys_game_ids = sys_table.column("game_id").to_pylist()
    sys_years = sys_table.column("year").to_pylist()
    sys_ids = sys_table.column("system_id").to_pylist()

    feature_cols = [sys_table.column(name).to_pylist() for name in SYSTEM_FEATURE_NAMES]
    X_rows: list[list[float]] = []
    for i in range(len(sys_game_ids)):
        X_rows.append([col[i] for col in feature_cols])

    # Load transitions to find colonised systems
    trans_table = pq.read_table(transitions_path)
    colonised: set[tuple[str, str]] = set()
    for i in range(len(trans_table)):
        if trans_table.column("action_type")[i].as_py() == "system_claimed":
            params = json.loads(trans_table.column("action_params")[i].as_py())
            game_id = trans_table.column("game_id")[i].as_py()
            sys_id = params.get("system_id", "")
            colonised.add((game_id, str(sys_id)))

    # Build labels
    y_list: list[int] = []
    for i in range(len(sys_game_ids)):
        key = (sys_game_ids[i], str(sys_ids[i]))
        y_list.append(1 if key in colonised else 0)

    # Build groups: contiguous runs of same (game_id, year)
    groups: list[int] = []
    prev_key = None
    for i in range(len(sys_game_ids)):
        key = (sys_game_ids[i], sys_years[i])
        if key != prev_key:
            groups.append(1)
            prev_key = key
        else:
            groups[-1] += 1

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y, groups


def _habitability_baseline_ndcg(
    X: Any, y: Any, groups: list[int], k: int = 10
) -> float:
    """NDCG@k using only the first feature (system_best_habitability) as score."""
    from sklearn.metrics import ndcg_score

    scores: list[float] = []
    offset = 0
    for g in groups:
        if g < 2:
            offset += g
            continue
        y_true = y[offset : offset + g].reshape(1, -1).astype(float)
        y_score = X[offset : offset + g, 0].reshape(1, -1)
        ndcg = ndcg_score(y_true, y_score, k=min(k, g))
        scores.append(ndcg)
        offset += g
    return float(np.mean(scores)) if scores else 0.0


def train_colonization(
    system_features_path: Path,
    transitions_path: Path,
    output_path: Path,
    n_estimators: int = 100,
    num_leaves: int = 31,
    learning_rate: float = 0.05,
) -> ColonizationTrainResult:
    """Train LightGBM LambdaMART colonization ranker."""
    import lightgbm as lgb

    X, y, groups = create_colonization_labels(system_features_path, transitions_path)

    # Ensure positive labels exist; if not, return degenerate result
    if y.sum() == 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ranker = lgb.LGBMRanker(n_estimators=1, num_leaves=2)
        # Train with synthetic positive to avoid crash
        y_synth = y.copy()
        y_synth[0] = 1
        ranker.fit(X, y_synth, group=groups)
        scorer = ColonizationScorer(ranker)
        scorer.save(output_path)
        return ColonizationTrainResult(
            model_path=output_path,
            ndcg_at_10=0.0,
            baseline_ndcg_at_10=0.0,
            improvement_pct=0.0,
            n_samples=len(X),
        )

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        verbose=-1,
    )
    ranker.fit(X, y, group=groups)

    # Evaluate
    from sklearn.metrics import ndcg_score

    pred_scores = ranker.predict(X)
    ndcg_scores: list[float] = []
    offset = 0
    for g in groups:
        if g < 2:
            offset += g
            continue
        y_true = y[offset : offset + g].reshape(1, -1).astype(float)
        y_pred = pred_scores[offset : offset + g].reshape(1, -1)
        ndcg = ndcg_score(y_true, y_pred, k=min(10, g))
        ndcg_scores.append(ndcg)
        offset += g
    model_ndcg = float(np.mean(ndcg_scores)) if ndcg_scores else 0.0

    baseline_ndcg = _habitability_baseline_ndcg(X, y, groups, k=10)
    improvement = ((model_ndcg - baseline_ndcg) / max(baseline_ndcg, 1e-8)) * 100

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scorer = ColonizationScorer(ranker)
    scorer.save(output_path)

    return ColonizationTrainResult(
        model_path=output_path,
        ndcg_at_10=model_ndcg,
        baseline_ndcg_at_10=baseline_ndcg,
        improvement_pct=improvement,
        n_samples=len(X),
    )
