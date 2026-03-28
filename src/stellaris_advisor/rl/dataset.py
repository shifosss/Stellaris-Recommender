"""Bridge between Parquet transitions and d3rlpy MDPDataset."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stellaris_advisor.rl.action_encoder import ActionEncoder


@dataclass(frozen=True)
class DataSplit:
    """Game-level train/val/test split."""

    train_game_ids: tuple[str, ...]
    val_game_ids: tuple[str, ...]
    test_game_ids: tuple[str, ...]


@dataclass(frozen=True)
class RLArrays:
    """Numpy arrays for RL training. All float32, not writable."""

    observations: Any  # numpy (N, state_dim)
    actions: Any  # numpy (N, action_dim)
    rewards: Any  # numpy (N,)
    terminals: Any  # numpy (N,)


def split_by_game(
    game_ids: list[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    holdout_patterns: tuple[str, ...] = (),
    seed: int = 42,
) -> DataSplit:
    """Split game IDs into train/val/test sets.

    Holdout games matching any pattern go to test unconditionally.
    Remaining games are shuffled deterministically and split by ratio.
    Guarantees at least 1 game per split (when possible).
    """
    holdout: list[str] = []
    remainder: list[str] = []
    for gid in game_ids:
        if any(pat in gid for pat in holdout_patterns):
            holdout.append(gid)
        else:
            remainder.append(gid)

    rng = random.Random(seed)
    rng.shuffle(remainder)

    n = len(remainder)
    n_train = max(1, round(n * train_ratio))
    n_val = max(1, round(n * val_ratio))

    # Ensure we don't exceed total
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = max(1, n - n_train - 1)

    train_ids = remainder[:n_train]
    val_ids = remainder[n_train : n_train + n_val]
    test_ids = remainder[n_train + n_val :] + holdout

    # Guarantee at least 1 per split
    if not test_ids and val_ids:
        test_ids = [val_ids.pop()]
    if not val_ids and len(train_ids) > 1:
        val_ids = [train_ids.pop()]

    return DataSplit(
        train_game_ids=tuple(sorted(train_ids)),
        val_game_ids=tuple(sorted(val_ids)),
        test_game_ids=tuple(sorted(test_ids)),
    )


def load_transitions(parquet_path: Path, game_ids: list[str]) -> RLArrays:
    """Load transitions from Parquet, filter by game_ids, encode actions."""
    import numpy as np
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    df_game_ids = table.column("game_id").to_pylist()

    # Filter rows by game_id
    mask = [gid in set(game_ids) for gid in df_game_ids]
    indices = [i for i, m in enumerate(mask) if m]

    states = [table.column("state_features")[i].as_py() for i in indices]
    action_types = [table.column("action_type")[i].as_py() for i in indices]
    action_params = [table.column("action_params")[i].as_py() for i in indices]
    rewards_list = [table.column("reward")[i].as_py() for i in indices]
    terminals_list = [table.column("terminal")[i].as_py() for i in indices]

    encoder = ActionEncoder()
    action_vecs = encoder.encode_batch(action_types, action_params)

    obs = np.array(states, dtype=np.float32)
    obs.flags.writeable = False
    acts = np.array(action_vecs, dtype=np.float32)
    acts.flags.writeable = False
    rews = np.array(rewards_list, dtype=np.float32)
    rews.flags.writeable = False
    terms = np.array([1.0 if t else 0.0 for t in terminals_list], dtype=np.float32)
    terms.flags.writeable = False

    return RLArrays(observations=obs, actions=acts, rewards=rews, terminals=terms)


def build_mdp_dataset(arrays: RLArrays) -> Any:
    """Wrap RLArrays in a d3rlpy MDPDataset."""
    import d3rlpy

    return d3rlpy.dataset.MDPDataset(
        observations=arrays.observations.copy(),
        actions=arrays.actions.copy(),
        rewards=arrays.rewards.copy(),
        terminals=arrays.terminals.copy(),
    )


def load_and_split(
    parquet_path: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    holdout_patterns: tuple[str, ...] = (),
    seed: int = 42,
) -> tuple[DataSplit, RLArrays, RLArrays, RLArrays]:
    """Full pipeline: load Parquet, split by game, return split + 3 array sets."""
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    game_ids = sorted(set(table.column("game_id").to_pylist()))

    split = split_by_game(
        game_ids, train_ratio, val_ratio, test_ratio, holdout_patterns, seed
    )
    train = load_transitions(parquet_path, list(split.train_game_ids))
    val = load_transitions(parquet_path, list(split.val_game_ids))
    test = load_transitions(parquet_path, list(split.test_game_ids))

    return split, train, val, test
