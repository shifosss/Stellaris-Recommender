"""d3rlpy CQL/IQL training orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stellaris_advisor.config import RLConfig
from stellaris_advisor.rl.dataset import RLArrays, build_mdp_dataset
from stellaris_advisor.rl.state_encoder import StateEncoder


@dataclass(frozen=True)
class TrainResult:
    """Immutable training outcome."""

    model_path: Path
    encoder_path: Path
    algorithm: str
    n_steps: int
    training_time_seconds: float


def create_algorithm(
    config: RLConfig,
    state_dim: int,
    action_dim: int,
    device: str = "cpu:0",
) -> Any:
    """Create a d3rlpy algorithm from config."""
    import d3rlpy

    encoder_factory = d3rlpy.models.VectorEncoderFactory(
        hidden_units=[256, 128, 64],
        use_batch_norm=True,
    )

    if config.algorithm == "cql":
        return d3rlpy.algos.CQLConfig(
            batch_size=config.batch_size,
            gamma=config.gamma,
            actor_learning_rate=config.learning_rate,
            critic_learning_rate=config.learning_rate,
            conservative_weight=config.cql_alpha,
            actor_encoder_factory=encoder_factory,
            critic_encoder_factory=encoder_factory,
        ).create(device=device)
    elif config.algorithm == "iql":
        return d3rlpy.algos.IQLConfig(
            batch_size=config.batch_size,
            gamma=config.gamma,
            actor_learning_rate=config.learning_rate,
            critic_learning_rate=config.learning_rate,
            actor_encoder_factory=encoder_factory,
            critic_encoder_factory=encoder_factory,
        ).create(device=device)
    else:
        msg = f"Unsupported algorithm: {config.algorithm!r}. Use 'cql' or 'iql'."
        raise ValueError(msg)


def train(
    config: RLConfig,
    train_arrays: RLArrays,
    val_arrays: RLArrays,
    output_dir: Path,
    device: str = "cpu:0",
    n_steps: int | None = None,
) -> TrainResult:
    """Train an offline RL model and save artifacts.

    Args:
        config: RL hyperparameters.
        train_arrays: Training data (from load_transitions).
        val_arrays: Validation data.
        output_dir: Directory for model + encoder artifacts.
        device: Torch device string.
        n_steps: Override total training steps (default: n_epochs * len(train) / batch_size).

    Returns:
        Frozen TrainResult with paths to saved artifacts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fit state encoder on training data
    state_dim = train_arrays.observations.shape[1]
    action_dim = train_arrays.actions.shape[1]
    encoder = StateEncoder(state_dim=state_dim)
    encoder.fit(train_arrays.observations.tolist())
    encoder_path = output_dir / "encoder_stats.json"
    encoder.save(encoder_path)

    # Build d3rlpy datasets
    train_dataset = build_mdp_dataset(train_arrays)

    # Create algorithm
    algo = create_algorithm(config, state_dim, action_dim, device)

    # Compute steps
    if n_steps is None:
        n_steps = max(
            1, config.n_epochs * len(train_arrays.observations) // config.batch_size
        )

    # Train
    start = time.monotonic()
    algo.fit(
        train_dataset,
        n_steps=n_steps,
        n_steps_per_epoch=max(1, n_steps // max(config.n_epochs, 1)),
        show_progress=False,
    )
    elapsed = time.monotonic() - start

    # Save
    model_path = output_dir / "model.d3"
    algo.save(str(model_path))

    return TrainResult(
        model_path=model_path,
        encoder_path=encoder_path,
        algorithm=config.algorithm,
        n_steps=n_steps,
        training_time_seconds=elapsed,
    )
