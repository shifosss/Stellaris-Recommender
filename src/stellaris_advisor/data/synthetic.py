"""Synthetic data generation for Stellaris RL training runs.

Provides configuration dataclasses and generator functions for producing
randomized Stellaris game configurations, launch scripts, and orchestration
configs for batch data collection.
"""
from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Valid difficulty levels
# ---------------------------------------------------------------------------

_DIFFICULTIES: tuple[str, ...] = (
    "civilian",
    "ensign",
    "captain",
    "commodore",
    "admiral",
    "grand_admiral",
)

_GALAXY_SIZES: tuple[int, ...] = (400, 500, 600, 700, 800, 900, 1000)

_HYPERLANE_DENSITIES: tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticConfig:
    """Configuration for a single synthetic Stellaris game run."""

    galaxy_size: int = 600
    num_empires: int = 15
    difficulty: str = "captain"
    crisis_strength: float = 1.0
    hyperlane_density: float = 1.0
    mid_game_start: int = 2275
    end_game_start: int = 2375
    autosave_interval: int = 12  # months


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_random_config() -> SyntheticConfig:
    """Return a SyntheticConfig with randomized galaxy parameters."""
    return SyntheticConfig(
        galaxy_size=random.choice(_GALAXY_SIZES),
        num_empires=random.randint(8, 30),
        difficulty=random.choice(_DIFFICULTIES),
        crisis_strength=round(random.uniform(1.0, 5.0), 2),
        hyperlane_density=random.choice(_HYPERLANE_DENSITIES),
        mid_game_start=random.randint(2250, 2300),
        end_game_start=random.randint(2350, 2400),
        autosave_interval=12,
    )


def generate_launch_script(config: SyntheticConfig) -> str:
    """Return a bash script that launches Stellaris in observer mode with the given config."""
    return (
        "#!/bin/bash\n"
        "# Auto-generated Stellaris observer run script\n"
        "# Do not edit — regenerate via generate_launch_script()\n"
        "\n"
        "stellaris \\\n"
        "    --observer \\\n"
        f"    --difficulty {config.difficulty} \\\n"
        f"    --galaxy-size {config.galaxy_size} \\\n"
        f"    --num-empires {config.num_empires} \\\n"
        f"    --crisis-strength {config.crisis_strength} \\\n"
        f"    --hyperlane-density {config.hyperlane_density} \\\n"
        f"    --autosave-interval {config.autosave_interval} \\\n"
        f"    --mid-game-start {config.mid_game_start} \\\n"
        f"    --end-game-start {config.end_game_start}\n"
    )


def generate_orchestrator_config(
    num_runs: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Return an orchestration config dict for running multiple synthetic games.

    Each run gets its own randomly generated SyntheticConfig (serialised as a
    plain dict so the result is JSON-serialisable).
    """
    configs = [dataclasses.asdict(generate_random_config()) for _ in range(num_runs)]
    return {
        "num_runs": num_runs,
        "output_dir": str(output_dir),
        "configs": configs,
    }
