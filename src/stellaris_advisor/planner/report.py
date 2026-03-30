# src/stellaris_advisor/planner/report.py
"""Report assembly — the top-level analyze() function."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from stellaris_advisor.colonization.benchmark import BenchmarkStore
from stellaris_advisor.config import PlannerConfig
from stellaris_advisor.features.empire import extract_empire_features
from stellaris_advisor.parser.models import EmpireState, GalacticState
from stellaris_advisor.planner.action_generator import generate_all_actions
from stellaris_advisor.planner.alerts import generate_alerts
from stellaris_advisor.planner.bottleneck import compute_empire_health
from stellaris_advisor.planner.models import Report
from stellaris_advisor.planner.ranker import rank_recommendations


def analyze(
    empire: EmpireState,
    galactic: GalacticState,
    config: PlannerConfig,
    game_graph: Any = None,
    rl_model_dir: Path | None = None,
    colonization_model_path: Path | None = None,
    benchmark_path: Path | None = None,
) -> Report:
    """Top-level planner entry point.

    Orchestrates: features → bottleneck → generate → score → rank → report.

    Args:
        empire: Player empire state.
        galactic: Full galactic state.
        config: Planner tuning knobs.
        game_graph: GameKnowledgeGraph (optional, enhances action generation).
        rl_model_dir: Path to trained RL model (optional, enables full mode).
        colonization_model_path: Path to colonization scorer (optional).
        benchmark_path: Path to benchmark JSON (optional, falls back to hardcoded).

    Returns:
        Frozen Report with recommendations, health, alerts.
    """
    # Step 1: Extract features
    empire_features = extract_empire_features(empire, galactic)

    # Step 2: Load benchmark (hardcoded defaults if no path)
    if benchmark_path and benchmark_path.exists():
        benchmark = BenchmarkStore.load(benchmark_path)
    else:
        benchmark = BenchmarkStore.hardcoded_defaults()

    # Step 3: Compute empire health + bottleneck
    empire_type = "standard"
    if empire.authority == "auth_hive_mind":
        empire_type = "hive"
    elif empire.authority == "auth_machine_intelligence":
        empire_type = "machine"

    # Hardcoded defaults are captain-only; use "captain" when no benchmark file
    difficulty = galactic.difficulty if benchmark_path else "captain"

    health = compute_empire_health(
        empire_features,
        galactic.year,
        difficulty,
        empire_type,
        benchmark,
    )

    # Step 4: Load optional models
    rl_predictor = None
    if rl_model_dir is not None:
        from stellaris_advisor.rl.inference import RLPredictor

        if RLPredictor.is_available(rl_model_dir):
            rl_predictor = RLPredictor(rl_model_dir)

    colonization_scorer = None
    if colonization_model_path is not None:
        from stellaris_advisor.colonization.scorer import ColonizationScorer

        if ColonizationScorer.is_available(colonization_model_path):
            colonization_scorer = ColonizationScorer.load(colonization_model_path)

    # RL scoring not yet wired into ranking pipeline — stay degraded
    mode = "degraded"

    # Step 5: Generate candidate actions
    candidates = generate_all_actions(
        empire,
        galactic,
        config,
        game_graph,
        colonization_scorer,
    )

    # Step 6: Rank candidates
    ranked = rank_recommendations(candidates, config, rl_predictor, empire_features)

    # Step 7: Generate alerts
    alerts = generate_alerts(empire, galactic, config)

    # Step 8: Assemble report
    return Report(
        empire_name=empire.name,
        year=galactic.year,
        mode=mode,
        empire_health=health,
        recommendations=ranked,
        tech_path_insight=None,
        tradition_perk_path=None,
        faction_influence_summary=None,
        alerts=alerts,
    )
