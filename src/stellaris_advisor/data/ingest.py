"""Batch ingestion pipeline for Stellaris save files.

Scans a folder of .sav files, parses each one, extracts empire and system
features, builds transitions (state, action, reward, next_state), and writes
four Parquet outputs to the output directory.

pyarrow is an optional [ml] dependency — imported lazily inside ingest_saves.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stellaris_advisor.data.actions import extract_actions
from stellaris_advisor.data.rewards import composite_score, compute_reward
from stellaris_advisor.features import (
    build_hyperlane_graph,
    compute_betweenness,
    extract_empire_features,
    extract_system_features,
    extract_temporal_features,
)
from stellaris_advisor.features.registry import EMPIRE_FEATURE_NAMES, TEMPORAL_FEATURE_NAMES
from stellaris_advisor.parser import parse_save
from stellaris_advisor.parser.models import EmpireState, GalacticState

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IngestResult:
    """Paths to the four Parquet outputs and pipeline diagnostics."""

    game_id: str
    saves_processed: int
    empire_features_path: Path
    system_features_path: Path
    transitions_path: Path
    metadata_path: Path
    errors: tuple[str, ...]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ingest_saves(
    save_folder: str | Path,
    output_dir: str | Path,
    game_id: str,
    stellaris_dir: str | Path | None = None,  # reserved for future game-data loading
) -> IngestResult:
    """Ingest a folder of .sav files into four Parquet datasets.

    Steps per save (sorted by in-game date):
      1. parse_save → GalacticState
      2. extract_empire_features for every empire
      3. extract_system_features for unowned systems
      4. For consecutive pairs: extract actions + compute rewards + temporal features
      5. Write empire_features, system_features, transitions, game_metadata Parquet
    """
    import pyarrow.parquet as pq  # noqa: F401 — confirms pyarrow is available
    from stellaris_advisor.data.schemas import (
        EMPIRE_FEATURES_SCHEMA,
        GAME_METADATA_SCHEMA,
        SYSTEM_FEATURES_SCHEMA,
        TRANSITIONS_SCHEMA,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    parsed, errors = _parse_saves(Path(save_folder))
    parsed.sort(key=lambda x: (x[1].year, x[1].month))

    emp_path = out / "empire_features.parquet"
    sys_path = out / "system_features.parquet"
    trans_path = out / "transitions.parquet"
    meta_path = out / "game_metadata.parquet"

    emp_rows, emp_errs = _collect_empire_rows(parsed, game_id)
    sys_rows, sys_errs = _collect_system_rows(parsed, game_id)
    trans_rows, trans_errs = _collect_transition_rows(parsed, game_id)
    errors.extend(emp_errs)
    errors.extend(sys_errs)
    errors.extend(trans_errs)

    _write_table(emp_rows, emp_path, EMPIRE_FEATURES_SCHEMA)
    _write_table(sys_rows, sys_path, SYSTEM_FEATURES_SCHEMA)
    _write_table(trans_rows, trans_path, TRANSITIONS_SCHEMA)
    _write_table(_build_metadata_rows(parsed, game_id), meta_path, GAME_METADATA_SCHEMA)

    return IngestResult(
        game_id=game_id,
        saves_processed=len(parsed),
        empire_features_path=emp_path,
        system_features_path=sys_path,
        transitions_path=trans_path,
        metadata_path=meta_path,
        errors=tuple(errors),
    )


# ---------------------------------------------------------------------------
# Internal helpers — parsing
# ---------------------------------------------------------------------------


def _parse_saves(
    folder: Path,
) -> tuple[list[tuple[Path, GalacticState]], list[str]]:
    """Try to parse every .sav in folder. Returns (parsed_list, error_list)."""
    parsed: list[tuple[Path, GalacticState]] = []
    errors: list[str] = []
    for sav in sorted(folder.glob("*.sav")):
        try:
            parsed.append((sav, parse_save(str(sav))))
        except Exception as exc:
            errors.append(f"{sav.name}: {exc}")
    return parsed, errors


# ---------------------------------------------------------------------------
# Internal helpers — feature collection
# ---------------------------------------------------------------------------


def _collect_empire_rows(
    parsed: list[tuple[Path, GalacticState]],
    game_id: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for sav, galactic in parsed:
        for empire in galactic.empires:
            try:
                feats = extract_empire_features(empire, galactic)
                row: dict[str, Any] = {"game_id": game_id, "year": galactic.year, "empire_id": empire.id}
                row.update(feats)
                rows.append(row)
            except Exception as exc:
                msg = f"{sav.name} empire {empire.id}: {exc}"
                _log.warning(msg)
                errors.append(msg)
    return rows, errors


def _collect_system_rows(
    parsed: list[tuple[Path, GalacticState]],
    game_id: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for sav, galactic in parsed:
        try:
            graph = build_hyperlane_graph(galactic)
            betweenness = compute_betweenness(graph)
            player = galactic.player_empire
            for system in galactic.systems:
                if system.owner is not None:
                    continue
                try:
                    feats = extract_system_features(system, player, galactic, graph, betweenness)
                    row: dict[str, Any] = {
                        "game_id": game_id,
                        "year": galactic.year,
                        "system_id": system.id,
                    }
                    row.update(feats)
                    rows.append(row)
                except Exception as exc:
                    msg = f"{sav.name} system {system.id}: {exc}"
                    _log.warning(msg)
                    errors.append(msg)
        except Exception as exc:
            msg = f"{sav.name} system graph: {exc}"
            _log.warning(msg)
            errors.append(msg)
    return rows, errors


def _precompute_empire_feats(
    parsed: list[tuple[Path, GalacticState]],
) -> tuple[list[dict[int, dict[str, float]]], list[str]]:
    """Pre-compute empire feature dicts for each save. Returns (snaps, errors)."""
    result: list[dict[int, dict[str, float]]] = []
    errors: list[str] = []
    for sav, galactic in parsed:
        snap: dict[int, dict[str, float]] = {}
        for empire in galactic.empires:
            try:
                snap[empire.id] = extract_empire_features(empire, galactic)
            except Exception as exc:
                msg = f"{sav.name} empire {empire.id} (transition precompute): {exc}"
                _log.warning(msg)
                errors.append(msg)
        result.append(snap)
    return result, errors


def _collect_transition_rows(
    parsed: list[tuple[Path, GalacticState]],
    game_id: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    all_feats, errors = _precompute_empire_feats(parsed)
    rows: list[dict[str, Any]] = []
    for i in range(1, len(parsed)):
        is_last = i == len(parsed) - 1
        prevprev_feats: dict[int, dict[str, float]] = all_feats[i - 2] if i >= 2 else {}
        rows.extend(
            _build_transition_rows(
                parsed[i - 1][1],
                parsed[i][1],
                all_feats[i - 1],
                all_feats[i],
                prevprev_feats,
                game_id,
                is_last,
            )
        )
    return rows, errors


def _build_transition_rows(
    prev_g: GalacticState,
    curr_g: GalacticState,
    prev_feats: dict[int, dict[str, float]],
    curr_feats: dict[int, dict[str, float]],
    prevprev_feats: dict[int, dict[str, float]],
    game_id: str,
    is_last: bool,
) -> list[dict[str, Any]]:
    """Build transition rows for one consecutive save pair. Returns a new list (no mutation)."""
    rows: list[dict[str, Any]] = []
    prev_map: dict[int, EmpireState] = {e.id: e for e in prev_g.empires}
    for curr_emp in curr_g.empires:
        prev_emp = prev_map.get(curr_emp.id)
        if prev_emp is None:
            continue
        p_feats = prev_feats.get(curr_emp.id)
        c_feats = curr_feats.get(curr_emp.id)
        if p_feats is None or c_feats is None:
            continue

        pp_feats = prevprev_feats.get(curr_emp.id)
        prev_history: tuple[dict[str, float], ...] = (pp_feats,) if pp_feats is not None else ()
        p_temp = extract_temporal_features(p_feats, prev_history)
        c_temp = extract_temporal_features(c_feats, (p_feats,))

        state_vec = [p_feats[n] for n in EMPIRE_FEATURE_NAMES] + [p_temp[n] for n in TEMPORAL_FEATURE_NAMES]
        next_state_vec = [c_feats[n] for n in EMPIRE_FEATURE_NAMES] + [c_temp[n] for n in TEMPORAL_FEATURE_NAMES]

        prev_score = composite_score(prev_emp, tuple(prev_g.empires))
        curr_score = composite_score(curr_emp, tuple(curr_g.empires))
        actions = extract_actions(curr_emp, prev_emp)
        action_seq = actions if actions else (None,)

        for action in action_seq:
            reward = compute_reward(
                curr_score,
                prev_score,
                is_terminal=is_last,
                final_rank=curr_emp.victory_rank,
                num_empires=len(curr_g.empires),
            )
            rows.append(
                {
                    "game_id": game_id,
                    "year": curr_g.year,
                    "empire_id": curr_emp.id,
                    "state_features": state_vec,
                    "action_type": action.action_type if action else "",
                    "action_params": json.dumps(dict(action.params)) if action else "{}",
                    "reward": reward,
                    "next_state_features": next_state_vec,
                    "terminal": is_last,
                    "final_rank": curr_emp.victory_rank,
                }
            )
    return rows


def _extract_game_metadata(galactic: GalacticState, game_id: str) -> dict[str, Any]:
    """Extract a metadata snapshot dict from a single GalacticState."""
    return {
        "game_id": game_id,
        "game_version": galactic.game_version,
        "galaxy_size": galactic.galaxy_size,
        "difficulty": galactic.difficulty,
        "crisis_active": galactic.crisis_active,
        "crisis_type": galactic.crisis_type or "",
        "war_in_heaven": galactic.war_in_heaven,
        "galactic_market_founded": galactic.galactic_market_founded,
        "player_empire_id": galactic.player_empire_id,
        "num_empires": len(galactic.empires),
        "year": galactic.year,
    }


def _sort_saves_by_date(
    save_folder: str | Path,
) -> list[tuple[Path, int, int]]:
    """Return (path, year, month) tuples for all parseable .sav files, sorted by date."""
    folder = Path(save_folder)
    result: list[tuple[Path, int, int]] = []
    for sav in sorted(folder.glob("*.sav")):
        try:
            state = parse_save(str(sav))
            result.append((sav, state.year, state.month))
        except Exception:
            pass
    result.sort(key=lambda t: (t[1], t[2]))
    return result


def _generate_game_id(save_folder: str | Path) -> str:
    """Generate a deterministic 12-character hex ID from the resolved save folder path."""
    import hashlib

    resolved = str(Path(save_folder).resolve())
    return hashlib.sha256(resolved.encode()).hexdigest()[:12]


def _build_metadata_rows(
    parsed: list[tuple[Path, GalacticState]],
    game_id: str,
) -> list[dict[str, Any]]:
    if not parsed:
        return []
    first = _extract_game_metadata(parsed[0][1], game_id)
    last = _extract_game_metadata(parsed[-1][1], game_id)
    return [
        {
            "game_id": first["game_id"],
            "game_version": first["game_version"],
            "galaxy_size": first["galaxy_size"],
            "difficulty": first["difficulty"],
            "crisis_active": last["crisis_active"],
            "crisis_type": last["crisis_type"],
            "war_in_heaven": last["war_in_heaven"],
            "galactic_market_founded": last["galactic_market_founded"],
            "player_empire_id": first["player_empire_id"],
            "num_empires": first["num_empires"],
            "start_year": first["year"],
            "end_year": last["year"],
            "total_saves": len(parsed),
        }
    ]


# ---------------------------------------------------------------------------
# Internal helpers — Parquet writing
# ---------------------------------------------------------------------------


def _write_table(rows: list[dict[str, Any]], path: Path, schema: Any) -> None:
    """Write rows as a Parquet file. Uses schema.empty_table() when rows is empty."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(rows, schema=schema) if rows else schema.empty_table()
    pq.write_table(table, str(path))
