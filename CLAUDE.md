# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Stellaris AI Advisor — a Python library + CLI that parses Stellaris save files, builds a game knowledge graph, trains offline RL models (d3rlpy/CQL), and delivers quantitative strategy recommendations. Licensed under GNU AGPL v3.

## Build & Development Commands

```bash
# Install (once pyproject.toml exists)
pip install -e ".[dev]"              # Core + dev tools
pip install -e ".[ml,dev]"           # Core + ML + dev tools

# Testing
pytest tests/                        # All tests
pytest tests/test_parser/ -v         # Single directory
pytest tests/test_config.py::test_name -v  # Single test
pytest -m unit                       # Fast tests only (no I/O)
pytest -m integration                # Integration tests
pytest -m "not slow"                 # Skip RL training tests
pytest --cov=src --cov-report=term-missing  # Coverage (80% min)

# Code quality
ruff check src/ tests/               # Lint
black src/ tests/                    # Format
mypy src/                            # Type check (strict mode)

# CLI (once implemented)
stellaris-advisor analyze <save_file>
stellaris-advisor parse <save_file>
```

## Architecture

**Package:** `src/stellaris_advisor/` — single package with optional `[ml]` dependencies.

**Layer stack (bottom-up, each depends on layers below):**

1. **parser/** — Custom Clausewitz tokenizer + recursive descent parser. Handles both save files (ZIP-compressed) and game definition files. Three layers: tokenize → parse to dict → extract to typed dataclasses.
2. **gamedata/** — Loads all 18+ game data file types into a `GameKnowledgeGraph`. Tech DAG via networkx. Each file (buildings.py, districts.py, etc.) handles one game data domain.
3. **features/** — Feature extraction from parsed game state. Empire (~107 features), system (~12), temporal deltas (~8). Used by both the data pipeline and the planner.
4. **data/** — Batch save ingestion to Parquet. Reward computation and action extraction from consecutive saves. Feeds the RL training pipeline.
5. **rl/** — [optional: ml] d3rlpy training orchestration (CQL/IQL), PyTorch V(s)/Q(s,a) networks, Optuna tuning, inference.
6. **colonization/** — [optional: ml] LightGBM LambdaMART scorer for system colonization ranking.
7. **planner/** — 9 action generators, delta simulator, 6 domain advisors, hybrid ranker (RL value + rule-based ROI). Works in degraded mode without trained models.
8. **cli.py** — Typer CLI with 8 commands. **config.py** — Pydantic v2 frozen config.

**Key design constraints:**

- All data structures are **immutable** (frozen dataclasses, tuples, frozensets, MappingProxyType)
- ML modules use **lazy imports** — core functionality works without PyTorch installed
- Pure functions preferred; side effects confined to CLI and I/O boundaries

## Implementation Plan

7 sequential phases documented in `docs/superpowers/plans/`:

```
Plan 1 (scaffold+parser) → Plan 2 (gamedata) → Plan 3 (features) → Plan 4 (data pipeline) → Plan 5 (RL)
                                  │                    │                                          │
                                  └────────────────────┴──────────────→ Plan 6 (planner) ←────────┘
                                                                              │
                                                                           Plan 7 (CLI)
```

**Key docs:**

- Design spec: `docs/superpowers/specs/2026-03-27-stellaris-advisor-design.md`
- PRD: `docs/stellaris-advisor-prd-v4-unified.md`
- Master plan: `docs/superpowers/plans/2026-03-27-master-plan.md`
- Phase plans: `docs/superpowers/plans/2026-03-27-plan-0N-*.md`

## Tool Configuration

- **ruff**: line-length 100, src = ["src"]
- **mypy**: python_version 3.10, strict = true
- **pytest**: markers `unit`, `integration`, `slow`; testpaths = ["tests"]
- **black**: default config

## Conventions

- Python 3.10+ with type annotations on all signatures
- Pydantic v2 for config; frozen dataclasses for domain models
- Test markers: `@pytest.mark.unit` (no I/O), `@pytest.mark.integration` (real files), `@pytest.mark.slow` (RL training)
- Hypothesis for property-based testing of parser

## Project agents

General-purpose roles, personalized for this codebase:

| Tier   | Agent              | Route when...                                                                          |
| ------ | ------------------ | -------------------------------------------------------------------------------------- |
| opus   | `architect`        | System design, cross-layer changes, pipeline architecture, 3+ file scope               |
| opus   | `debugger`         | Subtle bugs — parser edge cases, RL divergence, data corruption, graph validation      |
| opus   | `security-auditor` | File I/O, deserialization, save parsing, model loading, CLI input                      |
| sonnet | `code-reviewer`    | **Proactively after code changes** — correctness, immutability, lazy imports, coverage |
| sonnet | `implementer`      | Standard coding — new modules, features, refactoring, wiring                           |
| sonnet | `test-writer`      | Test generation, coverage gaps, fixtures, Hypothesis                                   |
| sonnet | `eval-designer`    | RL evaluation, metrics, benchmarks, cross-validation design                            |
| haiku  | `explorer`         | Fast codebase search — find definitions, trace imports, answer "where is X?"           |
| haiku  | `scaffolder`       | Boilerplate, directory structure, stub files for new modules                           |
| haiku  | `dep-checker`      | Dependency audit, version conflicts,`[ml]` optional group validation                   |
| haiku  | `log-parser`       | Training logs, ingestion output, pytest results, CI logs                               |
| haiku  | `doc-writer`       | Documentation, docstrings, ADRs, changelogs                                            |

Project-specific checklists (no model — inherit from parent):

| Checklist               | Purpose                                                                                     |
| ----------------------- | ------------------------------------------------------------------------------------------- |
| `integration-validator` | Module boundary data flow, feature registry alignment, lazy import enforcement              |
| `numerical-reviewer`    | Math correctness — reward normalization, feature scaling, CQL alpha, NDCG, division-by-zero |

### Parallel pairings

- **`implementer`** + **`test-writer`**: Implementation + test writing in parallel
- **`architect`** + **`explorer`**: Design + codebase research in parallel
- **`code-reviewer`** + **`security-auditor`**: Quality review + security audit in parallel
- **`implementer`** + **`scaffolder`**: Feature work + boilerplate for new module in parallel

Each agent owns their domain. No file overlap.

## Codex Cooperation

Use model `gpt-6.1-sol` when dispatching jobs to codex via the project `codex-cli` MCP (`mcp__codex-cli__*`).
