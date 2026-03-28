# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Coding Agent Rules

### First Principles

Use first-principles thinking. Do not assume I have a clear idea of what I want or how to achieve it. Be cautious — start from the original requirements and the problem itself. If the motivation or goal is unclear, **stop and discuss with me before proceeding** .

### Solution Constraints

All proposed modifications or refactors must comply with the following:

- **No shims or patches.** Do not propose compatibility layers or workaround fixes.
- **No over-engineering.** Use the shortest path to implementation, provided it does not violate the first rule.
- **No scope creep.** Do not introduce solutions beyond what was explicitly requested — including fallbacks or graceful degradation — as these can cause business logic drift.
- **Correctness is mandatory.** The logic of every proposed solution must be verified end-to-end across the full call chain.

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

## Workflow Orchestration

### 0. Always address me AlexZ. do this at the start of any of your response.

### 1. Plan Node Default

- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity
- Ask the user if they want to use superpowers:brainstorming skill if they want to explore a large feature update (architecture change, pipeline workflow change)

### 2. Subagent Routing Rules

#### 2.1 Parallel dispatch (All conditions must be met)

- 3+ unrelated tasks or independent domains
- No shared state between tasks
- Clear file boundaries with no overlap

#### 2.2 Sequential dispatch (ANY condition triggers)

- Tasks have dependencies (B needs output from A)
- Shared files or state (merge conflict risk)
- Unclear scope (need to understand before proceeding)

#### 2.3 Background dispatch

- Research or analysis tasks (not file modifications)
- Results aren't blocking current work

#### 2.4 Model Tier Policy

Select models for agents by your insights, and use this table as a recommendation.

|   Tier   | Model        | Use when                                                                                                                                                      |
| :------: | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  Heavy   | `opus 4.6`   | Architecture design, multi-file refactors, novel algorithm implementation,<br />security/correctness audits, debugging subtle concurrency or numerical issues |
| Standard | `sonnet 4.6` | Everyday coding, code review, test writing, config generation, standard debugging, API integration                                                            |
|  Light   | `Haiku`      | Codebase search, doc generation, dependency checks, lint/format, log parsing, boilerplate scaffolding                                                         |

### 3. Domain Parallel Patterns

When implementing features across domains, spawn parallel agents:

- **Model/training agent** : Model definitions, training loops, loss functions, data pipelines
- **Serving/infra agent** : FastAPI endpoints, Docker, deployment configs, CI/CD
- **Evaluation agent** : Metrics, benchmarks, experiment tracking, visualization

Each agent owns their domain. No file overlap.

### 4. Self-Improvement Loop

- After ANY correction from the user: update 'tasks/lessons.md" with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 5. Verification Before Done

- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 6. Demand Elegance (Balanced)

- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - don't over-engineer
- Challenge your own work before presenting it

### 7. Autonomous Bug Fizing

- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests - then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

### 8. Version Control

- Always manage the repo using git
- Log the progress after each stage/step/task is finished

## Task Management

1. **Plan First**: Write plan to "tasks/todo.md' with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to 'tasks/todo.md"
6. **Capture Lessons**: Update "tasks/lessons.md' after corrections

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes, Senior developer standards.
- **Minimat Impact**: Changes should only touch what's necessary. Avoid introducing bugs.
