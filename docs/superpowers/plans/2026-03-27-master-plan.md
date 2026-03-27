# Stellaris AI Advisor — Master Implementation Plan

> **For agentic workers:** Each sub-plan below is a self-contained implementation phase. Execute them in order. Each plan produces working, testable software.

**Goal:** Build a complete Stellaris AI Advisor — Python library + CLI that parses save files, loads the game knowledge graph, trains offline RL models, and delivers quantitative strategy recommendations.

**Design spec:** `docs/superpowers/specs/2026-03-27-stellaris-advisor-design.md`
**PRD:** `docs/stellaris-advisor-prd-v4-unified.md`

---

## Sub-Plans

### Plan 1: Project Scaffold + Clausewitz Parser
**File:** `2026-03-27-plan-01-scaffold-parser.md`
**Delivers:** pyproject.toml, config module, Clausewitz tokenizer + recursive descent parser, all parser data models (frozen dataclasses), typed extraction layer, `parse_save()` and `parse_game_files()` top-level functions, version detection.
**Depends on:** Nothing (foundation)
**Test fixtures:** Synthetic Clausewitz text + minimal real save/game data files (user provides)

### Plan 2: Game Knowledge Graph
**File:** `2026-03-27-plan-02-game-knowledge-graph.md`
**Delivers:** All gamedata models, loader for all 18+ game data file types, tech DAG construction, all Layer 1/2/3 query methods, integrity validation (11 checks), caching.
**Depends on:** Plan 1 (parser)
**Test fixtures:** Subset of real game definition files

### Plan 3: Feature Engineering
**File:** `2026-03-27-plan-03-feature-engineering.md`
**Delivers:** Empire feature extractor (~107), system feature extractor (~12), temporal feature extractor (~8), feature registry, hyperlane graph for chokepoint/distance calculations.
**Depends on:** Plan 1 (parser models), Plan 2 (game knowledge graph for system features)

### Plan 4: Data Pipeline
**File:** `2026-03-27-plan-04-data-pipeline.md`
**Delivers:** Ingestion pipeline (batch save → Parquet), action extraction from consecutive saves, reward computation, transition storage, synthetic data generation scripts, Parquet schema definitions.
**Depends on:** Plan 1 (parser), Plan 3 (features)

### Plan 5: RL + Colonization Models
**File:** `2026-03-27-plan-05-rl-models.md`
**Delivers:** State/action encoders, V(s) and Q(s,a) network architectures, d3rlpy training orchestration (CQL/IQL), evaluation metrics, Optuna tuning, colonization LightGBM scorer, benchmark percentile store.
**Depends on:** Plan 3 (features), Plan 4 (transitions data)

### Plan 6: Strategic Planner
**File:** `2026-03-27-plan-06-planner.md`
**Delivers:** 9 action generators, delta simulator, 6 domain advisors (tech, tradition, perk, megastructure, faction, council), hybrid ranker (degraded + full modes), bottleneck detector, 15 alert types, report assembly.
**Depends on:** Plan 2 (game graph queries), Plan 3 (features), Plan 5 (RL inference, optional)

### Plan 7: CLI
**File:** `2026-03-27-plan-07-cli.md`
**Delivers:** All 8 commands (analyze, parse, ingest, train, score-systems, tech-path, benchmark, compare, version), Rich terminal output, auto-detection of Stellaris paths, degraded mode indicator.
**Depends on:** All previous plans

---

## Build Order

```
Plan 1 ──→ Plan 2 ──→ Plan 3 ──→ Plan 4 ──→ Plan 5
              │           │                      │
              └───────────┴──────────→ Plan 6 ←──┘
                                         │
                                         ▼
                                      Plan 7
```

Plans 1→2→3 are strictly sequential (each depends on the previous).
Plan 4 depends on 1+3.
Plan 5 depends on 3+4.
Plan 6 depends on 2+3+5(optional).
Plan 7 depends on everything.
