# Stellaris AI Advisor — Design Specification

**Date:** 2026-03-27
**Status:** Approved
**PRD reference:** `docs/stellaris-advisor-prd-v4-unified.md`

---

## 1. Architecture Overview

Single Python package with optional dependencies:

```
pip install stellaris-advisor          # core: parser + gamedata + planner (rule-based)
pip install stellaris-advisor[ml]      # adds: PyTorch, d3rlpy, PyArrow, SHAP, LightGBM
pip install stellaris-advisor[all]     # everything
pip install stellaris-advisor[dev]     # + pytest, hypothesis, black, ruff, mypy
```

**Design principles:**
- All data structures are immutable (frozen dataclasses, tuples, frozensets, MappingProxyType)
- ML modules use lazy imports — degraded mode works without PyTorch
- Pure functions preferred; side effects confined to CLI and I/O boundaries
- Foundation-up build order: parser → game data → features → data pipeline → RL → planner → CLI

---

## 2. Package Structure

```
stellaris-advisor/
├── src/
│   └── stellaris_advisor/
│       ├── __init__.py
│       ├── parser/
│       │   ├── __init__.py
│       │   ├── clausewitz.py          # Tokenizer + recursive descent parser
│       │   ├── extractor.py           # Raw dict → typed dataclasses
│       │   ├── version.py             # Version detection + schema routing
│       │   └── models.py              # EmpireState, PlanetState, SystemState, etc.
│       ├── gamedata/
│       │   ├── __init__.py
│       │   ├── loader.py              # Master loader → GameKnowledgeGraph
│       │   ├── tech_graph.py          # Tech DAG + BFS/DFS queries
│       │   ├── buildings.py           # Building defs + constraint resolution
│       │   ├── districts.py           # District defs + planet class rules
│       │   ├── jobs.py                # Job defs + output estimation
│       │   ├── ships.py               # Ship class + component defs
│       │   ├── planet_classes.py      # Planet class defs + habitability
│       │   ├── empires.py             # Civic, authority, trait defs
│       │   ├── traditions.py          # Tradition trees + individual traditions
│       │   ├── ascension_perks.py     # Perk defs, tiers, mutual exclusions
│       │   ├── megastructures.py      # Megastructure defs, site reqs, stages
│       │   ├── policies.py            # Policy defs, faction issue mappings
│       │   ├── factions.py            # Faction defs, issue conditions, approval
│       │   ├── council.py             # Council positions, agendas
│       │   ├── federations.py         # Federation types, per-level bonuses
│       │   ├── validation.py          # DAG integrity + cross-reference checks
│       │   └── models.py              # TechNode, BuildingDef, TraditionTree, etc.
│       ├── features/
│       │   ├── __init__.py
│       │   ├── empire.py              # ~107 empire features
│       │   ├── system.py              # ~12 system features
│       │   └── temporal.py            # ~8 temporal delta features
│       ├── data/
│       │   ├── __init__.py
│       │   ├── ingest.py              # Batch save → Parquet pipeline
│       │   ├── synthetic.py           # Observer run automation
│       │   ├── labels.py              # Reward computation + action extraction
│       │   └── schema.py              # Parquet schema defs
│       ├── rl/                        # [optional: ml]
│       │   ├── __init__.py
│       │   ├── networks.py            # PyTorch V(s) and Q(s,a) architectures
│       │   ├── state_encoder.py       # Feature vector → normalized tensor
│       │   ├── action_encoder.py      # MacroAction → tensor encoding
│       │   ├── trainer.py             # d3rlpy training orchestration (CQL, IQL)
│       │   ├── inference.py           # Load trained model, predict value/Q
│       │   ├── evaluation.py          # Metrics, cross-validation
│       │   └── tuning.py              # Optuna hyperparameter search
│       ├── colonization/              # [optional: ml]
│       │   ├── __init__.py
│       │   ├── scorer.py              # LightGBM LambdaMART ranking model
│       │   ├── training.py            # Train + evaluate colonization scorer
│       │   └── benchmark.py           # Percentile data store
│       ├── planner/
│       │   ├── __init__.py
│       │   ├── action_generator.py    # Enumerate legal macro-actions
│       │   ├── delta_simulator.py     # Simulate feature deltas per action
│       │   ├── tech_optimizer.py      # Tech path BFS + ROI scoring
│       │   ├── tradition_advisor.py   # Tradition ordering + perk slot timing
│       │   ├── perk_advisor.py        # Perk selection + path recommendation
│       │   ├── megastructure_advisor.py # Readiness, site selection, build order
│       │   ├── faction_advisor.py     # Approval analysis, policy change impact
│       │   ├── council_advisor.py     # Leader matching, agenda optimization
│       │   ├── ranker.py              # Hybrid scoring (RL value + rule ROI)
│       │   ├── bottleneck.py          # 8-dimension bottleneck detection
│       │   ├── alerts.py              # 15 alert types
│       │   └── report.py              # Report assembly + formatting
│       ├── cli.py                     # Typer CLI entry point
│       └── config.py                  # Pydantic v2 config schema
├── scripts/
│   ├── generate_synthetic.sh          # Batch observer run launcher
│   └── generate_synthetic.py          # Python orchestrator
├── tests/
│   ├── conftest.py
│   ├── fixtures/
│   │   ├── sample_save_small.sav
│   │   ├── sample_save_midgame.sav
│   │   ├── sample_save_lategame.sav
│   │   ├── sample_gamestate.txt
│   │   ├── sample_meta.txt
│   │   ├── game_data/
│   │   └── expected/
│   ├── test_parser/
│   ├── test_gamedata/
│   ├── test_features/
│   ├── test_data/
│   ├── test_rl/
│   ├── test_colonization/
│   ├── test_planner/
│   └── test_cli/
├── data/                              # .gitignore'd
├── models/                            # Trained artifacts
├── pyproject.toml
├── config.yaml
└── README.md
```

---

## 3. Dependencies

### Core (always installed)

| Package | Version | Purpose |
|---|---|---|
| networkx | >=3.0 | Tech DAG, hyperlane graph, betweenness centrality |
| typer | >=0.9 | CLI framework |
| rich | >=13.0 | Terminal output rendering |
| pydantic | >=2.0 | Config validation |
| pyyaml | >=6.0 | Config file loading |

### Optional: ml

| Package | Version | Purpose |
|---|---|---|
| torch | >=2.0 | RL network architectures |
| d3rlpy | >=2.0 | Offline RL algorithms (CQL, IQL) |
| lightgbm | >=4.0 | Colonization scorer (LambdaMART) |
| pyarrow | >=14.0 | Parquet I/O |
| optuna | >=3.0 | Hyperparameter tuning |
| shap | >=0.42 | Explainability |
| scikit-learn | >=1.3 | Preprocessing, metrics |

### Dev

| Package | Version | Purpose |
|---|---|---|
| pytest | >=7.0 | Testing |
| pytest-cov | >=4.0 | Coverage |
| hypothesis | >=6.0 | Property-based testing |
| black | >=23.0 | Formatting |
| ruff | >=0.1 | Linting |
| mypy | >=1.0 | Type checking |

**Python version:** >=3.10

---

## 4. Clausewitz Parser

### 4.1 Architecture

Three layers:

**Layer 1 — Tokenizer** (`clausewitz.py`):
- Streaming buffered reader for 100+ MB files
- Token types: `IDENTIFIER`, `STRING`, `NUMBER`, `EQUALS`, `LBRACE`, `RBRACE`, `BOOL`, `COMMENT`
- Handles: quoted strings with escapes, unquoted identifiers, integers, floats, `yes`/`no` booleans, `#` comments

**Layer 2 — Recursive descent parser** (`clausewitz.py`):
- Consumes token stream → nested Python dict
- Rules:
  - `key = value` → `dict[key] = value`
  - `key = { ... }` → `dict[key] = parse_block()`
  - Duplicate keys at same level → collected into a list
  - Integer keys → stored as string keys
  - Bare value lists `{ "a" "b" "c" }` → detected by absence of `=`, returned as `list`
- Output: `dict[str, Any]`

**Layer 3 — Typed extraction** (`extractor.py`):
- Maps raw dict paths to frozen dataclasses via version-aware schemas
- Path navigation helper: `get_path(data, "country.0.budget.current_month.income.energy")` with safe defaults
- Pure functions: `extract_empire(raw_dict, empire_id, version) -> EmpireState`

### 4.2 Save File Handling

```python
def parse_save(path: str) -> GalacticState:
    # 1. Open ZIP archive
    # 2. Read 'meta' → detect version, extract metadata
    # 3. Read 'gamestate' → tokenize → parse → raw dict
    # 4. Extract GalacticState with all empires, systems, planets
```

### 4.3 Game Data File Handling

```python
def parse_game_files(stellaris_dir: str) -> dict[str, dict]:
    # 1. Glob common/technology/*.txt, common/buildings/*.txt, etc.
    # 2. Parse each file → raw dict
    # 3. Return merged dict keyed by category
```

### 4.4 Performance

- Selective parsing: `--empire-only` skips non-player blocks by jumping over nested `{ }` without full parse
- Cache parsed results to disk. Cache key = file hash + parser version.

### 4.5 Edge Cases

- Ironman saves (same format, specific ZIP settings)
- UTF-8 with BOM
- Empty blocks `key = { }`
- Deep nesting (>20 levels)
- Very long bare value lists (hundreds of entries)
- Duplicate keys at same level

---

## 5. Data Models

All dataclasses are frozen (immutable). Two model files with independent concerns.

### 5.1 Parser Models (`parser/models.py`)

Represents runtime state extracted from save files.

**Immutability conventions:**
- `tuple[T, ...]` instead of `list[T]` for all sequences
- `frozenset[str]` for tech sets (fast membership checks)
- `MappingProxyType[str, V]` for dicts (immutable view wrapping dict at construction)
- All fields are primitives, tuples, frozensets, or other frozen dataclasses

**Key types:**

- `GalacticState` — top-level: year, version, galaxy config, all empires + systems, player empire
- `EmpireState` — full empire snapshot: identity, resources (19 types), tech, military, territory, pops, diplomacy, traditions/perks, leaders/council, factions, policies, galcom, federation, megastructures, warfare, score
- `PlanetState` — districts, buildings, specializations, pops, stability, amenities, housing, features
- `SystemState` — star class, hyperlanes, owner, claims, starbase, planets, resources
- `StarbaseState` — level, modules, buildings, defense fleet power
- `FleetState` — power, ship composition, naval cap, location, commander
- `LeaderState` — class, level, traits, expertise, assignment
- `CouncilPosition` — position type, assigned leader
- `AgendaState` — type, progress, launch status
- `FactionState` — ethic, approval, size, support, issues
- `FederationState` — type, level, cohesion, experience, fleet power
- `MegastructureState` — type, stage, system
- `WarState`, `ClaimState`, `ColonyCandidate`

### 5.2 Game Data Models (`gamedata/models.py`)

Represents static definitions from game files. All frozen.

**Key types:**

- `TechNode` — id, area, category, tier, cost, weight, prerequisites, unlocks (buildings/districts/components/ships/edicts), weight modifiers, empire type restrictions
- `WeightModifier` — condition type/value, weight multiplier
- `BuildingDef` — prerequisites (tech/building/planet/empire), economics (cost/production/upkeep/jobs), uniqueness, upgrade chain
- `DistrictDef` — planet class availability, output, housing, jobs, specialization slots
- `SpecializationDef` — tech prereqs, extra jobs, extra building slots
- `JobDef` — category, resource production/upkeep, output modifiers
- `ShipClassDef` — hull, naval cap cost, tech prereqs, sections, build cost
- `PlanetClassDef` — climate, habitability base, available districts
- `TraditionTree` — adoption bonus, finisher bonus, tradition list, empire type restrictions
- `TraditionDef` — tree, effects, prerequisites within tree, cost modifiers
- `AscensionPerkDef` — tier, prerequisites (perks/techs/traditions), effects, mutual exclusions, is_ascension_path flag
- `MegastructureDef` — stages (cost/time/output each), site requirements, perk/tech prerequisites
- `MegastructureStageDef` — alloy cost, build days, output on completion
- `PolicyDef` — categories, options, effects per option, faction approval impacts
- `FactionDef` — ethic, issue list (conditions + approval deltas), influence formula, join conditions
- `FactionIssueDef` — binary condition, approval delta when fulfilled/violated
- `CouncilPositionDef` — required leader class, per-level bonus, empty penalty
- `AgendaDef` — effect, launch effect, preparation cost, prerequisites
- `FederationTypeDef` — per-level bonuses, law options, cohesion mechanics
- `CivicDef`, `TraitDef`, `EdictDef`

---

## 6. Game Knowledge Graph

### 6.1 Structure

```python
class GameKnowledgeGraph:
    # Layer 1: Core
    tech_dag: nx.DiGraph                              # nodes=tech IDs, edges=prereq→unlock
    tech_defs: MappingProxyType[str, TechNode]
    building_defs: MappingProxyType[str, BuildingDef]
    district_defs: MappingProxyType[str, DistrictDef]
    job_defs: MappingProxyType[str, JobDef]
    ship_class_defs: MappingProxyType[str, ShipClassDef]
    planet_class_defs: MappingProxyType[str, PlanetClassDef]

    # Layer 2: Meta-systems
    tradition_trees: MappingProxyType[str, TraditionTree]
    tradition_defs: MappingProxyType[str, TraditionDef]
    ascension_perk_defs: MappingProxyType[str, AscensionPerkDef]
    megastructure_defs: MappingProxyType[str, MegastructureDef]

    # Layer 3: Political
    policy_defs: MappingProxyType[str, PolicyDef]
    faction_defs: MappingProxyType[str, FactionDef]
    council_position_defs: MappingProxyType[str, CouncilPositionDef]
    agenda_defs: MappingProxyType[str, AgendaDef]
    federation_type_defs: MappingProxyType[str, FederationTypeDef]

    # Supporting
    civic_defs: MappingProxyType[str, CivicDef]
    trait_defs: MappingProxyType[str, TraitDef]
    edict_defs: MappingProxyType[str, EdictDef]
```

Loaded via `GameKnowledgeGraph.load(stellaris_dir, version)`. Immutable after construction. Cached to disk as pickle (keyed by version + directory hash).

### 6.2 Loading Pipeline

```
GameKnowledgeGraph.load(stellaris_dir)
    ├── parse common/technology/*.txt → tech_dag + tech_defs
    ├── parse common/buildings/*.txt → building_defs
    ├── parse common/districts/*.txt → district_defs
    ├── parse common/pop_jobs/*.txt → job_defs
    ├── parse common/ship_sizes/*.txt + component_templates/*.txt → ship_class_defs
    ├── parse common/planet_classes/*.txt → planet_class_defs
    ├── parse common/tradition_categories/*.txt + traditions/*.txt → tradition_trees + defs
    ├── parse common/ascension_perks/*.txt → ascension_perk_defs
    ├── parse common/megastructures/*.txt → megastructure_defs
    ├── parse common/policies/*.txt → policy_defs
    ├── parse common/factions/*.txt → faction_defs
    ├── parse common/council_positions/*.txt + agendas/*.txt → council + agenda defs
    ├── parse common/federation_types/*.txt → federation_type_defs
    ├── parse common/governments/civics/*.txt + authorities/*.txt → civic + authority defs
    ├── parse common/traits/*.txt → trait_defs
    ├── parse common/edicts/*.txt → edict_defs
    ├── cross-link: tech → buildings/districts/components they unlock
    └── validate integrity
```

### 6.3 Integrity Validation

Run on every load. Failures raise descriptive errors with offending IDs.

1. **Tech DAG acyclicity** — topological sort
2. **Referential integrity** — every prerequisite resolves to existing tech ID
3. **Tier consistency** — no tech at tier N has prereq at tier > N
4. **Building→tech linkage** — building tech_prerequisites reference valid IDs
5. **Planet class coverage** — district planet_class_availability references valid IDs
6. **Tradition→tree linkage** — every tradition belongs to a valid tree
7. **Perk tier prerequisites** — tier requirements are consistent
8. **Perk mutual exclusions** — symmetric and reference valid IDs
9. **Megastructure→tech/perk linkage** — prerequisites reference valid IDs
10. **Policy→faction issue linkage** — issue conditions reference valid policy categories
11. **Council position→leader class** — requirements reference valid leader classes

### 6.4 Query Methods

**Layer 1 — Core:**

- `reachable_techs(researched, empire_type, ethics)` → techs with prereqs met, tier unlocked, type allowed
- `available_buildings(researched, planet_class, empire_type, existing_buildings, capital_level)` → legal buildings
- `available_districts(planet_class, researched)` → legal districts
- `tech_path_to(target_tech, researched)` → all shortest prereq chains via BFS
- `building_unlock_chain(target_building, researched)` → (techs_needed, building_def)
- `tech_tier_counts(researched, area)` → count per tier for tier gating
- `downstream_unlocks(tech_id)` → buildings, districts, components, ships, further techs
- `estimate_building_net_output(building, empire_state)` → net resource delta with modifiers

**Layer 2 — Meta-systems:**

- `available_traditions(opened_trees, unlocked_traditions, empire_type)` → next unlockable traditions
- `tradition_tree_progress(tree_id, unlocked_traditions)` → (completed, total, remaining)
- `eligible_perks(taken_perks, researched, traditions_completed, empire_type)` → perks passing all checks
- `perk_prerequisites_met(perk_id, taken_perks, researched, traditions_completed)` → bool + missing
- `available_megastructures(empire_state)` → passing tech + perk + concurrent limit
- `megastructure_sites(mega_type, systems)` → systems meeting site requirements
- `megastructure_total_cost(mega_type)` → summed alloys + time across stages

**Layer 3 — Political:**

- `available_policy_options(policy_category, empire_type)` → legal options
- `compute_faction_approval_deltas(policy_category, new_option, empire_state)` → per-faction delta
- `compute_influence_delta(approval_deltas, empire_state)` → net influence change
- `identify_policy_side_effects(policy_category, option, empire_state)` → non-approval consequences
- `optimal_council_assignment(leaders, positions, tech_goal)` → best leader→position mapping
- `agenda_prerequisites_met(agenda_id, empire_state)` → bool + missing

---

## 7. Feature Engineering

Three extractors, all pure functions. Output: flat `dict[str, float]`.

### 7.1 Empire Features (~107)

| Category | Count | Key features |
|---|---|---|
| Resource rates | 20 | Net income × 10 resources, months-until-depletion, stockpile utilization |
| Economic efficiency | 8 | Alloys/pop, research/pop, energy/generator district, etc. |
| Expansion | 8 | Systems, planets, avg size, district/building utilization, border ratio |
| Military | 8 | Fleet power (absolute + relative), naval cap utilization, composition % |
| Tech progression | 12 | Total + per-area + per-tier counts, speed, milestones, frontier size |
| Planet portfolio | 10 | Climate distribution, habitability, remaining slots, specialized worlds |
| Pop metrics | 6 | Total, happiness, unemployment, diversity, faction spread, growth |
| Diplomatic | 6 | Rivals, pacts, federation, subjects |
| Empire archetype | 4 | One-hot: hive, machine, megacorp, standard |
| Traditions & ascension | 8 | Counts, path type, unity income, cost efficiency |
| Leaders & council | 5 | HoR level, council avg, legitimacy, agenda status |
| Factions & influence | 6 | Weighted avg approval, min/max, influence breakdown |
| Galactic community | 2 | Diplomatic weight, council membership |
| Federation | 2 | Level, fleet power |
| Megastructures | 4 | Completed count, under construction, dyson/decompressor flags |

**Top-level function:**
```python
def extract_empire_features(empire: EmpireState, galactic: GalacticState) -> dict[str, float]
```

Each category is a separate private function. Keys prefixed by category (e.g., `resource_net_energy`). All values float. Feature names registered in `FEATURE_REGISTRY: tuple[str, ...]` for consistent tensor ordering.

### 7.2 System Features (~12)

For colonization scoring. Only computed for uncolonized habitable systems.

Features: best_habitability, best_planet_size, district slots by type (4), resource_deposit_value, hyperlane_degree, chokepoint_score, distance_to_capital, distance_to_hostile_border, climate_match_score.

Hyperlane graph built as NetworkX graph from system connections. Betweenness centrality computed once per save.

### 7.3 Temporal Features (~8)

Computed by diffing current save against N previous saves (default N=5).

Features: delta income (energy, mineral, alloy, research), expansion_velocity, tech_velocity, delta_fleet_power, stall_flag.

Returns all zeros if history is empty.

---

## 8. Data Pipeline

### 8.1 Ingestion

```
stellaris-advisor ingest <save_folder>

Per .sav file (sorted by in-game date):
    1. Parse → typed dataclasses
    2. Extract empire features for every empire
    3. Extract system features for uncolonized systems
    4. If previous save exists: compute temporal features + extract actions
    5. Write to Parquet:
       - empire_features.parquet (partitioned by game_id/year)
       - system_features.parquet (partitioned by game_id/year)
       - transitions.parquet (state, action, reward, next_state)
       - game_metadata.parquet
```

### 8.2 Action Extraction

Diff consecutive empire states to infer macro-actions:

| Diff | Inferred action |
|---|---|
| New techs in researched set | Tech researched |
| New buildings on any planet | Building constructed |
| New districts on any planet | District built |
| New systems in owned set | System claimed/colonized |
| New fleet entries / ship counts | Ships built |
| New traditions in unlocked set | Tradition unlocked |
| New perks in taken set | Perk taken |
| New megastructures or stage advances | Megastructure construction |
| Changed policy values | Policy changed |
| Different leader in council position | Council reassignment |

### 8.3 Reward Computation

```
reward = Δ(composite_score) between consecutive saves

composite_score = w1 * norm(fleet_power)
               + w2 * norm(tech_count)
               + w3 * norm(total_income)
               + w4 * norm(pop_count)

Normalization: min-max across all empires in same game at same year.
Default weights: fleet=0.25, tech=0.25, income=0.30, pop=0.20.
Terminal bonus: weighted by final rank.
```

### 8.4 Transitions Schema (transitions.parquet)

| Column | Type | Description |
|---|---|---|
| game_id | string | Game run identifier |
| year | int | Year of state (before action) |
| empire_id | int | Empire index |
| state_features | float[] | ~107 feature vector at T |
| action_type | string | Macro-action category |
| action_params | string (JSON) | Action-specific parameters |
| reward | float | Δ(composite_score) |
| next_state_features | float[] | ~107 feature vector at T+1 |
| terminal | bool | Last transition in game |
| final_rank | int | Empire rank at game end |

### 8.5 Synthetic Data Generation

Scripts for running Stellaris in observer mode:

- `generate_synthetic.sh` — shell script, launches Stellaris with randomized galaxy configs, observer mode, fast speed, autosave at configured intervals
- `generate_synthetic.py` — Python orchestrator: monitors save folder, handles crashes, tracks progress, validates year progression

Randomized settings per run: galaxy size (400–1000), AI empires (8–30), difficulty (ensign–grand admiral), crisis strength (1x–5x), hyperlane density, mid/end-game start years.

Target: 50–100 full game runs for initial training data.

---

## 9. RL Architecture

### 9.1 Approach: Offline RL

Learning from fixed dataset of observer run trajectories. No interactive environment.

**Three phases:**

1. **Phase 1 — No data:** Pure rule-based planner (degraded mode). Functional without any trained model.
2. **Phase 2 — Value function:** Train V(s) and Q(s,a) from observer data using offline RL (CQL/IQL via d3rlpy). Score candidate actions via value delta.
3. **Phase 3 — Planning:** MCTS + V(s) for multi-step lookahead over the macro-action space. Depth-limited to 3–5 steps.

### 9.2 State Encoder

```python
class StateEncoder:
    """Feature dict → normalized torch tensor.
    Learns running mean/std for online normalization.
    Serialized alongside model for consistent inference."""
```

### 9.3 Action Encoder

```python
class ActionEncoder:
    """MacroAction → fixed-size tensor.
    Action type: one-hot over ~10 categories.
    Action params: category-specific continuous features.
    Fixed-size output via padding."""
```

### 9.4 Network Architectures

```python
class ValueNetwork(nn.Module):
    """V(s): 107 features → 256 → 128 → 64 → 1
    BatchNorm + ReLU between layers, dropout 0.1"""

class QNetwork(nn.Module):
    """Q(s, a): concat(107 state, ~20 action) → 256 → 128 → 64 → 1
    BatchNorm + ReLU, dropout 0.1"""
```

### 9.5 Training

**Primary algorithm: CQL (Conservative Q-Learning)**
- Penalizes Q-values for out-of-distribution actions
- Critical for learning from suboptimal AI observer play
- Prevents overestimation of actions the AI never took

**Secondary algorithm: IQL (Implicit Q-Learning)**
- Avoids querying Q for unseen actions entirely
- More conservative, potentially more stable with limited data

**Action space modeling:**
- Macro-actions are discrete categories (~10 types) with continuous parameters (e.g., which building, which planet)
- Approach: encode each (state, action) pair as a single vector. The action is represented as one-hot category + continuous parameter features. d3rlpy treats this as a continuous-action offline RL problem.
- At inference: enumerate all legal actions from action_generator, encode each, query Q(s,a) for all, rank by Q-value.

**Training config:**

```yaml
rl:
  algorithm: "cql"
  learning_rate: 3e-4
  batch_size: 256
  n_epochs: 100
  gamma: 0.99
  cql_alpha: 1.0
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  holdout_galaxy_configs: ["1000_grand_admiral"]
```

**Validation strategy:**
- Game-level split (hold out entire game runs)
- Temporal consistency within games
- Cross-galaxy generalization (reserved configs for test)
- Baselines: lag (V=0), random policy, heuristic policy (degraded mode rules)

**Evaluation metrics:**
- Value prediction: Spearman ρ on held-out games (target: >0.3)
- Action ranking: does Q(s,a) rank actions taken by winning empires higher?

### 9.6 Colonization Scorer (Separate)

LightGBM with LambdaMART ranking loss on system features → colonization priority. Same game-level split. Baseline: habitability-only ranking (target: NDCG@10 >10% improvement).

### 9.7 Benchmark Data Store

Pre-computed percentile distributions from training data:

```
Per (year_bucket, difficulty, empire_type):
    alloy_income, research_total, fleet_power, tech_count,
    system_count, influence_income, unity_income, traditions_completed
    → {p25, p50, p75, p90}
```

Year buckets: every 25 years (2200–2500). Stored as compact JSON shipped with models.

In degraded mode (no benchmark data): hardcoded thresholds by year.

---

## 10. Strategic Planner

### 10.1 Two Modes

```python
class Planner:
    def __init__(self, game_graph, rl_model=None, colonization_model=None):
        self.mode = "full" if rl_model is not None else "degraded"
```

- **Degraded:** `final_score = rule_based_roi` (normalized)
- **Full:** `final_score = α * rl_value_delta + β * rule_based_roi` (α=0.6, β=0.4)

### 10.2 Core Algorithm

```
analyze(empire_state, galactic_state, game_graph, rl_model?, col_model?) → Report

Step 1: Compute reachable frontier (techs, buildings, districts per planet, colony candidates)
Step 2: Generate candidate macro-actions across 9 categories
Step 3: Score each candidate (degraded or full mode)
Step 4: Rank, assemble report with recommendations + bottleneck + alerts
```

### 10.3 Action Generators

Nine generators, one per category. Each returns scored candidates. All respect game constraints — zero illegal actions generated.

```python
def generate_economy_actions(empire, game_graph) -> list[EconomyAction]
def generate_tech_actions(empire, game_graph) -> list[TechAction]
def generate_expansion_actions(empire, galactic, game_graph, col_model?) -> list[ExpansionAction]
def generate_military_actions(empire, galactic, game_graph) -> list[MilitaryAction]
def generate_tradition_actions(empire, game_graph, year) -> list[TraditionAction]
def generate_perk_actions(empire, game_graph, year) -> list[PerkAction]
def generate_megastructure_actions(empire, galactic, game_graph) -> list[MegastructureAction]
def generate_council_actions(empire, game_graph) -> list[CouncilAction]
def generate_faction_actions(empire, game_graph) -> list[FactionAction]
```

### 10.4 Delta Simulator

Approximates post-action state by modifying the feature vector. Pure function — returns new dict, never mutates input.

| Action type | Feature modifications |
|---|---|
| Build building | Adjust resource income/expense, increment building_slots_used |
| Build district | Adjust resource income, housing, district_utilization |
| Research tech | Increment tech counts (total, area, tier), update milestones, adjust frontier size |
| Colonize system | Increment system/planet counts, adjust expansion metrics |
| Build ships | Increment fleet_power, adjust naval_capacity_used, increase alloy expense |
| Open tradition tree | Increment traditions count, apply adoption bonus effects |
| Complete tradition tree | Increment completed count, apply finisher bonus, increment perk slots |
| Take perk | Increment perk count, apply perk effects (e.g., +10% research) |
| Begin megastructure | Increment under_construction, adjust alloy expense, project output discounted by time |
| Switch agenda | Update agenda features, estimate launch effect |
| Reassign HoR | Update HoR level/expertise features, adjust tech draw weights |
| Change policy | Recompute faction approval features, adjust influence income |
| Declare rival | Adjust influence_income by +0.5 per new rival |

### 10.5 Domain Advisors

**Tech path optimizer:** BFS from current state to target. Goal resolution: keyword → tech IDs, resource target → building prereqs → tech chain. Time-discounted downstream value per chain. Step-by-step breakdown.

**Tradition advisor:** Lifetime value of adoption bonuses (front-loaded → higher ROI early). Tree completion value = finisher bonus + perk slot value. Strategic timing: does completing this tree unlock a perk slot for a high-value perk?

**Perk advisor:** Time-discounted lifetime value per eligible perk. Ascension path recommendation based on ethics + economy. Prerequisite tracing. Mutual exclusion awareness.

**Megastructure advisor:** Readiness detection (Mega-Engineering + Galactic Wonders + site + alloys). Full prereq chain tracing when not ready. ROI comparison across types. Site scoring.

**Faction advisor:** Per-policy-change: per-faction approval delta → influence delta. Trade-off warnings. Rival recommendations. Triggers when influence is bottleneck.

**Council advisor:** HoR expertise alignment. Leader→position optimization. Agenda switching with progress loss warnings.

### 10.6 Bottleneck Detector

8 dimensions compared against benchmark percentiles (or hardcoded thresholds in degraded mode):

- alloy_income, research_total, fleet_power, tech_count, expansion, influence_income, unity_income
- megastructure_readiness (binary, checked after year 2300)

Returns (bottleneck_name, normalized_gap). Drives which advisor gets priority.

| Bottleneck | Resolution path |
|---|---|
| alloy_income | Buildings → tech unlocks → build/research chain |
| research_total | HoR expertise → council agenda → tradition bonuses |
| fleet_power | Ship construction → ship class tech path |
| tech_count | Research speed → scientist expertise → tech path |
| expansion | Influence budget → faction management → claims + colonization |
| influence_income | Faction approval → policy changes → rival count |
| unity_income | Unity buildings/jobs → tradition adoption bonuses |
| megastructure_readiness | Mega-Engineering → Galactic Wonders → site → alloy budget |

### 10.7 Alert Generator

15 rule-based alerts (no ML):

| Severity | Alerts |
|---|---|
| High | Resource deficit imminent (<12 months), hostile neighbor (>1.5x fleet), no ascension path (>2280) |
| Medium | Empire stall, faction crisis (<20 approval), influence starved (<2/mo), perk slot unused (10+ years), HoR expertise mismatch |
| Low | Naval cap underutilized (<0.5), building slots wasted (<0.6), megastructure readiness |
| Info | Tech tier gate approaching, crisis imminent, tradition tree nearly complete, council position empty |

### 10.8 Report

```python
@dataclass(frozen=True)
class Report:
    empire_health: EmpireHealth
    recommendations: tuple[Recommendation, ...]
    tech_path_insight: TechPathInsight
    tradition_perk_path: TraditionPerkPath
    faction_influence_summary: FactionSummary
    alerts: tuple[Alert, ...]
    mode: str  # "full" or "degraded"
```

Pure data structure. Rendering handled by CLI layer.

---

## 11. CLI

### 11.1 Commands

Built with Typer. All commands support `--help` and `--format text|json`.

```
stellaris-advisor [--stellaris-dir PATH] [--config PATH] [--verbose] COMMAND

  analyze <save> [--empire NAME|ID] [--top-n INT] [--format text|json] [--shap/--no-shap] [--output PATH]
  parse <save> [--output PATH] [--format json|parquet] [--empire-only] [--pretty]
  ingest <folder> [--output-dir PATH] [--recursive] [--label] [--parallel INT]
  train [--data-dir PATH] [--output-dir PATH] [--config PATH] [--tune/--no-tune] [--tune-trials INT]
  score-systems <save> [--top-n INT] [--format text|json] [--min-habitability FLOAT]
  tech-path <save> --goal GOAL [--top-n INT] [--format text|json]
  benchmark <save> [--format text|json] [--percentiles 25,50,75,90]
  compare <save1> <save2> [--empire NAME|ID] [--format text|json]
  version
```

### 11.2 Auto-Detection

- Stellaris install: macOS → Windows → Linux Steam paths
- Save dir: OS-specific `Documents/Paradox Interactive/Stellaris/save games/`
- Config: `~/.stellaris-advisor/config.yaml` → `./config.yaml` → defaults

### 11.3 Degraded Mode Behavior

- `analyze`, `score-systems`, `benchmark` work without models (rule-based + hardcoded thresholds)
- `train` requires `[ml]` deps — fails fast with install message
- Output includes `[DEGRADED MODE]` indicator

### 11.4 Output

- **text:** Rich-rendered (tables, progress bars, box-drawing report as shown in PRD §6.7.2)
- **json:** Machine-readable, same structure as Report dataclass

---

## 12. Configuration

Pydantic v2, all frozen models. YAML loading with defaults.

```yaml
stellaris:
  install_dir: null                    # Auto-detected
  supported_versions: ["4.3"]
  save_dir: null                       # Auto-detected

parser:
  cache_parsed: true
  cache_dir: "~/.stellaris-advisor/cache/"
  empire_only: false

data:
  processed_dir: "data/processed/"
  raw_dir: "data/raw/"
  parquet_compression: "snappy"
  temporal_lookback: 5

reward:
  fleet_power_weight: 0.25
  tech_count_weight: 0.25
  total_income_weight: 0.30
  pop_count_weight: 0.20
  lookahead_years: 50

rl:
  algorithm: "cql"
  learning_rate: 3e-4
  batch_size: 256
  n_epochs: 100
  gamma: 0.99
  cql_alpha: 1.0
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  holdout_galaxy_configs: ["1000_grand_admiral"]

planner:
  alpha: 0.6
  beta: 0.4
  top_n_recommendations: 7
  top_n_colony_candidates: 5
  naval_utilization_threshold: 0.7
  threat_power_ratio: 1.5
  discount_rate: 0.05
  ascension_path_deadline_year: 2300
  megastructure_readiness_year: 2300
  megastructure_alloy_threshold: 200.0
  influence_crisis_threshold: 2.0
  faction_approval_warning: 20.0
  agenda_switch_progress_threshold: 0.7

benchmark:
  year_bucket_size: 25
  percentiles: [25, 50, 75, 90]
```

---

## 13. Testing Strategy

### 13.1 Framework

pytest + hypothesis + pytest-cov. Target: 80%+ coverage.

### 13.2 Test Organization

Mirrors source structure: `test_parser/`, `test_gamedata/`, `test_features/`, `test_data/`, `test_rl/`, `test_colonization/`, `test_planner/`, `test_cli/`.

### 13.3 Fixtures

- Real save files (small subset) in `tests/fixtures/`
- Real game definition files (subset) in `tests/fixtures/game_data/`
- Constructed dataclass instances in `conftest.py` for fast unit tests
- Tiny synthetic datasets for RL smoke tests (generated, not committed)

### 13.4 Approach by Layer

| Layer | Technique | Key assertions |
|---|---|---|
| Parser (clausewitz) | Hypothesis property-based | Roundtrip fidelity; edge case handling |
| Parser (extractor) | Unit with fixtures | Extracted fields match manually verified values |
| Game data | Unit with real files | All integrity validations pass; queries match hand-verified results |
| Game data (validation) | Negative tests | Broken data triggers correct errors |
| Features | Unit with constructed state | Count matches registry; values in range; no NaN |
| Data pipeline | Integration | Parquet schema, row counts, partitioning correct |
| RL | Smoke with tiny data | Shapes correct; training completes; serialization roundtrips |
| Planner (constraints) | Unit per category | **Zero constraint violations** |
| Planner (tech paths) | Property tests | **100% valid DAG traversals** |
| Planner (advisors) | Unit with crafted scenarios | Ordering matches meta; exclusions respected; deltas directionally correct |
| CLI | Snapshot tests | Format matches expected; JSON is valid |

### 13.5 Markers

```python
@pytest.mark.unit          # Fast, no I/O
@pytest.mark.integration   # Real files, disk access
@pytest.mark.slow          # RL training, large saves
```

Default: `pytest -m "not slow"`. CI runs everything.

---

## 14. Evaluation Plan

### 14.1 Model Metrics

| Component | Primary metric | Target |
|---|---|---|
| RL Value Function | Spearman ρ on score delta prediction | >0.3 on held-out games |
| Colonization Scorer | NDCG@10 | >10% improvement over habitability heuristic |
| Planner | Player agreement rate (manual eval) | Sensible on 10+ real saves |
| Tech Path Optimizer | Path validity rate | 100% legal DAG traversals |
| Tradition/Perk Advisor | Coherence (manual eval) | No invalid suggestions; matches community meta |
| Faction Analyzer | Approval delta accuracy | Directionally correct on all mappings |
| Megastructure Advisor | Readiness detection accuracy | Correct when all prereqs met |
| Bottleneck Detector | Manual assessment agreement | ≥70% correct |

### 14.2 Baselines

- Lag: V(s) = 0 (no change)
- Random: random action ranking
- Heuristic: degraded mode rule-based planner
- Habitability-only: colonization by habitability only

### 14.3 Validation Split

- Game-level split: 70% train, 15% val, 15% test (by game count)
- Cross-galaxy: reserve 1000-star grand admiral for test only

---

## 15. Known Limitations

Carried forward from PRD §11, adjusted for RL approach:

1. Greedy single-step planning (Phase 1–2). MCTS in Phase 3 addresses this.
2. No opponent modeling.
3. No event/crisis anticipation.
4. No combat simulation (aggregate fleet power only).
5. No pop growth/job assignment modeling (approximate ROI).
6. Vanilla Stellaris 4.3 only.
7. No multiplayer-specific advice.
8. Cannot predict stochastic tech draws (can steer weights).
9. No Shroud interaction modeling.
10. No espionage, vassal optimization, galactic community strategy, or species rights optimization.
11. Offline RL learns from AI observer play, not optimal human play. Models capture "what typically happens" not "what's optimal." Community player saves would improve this post-launch.

---

## 16. Build Sequence

Foundation-up, layered:

**Foundation:**
1. Parser (clausewitz tokenizer + recursive descent + typed extraction)
2. Game knowledge graph Layer 1 (techs, buildings, districts, jobs, ships, planet classes)
3. Game knowledge graph Layer 2 (traditions, perks, megastructures)
4. Game knowledge graph Layer 3 (policies, factions, council, federations)

**Middle layers:**
5. Feature engineering (empire, system, temporal)
6. Data pipeline (ingest, action extraction, reward, Parquet)
7. RL training pipeline (encoders, networks, d3rlpy trainer, evaluation)
8. Colonization scorer (LightGBM training + benchmark store)

**Intelligence:**
9. Planner — action generators (all 9 categories)
10. Planner — delta simulator
11. Planner — domain advisors (tech, tradition, perk, megastructure, faction, council)
12. Planner — bottleneck detector + alerts + report assembly
13. Planner — ranker (degraded + full modes)

**Surface:**
14. CLI (all 8 commands)
15. Config (Pydantic schema + YAML loading)

Each step is independently testable.
