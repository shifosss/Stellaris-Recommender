# Stellaris AI Advisor — Product Requirements Document

**Project codename:** `stellaris-advisor`
**Author:** Alex Zhang
**Date:** 2026-03-26
**Status:** Draft v4.0 (unified)
**Timeline:** 2–4 weeks (side project)

---

## 1. Problem Statement

Stellaris generates enormous quantities of structured game-state data every autosave — resource flows, planet attributes, system topology, fleet compositions, tech progress — yet no existing tool applies trained ML models to this data. Current offerings fall into two camps:

- **Dashboard tools** (Stellaris Dashboard, Stellaru): parse saves and visualize historical data, but offer zero predictive or prescriptive capability.
- **LLM wrappers** (Stellaris Companion, ChatGPT GPTs): send extracted text to a language model and return natural-language advice. No trained models, no quantitative predictions, no evaluation rigor.

There is a clear gap: a tool that treats Stellaris save data as a proper ML dataset — extracts features, trains models, evaluates predictions — and combines learned value estimation with structured knowledge of the game's dependency graph to deliver quantitative, actionable recommendations.

---

## 2. Existing Landscape

### 2.1 Stellaris Dashboard (eliasdoehne/stellaris-dashboard)
- Open-source Python + Rust program; reads saves in real time while the game runs.
- Stores parsed data in a local DB; renders graph-based dashboards and a galaxy map in a browser.
- Updated for Stellaris 4.2; maintainer has stepped back as of February 2026.
- Uses a Rust-based Clausewitz parser exposed to Python via `maturin`. This parser is the most mature open-source Clausewitz parser available and is a strong candidate for reuse.
- **Gap:** Visualization only; no predictions, no recommendations.

### 2.2 Stellaru (benreid24/Stellaru)
- Python (Django) + JavaScript data visualization suite.
- Monitors autosaves, provides 30+ chart types, custom dashboards.
- Stores historical data as zipped pickle files alongside the save folder.
- Works in multiplayer.
- **Gap:** Visualization only; no ML, no prescriptive output.

### 2.3 Stellaris Companion (galacticfilingcabinet.com)
- Electron desktop app, launched ~February 2026.
- Reads save files locally, extracts a structured briefing, sends it to Gemini via user's API key.
- Features: natural-language strategy advice, empire chronicle (auto-generated narrative history), crisis detection, personality modes for gestalt empires.
- Discord overlay integration.
- Open source.
- **Gap:** No trained models; advice quality is entirely dependent on the LLM's general knowledge; no quantitative predictions or evaluation; no game data graph traversal; no ROI calculations.

### 2.4 ChatGPT GPTs (Stellaris Oracle, Galactic Advisor)
- Custom GPT wrappers on ChatGPT.
- Accept text descriptions of game state; return generic advice.
- No save file integration, no data pipeline, no trained models.
- **Gap:** No game state access; advice is generic and unverifiable.

### 2.5 pypdx-clausewitz
- JavaScript/TypeScript Clausewitz format parser.
- Converts Paradox save files to JSON.
- Used by the stellaris-save-vid project.
- **Reuse potential:** Useful as reference for format specification, but Python/Rust is preferred for this project's stack.

### 2.6 Stellaris-Save-API (AgainPsychoX)
- JavaScript library and CLI for modifying Stellaris saves.
- Parsing, proxy classes for data access, utility tools.
- Last verified for Stellaris v3.4.5 (Cepheus) — likely needs updates for 4.x format changes.

### 2.7 Competitive Differentiation Summary

| Capability | Dashboard | Stellaru | Companion | GPTs | **This project** |
|---|---|---|---|---|---|
| Save parsing | ✓ | ✓ | ✓ | ✗ | ✓ |
| Historical visualization | ✓ | ✓ | ✗ | ✗ | ✗ (post-MVP) |
| LLM-based advice | ✗ | ✗ | ✓ | ✓ | ✗ |
| Trained ML models | ✗ | ✗ | ✗ | ✗ | **✓** |
| Quantitative predictions | ✗ | ✗ | ✗ | ✗ | **✓** |
| Game knowledge graph | ✗ | ✗ | ✗ | ✗ | **✓** |
| Tech path optimization | ✗ | ✗ | ✗ | ✗ | **✓** |
| Constraint-aware recommendations | ✗ | ✗ | ✗ | ✗ | **✓** |
| Evaluation rigor | ✗ | ✗ | ✗ | ✗ | **✓** |
| Python library API | ✗ | ✗ | ✗ | ✗ | **✓** |

---

## 3. Complexity Analysis

### 3.1 Game Dimensions

| Dimension | Scale | Detail |
|---|---|---|
| Resource types | **19 total** | **Core (7):** Energy Credits, Minerals, Food, Consumer Goods, Alloys, Unity, Influence. **Research (3):** Physics Research, Society Research, Engineering Research. **Strategic (3):** Rare Crystals, Exotic Gases, Volatile Motes. **Rare (3):** Dark Matter, Zro, Living Metal. **Other (3):** Nanites, Minor Artifacts, Menace. Trade was reworked into a stockpilable logistical resource in 4.0 — planetary deficits and military fleets impose trade costs; trade can be spent on the market. |
| Technologies | **~400 base + ~50 rare/event** | 3 areas (Physics, Society, Engineering), 13 subcategories, 6 tiers (T0–T5). Tier gating: 6 techs from tier N required to unlock tier N+1 draws. Drawn via weighted card-shuffle: 3 semi-random options per slot, weights modified by ethics, civics, scientist traits, and prior draws (a tech in the previous hand has halved weight). ~50 additional techs are event-locked (debris analysis, precursors, archaeology, ascension). Repeatable end-game techs exist at the top. |
| Buildings | **~80–100+ types** | Categories: resource production, military (strongholds/fortresses with FTL inhibitors), law enforcement, refineries (strategic resources), research labs, unity, nanotech (4.0+), megastructures, unique/empire-limited, event buildings, fallen empire buildings, DLC-specific. Up to 12 building slots per planet; unlocked by capital upgrade, city districts, technologies, and civics. |
| Districts | **~5 base types + specializations** | City, Generator, Mining, Farming, Industrial. District specializations added in 4.0: city districts get 2 specialization slots, resource districts get 1. Specializations provide extra jobs per district and unlock additional building slots. Availability varies by planet class (e.g., no farming on tomb worlds without tech, no mining on habitats). |
| Ship classes | **5 hull types** | Corvette, Destroyer, Cruiser, Battleship, Titan. Each hull has section layouts with weapon/utility slots. Dozens of component types (weapons: energy, kinetic, explosive, strike craft; defense: shields, armor, hull; utility: reactors, thrusters, computers, sensors, auras). Component availability gated by techs. |
| Planet classes | **~20+ types** | Standard habitable: Continental, Ocean, Tropical, Arid, Desert, Savanna, Arctic, Tundra, Alpine, + Tomb World. Special: Gaia World, Ecumenopolis, Ring World segment, Habitat, Hive World, Machine World, Relic World, Nanite World. Each class constrains which districts/buildings are available and affects habitability per species. |
| Empire types | **4 authority × 2–3 ethics × 2 civics × origins** | Authority: Democratic, Oligarchic, Dictatorial, Imperial (+ gestalt: Hive Mind, Machine Intelligence). Ethics: 7 standard (3 with fanatic variants) + gestalt. Civics: ~50+ options with mutual exclusions. Origins: ~30+ options. Different combinations unlock or restrict specific techs, buildings, and traditions. |
| Victory conditions | **3 effective** | (1) Score victory at victory year — composite of economy, military, tech, pops, diplomacy, federation, subjects, crisis contribution. (2) Total galactic conquest / extermination. (3) Crisis path completion (Become the Crisis perk or Cosmogenesis ascension). Score is the de facto optimization target — it is continuous, multi-dimensional, and visible in the situation log. |
| Traditions | **14 trees × ~7 traditions = ~98** | Trees: Discovery, Expansion, Domination, Prosperity, Harmony, Supremacy, Unyielding, Diplomacy, Subterfuge, Mercantile, Aptitude, Statecraft, Politics (requires Galactic Community), + ascension trees (Synthetics, Cybernetics, Psionics, Genetics). Each tree has an adoption bonus (immediate on opening) and finisher bonus (on completion). Cost formula scales with number of traditions already unlocked and empire size. Completing a tree unlocks an Ascension Perk slot. Players can open multiple trees simultaneously to capture front-loaded adoption bonuses. |
| Ascension Perks | **~36 perks, 8 slots per empire** | Tiered (T0–T3, requiring 0–3 prior perks). Permanent once chosen. Key perks: Technological Ascendancy (+10% research speed), Interstellar Dominion (-20% claim/starbase influence cost), Galactic Wonders (unlocks Dyson Sphere, Matter Decompressor, Ring World, Science Nexus), Master Builders (+50% megastructure speed, +1 build capacity), Arcology Project (enables Ecumenopolis), Colossus Project (planet-killer ships + Total War casus belli). Ascension path perks (Tier 2, mutually exclusive) unlock species transformation tradition trees. |
| Ascension Paths | **4 mutually exclusive** | Synthetic Evolution (organic pops → synthetic: no food, +20% output, immortal leaders), Cybernetic Augmentation (+20% habitability, +various bonuses), Psionic Ascension (Shroud access, telepathy, high-variance covenants), Biomorphosis (extreme biological trait customization). Each triggers a multi-stage ascension situation with approach choices. The single most transformative strategic decision in the game. Choice should align with ethics: Materialist → Synthetic/Cybernetic; Spiritualist → Psionic. |
| Leaders & Council | **~5–10 leaders, 3–6 council positions** | Leader classes: Scientist, Admiral, General, Governor, Official. Skill levels 1–10, positive/negative traits, veteran classes (chosen at level 4). Council: Ruler + 3–5 named positions (Head of Research, Minister of Defense, Minister of State, + civic-granted). Head of Research skill level gives +2% research speed per level; empty = -25% research speed. Scientist expertise traits give 25% draw weight bonus to matching tech subcategory — the only way to steer the stochastic tech draw. Council agendas provide empire-wide bonuses: 7000 progress to prepare, progress rate = Σ(10 + councilor_level)/month × legitimacy modifier. Key agendas: Military Buildup, Chart the Unknown, Infinite Opportunities, ascension path agendas. |
| Factions | **9 standard factions** | One per ethic (Supremacist/Militarist, Progressive/Egalitarian, Technologist/Materialist, Traditionalist/Spiritualist, etc.). Approval (0–100) driven by binary issue fulfillment tied to policies, diplomacy, and game state. Factions produce influence proportional to support × approval. Primary influence engine for most empires. Faction approval also drives pop happiness → stability → all job output. Council legitimacy derived from faction approval → agenda preparation speed. Gestalt empires have no factions. |
| Policies | **~15 categories, 2–4 options each** | Economic Policy (trade conversion: energy/consumer goods/unity), War Philosophy (unrestricted/liberation/defensive), Orbital Bombardment, AI Policy (outlawed/servitude/full rights — gates robot buildings), Robot Workers, Population Controls, Resettlement, Leader Enhancement, etc. 10-year cooldown on changes. Policies are the primary lever for faction management: each policy option pleases some factions and angers others. AI Policy particularly consequential — outlawing AI blocks synthetic pop growth but pleases Spiritualists. |
| Galactic Community | **Galaxy-wide legislature** | Resolutions pass/fail by diplomatic weight (40% economy, 25% tech, 20% military, 15% diplomacy). Active resolutions impose rules on all members: research bonuses/penalties, banned weapons, market regulations, forced refugee acceptance. Galactic Council (top 3 by diplomatic weight) gets emergency measures and veto. Custodian path → Galactic Emperor (massive power, near-total control). Politics tradition tree unlocks special resolution powers. |
| Federations | **5 types, 5 levels** | Galactic Union (generic), Trade League (shared trade value), Martial Alliance (shared fleet), Research Cooperative (shared research speed), Hegemony (one dominant member). Laws (tax rates, fleet contribution, war voting), cohesion, experience. Higher levels unlock stronger bonuses: Research Cooperative level 5 = +10% research speed for all members. Federation fleet controlled by president. |
| Megastructures | **~15 types** | Key: Dyson Sphere (+4000 energy/mo), Matter Decompressor (+2000 minerals/mo), Ring World (4 × 50-district segments), Science Nexus (+300 research/mo), Strategic Coordination Center (+150 naval cap), Mega Shipyard (20 shipyards). Prerequisite chain: Mega-Engineering tech (requires Battleships + Citadels + restored/built megastructure) → Galactic Wonders perk → site requirements (specific star types) → multi-stage construction (5–10 years, thousands of alloys). Only 1 under construction at a time (2 with Master Builders). Economic phase transition: a single Dyson Sphere exceeds most empires' entire planetary energy output. |
| Warfare | **Claims, casus belli, war exhaustion, ground combat** | Claims cost influence per system (placed before war). Casus belli types: Claim, Subjugation, Ideology, Total War (Colossus). War exhaustion accumulates from battles, occupations, time; at 100% status quo can be forced. Ground armies (clone, psionic, xenomorph, gene warrior) required to occupy planets. Fleet combat determined by composition, components, admiral traits. |

### 3.2 Decision Space

At any given moment, a player can:

- Build/upgrade districts on each planet (~5 types × ~15 planets), constrained by planet class and available district slots
- Build/upgrade buildings on each planet (~80+ eligible types × 12 slots × ~15 planets), constrained by planet class, empire type, researched techs, and existing buildings (some are unique per planet or per empire)
- Choose 1 of 3 tech options per research area (3 choices), each with different research point costs (scaling with empire size), tech-tier prerequisites, and downstream unlock chains
- Construct/upgrade starbases and modules (6 module types, 4 building slots × ~30 starbases)
- Build ships across 5 hull classes with dozens of component combinations per section layout
- Issue edicts (~20+ types, costing unity or influence, with duration and prerequisites)
- Make diplomatic actions per known empire (~10 action types × ~15 empires): improve/harm relations, propose/break treaties, declare rivalry, form/leave federation, declare war with casus belli
- Colonize systems (requires colony ship, influence cost, habitability check)
- Resettle pops between planets
- Change policies and species rights (10-year cooldown per policy; each choice affects faction approval)
- Manage factions (promote, suppress, embrace — costing unity/influence, with faction approval and influence income consequences)
- Select and unlock traditions (spend unity; choose which tree to open/advance; tradition order is one of the top 5 strategic decisions)
- Choose Ascension Perks when slots become available (permanent, irreversible; 8 slots total; gates megastructures, ascension paths, crisis paths)
- Choose Ascension Path (mutually exclusive species transformation; the single most consequential decision in the mid-game)
- Assign leaders to council positions and manage agendas (select strategic focus; choose which leader occupies Head of Research based on expertise match)
- Engage with galactic community (propose/vote on resolutions; pursue custodian/emperor path)
- Join/leave federations and manage federation laws (tax rates, centralization, fleet contribution)
- Build megastructures (multi-year, multi-thousand-alloy investments with massive late-game output)
- Trigger planetary decisions (specialize, terraform, consecrate, etc.)
- Place claims and declare wars (influence investment; casus belli selection; ground army composition)

**Conservative combinatorial estimate: millions to billions of possible action combinations at any decision point.** This is closer to Go or StarCraft than to chess.

### 3.3 Critical Dependency Chains

Decisions in Stellaris are not independent. There are two interleaving dependency chains that constrain what actions are available.

**Chain 1: The Tech → Building → Resource Chain (from v3.0)**

```
Empire type (ethics, civics, authority, origin)
    + Species traits (habitability, resource bonuses)
    + Planet portfolio (class, size, features, existing districts/buildings)
        ↓ constrains
    Available technologies
        (filtered by empire type, weighted by ethics/traits/scientist expertise;
         tier-gated by count of prior-tier techs researched;
         drawn semi-randomly from eligible pool;
         Head of Research expertise gives 25% draw weight bonus)
        ↓ unlocks
    Available buildings, district specializations, ship components, edicts
        (each gated by specific tech prerequisites;
         further filtered by planet class and empire type)
        ↓ produces
    Resource income delta
        (deterministic: building/district definitions specify exact output;
         modified by empire-wide bonuses, species traits, stability, etc.)
        ↓ enables
    Next tier of decisions
```

**Chain 2: The Unity → Tradition → Perk → Transformation Chain (NEW)**

This is the **meta-decision layer** that sits above the tech tree and shapes the entire trajectory of an empire. The PRD v3.0 completely missed this chain.

```
Unity income (from jobs, factions, buildings, edicts)
    ↓ spent on
Tradition tree selection and ordering
    (14 trees; adoption bonus is front-loaded — opening 2–3 trees early
     captures multiple bonuses before completing any)
    ↓ completing a tree unlocks
Ascension Perk slot (8 slots total per empire)
    ↓ perk selection gates
Ascension Path choice (Tier 2 perk, mutually exclusive)
    OR Galactic Wonders / Master Builders / Arcology Project / other
    ↓ path perk unlocks
Ascension Tradition tree (Synthetics/Cybernetics/Psionics/Bio)
    + Ascension Situation (multi-stage with approach choices)
    ↓ completing the situation triggers
Species Transformation
    (all pops become Synthetic/Cybernetic/Psionic/Bio-enhanced;
     fundamentally changes resource needs, pop growth mechanics,
     leader traits, available buildings/techs)
```

This chain is **more consequential than any individual tech or building decision**. Choosing the wrong tradition order or the wrong ascension path can permanently weaken an empire compared to the optimal path for its build.

**Chain 3: The Faction → Influence → Expansion Loop (NEW)**

Influence is the scarcest resource in the game, and it gates expansion, diplomacy, and claims. Its generation is tightly coupled to faction management:

```
Policies (economic, war, AI, species rights)
    ↓ determine
Faction approval (each policy fulfills or violates faction issues)
    ↓ drives
Influence income (faction_support × approval × modifier)
    + Power Projection (fleet-to-empire ratio, max 2/mo)
    ↓ spent on
Expansion (starbases: 75 influence each)
    + Claims (75+ influence per system, placed before wars)
    + Diplomacy (agreements, federation actions)
    + Edicts (some cost influence)
    ↓ expansion increases
Empire size
    ↓ which increases
Tradition cost (scales with empire size)
    AND Tech cost (scales with empire size)
    ↓ creating tension with
Unity income and Research speed
```

This creates a fundamental tension: expanding (wide play) generates more resources but increases tradition and tech costs. Staying small (tall play) keeps costs low but limits resource generation. The optimal balance depends on empire type, galaxy configuration, and game phase.

### 3.4 Integrated Strategic Model

All three chains interact through causal loops:

```
                    ┌──────────────────────┐
                    │   VICTORY SCORE      │
                    │ (economy + tech +     │
                    │  military + pops +    │
                    │  diplomacy)           │
                    └──────┬───────────────┘
                           │ driven by
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    ┌───────────┐   ┌───────────┐    ┌───────────┐
    │ RESOURCES │   │ TECH      │    │ FLEET     │
    │ (alloys,  │   │ (research │    │ POWER     │
    │  energy,  │   │  speed,   │    │           │
    │  etc.)    │   │  count)   │    │           │
    └─────┬─────┘   └─────┬─────┘    └─────┬─────┘
          │               │                │
     produced by     driven by         built with
          │               │                │
    ┌─────▼─────┐   ┌─────▼─────┐    ┌─────▼─────┐
    │ BUILDINGS │   │ LEADERS   │    │ ALLOYS +  │
    │ DISTRICTS │   │ (council, │    │ NAVAL CAP │
    │ MEGASTR.  │   │  expertise│    │           │
    └─────┬─────┘   │  agendas) │    └───────────┘
          │         └─────┬─────┘
     gated by        affects
          │               │
    ┌─────▼─────┐   ┌─────▼─────┐
    │ TECHS     │   │ TECH DRAW │
    │ (prereqs) │   │ WEIGHTS   │
    └─────┬─────┘   └───────────┘
          │
     gated by                      ┌──────────────┐
          │                        │ EXPANSION    │
    ┌─────▼─────┐                  │ (systems,    │
    │ TRADITIONS│                  │  planets)    │
    │ → PERKS   │                  └──────┬───────┘
    │ → PATH    │                         │
    └─────┬─────┘                    gated by
          │                              │
     costs unity              ┌──────────▼──────┐
          │                   │  INFLUENCE      │
    ┌─────▼─────┐             └────────┬────────┘
    │ UNITY     │                      │
    │ PRODUCTION│                 produced by
    └─────┬─────┘                      │
          │                   ┌────────▼────────┐
     from jobs +              │  FACTIONS       │
     factions                 │  (approval ×    │
          │                   │   support)      │
          │                   └────────┬────────┘
          │                            │
          │                       driven by
          │                            │
          │                   ┌────────▼────────┐
          └──────────────────►│  POLICIES       │
                              └─────────────────┘
```

**Key insight for the planner:** The "best next step" is not just "which building produces the most alloys" — it requires identifying the **binding constraint** in this web of loops:

- If influence is the binding constraint → optimize faction approval → check policies → check rival count → recommend changes.
- If unity is the binding constraint → optimize tradition pace → check if adoption bonuses from a new tree would help → recommend specific tradition picks.
- If tech is the binding constraint → check Head of Research expertise alignment → check if a better scientist is available for council → check tradition bonuses affecting research → recommend research path.
- If alloys are the binding constraint → check if a building is available now → check if a tech unlock is needed first → trace the chain to the optimal build.
- If expansion is blocked by influence → trace to faction management.
- If late game and megastructures aren't started → trace the entire prerequisite chain: Mega-Engineering tech → perk slot availability → Galactic Wonders perk → suitable site → alloy budget → construction.
- If no ascension path is chosen by mid-game → recommend based on empire ethics, current economy, and planet portfolio.

### 3.5 What ML Can and Cannot Solve Here

**ML is well-suited for:**
- **State valuation** — "given these 60–80 empire features at year T, how does this empire rank at T+50?" (regression on tabular data)
- **Colonization ranking** — "which uncolonized system is most valuable?" (learning-to-rank on ~12 features per system)
- **Pattern detection** — identifying "empire stall" signatures, resource crisis precursors, benchmark comparisons against historical data
- **Resource trajectory forecasting** — time-series prediction on income curves

**ML alone is NOT sufficient for:**
- **Multi-step strategic planning** — the action space is too large and too coupled for a model to learn action values from data alone within this project scope
- **Tech path optimization** — the tech prerequisite graph is a known DAG with ~400 nodes; this is a graph search problem, not a learning problem
- **Building constraint resolution** — which buildings are available given (planet type, empire type, researched techs) is deterministic and fully specified in game files
- **Tradition/perk ordering** — tradition prerequisite structure and perk tier requirements are known; optimal ordering can be computed from bonuses and costs
- **Faction issue resolution** — which policies fulfill which faction issues is deterministic; the advisor can compute the exact approval delta of any policy change
- **Megastructure prerequisite chain** — the tech → perk → site → build chain is fully specified in game files

**The solution: a hybrid architecture** — ML models for state valuation (what's hard to compute from rules), combined with a rule-based graph planner that traverses the known game dependency graph (what's structured and deterministic).

---

## 4. Product Vision

`stellaris-advisor` is a **Python library + CLI tool** that:

1. Parses Stellaris save files into structured feature vectors.
2. Loads the game's tech/building/district dependency graph from game definition files.
3. Trains ML models on historical game-state data to learn empire state valuation.
4. Given a current save, combines ML value estimation with graph-based path planning to output: **empire health summary**, **scored next-best-action recommendations** (across economy, expansion, tech, and military), **optimal tech → building paths with ROI estimates**, and **system colonization rankings**.

The intelligence comes from two sources: **ML models** trained on actual game outcomes for state valuation, and a **rule-based planner** that leverages the known structure of the game's dependency graph.

---

## 5. Target User

**Primary:** The developer (Alex) — a Stellaris player who wants to combine ML and data engineering skills with a game they enjoy. The project should be fun to use on their own playthroughs.

**Secondary:** Stellaris community members who want quantitative strategy analysis. The library is pip-installable and the CLI works on any vanilla Stellaris 4.3 save file.

**Tertiary:** Stellaris modders and tool developers who may want to build on the parsing and game data layers.

---

## 6. Core Features (MVP)

### 6.1 Save Parser (`stellaris_advisor.parser`)

#### 6.1.1 Clausewitz Format Specification

Stellaris save files (`.sav`) are ZIP archives containing two plaintext files:

- **`gamestate`** — the full game state (~5–200 MB uncompressed depending on galaxy size and game year). Uses Paradox's proprietary Clausewitz serialization format: nested key-value blocks with `{ }` delimiters, no schema, no standard encoding beyond UTF-8 with UNIX-style newlines.
- **`meta`** — metadata shown on the load screen (empire name, year, date, version, DLC flags).

The Clausewitz format is shared across all Paradox titles (EU4, HOI4, CK3, etc.). Key format characteristics:

```
# Example Clausewitz structure (simplified)
country={
    0={
        name="United Nations of Earth"
        government={
            type="gov_democratic"
            authority="auth_democratic"
            civics={ "civic_beacon_of_liberty" "civic_idealistic_foundation" }
        }
        budget={
            current_month={
                income={
                    energy=142.50
                    minerals=98.30
                    alloys=43.00
                }
                expenses={
                    energy=130.20
                    minerals=45.00
                }
            }
        }
        tech_status={
            technology={
                0={
                    technology="tech_corvettes"
                    level=1
                }
                1={
                    technology="tech_destroyers"
                    level=1
                }
            }
        }
        owned_planets={ 0 4 12 27 }
    }
}
```

Key parsing challenges:
- No formal schema; field names and nesting can change between game versions.
- Integer keys used as indices (not named keys) for many collections.
- Ironman saves use the same format but file must be named `ironman.sav` and uses specific ZIP compression settings (UNIX newlines, correct timestamps, correct compression level).
- Very large files for late-game states on large galaxies (1000-star galaxies can produce 100+ MB gamestate files).

#### 6.1.2 Parser Implementation Strategy

**Option A (recommended): Reuse stellaris-dashboard Rust parser.**
- Located at `stellarisdashboard/parsing/rust_parser/`.
- Compiled via `maturin` into a Python-callable library.
- Fastest available parser; handles the full Clausewitz format including edge cases.
- Returns a Python dict tree.
- Risk: may need updates for 4.3-specific format changes.

**Option B (fallback): pypdx-clausewitz.**
- Pure JavaScript/TypeScript — would need a Python port or subprocess call.
- Slower than Rust but well-documented.

**Option C (last resort): Custom Python parser.**
- Write a recursive descent parser for the Clausewitz format.
- Most flexible but most development time.
- Only justified if Options A and B fail for 4.3 saves.

#### 6.1.3 Typed Extraction Layer

On top of the raw dict, build a typed extraction layer that maps game-version-specific dict paths to stable Python dataclasses.

```python
@dataclass
class EmpireState:
    id: int
    name: str
    # Identity
    authority: str                      # e.g., "auth_democratic"
    ethics: list[str]                   # e.g., ["ethic_fanatic_egalitarian", "ethic_xenophile"]
    civics: list[str]                   # e.g., ["civic_beacon_of_liberty", ...]
    origin: str                         # e.g., "origin_prosperous_unification"
    government_type: str
    ascension_perks: list[str]
    # Resources (all 19 types)
    resource_income: dict[str, float]   # resource_id → monthly income
    resource_expense: dict[str, float]  # resource_id → monthly expense
    resource_stockpile: dict[str, float]# resource_id → current stockpile
    resource_storage_cap: dict[str, float]
    # Tech
    researched_techs: set[str]          # set of tech IDs
    current_research: dict[str, str]    # area → tech_id currently being researched
    research_progress: dict[str, float] # area → % complete on current tech
    research_speed: dict[str, float]    # area → monthly research output
    stored_research: dict[str, float]   # area → banked research points
    # Military
    fleet_power: float
    naval_capacity_used: int
    naval_capacity_max: int
    fleets: list[FleetState]
    # Territory
    owned_systems: list[int]            # system IDs
    owned_planets: list[PlanetState]
    starbases: list[StarbaseState]
    # Pops
    total_pops: int
    pop_groups: list[PopGroupState]     # 4.0+ pop group system
    # Diplomacy
    rivals: list[int]                   # empire IDs
    defensive_pacts: list[int]
    federation_id: Optional[int]
    federation_power: Optional[float]
    subjects: list[int]                 # vassal empire IDs
    overlord: Optional[int]
    # Traditions & Ascension
    traditions_unlocked: list[str]       # individual tradition IDs
    tradition_trees_opened: list[str]    # tree IDs where adoption bonus is active
    tradition_trees_completed: list[str] # tree IDs fully completed
    ascension_perks: list[str]           # perk IDs taken
    ascension_perk_slots_available: int
    ascension_perk_slots_total: int
    ascension_path: Optional[str]        # "synthetic"/"cybernetic"/"psionic"/"biomorphosis"/None
    ascension_situation_progress: Optional[float]  # 0.0–1.0 if situation active
    # Leaders & Council
    leaders: list[LeaderState]           # all leaders (class, level, traits, expertise, assignment)
    council_positions: list[CouncilPosition]  # position + assigned leader
    current_agenda: Optional[AgendaState]     # type, progress %, launch status
    council_legitimacy: float            # 0–100, derived from faction approval
    # Factions
    factions: list[FactionState]         # ethic, approval, size, support, issues list
    # Policies
    policies: dict[str, str]             # policy_category → selected_option_id
    # Galactic Community
    galcom_member: bool
    galcom_diplomatic_weight: float
    galcom_council_member: bool
    galcom_custodian: bool
    galcom_emperor: bool
    active_resolutions: list[str]        # resolution IDs currently in effect
    # Federation
    federation: Optional[FederationState]  # type, level, cohesion, experience, fleet power
    # Megastructures
    megastructures: list[MegastructureState]  # type, stage, system_id
    megastructures_under_construction: int
    # Warfare
    active_wars: list[WarState]          # war goal, exhaustion, claims
    claims_placed: list[ClaimState]      # system_id, influence_cost
    total_army_strength: float
    # Score
    victory_score: float
    victory_rank: int

@dataclass
class PlanetState:
    id: int
    name: str
    planet_class: str                   # e.g., "pc_continental", "pc_ocean"
    size: int                           # planet size (affects district cap)
    # Districts
    districts: dict[str, int]           # district_type → count built
    district_slots: dict[str, int]      # district_type → max slots available
    total_district_cap: int             # total district cap (from size)
    districts_used: int                 # total districts built
    # Buildings
    buildings: list[str]                # building IDs in each slot
    building_slots_total: int           # unlocked building slots
    building_slots_used: int
    # Specializations (4.0+)
    district_specializations: dict[str, str]  # district_type → specialization_id
    # Population
    pops: int
    pop_groups: list[PopGroupSummary]
    # Modifiers
    stability: float
    crime: float
    amenities: float
    amenities_usage: float
    housing: float
    housing_usage: float
    designation: str                    # colony designation / specialization
    # Features
    planetary_features: list[str]       # resource deposits, blockers, modifiers
    is_capital: bool

@dataclass
class SystemState:
    id: int
    name: str
    star_class: str
    hyperlane_connections: list[int]     # connected system IDs
    owner: Optional[int]                # empire ID
    claimed_by: list[int]               # empire IDs with claims
    starbase: Optional[StarbaseState]
    planets: list[PlanetSummary]        # all planets (habitable and non-habitable)
    resources: dict[str, float]         # minable space resources
    has_megastructure: bool
    megastructure_type: Optional[str]

@dataclass
class StarbaseState:
    system_id: int
    level: str                          # outpost / starport / starhold / star fortress / citadel
    modules: list[str]                  # module IDs (shipyard, anchorage, gun battery, etc.)
    buildings: list[str]                # starbase building IDs
    defense_fleet_power: float

@dataclass
class FleetState:
    id: int
    name: str
    fleet_power: float
    ships: dict[str, int]               # ship_class → count
    naval_capacity_used: int
    location_system: int
    commander: Optional[LeaderSummary]

@dataclass
class ColonyCandidate:
    system_id: int
    system_name: str
    planet_id: int
    planet_name: str
    planet_class: str
    size: int
    habitability: float                 # for player's primary species
    district_slots: dict[str, int]
    resource_deposits: dict[str, float]
    distance_to_capital: int            # hyperlane hops
    distance_to_hostile_border: int
    is_in_claimed_space: bool

@dataclass
class GalacticState:
    year: int
    month: int
    game_version: str
    galaxy_size: int                    # number of stars
    difficulty: str
    crisis_active: bool
    crisis_type: Optional[str]
    war_in_heaven: bool
    galactic_market_founded: bool
    empires: list[EmpireSummary]        # all known empires (brief)
    player_empire_id: int
```

#### 6.1.4 Version Detection

The `meta` file contains the game version string (e.g., `version="4.3.0"`). The extraction layer uses this to select the correct field mapping schema. For MVP, only Stellaris 4.3.x is supported. If an unsupported version is detected, the parser raises a clear error with the detected version and supported range.

### 6.2 Game Knowledge Graph (`stellaris_advisor.gamedata`)

Parses Stellaris game definition files (from the game's install directory) into a structured dependency graph. This is a **static, deterministic** layer — no ML involved. These files use the same Clausewitz text format as save files, so the same parser can be reused.

#### 6.2.1 Source Files

| Source path | Content | Approximate scale |
|---|---|---|
| `common/technology/*.txt` | All technology definitions: ID, area, category, tier, cost, weight, prerequisites, unlock effects, weight modifiers by ethics/civics/traits | ~400 entries across ~15 files |
| `common/buildings/*.txt` | Building definitions: ID, cost, upkeep, production, jobs, prerequisites (tech, building, planet class, empire type), upgrade chains | ~100+ entries across ~10 files |
| `common/districts/*.txt` | District definitions: ID, resource output, housing, jobs, planet class availability, specialization slots and options (4.0+) | ~15–20 entries across ~5 files |
| `common/ship_sizes/*.txt` | Ship hull definitions: ID, hull points, sections, naval capacity cost, tech prerequisites | ~10 entries |
| `common/component_templates/*.txt` | Ship components: weapons, defenses, utilities; stats, costs, tech prerequisites | ~200+ entries |
| `common/edicts/*.txt` | Edict definitions: effects, costs (unity/influence), duration, prerequisites | ~30+ entries |
| `common/governments/civics/*.txt` | Civic definitions: effects on buildings/techs/resources, empire type restrictions | ~50+ entries |
| `common/governments/authorities/*.txt` | Authority definitions: allowed ethics, special mechanics | ~6 entries |
| `common/traits/*.txt` | Species trait definitions: resource output modifiers, habitability, other effects | ~50+ entries |
| `common/planet_classes/*.txt` | Planet class definitions: habitability base, district availability, special features | ~20+ entries |
| `common/pop_jobs/*.txt` | Job definitions: resource production per pop, conditions | ~80+ entries |
| `common/tradition_categories/*.txt` | Tradition tree definitions: ID, available traditions, adoption bonus, finisher bonus, empire type restrictions | ~14 trees |
| `common/traditions/*.txt` | Individual tradition definitions: ID, tree, effects, prerequisites, unity cost modifiers | ~98 entries |
| `common/ascension_perks/*.txt` | Perk definitions: ID, tier, prerequisites (prior perk count, techs, completed traditions), effects, mutual exclusions, empire type restrictions | ~36 entries |
| `common/megastructures/*.txt` | Megastructure definitions: ID, stages (with cost/time/output per stage), site requirements (star type restrictions), perk/tech prerequisites | ~15 entries |
| `common/policies/*.txt` | Policy definitions: ID, categories, option list, effects per option, faction approval impacts, prerequisites, 10-year cooldown | ~15 categories |
| `common/factions/*.txt` | Faction definitions: ethic association, issue list (binary conditions + approval delta), influence output formula, join conditions | ~9 standard factions |
| `common/council_positions/*.txt` | Council position definitions: required leader class, per-level bonus, empty penalty, civic-granted positions | ~6 positions |
| `common/agendas/*.txt` | Agenda definitions: continuous effect, launch effect, preparation cost (7000 base + empire size scaling), prerequisites (traditions, perks, ethics) | ~30+ entries |
| `common/leader_classes/*.txt` | Leader class definitions: available trait pools, veteran classes (3 per class), expertise fields (for scientists) | ~5 classes |
| `common/federation_types/*.txt` | Federation type definitions: per-level bonuses, law options, cohesion mechanics, experience thresholds | ~5 types |

#### 6.2.2 Core Data Structures

```python
@dataclass
class TechNode:
    id: str                             # e.g., "tech_destroyers"
    name: str                           # localized display name
    area: str                           # "physics" | "society" | "engineering"
    category: str                       # e.g., "voidcraft", "industry", "biology"
    tier: int                           # 0–5
    cost: int                           # base research cost (before scaling)
    weight: float                       # base draw weight
    is_rare: bool                       # rare tech flag (purple card)
    is_dangerous: bool                  # dangerous tech flag
    is_repeatable: bool                 # repeatable end-game tech
    is_starter: bool                    # always available at game start
    prerequisites: list[str]            # tech IDs required before this can appear
    # What this tech unlocks
    unlocks_buildings: list[str]        # building IDs
    unlocks_building_upgrades: list[str]# building upgrade paths
    unlocks_districts: list[str]        # district types or specializations
    unlocks_components: list[str]       # ship component IDs
    unlocks_ship_sizes: list[str]       # ship class IDs
    unlocks_edicts: list[str]           # edict IDs
    unlocks_traditions: list[str]       # tradition IDs
    unlocks_policies: list[str]         # policy options
    unlocks_decisions: list[str]        # planetary decision IDs
    # Draw weight modifiers
    weight_modifiers: list[WeightModifier]  # conditions that change draw weight
    # Empire type restrictions
    empire_type_allow: Optional[list[str]]  # if set, only these types can draw
    empire_type_block: Optional[list[str]]  # if set, these types cannot draw

@dataclass
class WeightModifier:
    """Represents a conditional modifier to a tech's draw weight."""
    condition_type: str                 # "ethic", "civic", "trait", "has_tech", "scientist_expertise", etc.
    condition_value: str                # e.g., "ethic_materialist", "civic_technocracy"
    weight_multiplier: float            # e.g., 1.25 for 25% boost, 0 for blocked

@dataclass
class BuildingDef:
    id: str                             # e.g., "building_alloy_foundry"
    name: str
    category: str                       # "resource", "military", "unity", "research", "refinery", etc.
    # Prerequisites
    tech_prerequisites: list[str]       # tech IDs required
    building_prerequisites: list[str]   # building IDs required (upgrade chains)
    planet_class_allow: Optional[list[str]]   # if set, only these planet classes
    planet_class_block: Optional[list[str]]   # if set, blocked on these classes
    empire_type_allow: Optional[list[str]]
    empire_type_block: Optional[list[str]]
    requires_capital_level: Optional[int]     # minimum capital building tier
    is_unique_planet: bool              # only 1 per planet
    is_unique_empire: bool              # only 1 per empire
    # Economics
    build_cost: dict[str, float]        # resource_id → amount
    build_time: int                     # days
    resource_production: dict[str, float]  # resource_id → monthly output
    resource_upkeep: dict[str, float]   # resource_id → monthly cost
    jobs_created: dict[str, int]        # job_id → count
    # Upgrade
    upgrades_to: Optional[str]          # building ID of upgrade target
    upgrades_from: Optional[str]        # building ID this upgrades from

@dataclass
class DistrictDef:
    id: str                             # e.g., "district_mining"
    name: str
    planet_class_availability: list[str]  # planet classes where this can be built
    resource_production: dict[str, float]
    resource_upkeep: dict[str, float]
    housing: int                        # housing provided
    jobs: dict[str, int]                # job_id → count
    build_cost: dict[str, float]
    build_time: int
    # Specializations (4.0+)
    specialization_slots: int           # 0, 1, or 2
    available_specializations: list[SpecializationDef]

@dataclass
class SpecializationDef:
    id: str
    name: str
    tech_prerequisites: list[str]
    extra_jobs: dict[str, int]          # additional jobs per district of this type
    extra_building_slots: int           # additional building slots unlocked

@dataclass
class JobDef:
    id: str                             # e.g., "metallurgist", "researcher", "farmer"
    name: str
    category: str                       # "specialist", "worker", "ruler"
    resource_production: dict[str, float]  # base output per pop working this job
    resource_upkeep: dict[str, float]
    output_modifiers: list[OutputModifier]  # conditions that change output (traits, techs, etc.)

@dataclass
class ShipClassDef:
    id: str
    name: str
    hull_points: int
    naval_capacity_cost: int
    tech_prerequisites: list[str]
    sections: list[ShipSectionDef]
    build_cost: dict[str, float]
    build_time: int

@dataclass
class PlanetClassDef:
    id: str                             # e.g., "pc_continental"
    name: str
    climate: str                        # "wet", "dry", "cold"
    habitability_base: float            # base habitability for matching preference
    available_districts: list[str]      # district type IDs buildable here
    special_features: list[str]         # e.g., guaranteed resource deposits
```

#### 6.2.3 The GameKnowledgeGraph Class

```python
class GameKnowledgeGraph:
    """Central registry of all game definitions. Loaded once from game files."""

    tech_dag: nx.DiGraph                # Nodes: tech IDs; edges: prerequisite → unlocked
    tech_defs: dict[str, TechNode]
    building_defs: dict[str, BuildingDef]
    district_defs: dict[str, DistrictDef]
    job_defs: dict[str, JobDef]
    ship_class_defs: dict[str, ShipClassDef]
    planet_class_defs: dict[str, PlanetClassDef]
    edict_defs: dict[str, EdictDef]
    civic_defs: dict[str, CivicDef]
    trait_defs: dict[str, TraitDef]

    # --- Query methods ---

    def reachable_techs(
        self,
        researched: set[str],
        empire_type: str,
        ethics: list[str],
    ) -> list[TechNode]:
        """
        Return techs whose:
        - All prerequisites are in `researched`
        - Tier is unlocked (≥6 techs from prior tier in same area)
        - Empire type is not blocked
        Not filtered by draw weight (that's stochastic at runtime).
        """

    def available_buildings(
        self,
        researched: set[str],
        planet_class: str,
        empire_type: str,
        existing_buildings: list[str],  # for uniqueness constraints
        capital_level: int,
    ) -> list[BuildingDef]:
        """
        Return buildings that can be built given:
        - Tech prerequisites met
        - Planet class allowed
        - Empire type allowed
        - Capital level sufficient
        - Uniqueness constraints not violated
        """

    def available_districts(
        self,
        planet_class: str,
        researched: set[str],
    ) -> list[DistrictDef]:
        """Return districts buildable on this planet class."""

    def tech_path_to(
        self,
        target_tech: str,
        researched: set[str],
    ) -> list[list[TechNode]]:
        """
        BFS from current researched set to target tech.
        Returns all shortest prerequisite chains.
        Each chain is ordered: [first tech to research, ..., target_tech].
        """

    def building_unlock_chain(
        self,
        target_building: str,
        researched: set[str],
    ) -> tuple[list[TechNode], BuildingDef]:
        """
        Return (techs_needed, building_def) to unlock a target building.
        techs_needed is the shortest path through the tech DAG.
        """

    def tech_tier_counts(
        self,
        researched: set[str],
        area: str,
    ) -> dict[int, int]:
        """Count researched techs per tier in a given area. Used for tier gating checks."""

    def downstream_unlocks(
        self,
        tech_id: str,
    ) -> dict[str, list]:
        """
        Return everything a tech unlocks:
        {
            "buildings": [...],
            "districts": [...],
            "components": [...],
            "ship_sizes": [...],
            "further_techs": [techs that have this as a prerequisite],
        }
        """

    def estimate_building_net_output(
        self,
        building: BuildingDef,
        empire_state: EmpireState,
    ) -> dict[str, float]:
        """
        Estimate the net resource output of a building:
        production - upkeep, adjusted for empire-wide modifiers
        (ethics bonuses, trait bonuses, stability multiplier).
        This is an approximation — doesn't model pop assignment.
        """
```

#### 6.2.4 Tech DAG Integrity Validation

On loading, the game knowledge graph runs automated integrity checks:

1. **Acyclicity:** The tech prerequisite graph must be a DAG (no cycles). Checked via topological sort.
2. **Referential integrity:** Every prerequisite reference must resolve to an existing tech ID.
3. **Tier consistency:** No tech at tier N should have a prerequisite at tier N+1 or higher.
4. **Building-tech linkage:** Every building's `tech_prerequisites` must reference valid tech IDs.
5. **Planet class coverage:** Every district's `planet_class_availability` must reference valid planet class IDs.

Failed checks raise descriptive errors with the offending IDs, since game file modding or DLC mismatches can cause inconsistencies.

### 6.3 Feature Engineering (`stellaris_advisor.features`)

Transforms parsed entities into model-ready feature vectors.

#### 6.3.1 Empire-Level Features (~107 features)

**Resource rates (20 features):**
- Net income for each of 10 trackable resources: energy, minerals, food, consumer_goods, alloys, unity, influence, physics_research, society_research, engineering_research
- Net income total (sum of core 7 weighted by market value)
- Months-until-depletion for each resource in deficit (0 if positive income)
- Resource stockpile utilization: stockpile / storage_cap for energy, minerals, food, alloys, consumer_goods

**Economic efficiency ratios (8 features):**
- Alloy production per pop
- Total research output per pop
- Energy production per generator district
- Mineral production per mining district
- Alloy production per industrial district
- Consumer goods production per industrial district
- Research per research lab building
- Unity per pop

**Expansion metrics (8 features):**
- Total systems owned
- Total planets colonized
- Average planet size
- Average district utilization (districts_used / total_district_cap) across all planets
- Average building slot utilization across all planets
- Border-to-interior system ratio (systems with hostile neighbor adjacency / total systems)
- Average system resource value (sum of minable space resources / system count)
- Expansion velocity (systems gained per decade, from temporal data)

**Military posture (8 features):**
- Fleet power (total)
- Naval capacity utilization ratio (used / max)
- Fleet power relative to strongest neighbor (own / max_neighbor)
- Fleet power relative to average neighbor
- Fleet composition: % corvettes, % destroyers, % cruisers, % battleships+titans
- Starbase defense power (total across all starbases)

**Tech progression (12 features):**
- Total techs researched
- Per-area counts: physics, society, engineering
- Per-tier distribution: count at T1, T2, T3, T4, T5
- Research speed per area (monthly output)
- Milestone tech flags (binary): has_cruisers, has_battleships, has_titans, has_megastructures, has_repeatables
- Tech frontier size: number of techs reachable on next draw per area

**Planet portfolio (10 features):**
- Planet count by climate type (wet, dry, cold, special)
- Average habitability across owned planets
- Total district slots remaining (all planets)
- Total building slots remaining (all planets)
- Number of specialized worlds (forge, tech, unity, etc.)
- Has ecumenopolis (binary)
- Has ring world (binary)
- Has habitat(s) (count)

**Pop metrics (6 features):**
- Total pops
- Average pop happiness / approval
- Unemployment rate (unemployed pops / total pops)
- Species diversity (number of distinct species)
- Faction approval spread (max - min faction approval)
- Pop growth rate (from temporal data)

**Diplomatic context (6 features):**
- Number of rivals
- Number of defensive pacts
- Is in federation (binary)
- Federation total fleet power (if applicable)
- Number of subjects (vassals)
- Is subject (binary)

**Empire archetype (4 features, one-hot):**
- is_gestalt_hive
- is_gestalt_machine
- is_megacorp
- is_standard

**Traditions & Ascension (8 features):**
- traditions_unlocked_count (total individual traditions unlocked)
- tradition_trees_completed_count (completed trees — gates perk slot availability)
- ascension_perks_count (perks taken, 0–8)
- has_ascension_path (binary: has any path perk been taken?)
- ascension_path_type (categorical: synthetic/cybernetic/psionic/bio/none)
- unity_income (monthly unity production)
- tradition_cost_efficiency (unity_income / next_tradition_cost — measures how fast traditions are being unlocked)
- ascension_situation_progress (0.0–1.0 if active, 0.0 if not)

**Leaders & Council (5 features):**
- head_of_research_level (1–10; directly adds +2% research speed per level)
- council_average_level (average skill level across all filled positions)
- council_legitimacy (0–100; derived from faction approval; multiplies agenda speed)
- has_active_agenda (binary)
- agenda_progress_pct (0.0–1.0; how close the current agenda is to launching)

**Factions & Influence (6 features):**
- weighted_avg_faction_approval (approval weighted by faction support — captures overall political health)
- max_faction_approval (best faction — indicates which ethic alignment is strongest)
- min_faction_approval (worst faction — indicates policy conflicts)
- influence_income_total (total monthly influence from all sources)
- influence_from_factions (faction contribution to influence — the primary engine)
- power_projection_score (fleet-to-empire ratio for influence; max 2.0/month)

**Galactic Community (2 features):**
- galcom_diplomatic_weight (determines voting power and council eligibility)
- is_galcom_council_member (binary — top 3 by weight get council powers)

**Federation (2 features):**
- federation_level (0 if none, 1–5 otherwise; higher = stronger bonuses)
- federation_fleet_power (0 if none; shared fleet power)

**Megastructures (4 features):**
- megastructures_completed_count (total completed megastructures)
- megastructures_under_construction (0, 1, or 2 with Master Builders)
- has_dyson_sphere (binary — marks the energy economic phase transition)
- has_matter_decompressor (binary — marks the mineral economic phase transition)

#### 6.3.2 System-Level Features (~12 features per system, for colonization scoring)

- Habitability of best candidate planet (for player's primary species, accounting for traits and preferences)
- Planet size (of best candidate)
- Total district slots by category (generator, mining, farming, industrial)
- Resource deposit value: weighted sum of all planetary resource deposits (energy weight=1.0, mineral=1.0, rare_crystal=3.0, exotic_gas=3.0, volatile_motes=3.0)
- Hyperlane degree (number of connections — higher = more accessible but less defensible)
- Chokepoint score (betweenness centrality in the hyperlane graph — high = strategically valuable)
- Distance to empire capital (shortest path in hyperlane hops)
- Distance to nearest hostile border (hyperlane hops)
- Adjacent-system resource potential (sum of minable resources in all directly connected systems)
- Planet class match score (1.0 if same climate as majority of existing colonies, 0.5 if different climate, 0.0 if special class)
- Is in claimed space (binary — does the player already have a claim?)
- System already has starbase/outpost (binary — ownership established?)

#### 6.3.3 Temporal Features (from save series, ~8 features)

Computed by comparing the current save against N previous saves from the same game:

- Energy income Δ (current - N saves ago)
- Mineral income Δ
- Alloy income Δ
- Research income Δ (total across 3 areas)
- Expansion velocity (systems gained over last N saves)
- Tech velocity (techs researched over last N saves)
- Fleet power Δ
- Stall detection flag: binary, set to 1 if alloy income Δ and research income Δ are both < threshold for 3+ consecutive save intervals

Default: N = 5 saves (~5 years at yearly autosave; adjustable in config).

### 6.4 Data Pipeline (`stellaris_advisor.data`)

#### 6.4.1 Data Sources

1. **Player saves** — user's own save files, parsed chronologically from a game folder. Typically 20–100+ saves per playthrough (autosave every 1–12 months in-game). Located at `Documents/Paradox Interactive/Stellaris/save games/<game_name>/`.

2. **Synthetic observer runs** — scripted batch runs using Stellaris console `observe` mode with autosave enabled at short intervals.

#### 6.4.2 Synthetic Data Generation Pipeline

```
[generate_synthetic.sh]
    For run_id in 1..N:
        1. Generate randomized game settings:
           - Galaxy size: randomly sample from {400, 600, 800, 1000}
           - AI empires: randomly sample from {8, 12, 16, 20, 24, 30}
           - Difficulty: randomly sample from {ensign, captain, admiral, grand_admiral}
           - Crisis strength: randomly sample from {1.0, 2.0, 3.0, 5.0}
           - Advanced AI starts: randomly sample from {0, 1, 2, 4}
           - Mid-game start year: 2250 or 2275
           - End-game start year: 2350 or 2400
           - Victory year: 2500
           - AI aggression: randomly sample from {low, normal, high}
           - Hyperlane density: randomly sample from {0.5, 0.75, 1.0, 1.25}
        2. Write settings to a game configuration file
        3. Launch Stellaris:
           - Pass settings file
           - Execute console commands on startup:
             `observe` (enter observer mode)
             `autosave_interval 12` (save every 12 months)
             `game_speed 5` (maximum speed)
           - Optionally use --headless or minimized window
        4. Wait for game to reach year 2500 or end
           (estimated: 20–60 minutes per run depending on galaxy size)
        5. Copy all .sav files from save folder to:
           data/raw/{run_id}/
        6. Log metadata:
           {run_id, settings, start_time, end_time, num_saves, final_year}

[generate_synthetic.py]
    - Python orchestrator that wraps the shell script
    - Monitors save folder for new files
    - Handles Stellaris crashes (restart and continue)
    - Tracks progress across runs
    - Validates that saves were generated correctly (checks year progression)
```

**Target: 50–100 full game runs** for MVP, yielding:
- ~300 saves per run (years 2200–2500, monthly autosave = 300 saves; yearly = 300 saves)
- ~15 empires per game (average)
- = ~225,000–450,000 (empire, year) training rows at yearly resolution
- = ~15,000–30,000 at 12-month resolution

#### 6.4.3 Ingestion Pipeline

```
stellaris-advisor ingest <save_folder> [--output-dir data/processed/]

For each .sav file in the folder (sorted by in-game date):
    1. Parse via Clausewitz parser → raw dict
    2. Extract all empire states, planet states, system states
    3. Compute empire-level features for each empire
    4. Compute system-level features for uncolonized systems
    5. If previous save from same game exists in memory:
       compute temporal features (deltas)
    6. Write to Parquet:
       - empire_features.parquet: partitioned by game_id/year
       - system_features.parquet: partitioned by game_id/year
       - game_metadata.parquet: one row per game run
```

#### 6.4.4 Storage Schema

**empire_features.parquet:**

| Column | Type | Description |
|---|---|---|
| game_id | string | Unique identifier for the game run |
| year | int | In-game year |
| empire_id | int | Empire index within the game |
| empire_name | string | Display name |
| empire_type | string | Archetype (standard/hive/machine/megacorp) |
| authority | string | Authority type |
| ethics | string[] | Ethics list |
| feature_* | float | All ~80 empire-level features (one column each) |
| temporal_* | float | All ~8 temporal features (null for first save in series) |
| label_score_delta_50 | float | Score delta at T+50 (null if <50 years remain, filled during labeling) |
| label_final_rank | int | Final rank at game end (filled during labeling) |
| label_survived_endgame | bool | Survived to victory year (filled during labeling) |

**system_features.parquet:**

| Column | Type | Description |
|---|---|---|
| game_id | string | |
| year | int | |
| system_id | int | |
| system_name | string | |
| owner_empire_id | int | -1 if unowned |
| feature_* | float | All ~12 system-level features |
| label_colonized_year | int | Year this system was colonized (null if never) |
| label_colonized_by_rank | int | Final rank of empire that colonized it |

**game_metadata.parquet:**

| Column | Type | Description |
|---|---|---|
| game_id | string | |
| galaxy_size | int | |
| num_empires | int | |
| difficulty | string | |
| crisis_strength | float | |
| crisis_type | string | |
| final_year | int | |
| stellaris_version | string | |
| generation_settings | json | Full settings blob |

#### 6.4.5 Labeling Strategy

**Empire outcome label (Model A):**

```
score(empire, year) = w1 * norm(fleet_power)
                    + w2 * norm(tech_count)
                    + w3 * norm(total_resource_income)
                    + w4 * norm(pop_count)
```

- Each component normalized across all empires in the same game at the same year using min-max scaling.
- Default weights: w1=0.25, w2=0.25, w3=0.30, w4=0.20. Configurable in `config.yaml`.
- Primary label: `label_score_delta_50 = score(empire, year+50) - score(empire, year)`.
- Secondary labels: `label_final_rank` (ordinal rank at game end), `label_survived_endgame` (binary: empire exists at victory year).
- For (empire, year) pairs where year+50 > final game year: use game end score instead of year+50.

**Colonization label (Model B):**

For each system in each game:
- If colonized: `label = (max_year - year_colonized) * colonizer_final_score`. Earlier colonization by stronger empires = higher label.
- If never colonized: `label = 0`.
- Normalized within each game to [0, 1] range.

### 6.5 ML Models (`stellaris_advisor.models`)

#### 6.5.1 Model A: Empire State Valuator

**Task:** Given empire-level features at year T, predict:
- (Primary) `score_delta_50`: continuous regression target — change in relative power score over the next 50 years.
- (Secondary) `survived_endgame`: binary classification — whether the empire survives to the victory year.

**Architecture:** LightGBM gradient-boosted decision tree ensemble.

**Training configuration:**

```yaml
model_a:
  task: regression  # primary task
  objective: regression_l2
  metric: [rmse, mae]
  num_leaves: 63
  learning_rate: 0.05
  n_estimators: 1000
  early_stopping_rounds: 50
  feature_fraction: 0.8
  bagging_fraction: 0.8
  bagging_freq: 5
  min_child_samples: 20
  lambda_l1: 0.1
  lambda_l2: 0.1
  categorical_features: [empire_type, authority]
  # Secondary binary classifier trained separately
  secondary_task: binary
  secondary_objective: binary_logloss
```

**Hyperparameter search space (Optuna):**

| Parameter | Range | Distribution |
|---|---|---|
| num_leaves | [15, 127] | int |
| learning_rate | [0.01, 0.3] | log-uniform |
| n_estimators | [200, 2000] | int |
| feature_fraction | [0.5, 1.0] | uniform |
| bagging_fraction | [0.5, 1.0] | uniform |
| min_child_samples | [5, 50] | int |
| lambda_l1 | [1e-3, 10] | log-uniform |
| lambda_l2 | [1e-3, 10] | log-uniform |

**Feature importance analysis:** After training, compute SHAP values for all features. Store the top-20 features by mean |SHAP| value. These are used in the CLI output to explain recommendations.

**Model serialization:** Save trained model as `.txt` (LightGBM native format) + a companion `.json` with metadata (training date, dataset stats, feature names, hyperparams, eval metrics).

#### 6.5.2 Model B: System Colonization Scorer

**Task:** Given system-level features, rank uncolonized systems by colonization priority.

**Architecture:** LightGBM with LambdaMART ranking loss.

**Training configuration:**

```yaml
model_b:
  task: ranking
  objective: lambdarank
  metric: ndcg
  ndcg_eval_at: [5, 10]
  num_leaves: 31
  learning_rate: 0.05
  n_estimators: 500
  early_stopping_rounds: 30
  group_column: game_id_year  # ranking groups = all systems in same game at same year
```

**Alternative approach (if ranking data is too sparse):** Train as regression on the colonization label directly, then rank by predicted score at inference time.

#### 6.5.3 Benchmark Data Store

For the `benchmark` CLI command, pre-compute percentile distributions from the training data:

```
For each (year_bucket, difficulty, empire_type):
    Store: {
        alloy_income: {p25, p50, p75, p90},
        research_total: {p25, p50, p75, p90},
        fleet_power: {p25, p50, p75, p90},
        tech_count: {p25, p50, p75, p90},
        pop_count: {p25, p50, p75, p90},
        system_count: {p25, p50, p75, p90},
    }
```

Year buckets: every 25 years (2200, 2225, 2250, ..., 2500). Stored as a compact JSON or Parquet file shipped with the trained models.

### 6.6 Strategic Planner (`stellaris_advisor.planner`)

A **hybrid rule-based + ML-scored** system. Uses the Game Knowledge Graph (§6.2) for structured reasoning and Model A (§6.5) for value estimation.

#### 6.6.1 Design Rationale

The full action space is combinatorially intractable for pure ML. But the dependency structure of the game is **known and static** — it lives in the game definition files. The planner exploits this by:

1. Using **game rules** for what's possible (prerequisite chains, planet/empire constraints)
2. Using **game formulas** for deterministic outcomes (building X produces Y alloys/month, job Z produces W resources)
3. Using **ML** only for what's uncertain (long-term empire trajectory valuation)

#### 6.6.2 Action Space Definition

The planner operates on **macro-actions** — coarse-grained decisions that abstract away low-level micromanagement. Each macro-action is a self-contained recommendation.

**ECONOMY actions:**

| Action template | Parameters | Constraint checks |
|---|---|---|
| Build district on planet | (planet_id, district_type) | planet class allows district type; district slots available; empire can afford build cost |
| Build building on planet | (planet_id, building_id) | tech prerequisites met; planet class allowed; empire type allowed; building slots available; uniqueness constraints satisfied; capital level sufficient; empire can afford build cost |
| Upgrade building on planet | (planet_id, building_id, upgrade_target) | upgrade path exists in building defs; tech prerequisite for upgrade met; empire can afford cost |
| Set district specialization | (planet_id, district_type, specialization_id) | specialization tech prerequisites met; specialization slot available |

**TECH actions:**

| Action template | Parameters | Constraint checks |
|---|---|---|
| Prioritize tech for research | (tech_id) | tech is on reachable frontier (prerequisites met, tier unlocked, empire type allowed); tech is in same area as an active research slot |
| Research tech path toward goal | (goal_tech_id) | goal tech exists; at least one valid prerequisite chain exists from current state |

**EXPANSION actions:**

| Action template | Parameters | Constraint checks |
|---|---|---|
| Colonize system | (system_id) | system has habitable planet with habitability > 0; system is reachable (adjacent to owned space or has claim); player has or can afford colony ship + influence cost |
| Build starbase in system | (system_id) | system is unclaimed or claimed by player; player has starbase capacity |
| Upgrade starbase | (system_id, target_level) | tech prerequisites for level met; can afford cost |

**MILITARY actions:**

| Action template | Parameters | Constraint checks |
|---|---|---|
| Build ships to fill naval capacity | (ship_class, count) | tech prerequisites for ship class met; shipyard available; can afford alloy cost; naval capacity headroom > 0 |
| Upgrade fleet components | — | newer components available via researched techs than currently equipped |
| Place claims for war | (system_id, count) | target systems identified; influence budget sufficient (75+ per claim); casus belli available |

**TRADITION actions:**

| Action template | Parameters | Constraint checks |
|---|---|---|
| Open tradition tree | (tree_id) | tree not already opened; tree available for empire type (e.g., Politics requires Galactic Community); unity budget can afford adoption |
| Unlock next tradition in tree | (tree_id, tradition_id) | tree is opened; tradition prerequisites within tree met; unity budget sufficient |
| Complete tradition tree | (tree_id) | all traditions in tree unlocked (or enough unity to unlock remaining); completing unlocks a perk slot |

The tradition recommender computes ROI differently from buildings/techs. Tradition bonuses are typically percentage modifiers that compound over time. The lifetime value of a tradition is: `bonus_magnitude × time_remaining_in_game × affected_resource_base`. Front-loaded adoption bonuses (e.g., Discovery's Map the Stars edict) have higher ROI when taken earlier. The recommender should also consider: does completing this tree unlock a perk slot that enables a high-value perk (e.g., completing any tree when Galactic Wonders is available and Mega-Engineering is researched)?

**ASCENSION PERK actions:**

| Action template | Parameters | Constraint checks |
|---|---|---|
| Take Ascension Perk | (perk_id) | perk slot available; tier prerequisite met (sufficient prior perks); tech prerequisites met (if any); mutual exclusion check (no conflicting perk taken); empire type allowed |
| Take Ascension Path perk | (path_perk_id) | 2+ prior perks taken; empty tradition slot available (for the path tradition tree); no other path perk taken; empire type and origin compatible |

Perk recommendation uses time-discounted lifetime value: `perk_effect_value × remaining_game_years / (1 + discount_rate × years_until_effect_realized)`. For Technological Ascendancy (+10% research speed), effect is immediate and compounds, so it should almost always be the first perk. For Galactic Wonders, effect is delayed by megastructure build time (~10 years), so the discount is larger but the eventual output is massive. The recommender must also check whether the perk's prerequisites are met — recommending Galactic Wonders when Mega-Engineering isn't researched yet should instead recommend the tech path to Mega-Engineering first.

**MEGASTRUCTURE actions:**

| Action template | Parameters | Constraint checks |
|---|---|---|
| Begin megastructure construction | (megastructure_type, system_id) | perk prerequisite met (Galactic Wonders for most); tech prerequisite met; site requirements met (star type for Dyson/Decompressor, black hole for Decompressor); no other megastructure under construction (unless Master Builders perk); alloy budget sufficient for first stage; influence budget sufficient |
| Upgrade megastructure stage | (megastructure_id) | previous stage complete; alloy budget for next stage |

Megastructure ROI dwarfs planetary development in late game. A Dyson Sphere (+4000 energy/mo when complete) has an ROI that exceeds building generator districts on every planet in most empires. The recommender should flag megastructure readiness as a distinct late-game trigger: when `has_mega_engineering_tech AND has_galactic_wonders_perk AND suitable_site_exists AND alloy_income > threshold`, megastructure construction should be the top recommendation. System scoring should tag black holes and isolated single stars as megastructure candidates alongside colonization candidates.

**COUNCIL & AGENDA actions:**

| Action template | Parameters | Constraint checks |
|---|---|---|
| Switch agenda | (agenda_id) | agenda prerequisites met (ethics, traditions, perks); agenda not on 30-year cooldown; warning if current agenda has significant progress that would be lost |
| Reassign leader to council position | (leader_id, position_id) | leader class matches position requirement; leader is not currently in a critical field assignment (e.g., only admiral on an active fleet) |

The council recommender focuses on two things: (1) Is the Head of Research's expertise aligned with the player's tech goals? If the player needs Voidcraft techs (battleships, megastructures) but the Head of Research has Biology expertise, recommend reassigning a Voidcraft-expertise scientist. (2) Is the current agenda optimal for the detected bottleneck? If alloy production is the bottleneck and the current agenda is Chart the Unknown (exploration bonus), recommend switching to an economy-relevant agenda. Agenda switching should account for progress loss — don't recommend switching if the current agenda is >70% prepared unless the alternative is significantly higher value.

**FACTION MANAGEMENT actions:**

| Action template | Parameters | Constraint checks |
|---|---|---|
| Change policy | (policy_category, new_option) | policy not on 10-year cooldown; new option available for empire type; estimate faction approval delta across all factions |
| Promote faction | (faction_id) | costs unity; faction exists with ≥5 pops |
| Declare rival | (empire_id) | not already rival; target empire known; provides +0.5 influence/month per rival (max 3) |

The faction recommender activates when influence is identified as the binding constraint. It computes: for each possible policy change, what is the net influence delta? This requires: (1) computing which faction issues each policy option fulfills/violates, (2) estimating the approval change per faction, (3) translating approval changes to influence output changes via `Δinfluence = Σ(faction_support × Δapproval × influence_modifier)`. The recommender should also flag policy changes that have negative side effects (e.g., changing AI Policy to "Outlawed" would improve Spiritualist faction approval but block robot pop assembly — the recommender should note this trade-off explicitly).

#### 6.6.3 Core Algorithm

```
FUNCTION analyze(empire_state, game_state, game_graph, model_a, model_b) -> Report:

    // Step 1: Compute reachable frontier
    reachable_techs = game_graph.reachable_techs(
        empire_state.researched_techs, empire_state.authority, empire_state.ethics)
    
    For each planet in empire_state.owned_planets:
        available_buildings[planet.id] = game_graph.available_buildings(
            empire_state.researched_techs, planet.planet_class,
            empire_state.authority, planet.buildings, planet.capital_level)
        available_districts[planet.id] = game_graph.available_districts(
            planet.planet_class, empire_state.researched_techs)

    colony_candidates = extract_colony_candidates(game_state, empire_state)
    colony_scores = model_b.predict(colony_candidates)  // ranked list

    // Step 2: Generate candidate macro-actions
    candidates = []

    // ECONOMY: buildings
    For each planet in empire_state.owned_planets:
        if planet.building_slots_used < planet.building_slots_total:
            For each building in available_buildings[planet.id]:
                net_output = game_graph.estimate_building_net_output(building, empire_state)
                roi = sum(net_output.values()) / max(building.build_time / 30, 1)
                candidates.append(EconomyAction(
                    type="build_building",
                    planet=planet, building=building,
                    estimated_net_output=net_output,
                    estimated_roi=roi,
                    time_days=building.build_time))

    // ECONOMY: districts
    For each planet in empire_state.owned_planets:
        For each district in available_districts[planet.id]:
            if planet.district_slots[district.id] > planet.districts.get(district.id, 0):
                net_output = estimate_district_net_output(district, empire_state)
                roi = sum(net_output.values()) / max(district.build_time / 30, 1)
                candidates.append(EconomyAction(
                    type="build_district",
                    planet=planet, district=district,
                    estimated_net_output=net_output,
                    estimated_roi=roi,
                    time_days=district.build_time))

    // TECH: frontier techs with downstream value
    For each tech in reachable_techs:
        research_months = tech.cost / empire_state.research_speed[tech.area]
        downstream = game_graph.downstream_unlocks(tech.id)

        // Compute downstream building value across planet portfolio
        downstream_value = 0
        dependency_chain = [tech.name]
        For each building_id in downstream["buildings"]:
            building = game_graph.building_defs[building_id]
            // Find best planet for this building
            For each planet in empire_state.owned_planets:
                if game_graph.can_build(building, planet, empire_state):
                    net = game_graph.estimate_building_net_output(building, empire_state)
                    downstream_value += sum(net.values())
                    dependency_chain.append(f"→ Build {building.name} on {planet.name}")
                    break  // one placement per building for estimation

        // Milestone bonus for key techs
        milestone_bonus = 0
        if tech.id in MILESTONE_TECHS:
            milestone_bonus = MILESTONE_WEIGHTS[tech.id]

        chained_roi = (downstream_value + milestone_bonus) / max(research_months, 1)
        candidates.append(TechAction(
            tech=tech,
            research_months=research_months,
            downstream_value=downstream_value,
            chained_roi=chained_roi,
            dependency_chain=dependency_chain))

    // EXPANSION: top colony candidates
    For top_n in colony_scores[:5]:
        system = top_n.system
        planet = top_n.best_planet
        time_to_colony = estimate_colonization_time(empire_state, system)
        long_term_output = estimate_colony_output(planet, game_graph, empire_state)
        candidates.append(ExpansionAction(
            system=system, planet=planet,
            colony_score=top_n.score,
            time_estimate=time_to_colony,
            projected_output=long_term_output))

    // MILITARY
    naval_utilization = empire_state.naval_capacity_used / max(empire_state.naval_capacity_max, 1)
    if naval_utilization < 0.7:
        headroom = empire_state.naval_capacity_max - empire_state.naval_capacity_used
        best_class = select_best_available_ship_class(empire_state, game_graph)
        count = headroom // game_graph.ship_class_defs[best_class].naval_capacity_cost
        candidates.append(MilitaryAction(
            type="build_ships", ship_class=best_class, count=count,
            alloy_cost=count * game_graph.ship_class_defs[best_class].build_cost["alloys"],
            fleet_power_gain=estimate_fleet_power_gain(best_class, count, game_graph)))

    max_neighbor_power = max(e.fleet_power for e in game_state.empires if e.id != empire_state.id)
    if max_neighbor_power > 1.5 * empire_state.fleet_power:
        candidates.append(MilitaryAlert(
            type="threat_warning",
            threat_empire=get_strongest_neighbor(game_state, empire_state),
            power_ratio=max_neighbor_power / max(empire_state.fleet_power, 1)))

    // TRADITIONS
    For each tree in game_graph.tradition_trees:
        if tree.id not in empire_state.tradition_trees_opened:
            if tree.available_for(empire_state):
                adoption_value = estimate_tradition_bonus_lifetime_value(
                    tree.adoption_bonus, empire_state, game_state.year)
                unity_cost = compute_tradition_cost(empire_state)
                candidates.append(TraditionAction(
                    type="open_tree", tree=tree,
                    adoption_bonus=tree.adoption_bonus,
                    estimated_lifetime_value=adoption_value,
                    unity_cost=unity_cost,
                    unlocks_perk_slot=False))
        elif tree.id not in empire_state.tradition_trees_completed:
            remaining = tree.remaining_traditions(empire_state.traditions_unlocked)
            finisher_value = estimate_tradition_bonus_lifetime_value(
                tree.finisher_bonus, empire_state, game_state.year)
            perk_slot_value = estimate_next_perk_value(empire_state, game_graph)
            candidates.append(TraditionAction(
                type="complete_tree", tree=tree,
                estimated_lifetime_value=finisher_value + perk_slot_value,
                unity_cost=sum(compute_tradition_cost(empire_state) for _ in remaining),
                unlocks_perk_slot=True))

    // ASCENSION PERKS
    if empire_state.ascension_perk_slots_available > 0:
        eligible_perks = game_graph.eligible_perks(
            taken=empire_state.ascension_perks,
            researched=empire_state.researched_techs,
            traditions_completed=empire_state.tradition_trees_completed,
            empire_type=empire_state.authority)
        For each perk in eligible_perks:
            lifetime_value = estimate_perk_lifetime_value(
                perk, empire_state, game_state.year, game_graph)
            candidates.append(PerkAction(
                perk=perk, lifetime_value=lifetime_value,
                is_ascension_path=perk.is_ascension_path,
                is_megastructure_enabler=perk.id in MEGASTRUCTURE_PERKS))

    // MEGASTRUCTURES
    if empire_state.megastructures_under_construction < game_graph.max_concurrent(empire_state):
        for mega_type in game_graph.available_megastructures(empire_state):
            sites = find_megastructure_sites(mega_type, game_state, empire_state)
            if sites:
                best_site = sites[0]
                total_cost = sum(stage.alloy_cost for stage in mega_type.stages)
                total_time = sum(stage.build_days for stage in mega_type.stages)
                final_output = mega_type.stages[-1].output
                roi = sum(final_output.values()) * (2500 - game_state.year) / total_cost
                candidates.append(MegastructureAction(
                    type=mega_type, site=best_site,
                    total_alloy_cost=total_cost,
                    total_build_months=total_time // 30,
                    final_output=final_output,
                    roi=roi))

    // COUNCIL & AGENDA
    if empire_state.current_agenda:
        bottleneck = detect_bottleneck(empire_state, benchmark_data)
        better_agenda = find_agenda_for_bottleneck(bottleneck, empire_state, game_graph)
        if better_agenda and better_agenda.id != empire_state.current_agenda.type:
            progress_loss = empire_state.current_agenda.progress
            if progress_loss < 0.7:  // don't recommend switching if >70% prepared
                candidates.append(AgendaAction(
                    type="switch_agenda", new_agenda=better_agenda,
                    progress_loss=progress_loss,
                    estimated_benefit=better_agenda.launch_effect_value))

    // Check Head of Research expertise alignment
    desired_expertise = determine_desired_expertise(empire_state, game_graph)
    current_expertise = get_head_of_research_expertise(empire_state)
    if current_expertise != desired_expertise:
        better_scientist = find_scientist_with_expertise(
            desired_expertise, empire_state.leaders)
        if better_scientist:
            candidates.append(CouncilAction(
                type="reassign_head_of_research",
                new_leader=better_scientist,
                expertise_match=desired_expertise,
                estimated_tech_draw_improvement=0.25))  // 25% weight bonus

    // FACTION MANAGEMENT (when influence is the bottleneck)
    if detect_bottleneck(empire_state, benchmark_data) == "influence_income":
        for policy_cat, options in game_graph.policy_options(empire_state):
            current = empire_state.policies[policy_cat]
            for option in options:
                if option != current and not policy_on_cooldown(empire_state, policy_cat):
                    approval_deltas = compute_faction_approval_deltas(
                        policy_cat, option, empire_state, game_graph)
                    influence_delta = compute_influence_delta(approval_deltas, empire_state)
                    side_effects = identify_policy_side_effects(
                        policy_cat, option, empire_state, game_graph)
                    if influence_delta > 0:
                        candidates.append(FactionAction(
                            type="change_policy",
                            policy=policy_cat, new_option=option,
                            approval_deltas=approval_deltas,
                            influence_delta=influence_delta,
                            side_effects=side_effects))

        // Recommend declaring rivals if < 3 active
        rival_count = len(empire_state.rivals)
        if rival_count < 3:
            candidates.append(FactionAction(
                type="declare_rival",
                influence_gain=0.5 * (3 - rival_count)))

    // Step 3: Score each candidate via Model A
    current_features = extract_empire_features(empire_state)
    current_score = model_a.predict(current_features)

    For each candidate in candidates:
        modified_features = simulate_feature_delta(current_features, candidate)
        projected_score = model_a.predict(modified_features)
        candidate.model_a_delta = projected_score - current_score

    // Step 4: Hybrid ranking
    For each candidate in candidates:
        candidate.final_score = (
            ALPHA * candidate.model_a_delta +
            BETA * normalize(candidate.estimated_roi)
        )
        // ALPHA default: 0.6, BETA default: 0.4 — configurable

    candidates.sort(key=lambda c: c.final_score, reverse=True)

    // Step 5: Assemble report
    report = Report(
        empire_health=compute_health_dashboard(empire_state, benchmark_data),
        recommendations=candidates[:7],  // increased from 5 to accommodate more categories
        tech_path_insight=auto_detect_bottleneck_and_suggest_path(
            empire_state, game_graph, model_a, benchmark_data),
        tradition_perk_path=compute_tradition_perk_recommendations(
            empire_state, game_graph, game_state.year),
        faction_influence_summary=compute_faction_summary(
            empire_state, game_graph),
        alerts=generate_alerts(empire_state, game_state, candidates),
    )
    return report
```

#### 6.6.4 Delta Simulation

The `simulate_feature_delta` function modifies the current feature vector to approximate the state after an action:

| Action type | Feature modifications |
|---|---|
| Build building | Adjust relevant resource income/expense features by building's net output; increment building_slots_used |
| Build district | Adjust resource income features; adjust housing; adjust district_utilization |
| Research tech | Increment tech_count; increment per-area and per-tier counts; update milestone flags if applicable; adjust tech_frontier_size |
| Colonize system | Increment system_count, planet_count; adjust expansion metrics; add estimated resource contribution |
| Build ships | Increment fleet_power by estimated gain; adjust naval_capacity_used; adjust alloy expense |
| Open tradition tree | Increment traditions_unlocked_count; apply adoption bonus effects to relevant resource/modifier features |
| Complete tradition tree | Increment tradition_trees_completed_count; apply finisher bonus; increment ascension_perk_slots_available |
| Take ascension perk | Increment ascension_perks_count; apply perk effects (e.g., Technological Ascendancy: +10% to all research features); set has_ascension_path if path perk |
| Begin megastructure | Increment megastructures_under_construction; adjust alloy expense; project output at completion into resource features (discounted by build time) |
| Switch agenda | Update agenda_type feature; estimate agenda launch effect on relevant features |
| Reassign Head of Research | Update head_of_research_level and expertise features; adjust estimated tech draw weights |
| Change policy | Recompute faction approval features based on new issue fulfillment; adjust influence_income_total and influence_from_factions |
| Declare rival | Adjust influence_income_total by +0.5 per new rival |

**This is an approximation.** It does not model second-order effects (pop growth, job reassignment, stability changes, trade cost changes). The approximation is acceptable because Model A was trained on real game data that implicitly captures these dynamics — the delta simulation just needs to be directionally correct for ranking purposes.

#### 6.6.5 Tech Path Optimizer

```python
def optimize_tech_path(
    goal: str,
    empire_state: EmpireState,
    game_graph: GameKnowledgeGraph,
    model_a: EmpireValuator,
) -> TechPathResult:
    """
    Goals can be:
    - A specific tech ID: "tech_battleships"
    - A keyword: "battleships", "megastructures", "synthetic_ascension"
    - A resource target: "max_alloy_income", "max_research"

    Algorithm:
    1. Resolve goal to target tech(s):
       - If specific tech ID: target = [that tech]
       - If keyword: lookup in GOAL_TECH_MAP → set of relevant techs
       - If resource target: identify buildings with highest output for that
         resource → trace back to their tech prerequisites → target = those techs

    2. For each target tech:
       a. If already researched: skip
       b. Compute prerequisite chain via BFS on tech_dag from current state
       c. For each chain:
          - Compute total_research_cost = sum(tech.cost for tech in chain)
          - Compute total_time = sum(tech.cost / research_speed[tech.area] for tech in chain)
          - Compute downstream_value:
            For each tech in chain:
              For each building/district it unlocks:
                For each planet in empire where that building/district is buildable:
                  value += estimated_net_output
          - Compute time_discounted_value = downstream_value / (1 + discount_rate * total_time)
       d. Rank chains by time_discounted_value

    3. Return best chain with step-by-step breakdown:
       [TechPathStep(tech, months, unlocks, cumulative_value), ...]
    """

GOAL_TECH_MAP = {
    "battleships": ["tech_battleships"],
    "cruisers": ["tech_cruisers"],
    "titans": ["tech_titans"],
    "megastructures": ["tech_mega_engineering"],
    "synthetic_ascension": ["tech_synthetic_workers", "tech_synthetic_evolution"],
    "psionic_ascension": ["tech_psionic_theory", "tech_telepathy", "tech_psi_jump_drive"],
    "habitats": ["tech_habitat_1"],
    "gaia_worlds": ["tech_gaia_creation"],
    "colossus": ["tech_colossus"],
    # ... extensible
}

MILESTONE_TECHS = {
    "tech_cruisers": 3.0,
    "tech_battleships": 5.0,
    "tech_titans": 4.0,
    "tech_mega_engineering": 8.0,
    "tech_repeatables_start": 2.0,  # first repeatable in each area
}
```

#### 6.6.6 Bottleneck Auto-Detection

The planner automatically identifies the empire's current bottleneck by comparing each dimension against the benchmark percentile data:

```python
def auto_detect_bottleneck(empire_state, benchmark_data):
    """
    Compare empire metrics against p50 (median) of benchmark data
    for the same year bucket, difficulty, and empire type.
    Return the dimension with the largest negative gap.

    8 dimensions capture both direct economic metrics and the meta-systems
    (influence, unity, megastructure readiness) that gate long-term growth.
    """
    year_bucket = round_to_nearest(empire_state.year, 25)
    benchmarks = benchmark_data[year_bucket][difficulty][empire_type]

    gaps = {
        "alloy_income": empire_state.alloy_income - benchmarks["alloy_income"]["p50"],
        "research_total": empire_state.research_total - benchmarks["research_total"]["p50"],
        "fleet_power": empire_state.fleet_power - benchmarks["fleet_power"]["p50"],
        "tech_count": empire_state.tech_count - benchmarks["tech_count"]["p50"],
        "expansion": empire_state.system_count - benchmarks["system_count"]["p50"],
        # NEW: meta-system bottlenecks
        "influence_income": empire_state.influence_income - benchmarks["influence_income"]["p50"],
        "unity_income": empire_state.unity_income - benchmarks["unity_income"]["p50"],
    }

    # Normalize gaps by benchmark p50 to make them comparable
    normalized_gaps = {k: v / max(benchmarks[k]["p50"], 1) for k, v in gaps.items()}

    # Special: megastructure readiness is binary, not a continuous gap
    if empire_state.year >= 2300:  # late enough that megastructures should be considered
        has_mega_eng = "tech_mega_engineering" in empire_state.researched_techs
        has_gw_perk = "ap_galactic_wonders" in empire_state.ascension_perks
        has_site = any(is_megastructure_candidate(s) for s in game_state.systems)
        mega_ready = has_mega_eng and has_gw_perk and has_site
        if not mega_ready and empire_state.megastructures_completed_count == 0:
            normalized_gaps["megastructure_readiness"] = -1.0  # strong signal

    bottleneck = min(normalized_gaps, key=normalized_gaps.get)
    return bottleneck, normalized_gaps[bottleneck]
```

The bottleneck result drives the planner's action generation focus. When the bottleneck is a meta-system dimension, the planner follows specific resolution paths:

| Bottleneck | Resolution path |
|---|---|
| alloy_income | Check available buildings → check tech unlocks → recommend build/research chain |
| research_total | Check Head of Research expertise alignment → check council agenda → check tradition bonuses → recommend changes |
| fleet_power | Recommend ship construction to fill naval capacity → check if higher ship classes are available → check tech path to cruisers/battleships |
| tech_count | Check research speed → check if scientist expertise matches goals → recommend tech path |
| expansion | Check influence budget → if insufficient, trace to faction management → recommend claims + colonization |
| influence_income | Analyze faction approval gaps → identify policy changes that improve approval → check rival count → recommend changes |
| unity_income | Check unity-producing buildings/jobs → check if a tradition adoption bonus would help → recommend tradition picks |
| megastructure_readiness | Trace prerequisite chain: Mega-Engineering → Galactic Wonders perk slot → perk selection → site identification → alloy budget |

#### 6.6.7 Alert Generation

Alerts are generated from simple rule checks, not ML:

| Alert type | Condition | Severity |
|---|---|---|
| Resource deficit imminent | months_until_depletion < 12 for any core resource | High |
| Hostile neighbor threat | strongest_neighbor_fleet / own_fleet > 1.5 | High |
| No ascension path chosen | year > 2280 and ascension_path == None | High |
| Empire stall | stall_detection_flag == True | Medium |
| Faction crisis | any faction approval < 20 and faction support > 15% | Medium |
| Influence starved | influence_income < 2.0/mo and expansion is blocked | Medium |
| Perk slot unused | ascension_perk_slots_available > 0 for 10+ years | Medium |
| Head of Research expertise mismatch | expertise doesn't match current tech goal | Medium |
| Naval capacity underutilized | naval_used / naval_max < 0.5 | Low |
| Building slots wasted | average building utilization < 0.6 | Low |
| Megastructure readiness | has Mega-Engineering + Galactic Wonders but no megastructure started | Low |
| Tech tier gate approaching | within 2 techs of next tier unlock | Info |
| Crisis imminent | year > endgame_start_year and no crisis yet | Info |
| Tradition tree nearly complete | 1 tradition away from completing a tree (= perk slot) | Info |
| Council position empty | any council position unfilled (incurs -25% penalty) | Info |

### 6.7 CLI Interface (`stellaris_advisor.cli`)

Built with **Typer**. All commands support `--help` auto-generated from type hints.

#### 6.7.1 Command Reference

```
stellaris-advisor [--stellaris-dir PATH] [--config PATH] COMMAND [OPTIONS]

Global options:
  --stellaris-dir PATH    Stellaris install directory (for game data files).
                          Auto-detected from common Steam paths:
                            Linux: ~/.steam/steam/steamapps/common/Stellaris/
                            Windows: C:\Program Files (x86)\Steam\steamapps\common\Stellaris\
                            macOS: ~/Library/Application Support/Steam/steamapps/common/Stellaris/
  --config PATH           Path to config.yaml (defaults to ~/.stellaris-advisor/config.yaml)
  --verbose / -v          Enable debug logging

Commands:
  analyze      Full analysis of a save file → empire report + recommendations
  parse        Parse a save file and export structured data
  ingest       Batch-parse a folder of saves into the training dataset
  train        Train or retrain ML models on the current dataset
  score-systems  Rank uncolonized systems for colonization priority
  tech-path    Find optimal research path to a specified goal
  benchmark    Compare empire metrics against historical data
  compare      Compare empire state between two saves
  version      Show tool version and supported Stellaris versions
```

**`analyze` command (primary):**

```
stellaris-advisor analyze <save_file> [OPTIONS]

Options:
  --empire NAME|ID     Empire to analyze (default: player empire)
  --top-n INT          Number of recommendations to show (default: 5)
  --format text|json   Output format (default: text)
  --shap / --no-shap   Include SHAP explanations (default: yes)
  --output PATH        Write report to file instead of stdout
```

**`parse` command:**

```
stellaris-advisor parse <save_file> [OPTIONS]

Options:
  --output PATH           Output file path
  --format json|parquet   Output format (default: json)
  --empire-only           Only extract player empire (faster)
  --pretty                Pretty-print JSON output
```

**`ingest` command:**

```
stellaris-advisor ingest <save_folder> [OPTIONS]

Options:
  --output-dir PATH       Where to write Parquet files (default: data/processed/)
  --recursive             Recurse into subdirectories
  --label                 Also compute labels (requires full game series)
  --parallel INT          Number of parallel parse workers (default: CPU count)
```

**`train` command:**

```
stellaris-advisor train [OPTIONS]

Options:
  --data-dir PATH         Path to processed Parquet data (default: data/processed/)
  --output-dir PATH       Where to save trained models (default: models/)
  --config PATH           Training config YAML
  --tune / --no-tune      Run Optuna hyperparameter tuning (default: no)
  --tune-trials INT       Number of Optuna trials (default: 50)
  --eval-report PATH      Write evaluation metrics to file
```

**`score-systems` command:**

```
stellaris-advisor score-systems <save_file> [OPTIONS]

Options:
  --top-n INT             Number of systems to show (default: 10)
  --format text|json      Output format
  --min-habitability FLOAT  Minimum habitability threshold (default: 0.2)
```

**`tech-path` command:**

```
stellaris-advisor tech-path <save_file> --goal GOAL [OPTIONS]

Options:
  --goal TEXT              Target: tech ID, keyword, or resource target
                           Keywords: battleships, cruisers, megastructures,
                           synthetic_ascension, habitats, max_alloy_income, max_research
  --top-n INT              Number of alternative paths to show (default: 1)
  --format text|json       Output format
```

**`benchmark` command:**

```
stellaris-advisor benchmark <save_file> [OPTIONS]

Options:
  --format text|json       Output format
  --percentiles 25,50,75,90  Which percentiles to show (default: 25,50,75)
```

**`compare` command:**

```
stellaris-advisor compare <save1> <save2> [OPTIONS]

Options:
  --empire NAME|ID         Empire to compare (default: player empire)
  --format text|json       Output format
```

#### 6.7.2 Output Format (analyze command)

```
╔══════════════════════════════════════════════════════════════╗
║  STELLARIS ADVISOR — Empire Report                           ║
║  Empire: United Nations of Earth  |  Year: 2280              ║
║  Type: Democratic / Fanatic Egalitarian + Xenophile           ║
╠══════════════════════════════════════════════════════════════╣
║  EMPIRE HEALTH                                               ║
║  Overall Score:  72/100  (rank 3 of 12)                      ║
║  Economy:   ████████░░  78   Alloy: +43/mo  Energy: +12      ║
║  Tech:      ██████░░░░  61   Researched: 48/~400  Speed: 142 ║
║  Military:  █████░░░░░  52   Fleet: 8.2k / Cap: 120          ║
║  Expansion: ████████░░  81   Systems: 34  Planets: 7         ║
║                                                              ║
║  Benchmark (vs Grand Admiral median at year 2280):           ║
║    Alloy income:   +43/mo  (median: +65, you: p32)           ║
║    Research total:  142/mo  (median: 180, you: p38)          ║
║    Fleet power:    8.2k    (median: 14k, you: p29)           ║
╠══════════════════════════════════════════════════════════════╣
║  TOP RECOMMENDED ACTIONS                                     ║
║                                                              ║
║  1. [+9.2] Colonize Epsilon Crateris                         ║
║     → 18-slot continental world, 4 energy + 3 mineral dep.   ║
║     → Projected: +22 energy, +14 minerals/mo by 2300         ║
║     → Time: ~2 years to establish colony                     ║
║     → Why: expansion metrics drove 62% of score improvement  ║
║                                                              ║
║  2. [+7.4] Research "Alloy Nano-Refining" (Engineering)      ║
║     → Unlocks Advanced Alloy Foundry                         ║
║     → Chain: 8 months research → build on Forge World Alpha  ║
║     → Projected: +18 alloys/mo (current bottleneck)          ║
║     → ROI: 2.25 alloys/mo per month invested                 ║
║                                                              ║
║  3. [+6.1] Build 2 Industrial Districts on Earth             ║
║     → Available now (no tech needed, 4 slots remaining)      ║
║     → Projected: +12 alloys, +4 consumer goods/mo            ║
║                                                              ║
║  4. [+4.3] Expand starbase to Chokepoint Aldebaran           ║
║     → Betweenness centrality: 0.91 (top 3%)                  ║
║     → Secures 3 resource-rich interior systems                ║
║                                                              ║
║  5. [+3.7] Build 12 corvettes (fill naval capacity)          ║
║     → Current: 52/120 naval cap utilization                   ║
║     → Deterrence threshold vs neighbor Yondarim: 11.4k       ║
╠══════════════════════════════════════════════════════════════╣
║  TECH PATH INSIGHT                                           ║
║  Current bottleneck: Alloy production (p32, below median)    ║
║  Recommended path (14 months total):                         ║
║    1. Alloy Nano-Refining (8mo, Engineering)                 ║
║       → Unlocks: Advanced Alloy Foundry                      ║
║       → Best placement: Forge World Alpha (+18 alloys/mo)    ║
║    2. Ministry of Production (6mo, Engineering)              ║
║       → Unlocks: empire-wide +5% resource output             ║
║  Projected cumulative impact: +36 alloys/mo by year 2282     ║
╠══════════════════════════════════════════════════════════════╣
║  TRADITION & PERK PATH                                       ║
║  Traditions: 3/14 trees complete  |  Perks: 2/8 taken        ║
║  Unity: 42/mo  |  Next tradition cost: 890 (21 months)       ║
║  Ascension path: None chosen yet                             ║
║                                                              ║
║  Recommended: Open Prosperity tree                           ║
║    → Adoption bonus: -10% empire size from pops              ║
║    → Completing this tree unlocks perk slot #4               ║
║                                                              ║
║  Next perk recommendation: Galactic Wonders                  ║
║    → Requires: Mega-Engineering tech (you have it ✓)         ║
║    → Unlocks: Dyson Sphere, Matter Decompressor, Ring World  ║
║    → Estimated ROI: +4000 energy/mo after ~8 year build      ║
║                                                              ║
║  ⚠ No ascension path chosen — recommend Synthetic Evolution  ║
║    → Matches Materialist ethics; frees food districts         ║
║    → Prerequisite: The Flesh is Weak (Tier 2, need 2 perks ✓)║
╠══════════════════════════════════════════════════════════════╣
║  FACTION & INFLUENCE                                         ║
║  Influence: 4.2/mo  (factions: 1.8, projection: 1.5, base: 3)║
║  Council legitimacy: 68% (faction approval driven)           ║
║                                                              ║
║  Top faction:     Technologist  72% approval, 35% support    ║
║  Problem faction: Traditionalist 28% approval, 18% support   ║
║    → Issue: "AI Servitude" policy angers this faction         ║
║    → Changing AI policy → +15 approval → +0.3 influence/mo   ║
║    → BUT: would block robot pop assembly — not recommended   ║
║                                                              ║
║  Action: Declare 1 more rival → +0.5 influence/mo            ║
╠══════════════════════════════════════════════════════════════╣
║  ALERTS                                                      ║
║  ⚠ HIGH: Consumer goods deficit in ~8 years at current rate  ║
║  ⚠ HIGH: Neighbor Prikkiki-Ti fleet power 2.1x yours         ║
║  ⚠ HIGH: No ascension path chosen (year 2280)                ║
║  ⚠ MED:  Empire stall — mineral income flat for 20 years     ║
║  ⚠ MED:  Head of Research has Biology expertise, but you     ║
║          need Voidcraft for Mega-Engineering path             ║
║  ℹ INFO: 1 tech from unlocking Tier 4 Engineering            ║
║  ℹ INFO: Prosperity tree 1 tradition from completion (perk!) ║
╚══════════════════════════════════════════════════════════════╝
```

### 6.8 Library API (`stellaris_advisor`)

```python
from stellaris_advisor.parser import parse_save
from stellaris_advisor.gamedata import GameKnowledgeGraph
from stellaris_advisor.features import extract_empire_features, extract_system_features
from stellaris_advisor.models import EmpireValuator, ColonizationScorer
from stellaris_advisor.planner import analyze, optimize_tech_path
from stellaris_advisor.config import Config

# --- Setup ---
config = Config.from_yaml("config.yaml")
game_data = GameKnowledgeGraph.load(config.stellaris_dir)
model_a = EmpireValuator.load("models/empire_valuator.txt")
model_b = ColonizationScorer.load("models/colonization_scorer.txt")

# --- Parse a save ---
game_state = parse_save("path/to/save.sav")
player = game_state.player_empire

# Access parsed data
print(player.name)                          # "United Nations of Earth"
print(player.resource_income["alloys"])      # 43.0
print(player.researched_techs)              # {"tech_corvettes", "tech_destroyers", ...}
print(player.owned_planets[0].planet_class) # "pc_continental"
print(player.owned_planets[0].buildings)    # ["building_capital_2", "building_alloy_foundry", ...]

# --- Query game knowledge graph ---
frontier = game_data.reachable_techs(
    researched=player.researched_techs,
    empire_type=player.authority,
    ethics=player.ethics,
)
print(f"{len(frontier)} techs available on next draw")

available = game_data.available_buildings(
    researched=player.researched_techs,
    planet_class="pc_continental",
    empire_type=player.authority,
    existing_buildings=player.owned_planets[0].buildings,
    capital_level=2,
)
print(f"{len(available)} buildings can be built on planet 0")

chain = game_data.tech_path_to("tech_battleships", player.researched_techs)
print(f"Need {len(chain)} techs to unlock battleships")

# --- Run full analysis ---
report = analyze(
    empire_state=player,
    game_state=game_state,
    game_graph=game_data,
    model_a=model_a,
    model_b=model_b,
)

print(f"Health score: {report.health_score}/100")
for rec in report.recommendations:
    print(f"[{rec.final_score:+.1f}] {rec.description}")
    if rec.dependency_chain:
        for step in rec.dependency_chain:
            print(f"       {step}")
    print(f"       Model A Δ: {rec.model_a_delta:+.2f}, ROI: {rec.estimated_roi:.2f}")

for alert in report.alerts:
    print(f"[{alert.severity}] {alert.message}")

# --- Tech path optimization ---
path = optimize_tech_path(
    goal="megastructures",
    empire_state=player,
    game_graph=game_data,
    model_a=model_a,
)
print(f"Path to megastructures: {path.total_months} months")
for step in path.steps:
    print(f"  {step.tech_name} ({step.area}, {step.estimated_months}mo, cost={step.cost})")
    for unlock in step.unlocks:
        print(f"    → {unlock}")

# --- Feature extraction (for custom analysis) ---
features = extract_empire_features(player, game_state)
print(f"Feature vector: {len(features)} dimensions")
prediction = model_a.predict(features)
print(f"Predicted score delta at T+50: {prediction:.3f}")

# --- Training ---
from stellaris_advisor.models import train_all
train_all(data_dir="data/processed/", output_dir="models/", config=config)
```

---

## 7. Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                      CLI / Library API                        │
├──────────┬──────────┬───────────────┬──────────┬─────────────┤
│  parse   │  ingest  │   analyze     │  train   │  tech-path  │
│          │          │  benchmark    │          │  score-sys  │
│          │          │  compare      │          │             │
├──────────┴──────────┴───────┬───────┴──────────┴─────────────┤
│                             │                                 │
│        ML Models            │       Strategic Planner         │
│   (learned, statistical)    │    (rule-based + ML-scored)     │
│                             │                                 │
│  ┌────────────────────┐     │  ┌───────────────────────────┐  │
│  │ Model A: Empire    │◄────┼──┤ Action Generator          │  │
│  │ State Valuator     │     │  │ + Delta Simulator          │  │
│  │ (LightGBM)        │─────┼─►│ + Hybrid Ranker            │  │
│  └────────────────────┘     │  └──────────┬────────────────┘  │
│  ┌────────────────────┐     │             │                   │
│  │ Model B: Colony    │     │  ┌──────────▼────────────────┐  │
│  │ Scorer (LightGBM)  │     │  │ Tech Path Optimizer       │  │
│  └────────────────────┘     │  │ + Bottleneck Detector      │  │
│                             │  │ + Alert Generator          │  │
│  ┌────────────────────┐     │  └──────────┬────────────────┘  │
│  │ Benchmark Data     │     │             │                   │
│  │ (percentile store) │     │             │                   │
│  └────────────────────┘     │             │                   │
├─────────────────────────────┤             │                   │
│       Feature Engineering   │  ┌──────────▼────────────────┐  │
│  Empire │ System │ Temporal  │  │ Game Knowledge Graph      │  │
│         │        │           │  │  Tech DAG (~400 nodes)    │  │
│         │        │           │  │  Building defs (~100)     │  │
│         │        │           │  │  District defs (~20)      │  │
│         │        │           │  │  Job defs (~80)           │  │
│         │        │           │  │  Ship defs, Edicts, etc.  │  │
│         │        │           │  │  Planet class constraints  │  │
│         │        │           │  │  Empire type constraints   │  │
├─────────┴────────┴───────────┤  │ Source: game files (static)│  │
│          Data Layer          │  └───────────────────────────┘  │
│  Clausewitz Parser (Rust)    │                                 │
│  → Typed Dataclasses         │                                 │
│  → Parquet Store             │                                 │
├──────────────────────────────┴─────────────────────────────────┤
│                       Input Files                              │
│   Save File (.sav)              Stellaris Install Dir          │
│   ZIP(gamestate + meta)         common/technology/*.txt        │
│                                 common/buildings/*.txt         │
│                                 common/districts/*.txt         │
│                                 common/ship_sizes/*.txt        │
│                                 common/pop_jobs/*.txt          │
│                                 common/planet_classes/*.txt    │
│                                 common/governments/*.txt       │
│                                 common/traits/*.txt            │
└────────────────────────────────────────────────────────────────┘
```

**Package structure:**

```
stellaris-advisor/
├── stellaris_advisor/
│   ├── __init__.py
│   ├── parser/
│   │   ├── __init__.py
│   │   ├── clausewitz.py          # Low-level Clausewitz format parser (or Rust FFI)
│   │   ├── extractor.py           # Typed extraction: raw dict → dataclasses
│   │   ├── version.py             # Version detection and schema selection
│   │   └── models.py              # Dataclass definitions (EmpireState, PlanetState, etc.)
│   ├── gamedata/
│   │   ├── __init__.py
│   │   ├── loader.py              # Master loader: reads all game files → GameKnowledgeGraph
│   │   ├── tech_graph.py          # Tech DAG construction, BFS/DFS queries
│   │   ├── buildings.py           # Building definitions + constraint resolution
│   │   ├── districts.py           # District definitions + planet class rules
│   │   ├── jobs.py                # Job definitions + output estimation
│   │   ├── ships.py               # Ship class definitions
│   │   ├── planet_classes.py      # Planet class definitions + habitability
│   │   ├── empires.py             # Civic, authority, trait definitions
│   │   ├── traditions.py          # Tradition tree + individual tradition definitions
│   │   ├── ascension_perks.py     # Perk definitions, tier logic, mutual exclusions
│   │   ├── megastructures.py      # Megastructure definitions, site requirements, stage costs
│   │   ├── policies.py            # Policy definitions, faction issue mappings
│   │   ├── factions.py            # Faction definitions, issue conditions, approval formulas
│   │   ├── council.py             # Council position definitions, agenda definitions
│   │   ├── federations.py         # Federation type definitions, per-level bonuses
│   │   ├── validation.py          # DAG integrity + cross-reference checks
│   │   └── models.py              # Dataclass definitions (TechNode, BuildingDef, TraditionTree, etc.)
│   ├── features/
│   │   ├── __init__.py
│   │   ├── empire.py              # Empire-level feature extraction (~107 features)
│   │   ├── system.py              # System-level feature extraction (~12 features)
│   │   └── temporal.py            # Time-series delta features (~8 features)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingest.py              # Batch save → Parquet pipeline
│   │   ├── synthetic.py           # Observer run automation + monitoring
│   │   ├── labels.py              # Label generation (outcome scores, colonization)
│   │   └── schema.py              # Parquet schema definitions
│   ├── models/
│   │   ├── __init__.py
│   │   ├── valuator.py            # Model A: Empire state valuator (train + predict)
│   │   ├── colonization.py        # Model B: System colonization scorer
│   │   ├── benchmark.py           # Benchmark percentile data store
│   │   ├── evaluation.py          # Metrics, cross-validation, SHAP analysis
│   │   └── tuning.py              # Optuna hyperparameter search
│   ├── planner/
│   │   ├── __init__.py
│   │   ├── action_generator.py    # Enumerate candidate macro-actions with constraints
│   │   ├── delta_simulator.py     # Simulate feature deltas for actions
│   │   ├── tech_optimizer.py      # Tech path BFS/DFS + ROI scoring
│   │   ├── tradition_advisor.py   # Tradition ordering + perk slot timing recommendations
│   │   ├── perk_advisor.py        # Ascension perk selection + path recommendation
│   │   ├── megastructure_advisor.py # Megastructure readiness, site selection, build order
│   │   ├── faction_advisor.py     # Faction approval analysis, policy change impact estimation
│   │   ├── council_advisor.py     # Leader-to-position matching, agenda optimization
│   │   ├── ranker.py              # Hybrid scoring (Model A + rule-based ROI)
│   │   ├── bottleneck.py          # Auto-detect empire bottleneck vs benchmarks (8 dimensions)
│   │   ├── alerts.py              # Rule-based alert generation (15 alert types)
│   │   └── report.py              # Report assembly and formatting
│   ├── cli.py                     # Typer CLI entry point + all subcommands
│   └── config.py                  # Pydantic config schema
├── scripts/
│   ├── generate_synthetic.sh      # Batch observer run launcher (shell)
│   └── generate_synthetic.py      # Python orchestrator for observer runs
├── data/                          # .gitignore'd, local data store
│   ├── raw/                       # Raw .sav files organized by game_id
│   └── processed/                 # Parquet files (empire_features, system_features, metadata)
├── models/                        # Trained model artifacts + benchmark data
├── tests/
│   ├── test_parser/
│   ├── test_gamedata/
│   ├── test_features/
│   ├── test_models/
│   ├── test_planner/
│   ├── test_cli/
│   └── fixtures/                  # Sample save files and game data for testing
├── pyproject.toml
├── config.yaml                    # Default configuration
└── README.md
```

---

## 8. Configuration (`config.yaml`)

```yaml
# Stellaris AI Advisor configuration

stellaris:
  install_dir: null              # Auto-detected if null
  supported_versions: ["4.3"]
  save_dir: null                 # Auto-detected from OS-specific default path

parser:
  backend: "rust"                # "rust" (via maturin) or "python"
  cache_parsed: true             # Cache parsed saves to avoid re-parsing
  cache_dir: "~/.stellaris-advisor/cache/"
  empire_only: false             # If true, only parse player empire (faster)

data:
  processed_dir: "data/processed/"
  raw_dir: "data/raw/"
  parquet_compression: "snappy"
  temporal_lookback: 5           # Number of prior saves for temporal features

synthetic:
  runs: 50                       # Number of observer runs to generate
  galaxy_sizes: [400, 600, 800, 1000]
  ai_empires_range: [8, 30]
  difficulties: ["ensign", "captain", "admiral", "grand_admiral"]
  crisis_strengths: [1.0, 2.0, 3.0, 5.0]
  autosave_interval_months: 12
  max_year: 2500

features:
  empire_feature_count: 80       # Expected count (for validation)
  system_feature_count: 12
  temporal_feature_count: 8

labels:
  score_weights:
    fleet_power: 0.25
    tech_count: 0.25
    total_income: 0.30
    pop_count: 0.20
  lookahead_years: 50

models:
  output_dir: "models/"
  model_a:
    type: "lightgbm"
    objective: "regression_l2"
    num_leaves: 63
    learning_rate: 0.05
    n_estimators: 1000
    early_stopping_rounds: 50
    feature_fraction: 0.8
    bagging_fraction: 0.8
    bagging_freq: 5
    min_child_samples: 20
  model_b:
    type: "lightgbm"
    objective: "lambdarank"
    num_leaves: 31
    learning_rate: 0.05
    n_estimators: 500
    early_stopping_rounds: 30
  tuning:
    enabled: false
    n_trials: 50
    timeout_seconds: 3600

planner:
  alpha: 0.6                     # Weight for Model A score delta
  beta: 0.4                      # Weight for rule-based ROI
  top_n_recommendations: 7       # Increased from 5 to accommodate more categories
  top_n_colony_candidates: 5
  naval_utilization_threshold: 0.7
  threat_power_ratio: 1.5
  discount_rate: 0.05            # For time-discounted tech/tradition/perk values
  # Tradition & perk advisor
  tradition_adoption_bonus_weight: 1.5  # Multiplier for front-loaded adoption bonuses
  perk_slot_value_weight: 2.0           # How much to value completing a tree for the perk slot
  ascension_path_deadline_year: 2300    # Alert if no path chosen after this year
  # Megastructure advisor
  megastructure_readiness_year: 2300    # Year after which megastructure readiness is checked
  megastructure_alloy_threshold: 200    # Min alloy/mo income before recommending construction
  # Faction & influence advisor
  influence_crisis_threshold: 2.0       # Below this influence/mo, flag as bottleneck
  faction_approval_warning: 20          # Below this approval, flag faction as problematic
  policy_cooldown_warning: 0.7          # Don't recommend policy change if agenda progress > this
  # Council advisor
  expertise_mismatch_alert: true        # Alert when HoR expertise doesn't match tech goal
  agenda_switch_progress_threshold: 0.7 # Don't recommend agenda switch above this progress

benchmark:
  year_bucket_size: 25
  percentiles: [25, 50, 75, 90]
  # Added dimensions for new mechanics
  dimensions:
    - alloy_income
    - research_total
    - fleet_power
    - tech_count
    - system_count
    - influence_income
    - unity_income
    - traditions_completed
```

---

## 9. Evaluation Plan

### 9.1 Model Metrics

| Component | Primary metric | Secondary metrics | Target |
|---|---|---|---|
| Model A: Empire Valuator | RMSE on score delta prediction | R², Spearman ρ (rank correlation), MAE | Spearman ρ > 0.3 on held-out games |
| Model B: Colonization Scorer | NDCG@10 | Precision@5, Spearman ρ vs. actual colonization order | NDCG@10 > 10% improvement over habitability heuristic |
| Planner: Action Recommendations | Player agreement rate (manual evaluation) | Constraint violation rate (must be 0%), comparison vs. pure-heuristic recommendations | Sensible output on 10+ real saves |
| Tech Path Optimizer | Path validity rate | Average path length, ROI estimate correlation with observed outcomes | 100% valid paths (no DAG violations) |
| Tradition/Perk Advisor | Recommendation coherence (manual eval) | Alignment with community meta; perk prerequisite/exclusion validity rate | No invalid perk suggestions; tradition order matches established wisdom for standard builds |
| Faction Influence Analyzer | Approval delta accuracy | Comparison of predicted approval changes against wiki issue tables | Directionally correct on all policy→approval mappings |
| Megastructure Advisor | Readiness detection accuracy | False positive/negative rate on "should have started megastructure" | Correctly flags readiness when all prerequisites are met |
| Bottleneck Detector | Agreement with manual assessment | Breakdown of detected bottlenecks by category across test saves | Correct on ≥70% of manual assessments |
| Benchmark | Percentile accuracy | Comparison against manually verified game states | Directionally correct on all dimensions |

### 9.2 Baselines

- **Heuristic baseline (planner):** Hand-coded rules: "always colonize highest-habitability planet"; "build alloy foundries when alloy income < threshold"; "research cheapest available tech." The hybrid planner must produce more nuanced recommendations.
- **Random baseline (planner):** Randomly ordered action ranking. Establishes a floor.
- **Lag baseline (Model A):** Predict score_delta = 0 (no change). Tests whether the model captures any real signal beyond "nothing changes."
- **Pure-ML baseline (planner):** The v1.0 approach — Model A with naive feature-delta simulation, no game knowledge graph. Tests whether the game knowledge graph adds value over blind ML scoring.
- **Habitability-only baseline (Model B):** Rank systems solely by habitability of best planet. The colonization scorer must beat this.

### 9.3 Validation Strategy

- **Temporal split:** Train on saves from years 2200–2350, validate on 2350–2450, test on 2450–2500. Prevents leaking future game knowledge.
- **Game-level split:** Hold out entire game runs (not individual snapshots) to prevent data leakage across saves from the same game. Recommended: 70% train, 15% validation, 15% test by game count.
- **Cross-galaxy generalization:** Reserve some galaxy configurations (e.g., 1000-star, grand admiral) exclusively for the test set.
- **Tech path validation:** Automated test that all recommended paths are valid traversals of the tech DAG with all constraints satisfied (this is a correctness test, not a performance test — must be 100%).
- **Constraint violation audit:** On every recommendation output, verify that all suggested buildings are legal given (planet class, empire type, researched techs, slot availability). Violation rate must be 0%.

---

## 10. Technical Decisions

| Decision | Choice | Rationale |
|---|---|---|
| ML framework | LightGBM | Best-in-class for tabular data at this scale; faster than XGBoost; native categorical support; built-in LambdaMART ranking loss for Model B |
| Explainability | SHAP (TreeSHAP) | Exact and fast for tree ensembles; powers recommendation explanations |
| Hyperparameter tuning | Optuna | Bayesian optimization; Pruner support for early stopping; easy LightGBM integration |
| Data format | Parquet (via PyArrow) | Columnar, compressed, fast reads for feature subsets; industry standard for data engineering |
| CLI framework | Typer | Type-hint driven, auto-generates help, cleaner than Click for this scope |
| Config | Pydantic v2 | Typed config validation; YAML/JSON input; good error messages |
| Save parsing | Stellaris-dashboard Rust parser via maturin (preferred) or custom Python (fallback) | Rust parser is fastest; already handles Clausewitz edge cases; Python fallback avoids Rust build dependency |
| Game data parsing | Reuse Clausewitz parser (same format as saves) | Game definition files use the same Clausewitz text format |
| Graph operations | NetworkX | Hyperlane graph for chokepoint detection (betweenness centrality ~O(VE), V≤1000); tech DAG for BFS/DFS (V≤400); both small enough for NetworkX |
| Testing | pytest + hypothesis | Property-based testing for parser edge cases; DAG validity assertions for tech paths; snapshot testing for CLI output |
| Packaging | pyproject.toml + hatch | Modern Python packaging; editable installs; PEP 517/518 compliant |
| Python version | ≥3.10 | For match/case, type union syntax, dataclass improvements |
| Rich terminal output | Rich | For the CLI report rendering (progress bars, tables, styled text) |

---

## 11. Known Limitations and Honest Scope

### What it does well:
1. **State valuation** — quantitative answer to "how healthy is my empire relative to this galaxy?"
2. **Colonization ranking** — data-driven scoring of systems, better than habitability-only heuristics.
3. **Single-step action recommendations across 10 categories** — economy, tech, expansion, military, traditions, ascension perks, megastructures, council/agenda, faction management, and policy changes — each with dependency chains explained and constraint violations guaranteed absent.
4. **Tech path planning** — "what's the fastest/most valuable research path to unlock X?" using exact game rules, evaluated against the current planet portfolio with building constraints respected.
5. **Tradition and perk path advising** — "which tradition tree should I open next?" and "which ascension perk should I take?" with time-discounted lifetime value estimates. Ascension path recommendation based on empire ethics and economy.
6. **Faction and influence analysis** — "why is my influence low, and which policy changes would fix it?" with exact approval delta calculations per faction per policy option, including trade-off warnings.
7. **Megastructure readiness assessment** — traces the full prerequisite chain (tech → perk → site → alloy budget) and flags when the empire should begin the megastructure transition.
8. **Council optimization** — identifies when the Head of Research's expertise is misaligned with the tech goal and recommends leader reassignment.
9. **Benchmarking** — "how does my empire compare to historical data at this year/difficulty?"
10. **Bottleneck detection across 8 dimensions** — identifies the binding constraint (alloys, research, fleet, tech, expansion, influence, unity, megastructure readiness) and traces to the root cause.

### What it does NOT do:
1. **Multi-step strategic planning** — the planner evaluates each action independently (greedy single-step). It cannot reason about multi-turn plans like "sacrifice alloys now → build research → tech rush → unlock megastructures → dominate late game." The tech path optimizer and tradition/perk advisor handle chains within their specific domains, but cross-domain action sequence optimization is not implemented.
2. **Opponent modeling** — recommendations are based on the player's empire state, not on predicting what AI empires will do next.
3. **Event/crisis anticipation** — the advisor does not predict when the endgame crisis will spawn or which events will fire. It can only detect a crisis that is already active in the save.
4. **Combat simulation** — fleet composition advice is based on aggregate fleet power, not on detailed weapon-vs-defense matchups (energy vs shields, kinetic vs armor, etc.).
5. **Pop growth and job assignment modeling** — resource projections are based on building/district definitions, not on simulating pop growth curves, migration, or optimal job assignment. This makes ROI estimates approximate.
6. **Mod support** — MVP is vanilla Stellaris 4.3 only. Modded tech trees, buildings, traditions, and perks would require loading mod files (the architecture supports this, but it's not implemented).
7. **Multiplayer-specific advice** — no differentiation between human and AI opponents.
8. **Stochastic tech draw** — the advisor can identify which techs are *reachable* and which expertise would boost their draw weight, but cannot predict which 3 options the card shuffle will present.
9. **Shroud interaction advice** — the Psionic ascension path's Shroud covenants involve high-variance RNG outcomes (including the catastrophic End of the Cycle). The advisor cannot model these stochastic outcomes.
10. **Espionage operations** — spy network management, operation targeting, and counter-espionage are not modeled. These are situational and rarely decisive for empire outcomes.
11. **Galactic Community resolution proposal strategy** — the advisor can flag the impact of active resolutions but does not recommend which resolutions to propose or how to build diplomatic weight coalitions.
12. **Vassal agreement optimization** — subject empires (tax rates, diplomatic freedom, integration timing) are tracked but not actively optimized.

### Why these limitations are acceptable for MVP:
- Single-step greedy advice across 10 action categories is still far more useful than no quantitative advice. The tech path, tradition/perk, and megastructure advisors each handle multi-step chains within their domains, compensating for the lack of cross-domain planning.
- The architecture is designed so that Model A serves as a value function for a future RL agent or MCTS planner that could replace the greedy approach.
- The low-impact mechanics (espionage, vassal agreements, Shroud, galactic community strategy) are primarily reactive or situational — they don't drive the core economic/tech/expansion loop that determines most game outcomes.

---

## 12. Risks and Mitigations

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| Clausewitz format changes between Stellaris versions | Parser breaks | Medium (happens with major patches) | Version detection + versioned extraction schemas; pin to 4.3 for MVP; abstract field paths into a schema file |
| Game definition file format changes | Game knowledge graph breaks | Medium | Version-pin game data loader; abstract parser so format changes are localized to one file |
| Rust parser build dependency is problematic for users | Installation friction | Medium | Provide pre-built wheels for common platforms; implement Python fallback parser |
| Insufficient training data from observer runs | Weak models | Medium | Start with 50 runs; augment with varied galaxy settings; fall back to heuristic baselines if models don't beat them; clearly report model confidence |
| Observer mode behavior ≠ player behavior | Models learn AI patterns, not optimal play | High | Frame Model A as "predicting what typically happens" not "what's optimal"; observer data establishes baseline; community player saves improve this post-MVP |
| Feature engineering is the bottleneck, not model architecture | Time sink | High | Start with a small feature set (~30 features), validate signal exists, then expand incrementally |
| Save parsing is slow for large late-game galaxies | Poor UX (>30 second target) | Medium | Use Rust parser; cache parsed results; parse only player empire by default; lazy-load system data |
| Game balance changes invalidate trained models | Stale advice | Medium | Version-tag all training data; retrain per major patch; document supported versions clearly |
| Game data parsing covers only vanilla content | Useless for modded games | Low for MVP | Document vanilla-only limitation clearly; design loader with a `mod_dirs` parameter for post-MVP |
| Tech ROI estimates are inaccurate (missing pop growth, stability, trade cost effects) | Misleading recommendations | Medium | Label estimates as approximate in output; use Model A score delta as primary ranking signal (trained on real data that captures these effects); rule-based ROI is secondary |
| NetworkX is too slow for betweenness centrality on 1000-star galaxies | Slow analysis | Low | Betweenness centrality on 1000-node graphs is fast (~seconds); cache results; fallback to degree centrality if needed |
| Parquet data grows large (hundreds of game runs × hundreds of saves each) | Disk usage | Low | Parquet compression (snappy); configurable retention; only store feature vectors, not raw parsed data |
| Expanded game data parsing scope (traditions, perks, factions, megastructures, policies, council) requires significantly more development than v3.0 | Timeline overrun | High | Prioritize by strategic impact: traditions/perks first (highest impact on recommendations), then factions/influence (binding constraint analysis), then megastructures (late-game), then council/policies (refinement). Each layer is independently useful. |
| Faction issue → approval mapping is complex and version-sensitive | Incorrect influence recommendations | Medium | Validate against wiki-documented issue tables; include automated regression tests comparing computed approval against known save states |
| Tradition/perk ordering recommendations may not match player's playstyle | Unhelpful advice | Medium | Frame recommendations as "highest lifetime value" rather than "you must do this"; allow user to specify playstyle preference (tall/wide/military/tech rush) in config |
| Council and agenda mechanics changed significantly in Galactic Paragons DLC | Feature may not work without DLC | Medium | Detect DLC presence from save/game files; gracefully degrade council features if DLC not present |

---

## 13. Deliverables

The following components constitute the MVP. No prescribed ordering or schedule — the developer determines the build sequence.

1. **Save parser** — Clausewitz parser integration, typed extraction layer, version detection. Produces `EmpireState`, `PlanetState`, `SystemState`, etc. from any Stellaris 4.3 `.sav` file.

2. **Game knowledge graph** — Loader for all game definition files (tech DAG, buildings, districts, ships, jobs, planet classes, traditions, ascension perks, megastructures, policies, factions, council positions, agendas, federation types). Produces `GameKnowledgeGraph` with constraint resolution queries. Passes integrity validation.

3. **Feature engineering** — Empire-level (~107), system-level (~12), and temporal (~8) feature extractors. Covers resources, efficiency ratios, expansion, military, tech, planet portfolio, pops, diplomacy, empire archetype, traditions/perks, leaders/council, factions/influence, galactic community, federation, and megastructures.

4. **Data pipeline** — Ingestion pipeline (batch save → Parquet). Labeling logic (empire outcome, colonization). Synthetic data generation scripts (observer run automation).

5. **ML models** — Model A (empire state valuator, LightGBM regression). Model B (colonization scorer, LightGBM ranking). Training, evaluation, SHAP analysis. Benchmark percentile data store.

6. **Strategic planner** — Action generator across 10 categories (economy, tech, expansion, military, traditions, perks, megastructures, council/agenda, faction management, policy). Delta simulator for all action types. Hybrid ranker (Model A + rule-based ROI). Tech path optimizer. Tradition/perk path advisor. Megastructure readiness assessor. Faction influence analyzer. Council expertise matcher. Bottleneck detector (8 dimensions). Alert generator (15 alert types).

7. **CLI** — All commands: `analyze`, `parse`, `ingest`, `train`, `score-systems`, `tech-path`, `benchmark`, `compare`, `version`. Rich terminal output.

8. **Library API** — Public Python API for programmatic access to all capabilities.

9. **Tests** — Unit tests for parser, game data integrity, feature extraction, planner constraint validation, CLI output. Fixtures with sample save files.

10. **Documentation** — README with quickstart guide, config reference, supported Stellaris versions, API examples.

---

## 14. Post-MVP Extensions

Listed roughly by value and feasibility:

1. **Multi-step lookahead** — Extend the planner from greedy single-step to N-step lookahead using Model A as heuristic in a tree search (similar to MCTS). More tractable than full RL and directly improves recommendation quality.

2. **True RL agent** — Train a policy network over action sequences using observer data as environment trajectories. Model A already serves as a value function; the planner's macro-action space becomes the RL action space. Requires defining a formal MDP (state, action, transition, reward).

3. **Community data platform** — Accept anonymized save uploads from the community. Larger and more diverse dataset = better models. Could crowd-source "expert labels" (what did good players actually do at each decision point?).

4. **War outcome predictor** — Model fleet combat outcomes given fleet compositions, component loadouts, tech levels, and admiral traits. Stellaris combat is deterministic enough to model well with classification/regression.

5. **Web dashboard** — FastAPI backend + React frontend for visual save analysis: charts, galaxy map with system scores overlaid, interactive tech tree with recommended path highlighted. Extends beyond CLI.

6. **Multiplayer opponent modeling** — In multiplayer saves, build models of each player's tendencies and predict their next moves based on historical behavior.

7. **Mod compatibility** — Load modded tech trees, buildings, and districts from mod directories. The `gamedata` module already accepts a file path; extending to mod directories is structurally straightforward.

8. **Pop growth and job assignment modeling** — More accurate resource projections by modeling pop growth curves (logistic growth, immigration/emigration), job priority assignment, and the effect of stability on output. Directly improves the planner's ROI estimates.

9. **Fleet composition optimizer** — Given available components and a budget, find the fleet composition that maximizes effective power against likely enemy compositions.

10. **Diplomacy advisor** — Model the AI's diplomatic disposition (based on ethics, personality, opinion, relative power) to suggest federation targets, rivalry timing, and war declaration windows.

11. **Espionage advisor** — Model spy network management, operation targeting (steal tech, sabotage starbase, assassination), and counter-espionage allocation.

12. **Vassal agreement optimizer** — Optimize subject empire tax rates, diplomatic freedom, and integration timing for maximum resource extraction and loyalty management.

13. **Galactic Community resolution strategy** — Recommend which resolutions to propose/support/oppose based on diplomatic weight analysis and resolution effect estimation; advise on Custodian/Emperor path timing.

14. **Shroud interaction modeling** — For Psionic empires, model expected value of Shroud diving outcomes including covenant bonuses and risks (especially End of the Cycle probability).

15. **Species rights optimizer** — Recommend species rights settings (citizenship, living standards, military service) that maximize output while maintaining faction approval targets.

---

## 15. Success Criteria

The MVP is successful if:

1. **Parser works reliably** on Stellaris 4.3 saves — both observer and player saves, ironman and non-ironman, across galaxy sizes 400–1000. All new fields (traditions, perks, leaders, factions, policies, galcom, federation, megastructures) correctly extracted.
2. **Game knowledge graph is correct and comprehensive** — all tech prerequisites, building constraints, planet class rules, empire type restrictions, tradition definitions, ascension perk prerequisites/exclusions, megastructure site requirements, policy→faction issue mappings, and council position definitions match the actual game. Validated by automated integrity tests against game files.
3. **Empire valuator beats the lag baseline** — Spearman ρ > 0.3 on score delta prediction on held-out game runs.
4. **Colonization scorer beats the habitability heuristic** — NDCG@10 improvement > 10%.
5. **Tech path optimizer produces valid paths** — 100% of paths are legal DAG traversals with all constraints satisfied.
6. **Tradition and perk recommendations are coherent** — on manual inspection, tradition ordering aligns with established community meta (Discovery/Expansion early, ascension path by mid-game); perk recommendations respect tier prerequisites and mutual exclusions; ascension path recommendation matches empire ethics.
7. **Faction influence analysis is accurate** — for 10+ test saves, the advisor's computed faction approval deltas from policy changes match the actual game mechanics (validated against wiki issue tables).
8. **Strategic planner produces sensible recommendations across all 10 categories** — on manual inspection of 10+ saves from real playthroughs, with correct dependency chains, no illegal building/tradition/perk suggestions, and no constraint violations.
9. **Bottleneck detection correctly identifies the binding constraint** — on manual inspection, the detected bottleneck matches the player's actual strategic situation (e.g., influence-starved empires get faction advice, late-game empires without megastructures get megastructure readiness warnings).
10. **CLI runs in < 30 seconds** on a single save file (parsing + game data loading + inference + planning) for a 600-star galaxy mid-game save.
11. **The project is fun to use** on the developer's own Stellaris games.
