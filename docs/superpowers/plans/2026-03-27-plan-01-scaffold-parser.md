# Plan 1: Project Scaffold + Clausewitz Parser

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the project foundation — packaging, configuration, and a complete Clausewitz parser that can read Stellaris save files and game definition files into typed Python dataclasses.

**Architecture:** Three-layer parser (tokenizer → recursive descent → typed extraction) producing frozen dataclasses. Custom Python implementation, no external parser dependencies. Handles both save files (ZIP archives) and game definition files (plain text).

**Tech Stack:** Python >=3.10, Pydantic v2, PyYAML, pytest, hypothesis

---

## Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `src/stellaris_advisor/__init__.py`
- Create: `src/stellaris_advisor/parser/__init__.py`
- Create: `src/stellaris_advisor/gamedata/__init__.py`
- Create: `src/stellaris_advisor/features/__init__.py`
- Create: `src/stellaris_advisor/data/__init__.py`
- Create: `src/stellaris_advisor/rl/__init__.py`
- Create: `src/stellaris_advisor/colonization/__init__.py`
- Create: `src/stellaris_advisor/planner/__init__.py`
- Create: `.gitignore`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "stellaris-advisor"
version = "0.1.0"
description = "AI-powered strategy advisor for Stellaris"
readme = "README.md"
license = "MIT"
requires-python = ">=3.10"
authors = [
    { name = "Alex Zhang" },
]
dependencies = [
    "networkx>=3.0",
    "typer>=0.9",
    "rich>=13.0",
    "pydantic>=2.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
ml = [
    "torch>=2.0",
    "d3rlpy>=2.0",
    "lightgbm>=4.0",
    "pyarrow>=14.0",
    "optuna>=3.0",
    "shap>=0.42",
    "scikit-learn>=1.3",
]
all = ["stellaris-advisor[ml]"]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "hypothesis>=6.0",
    "black>=23.0",
    "ruff>=0.1",
    "mypy>=1.0",
]

[project.scripts]
stellaris-advisor = "stellaris_advisor.cli:app"

[tool.hatch.build.targets.wheel]
packages = ["src/stellaris_advisor"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "unit: fast tests with no I/O",
    "integration: tests that parse real files",
    "slow: RL training and large save parsing",
]

[tool.ruff]
src = ["src"]
line-length = 100

[tool.mypy]
python_version = "3.10"
strict = true

[tool.coverage.run]
source = ["stellaris_advisor"]

[tool.coverage.report]
fail_under = 80
```

- [ ] **Step 2: Create directory structure and __init__.py files**

Create all package directories with empty `__init__.py` files:

```python
# src/stellaris_advisor/__init__.py
"""Stellaris AI Advisor — quantitative strategy recommendations for Stellaris."""

__version__ = "0.1.0"
```

All other `__init__.py` files are empty (zero bytes).

Create the test directory:

```python
# tests/__init__.py
# (empty)
```

```python
# tests/conftest.py
"""Shared test fixtures for stellaris-advisor."""

import pytest
```

- [ ] **Step 3: Create .gitignore**

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
dist/
build/
*.egg
.eggs/

# Virtual environments
.venv/
venv/
env/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Testing
.pytest_cache/
htmlcov/
.coverage
.coverage.*

# Type checking
.mypy_cache/

# Project-specific
data/
models/
*.sav
.stellaris-advisor/
.superpowers/
```

- [ ] **Step 4: Install the package in editable mode and verify**

Run: `pip install -e ".[dev]"`
Expected: Successful installation

Run: `python -c "import stellaris_advisor; print(stellaris_advisor.__version__)"`
Expected: `0.1.0`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/ tests/ .gitignore
git commit -m "feat: project scaffold with package structure and dev dependencies"
```

---

## Task 2: Configuration Module

**Files:**
- Create: `src/stellaris_advisor/config.py`
- Create: `tests/test_config.py`
- Create: `config.yaml`

- [ ] **Step 1: Write failing tests for config**

```python
# tests/test_config.py
"""Tests for configuration module."""

import pytest
from pathlib import Path


@pytest.mark.unit
class TestConfig:
    def test_default_config_creates_successfully(self):
        from stellaris_advisor.config import Config

        config = Config.default()
        assert config.stellaris.supported_versions == ("4.3",)
        assert config.parser.cache_parsed is True
        assert config.planner.alpha == 0.6
        assert config.planner.beta == 0.4
        assert config.rl.algorithm == "cql"

    def test_config_is_immutable(self):
        from stellaris_advisor.config import Config

        config = Config.default()
        with pytest.raises(Exception):
            config.planner.alpha = 0.9  # type: ignore[misc]

    def test_config_from_yaml(self, tmp_path: Path):
        from stellaris_advisor.config import Config

        yaml_content = """
stellaris:
  supported_versions: ["4.3"]
planner:
  alpha: 0.7
  beta: 0.3
  top_n_recommendations: 10
"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content)

        config = Config.from_yaml(yaml_file)
        assert config.planner.alpha == 0.7
        assert config.planner.beta == 0.3
        assert config.planner.top_n_recommendations == 10
        # Defaults still apply for unset fields
        assert config.parser.cache_parsed is True

    def test_config_from_yaml_missing_file_raises(self):
        from stellaris_advisor.config import Config

        with pytest.raises(FileNotFoundError):
            Config.from_yaml(Path("/nonexistent/config.yaml"))

    def test_reward_weights_accessible(self):
        from stellaris_advisor.config import Config

        config = Config.default()
        assert config.reward.fleet_power_weight == 0.25
        assert config.reward.tech_count_weight == 0.25
        assert config.reward.total_income_weight == 0.30
        assert config.reward.pop_count_weight == 0.20
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement config module**

```python
# src/stellaris_advisor/config.py
"""Configuration schema for stellaris-advisor. All models are frozen (immutable)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class StellarisDirConfig(BaseModel, frozen=True):
    install_dir: Path | None = None
    supported_versions: tuple[str, ...] = ("4.3",)
    save_dir: Path | None = None


class ParserConfig(BaseModel, frozen=True):
    cache_parsed: bool = True
    cache_dir: Path = Path("~/.stellaris-advisor/cache/")
    empire_only: bool = False


class DataConfig(BaseModel, frozen=True):
    processed_dir: Path = Path("data/processed/")
    raw_dir: Path = Path("data/raw/")
    parquet_compression: str = "snappy"
    temporal_lookback: int = 5


class RewardConfig(BaseModel, frozen=True):
    fleet_power_weight: float = 0.25
    tech_count_weight: float = 0.25
    total_income_weight: float = 0.30
    pop_count_weight: float = 0.20
    lookahead_years: int = 50


class RLConfig(BaseModel, frozen=True):
    algorithm: str = "cql"
    learning_rate: float = 3e-4
    batch_size: int = 256
    n_epochs: int = 100
    gamma: float = 0.99
    cql_alpha: float = 1.0
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    holdout_galaxy_configs: tuple[str, ...] = ("1000_grand_admiral",)


class PlannerConfig(BaseModel, frozen=True):
    alpha: float = 0.6
    beta: float = 0.4
    top_n_recommendations: int = 7
    top_n_colony_candidates: int = 5
    naval_utilization_threshold: float = 0.7
    threat_power_ratio: float = 1.5
    discount_rate: float = 0.05
    ascension_path_deadline_year: int = 2300
    megastructure_readiness_year: int = 2300
    megastructure_alloy_threshold: float = 200.0
    influence_crisis_threshold: float = 2.0
    faction_approval_warning: float = 20.0
    agenda_switch_progress_threshold: float = 0.7


class BenchmarkConfig(BaseModel, frozen=True):
    year_bucket_size: int = 25
    percentiles: tuple[int, ...] = (25, 50, 75, 90)


class Config(BaseModel, frozen=True):
    stellaris: StellarisDirConfig = StellarisDirConfig()
    parser: ParserConfig = ParserConfig()
    data: DataConfig = DataConfig()
    reward: RewardConfig = RewardConfig()
    rl: RLConfig = RLConfig()
    planner: PlannerConfig = PlannerConfig()
    benchmark: BenchmarkConfig = BenchmarkConfig()

    @classmethod
    def from_yaml(cls, path: Path) -> Config:
        """Load config from a YAML file, merging with defaults."""
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Config file not found: {resolved}")
        with open(resolved) as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        return cls.model_validate(raw)

    @classmethod
    def default(cls) -> Config:
        """Return config with all defaults."""
        return cls()
```

- [ ] **Step 4: Create default config.yaml**

```yaml
# config.yaml — Stellaris AI Advisor default configuration
# See docs/superpowers/specs/2026-03-27-stellaris-advisor-design.md for details

stellaris:
  install_dir: null
  supported_versions: ["4.3"]
  save_dir: null

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
  learning_rate: 3.0e-4
  batch_size: 256
  n_epochs: 100
  gamma: 0.99
  cql_alpha: 1.0
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15

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

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: All 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/stellaris_advisor/config.py tests/test_config.py config.yaml
git commit -m "feat: add Pydantic v2 config module with YAML loading"
```

---

## Task 3: Clausewitz Tokenizer

**Files:**
- Create: `src/stellaris_advisor/parser/clausewitz.py`
- Create: `tests/test_parser/__init__.py`
- Create: `tests/test_parser/test_clausewitz.py`

- [ ] **Step 1: Write failing tests for tokenizer**

```python
# tests/test_parser/__init__.py
# (empty)
```

```python
# tests/test_parser/test_clausewitz.py
"""Tests for Clausewitz format tokenizer and parser."""

import pytest
from hypothesis import given, strategies as st


@pytest.mark.unit
class TestTokenizer:
    def test_simple_key_value(self):
        from stellaris_advisor.parser.clausewitz import TokenType, tokenize

        tokens = list(tokenize('name="Earth"'))
        assert len(tokens) == 4  # IDENTIFIER, EQUALS, STRING, EOF
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "name"
        assert tokens[1].type == TokenType.EQUALS
        assert tokens[2].type == TokenType.STRING
        assert tokens[2].value == "Earth"
        assert tokens[3].type == TokenType.EOF

    def test_integer_value(self):
        from stellaris_advisor.parser.clausewitz import TokenType, tokenize

        tokens = list(tokenize("size=25"))
        assert tokens[0].value == "size"
        assert tokens[2].type == TokenType.NUMBER
        assert tokens[2].value == 25

    def test_float_value(self):
        from stellaris_advisor.parser.clausewitz import TokenType, tokenize

        tokens = list(tokenize("energy=142.50"))
        assert tokens[2].type == TokenType.NUMBER
        assert tokens[2].value == pytest.approx(142.50)

    def test_negative_number(self):
        from stellaris_advisor.parser.clausewitz import TokenType, tokenize

        tokens = list(tokenize("modifier=-0.25"))
        assert tokens[2].type == TokenType.NUMBER
        assert tokens[2].value == pytest.approx(-0.25)

    def test_boolean_yes(self):
        from stellaris_advisor.parser.clausewitz import TokenType, tokenize

        tokens = list(tokenize("is_active=yes"))
        assert tokens[2].type == TokenType.BOOL
        assert tokens[2].value is True

    def test_boolean_no(self):
        from stellaris_advisor.parser.clausewitz import TokenType, tokenize

        tokens = list(tokenize("is_active=no"))
        assert tokens[2].type == TokenType.BOOL
        assert tokens[2].value is False

    def test_braces(self):
        from stellaris_advisor.parser.clausewitz import TokenType, tokenize

        tokens = list(tokenize("block={ }"))
        types = [t.type for t in tokens]
        assert TokenType.LBRACE in types
        assert TokenType.RBRACE in types

    def test_comment_skipped(self):
        from stellaris_advisor.parser.clausewitz import tokenize

        tokens = list(tokenize("# this is a comment\nname=\"Earth\""))
        non_eof = [t for t in tokens if t.type.name != "EOF"]
        assert len(non_eof) == 3  # name, =, "Earth"

    def test_inline_comment_skipped(self):
        from stellaris_advisor.parser.clausewitz import tokenize

        tokens = list(tokenize("name=\"Earth\" # inline comment"))
        non_eof = [t for t in tokens if t.type.name != "EOF"]
        assert len(non_eof) == 3

    def test_multiline(self):
        from stellaris_advisor.parser.clausewitz import tokenize

        text = """name="Earth"
size=25
is_capital=yes"""
        tokens = list(tokenize(text))
        non_eof = [t for t in tokens if t.type.name != "EOF"]
        assert len(non_eof) == 9  # 3 key=value pairs × 3 tokens each

    def test_integer_key(self):
        from stellaris_advisor.parser.clausewitz import TokenType, tokenize

        tokens = list(tokenize("0={ }"))
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "0"

    def test_empty_string(self):
        from stellaris_advisor.parser.clausewitz import TokenType, tokenize

        tokens = list(tokenize('name=""'))
        assert tokens[2].type == TokenType.STRING
        assert tokens[2].value == ""

    def test_string_with_spaces(self):
        from stellaris_advisor.parser.clausewitz import tokenize

        tokens = list(tokenize('name="United Nations of Earth"'))
        assert tokens[2].value == "United Nations of Earth"

    def test_unquoted_identifier_value(self):
        from stellaris_advisor.parser.clausewitz import TokenType, tokenize

        tokens = list(tokenize("type=gov_democratic"))
        assert tokens[2].type == TokenType.IDENTIFIER
        assert tokens[2].value == "gov_democratic"

    def test_line_tracking(self):
        from stellaris_advisor.parser.clausewitz import tokenize

        text = "a=1\nb=2\nc=3"
        tokens = list(tokenize(text))
        lines = [t.line for t in tokens if t.value in ("a", "b", "c")]
        assert lines == [1, 2, 3]

    def test_empty_input(self):
        from stellaris_advisor.parser.clausewitz import TokenType, tokenize

        tokens = list(tokenize(""))
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF

    def test_tabs_and_mixed_whitespace(self):
        from stellaris_advisor.parser.clausewitz import tokenize

        text = "\t name = \t \"Earth\" \t"
        tokens = list(tokenize(text))
        non_eof = [t for t in tokens if t.type.name != "EOF"]
        assert len(non_eof) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_parser/test_clausewitz.py::TestTokenizer -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement tokenizer**

```python
# src/stellaris_advisor/parser/clausewitz.py
"""Clausewitz format tokenizer and recursive descent parser.

Handles the Paradox Clausewitz serialization format used by Stellaris
save files and game definition files.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Iterator


class TokenType(Enum):
    IDENTIFIER = auto()
    STRING = auto()
    NUMBER = auto()
    BOOL = auto()
    EQUALS = auto()
    LBRACE = auto()
    RBRACE = auto()
    EOF = auto()


@dataclass(frozen=True)
class Token:
    type: TokenType
    value: str | int | float | bool
    line: int
    col: int


class TokenizeError(Exception):
    """Raised when the tokenizer encounters invalid input."""

    def __init__(self, message: str, line: int, col: int) -> None:
        super().__init__(f"Line {line}, col {col}: {message}")
        self.line = line
        self.col = col


def tokenize(text: str) -> Iterator[Token]:
    """Tokenize Clausewitz format text into a stream of tokens.

    Yields Token objects. The final token is always EOF.
    """
    pos = 0
    line = 1
    col = 1
    length = len(text)

    while pos < length:
        ch = text[pos]

        # Skip whitespace
        if ch in (" ", "\t", "\r"):
            pos += 1
            col += 1
            continue

        # Newline
        if ch == "\n":
            pos += 1
            line += 1
            col = 1
            continue

        # Comment — skip to end of line
        if ch == "#":
            while pos < length and text[pos] != "\n":
                pos += 1
            continue

        # Single-character tokens
        if ch == "=":
            yield Token(TokenType.EQUALS, "=", line, col)
            pos += 1
            col += 1
            continue

        if ch == "{":
            yield Token(TokenType.LBRACE, "{", line, col)
            pos += 1
            col += 1
            continue

        if ch == "}":
            yield Token(TokenType.RBRACE, "}", line, col)
            pos += 1
            col += 1
            continue

        # Quoted string
        if ch == '"':
            start_col = col
            pos += 1
            col += 1
            chars: list[str] = []
            while pos < length and text[pos] != '"':
                if text[pos] == "\\":
                    pos += 1
                    col += 1
                    if pos < length:
                        escape_ch = text[pos]
                        if escape_ch == "n":
                            chars.append("\n")
                        elif escape_ch == "t":
                            chars.append("\t")
                        elif escape_ch == '"':
                            chars.append('"')
                        elif escape_ch == "\\":
                            chars.append("\\")
                        else:
                            chars.append(escape_ch)
                elif text[pos] == "\n":
                    chars.append("\n")
                    line += 1
                    col = 0
                else:
                    chars.append(text[pos])
                pos += 1
                col += 1
            if pos >= length:
                raise TokenizeError("Unterminated string", line, start_col)
            pos += 1  # skip closing quote
            col += 1
            yield Token(TokenType.STRING, "".join(chars), line, start_col)
            continue

        # Number or identifier starting with digit/minus
        if ch == "-" or ch.isdigit():
            start_col = col
            start_pos = pos
            if ch == "-":
                pos += 1
                col += 1
                if pos >= length or not text[pos].isdigit():
                    # Bare minus — treat as identifier
                    while pos < length and _is_ident_char(text[pos]):
                        pos += 1
                        col += 1
                    word = text[start_pos:pos]
                    yield Token(TokenType.IDENTIFIER, word, line, start_col)
                    continue

            # Read digits
            while pos < length and text[pos].isdigit():
                pos += 1
                col += 1

            is_float = False
            if pos < length and text[pos] == ".":
                is_float = True
                pos += 1
                col += 1
                while pos < length and text[pos].isdigit():
                    pos += 1
                    col += 1

            # Check if followed by identifier chars (e.g., "4.3.0" version string)
            if pos < length and _is_ident_char(text[pos]) and text[pos] != "." :
                while pos < length and _is_ident_char(text[pos]):
                    pos += 1
                    col += 1
                word = text[start_pos:pos]
                yield Token(TokenType.IDENTIFIER, word, line, start_col)
                continue

            num_str = text[start_pos:pos]
            if is_float:
                yield Token(TokenType.NUMBER, float(num_str), line, start_col)
            else:
                yield Token(TokenType.NUMBER, int(num_str), line, start_col)
            continue

        # Identifier (including keywords yes/no)
        if _is_ident_start(ch):
            start_col = col
            start_pos = pos
            while pos < length and _is_ident_char(text[pos]):
                pos += 1
                col += 1
            word = text[start_pos:pos]
            if word == "yes":
                yield Token(TokenType.BOOL, True, line, start_col)
            elif word == "no":
                yield Token(TokenType.BOOL, False, line, start_col)
            else:
                yield Token(TokenType.IDENTIFIER, word, line, start_col)
            continue

        raise TokenizeError(f"Unexpected character: {ch!r}", line, col)

    yield Token(TokenType.EOF, "", line, col)


def _is_ident_start(ch: str) -> bool:
    return ch.isalpha() or ch == "_"


def _is_ident_char(ch: str) -> bool:
    return ch.isalnum() or ch in ("_", ".", ":", "-", "@", "'")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_parser/test_clausewitz.py::TestTokenizer -v`
Expected: All 17 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/stellaris_advisor/parser/clausewitz.py tests/test_parser/
git commit -m "feat: implement Clausewitz format tokenizer"
```

---

## Task 4: Clausewitz Recursive Descent Parser

**Files:**
- Modify: `src/stellaris_advisor/parser/clausewitz.py`
- Modify: `tests/test_parser/test_clausewitz.py`

- [ ] **Step 1: Write failing tests for parser**

Add to `tests/test_parser/test_clausewitz.py`:

```python
class ParseError(Exception):
    pass


@pytest.mark.unit
class TestParser:
    def test_simple_key_value(self):
        from stellaris_advisor.parser.clausewitz import parse

        result = parse('name="Earth"')
        assert result == {"name": "Earth"}

    def test_integer_value(self):
        from stellaris_advisor.parser.clausewitz import parse

        result = parse("size=25")
        assert result == {"size": 25}

    def test_float_value(self):
        from stellaris_advisor.parser.clausewitz import parse

        result = parse("energy=142.50")
        assert result["energy"] == pytest.approx(142.50)

    def test_boolean_value(self):
        from stellaris_advisor.parser.clausewitz import parse

        result = parse("is_active=yes\nis_blocked=no")
        assert result == {"is_active": True, "is_blocked": False}

    def test_nested_block(self):
        from stellaris_advisor.parser.clausewitz import parse

        text = """government={
    type="gov_democratic"
    authority="auth_democratic"
}"""
        result = parse(text)
        assert result["government"]["type"] == "gov_democratic"
        assert result["government"]["authority"] == "auth_democratic"

    def test_deeply_nested(self):
        from stellaris_advisor.parser.clausewitz import parse

        text = """a={
    b={
        c={
            d=42
        }
    }
}"""
        result = parse(text)
        assert result["a"]["b"]["c"]["d"] == 42

    def test_bare_value_list_strings(self):
        from stellaris_advisor.parser.clausewitz import parse

        text = 'civics={ "civic_beacon_of_liberty" "civic_idealistic_foundation" }'
        result = parse(text)
        assert result["civics"] == ["civic_beacon_of_liberty", "civic_idealistic_foundation"]

    def test_bare_value_list_integers(self):
        from stellaris_advisor.parser.clausewitz import parse

        text = "owned_planets={ 0 4 12 27 }"
        result = parse(text)
        assert result["owned_planets"] == [0, 4, 12, 27]

    def test_empty_block(self):
        from stellaris_advisor.parser.clausewitz import parse

        result = parse("empty={ }")
        assert result["empty"] == {}

    def test_duplicate_keys_become_list(self):
        from stellaris_advisor.parser.clausewitz import parse

        text = """weight_modifier={
    factor=1.5
}
weight_modifier={
    factor=2.0
}"""
        result = parse(text)
        assert isinstance(result["weight_modifier"], list)
        assert len(result["weight_modifier"]) == 2
        assert result["weight_modifier"][0]["factor"] == pytest.approx(1.5)
        assert result["weight_modifier"][1]["factor"] == pytest.approx(2.0)

    def test_integer_keys(self):
        from stellaris_advisor.parser.clausewitz import parse

        text = """technology={
    0={
        technology="tech_corvettes"
        level=1
    }
    1={
        technology="tech_destroyers"
        level=1
    }
}"""
        result = parse(text)
        assert result["technology"]["0"]["technology"] == "tech_corvettes"
        assert result["technology"]["1"]["technology"] == "tech_destroyers"

    def test_multiple_top_level_keys(self):
        from stellaris_advisor.parser.clausewitz import parse

        text = """name="Earth"
size=25
planet_class="pc_continental"
is_capital=yes"""
        result = parse(text)
        assert result["name"] == "Earth"
        assert result["size"] == 25
        assert result["planet_class"] == "pc_continental"
        assert result["is_capital"] is True

    def test_identifier_value_unquoted(self):
        from stellaris_advisor.parser.clausewitz import parse

        result = parse("authority=auth_democratic")
        assert result["authority"] == "auth_democratic"

    def test_mixed_content(self):
        from stellaris_advisor.parser.clausewitz import parse

        text = """country={
    0={
        name="United Nations of Earth"
        government={
            type="gov_democratic"
            civics={ "civic_beacon_of_liberty" "civic_idealistic_foundation" }
        }
        owned_planets={ 0 4 12 }
        fleet_power=8200.50
        is_player=yes
    }
}"""
        result = parse(text)
        empire = result["country"]["0"]
        assert empire["name"] == "United Nations of Earth"
        assert empire["government"]["type"] == "gov_democratic"
        assert empire["government"]["civics"] == [
            "civic_beacon_of_liberty",
            "civic_idealistic_foundation",
        ]
        assert empire["owned_planets"] == [0, 4, 12]
        assert empire["fleet_power"] == pytest.approx(8200.50)
        assert empire["is_player"] is True

    def test_empty_input(self):
        from stellaris_advisor.parser.clausewitz import parse

        result = parse("")
        assert result == {}

    def test_comment_only(self):
        from stellaris_advisor.parser.clausewitz import parse

        result = parse("# just a comment\n# another comment")
        assert result == {}

    def test_bare_value_list_with_identifiers(self):
        from stellaris_advisor.parser.clausewitz import parse

        text = "ethics={ ethic_fanatic_egalitarian ethic_xenophile }"
        result = parse(text)
        assert result["ethics"] == ["ethic_fanatic_egalitarian", "ethic_xenophile"]

    def test_single_element_list(self):
        from stellaris_advisor.parser.clausewitz import parse

        text = "planets={ 7 }"
        result = parse(text)
        # Single element inside braces with no = → list with one item
        assert result["planets"] == [7]

    def test_triple_duplicate_keys(self):
        from stellaris_advisor.parser.clausewitz import parse

        text = "a=1\na=2\na=3"
        result = parse(text)
        assert result["a"] == [1, 2, 3]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_parser/test_clausewitz.py::TestParser -v`
Expected: FAIL (ImportError — `parse` not defined)

- [ ] **Step 3: Implement recursive descent parser**

Add to `src/stellaris_advisor/parser/clausewitz.py`:

```python
class ParseError(Exception):
    """Raised when the parser encounters invalid structure."""

    def __init__(self, message: str, line: int, col: int) -> None:
        super().__init__(f"Line {line}, col {col}: {message}")
        self.line = line
        self.col = col


def parse(text: str) -> dict[str, Any]:
    """Parse Clausewitz format text into a nested Python dict.

    Handles:
    - key = value (scalars: str, int, float, bool, identifier)
    - key = { ... } (nested blocks → dict)
    - key = { val1 val2 ... } (bare value lists → list)
    - Duplicate keys at same level → collected into a list
    - Integer keys → stored as string keys
    """
    tokens = list(tokenize(text))
    parser = _Parser(tokens)
    return parser.parse_top_level()


class _Parser:
    """Stateful recursive descent parser over a token list."""

    def __init__(self, tokens: list[Token]) -> None:
        self._tokens = tokens
        self._pos = 0

    def _peek(self) -> Token:
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return Token(TokenType.EOF, "", 0, 0)

    def _advance(self) -> Token:
        token = self._peek()
        self._pos += 1
        return token

    def _expect(self, token_type: TokenType) -> Token:
        token = self._advance()
        if token.type != token_type:
            raise ParseError(
                f"Expected {token_type.name}, got {token.type.name} ({token.value!r})",
                token.line,
                token.col,
            )
        return token

    def parse_top_level(self) -> dict[str, Any]:
        """Parse the top level as a series of key=value pairs."""
        return self._parse_pairs(top_level=True)

    def _parse_pairs(self, top_level: bool = False) -> dict[str, Any]:
        """Parse key=value pairs until RBRACE or EOF."""
        result: dict[str, Any] = {}
        stop_type = TokenType.EOF if top_level else TokenType.RBRACE

        while True:
            token = self._peek()
            if token.type == stop_type:
                if not top_level:
                    self._advance()  # consume RBRACE
                return result
            if token.type == TokenType.EOF and not top_level:
                raise ParseError("Unexpected end of input in block", token.line, token.col)

            # Read the key
            key_token = self._advance()
            if key_token.type not in (TokenType.IDENTIFIER, TokenType.NUMBER, TokenType.STRING):
                raise ParseError(
                    f"Expected key, got {key_token.type.name} ({key_token.value!r})",
                    key_token.line,
                    key_token.col,
                )
            key = str(key_token.value)

            # Expect =
            self._expect(TokenType.EQUALS)

            # Read the value
            value = self._parse_value()

            # Handle duplicate keys
            if key in result:
                existing = result[key]
                if isinstance(existing, list) and not self._is_parsed_list(existing, key, result):
                    existing.append(value)
                else:
                    result[key] = [existing, value]
            else:
                result[key] = value

        return result

    def _is_parsed_list(self, value: list[Any], key: str, result: dict[str, Any]) -> bool:
        """Distinguish between a bare-value list and a duplicate-key list.

        A bare-value list was created by _parse_block_or_list.
        A duplicate-key list was created by us in _parse_pairs.
        We track this by checking if the list was assigned directly (bare) vs built up.

        Heuristic: if the value is a list and the first time we see this key,
        it means it came from a bare-value list parse. We use a sentinel approach:
        we wrap duplicate-key lists in a special way. Actually, simpler: we check
        if the existing value is a list whose elements are the types we'd see
        from bare value lists vs block parses.

        Simplest correct approach: always box duplicates into a new list.
        """
        return False

    def _parse_value(self) -> Any:
        """Parse a single value (scalar or block)."""
        token = self._peek()
        if token.type == TokenType.LBRACE:
            self._advance()  # consume {
            return self._parse_block_or_list()
        else:
            return self._advance().value

    def _parse_block_or_list(self) -> dict[str, Any] | list[Any]:
        """After consuming '{', determine if this is a dict or list and parse."""
        # Empty block
        if self._peek().type == TokenType.RBRACE:
            self._advance()
            return {}

        # Look at first token and second token to decide
        first = self._peek()

        # If first token is RBRACE, empty (already handled above)
        # If first is LBRACE, it's a list of nested blocks (rare but possible)

        # Peek two ahead: if pattern is KEY = VALUE, it's a dict
        # Otherwise it's a bare value list
        if self._is_dict_pattern():
            return self._parse_pairs(top_level=False)
        else:
            return self._parse_bare_list()

    def _is_dict_pattern(self) -> bool:
        """Look ahead to determine if current position starts key=value pairs."""
        # Save position
        saved = self._pos

        # Read potential key
        first = self._peek()
        if first.type not in (TokenType.IDENTIFIER, TokenType.NUMBER, TokenType.STRING):
            self._pos = saved
            return False

        # Check if next-next is EQUALS
        if self._pos + 1 < len(self._tokens):
            second = self._tokens[self._pos + 1]
            self._pos = saved
            return second.type == TokenType.EQUALS

        self._pos = saved
        return False

    def _parse_bare_list(self) -> list[Any]:
        """Parse a bare value list: { val1 val2 val3 ... }"""
        values: list[Any] = []
        while self._peek().type != TokenType.RBRACE:
            if self._peek().type == TokenType.EOF:
                raise ParseError(
                    "Unexpected end of input in list",
                    self._peek().line,
                    self._peek().col,
                )
            token = self._advance()
            if token.type == TokenType.LBRACE:
                values.append(self._parse_block_or_list())
            else:
                values.append(token.value)
        self._advance()  # consume RBRACE
        return values
```

Also fix the `_is_parsed_list` method and duplicate key handling. Replace the `_parse_pairs` duplicate key section with:

```python
            # Handle duplicate keys
            if key in result:
                existing = result[key]
                if isinstance(existing, list) and _is_duplicate_list(existing):
                    existing.append(value)
                else:
                    result[key] = _DuplicateList([existing, value])
            else:
                result[key] = value
```

And add a helper class above `_Parser`:

```python
class _DuplicateList(list):
    """Marker subclass to distinguish duplicate-key lists from bare-value lists."""
    pass


def _is_duplicate_list(value: Any) -> bool:
    return isinstance(value, _DuplicateList)
```

And add a post-processing step to `parse()`:

```python
def parse(text: str) -> dict[str, Any]:
    tokens = list(tokenize(text))
    parser = _Parser(tokens)
    result = parser.parse_top_level()
    return _normalize(result)


def _normalize(value: Any) -> Any:
    """Convert _DuplicateList markers back to plain lists. Recurse into dicts."""
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items()}
    if isinstance(value, _DuplicateList):
        return [_normalize(item) for item in value]
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_parser/test_clausewitz.py -v`
Expected: All tokenizer + parser tests PASS

- [ ] **Step 5: Add hypothesis property-based tests**

Add to `tests/test_parser/test_clausewitz.py`:

```python
@pytest.mark.unit
class TestParserProperties:
    @given(st.text(alphabet=st.characters(whitelist_categories=("L", "N")), min_size=1, max_size=20))
    def test_identifier_roundtrip(self, ident: str):
        """Any alphanumeric string used as a key should parse back."""
        from stellaris_advisor.parser.clausewitz import parse

        # Only test valid identifiers (start with letter or underscore)
        if not (ident[0].isalpha() or ident[0] == "_"):
            return
        if ident in ("yes", "no"):
            return  # reserved words

        text = f'{ident}="value"'
        result = parse(text)
        assert ident in result
        assert result[ident] == "value"

    @given(st.integers(min_value=-999999, max_value=999999))
    def test_integer_roundtrip(self, n: int):
        """Integers should parse back to the same value."""
        from stellaris_advisor.parser.clausewitz import parse

        text = f"key={n}"
        result = parse(text)
        assert result["key"] == n

    @given(st.lists(st.integers(min_value=0, max_value=9999), min_size=0, max_size=20))
    def test_integer_list_roundtrip(self, nums: list[int]):
        """Integer lists should preserve all elements."""
        from stellaris_advisor.parser.clausewitz import parse

        inner = " ".join(str(n) for n in nums)
        text = f"items={{ {inner} }}"
        result = parse(text)
        if len(nums) == 0:
            assert result["items"] == {}  # empty block
        else:
            assert result["items"] == nums
```

- [ ] **Step 6: Run all tests**

Run: `pytest tests/test_parser/test_clausewitz.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/stellaris_advisor/parser/clausewitz.py tests/test_parser/test_clausewitz.py
git commit -m "feat: implement Clausewitz recursive descent parser"
```

---

## Task 5: Parser Data Models — Core Types

**Files:**
- Create: `src/stellaris_advisor/parser/models.py`
- Create: `tests/test_parser/test_models.py`

- [ ] **Step 1: Write failing tests for data models**

```python
# tests/test_parser/test_models.py
"""Tests for parser data models (frozen dataclasses)."""

import pytest
from types import MappingProxyType


@pytest.mark.unit
class TestEmpireState:
    def test_empire_state_is_frozen(self):
        from stellaris_advisor.parser.models import EmpireState

        empire = EmpireState(
            id=0,
            name="Test Empire",
            authority="auth_democratic",
            ethics=("ethic_fanatic_egalitarian", "ethic_xenophile"),
            civics=("civic_beacon_of_liberty",),
            origin="origin_prosperous_unification",
            government_type="gov_democratic",
            resource_income=MappingProxyType({"energy": 100.0}),
            resource_expense=MappingProxyType({"energy": 80.0}),
            resource_stockpile=MappingProxyType({"energy": 500.0}),
            resource_storage_cap=MappingProxyType({"energy": 1000.0}),
            researched_techs=frozenset({"tech_corvettes"}),
            current_research=MappingProxyType({"physics": "tech_lasers_2"}),
            research_progress=MappingProxyType({"physics": 0.5}),
            research_speed=MappingProxyType({"physics": 50.0}),
            stored_research=MappingProxyType({"physics": 0.0}),
            fleet_power=5000.0,
            naval_capacity_used=60,
            naval_capacity_max=120,
            fleets=(),
            owned_systems=(0, 1, 2),
            owned_planets=(),
            starbases=(),
            total_pops=50,
            pop_groups=(),
            rivals=(3,),
            defensive_pacts=(),
            federation_id=None,
            federation_power=None,
            subjects=(),
            overlord=None,
            traditions_unlocked=(),
            tradition_trees_opened=(),
            tradition_trees_completed=(),
            ascension_perks=(),
            ascension_perk_slots_available=0,
            ascension_perk_slots_total=0,
            ascension_path=None,
            ascension_situation_progress=None,
            leaders=(),
            council_positions=(),
            current_agenda=None,
            council_legitimacy=50.0,
            factions=(),
            policies=MappingProxyType({}),
            galcom_member=False,
            galcom_diplomatic_weight=0.0,
            galcom_council_member=False,
            galcom_custodian=False,
            galcom_emperor=False,
            active_resolutions=(),
            federation=None,
            megastructures=(),
            megastructures_under_construction=0,
            active_wars=(),
            claims_placed=(),
            total_army_strength=0.0,
            victory_score=100.0,
            victory_rank=3,
        )
        assert empire.name == "Test Empire"
        assert "tech_corvettes" in empire.researched_techs
        with pytest.raises(AttributeError):
            empire.name = "Modified"  # type: ignore[misc]

    def test_empire_techs_are_frozenset(self):
        from stellaris_advisor.parser.models import EmpireState

        empire = _make_test_empire(researched_techs=frozenset({"tech_a", "tech_b"}))
        assert isinstance(empire.researched_techs, frozenset)
        assert "tech_a" in empire.researched_techs

    def test_empire_resources_are_immutable(self):
        from stellaris_advisor.parser.models import EmpireState

        empire = _make_test_empire(
            resource_income=MappingProxyType({"energy": 100.0, "minerals": 50.0})
        )
        with pytest.raises(TypeError):
            empire.resource_income["energy"] = 200.0  # type: ignore[index]


@pytest.mark.unit
class TestPlanetState:
    def test_planet_state_is_frozen(self):
        from stellaris_advisor.parser.models import PlanetState

        planet = PlanetState(
            id=0,
            name="Earth",
            planet_class="pc_continental",
            size=25,
            districts=MappingProxyType({"district_city": 3}),
            district_slots=MappingProxyType({"district_city": 10}),
            total_district_cap=16,
            districts_used=5,
            buildings=("building_capital_2", "building_alloy_foundry"),
            building_slots_total=8,
            building_slots_used=2,
            district_specializations=MappingProxyType({}),
            pops=30,
            pop_groups=(),
            stability=55.0,
            crime=5.0,
            amenities=10.0,
            amenities_usage=8.0,
            housing=40.0,
            housing_usage=30.0,
            designation="col_default",
            planetary_features=(),
            is_capital=True,
        )
        assert planet.name == "Earth"
        assert planet.is_capital is True
        with pytest.raises(AttributeError):
            planet.name = "Mars"  # type: ignore[misc]


@pytest.mark.unit
class TestSystemState:
    def test_system_state_is_frozen(self):
        from stellaris_advisor.parser.models import SystemState

        system = SystemState(
            id=0,
            name="Sol",
            star_class="sc_g",
            hyperlane_connections=(1, 2, 3),
            owner=0,
            claimed_by=(),
            starbase=None,
            planets=(),
            resources=MappingProxyType({}),
            has_megastructure=False,
            megastructure_type=None,
        )
        assert system.name == "Sol"


def _make_test_empire(**overrides):
    """Factory for creating test EmpireState with sensible defaults."""
    from stellaris_advisor.parser.models import EmpireState

    defaults = dict(
        id=0,
        name="Test Empire",
        authority="auth_democratic",
        ethics=(),
        civics=(),
        origin="origin_default",
        government_type="gov_democratic",
        resource_income=MappingProxyType({}),
        resource_expense=MappingProxyType({}),
        resource_stockpile=MappingProxyType({}),
        resource_storage_cap=MappingProxyType({}),
        researched_techs=frozenset(),
        current_research=MappingProxyType({}),
        research_progress=MappingProxyType({}),
        research_speed=MappingProxyType({}),
        stored_research=MappingProxyType({}),
        fleet_power=0.0,
        naval_capacity_used=0,
        naval_capacity_max=0,
        fleets=(),
        owned_systems=(),
        owned_planets=(),
        starbases=(),
        total_pops=0,
        pop_groups=(),
        rivals=(),
        defensive_pacts=(),
        federation_id=None,
        federation_power=None,
        subjects=(),
        overlord=None,
        traditions_unlocked=(),
        tradition_trees_opened=(),
        tradition_trees_completed=(),
        ascension_perks=(),
        ascension_perk_slots_available=0,
        ascension_perk_slots_total=0,
        ascension_path=None,
        ascension_situation_progress=None,
        leaders=(),
        council_positions=(),
        current_agenda=None,
        council_legitimacy=0.0,
        factions=(),
        policies=MappingProxyType({}),
        galcom_member=False,
        galcom_diplomatic_weight=0.0,
        galcom_council_member=False,
        galcom_custodian=False,
        galcom_emperor=False,
        active_resolutions=(),
        federation=None,
        megastructures=(),
        megastructures_under_construction=0,
        active_wars=(),
        claims_placed=(),
        total_army_strength=0.0,
        victory_score=0.0,
        victory_rank=0,
    )
    defaults.update(overrides)
    return EmpireState(**defaults)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_parser/test_models.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement all parser data models**

```python
# src/stellaris_advisor/parser/models.py
"""Typed data models for parsed Stellaris save state.

All dataclasses are frozen (immutable). Collections use tuple (not list),
frozenset (for sets), and MappingProxyType (for dicts).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType


# --- Supporting types ---


@dataclass(frozen=True)
class PopGroupState:
    species: str
    count: int
    happiness: float


@dataclass(frozen=True)
class PopGroupSummary:
    species: str
    count: int


@dataclass(frozen=True)
class LeaderState:
    id: int
    name: str
    leader_class: str  # "scientist", "admiral", "general", "governor", "official"
    level: int
    traits: tuple[str, ...]
    expertise: str | None  # for scientists: "expertise_voidcraft", etc.
    assignment: str | None  # current assignment description


@dataclass(frozen=True)
class CouncilPosition:
    position_type: str
    leader_id: int | None
    leader_name: str | None
    leader_level: int


@dataclass(frozen=True)
class AgendaState:
    agenda_type: str
    progress: float  # 0.0–1.0
    is_launched: bool


@dataclass(frozen=True)
class FactionIssueState:
    issue_id: str
    is_fulfilled: bool


@dataclass(frozen=True)
class FactionState:
    ethic: str
    approval: float  # 0–100
    size: int  # pop count
    support: float  # 0.0–1.0
    issues: tuple[FactionIssueState, ...]


@dataclass(frozen=True)
class FederationState:
    federation_type: str
    level: int  # 1–5
    cohesion: float
    experience: float
    fleet_power: float


@dataclass(frozen=True)
class MegastructureState:
    megastructure_type: str
    stage: int
    system_id: int


@dataclass(frozen=True)
class WarState:
    war_id: int
    name: str
    war_goal: str
    exhaustion: float  # 0.0–1.0
    is_attacker: bool


@dataclass(frozen=True)
class ClaimState:
    system_id: int
    influence_cost: int


@dataclass(frozen=True)
class LeaderSummary:
    id: int
    name: str
    level: int


@dataclass(frozen=True)
class StarbaseState:
    system_id: int
    level: str  # "outpost", "starport", "starhold", "star_fortress", "citadel"
    modules: tuple[str, ...]
    buildings: tuple[str, ...]
    defense_fleet_power: float


@dataclass(frozen=True)
class FleetState:
    id: int
    name: str
    fleet_power: float
    ships: MappingProxyType[str, int]  # ship_class → count
    naval_capacity_used: int
    location_system: int
    commander: LeaderSummary | None


@dataclass(frozen=True)
class PlanetSummary:
    id: int
    name: str
    planet_class: str
    size: int
    is_habitable: bool
    is_colonized: bool


@dataclass(frozen=True)
class PlanetState:
    id: int
    name: str
    planet_class: str
    size: int
    # Districts
    districts: MappingProxyType[str, int]  # district_type → count built
    district_slots: MappingProxyType[str, int]  # district_type → max
    total_district_cap: int
    districts_used: int
    # Buildings
    buildings: tuple[str, ...]  # building IDs in slots
    building_slots_total: int
    building_slots_used: int
    # Specializations (4.0+)
    district_specializations: MappingProxyType[str, str]  # district_type → spec_id
    # Population
    pops: int
    pop_groups: tuple[PopGroupSummary, ...]
    # Modifiers
    stability: float
    crime: float
    amenities: float
    amenities_usage: float
    housing: float
    housing_usage: float
    designation: str
    # Features
    planetary_features: tuple[str, ...]
    is_capital: bool


@dataclass(frozen=True)
class SystemState:
    id: int
    name: str
    star_class: str
    hyperlane_connections: tuple[int, ...]
    owner: int | None
    claimed_by: tuple[int, ...]
    starbase: StarbaseState | None
    planets: tuple[PlanetSummary, ...]
    resources: MappingProxyType[str, float]
    has_megastructure: bool
    megastructure_type: str | None


@dataclass(frozen=True)
class ColonyCandidate:
    system_id: int
    system_name: str
    planet_id: int
    planet_name: str
    planet_class: str
    size: int
    habitability: float
    district_slots: MappingProxyType[str, int]
    resource_deposits: MappingProxyType[str, float]
    distance_to_capital: int
    distance_to_hostile_border: int
    is_in_claimed_space: bool


@dataclass(frozen=True)
class EmpireSummary:
    id: int
    name: str
    authority: str
    ethics: tuple[str, ...]
    fleet_power: float
    tech_count: int
    system_count: int
    victory_score: float


# --- Main empire type ---


@dataclass(frozen=True)
class EmpireState:
    id: int
    name: str
    # Identity
    authority: str
    ethics: tuple[str, ...]
    civics: tuple[str, ...]
    origin: str
    government_type: str
    # Resources
    resource_income: MappingProxyType[str, float]
    resource_expense: MappingProxyType[str, float]
    resource_stockpile: MappingProxyType[str, float]
    resource_storage_cap: MappingProxyType[str, float]
    # Tech
    researched_techs: frozenset[str]
    current_research: MappingProxyType[str, str]
    research_progress: MappingProxyType[str, float]
    research_speed: MappingProxyType[str, float]
    stored_research: MappingProxyType[str, float]
    # Military
    fleet_power: float
    naval_capacity_used: int
    naval_capacity_max: int
    fleets: tuple[FleetState, ...]
    # Territory
    owned_systems: tuple[int, ...]
    owned_planets: tuple[PlanetState, ...]
    starbases: tuple[StarbaseState, ...]
    # Pops
    total_pops: int
    pop_groups: tuple[PopGroupState, ...]
    # Diplomacy
    rivals: tuple[int, ...]
    defensive_pacts: tuple[int, ...]
    federation_id: int | None
    federation_power: float | None
    subjects: tuple[int, ...]
    overlord: int | None
    # Traditions & Ascension
    traditions_unlocked: tuple[str, ...]
    tradition_trees_opened: tuple[str, ...]
    tradition_trees_completed: tuple[str, ...]
    ascension_perks: tuple[str, ...]
    ascension_perk_slots_available: int
    ascension_perk_slots_total: int
    ascension_path: str | None
    ascension_situation_progress: float | None
    # Leaders & Council
    leaders: tuple[LeaderState, ...]
    council_positions: tuple[CouncilPosition, ...]
    current_agenda: AgendaState | None
    council_legitimacy: float
    # Factions
    factions: tuple[FactionState, ...]
    # Policies
    policies: MappingProxyType[str, str]
    # Galactic Community
    galcom_member: bool
    galcom_diplomatic_weight: float
    galcom_council_member: bool
    galcom_custodian: bool
    galcom_emperor: bool
    active_resolutions: tuple[str, ...]
    # Federation
    federation: FederationState | None
    # Megastructures
    megastructures: tuple[MegastructureState, ...]
    megastructures_under_construction: int
    # Warfare
    active_wars: tuple[WarState, ...]
    claims_placed: tuple[ClaimState, ...]
    total_army_strength: float
    # Score
    victory_score: float
    victory_rank: int


# --- Top-level game state ---


@dataclass(frozen=True)
class GalacticState:
    year: int
    month: int
    game_version: str
    galaxy_size: int
    difficulty: str
    crisis_active: bool
    crisis_type: str | None
    war_in_heaven: bool
    galactic_market_founded: bool
    empires: tuple[EmpireSummary, ...]
    player_empire_id: int
    systems: tuple[SystemState, ...]
    player_empire: EmpireState
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_parser/test_models.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/stellaris_advisor/parser/models.py tests/test_parser/test_models.py
git commit -m "feat: add frozen dataclass models for parsed save state"
```

---

## Task 6: Version Detection and Path Navigation Helpers

**Files:**
- Create: `src/stellaris_advisor/parser/version.py`
- Create: `src/stellaris_advisor/parser/extractor.py` (helpers only, extraction in Task 7)
- Create: `tests/test_parser/test_version.py`
- Create: `tests/test_parser/test_extractor.py`

- [ ] **Step 1: Write failing tests for version detection**

```python
# tests/test_parser/test_version.py
"""Tests for version detection."""

import pytest


@pytest.mark.unit
class TestVersionDetection:
    def test_detect_version_from_meta(self):
        from stellaris_advisor.parser.version import detect_version

        meta = {"version": "4.3.0.0"}
        assert detect_version(meta) == "4.3"

    def test_detect_version_strips_patch(self):
        from stellaris_advisor.parser.version import detect_version

        meta = {"version": "4.3.2.1"}
        assert detect_version(meta) == "4.3"

    def test_unsupported_version_raises(self):
        from stellaris_advisor.parser.version import UnsupportedVersionError, detect_version

        meta = {"version": "3.12.0.0"}
        with pytest.raises(UnsupportedVersionError, match="3.12"):
            detect_version(meta, supported=("4.3",))

    def test_missing_version_key_raises(self):
        from stellaris_advisor.parser.version import detect_version

        with pytest.raises(KeyError):
            detect_version({})
```

- [ ] **Step 2: Write failing tests for path navigation**

```python
# tests/test_parser/test_extractor.py
"""Tests for extraction helpers."""

import pytest


@pytest.mark.unit
class TestGetPath:
    def test_simple_path(self):
        from stellaris_advisor.parser.extractor import get_path

        data = {"a": {"b": {"c": 42}}}
        assert get_path(data, "a.b.c") == 42

    def test_missing_path_returns_default(self):
        from stellaris_advisor.parser.extractor import get_path

        data = {"a": {"b": 1}}
        assert get_path(data, "a.x.y", default=0) == 0

    def test_integer_index_in_path(self):
        from stellaris_advisor.parser.extractor import get_path

        data = {"country": {"0": {"name": "Earth"}}}
        assert get_path(data, "country.0.name") == "Earth"

    def test_top_level_key(self):
        from stellaris_advisor.parser.extractor import get_path

        data = {"version": "4.3.0"}
        assert get_path(data, "version") == "4.3.0"

    def test_empty_path_returns_data(self):
        from stellaris_advisor.parser.extractor import get_path

        data = {"a": 1}
        assert get_path(data, "") == data

    def test_none_intermediate_returns_default(self):
        from stellaris_advisor.parser.extractor import get_path

        data = {"a": None}
        assert get_path(data, "a.b", default="missing") == "missing"


@pytest.mark.unit
class TestSafeConversions:
    def test_safe_float(self):
        from stellaris_advisor.parser.extractor import safe_float

        assert safe_float(42) == 42.0
        assert safe_float("3.14") == pytest.approx(3.14)
        assert safe_float(None, default=0.0) == 0.0
        assert safe_float("not_a_number", default=-1.0) == -1.0

    def test_safe_int(self):
        from stellaris_advisor.parser.extractor import safe_int

        assert safe_int(42) == 42
        assert safe_int(42.7) == 42
        assert safe_int(None, default=0) == 0

    def test_safe_str(self):
        from stellaris_advisor.parser.extractor import safe_str

        assert safe_str("hello") == "hello"
        assert safe_str(42) == "42"
        assert safe_str(None, default="") == ""

    def test_safe_bool(self):
        from stellaris_advisor.parser.extractor import safe_bool

        assert safe_bool(True) is True
        assert safe_bool("yes") is True
        assert safe_bool(None, default=False) is False

    def test_to_str_tuple(self):
        from stellaris_advisor.parser.extractor import to_str_tuple

        assert to_str_tuple(["a", "b", "c"]) == ("a", "b", "c")
        assert to_str_tuple(None) == ()
        assert to_str_tuple("single") == ("single",)

    def test_to_int_tuple(self):
        from stellaris_advisor.parser.extractor import to_int_tuple

        assert to_int_tuple([0, 4, 12]) == (0, 4, 12)
        assert to_int_tuple(None) == ()

    def test_to_frozen_map(self):
        from stellaris_advisor.parser.extractor import to_frozen_map

        result = to_frozen_map({"a": 1.0, "b": 2.0})
        assert result["a"] == 1.0
        with pytest.raises(TypeError):
            result["c"] = 3.0  # type: ignore[index]

    def test_to_frozen_map_none(self):
        from stellaris_advisor.parser.extractor import to_frozen_map

        result = to_frozen_map(None)
        assert len(result) == 0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_parser/test_version.py tests/test_parser/test_extractor.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 4: Implement version detection**

```python
# src/stellaris_advisor/parser/version.py
"""Stellaris version detection from save file metadata."""

from __future__ import annotations

from typing import Any

SUPPORTED_VERSIONS: tuple[str, ...] = ("4.3",)


class UnsupportedVersionError(Exception):
    """Raised when a save file version is not supported."""

    def __init__(self, detected: str, supported: tuple[str, ...]) -> None:
        self.detected = detected
        self.supported = supported
        super().__init__(
            f"Stellaris version {detected} is not supported. "
            f"Supported versions: {', '.join(supported)}"
        )


def detect_version(
    meta: dict[str, Any],
    supported: tuple[str, ...] = SUPPORTED_VERSIONS,
) -> str:
    """Extract and validate the game version from parsed meta dict.

    Returns the major.minor version string (e.g., '4.3').
    Raises UnsupportedVersionError if not in the supported set.
    Raises KeyError if the 'version' key is missing.
    """
    raw_version: str = meta["version"]
    # Format: "4.3.0.0" or "4.3.2.1" → extract "4.3"
    parts = str(raw_version).split(".")
    major_minor = f"{parts[0]}.{parts[1]}"

    if major_minor not in supported:
        raise UnsupportedVersionError(major_minor, supported)

    return major_minor
```

- [ ] **Step 5: Implement path navigation and safe conversion helpers**

```python
# src/stellaris_advisor/parser/extractor.py
"""Extraction helpers: path navigation and safe type conversions.

All functions are pure — no side effects, no mutation.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, TypeVar

T = TypeVar("T")


def get_path(data: dict[str, Any], path: str, default: Any = None) -> Any:
    """Navigate a nested dict by dot-separated path.

    Examples:
        get_path(data, "a.b.c") → data["a"]["b"]["c"]
        get_path(data, "country.0.name") → data["country"]["0"]["name"]
        get_path(data, "missing.key", default=0) → 0
    """
    if not path:
        return data

    current: Any = data
    for key in path.split("."):
        if not isinstance(current, dict):
            return default
        current = current.get(key, None)
        if current is None:
            return default
    return current


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float, returning default on failure."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Convert a value to int, returning default on failure."""
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def safe_str(value: Any, default: str = "") -> str:
    """Convert a value to str, returning default if None."""
    if value is None:
        return default
    return str(value)


def safe_bool(value: Any, default: bool = False) -> bool:
    """Convert a value to bool, returning default if None."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("yes", "true", "1")
    return bool(value)


def to_str_tuple(value: Any) -> tuple[str, ...]:
    """Convert a value to a tuple of strings.

    Handles: list → tuple, None → empty tuple, single string → (string,).
    """
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    return (str(value),)


def to_int_tuple(value: Any) -> tuple[int, ...]:
    """Convert a value to a tuple of ints.

    Handles: list → tuple, None → empty tuple.
    """
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    return (int(value),)


def to_frozen_map(
    value: dict[str, Any] | None,
    value_converter: type = float,
) -> MappingProxyType[str, Any]:
    """Convert a dict to an immutable MappingProxyType.

    Returns an empty MappingProxyType if value is None.
    """
    if value is None:
        return MappingProxyType({})
    return MappingProxyType({str(k): value_converter(v) for k, v in value.items()})
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_parser/test_version.py tests/test_parser/test_extractor.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/stellaris_advisor/parser/version.py src/stellaris_advisor/parser/extractor.py tests/test_parser/test_version.py tests/test_parser/test_extractor.py
git commit -m "feat: add version detection and extraction helpers"
```

---

## Task 7: Typed Extraction — Empire, Planet, System

**Files:**
- Modify: `src/stellaris_advisor/parser/extractor.py`
- Modify: `tests/test_parser/test_extractor.py`

This task adds the core extraction functions that convert raw parsed dicts into typed dataclasses. The extraction logic is version-aware and uses the path helpers from Task 6.

**Important:** The exact dict paths depend on the actual Stellaris 4.3 save file format. The paths shown below are based on the PRD's example and known Clausewitz conventions. They will need adjustment once real save fixtures are available. The architecture is designed so that path adjustments are localized to this one file.

- [ ] **Step 1: Write failing tests for empire extraction**

Add to `tests/test_parser/test_extractor.py`:

```python
@pytest.mark.unit
class TestExtractEmpire:
    def test_extract_empire_basic_fields(self):
        from stellaris_advisor.parser.extractor import extract_empire

        raw_empire = {
            "name": "United Nations of Earth",
            "government": {
                "type": "gov_democratic",
                "authority": "auth_democratic",
                "civics": ["civic_beacon_of_liberty", "civic_idealistic_foundation"],
            },
            "origin": "origin_prosperous_unification",
            "ethos": {
                "ethic": ["ethic_fanatic_egalitarian", "ethic_xenophile"],
            },
            "budget": {
                "current_month": {
                    "income": {"energy": 142.5, "minerals": 98.3, "alloys": 43.0},
                    "expenses": {"energy": 130.2, "minerals": 45.0},
                },
            },
            "tech_status": {
                "technology": {
                    "0": {"technology": "tech_corvettes", "level": 1},
                    "1": {"technology": "tech_destroyers", "level": 1},
                },
            },
            "owned_planets": [0, 4, 12],
            "fleet_power": 8200.5,
            "naval_capacity": {"used": 60, "max": 120},
            "victory_score": 100.0,
            "victory_rank": 3,
        }

        empire = extract_empire(raw_empire, empire_id=0, version="4.3")
        assert empire.name == "United Nations of Earth"
        assert empire.authority == "auth_democratic"
        assert empire.ethics == ("ethic_fanatic_egalitarian", "ethic_xenophile")
        assert empire.civics == ("civic_beacon_of_liberty", "civic_idealistic_foundation")
        assert empire.origin == "origin_prosperous_unification"
        assert empire.resource_income["energy"] == pytest.approx(142.5)
        assert empire.resource_income["alloys"] == pytest.approx(43.0)
        assert "tech_corvettes" in empire.researched_techs
        assert "tech_destroyers" in empire.researched_techs
        assert empire.fleet_power == pytest.approx(8200.5)
        assert empire.naval_capacity_used == 60
        assert empire.naval_capacity_max == 120

    def test_extract_empire_missing_optional_fields(self):
        from stellaris_advisor.parser.extractor import extract_empire

        raw_empire = {
            "name": "Minimal Empire",
            "government": {
                "type": "gov_democratic",
                "authority": "auth_democratic",
            },
        }

        empire = extract_empire(raw_empire, empire_id=0, version="4.3")
        assert empire.name == "Minimal Empire"
        assert empire.ethics == ()
        assert empire.civics == ()
        assert len(empire.resource_income) == 0
        assert len(empire.researched_techs) == 0
        assert empire.fleet_power == 0.0
        assert empire.federation is None
        assert empire.ascension_path is None


@pytest.mark.unit
class TestExtractPlanet:
    def test_extract_planet_basic(self):
        from stellaris_advisor.parser.extractor import extract_planet

        raw_planet = {
            "name": "Earth",
            "planet_class": "pc_continental",
            "planet_size": 25,
            "districts": {
                "district_city": 3,
                "district_mining": 2,
            },
            "buildings": ["building_capital_2", "building_alloy_foundry"],
            "num_buildings_slots": 8,
            "pop_count": 30,
            "stability": 55.0,
            "crime": 5.0,
            "amenities": 10.0,
            "amenities_usage": 8.0,
            "free_housing": 10.0,
            "total_housing": 40.0,
            "colony_type": "col_default",
            "is_capital": True,
        }

        planet = extract_planet(raw_planet, planet_id=0, version="4.3")
        assert planet.name == "Earth"
        assert planet.planet_class == "pc_continental"
        assert planet.size == 25
        assert planet.districts["district_city"] == 3
        assert planet.buildings == ("building_capital_2", "building_alloy_foundry")
        assert planet.is_capital is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_parser/test_extractor.py::TestExtractEmpire tests/test_parser/test_extractor.py::TestExtractPlanet -v`
Expected: FAIL (ImportError — `extract_empire` not defined)

- [ ] **Step 3: Implement extraction functions**

Add to `src/stellaris_advisor/parser/extractor.py`:

```python
from stellaris_advisor.parser.models import (
    AgendaState,
    ClaimState,
    CouncilPosition,
    EmpireSummary,
    EmpireState,
    FactionIssueState,
    FactionState,
    FederationState,
    FleetState,
    GalacticState,
    LeaderState,
    LeaderSummary,
    MegastructureState,
    PlanetState,
    PlanetSummary,
    PopGroupState,
    PopGroupSummary,
    StarbaseState,
    SystemState,
    WarState,
)


def extract_empire(raw: dict[str, Any], empire_id: int, version: str) -> EmpireState:
    """Extract a typed EmpireState from a raw parsed empire dict.

    Paths are version-aware. Currently supports version '4.3'.
    Missing fields use safe defaults (empty tuples, zero values, None).
    """
    gov = raw.get("government", {}) or {}
    budget = get_path(raw, "budget.current_month", default={}) or {}
    income = budget.get("income", {}) or {}
    expenses = budget.get("expenses", {}) or {}
    stockpile = raw.get("resource_stockpile", {}) or {}
    storage_cap = raw.get("resource_storage_cap", {}) or {}

    # Extract researched techs from tech_status
    researched = _extract_researched_techs(raw.get("tech_status", {}))

    # Extract current research
    current_research: dict[str, str] = {}
    research_progress: dict[str, float] = {}
    research_speed: dict[str, float] = {}
    for area in ("physics", "society", "engineering"):
        area_data = get_path(raw, f"tech_status.{area}", default={})
        if isinstance(area_data, dict):
            tech = area_data.get("technology", "")
            if tech:
                current_research[area] = safe_str(tech)
            progress = area_data.get("progress", 0.0)
            if progress:
                research_progress[area] = safe_float(progress)
        speed_val = get_path(raw, f"research_speed.{area}", default=0.0)
        if speed_val:
            research_speed[area] = safe_float(speed_val)

    # Extract naval capacity
    naval = raw.get("naval_capacity", {}) or {}

    # Extract traditions, perks, ascension
    traditions = to_str_tuple(raw.get("traditions", None))
    trees_opened = to_str_tuple(raw.get("tradition_trees_opened", None))
    trees_completed = to_str_tuple(raw.get("tradition_trees_completed", None))
    perks = to_str_tuple(raw.get("ascension_perks", None))

    # Extract leaders
    leaders = _extract_leaders(raw.get("leaders", {}))

    # Extract council
    council = _extract_council(raw.get("council", {}))

    # Extract agenda
    agenda = _extract_agenda(raw.get("current_agenda", None))

    # Extract factions
    factions = _extract_factions(raw.get("factions", {}))

    # Extract federation
    federation = _extract_federation(raw.get("federation", None))

    # Extract megastructures
    megas = _extract_megastructures(raw.get("megastructures", {}))

    # Extract wars
    wars = _extract_wars(raw.get("active_wars", {}))

    # Extract claims
    claims = _extract_claims(raw.get("claims", {}))

    # Policies
    policies_raw = raw.get("policies", {}) or {}
    policies_dict: dict[str, str] = {}
    if isinstance(policies_raw, dict):
        for cat, val in policies_raw.items():
            if isinstance(val, dict):
                policies_dict[cat] = safe_str(val.get("selected", ""))
            else:
                policies_dict[cat] = safe_str(val)

    return EmpireState(
        id=empire_id,
        name=safe_str(raw.get("name", ""), default="Unknown"),
        authority=safe_str(gov.get("authority", ""), default=""),
        ethics=to_str_tuple(get_path(raw, "ethos.ethic", default=None)),
        civics=to_str_tuple(gov.get("civics", None)),
        origin=safe_str(raw.get("origin", ""), default=""),
        government_type=safe_str(gov.get("type", ""), default=""),
        resource_income=to_frozen_map(income),
        resource_expense=to_frozen_map(expenses),
        resource_stockpile=to_frozen_map(stockpile),
        resource_storage_cap=to_frozen_map(storage_cap),
        researched_techs=researched,
        current_research=MappingProxyType(current_research),
        research_progress=MappingProxyType(research_progress),
        research_speed=MappingProxyType(research_speed),
        stored_research=to_frozen_map(raw.get("stored_research", None)),
        fleet_power=safe_float(raw.get("fleet_power", 0.0)),
        naval_capacity_used=safe_int(naval.get("used", 0)),
        naval_capacity_max=safe_int(naval.get("max", 0)),
        fleets=(),  # Extracted separately via extract_fleets
        owned_systems=to_int_tuple(raw.get("owned_systems", None)),
        owned_planets=(),  # Extracted separately via extract_planets
        starbases=(),  # Extracted separately
        total_pops=safe_int(raw.get("pop_count", 0)),
        pop_groups=(),  # Extracted separately
        rivals=to_int_tuple(raw.get("rivals", None)),
        defensive_pacts=to_int_tuple(raw.get("defensive_pacts", None)),
        federation_id=raw.get("federation_id", None),
        federation_power=safe_float(raw.get("federation_power"), default=0.0)
        if raw.get("federation_id") is not None
        else None,
        subjects=to_int_tuple(raw.get("subjects", None)),
        overlord=raw.get("overlord", None),
        traditions_unlocked=traditions,
        tradition_trees_opened=trees_opened,
        tradition_trees_completed=trees_completed,
        ascension_perks=perks,
        ascension_perk_slots_available=safe_int(raw.get("ascension_perk_slots_available", 0)),
        ascension_perk_slots_total=safe_int(raw.get("ascension_perk_slots_total", 0)),
        ascension_path=raw.get("ascension_path", None),
        ascension_situation_progress=raw.get("ascension_situation_progress", None),
        leaders=leaders,
        council_positions=council,
        current_agenda=agenda,
        council_legitimacy=safe_float(raw.get("council_legitimacy", 0.0)),
        factions=factions,
        policies=MappingProxyType(policies_dict),
        galcom_member=safe_bool(raw.get("galcom_member", False)),
        galcom_diplomatic_weight=safe_float(raw.get("galcom_diplomatic_weight", 0.0)),
        galcom_council_member=safe_bool(raw.get("galcom_council_member", False)),
        galcom_custodian=safe_bool(raw.get("galcom_custodian", False)),
        galcom_emperor=safe_bool(raw.get("galcom_emperor", False)),
        active_resolutions=to_str_tuple(raw.get("active_resolutions", None)),
        federation=federation,
        megastructures=megas,
        megastructures_under_construction=safe_int(
            raw.get("megastructures_under_construction", 0)
        ),
        active_wars=wars,
        claims_placed=claims,
        total_army_strength=safe_float(raw.get("total_army_strength", 0.0)),
        victory_score=safe_float(raw.get("victory_score", 0.0)),
        victory_rank=safe_int(raw.get("victory_rank", 0)),
    )


def extract_planet(raw: dict[str, Any], planet_id: int, version: str) -> PlanetState:
    """Extract a typed PlanetState from a raw parsed planet dict."""
    districts_raw = raw.get("districts", {}) or {}
    districts_map = {str(k): safe_int(v) for k, v in districts_raw.items()} if isinstance(districts_raw, dict) else {}
    districts_used = sum(districts_map.values())

    district_slots_raw = raw.get("district_slots", {}) or {}
    district_slots = {str(k): safe_int(v) for k, v in district_slots_raw.items()} if isinstance(district_slots_raw, dict) else {}

    buildings_raw = raw.get("buildings", [])
    buildings = to_str_tuple(buildings_raw) if buildings_raw else ()

    specs_raw = raw.get("district_specializations", {}) or {}
    specs = {str(k): safe_str(v) for k, v in specs_raw.items()} if isinstance(specs_raw, dict) else {}

    total_housing = safe_float(raw.get("total_housing", 0.0))
    free_housing = safe_float(raw.get("free_housing", 0.0))
    housing_usage = total_housing - free_housing

    return PlanetState(
        id=planet_id,
        name=safe_str(raw.get("name", ""), default="Unknown"),
        planet_class=safe_str(raw.get("planet_class", ""), default=""),
        size=safe_int(raw.get("planet_size", 0)),
        districts=MappingProxyType(districts_map),
        district_slots=MappingProxyType(district_slots),
        total_district_cap=safe_int(raw.get("total_district_cap", 0)),
        districts_used=districts_used,
        buildings=buildings,
        building_slots_total=safe_int(raw.get("num_buildings_slots", 0)),
        building_slots_used=len(buildings),
        district_specializations=MappingProxyType(specs),
        pops=safe_int(raw.get("pop_count", 0)),
        pop_groups=(),  # Extracted separately if needed
        stability=safe_float(raw.get("stability", 0.0)),
        crime=safe_float(raw.get("crime", 0.0)),
        amenities=safe_float(raw.get("amenities", 0.0)),
        amenities_usage=safe_float(raw.get("amenities_usage", 0.0)),
        housing=total_housing,
        housing_usage=housing_usage,
        designation=safe_str(raw.get("colony_type", ""), default=""),
        planetary_features=to_str_tuple(raw.get("planetary_features", None)),
        is_capital=safe_bool(raw.get("is_capital", False)),
    )


def extract_system(raw: dict[str, Any], system_id: int, version: str) -> SystemState:
    """Extract a typed SystemState from a raw parsed system dict."""
    hyperlanes_raw = raw.get("hyperlane", [])
    connections: list[int] = []
    if isinstance(hyperlanes_raw, list):
        for hl in hyperlanes_raw:
            if isinstance(hl, dict):
                connections.append(safe_int(hl.get("to", 0)))
    elif isinstance(hyperlanes_raw, dict):
        for hl in hyperlanes_raw.values():
            if isinstance(hl, dict):
                connections.append(safe_int(hl.get("to", 0)))

    planets_raw = raw.get("planet", [])
    planets: list[PlanetSummary] = []
    if isinstance(planets_raw, (list, tuple)):
        for pid in planets_raw:
            planets.append(PlanetSummary(
                id=safe_int(pid),
                name="",
                planet_class="",
                size=0,
                is_habitable=False,
                is_colonized=False,
            ))

    resources_raw = raw.get("resources", {}) or {}
    resources = {str(k): safe_float(v) for k, v in resources_raw.items()} if isinstance(resources_raw, dict) else {}

    return SystemState(
        id=system_id,
        name=safe_str(raw.get("name", ""), default="Unknown"),
        star_class=safe_str(raw.get("star_class", ""), default=""),
        hyperlane_connections=tuple(connections),
        owner=raw.get("owner", None),
        claimed_by=to_int_tuple(raw.get("claimed_by", None)),
        starbase=None,  # Linked separately
        planets=tuple(planets),
        resources=MappingProxyType(resources),
        has_megastructure=safe_bool(raw.get("has_megastructure", False)),
        megastructure_type=raw.get("megastructure_type", None),
    )


# --- Private extraction helpers ---


def _extract_researched_techs(tech_status: Any) -> frozenset[str]:
    """Extract set of researched tech IDs from tech_status block."""
    if not isinstance(tech_status, dict):
        return frozenset()
    techs: set[str] = set()
    technology = tech_status.get("technology", {})
    if isinstance(technology, dict):
        for entry in technology.values():
            if isinstance(entry, dict):
                tech_id = entry.get("technology", "")
                if tech_id:
                    techs.add(str(tech_id))
    return frozenset(techs)


def _extract_leaders(leaders_raw: Any) -> tuple[LeaderState, ...]:
    """Extract leader states from raw leaders block."""
    if not isinstance(leaders_raw, dict):
        return ()
    results: list[LeaderState] = []
    for lid, data in leaders_raw.items():
        if not isinstance(data, dict):
            continue
        results.append(LeaderState(
            id=safe_int(lid),
            name=safe_str(data.get("name", ""), default=""),
            leader_class=safe_str(data.get("class", ""), default=""),
            level=safe_int(data.get("level", 1)),
            traits=to_str_tuple(data.get("traits", None)),
            expertise=data.get("expertise", None),
            assignment=data.get("assignment", None),
        ))
    return tuple(results)


def _extract_council(council_raw: Any) -> tuple[CouncilPosition, ...]:
    """Extract council positions from raw council block."""
    if not isinstance(council_raw, dict):
        return ()
    results: list[CouncilPosition] = []
    for pos_type, data in council_raw.items():
        if not isinstance(data, dict):
            continue
        results.append(CouncilPosition(
            position_type=str(pos_type),
            leader_id=data.get("leader_id", None),
            leader_name=data.get("leader_name", None),
            leader_level=safe_int(data.get("leader_level", 0)),
        ))
    return tuple(results)


def _extract_agenda(agenda_raw: Any) -> AgendaState | None:
    """Extract current agenda state."""
    if not isinstance(agenda_raw, dict):
        return None
    return AgendaState(
        agenda_type=safe_str(agenda_raw.get("type", ""), default=""),
        progress=safe_float(agenda_raw.get("progress", 0.0)),
        is_launched=safe_bool(agenda_raw.get("is_launched", False)),
    )


def _extract_factions(factions_raw: Any) -> tuple[FactionState, ...]:
    """Extract faction states from raw factions block."""
    if not isinstance(factions_raw, dict):
        return ()
    results: list[FactionState] = []
    for _fid, data in factions_raw.items():
        if not isinstance(data, dict):
            continue
        issues_raw = data.get("issues", {})
        issues: list[FactionIssueState] = []
        if isinstance(issues_raw, dict):
            for issue_id, issue_data in issues_raw.items():
                if isinstance(issue_data, dict):
                    issues.append(FactionIssueState(
                        issue_id=str(issue_id),
                        is_fulfilled=safe_bool(issue_data.get("fulfilled", False)),
                    ))
        results.append(FactionState(
            ethic=safe_str(data.get("ethic", ""), default=""),
            approval=safe_float(data.get("approval", 0.0)),
            size=safe_int(data.get("size", 0)),
            support=safe_float(data.get("support", 0.0)),
            issues=tuple(issues),
        ))
    return tuple(results)


def _extract_federation(fed_raw: Any) -> FederationState | None:
    """Extract federation state."""
    if not isinstance(fed_raw, dict):
        return None
    return FederationState(
        federation_type=safe_str(fed_raw.get("type", ""), default=""),
        level=safe_int(fed_raw.get("level", 1)),
        cohesion=safe_float(fed_raw.get("cohesion", 0.0)),
        experience=safe_float(fed_raw.get("experience", 0.0)),
        fleet_power=safe_float(fed_raw.get("fleet_power", 0.0)),
    )


def _extract_megastructures(megas_raw: Any) -> tuple[MegastructureState, ...]:
    """Extract megastructure states."""
    if not isinstance(megas_raw, dict):
        return ()
    results: list[MegastructureState] = []
    for _mid, data in megas_raw.items():
        if not isinstance(data, dict):
            continue
        results.append(MegastructureState(
            megastructure_type=safe_str(data.get("type", ""), default=""),
            stage=safe_int(data.get("stage", 0)),
            system_id=safe_int(data.get("system_id", 0)),
        ))
    return tuple(results)


def _extract_wars(wars_raw: Any) -> tuple[WarState, ...]:
    """Extract active war states."""
    if not isinstance(wars_raw, dict):
        return ()
    results: list[WarState] = []
    for wid, data in wars_raw.items():
        if not isinstance(data, dict):
            continue
        results.append(WarState(
            war_id=safe_int(wid),
            name=safe_str(data.get("name", ""), default=""),
            war_goal=safe_str(data.get("war_goal", ""), default=""),
            exhaustion=safe_float(data.get("exhaustion", 0.0)),
            is_attacker=safe_bool(data.get("is_attacker", False)),
        ))
    return tuple(results)


def _extract_claims(claims_raw: Any) -> tuple[ClaimState, ...]:
    """Extract placed claims."""
    if not isinstance(claims_raw, (dict, list)):
        return ()
    results: list[ClaimState] = []
    items = claims_raw.values() if isinstance(claims_raw, dict) else claims_raw
    for data in items:
        if not isinstance(data, dict):
            continue
        results.append(ClaimState(
            system_id=safe_int(data.get("system_id", 0)),
            influence_cost=safe_int(data.get("influence_cost", 0)),
        ))
    return tuple(results)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_parser/test_extractor.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/stellaris_advisor/parser/extractor.py tests/test_parser/test_extractor.py
git commit -m "feat: implement typed extraction for empire, planet, and system"
```

---

## Task 8: Save File Handler (parse_save + parse_game_files)

**Files:**
- Create: `src/stellaris_advisor/parser/save_handler.py`
- Modify: `src/stellaris_advisor/parser/__init__.py`
- Create: `tests/test_parser/test_save_handler.py`
- Create: `tests/fixtures/` (directory with synthetic test data)

- [ ] **Step 1: Create a synthetic test fixture**

Create `tests/fixtures/` directory and a minimal synthetic save for testing:

```python
# tests/fixtures/create_test_save.py
"""Script to create a minimal synthetic .sav fixture for testing.
Run once: python tests/fixtures/create_test_save.py
"""

import zipfile
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent

META = """version="4.3.0.0"
name="Test Galaxy"
date="2250.01.01"
"""

GAMESTATE = """version="4.3.0.0"
date="2250.01.01"
galaxy={
    star_count=400
}
difficulty="grand_admiral"
player={
    0={
        country=0
    }
}
country={
    0={
        name="United Nations of Earth"
        government={
            type="gov_democratic"
            authority="auth_democratic"
            civics={ "civic_beacon_of_liberty" "civic_idealistic_foundation" }
        }
        origin="origin_prosperous_unification"
        ethos={
            ethic={ "ethic_fanatic_egalitarian" "ethic_xenophile" }
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
        owned_planets={ 0 4 }
        owned_systems={ 0 1 2 3 }
        fleet_power=5200.00
        naval_capacity={
            used=52
            max=120
        }
        pop_count=45
        victory_score=85.0
        victory_rank=3
    }
    1={
        name="Tzynn Empire"
        government={
            type="gov_dictatorial"
            authority="auth_dictatorial"
        }
        fleet_power=7800.00
        victory_score=120.0
        victory_rank=1
    }
}
galactic_object={
    0={
        name="Sol"
        star_class="sc_g"
        hyperlane={
            0={ to=1 }
            1={ to=2 }
        }
        planet={ 0 1 2 3 }
    }
    1={
        name="Alpha Centauri"
        star_class="sc_k"
        hyperlane={
            0={ to=0 }
            1={ to=3 }
        }
        planet={ 4 5 }
    }
}
planets={
    planet={
        0={
            name="Earth"
            planet_class="pc_continental"
            planet_size=25
            is_capital=yes
            districts={
                district_city=3
                district_mining=2
            }
            buildings={ "building_capital_2" "building_alloy_foundry" }
            num_buildings_slots=8
            pop_count=30
            stability=55.0
            crime=5.0
            amenities=10.0
            amenities_usage=8.0
            total_housing=40.0
            free_housing=10.0
            colony_type="col_default"
        }
        4={
            name="Mars"
            planet_class="pc_arid"
            planet_size=16
            districts={
                district_mining=3
            }
            buildings={ "building_capital_1" }
            num_buildings_slots=4
            pop_count=12
            stability=48.0
            crime=2.0
            amenities=5.0
            amenities_usage=4.0
            total_housing=20.0
            free_housing=8.0
            colony_type="col_mining"
        }
    }
}
"""

def create_test_save():
    save_path = FIXTURES_DIR / "sample_save_small.sav"
    with zipfile.ZipFile(save_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("meta", META)
        zf.writestr("gamestate", GAMESTATE)
    print(f"Created {save_path}")


if __name__ == "__main__":
    create_test_save()
```

Run: `python tests/fixtures/create_test_save.py`

- [ ] **Step 2: Write failing tests for save handler**

```python
# tests/test_parser/test_save_handler.py
"""Tests for save file handler."""

import pytest
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.mark.integration
class TestParseSave:
    def test_parse_synthetic_save(self):
        from stellaris_advisor.parser.save_handler import parse_save

        save_path = FIXTURES_DIR / "sample_save_small.sav"
        if not save_path.exists():
            pytest.skip("Test fixture not generated yet")

        state = parse_save(str(save_path))
        assert state.year == 2250
        assert state.game_version == "4.3"
        assert state.player_empire_id == 0
        assert state.player_empire.name == "United Nations of Earth"
        assert state.player_empire.authority == "auth_democratic"
        assert "tech_corvettes" in state.player_empire.researched_techs
        assert state.player_empire.resource_income["energy"] == pytest.approx(142.5)
        assert len(state.empires) == 2
        assert len(state.systems) == 2

    def test_parse_save_invalid_path_raises(self):
        from stellaris_advisor.parser.save_handler import parse_save

        with pytest.raises(FileNotFoundError):
            parse_save("/nonexistent/save.sav")

    def test_parse_save_extracts_planets(self):
        from stellaris_advisor.parser.save_handler import parse_save

        save_path = FIXTURES_DIR / "sample_save_small.sav"
        if not save_path.exists():
            pytest.skip("Test fixture not generated yet")

        state = parse_save(str(save_path))
        empire = state.player_empire
        assert len(empire.owned_planets) >= 1
        earth = next((p for p in empire.owned_planets if p.name == "Earth"), None)
        assert earth is not None
        assert earth.planet_class == "pc_continental"
        assert earth.size == 25
        assert earth.is_capital is True


@pytest.mark.integration
class TestParseGameFiles:
    def test_parse_game_files_returns_dict(self, tmp_path: Path):
        from stellaris_advisor.parser.save_handler import parse_game_files

        # Create a minimal game data file
        tech_dir = tmp_path / "common" / "technology"
        tech_dir.mkdir(parents=True)
        (tech_dir / "00_physics.txt").write_text("""
tech_lasers_1={
    area=physics
    category={ particles }
    tier=0
    cost=100
    weight=100
    prerequisites={}
}
tech_lasers_2={
    area=physics
    category={ particles }
    tier=1
    cost=500
    weight=80
    prerequisites={ "tech_lasers_1" }
}
""")

        result = parse_game_files(str(tmp_path))
        assert "technology" in result
        assert "tech_lasers_1" in result["technology"]
        assert result["technology"]["tech_lasers_1"]["area"] == "physics"
        assert result["technology"]["tech_lasers_2"]["cost"] == 500
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_parser/test_save_handler.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 4: Implement save handler**

```python
# src/stellaris_advisor/parser/save_handler.py
"""Top-level functions for parsing Stellaris save files and game data files."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

from stellaris_advisor.parser.clausewitz import parse
from stellaris_advisor.parser.extractor import (
    extract_empire,
    extract_planet,
    extract_system,
    get_path,
    safe_bool,
    safe_float,
    safe_int,
    safe_str,
    to_str_tuple,
)
from stellaris_advisor.parser.models import (
    EmpireSummary,
    GalacticState,
    PlanetState,
    SystemState,
)
from stellaris_advisor.parser.version import detect_version


def parse_save(path: str, supported_versions: tuple[str, ...] = ("4.3",)) -> GalacticState:
    """Parse a Stellaris .sav file into a typed GalacticState.

    The .sav file is a ZIP archive containing 'meta' and 'gamestate' entries.
    """
    save_path = Path(path)
    if not save_path.exists():
        raise FileNotFoundError(f"Save file not found: {save_path}")

    with zipfile.ZipFile(save_path, "r") as zf:
        meta_text = zf.read("meta").decode("utf-8-sig")
        gamestate_text = zf.read("gamestate").decode("utf-8-sig")

    meta = parse(meta_text)
    version = detect_version(meta, supported=supported_versions)
    gamestate = parse(gamestate_text)

    return _build_galactic_state(gamestate, meta, version)


def parse_game_files(stellaris_dir: str) -> dict[str, dict[str, Any]]:
    """Parse all game definition files from a Stellaris install directory.

    Returns a dict keyed by category (e.g., "technology", "buildings").
    Each value is a merged dict of all definitions in that category.
    """
    base = Path(stellaris_dir) / "common"
    if not base.exists():
        raise FileNotFoundError(f"Game data directory not found: {base}")

    categories = {
        "technology": "technology",
        "buildings": "buildings",
        "districts": "districts",
        "pop_jobs": "pop_jobs",
        "ship_sizes": "ship_sizes",
        "component_templates": "component_templates",
        "planet_classes": "planet_classes",
        "tradition_categories": "tradition_categories",
        "traditions": "traditions",
        "ascension_perks": "ascension_perks",
        "megastructures": "megastructures",
        "policies": "policies",
        "factions": "factions",
        "council_positions": "council_positions",
        "agendas": "agendas",
        "federation_types": "federation_types",
        "governments/civics": "civics",
        "governments/authorities": "authorities",
        "traits": "traits",
        "edicts": "edicts",
    }

    result: dict[str, dict[str, Any]] = {}
    for dir_name, category_key in categories.items():
        category_dir = base / dir_name
        if not category_dir.exists():
            continue
        merged: dict[str, Any] = {}
        for txt_file in sorted(category_dir.glob("*.txt")):
            try:
                text = txt_file.read_text(encoding="utf-8-sig")
                parsed = parse(text)
                merged.update(parsed)
            except Exception:
                continue  # Skip unparseable files, log in future
        if merged:
            result[category_key] = merged

    return result


def _build_galactic_state(
    gamestate: dict[str, Any],
    meta: dict[str, Any],
    version: str,
) -> GalacticState:
    """Assemble a GalacticState from parsed gamestate and meta dicts."""
    # Parse date: "2250.01.01" → year=2250, month=1
    date_str = safe_str(gamestate.get("date", "2200.01.01"))
    date_parts = date_str.split(".")
    year = safe_int(date_parts[0], default=2200)
    month = safe_int(date_parts[1], default=1) if len(date_parts) > 1 else 1

    # Galaxy info
    galaxy = gamestate.get("galaxy", {}) or {}
    galaxy_size = safe_int(galaxy.get("star_count", 0))
    difficulty = safe_str(gamestate.get("difficulty", ""), default="")

    # Player empire ID
    player_data = gamestate.get("player", {}) or {}
    player_empire_id = 0
    if isinstance(player_data, dict):
        first_player = next(iter(player_data.values()), {})
        if isinstance(first_player, dict):
            player_empire_id = safe_int(first_player.get("country", 0))

    # Extract all planets (needed for linking to empires)
    all_planets = _extract_all_planets(gamestate, version)

    # Extract empires
    countries_raw = gamestate.get("country", {}) or {}
    empires: list[EmpireSummary] = []
    player_empire = None

    for eid_str, empire_raw in countries_raw.items():
        if not isinstance(empire_raw, dict):
            continue
        eid = safe_int(eid_str)

        if eid == player_empire_id:
            full_empire = extract_empire(empire_raw, empire_id=eid, version=version)
            # Link planets to empire
            empire_planet_ids = set(full_empire.owned_systems)  # placeholder
            owned_planet_ids = empire_raw.get("owned_planets", [])
            if isinstance(owned_planet_ids, (list, tuple)):
                linked_planets = tuple(
                    all_planets[pid] for pid in owned_planet_ids
                    if pid in all_planets
                )
            else:
                linked_planets = ()
            # Rebuild with planets
            player_empire = EmpireState(
                **{
                    **{f.name: getattr(full_empire, f.name) for f in full_empire.__dataclass_fields__.values()},
                    "owned_planets": linked_planets,
                }
            )

        # Summary for all empires
        empires.append(EmpireSummary(
            id=eid,
            name=safe_str(empire_raw.get("name", ""), default=""),
            authority=safe_str(get_path(empire_raw, "government.authority", default=""), default=""),
            ethics=to_str_tuple(get_path(empire_raw, "ethos.ethic", default=None)),
            fleet_power=safe_float(empire_raw.get("fleet_power", 0.0)),
            tech_count=_count_techs(empire_raw),
            system_count=len(empire_raw.get("owned_systems", []) or []),
            victory_score=safe_float(empire_raw.get("victory_score", 0.0)),
        ))

    if player_empire is None:
        raise ValueError(f"Player empire ID {player_empire_id} not found in save")

    # Extract systems
    systems_raw = gamestate.get("galactic_object", {}) or {}
    systems: list[SystemState] = []
    for sid_str, sys_raw in systems_raw.items():
        if not isinstance(sys_raw, dict):
            continue
        systems.append(extract_system(sys_raw, system_id=safe_int(sid_str), version=version))

    return GalacticState(
        year=year,
        month=month,
        game_version=version,
        galaxy_size=galaxy_size,
        difficulty=difficulty,
        crisis_active=safe_bool(gamestate.get("crisis_active", False)),
        crisis_type=gamestate.get("crisis_type", None),
        war_in_heaven=safe_bool(gamestate.get("war_in_heaven", False)),
        galactic_market_founded=safe_bool(gamestate.get("galactic_market_founded", False)),
        empires=tuple(empires),
        player_empire_id=player_empire_id,
        systems=tuple(systems),
        player_empire=player_empire,
    )


def _extract_all_planets(
    gamestate: dict[str, Any], version: str
) -> dict[int, PlanetState]:
    """Extract all planets from the gamestate into a lookup dict."""
    planets_section = get_path(gamestate, "planets.planet", default={})
    if not isinstance(planets_section, dict):
        return {}
    result: dict[int, PlanetState] = {}
    for pid_str, planet_raw in planets_section.items():
        if not isinstance(planet_raw, dict):
            continue
        pid = safe_int(pid_str)
        result[pid] = extract_planet(planet_raw, planet_id=pid, version=version)
    return result


def _count_techs(empire_raw: dict[str, Any]) -> int:
    """Count researched techs in a raw empire dict."""
    tech_status = empire_raw.get("tech_status", {})
    if not isinstance(tech_status, dict):
        return 0
    technology = tech_status.get("technology", {})
    if not isinstance(technology, dict):
        return 0
    return len(technology)


# Re-import for the dataclass rebuild
from stellaris_advisor.parser.models import EmpireState  # noqa: E402
```

- [ ] **Step 5: Update parser __init__.py to export public API**

```python
# src/stellaris_advisor/parser/__init__.py
"""Stellaris save file and game data parser."""

from stellaris_advisor.parser.save_handler import parse_game_files, parse_save

__all__ = ["parse_save", "parse_game_files"]
```

- [ ] **Step 6: Generate the test fixture and run tests**

Run: `python tests/fixtures/create_test_save.py && pytest tests/test_parser/ -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/stellaris_advisor/parser/ tests/test_parser/ tests/fixtures/
git commit -m "feat: implement save file handler with parse_save and parse_game_files"
```

---

## Task 9: Run Full Test Suite and Coverage

**Files:**
- No new files

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 2: Check coverage**

Run: `pytest tests/ --cov=stellaris_advisor --cov-report=term-missing`
Expected: Coverage >80% on parser module

- [ ] **Step 3: Fix any coverage gaps**

If coverage is below 80% on any file, add targeted tests for uncovered branches. Focus on:
- Error paths in the tokenizer (unterminated strings, unexpected characters)
- Edge cases in the parser (empty input, deeply nested blocks)
- Missing field handling in extractors

- [ ] **Step 4: Run linting and type checking**

Run: `ruff check src/ tests/`
Run: `black --check src/ tests/`
Run: `mypy src/stellaris_advisor/parser/`

Fix any issues found.

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "chore: ensure full test coverage and clean linting for parser"
```

---

## Summary

After completing all 9 tasks, you will have:

1. A working Python package installable via `pip install -e ".[dev]"`
2. A Pydantic v2 config system with YAML loading
3. A complete Clausewitz format tokenizer and recursive descent parser
4. Frozen dataclass models for all game state types (empire, planet, system, fleet, leader, faction, etc.)
5. Version detection for Stellaris 4.3
6. Safe extraction helpers with path navigation
7. Typed extraction functions that convert raw parsed dicts to dataclasses
8. Top-level `parse_save()` and `parse_game_files()` functions
9. Full test coverage with unit, integration, and property-based tests

**Next plan:** Plan 2 — Game Knowledge Graph (loads game definitions, builds tech DAG, validates integrity)
