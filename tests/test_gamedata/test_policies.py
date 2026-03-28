"""Tests for gamedata/policies.py and gamedata/factions.py — TDD.

Covers:
  policies: load_policy_defs, available_policy_options,
            compute_faction_approval_deltas, compute_influence_delta,
            identify_policy_side_effects
  factions: load_faction_defs
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "game_data"


# ---------------------------------------------------------------------------
# Raw-dict helpers
# ---------------------------------------------------------------------------


def _raw_option(
    name: str,
    modifier: dict[str, Any] | None = None,
    faction_approval: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {"name": name}
    if modifier is not None:
        data["modifier"] = modifier
    if faction_approval is not None:
        data["faction_approval"] = faction_approval
    return data


def _raw_policy(
    pid: str,
    options: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a raw policy dict.  *options* is already a list of raw option dicts."""
    data: dict[str, Any] = {"potential": {"always": True}, "allow": {"always": True}}
    opts = options or [_raw_option(f"{pid}_opt_a"), _raw_option(f"{pid}_opt_b")]
    # Simulate parser output: multiple options → list, single option → dict
    data["option"] = opts if len(opts) > 1 else opts[0]
    return data


def _raw_faction(
    fid: str,
    ethic: str = "ethic_militarist",
    issues: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a raw faction dict with optional political issues."""
    raw_issues: list[dict[str, Any]] = issues or []
    if not raw_issues:
        return {
            "name": fid,
            "guiding_ethic": ethic,
        }
    issue_block: Any = raw_issues if len(raw_issues) > 1 else raw_issues[0]
    return {
        "name": fid,
        "guiding_ethic": ethic,
        "issues": {"political_issue": issue_block},
    }


def _raw_issue(option: str, approval: int, title: str = "") -> dict[str, Any]:
    return {
        "option": option,
        "title": title or f"issue_{option}",
        "effect": {"faction_approval": approval},
    }


# ===========================================================================
# load_policy_defs
# ===========================================================================


@pytest.mark.unit
class TestLoadPolicyDefs:
    def test_basic_fields(self):
        from stellaris_advisor.gamedata.policies import load_policy_defs

        raw = {"policy_eco": _raw_policy("policy_eco")}
        defs = load_policy_defs(raw)
        assert "policy_eco" in defs
        assert defs["policy_eco"].id == "policy_eco"

    def test_options_extracted_from_list(self):
        from stellaris_advisor.gamedata.policies import load_policy_defs

        opts = [_raw_option("opt_a"), _raw_option("opt_b"), _raw_option("opt_c")]
        raw = {"policy_x": _raw_policy("policy_x", options=opts)}
        defs = load_policy_defs(raw)
        assert len(defs["policy_x"].options) == 3

    def test_option_ids_from_name(self):
        from stellaris_advisor.gamedata.policies import load_policy_defs

        opts = [_raw_option("policy_x_free"), _raw_option("policy_x_state")]
        raw = {"policy_x": _raw_policy("policy_x", options=opts)}
        defs = load_policy_defs(raw)
        ids = {o.id for o in defs["policy_x"].options}
        assert ids == {"policy_x_free", "policy_x_state"}

    def test_option_effects_extracted(self):
        from stellaris_advisor.gamedata.policies import load_policy_defs

        opts = [_raw_option("opt_a", modifier={"trade_value_mult": 0.15})]
        raw = {"policy_x": _raw_policy("policy_x", options=opts)}
        defs = load_policy_defs(raw)
        effects = {m.resource: m.amount for m in defs["policy_x"].options[0].effects}
        assert effects["trade_value_mult"] == pytest.approx(0.15)

    def test_option_faction_effects_extracted(self):
        from stellaris_advisor.gamedata.policies import load_policy_defs

        opts = [_raw_option("opt_a", faction_approval={"materialist": 10, "egalitarian": 5})]
        raw = {"policy_x": _raw_policy("policy_x", options=opts)}
        defs = load_policy_defs(raw)
        feff = dict(defs["policy_x"].options[0].faction_effects)
        assert feff["materialist"] == pytest.approx(10.0)
        assert feff["egalitarian"] == pytest.approx(5.0)

    def test_single_option_as_dict_handled(self):
        """When only one option exists, parser emits a dict not a list."""
        from stellaris_advisor.gamedata.policies import load_policy_defs

        raw = {"policy_x": _raw_policy("policy_x", options=[_raw_option("only_opt")])}
        defs = load_policy_defs(raw)
        assert len(defs["policy_x"].options) == 1

    def test_frozen_dataclass(self):
        from stellaris_advisor.gamedata.policies import load_policy_defs

        defs = load_policy_defs({"p": _raw_policy("p")})
        with pytest.raises(AttributeError):
            defs["p"].category = "changed"  # type: ignore[misc]

    def test_empty_raw_returns_empty(self):
        from stellaris_advisor.gamedata.policies import load_policy_defs

        assert load_policy_defs({}) == {}


# ===========================================================================
# available_policy_options
# ===========================================================================


@pytest.mark.unit
class TestAvailablePolicyOptions:
    def test_returns_all_options(self):
        from stellaris_advisor.gamedata.policies import (
            available_policy_options,
            load_policy_defs,
        )

        opts = [_raw_option("opt_a"), _raw_option("opt_b")]
        defs = load_policy_defs({"p": _raw_policy("p", options=opts)})
        result = available_policy_options(defs["p"], frozenset())
        assert len(result) == 2

    def test_returns_tuple_of_policy_option_defs(self):
        from stellaris_advisor.gamedata.models import PolicyOptionDef
        from stellaris_advisor.gamedata.policies import (
            available_policy_options,
            load_policy_defs,
        )

        defs = load_policy_defs({"p": _raw_policy("p")})
        result = available_policy_options(defs["p"], frozenset())
        assert all(isinstance(o, PolicyOptionDef) for o in result)

    def test_empty_policy_returns_empty_tuple(self):
        from stellaris_advisor.gamedata.policies import (
            available_policy_options,
            load_policy_defs,
        )

        raw_policy: dict[str, Any] = {"potential": {}, "allow": {}}
        defs = load_policy_defs({"p": raw_policy})
        result = available_policy_options(defs["p"], frozenset())
        assert result == ()


# ===========================================================================
# compute_faction_approval_deltas
# ===========================================================================


@pytest.mark.unit
class TestComputeFactionApprovalDeltas:
    def _make_faction_defs(self):
        from stellaris_advisor.gamedata.factions import load_faction_defs

        return load_faction_defs(
            {
                "faction_a": _raw_faction(
                    "faction_a",
                    issues=[_raw_issue("opt_x", 10), _raw_issue("opt_y", 5)],
                ),
                "faction_b": _raw_faction(
                    "faction_b",
                    issues=[_raw_issue("opt_x", -5)],
                ),
            }
        )

    def test_matching_option_adds_approval(self):
        from stellaris_advisor.gamedata.policies import compute_faction_approval_deltas

        defs = self._make_faction_defs()
        result = compute_faction_approval_deltas(defs, frozenset({"opt_x"}))
        assert result["faction_a"] == pytest.approx(10.0)

    def test_multiple_matching_options_summed(self):
        from stellaris_advisor.gamedata.policies import compute_faction_approval_deltas

        defs = self._make_faction_defs()
        result = compute_faction_approval_deltas(defs, frozenset({"opt_x", "opt_y"}))
        assert result["faction_a"] == pytest.approx(15.0)

    def test_no_matching_options_returns_zero(self):
        from stellaris_advisor.gamedata.policies import compute_faction_approval_deltas

        defs = self._make_faction_defs()
        result = compute_faction_approval_deltas(defs, frozenset())
        assert result["faction_a"] == pytest.approx(0.0)

    def test_negative_approval_effect(self):
        from stellaris_advisor.gamedata.policies import compute_faction_approval_deltas

        defs = self._make_faction_defs()
        result = compute_faction_approval_deltas(defs, frozenset({"opt_x"}))
        assert result["faction_b"] == pytest.approx(-5.0)


# ===========================================================================
# compute_influence_delta
# ===========================================================================


@pytest.mark.unit
class TestComputeInfluenceDelta:
    def _make_policy(self):
        from stellaris_advisor.gamedata.policies import load_policy_defs

        opts = [_raw_option("opt_a"), _raw_option("opt_b")]
        return load_policy_defs({"p": _raw_policy("p", options=opts)})["p"]

    def test_same_option_returns_zero(self):
        from stellaris_advisor.gamedata.policies import compute_influence_delta

        policy = self._make_policy()
        assert compute_influence_delta(policy, "opt_a", "opt_a") == pytest.approx(0.0)

    def test_different_option_returns_float(self):
        from stellaris_advisor.gamedata.policies import compute_influence_delta

        policy = self._make_policy()
        result = compute_influence_delta(policy, "opt_a", "opt_b")
        assert isinstance(result, float)


# ===========================================================================
# identify_policy_side_effects
# ===========================================================================


@pytest.mark.unit
class TestIdentifyPolicySideEffects:
    def test_returns_effects_for_known_option(self):
        from stellaris_advisor.gamedata.policies import (
            identify_policy_side_effects,
            load_policy_defs,
        )

        opts = [_raw_option("opt_a", modifier={"trade_value_mult": 0.10})]
        defs = load_policy_defs({"p": _raw_policy("p", options=opts)})
        effects = identify_policy_side_effects(defs["p"], "opt_a")
        resources = {m.resource: m.amount for m in effects}
        assert resources["trade_value_mult"] == pytest.approx(0.10)

    def test_returns_empty_for_unknown_option(self):
        from stellaris_advisor.gamedata.policies import (
            identify_policy_side_effects,
            load_policy_defs,
        )

        defs = load_policy_defs({"p": _raw_policy("p")})
        assert identify_policy_side_effects(defs["p"], "nonexistent") == ()

    def test_returns_tuple_of_output_modifiers(self):
        from stellaris_advisor.gamedata.models import OutputModifier
        from stellaris_advisor.gamedata.policies import (
            identify_policy_side_effects,
            load_policy_defs,
        )

        opts = [_raw_option("opt_a", modifier={"x": 1.0})]
        defs = load_policy_defs({"p": _raw_policy("p", options=opts)})
        effects = identify_policy_side_effects(defs["p"], "opt_a")
        assert all(isinstance(e, OutputModifier) for e in effects)


# ===========================================================================
# load_faction_defs
# ===========================================================================


@pytest.mark.unit
class TestLoadFactionDefs:
    def test_basic_fields(self):
        from stellaris_advisor.gamedata.factions import load_faction_defs

        raw = {"faction_militarist": _raw_faction("faction_militarist", ethic="ethic_militarist")}
        defs = load_faction_defs(raw)
        assert "faction_militarist" in defs
        assert defs["faction_militarist"].id == "faction_militarist"
        assert defs["faction_militarist"].ethic == "ethic_militarist"

    def test_issues_extracted_multiple(self):
        from stellaris_advisor.gamedata.factions import load_faction_defs

        raw = {
            "f": _raw_faction(
                "f",
                issues=[_raw_issue("opt_x", 10), _raw_issue("opt_y", 5)],
            )
        }
        defs = load_faction_defs(raw)
        assert len(defs["f"].issues) == 2

    def test_issues_extracted_single(self):
        from stellaris_advisor.gamedata.factions import load_faction_defs

        raw = {"f": _raw_faction("f", issues=[_raw_issue("opt_x", 10)])}
        defs = load_faction_defs(raw)
        assert len(defs["f"].issues) == 1

    def test_issue_desired_option(self):
        from stellaris_advisor.gamedata.factions import load_faction_defs

        raw = {"f": _raw_faction("f", issues=[_raw_issue("policy_military_army", 10)])}
        defs = load_faction_defs(raw)
        assert defs["f"].issues[0].desired_option == "policy_military_army"

    def test_issue_approval_effect(self):
        from stellaris_advisor.gamedata.factions import load_faction_defs

        raw = {"f": _raw_faction("f", issues=[_raw_issue("opt", -5)])}
        defs = load_faction_defs(raw)
        assert defs["f"].issues[0].approval_effect == pytest.approx(-5.0)

    def test_no_issues(self):
        from stellaris_advisor.gamedata.factions import load_faction_defs

        raw = {"f": _raw_faction("f")}
        defs = load_faction_defs(raw)
        assert defs["f"].issues == ()

    def test_frozen_dataclass(self):
        from stellaris_advisor.gamedata.factions import load_faction_defs

        defs = load_faction_defs({"f": _raw_faction("f")})
        with pytest.raises(AttributeError):
            defs["f"].ethic = "changed"  # type: ignore[misc]

    def test_empty_raw_returns_empty(self):
        from stellaris_advisor.gamedata.factions import load_faction_defs

        assert load_faction_defs({}) == {}


# ===========================================================================
# Integration — real policies fixture
# ===========================================================================


@pytest.mark.integration
class TestPolicyDefsFromFixtures:
    @pytest.fixture
    def policy_raw(self):
        from stellaris_advisor.parser.clausewitz import parse

        return parse(_FIXTURE_DIR.joinpath("policies.txt").read_text())

    def test_loads_all_policies(self, policy_raw):
        from stellaris_advisor.gamedata.policies import load_policy_defs

        defs = load_policy_defs(policy_raw)
        for pid in ("policy_economic", "policy_military", "policy_diplomatic"):
            assert pid in defs

    def test_economic_policy_three_options(self, policy_raw):
        from stellaris_advisor.gamedata.policies import load_policy_defs

        defs = load_policy_defs(policy_raw)
        assert len(defs["policy_economic"].options) == 3

    def test_free_market_faction_effects(self, policy_raw):
        from stellaris_advisor.gamedata.policies import load_policy_defs

        defs = load_policy_defs(policy_raw)
        options = {o.id: o for o in defs["policy_economic"].options}
        feff = dict(options["policy_economic_free_market"].faction_effects)
        assert feff["materialist"] == pytest.approx(10.0)
        assert feff["egalitarian"] == pytest.approx(5.0)

    def test_free_market_effects(self, policy_raw):
        from stellaris_advisor.gamedata.policies import load_policy_defs

        defs = load_policy_defs(policy_raw)
        options = {o.id: o for o in defs["policy_economic"].options}
        effects = {m.resource: m.amount for m in options["policy_economic_free_market"].effects}
        assert effects["trade_value_mult"] == pytest.approx(0.15)

    def test_side_effects_query(self, policy_raw):
        from stellaris_advisor.gamedata.policies import (
            identify_policy_side_effects,
            load_policy_defs,
        )

        defs = load_policy_defs(policy_raw)
        effects = identify_policy_side_effects(
            defs["policy_military"], "policy_military_professional_army"
        )
        resources = {m.resource: m.amount for m in effects}
        assert "ship_fire_rate_mult" in resources


# ===========================================================================
# Integration — real factions fixture
# ===========================================================================


@pytest.mark.integration
class TestFactionDefsFromFixtures:
    @pytest.fixture
    def faction_raw(self):
        from stellaris_advisor.parser.clausewitz import parse

        return parse(_FIXTURE_DIR.joinpath("factions.txt").read_text())

    def test_loads_all_factions(self, faction_raw):
        from stellaris_advisor.gamedata.factions import load_faction_defs

        defs = load_faction_defs(faction_raw)
        for fid in ("faction_militarist", "faction_materialist", "faction_spiritualist"):
            assert fid in defs

    def test_militarist_ethic(self, faction_raw):
        from stellaris_advisor.gamedata.factions import load_faction_defs

        defs = load_faction_defs(faction_raw)
        assert defs["faction_militarist"].ethic == "ethic_militarist"

    def test_militarist_has_two_issues(self, faction_raw):
        from stellaris_advisor.gamedata.factions import load_faction_defs

        defs = load_faction_defs(faction_raw)
        assert len(defs["faction_militarist"].issues) == 2

    def test_materialist_has_one_issue(self, faction_raw):
        from stellaris_advisor.gamedata.factions import load_faction_defs

        defs = load_faction_defs(faction_raw)
        assert len(defs["faction_materialist"].issues) == 1

    def test_faction_approval_delta_integration(self, faction_raw):
        from stellaris_advisor.gamedata.factions import load_faction_defs
        from stellaris_advisor.gamedata.policies import compute_faction_approval_deltas

        defs = load_faction_defs(faction_raw)
        # policy_military_professional_army is desired by militarist (10) and opposed by none
        result = compute_faction_approval_deltas(
            defs, frozenset({"policy_military_professional_army"})
        )
        assert result["faction_militarist"] == pytest.approx(10.0)
