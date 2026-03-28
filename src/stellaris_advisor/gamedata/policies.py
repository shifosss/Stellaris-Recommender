"""Policy loader and query functions for the Stellaris gamedata layer."""

from __future__ import annotations

from typing import Any

from stellaris_advisor.gamedata.models import (
    FactionDef,
    OutputModifier,
    PolicyDef,
    PolicyOptionDef,
)
from stellaris_advisor.parser.extractor import safe_float, safe_str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_modifiers(raw_dict: Any) -> tuple[OutputModifier, ...]:
    if not isinstance(raw_dict, dict):
        return ()
    return tuple(
        OutputModifier(resource=str(k), amount=safe_float(v))
        for k, v in raw_dict.items()
    )


def _extract_faction_effects(raw: Any) -> tuple[tuple[str, float], ...]:
    """Convert a faction_approval dict to a tuple of (faction_id, delta) pairs."""
    if not isinstance(raw, dict):
        return ()
    return tuple((str(k), safe_float(v)) for k, v in raw.items())


def _parse_option(data: Any) -> PolicyOptionDef | None:
    if not isinstance(data, dict):
        return None
    option_id = safe_str(data.get("name"))
    return PolicyOptionDef(
        id=option_id,
        name=option_id,
        effects=_extract_modifiers(data.get("modifier")),
        faction_effects=_extract_faction_effects(data.get("faction_approval")),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_policy_defs(raw: dict[str, Any]) -> dict[str, PolicyDef]:
    """Parse a raw Clausewitz policy dict into typed PolicyDef instances.

    Each policy entry may have one (dict) or multiple (list) ``option`` blocks,
    depending on whether the parser de-duplicated them.
    """
    result: dict[str, PolicyDef] = {}

    for pid, data in raw.items():
        if not isinstance(data, dict):
            continue

        options_raw = data.get("option")
        if options_raw is None:
            options_raw = []
        elif isinstance(options_raw, dict):
            options_raw = [options_raw]

        options: list[PolicyOptionDef] = []
        for opt_data in options_raw:
            parsed = _parse_option(opt_data)
            if parsed is not None:
                options.append(parsed)

        result[pid] = PolicyDef(
            id=pid,
            name=safe_str(data.get("name"), default=pid),
            category="",
            options=tuple(options),
        )

    return result


def available_policy_options(
    policy_def: PolicyDef,
    researched: frozenset[str] = frozenset(),
) -> tuple[PolicyOptionDef, ...]:
    """Return all options for *policy_def*.

    Clausewitz ``possible`` blocks contain complex runtime conditions that
    cannot be evaluated from static game data; all options are returned.
    """
    return policy_def.options


def compute_faction_approval_deltas(
    faction_defs: dict[str, FactionDef],
    active_options: frozenset[str],
) -> dict[str, float]:
    """Return approval delta for each faction given the currently active policy options.

    For each faction, sums ``approval_effect`` for every issue whose
    ``desired_option`` is in *active_options*.
    """
    result: dict[str, float] = {}
    for fid, faction in faction_defs.items():
        delta = sum(
            issue.approval_effect
            for issue in faction.issues
            if issue.desired_option in active_options
        )
        result[fid] = delta
    return result


def compute_influence_delta(
    policy_def: PolicyDef,
    current_option_id: str,
    new_option_id: str,
) -> float:
    """Return the influence cost to switch *policy_def* from *current_option_id* to *new_option_id*.

    Returns 0.0 when the options are identical.  In the simplified gamedata
    model, no explicit influence cost is stored on policy options, so the cost
    is 0.0 for all transitions (runtime costs depend on empire state).
    """
    if current_option_id == new_option_id:
        return 0.0
    return 0.0


def identify_policy_side_effects(
    policy_def: PolicyDef,
    option_id: str,
) -> tuple[OutputModifier, ...]:
    """Return the resource/modifier effects of *option_id* within *policy_def*.

    Returns an empty tuple if the option is not found.
    """
    for option in policy_def.options:
        if option.id == option_id:
            return option.effects
    return ()
