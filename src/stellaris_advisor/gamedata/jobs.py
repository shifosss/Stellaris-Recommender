"""Job loader for the Stellaris gamedata layer."""

from __future__ import annotations

from typing import Any

from stellaris_advisor.gamedata.models import JobDef, OutputModifier
from stellaris_advisor.parser.extractor import get_path, safe_float, safe_str


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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_job_defs(raw: dict[str, Any]) -> dict[str, JobDef]:
    """Parse a raw Clausewitz jobs dict into typed JobDef instances.

    *raw* is the output of ``clausewitz.parse()`` applied to one or more
    job definition files — a flat mapping of job_id → attribute dict.

    Production and upkeep are nested under ``resources.produces`` and
    ``resources.upkeep`` respectively. Job weight is under ``weight.weight``.
    """
    result: dict[str, JobDef] = {}

    for jid, data in raw.items():
        if not isinstance(data, dict):
            continue

        produces = _extract_modifiers(get_path(data, "resources.produces"))
        upkeep = _extract_modifiers(get_path(data, "resources.upkeep"))

        # weight may be a plain number or a block { weight = N modifier = { ... } }
        raw_weight = data.get("weight")
        if isinstance(raw_weight, dict):
            weight = safe_float(raw_weight.get("weight"), default=1.0)
        else:
            weight = safe_float(raw_weight, default=1.0)

        result[jid] = JobDef(
            id=jid,
            name=safe_str(data.get("name"), default=jid),
            category=safe_str(data.get("category")),
            produces=produces,
            upkeep=upkeep,
            weight=weight,
        )

    return result
