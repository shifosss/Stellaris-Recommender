"""Tests for the feature name registry.

Verifies structure, naming conventions, and uniqueness
across all three feature name tuples.
"""

from __future__ import annotations

import re

import pytest

from stellaris_advisor.features.registry import (
    ALL_FEATURE_NAMES,
    EMPIRE_FEATURE_NAMES,
    SYSTEM_FEATURE_NAMES,
    TEMPORAL_FEATURE_NAMES,
)

_SNAKE_CASE = re.compile(r"^[a-z][a-z0-9_]*$")

_EMPIRE_PREFIXES = frozenset(
    {
        "resource_",
        "economic_",
        "expansion_",
        "military_",
        "tech_",
        "planet_",
        "pop_",
        "diplomatic_",
        "archetype_",
        "tradition_",
        "leader_",
        "faction_",
        "galcom_",
        "federation_",
        "megastructure_",
    }
)


class TestEmpireFeatureNames:
    @pytest.mark.unit
    def test_is_tuple(self) -> None:
        assert isinstance(EMPIRE_FEATURE_NAMES, tuple)

    @pytest.mark.unit
    def test_length_approximately_107(self) -> None:
        assert (
            100 <= len(EMPIRE_FEATURE_NAMES) <= 115
        ), f"Expected ~107 empire features, got {len(EMPIRE_FEATURE_NAMES)}"

    @pytest.mark.unit
    def test_all_strings(self) -> None:
        assert all(isinstance(name, str) for name in EMPIRE_FEATURE_NAMES)

    @pytest.mark.unit
    def test_snake_case(self) -> None:
        bad = [n for n in EMPIRE_FEATURE_NAMES if not _SNAKE_CASE.match(n)]
        assert not bad, f"Non-snake_case empire names: {bad}"

    @pytest.mark.unit
    def test_category_prefixes(self) -> None:
        bad = [
            n
            for n in EMPIRE_FEATURE_NAMES
            if not any(n.startswith(p) for p in _EMPIRE_PREFIXES)
        ]
        assert not bad, f"Empire names without category prefix: {bad}"

    @pytest.mark.unit
    def test_no_duplicates(self) -> None:
        assert len(EMPIRE_FEATURE_NAMES) == len(
            set(EMPIRE_FEATURE_NAMES)
        ), "Duplicate names in EMPIRE_FEATURE_NAMES"


class TestSystemFeatureNames:
    @pytest.mark.unit
    def test_is_tuple(self) -> None:
        assert isinstance(SYSTEM_FEATURE_NAMES, tuple)

    @pytest.mark.unit
    def test_length_approximately_12(self) -> None:
        assert (
            10 <= len(SYSTEM_FEATURE_NAMES) <= 15
        ), f"Expected ~12 system features, got {len(SYSTEM_FEATURE_NAMES)}"

    @pytest.mark.unit
    def test_all_strings(self) -> None:
        assert all(isinstance(name, str) for name in SYSTEM_FEATURE_NAMES)

    @pytest.mark.unit
    def test_snake_case(self) -> None:
        bad = [n for n in SYSTEM_FEATURE_NAMES if not _SNAKE_CASE.match(n)]
        assert not bad, f"Non-snake_case system names: {bad}"

    @pytest.mark.unit
    def test_system_prefix(self) -> None:
        bad = [n for n in SYSTEM_FEATURE_NAMES if not n.startswith("system_")]
        assert not bad, f"System names not starting with 'system_': {bad}"

    @pytest.mark.unit
    def test_no_duplicates(self) -> None:
        assert len(SYSTEM_FEATURE_NAMES) == len(set(SYSTEM_FEATURE_NAMES))


class TestTemporalFeatureNames:
    @pytest.mark.unit
    def test_is_tuple(self) -> None:
        assert isinstance(TEMPORAL_FEATURE_NAMES, tuple)

    @pytest.mark.unit
    def test_length_approximately_8(self) -> None:
        assert (
            6 <= len(TEMPORAL_FEATURE_NAMES) <= 10
        ), f"Expected ~8 temporal features, got {len(TEMPORAL_FEATURE_NAMES)}"

    @pytest.mark.unit
    def test_all_strings(self) -> None:
        assert all(isinstance(name, str) for name in TEMPORAL_FEATURE_NAMES)

    @pytest.mark.unit
    def test_snake_case(self) -> None:
        bad = [n for n in TEMPORAL_FEATURE_NAMES if not _SNAKE_CASE.match(n)]
        assert not bad, f"Non-snake_case temporal names: {bad}"

    @pytest.mark.unit
    def test_temporal_prefix(self) -> None:
        bad = [n for n in TEMPORAL_FEATURE_NAMES if not n.startswith("temporal_")]
        assert not bad, f"Temporal names not starting with 'temporal_': {bad}"

    @pytest.mark.unit
    def test_no_duplicates(self) -> None:
        assert len(TEMPORAL_FEATURE_NAMES) == len(set(TEMPORAL_FEATURE_NAMES))


class TestAllFeatureNames:
    @pytest.mark.unit
    def test_is_tuple(self) -> None:
        assert isinstance(ALL_FEATURE_NAMES, tuple)

    @pytest.mark.unit
    def test_is_ordered_concatenation(self) -> None:
        assert (
            ALL_FEATURE_NAMES
            == EMPIRE_FEATURE_NAMES + SYSTEM_FEATURE_NAMES + TEMPORAL_FEATURE_NAMES
        )

    @pytest.mark.unit
    def test_no_global_duplicates(self) -> None:
        assert len(ALL_FEATURE_NAMES) == len(
            set(ALL_FEATURE_NAMES)
        ), "Duplicate names found in ALL_FEATURE_NAMES"

    @pytest.mark.unit
    def test_no_cross_registry_duplicates(self) -> None:
        empire_set = set(EMPIRE_FEATURE_NAMES)
        system_set = set(SYSTEM_FEATURE_NAMES)
        temporal_set = set(TEMPORAL_FEATURE_NAMES)
        assert not (
            empire_set & system_set
        ), f"Empire/system share names: {empire_set & system_set}"
        assert not (
            empire_set & temporal_set
        ), f"Empire/temporal share names: {empire_set & temporal_set}"
        assert not (
            system_set & temporal_set
        ), f"System/temporal share names: {system_set & temporal_set}"
