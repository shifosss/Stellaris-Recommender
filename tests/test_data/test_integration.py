"""End-to-end integration tests for the data pipeline.

Runs ingest_saves on real save fixtures and verifies the four Parquet outputs
have correct schemas, row counts, column types, and data validity.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import pytest

pyarrow = pytest.importorskip("pyarrow")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from stellaris_advisor.data.ingest import ingest_saves  # noqa: E402
from stellaris_advisor.data.schemas import (  # noqa: E402
    EMPIRE_FEATURES_SCHEMA,
    GAME_METADATA_SCHEMA,
    SYSTEM_FEATURES_SCHEMA,
    TRANSITIONS_SCHEMA,
)

SAMPLE_SAVE = Path(__file__).parent.parent / "fixtures" / "sample_save_small.sav"

# 109 empire features + 8 temporal features
_EXPECTED_STATE_VEC_LEN = 117


# ---------------------------------------------------------------------------
# Single-save pipeline
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSingleSavePipeline:
    """Full pipeline with one save — no transitions, but all four files written."""

    def test_empire_features_schema_matches(self, tmp_path: Path) -> None:
        save_dir = tmp_path / "saves"
        save_dir.mkdir()
        shutil.copy(SAMPLE_SAVE, save_dir / "save_01.sav")
        result = ingest_saves(save_dir, tmp_path / "out", game_id="e2e_1")
        table = pq.read_table(str(result.empire_features_path))
        for field in EMPIRE_FEATURES_SCHEMA:
            assert (
                field.name in table.schema.names
            ), f"Missing empire field: {field.name}"

    def test_system_features_schema_matches(self, tmp_path: Path) -> None:
        save_dir = tmp_path / "saves"
        save_dir.mkdir()
        shutil.copy(SAMPLE_SAVE, save_dir / "save_01.sav")
        result = ingest_saves(save_dir, tmp_path / "out", game_id="e2e_1")
        table = pq.read_table(str(result.system_features_path))
        for field in SYSTEM_FEATURES_SCHEMA:
            assert field.name in table.schema.names

    def test_transitions_schema_matches(self, tmp_path: Path) -> None:
        save_dir = tmp_path / "saves"
        save_dir.mkdir()
        shutil.copy(SAMPLE_SAVE, save_dir / "save_01.sav")
        result = ingest_saves(save_dir, tmp_path / "out", game_id="e2e_1")
        table = pq.read_table(str(result.transitions_path))
        for field in TRANSITIONS_SCHEMA:
            assert field.name in table.schema.names

    def test_metadata_schema_matches(self, tmp_path: Path) -> None:
        save_dir = tmp_path / "saves"
        save_dir.mkdir()
        shutil.copy(SAMPLE_SAVE, save_dir / "save_01.sav")
        result = ingest_saves(save_dir, tmp_path / "out", game_id="e2e_1")
        table = pq.read_table(str(result.metadata_path))
        for field in GAME_METADATA_SCHEMA:
            assert field.name in table.schema.names

    def test_metadata_has_exactly_one_row(self, tmp_path: Path) -> None:
        save_dir = tmp_path / "saves"
        save_dir.mkdir()
        shutil.copy(SAMPLE_SAVE, save_dir / "save_01.sav")
        result = ingest_saves(save_dir, tmp_path / "out", game_id="e2e_1")
        table = pq.read_table(str(result.metadata_path))
        assert table.num_rows == 1

    def test_metadata_game_id_matches(self, tmp_path: Path) -> None:
        save_dir = tmp_path / "saves"
        save_dir.mkdir()
        shutil.copy(SAMPLE_SAVE, save_dir / "save_01.sav")
        result = ingest_saves(save_dir, tmp_path / "out", game_id="myrun")
        table = pq.read_table(str(result.metadata_path))
        assert table.column("game_id")[0].as_py() == "myrun"

    def test_metadata_num_empires_positive(self, tmp_path: Path) -> None:
        save_dir = tmp_path / "saves"
        save_dir.mkdir()
        shutil.copy(SAMPLE_SAVE, save_dir / "save_01.sav")
        result = ingest_saves(save_dir, tmp_path / "out", game_id="e2e_1")
        table = pq.read_table(str(result.metadata_path))
        assert table.column("num_empires")[0].as_py() > 0

    def test_metadata_total_saves_is_one(self, tmp_path: Path) -> None:
        save_dir = tmp_path / "saves"
        save_dir.mkdir()
        shutil.copy(SAMPLE_SAVE, save_dir / "save_01.sav")
        result = ingest_saves(save_dir, tmp_path / "out", game_id="e2e_1")
        table = pq.read_table(str(result.metadata_path))
        assert table.column("total_saves")[0].as_py() == 1

    def test_empire_features_columns_are_float64(self, tmp_path: Path) -> None:
        save_dir = tmp_path / "saves"
        save_dir.mkdir()
        shutil.copy(SAMPLE_SAVE, save_dir / "save_01.sav")
        result = ingest_saves(save_dir, tmp_path / "out", game_id="e2e_1")
        table = pq.read_table(str(result.empire_features_path))
        skip = {"game_id", "year", "empire_id"}
        for name in table.schema.names:
            if name in skip:
                continue
            assert table.schema.field(name).type == pa.float64(), f"Not float64: {name}"

    def test_no_parse_errors_from_valid_save(self, tmp_path: Path) -> None:
        save_dir = tmp_path / "saves"
        save_dir.mkdir()
        shutil.copy(SAMPLE_SAVE, save_dir / "save_01.sav")
        result = ingest_saves(save_dir, tmp_path / "out", game_id="e2e_1")
        assert result.errors == ()

    def test_single_save_no_transitions(self, tmp_path: Path) -> None:
        save_dir = tmp_path / "saves"
        save_dir.mkdir()
        shutil.copy(SAMPLE_SAVE, save_dir / "save_01.sav")
        result = ingest_saves(save_dir, tmp_path / "out", game_id="e2e_1")
        table = pq.read_table(str(result.transitions_path))
        assert table.num_rows == 0


# ---------------------------------------------------------------------------
# Two-save pipeline — transition and temporal feature checks
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestTwoSavePipeline:
    """Full pipeline with two identical saves — transitions with temporal features."""

    def _run(self, tmp_path: Path) -> tuple:
        save_dir = tmp_path / "saves"
        save_dir.mkdir()
        shutil.copy(SAMPLE_SAVE, save_dir / "save_01.sav")
        shutil.copy(SAMPLE_SAVE, save_dir / "save_02.sav")
        result = ingest_saves(save_dir, tmp_path / "out", game_id="e2e_2")
        table = pq.read_table(str(result.transitions_path))
        return result, table

    def test_transitions_has_two_rows(self, tmp_path: Path) -> None:
        # 2 empires × 1 pair × 1 no-op action each = 2 rows
        _, table = self._run(tmp_path)
        assert table.num_rows == 2

    def test_state_features_length_is_117(self, tmp_path: Path) -> None:
        _, table = self._run(tmp_path)
        for i in range(table.num_rows):
            vec = table.column("state_features")[i].as_py()
            assert len(vec) == _EXPECTED_STATE_VEC_LEN, f"Row {i}: {len(vec)} != 117"

    def test_next_state_features_length_is_117(self, tmp_path: Path) -> None:
        _, table = self._run(tmp_path)
        for i in range(table.num_rows):
            vec = table.column("next_state_features")[i].as_py()
            assert len(vec) == _EXPECTED_STATE_VEC_LEN

    def test_state_features_are_all_finite(self, tmp_path: Path) -> None:
        _, table = self._run(tmp_path)
        for i in range(table.num_rows):
            for v in table.column("state_features")[i].as_py():
                assert isinstance(v, float) and math.isfinite(
                    v
                ), f"Row {i}: non-finite {v}"

    def test_rewards_are_floats(self, tmp_path: Path) -> None:
        _, table = self._run(tmp_path)
        for v in table.column("reward"):
            assert isinstance(v.as_py(), float)

    def test_no_op_action_type_is_empty_string(self, tmp_path: Path) -> None:
        # identical saves → extract_actions returns () → action_type = ""
        _, table = self._run(tmp_path)
        for v in table.column("action_type"):
            assert v.as_py() == ""

    def test_action_params_is_empty_json_object(self, tmp_path: Path) -> None:
        _, table = self._run(tmp_path)
        for v in table.column("action_params"):
            assert v.as_py() == "{}"

    def test_terminal_flag_true_for_only_pair(self, tmp_path: Path) -> None:
        # only 1 pair with 2 saves → all rows are terminal
        _, table = self._run(tmp_path)
        for v in table.column("terminal"):
            assert v.as_py() is True

    def test_empire_id_column_contains_integers(self, tmp_path: Path) -> None:
        _, table = self._run(tmp_path)
        for v in table.column("empire_id"):
            assert isinstance(v.as_py(), int)

    def test_metadata_total_saves_is_two(self, tmp_path: Path) -> None:
        result, _ = self._run(tmp_path)
        meta = pq.read_table(str(result.metadata_path))
        assert meta.column("total_saves")[0].as_py() == 2

    def test_empire_features_has_four_rows(self, tmp_path: Path) -> None:
        # 2 empires × 2 saves = 4 rows
        result, _ = self._run(tmp_path)
        emp = pq.read_table(str(result.empire_features_path))
        assert emp.num_rows == 4


# ---------------------------------------------------------------------------
# Error handling pipeline
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestErrorHandlingPipeline:
    """Pipeline robustness: corrupted saves are captured in errors, valid saves proceed."""

    def test_corrupted_save_appears_in_errors(self, tmp_path: Path) -> None:
        save_dir = tmp_path / "saves"
        save_dir.mkdir()
        (save_dir / "corrupted.sav").write_bytes(b"not a zip file")
        shutil.copy(SAMPLE_SAVE, save_dir / "valid.sav")
        result = ingest_saves(save_dir, tmp_path / "out", game_id="err_g")
        assert any("corrupted.sav" in e for e in result.errors)

    def test_corrupted_save_does_not_block_valid_save(self, tmp_path: Path) -> None:
        save_dir = tmp_path / "saves"
        save_dir.mkdir()
        (save_dir / "aaa_bad.sav").write_bytes(b"garbage")
        shutil.copy(SAMPLE_SAVE, save_dir / "zzz_good.sav")
        result = ingest_saves(save_dir, tmp_path / "out", game_id="err_g")
        emp = pq.read_table(str(result.empire_features_path))
        assert emp.num_rows > 0

    def test_all_corrupted_saves_produces_empty_empire_table(
        self, tmp_path: Path
    ) -> None:
        save_dir = tmp_path / "saves"
        save_dir.mkdir()
        (save_dir / "bad1.sav").write_bytes(b"garbage1")
        (save_dir / "bad2.sav").write_bytes(b"garbage2")
        result = ingest_saves(save_dir, tmp_path / "out", game_id="err_g")
        emp = pq.read_table(str(result.empire_features_path))
        assert emp.num_rows == 0

    def test_all_corrupted_saves_produces_empty_transitions_table(
        self, tmp_path: Path
    ) -> None:
        save_dir = tmp_path / "saves"
        save_dir.mkdir()
        (save_dir / "bad1.sav").write_bytes(b"garbage1")
        result = ingest_saves(save_dir, tmp_path / "out", game_id="err_g")
        trans = pq.read_table(str(result.transitions_path))
        assert trans.num_rows == 0

    def test_all_corrupted_errors_are_captured(self, tmp_path: Path) -> None:
        save_dir = tmp_path / "saves"
        save_dir.mkdir()
        (save_dir / "bad1.sav").write_bytes(b"garbage1")
        (save_dir / "bad2.sav").write_bytes(b"garbage2")
        result = ingest_saves(save_dir, tmp_path / "out", game_id="err_g")
        assert len(result.errors) == 2

    def test_saves_processed_count_excludes_failures(self, tmp_path: Path) -> None:
        save_dir = tmp_path / "saves"
        save_dir.mkdir()
        (save_dir / "bad.sav").write_bytes(b"garbage")
        shutil.copy(SAMPLE_SAVE, save_dir / "good.sav")
        result = ingest_saves(save_dir, tmp_path / "out", game_id="err_g")
        assert result.saves_processed == 1
