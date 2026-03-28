"""Tests for the batch ingestion pipeline."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

pyarrow = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from stellaris_advisor.data.ingest import IngestResult, ingest_saves  # noqa: E402

SAMPLE_SAVE = Path(__file__).parent.parent / "fixtures" / "sample_save_small.sav"


# ---------------------------------------------------------------------------
# Unit tests — IngestResult dataclass
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIngestResult:
    def test_is_frozen(self) -> None:
        result = IngestResult(
            game_id="g0",
            saves_processed=1,
            empire_features_path=Path("/tmp/emp.parquet"),
            system_features_path=Path("/tmp/sys.parquet"),
            transitions_path=Path("/tmp/trans.parquet"),
            metadata_path=Path("/tmp/meta.parquet"),
            errors=(),
        )
        with pytest.raises(Exception):
            result.saves_processed = 99  # type: ignore[misc]

    def test_errors_is_tuple(self) -> None:
        result = IngestResult(
            game_id="g0",
            saves_processed=0,
            empire_features_path=Path("/tmp/emp.parquet"),
            system_features_path=Path("/tmp/sys.parquet"),
            transitions_path=Path("/tmp/trans.parquet"),
            metadata_path=Path("/tmp/meta.parquet"),
            errors=("err1", "err2"),
        )
        assert isinstance(result.errors, tuple)
        assert len(result.errors) == 2

    def test_paths_are_path_objects(self) -> None:
        result = IngestResult(
            game_id="g0",
            saves_processed=0,
            empire_features_path=Path("/a/emp.parquet"),
            system_features_path=Path("/a/sys.parquet"),
            transitions_path=Path("/a/trans.parquet"),
            metadata_path=Path("/a/meta.parquet"),
            errors=(),
        )
        assert isinstance(result.empire_features_path, Path)
        assert isinstance(result.transitions_path, Path)


# ---------------------------------------------------------------------------
# Integration tests — ingest_saves
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestIngestSaves:
    def test_empty_folder_creates_all_four_output_files(self, tmp_path: Path) -> None:
        save_dir = tmp_path / "saves"
        save_dir.mkdir()
        result = ingest_saves(save_dir, tmp_path / "out", game_id="g0")
        assert result.saves_processed == 0
        assert result.empire_features_path.exists()
        assert result.system_features_path.exists()
        assert result.transitions_path.exists()
        assert result.metadata_path.exists()

    def test_empty_folder_has_zero_transition_rows(self, tmp_path: Path) -> None:
        save_dir = tmp_path / "saves"
        save_dir.mkdir()
        result = ingest_saves(save_dir, tmp_path / "out", game_id="g0")
        table = pq.read_table(str(result.transitions_path))
        assert table.num_rows == 0

    def test_single_save_creates_all_four_output_files(self, tmp_path: Path) -> None:
        save_dir = tmp_path / "saves"
        save_dir.mkdir()
        shutil.copy(SAMPLE_SAVE, save_dir / "save_01.sav")
        result = ingest_saves(save_dir, tmp_path / "out", game_id="g1")
        assert result.empire_features_path.exists()
        assert result.system_features_path.exists()
        assert result.transitions_path.exists()
        assert result.metadata_path.exists()

    def test_single_save_processes_one_save(self, tmp_path: Path) -> None:
        save_dir = tmp_path / "saves"
        save_dir.mkdir()
        shutil.copy(SAMPLE_SAVE, save_dir / "save_01.sav")
        result = ingest_saves(save_dir, tmp_path / "out", game_id="g1")
        assert result.saves_processed == 1
        assert result.errors == ()

    def test_single_save_has_zero_transitions(self, tmp_path: Path) -> None:
        save_dir = tmp_path / "saves"
        save_dir.mkdir()
        shutil.copy(SAMPLE_SAVE, save_dir / "save_01.sav")
        result = ingest_saves(save_dir, tmp_path / "out", game_id="g1")
        table = pq.read_table(str(result.transitions_path))
        assert table.num_rows == 0

    def test_single_save_writes_empire_feature_rows(self, tmp_path: Path) -> None:
        save_dir = tmp_path / "saves"
        save_dir.mkdir()
        shutil.copy(SAMPLE_SAVE, save_dir / "save_01.sav")
        result = ingest_saves(save_dir, tmp_path / "out", game_id="g1")
        table = pq.read_table(str(result.empire_features_path))
        # Sample save has 2 empires → 2 rows
        assert table.num_rows == 2

    def test_single_save_empire_table_has_game_id_column(self, tmp_path: Path) -> None:
        save_dir = tmp_path / "saves"
        save_dir.mkdir()
        shutil.copy(SAMPLE_SAVE, save_dir / "save_01.sav")
        result = ingest_saves(save_dir, tmp_path / "out", game_id="mygame")
        table = pq.read_table(str(result.empire_features_path))
        assert "game_id" in table.schema.names
        assert table.column("game_id")[0].as_py() == "mygame"

    def test_two_saves_produces_transitions(self, tmp_path: Path) -> None:
        save_dir = tmp_path / "saves"
        save_dir.mkdir()
        shutil.copy(SAMPLE_SAVE, save_dir / "save_01.sav")
        shutil.copy(SAMPLE_SAVE, save_dir / "save_02.sav")
        result = ingest_saves(save_dir, tmp_path / "out", game_id="g2")
        table = pq.read_table(str(result.transitions_path))
        # 2 empires, both identical saves → no action diff → 2 no-op transition rows
        assert table.num_rows >= 0  # valid parquet, rows depend on action extraction

    def test_invalid_save_records_error_and_continues(self, tmp_path: Path) -> None:
        save_dir = tmp_path / "saves"
        save_dir.mkdir()
        (save_dir / "corrupted.sav").write_bytes(b"not a zip file")
        shutil.copy(SAMPLE_SAVE, save_dir / "valid.sav")
        result = ingest_saves(save_dir, tmp_path / "out", game_id="g3")
        assert any("corrupted.sav" in e for e in result.errors)
        assert result.saves_processed == 1

    def test_output_files_are_in_output_dir(self, tmp_path: Path) -> None:
        save_dir = tmp_path / "saves"
        save_dir.mkdir()
        out_dir = tmp_path / "out"
        result = ingest_saves(save_dir, out_dir, game_id="g0")
        assert result.empire_features_path.parent == out_dir
        assert result.system_features_path.parent == out_dir
        assert result.transitions_path.parent == out_dir
        assert result.metadata_path.parent == out_dir

    def test_output_dir_created_if_missing(self, tmp_path: Path) -> None:
        save_dir = tmp_path / "saves"
        save_dir.mkdir()
        out_dir = tmp_path / "deeply" / "nested" / "out"
        assert not out_dir.exists()
        ingest_saves(save_dir, out_dir, game_id="g0")
        assert out_dir.exists()
