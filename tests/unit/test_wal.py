"""
Unit tests for Write-Ahead Log implementation.

Tests cover:
- Log record serialization
- WAL logging operations
- Log iteration
- Recovery phases
"""

import pytest
import tempfile
from pathlib import Path

from clade.wal.logger import WALManager, LogRecord, LogType
from clade.wal.recovery import RecoveryManager, TransactionStatus
from clade.storage.interfaces import RecordID


@pytest.fixture
def temp_wal_path():
    """Create a temporary directory for WAL."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "wal"


@pytest.fixture
def wal_manager(temp_wal_path):
    """Create a WAL manager for testing."""
    wm = WALManager(temp_wal_path, group_commit_enabled=False)
    yield wm
    wm.close()


class TestLogRecord:
    """Tests for LogRecord serialization."""

    def test_serialize_deserialize(self):
        """LogRecord should serialize and deserialize correctly."""
        record = LogRecord(
            lsn=42,
            txn_id=100,
            prev_lsn=41,
            log_type=LogType.UPDATE,
            page_id=5,
            slot_id=3,
            before_image=b"before",
            after_image=b"after",
        )

        data = record.to_bytes()
        restored, consumed = LogRecord.from_bytes(data)

        assert restored.lsn == 42
        assert restored.txn_id == 100
        assert restored.prev_lsn == 41
        assert restored.log_type == LogType.UPDATE
        assert restored.page_id == 5
        assert restored.slot_id == 3
        assert restored.before_image == b"before"
        assert restored.after_image == b"after"

    def test_serialize_empty_images(self):
        """Should handle empty before/after images."""
        record = LogRecord(
            lsn=1,
            txn_id=1,
            prev_lsn=0,
            log_type=LogType.BEGIN,
            page_id=0,
            slot_id=0,
            before_image=b"",
            after_image=b"",
        )

        data = record.to_bytes()
        restored, _ = LogRecord.from_bytes(data)

        assert restored.before_image == b""
        assert restored.after_image == b""

    def test_serialize_large_images(self):
        """Should handle large before/after images."""
        large_before = b"X" * 10000
        large_after = b"Y" * 10000

        record = LogRecord(
            lsn=1,
            txn_id=1,
            prev_lsn=0,
            log_type=LogType.UPDATE,
            page_id=1,
            slot_id=1,
            before_image=large_before,
            after_image=large_after,
        )

        data = record.to_bytes()
        restored, _ = LogRecord.from_bytes(data)

        assert restored.before_image == large_before
        assert restored.after_image == large_after


class TestWALManagerCreation:
    """Tests for WAL manager creation."""

    def test_create_new_wal(self, temp_wal_path):
        """Should create a new WAL file."""
        wm = WALManager(temp_wal_path)

        assert (temp_wal_path / "wal.log").exists()
        assert wm.current_lsn == 1

        wm.close()

    def test_reopen_existing_wal(self, temp_wal_path):
        """Should reopen existing WAL and recover LSN."""
        # Create and write some records
        wm1 = WALManager(temp_wal_path, group_commit_enabled=False)
        wm1.log_begin(1)
        wm1.log_begin(2)
        wm1.flush()  # Ensure records are written to disk
        wm1.close()

        # Reopen
        wm2 = WALManager(temp_wal_path, group_commit_enabled=False)
        assert wm2.current_lsn == 3  # Next available LSN

        wm2.close()


class TestWALLogging:
    """Tests for WAL logging operations."""

    def test_log_begin(self, wal_manager):
        """Should log transaction begin."""
        lsn = wal_manager.log_begin(txn_id=1)

        assert lsn == 1
        assert wal_manager.current_lsn == 2

    def test_log_commit(self, wal_manager):
        """Should log transaction commit."""
        begin_lsn = wal_manager.log_begin(txn_id=1)
        commit_lsn = wal_manager.log_commit(txn_id=1, prev_lsn=begin_lsn)

        assert commit_lsn == 2

    def test_log_abort(self, wal_manager):
        """Should log transaction abort."""
        begin_lsn = wal_manager.log_begin(txn_id=1)
        abort_lsn = wal_manager.log_abort(txn_id=1, prev_lsn=begin_lsn)

        assert abort_lsn == 2

    def test_log_update(self, wal_manager):
        """Should log page update."""
        begin_lsn = wal_manager.log_begin(txn_id=1)
        update_lsn = wal_manager.log_update(
            txn_id=1,
            prev_lsn=begin_lsn,
            page_id=5,
            slot_id=3,
            before_image=b"old",
            after_image=b"new",
        )

        assert update_lsn == 2

    def test_log_insert(self, wal_manager):
        """Should log record insert."""
        begin_lsn = wal_manager.log_begin(txn_id=1)
        insert_lsn = wal_manager.log_insert(
            txn_id=1,
            prev_lsn=begin_lsn,
            page_id=1,
            slot_id=0,
            record=b"new record",
        )

        assert insert_lsn == 2

    def test_log_delete(self, wal_manager):
        """Should log record delete."""
        begin_lsn = wal_manager.log_begin(txn_id=1)
        delete_lsn = wal_manager.log_delete(
            txn_id=1,
            prev_lsn=begin_lsn,
            page_id=1,
            slot_id=0,
            record=b"deleted record",
        )

        assert delete_lsn == 2

    def test_log_checkpoint(self, wal_manager):
        """Should log checkpoint."""
        wal_manager.log_begin(txn_id=1)
        wal_manager.log_begin(txn_id=2)

        active_txns = {1: 1, 2: 2}
        cp_lsn = wal_manager.log_checkpoint(active_txns)

        assert cp_lsn == 3


class TestWALIteration:
    """Tests for WAL record iteration."""

    def test_iter_empty_wal(self, wal_manager):
        """Should iterate empty WAL without error."""
        records = list(wal_manager.iter_records())

        assert records == []

    def test_iter_all_records(self, wal_manager):
        """Should iterate all logged records."""
        wal_manager.log_begin(txn_id=1)
        wal_manager.log_update(1, 1, 5, 0, b"before", b"after")
        wal_manager.log_commit(1, 2)

        wal_manager.flush()
        records = list(wal_manager.iter_records())

        assert len(records) == 3
        assert records[0].log_type == LogType.BEGIN
        assert records[1].log_type == LogType.UPDATE
        assert records[2].log_type == LogType.COMMIT

    def test_iter_from_lsn(self, wal_manager):
        """Should iterate from specific LSN."""
        wal_manager.log_begin(txn_id=1)
        wal_manager.log_begin(txn_id=2)
        wal_manager.log_begin(txn_id=3)

        wal_manager.flush()
        records = list(wal_manager.iter_records(start_lsn=2))

        assert len(records) == 2
        assert records[0].lsn == 2
        assert records[1].lsn == 3


class TestWALPersistence:
    """Tests for WAL persistence."""

    def test_records_persist_after_close(self, temp_wal_path):
        """Records should persist after close and reopen."""
        # Write records
        wm1 = WALManager(temp_wal_path)
        wm1.log_begin(txn_id=1)
        wm1.log_update(1, 1, 5, 0, b"before", b"after")
        wm1.log_commit(1, 2)
        wm1.close()

        # Reopen and verify
        wm2 = WALManager(temp_wal_path)
        records = list(wm2.iter_records())

        assert len(records) == 3
        assert records[0].log_type == LogType.BEGIN
        assert records[1].log_type == LogType.UPDATE
        assert records[2].log_type == LogType.COMMIT

        wm2.close()


class TestRecoveryManager:
    """Tests for recovery manager."""

    def test_analysis_phase_empty(self, temp_wal_path):
        """Analysis on empty WAL should find nothing."""
        wm = WALManager(temp_wal_path)
        rm = RecoveryManager(wm)

        result = rm.analysis_phase()

        assert result["records_scanned"] == 0
        assert result["active_transactions"] == 0

        wm.close()

    def test_analysis_finds_active_transaction(self, temp_wal_path):
        """Analysis should find uncommitted transactions."""
        wm = WALManager(temp_wal_path)
        wm.log_begin(txn_id=1)
        wm.log_update(1, 1, 5, 0, b"before", b"after")
        wm.flush()

        rm = RecoveryManager(wm)
        result = rm.analysis_phase()

        assert result["active_transactions"] == 1
        assert 1 in rm.active_transactions
        assert rm.active_transactions[1].status == "active"

        wm.close()

    def test_analysis_ignores_committed(self, temp_wal_path):
        """Analysis should not count committed transactions as active."""
        wm = WALManager(temp_wal_path)
        lsn1 = wm.log_begin(txn_id=1)
        lsn2 = wm.log_update(1, lsn1, 5, 0, b"before", b"after")
        wm.log_commit(1, lsn2)
        wm.flush()

        rm = RecoveryManager(wm)
        result = rm.analysis_phase()

        assert result["active_transactions"] == 0
        assert rm.active_transactions[1].status == "committed"

        wm.close()

    def test_analysis_tracks_dirty_pages(self, temp_wal_path):
        """Analysis should track dirty pages."""
        wm = WALManager(temp_wal_path)
        wm.log_begin(txn_id=1)
        wm.log_update(1, 1, 5, 0, b"before", b"after")
        wm.log_update(1, 2, 10, 0, b"before", b"after")
        wm.flush()

        rm = RecoveryManager(wm)
        rm.analysis_phase()

        assert 5 in rm.dirty_pages
        assert 10 in rm.dirty_pages

        wm.close()

    def test_full_recovery(self, temp_wal_path):
        """Full recovery should complete all phases."""
        wm = WALManager(temp_wal_path)
        wm.log_begin(txn_id=1)
        wm.log_update(1, 1, 5, 0, b"before", b"after")
        wm.flush()

        rm = RecoveryManager(wm)
        result = rm.recover()

        assert result["recovered"]
        assert "analysis" in result
        assert "redo" in result
        assert "undo" in result

        wm.close()


class TestWALProperties:
    """Tests for WAL manager properties."""

    def test_current_lsn(self, wal_manager):
        """current_lsn should track next LSN."""
        assert wal_manager.current_lsn == 1

        wal_manager.log_begin(1)
        assert wal_manager.current_lsn == 2

    def test_flushed_lsn(self, wal_manager):
        """flushed_lsn should track flushed records."""
        wal_manager.log_begin(1)
        wal_manager.log_begin(2)

        wal_manager.flush()

        assert wal_manager.flushed_lsn == 2

    def test_repr(self, wal_manager):
        """__repr__ should return useful info."""
        repr_str = repr(wal_manager)

        assert "WALManager" in repr_str
        assert "lsn=" in repr_str
