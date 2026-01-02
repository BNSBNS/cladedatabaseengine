"""
Unit tests for transaction management and MVCC.

Tests cover:
- Timestamp manager
- Transaction lifecycle
- MVCC visibility rules
- Write-write conflict detection
"""

import pytest
import threading

from clade.transaction.timestamp import TimestampManager
from clade.transaction.mvcc import MVCCManager, Version, MAX_TIMESTAMP
from clade.storage.interfaces import RecordID
from clade.utils.errors import WriteWriteConflictError, TransactionAbortedError


def rid(page: int, slot: int = 0) -> RecordID:
    """Helper to create RecordID."""
    return RecordID(page_id=page, slot_id=slot)


class TestTimestampManager:
    """Tests for TimestampManager."""

    def test_initial_timestamp(self):
        """Should start with specified initial timestamp."""
        tm = TimestampManager(initial_ts=100)
        assert tm.current_timestamp() == 100

    def test_next_timestamp_increments(self):
        """next_timestamp should return incrementing values."""
        tm = TimestampManager()

        ts1 = tm.next_timestamp()
        ts2 = tm.next_timestamp()
        ts3 = tm.next_timestamp()

        assert ts1 == 1
        assert ts2 == 2
        assert ts3 == 3

    def test_current_timestamp_readonly(self):
        """current_timestamp should not increment."""
        tm = TimestampManager()
        tm.next_timestamp()

        ts1 = tm.current_timestamp()
        ts2 = tm.current_timestamp()

        assert ts1 == ts2 == 1

    def test_thread_safety(self):
        """Timestamps should be unique across threads."""
        tm = TimestampManager()
        timestamps = []
        lock = threading.Lock()

        def get_timestamps():
            for _ in range(100):
                ts = tm.next_timestamp()
                with lock:
                    timestamps.append(ts)

        threads = [threading.Thread(target=get_timestamps) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All timestamps should be unique
        assert len(timestamps) == 400
        assert len(set(timestamps)) == 400


class TestMVCCBasicOperations:
    """Tests for basic MVCC operations."""

    def test_begin_transaction(self):
        """Should begin a new transaction."""
        mvcc = MVCCManager()

        txn1 = mvcc.begin_transaction()
        txn2 = mvcc.begin_transaction()

        assert txn1 == 1
        assert txn2 == 2
        assert mvcc.active_transaction_count == 2

    def test_commit_transaction(self):
        """Should commit a transaction."""
        mvcc = MVCCManager()
        txn_id = mvcc.begin_transaction()

        mvcc.commit_transaction(txn_id)

        assert not mvcc.is_active(txn_id)
        assert mvcc.active_transaction_count == 0

    def test_abort_transaction(self):
        """Should abort a transaction."""
        mvcc = MVCCManager()
        txn_id = mvcc.begin_transaction()

        mvcc.abort_transaction(txn_id)

        assert not mvcc.is_active(txn_id)

    def test_commit_nonexistent_fails(self):
        """Committing nonexistent transaction should fail."""
        mvcc = MVCCManager()

        with pytest.raises(ValueError):
            mvcc.commit_transaction(999)


class TestMVCCVersioning:
    """Tests for MVCC version management."""

    def test_insert_and_read(self):
        """Should insert and read a version."""
        mvcc = MVCCManager()
        txn_id = mvcc.begin_transaction()

        mvcc.insert_version(txn_id, rid(1, 0), b"Hello")
        result = mvcc.read_version(txn_id, rid(1, 0))

        assert result == b"Hello"

    def test_read_nonexistent(self):
        """Reading nonexistent record should return None."""
        mvcc = MVCCManager()
        txn_id = mvcc.begin_transaction()

        result = mvcc.read_version(txn_id, rid(999, 0))

        assert result is None

    def test_update_version(self):
        """Should update a version."""
        mvcc = MVCCManager()
        txn_id = mvcc.begin_transaction()

        mvcc.insert_version(txn_id, rid(1, 0), b"Original")
        mvcc.update_version(txn_id, rid(1, 0), b"Updated")

        result = mvcc.read_version(txn_id, rid(1, 0))
        assert result == b"Updated"

    def test_delete_version(self):
        """Should delete a version."""
        mvcc = MVCCManager()
        txn_id = mvcc.begin_transaction()

        mvcc.insert_version(txn_id, rid(1, 0), b"Data")
        mvcc.delete_version(txn_id, rid(1, 0))

        # Deleted version not visible after commit
        mvcc.commit_transaction(txn_id)

        txn2 = mvcc.begin_transaction()
        result = mvcc.read_version(txn2, rid(1, 0))
        assert result is None


class TestMVCCIsolation:
    """Tests for MVCC isolation properties."""

    def test_snapshot_isolation(self):
        """Transactions should see consistent snapshots."""
        mvcc = MVCCManager()

        # T1 inserts and commits
        t1 = mvcc.begin_transaction()
        mvcc.insert_version(t1, rid(1, 0), b"V1")
        mvcc.commit_transaction(t1)

        # T2 starts (sees V1)
        t2 = mvcc.begin_transaction()

        # T3 updates to V2 and commits
        t3 = mvcc.begin_transaction()
        mvcc.update_version(t3, rid(1, 0), b"V2")
        mvcc.commit_transaction(t3)

        # T2 should still see V1 (its snapshot)
        result = mvcc.read_version(t2, rid(1, 0))
        assert result == b"V1"

        # New transaction T4 sees V2
        t4 = mvcc.begin_transaction()
        result = mvcc.read_version(t4, rid(1, 0))
        assert result == b"V2"

    def test_uncommitted_invisible_to_others(self):
        """Uncommitted changes should not be visible to other transactions."""
        mvcc = MVCCManager()

        t1 = mvcc.begin_transaction()
        mvcc.insert_version(t1, rid(1, 0), b"Uncommitted")

        t2 = mvcc.begin_transaction()
        result = mvcc.read_version(t2, rid(1, 0))

        # T2 should not see T1's uncommitted insert
        assert result is None


class TestMVCCConflicts:
    """Tests for write-write conflict detection."""

    def test_write_write_conflict(self):
        """Concurrent updates should cause conflict."""
        mvcc = MVCCManager()

        # T1 inserts and commits
        t1 = mvcc.begin_transaction()
        mvcc.insert_version(t1, rid(1, 0), b"Original")
        mvcc.commit_transaction(t1)

        # T2 and T3 both try to update
        t2 = mvcc.begin_transaction()
        t3 = mvcc.begin_transaction()

        mvcc.update_version(t2, rid(1, 0), b"T2 update")

        # T3's update should fail
        with pytest.raises(WriteWriteConflictError):
            mvcc.update_version(t3, rid(1, 0), b"T3 update")

    def test_no_conflict_after_commit(self):
        """After commit, other transactions can update."""
        mvcc = MVCCManager()

        t1 = mvcc.begin_transaction()
        mvcc.insert_version(t1, rid(1, 0), b"Original")
        mvcc.commit_transaction(t1)

        t2 = mvcc.begin_transaction()
        mvcc.update_version(t2, rid(1, 0), b"T2 update")
        mvcc.commit_transaction(t2)

        t3 = mvcc.begin_transaction()
        mvcc.update_version(t3, rid(1, 0), b"T3 update")

        result = mvcc.read_version(t3, rid(1, 0))
        assert result == b"T3 update"


class TestMVCCAbort:
    """Tests for transaction abort behavior."""

    def test_abort_rolls_back_insert(self):
        """Abort should roll back inserted versions."""
        mvcc = MVCCManager()

        t1 = mvcc.begin_transaction()
        mvcc.insert_version(t1, rid(1, 0), b"Data")
        mvcc.abort_transaction(t1)

        t2 = mvcc.begin_transaction()
        result = mvcc.read_version(t2, rid(1, 0))

        assert result is None

    def test_abort_restores_previous_version(self):
        """Abort should restore previous version after update."""
        mvcc = MVCCManager()

        # Insert and commit V1
        t1 = mvcc.begin_transaction()
        mvcc.insert_version(t1, rid(1, 0), b"V1")
        mvcc.commit_transaction(t1)

        # Update to V2 then abort
        t2 = mvcc.begin_transaction()
        mvcc.update_version(t2, rid(1, 0), b"V2")
        mvcc.abort_transaction(t2)

        # Should see V1 again
        t3 = mvcc.begin_transaction()
        result = mvcc.read_version(t3, rid(1, 0))

        assert result == b"V1"

    def test_commit_after_abort_fails(self):
        """Cannot commit an aborted transaction."""
        mvcc = MVCCManager()

        t1 = mvcc.begin_transaction()
        mvcc.abort_transaction(t1)

        # Transaction is gone, commit will fail
        with pytest.raises(ValueError):
            mvcc.commit_transaction(t1)


class TestMVCCProperties:
    """Tests for MVCC manager properties."""

    def test_active_transaction_count(self):
        """active_transaction_count should track active transactions."""
        mvcc = MVCCManager()

        assert mvcc.active_transaction_count == 0

        t1 = mvcc.begin_transaction()
        assert mvcc.active_transaction_count == 1

        t2 = mvcc.begin_transaction()
        assert mvcc.active_transaction_count == 2

        mvcc.commit_transaction(t1)
        assert mvcc.active_transaction_count == 1

    def test_is_active(self):
        """is_active should return correct status."""
        mvcc = MVCCManager()

        t1 = mvcc.begin_transaction()
        assert mvcc.is_active(t1)

        mvcc.commit_transaction(t1)
        assert not mvcc.is_active(t1)

    def test_repr(self):
        """__repr__ should return useful info."""
        mvcc = MVCCManager()
        mvcc.begin_transaction()

        repr_str = repr(mvcc)

        assert "MVCCManager" in repr_str
        assert "active_txns=" in repr_str
