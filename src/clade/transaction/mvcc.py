"""
Multi-Version Concurrency Control (MVCC) implementation.

MVCC allows multiple transactions to see consistent snapshots of data
without blocking reads with writes. Each transaction sees data as of
its start timestamp.

Key Concepts:
- xmin: Transaction that created this version
- xmax: Transaction that deleted this version (MAX_INT if current)
- Visibility rule: xmin <= snapshot_ts < xmax

Thread Safety: Thread-safe via internal locking.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Set
import threading

from clade.storage.interfaces import RecordID
from clade.transaction.timestamp import TimestampManager
from clade.utils.errors import WriteWriteConflictError, TransactionAbortedError


MAX_TIMESTAMP = 2**63 - 1  # Represents "infinity" (version is current)


@dataclass
class Version:
    """
    A versioned record in MVCC.

    Each version contains:
    - xmin: Creating transaction timestamp
    - xmax: Deleting transaction timestamp (MAX_TIMESTAMP if current)
    - data: The record data
    - prev: Pointer to older version (for version chain)
    """

    __slots__ = ("xmin", "xmax", "data", "prev", "rid")

    xmin: int  # Creating transaction
    xmax: int  # Deleting transaction (MAX_TIMESTAMP if current)
    data: bytes  # Record data
    prev: Optional["Version"]  # Older version
    rid: RecordID  # Physical location


@dataclass
class Transaction:
    """
    A transaction in the MVCC system.

    Tracks:
    - txn_id: Unique transaction ID
    - start_ts: Snapshot timestamp
    - status: "active", "committed", "aborted"
    - write_set: Records modified by this transaction
    """

    __slots__ = ("txn_id", "start_ts", "commit_ts", "status", "write_set")

    txn_id: int
    start_ts: int
    commit_ts: int
    status: str  # "active", "committed", "aborted"
    write_set: Set[RecordID]


class MVCCManager:
    """
    MVCC manager for snapshot isolation.

    Provides:
    - Snapshot-based reads (readers don't block writers)
    - Write-write conflict detection
    - Version chain management

    Thread Safety: Thread-safe via internal locking.
    """

    __slots__ = (
        "_timestamp_manager",
        "_active_txns",
        "_committed_txns",
        "_versions",
        "_lock",
        "_next_txn_id",
    )

    def __init__(self, timestamp_manager: Optional[TimestampManager] = None) -> None:
        """
        Initialize MVCC manager.

        Args:
            timestamp_manager: Timestamp manager (creates one if not provided)
        """
        self._timestamp_manager = timestamp_manager or TimestampManager()
        self._active_txns: Dict[int, Transaction] = {}
        self._committed_txns: Dict[int, int] = {}  # txn_id -> commit_ts
        self._versions: Dict[RecordID, Version] = {}  # Latest version per RID
        self._lock = threading.RLock()
        self._next_txn_id = 1

    # ─────────────────────────────────────────────────────────────────────────
    # Transaction Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def begin_transaction(self) -> int:
        """
        Begin a new transaction.

        Returns:
            Transaction ID
        """
        with self._lock:
            txn_id = self._next_txn_id
            self._next_txn_id += 1

            start_ts = self._timestamp_manager.next_timestamp()

            txn = Transaction(
                txn_id=txn_id,
                start_ts=start_ts,
                commit_ts=0,
                status="active",
                write_set=set(),
            )

            self._active_txns[txn_id] = txn
            return txn_id

    def commit_transaction(self, txn_id: int) -> None:
        """
        Commit a transaction.

        Makes all changes visible to future transactions.

        Args:
            txn_id: Transaction to commit

        Raises:
            TransactionAbortedError: If transaction was aborted
        """
        with self._lock:
            if txn_id not in self._active_txns:
                raise ValueError(f"Transaction {txn_id} not found")

            txn = self._active_txns[txn_id]

            if txn.status == "aborted":
                raise TransactionAbortedError(txn_id, "Transaction was already aborted")

            # Get commit timestamp
            commit_ts = self._timestamp_manager.next_timestamp()
            txn.commit_ts = commit_ts
            txn.status = "committed"

            # Track this transaction as committed (keep xmin as txn_id, lookup commit_ts)
            self._committed_txns[txn_id] = commit_ts

            # Remove from active transactions
            del self._active_txns[txn_id]

    def abort_transaction(self, txn_id: int) -> None:
        """
        Abort a transaction.

        Rolls back all changes made by the transaction.

        Args:
            txn_id: Transaction to abort
        """
        with self._lock:
            if txn_id not in self._active_txns:
                return

            txn = self._active_txns[txn_id]
            txn.status = "aborted"

            # Remove versions created by this transaction
            for rid in txn.write_set:
                if rid in self._versions:
                    version = self._versions[rid]
                    if version.xmin == txn_id:
                        # Restore previous version
                        if version.prev is not None:
                            self._versions[rid] = version.prev
                        else:
                            del self._versions[rid]

            del self._active_txns[txn_id]

    # ─────────────────────────────────────────────────────────────────────────
    # Version Management
    # ─────────────────────────────────────────────────────────────────────────

    def insert_version(self, txn_id: int, rid: RecordID, data: bytes) -> None:
        """
        Insert a new version.

        Args:
            txn_id: Transaction performing the insert
            rid: Record location
            data: Record data

        Raises:
            WriteWriteConflictError: If another active transaction modified this record
        """
        with self._lock:
            txn = self._get_active_txn(txn_id)

            # Check for write-write conflict
            if rid in self._versions:
                existing = self._versions[rid]
                if existing.xmax == MAX_TIMESTAMP:
                    # Check if another active transaction owns this
                    if existing.xmin in self._active_txns:
                        raise WriteWriteConflictError(
                            txn_id, existing.xmin, str(rid)
                        )

            # Create new version
            version = Version(
                xmin=txn_id,  # Will be updated to commit_ts on commit
                xmax=MAX_TIMESTAMP,
                data=data,
                prev=self._versions.get(rid),
                rid=rid,
            )

            self._versions[rid] = version
            txn.write_set.add(rid)

    def update_version(
        self, txn_id: int, rid: RecordID, new_data: bytes
    ) -> None:
        """
        Update a version (creates new version in chain).

        Args:
            txn_id: Transaction performing the update
            rid: Record location
            new_data: New record data

        Raises:
            WriteWriteConflictError: If another active transaction modified this record
            ValueError: If record doesn't exist
        """
        with self._lock:
            txn = self._get_active_txn(txn_id)

            if rid not in self._versions:
                raise ValueError(f"Record {rid} not found")

            latest_version = self._versions[rid]

            # Check for write-write conflict: if latest version was created by
            # another active transaction, we have a conflict
            if latest_version.xmin in self._active_txns and latest_version.xmin != txn_id:
                raise WriteWriteConflictError(txn_id, latest_version.xmin, str(rid))

            # Find the visible version in the chain
            old_version = latest_version
            while old_version is not None:
                if self._is_visible(old_version, txn.start_ts, txn_id):
                    break
                old_version = old_version.prev

            if old_version is None:
                raise ValueError(f"Record {rid} not visible to transaction")

            # Check for write-write conflict
            if old_version.xmax != MAX_TIMESTAMP:
                # Check if xmax is from another active transaction
                if old_version.xmax in self._active_txns and old_version.xmax != txn_id:
                    raise WriteWriteConflictError(txn_id, old_version.xmax, str(rid))
                # Otherwise it was already deleted by a committed transaction
                raise ValueError(f"Record {rid} was already deleted")

            # Mark old version as deleted
            old_version.xmax = txn_id

            # Create new version
            new_version = Version(
                xmin=txn_id,
                xmax=MAX_TIMESTAMP,
                data=new_data,
                prev=old_version,
                rid=rid,
            )

            self._versions[rid] = new_version
            txn.write_set.add(rid)

    def delete_version(self, txn_id: int, rid: RecordID) -> None:
        """
        Delete a version (marks xmax).

        Args:
            txn_id: Transaction performing the delete
            rid: Record location

        Raises:
            WriteWriteConflictError: If another active transaction modified this record
            ValueError: If record doesn't exist
        """
        with self._lock:
            txn = self._get_active_txn(txn_id)

            if rid not in self._versions:
                raise ValueError(f"Record {rid} not found")

            latest_version = self._versions[rid]

            # Check for write-write conflict: if latest version was created by
            # another active transaction, we have a conflict
            if latest_version.xmin in self._active_txns and latest_version.xmin != txn_id:
                raise WriteWriteConflictError(txn_id, latest_version.xmin, str(rid))

            # Find the visible version in the chain
            version = latest_version
            while version is not None:
                if self._is_visible(version, txn.start_ts, txn_id):
                    break
                version = version.prev

            if version is None:
                raise ValueError(f"Record {rid} not visible to transaction")

            # Check for write-write conflict on the visible version
            if version.xmax != MAX_TIMESTAMP:
                if version.xmax in self._active_txns and version.xmax != txn_id:
                    raise WriteWriteConflictError(txn_id, version.xmax, str(rid))
                raise ValueError(f"Record {rid} was already deleted")

            # Mark as deleted
            version.xmax = txn_id
            txn.write_set.add(rid)

    def read_version(self, txn_id: int, rid: RecordID) -> Optional[bytes]:
        """
        Read the visible version of a record.

        Args:
            txn_id: Transaction performing the read
            rid: Record location

        Returns:
            Record data if visible, None otherwise
        """
        with self._lock:
            txn = self._get_active_txn(txn_id)

            if rid not in self._versions:
                return None

            # Find visible version in chain
            version = self._versions[rid]
            while version is not None:
                if self._is_visible(version, txn.start_ts, txn_id):
                    return version.data
                version = version.prev

            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Visibility Rules
    # ─────────────────────────────────────────────────────────────────────────

    def _is_visible(self, version: Version, snapshot_ts: int, txn_id: int = 0) -> bool:
        """
        Check if a version is visible at a given snapshot timestamp.

        Visibility rules:
        1. xmin must be committed before snapshot, OR be our own transaction
        2. xmax must be MAX_TIMESTAMP, or from an uncommitted transaction,
           or committed after our snapshot

        Args:
            version: The version to check
            snapshot_ts: The snapshot timestamp
            txn_id: The requesting transaction ID (for seeing own changes)
        """
        xmin = version.xmin
        xmax = version.xmax

        # Check xmin (creation) visibility
        if xmin == txn_id:
            # Our own change - always visible to us
            pass
        elif xmin in self._active_txns:
            # Created by another active (uncommitted) transaction - invisible
            return False
        elif xmin in self._committed_txns:
            # Created by a committed transaction - check commit time
            commit_ts = self._committed_txns[xmin]
            if commit_ts > snapshot_ts:
                # Committed after our snapshot - invisible
                return False
        else:
            # Unknown transaction - shouldn't happen in normal operation
            # Treat as visible (legacy data from before tracking)
            pass

        # Check xmax (deletion) visibility
        if xmax == MAX_TIMESTAMP:
            # Not deleted - visible
            return True

        if xmax == txn_id:
            # We deleted it - invisible to us
            return False

        if xmax in self._active_txns:
            # Deleted by another active (uncommitted) transaction
            # Still visible to us (delete not committed)
            return True

        if xmax in self._committed_txns:
            # Deleted by a committed transaction - check commit time
            commit_ts = self._committed_txns[xmax]
            if commit_ts <= snapshot_ts:
                # Deleted before or at our snapshot - invisible
                return False
            # Deleted after our snapshot - visible
            return True

        # Unknown deletion transaction - shouldn't happen
        return True

    def _get_active_txn(self, txn_id: int) -> Transaction:
        """Get an active transaction or raise error."""
        if txn_id not in self._active_txns:
            raise ValueError(f"Transaction {txn_id} not found or not active")
        return self._active_txns[txn_id]

    # ─────────────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def active_transaction_count(self) -> int:
        """Get the number of active transactions."""
        return len(self._active_txns)

    def is_active(self, txn_id: int) -> bool:
        """Check if a transaction is active."""
        return txn_id in self._active_txns

    def __repr__(self) -> str:
        """String representation."""
        return f"MVCCManager(active_txns={len(self._active_txns)})"
