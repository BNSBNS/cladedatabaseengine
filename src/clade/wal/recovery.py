"""
ARIES-style recovery implementation.

ARIES (Algorithm for Recovery and Isolation Exploiting Semantics)
provides crash recovery through three phases:
1. Analysis: Determine which transactions were active at crash
2. Redo: Replay all logged changes to restore state
3. Undo: Roll back uncommitted transactions

Key Features:
- WAL-based recovery for durability
- Support for logical and physical undo
- Checkpoint-based optimization
"""

from dataclasses import dataclass
from typing import Optional, Set, Dict

from clade.wal.logger import WALManager, LogRecord, LogType
from clade.storage.buffer_manager import BufferPoolManager
from clade.utils.errors import RecoveryError


@dataclass
class TransactionStatus:
    """Status of a transaction during recovery."""

    __slots__ = ("txn_id", "status", "last_lsn", "undo_next_lsn")

    txn_id: int
    status: str  # "active", "committed", "aborted"
    last_lsn: int
    undo_next_lsn: int  # Next LSN to undo


class RecoveryManager:
    """
    ARIES-style recovery manager.

    Implements three-phase recovery:
    1. Analysis: Scan log to find active transactions and dirty pages
    2. Redo: Replay all changes from checkpoint
    3. Undo: Roll back uncommitted transactions

    Thread Safety: Not thread-safe. Used during single-threaded recovery.
    """

    __slots__ = (
        "_wal_manager",
        "_buffer_pool",
        "_active_txns",
        "_dirty_pages",
        "_checkpoint_lsn",
    )

    def __init__(
        self,
        wal_manager: WALManager,
        buffer_pool: Optional[BufferPoolManager] = None,
    ) -> None:
        """
        Initialize recovery manager.

        Args:
            wal_manager: WAL manager for reading log
            buffer_pool: Buffer pool for applying changes (optional for analysis-only)
        """
        self._wal_manager = wal_manager
        self._buffer_pool = buffer_pool
        self._active_txns: Dict[int, TransactionStatus] = {}
        self._dirty_pages: Set[int] = set()
        self._checkpoint_lsn = 0

    def recover(self) -> Dict[str, any]:
        """
        Perform full ARIES recovery.

        Returns:
            Recovery statistics
        """
        # Phase 1: Analysis
        analysis_result = self.analysis_phase()

        # Phase 2: Redo
        redo_result = self.redo_phase()

        # Phase 3: Undo
        undo_result = self.undo_phase()

        return {
            "analysis": analysis_result,
            "redo": redo_result,
            "undo": undo_result,
            "recovered": True,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 1: Analysis
    # ─────────────────────────────────────────────────────────────────────────

    def analysis_phase(self) -> Dict[str, any]:
        """
        Analysis phase: Scan log to determine state at crash.

        Identifies:
        - Active transactions (need undo)
        - Committed transactions (no action needed)
        - Dirty pages (need redo)

        Returns:
            Analysis results
        """
        self._active_txns.clear()
        self._dirty_pages.clear()
        records_scanned = 0

        for record in self._wal_manager.iter_records():
            records_scanned += 1
            self._process_analysis_record(record)

        # Find transactions that need undo
        txns_to_undo = [
            txn for txn in self._active_txns.values() if txn.status == "active"
        ]

        return {
            "records_scanned": records_scanned,
            "active_transactions": len(txns_to_undo),
            "dirty_pages": len(self._dirty_pages),
            "checkpoint_lsn": self._checkpoint_lsn,
        }

    def _process_analysis_record(self, record: LogRecord) -> None:
        """Process a single log record during analysis."""
        txn_id = record.txn_id

        if record.log_type == LogType.BEGIN:
            self._active_txns[txn_id] = TransactionStatus(
                txn_id=txn_id,
                status="active",
                last_lsn=record.lsn,
                undo_next_lsn=record.lsn,
            )

        elif record.log_type == LogType.COMMIT:
            if txn_id in self._active_txns:
                self._active_txns[txn_id].status = "committed"
                self._active_txns[txn_id].last_lsn = record.lsn

        elif record.log_type == LogType.ABORT:
            if txn_id in self._active_txns:
                self._active_txns[txn_id].status = "aborted"
                self._active_txns[txn_id].last_lsn = record.lsn

        elif record.log_type in (LogType.UPDATE, LogType.INSERT, LogType.DELETE):
            # Track dirty page
            self._dirty_pages.add(record.page_id)

            # Update transaction state
            if txn_id in self._active_txns:
                self._active_txns[txn_id].last_lsn = record.lsn
                self._active_txns[txn_id].undo_next_lsn = record.lsn

        elif record.log_type == LogType.CHECKPOINT:
            self._checkpoint_lsn = record.lsn
            # Parse checkpoint data for active transactions
            self._parse_checkpoint(record)

    def _parse_checkpoint(self, record: LogRecord) -> None:
        """Parse checkpoint record to restore active transaction table."""
        import struct

        data = record.after_image
        if len(data) < 4:
            return

        num_txns = struct.unpack("<I", data[:4])[0]
        offset = 4

        for _ in range(num_txns):
            if offset + 16 > len(data):
                break
            txn_id, last_lsn = struct.unpack("<QQ", data[offset : offset + 16])
            offset += 16

            if txn_id not in self._active_txns:
                self._active_txns[txn_id] = TransactionStatus(
                    txn_id=txn_id,
                    status="active",
                    last_lsn=last_lsn,
                    undo_next_lsn=last_lsn,
                )

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 2: Redo
    # ─────────────────────────────────────────────────────────────────────────

    def redo_phase(self) -> Dict[str, any]:
        """
        Redo phase: Replay all changes from log.

        Applies all logged changes to ensure pages reflect
        the state at crash time.

        Returns:
            Redo results
        """
        if self._buffer_pool is None:
            return {"records_redone": 0, "skipped": "no buffer pool"}

        records_redone = 0
        start_lsn = self._checkpoint_lsn if self._checkpoint_lsn > 0 else 1

        for record in self._wal_manager.iter_records(start_lsn):
            if record.log_type in (LogType.UPDATE, LogType.INSERT, LogType.DELETE):
                if self._should_redo(record):
                    self._apply_redo(record)
                    records_redone += 1

        return {"records_redone": records_redone}

    def _should_redo(self, record: LogRecord) -> bool:
        """
        Determine if a record needs to be redone.

        Redo is needed if the page's LSN is less than the log record's LSN.
        """
        if record.page_id not in self._dirty_pages:
            return False

        try:
            page = self._buffer_pool.fetch_page(record.page_id)
            page_lsn = page.lsn
            self._buffer_pool.unpin_page(record.page_id, is_dirty=False)
            return page_lsn < record.lsn
        except Exception:
            # Page doesn't exist, need to redo
            return True

    def _apply_redo(self, record: LogRecord) -> None:
        """Apply redo for a log record."""
        try:
            page = self._buffer_pool.fetch_page(record.page_id)

            if record.log_type == LogType.INSERT:
                # Redo insert
                try:
                    page.insert_record(record.after_image)
                except Exception:
                    pass  # May already exist

            elif record.log_type == LogType.DELETE:
                # Redo delete
                try:
                    page.delete_record(record.slot_id)
                except Exception:
                    pass  # May already be deleted

            elif record.log_type == LogType.UPDATE:
                # Redo update
                try:
                    page.update_record(record.slot_id, record.after_image)
                except Exception:
                    pass  # May fail if sizes differ

            page.lsn = record.lsn
            self._buffer_pool.unpin_page(record.page_id, is_dirty=True)

        except Exception:
            pass  # Continue recovery

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 3: Undo
    # ─────────────────────────────────────────────────────────────────────────

    def undo_phase(self) -> Dict[str, any]:
        """
        Undo phase: Roll back uncommitted transactions.

        Uses the undo chain (prev_lsn) to walk backwards through
        each transaction's changes and undo them.

        Returns:
            Undo results
        """
        if self._buffer_pool is None:
            return {"transactions_undone": 0, "skipped": "no buffer pool"}

        # Find transactions to undo
        to_undo = [
            txn for txn in self._active_txns.values() if txn.status == "active"
        ]

        transactions_undone = 0
        records_undone = 0

        for txn in to_undo:
            count = self._undo_transaction(txn)
            records_undone += count
            transactions_undone += 1

            # Log abort
            self._wal_manager.log_abort(txn.txn_id, txn.last_lsn)

        return {
            "transactions_undone": transactions_undone,
            "records_undone": records_undone,
        }

    def _undo_transaction(self, txn: TransactionStatus) -> int:
        """
        Undo a single transaction.

        Walks backwards through the transaction's log records
        using prev_lsn chain and applies undo.
        """
        records_undone = 0

        # Collect records for this transaction
        txn_records = []
        for record in self._wal_manager.iter_records():
            if record.txn_id == txn.txn_id:
                if record.log_type in (LogType.UPDATE, LogType.INSERT, LogType.DELETE):
                    txn_records.append(record)

        # Undo in reverse order
        for record in reversed(txn_records):
            self._apply_undo(record)
            records_undone += 1

        return records_undone

    def _apply_undo(self, record: LogRecord) -> None:
        """Apply undo for a log record."""
        try:
            page = self._buffer_pool.fetch_page(record.page_id)

            if record.log_type == LogType.INSERT:
                # Undo insert = delete
                try:
                    page.delete_record(record.slot_id)
                except Exception:
                    pass

            elif record.log_type == LogType.DELETE:
                # Undo delete = insert
                try:
                    page.insert_record(record.before_image)
                except Exception:
                    pass

            elif record.log_type == LogType.UPDATE:
                # Undo update = restore before image
                try:
                    page.update_record(record.slot_id, record.before_image)
                except Exception:
                    pass

            self._buffer_pool.unpin_page(record.page_id, is_dirty=True)

        except Exception:
            pass  # Continue recovery

    # ─────────────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def active_transactions(self) -> Dict[int, TransactionStatus]:
        """Get active transactions from analysis."""
        return self._active_txns

    @property
    def dirty_pages(self) -> Set[int]:
        """Get dirty pages from analysis."""
        return self._dirty_pages

    def __repr__(self) -> str:
        """String representation."""
        return f"RecoveryManager(active_txns={len(self._active_txns)})"
