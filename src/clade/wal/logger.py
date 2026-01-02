"""
Write-Ahead Log (WAL) implementation for durability.

The WAL ensures durability by logging all changes before they are
applied to data pages. Supports group commit for batching fsync
operations and ARIES-style recovery.

Key Features:
- Append-only log with sequential writes
- Group commit for batching fsync (reduces I/O)
- Log Sequence Numbers (LSN) for ordering
- Supports redo/undo logging
"""

import os
import struct
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator, Optional, BinaryIO, Union, List

from clade.storage.interfaces import RecordID
from clade.utils.errors import WALError, WALCorruptionError


# WAL record header size
RECORD_HEADER_SIZE = 32  # LSN(8) + txn_id(8) + prev_lsn(8) + type(4) + length(4)
WAL_MAGIC = 0x57414C31  # "WAL1"


class LogType(Enum):
    """Type of log record."""

    BEGIN = 1  # Transaction begin
    COMMIT = 2  # Transaction commit
    ABORT = 3  # Transaction abort
    UPDATE = 4  # Page update (with before/after images)
    INSERT = 5  # Insert record
    DELETE = 6  # Delete record
    CHECKPOINT = 7  # Checkpoint record
    CLR = 8  # Compensation Log Record (for undo)


@dataclass
class LogRecord:
    """
    A record in the write-ahead log.

    Layout:
    - lsn: Log Sequence Number (global ordering)
    - txn_id: Transaction ID
    - prev_lsn: Previous LSN in same transaction (for undo chain)
    - log_type: Type of operation
    - page_id: Target page (for UPDATE/INSERT/DELETE)
    - slot_id: Target slot (for record operations)
    - before_image: Data before change (for undo)
    - after_image: Data after change (for redo)
    """

    __slots__ = (
        "lsn",
        "txn_id",
        "prev_lsn",
        "log_type",
        "page_id",
        "slot_id",
        "before_image",
        "after_image",
    )

    lsn: int
    txn_id: int
    prev_lsn: int
    log_type: LogType
    page_id: int
    slot_id: int
    before_image: bytes
    after_image: bytes

    def to_bytes(self) -> bytes:
        """Serialize log record to bytes."""
        before_len = len(self.before_image)
        after_len = len(self.after_image)

        # Header: LSN, txn_id, prev_lsn, type, total_length
        # Body: page_id(4) + slot_id(4) + before_len(4) + after_len(4) = 16 bytes
        total_len = RECORD_HEADER_SIZE + 16 + before_len + after_len
        header = struct.pack(
            "<QQQII",
            self.lsn,
            self.txn_id,
            self.prev_lsn,
            self.log_type.value,
            total_len,
        )

        # Body: page_id, slot_id, before_len, after_len, before, after
        body = struct.pack(
            "<IIII",
            self.page_id,
            self.slot_id,
            before_len,
            after_len,
        )

        return header + body + self.before_image + self.after_image

    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> tuple["LogRecord", int]:
        """
        Deserialize log record from bytes.

        Returns:
            Tuple of (LogRecord, bytes_consumed)
        """
        if len(data) - offset < RECORD_HEADER_SIZE:
            raise WALCorruptionError(0, "Incomplete header")

        # Parse header
        lsn, txn_id, prev_lsn, log_type_val, total_len = struct.unpack_from(
            "<QQQII", data, offset
        )

        if len(data) - offset < total_len:
            raise WALCorruptionError(lsn, "Incomplete record")

        # Parse body
        body_offset = offset + RECORD_HEADER_SIZE
        page_id, slot_id, before_len, after_len = struct.unpack_from(
            "<IIII", data, body_offset
        )

        # Parse images
        image_offset = body_offset + 16
        before_image = data[image_offset : image_offset + before_len]
        after_image = data[
            image_offset + before_len : image_offset + before_len + after_len
        ]

        return (
            cls(
                lsn=lsn,
                txn_id=txn_id,
                prev_lsn=prev_lsn,
                log_type=LogType(log_type_val),
                page_id=page_id,
                slot_id=slot_id,
                before_image=before_image,
                after_image=after_image,
            ),
            total_len,
        )


class WALManager:
    """
    Write-Ahead Log manager.

    Manages the WAL file and provides logging operations.
    Supports group commit for batching fsync operations.

    Thread Safety: Thread-safe via internal locking.
    """

    __slots__ = (
        "_path",
        "_file",
        "_current_lsn",
        "_flushed_lsn",
        "_lock",
        "_group_commit_enabled",
        "_group_commit_interval_ms",
        "_last_flush_time",
        "_pending_commits",
    )

    def __init__(
        self,
        path: Union[str, Path],
        group_commit_enabled: bool = True,
        group_commit_interval_ms: int = 10,
    ) -> None:
        """
        Initialize WAL manager.

        Args:
            path: Path to WAL directory
            group_commit_enabled: Enable group commit batching
            group_commit_interval_ms: Max wait time for group commit
        """
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._group_commit_enabled = group_commit_enabled
        self._group_commit_interval_ms = group_commit_interval_ms
        self._last_flush_time = time.time()
        self._pending_commits: List[int] = []

        # Open or create WAL file
        wal_file = self._path / "wal.log"
        if wal_file.exists():
            self._file = open(wal_file, "r+b")
            self._current_lsn = self._recover_lsn()
        else:
            self._file = open(wal_file, "w+b")
            self._write_header()
            self._current_lsn = 1

        self._flushed_lsn = self._current_lsn - 1

    def _write_header(self) -> None:
        """Write WAL file header."""
        header = struct.pack("<II", WAL_MAGIC, 1)  # Magic + version
        self._file.write(header)
        self._file.flush()

    def _recover_lsn(self) -> int:
        """Recover the next LSN from existing WAL."""
        self._file.seek(0, 2)  # Seek to end
        file_size = self._file.tell()

        if file_size < 8:
            return 1

        # Scan to find last valid LSN
        self._file.seek(8)  # Skip header
        max_lsn = 0

        try:
            while self._file.tell() < file_size:
                pos = self._file.tell()
                header_data = self._file.read(RECORD_HEADER_SIZE)
                if len(header_data) < RECORD_HEADER_SIZE:
                    break

                lsn, _, _, _, total_len = struct.unpack("<QQQII", header_data)

                # Validate total_len to avoid infinite loop
                if total_len < RECORD_HEADER_SIZE or total_len > file_size:
                    break

                max_lsn = max(max_lsn, lsn)

                # Skip to next record
                self._file.seek(pos + total_len)
        except Exception:
            pass

        return max_lsn + 1

    # ─────────────────────────────────────────────────────────────────────────
    # Logging Operations
    # ─────────────────────────────────────────────────────────────────────────

    def log_begin(self, txn_id: int) -> int:
        """
        Log transaction begin.

        Args:
            txn_id: Transaction ID

        Returns:
            LSN of the log record
        """
        return self._append_log(
            txn_id=txn_id,
            prev_lsn=0,
            log_type=LogType.BEGIN,
            page_id=0,
            slot_id=0,
            before_image=b"",
            after_image=b"",
        )

    def log_commit(self, txn_id: int, prev_lsn: int) -> int:
        """
        Log transaction commit.

        Args:
            txn_id: Transaction ID
            prev_lsn: Previous LSN in transaction

        Returns:
            LSN of the log record
        """
        lsn = self._append_log(
            txn_id=txn_id,
            prev_lsn=prev_lsn,
            log_type=LogType.COMMIT,
            page_id=0,
            slot_id=0,
            before_image=b"",
            after_image=b"",
        )

        # Force flush for commit (durability)
        if self._group_commit_enabled:
            self._pending_commits.append(lsn)
            self._maybe_group_flush()
        else:
            self.flush()

        return lsn

    def log_abort(self, txn_id: int, prev_lsn: int) -> int:
        """Log transaction abort."""
        return self._append_log(
            txn_id=txn_id,
            prev_lsn=prev_lsn,
            log_type=LogType.ABORT,
            page_id=0,
            slot_id=0,
            before_image=b"",
            after_image=b"",
        )

    def log_update(
        self,
        txn_id: int,
        prev_lsn: int,
        page_id: int,
        slot_id: int,
        before_image: bytes,
        after_image: bytes,
    ) -> int:
        """
        Log a page update.

        Args:
            txn_id: Transaction ID
            prev_lsn: Previous LSN in transaction
            page_id: Target page
            slot_id: Target slot
            before_image: Data before change (for undo)
            after_image: Data after change (for redo)

        Returns:
            LSN of the log record
        """
        return self._append_log(
            txn_id=txn_id,
            prev_lsn=prev_lsn,
            log_type=LogType.UPDATE,
            page_id=page_id,
            slot_id=slot_id,
            before_image=before_image,
            after_image=after_image,
        )

    def log_insert(
        self,
        txn_id: int,
        prev_lsn: int,
        page_id: int,
        slot_id: int,
        record: bytes,
    ) -> int:
        """Log a record insert."""
        return self._append_log(
            txn_id=txn_id,
            prev_lsn=prev_lsn,
            log_type=LogType.INSERT,
            page_id=page_id,
            slot_id=slot_id,
            before_image=b"",
            after_image=record,
        )

    def log_delete(
        self,
        txn_id: int,
        prev_lsn: int,
        page_id: int,
        slot_id: int,
        record: bytes,
    ) -> int:
        """Log a record delete."""
        return self._append_log(
            txn_id=txn_id,
            prev_lsn=prev_lsn,
            log_type=LogType.DELETE,
            page_id=page_id,
            slot_id=slot_id,
            before_image=record,
            after_image=b"",
        )

    def log_checkpoint(self, active_txns: dict[int, int]) -> int:
        """
        Log a checkpoint.

        Args:
            active_txns: Map of txn_id -> last_lsn for active transactions

        Returns:
            LSN of checkpoint record
        """
        # Serialize active transactions
        data = struct.pack("<I", len(active_txns))
        for txn_id, last_lsn in active_txns.items():
            data += struct.pack("<QQ", txn_id, last_lsn)

        lsn = self._append_log(
            txn_id=0,
            prev_lsn=0,
            log_type=LogType.CHECKPOINT,
            page_id=0,
            slot_id=0,
            before_image=b"",
            after_image=data,
        )

        self.flush()
        return lsn

    def _append_log(
        self,
        txn_id: int,
        prev_lsn: int,
        log_type: LogType,
        page_id: int,
        slot_id: int,
        before_image: bytes,
        after_image: bytes,
    ) -> int:
        """Append a log record to the WAL."""
        with self._lock:
            lsn = self._current_lsn
            self._current_lsn += 1

            record = LogRecord(
                lsn=lsn,
                txn_id=txn_id,
                prev_lsn=prev_lsn,
                log_type=log_type,
                page_id=page_id,
                slot_id=slot_id,
                before_image=before_image,
                after_image=after_image,
            )

            data = record.to_bytes()
            self._file.seek(0, 2)  # Seek to end
            self._file.write(data)

            return lsn

    # ─────────────────────────────────────────────────────────────────────────
    # Flush and Sync
    # ─────────────────────────────────────────────────────────────────────────

    def flush(self) -> None:
        """Force flush WAL to disk."""
        with self._lock:
            self._file.flush()
            os.fsync(self._file.fileno())
            self._flushed_lsn = self._current_lsn - 1
            self._pending_commits.clear()
            self._last_flush_time = time.time()

    def _maybe_group_flush(self) -> None:
        """Flush if group commit interval has elapsed."""
        now = time.time()
        elapsed_ms = (now - self._last_flush_time) * 1000

        if elapsed_ms >= self._group_commit_interval_ms or len(self._pending_commits) > 0:
            self.flush()

    # ─────────────────────────────────────────────────────────────────────────
    # Reading and Iteration
    # ─────────────────────────────────────────────────────────────────────────

    def iter_records(self, start_lsn: int = 1) -> Iterator[LogRecord]:
        """
        Iterate over log records from a starting LSN.

        Args:
            start_lsn: Starting LSN (inclusive)

        Yields:
            LogRecord instances
        """
        with self._lock:
            self._file.seek(8)  # Skip header

            while True:
                pos = self._file.tell()
                header_data = self._file.read(RECORD_HEADER_SIZE)

                if len(header_data) < RECORD_HEADER_SIZE:
                    break

                lsn, _, _, _, total_len = struct.unpack("<QQQII", header_data)

                # Read full record
                self._file.seek(pos)
                record_data = self._file.read(total_len)

                if len(record_data) < total_len:
                    break

                record, _ = LogRecord.from_bytes(record_data)

                if record.lsn >= start_lsn:
                    yield record

    # ─────────────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def current_lsn(self) -> int:
        """Get the next LSN to be assigned."""
        return self._current_lsn

    @property
    def flushed_lsn(self) -> int:
        """Get the highest LSN that has been flushed to disk."""
        return self._flushed_lsn

    def close(self) -> None:
        """Close the WAL manager."""
        with self._lock:
            self.flush()
            self._file.close()

    def __enter__(self) -> "WALManager":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        """String representation."""
        return f"WALManager(path={self._path}, lsn={self._current_lsn})"
