"""
Abstract interfaces for storage engine components.

This module defines the core contracts for all storage-related components.
Following the Dependency Inversion Principle, high-level modules depend on
these abstractions, not on concrete implementations.

This enables:
- Easy swapping of implementations (e.g., mmap vs direct I/O)
- Clear contracts for testing (mock implementations)
- Separation of concerns between layers
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Optional, Any


# Enumerations


class PageType(Enum):
    """Type of page in the storage system."""

    DATA = 1  # Regular data page (heap file)
    INDEX = 2  # B+Tree index page
    FREESPACE = 3  # Free space map page
    METADATA = 4  # System metadata page


# Value Objects


@dataclass(frozen=True)
class RecordID:
    """
    Unique identifier for a record in the database.

    A RecordID consists of:
    - page_id: The page containing the record
    - slot_id: The slot within the page

    Using __slots__ to reduce memory overhead (60% reduction).
    """

    __slots__ = ("page_id", "slot_id")

    page_id: int
    slot_id: int

    def __post_init__(self) -> None:
        """Validate RecordID components."""
        if self.page_id < 0:
            raise ValueError(f"Invalid page_id: {self.page_id}. Must be non-negative.")
        if self.slot_id < 0:
            raise ValueError(f"Invalid slot_id: {self.slot_id}. Must be non-negative.")

    def __str__(self) -> str:
        return f"RID({self.page_id}:{self.slot_id})"


# Page Interface


class IPage(ABC):
    """
    Interface for page operations.

    A page is the fundamental unit of storage, representing a fixed-size
    block of data (typically 8KB). Pages use a slotted layout to store
    variable-length records efficiently.
    """

    @abstractmethod
    def insert_record(self, record: bytes) -> int:
        """
        Insert a record into the page.

        Args:
            record: The record data as bytes

        Returns:
            The slot_id of the inserted record

        Raises:
            PageFullError: If insufficient space for record
        """
        pass

    @abstractmethod
    def delete_record(self, slot_id: int) -> None:
        """
        Mark a record as deleted (tombstone).

        The actual space is reclaimed during page compaction.

        Args:
            slot_id: The slot containing the record to delete

        Raises:
            ValueError: If slot_id is invalid
        """
        pass

    @abstractmethod
    def update_record(self, slot_id: int, record: bytes) -> None:
        """
        Update a record in-place if possible.

        If the new record doesn't fit in the old slot, raises an error.
        The caller should then delete and insert as a new record.

        Args:
            slot_id: The slot to update
            record: The new record data

        Raises:
            ValueError: If slot_id is invalid or update not possible
        """
        pass

    @abstractmethod
    def get_record(self, slot_id: int) -> bytes:
        """
        Retrieve a record by its slot_id.

        Args:
            slot_id: The slot containing the record

        Returns:
            The record data as bytes

        Raises:
            ValueError: If slot_id is invalid or record is deleted
        """
        pass

    @abstractmethod
    def get_free_space(self) -> int:
        """
        Get the available free space in the page.

        Returns:
            Number of bytes available for new records
        """
        pass

    @abstractmethod
    def to_bytes(self) -> bytes:
        """
        Serialize the page to bytes for disk storage.

        Returns:
            The page data as bytes
        """
        pass

    @classmethod
    @abstractmethod
    def from_bytes(cls, data: bytes) -> "IPage":
        """
        Deserialize a page from bytes.

        Args:
            data: The page data as bytes

        Returns:
            A new Page instance

        Raises:
            PageCorruptionError: If checksum validation fails
        """
        pass


# Disk Manager Interface


class IDiskManager(ABC):
    """
    Low-level disk I/O abstraction.

    Handles reading and writing raw pages to/from disk.
    Implementations may use mmap, direct I/O, or standard file I/O.
    """

    @abstractmethod
    def read_page(self, page_id: int) -> bytes:
        """
        Read a page from disk.

        Args:
            page_id: The page identifier

        Returns:
            The raw page data as bytes

        Raises:
            PageNotFoundError: If page doesn't exist
            DiskIOError: If I/O operation fails
        """
        pass

    @abstractmethod
    def write_page(self, page_id: int, data: bytes) -> None:
        """
        Write a page to disk.

        Args:
            page_id: The page identifier
            data: The page data to write

        Raises:
            DiskIOError: If I/O operation fails
        """
        pass

    @abstractmethod
    def allocate_page(self) -> int:
        """
        Allocate a new page.

        Returns:
            The page_id of the newly allocated page
        """
        pass

    @abstractmethod
    def deallocate_page(self, page_id: int) -> None:
        """
        Deallocate a page, marking it as free.

        Args:
            page_id: The page to deallocate
        """
        pass

    @abstractmethod
    def sync(self) -> None:
        """
        Synchronize all dirty data to disk (fsync).

        Ensures durability by forcing OS to write to physical storage.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the disk manager and release resources."""
        pass


# Buffer Pool Interface


class IBufferPool(ABC):
    """
    Buffer pool (page cache) interface.

    Manages in-memory cache of pages with eviction policies.
    Coordinates with WAL for proper ordering of writes.
    """

    @abstractmethod
    def fetch_page(self, page_id: int) -> IPage:
        """
        Fetch a page from the buffer pool or disk.

        The page is automatically pinned. Caller must unpin when done.

        Args:
            page_id: The page to fetch

        Returns:
            The requested page

        Raises:
            PageNotFoundError: If page doesn't exist
            BufferPoolFullError: If can't evict any pages
        """
        pass

    @abstractmethod
    def unpin_page(self, page_id: int, is_dirty: bool) -> None:
        """
        Unpin a page, marking it as dirty if modified.

        Args:
            page_id: The page to unpin
            is_dirty: True if page was modified
        """
        pass

    @abstractmethod
    def flush_page(self, page_id: int) -> None:
        """
        Force write a specific page to disk.

        Args:
            page_id: The page to flush

        Raises:
            DiskIOError: If write fails
        """
        pass

    @abstractmethod
    def flush_all_dirty(self) -> None:
        """
        Flush all dirty pages to disk.

        Used during checkpoints and shutdown.
        """
        pass

    @abstractmethod
    def new_page(self) -> tuple[int, IPage]:
        """
        Allocate and return a new page.

        The page is automatically pinned. Caller must unpin when done.

        Returns:
            Tuple of (page_id, page)

        Raises:
            BufferPoolFullError: If can't evict any pages
        """
        pass


# Storage Engine Interface


class IStorageEngine(ABC):
    """
    High-level storage interface for record-level operations.

    Abstracts away page-level details, providing CRUD operations
    on logical records identified by RecordID.
    """

    @abstractmethod
    def insert(self, record: bytes) -> RecordID:
        """
        Insert a record into storage.

        Args:
            record: The record data as bytes

        Returns:
            The RecordID of the inserted record
        """
        pass

    @abstractmethod
    def delete(self, rid: RecordID) -> None:
        """
        Delete a record.

        Args:
            rid: The record identifier

        Raises:
            ValueError: If record doesn't exist or already deleted
        """
        pass

    @abstractmethod
    def update(self, rid: RecordID, record: bytes) -> None:
        """
        Update a record.

        May create a new version (MVCC) or update in-place.

        Args:
            rid: The record identifier
            record: The new record data

        Raises:
            ValueError: If record doesn't exist
        """
        pass

    @abstractmethod
    def get(self, rid: RecordID) -> bytes:
        """
        Retrieve a record by its identifier.

        Args:
            rid: The record identifier

        Returns:
            The record data as bytes

        Raises:
            ValueError: If record doesn't exist or deleted
        """
        pass

    @abstractmethod
    def scan(self) -> Iterator[tuple[RecordID, bytes]]:
        """
        Full table scan, yielding all records.

        Yields:
            Tuples of (RecordID, record_data)
        """
        pass


# Index Interface


class IIndex(ABC):
    """
    Interface for index structures (B+Tree, Hash, Bitmap, etc.).

    Provides key-based access to records.
    """

    @abstractmethod
    def insert(self, key: Any, rid: RecordID) -> None:
        """
        Insert a key-value pair into the index.

        Args:
            key: The key to index
            rid: The record identifier

        Raises:
            DuplicateKeyError: If key already exists (for unique indexes)
        """
        pass

    @abstractmethod
    def delete(self, key: Any) -> None:
        """
        Delete a key from the index.

        Args:
            key: The key to delete

        Raises:
            KeyNotFoundError: If key doesn't exist
        """
        pass

    @abstractmethod
    def search(self, key: Any) -> Optional[RecordID]:
        """
        Search for a key in the index.

        Args:
            key: The key to search for

        Returns:
            The RecordID if found, None otherwise
        """
        pass

    @abstractmethod
    def range_scan(self, start_key: Any, end_key: Any) -> Iterator[tuple[Any, RecordID]]:
        """
        Scan a range of keys.

        Args:
            start_key: The inclusive start of the range
            end_key: The inclusive end of the range

        Yields:
            Tuples of (key, RecordID) in sorted order
        """
        pass


# Transaction Interface


class ITransaction(ABC):
    """
    Interface for transaction management.

    Provides ACID guarantees through MVCC and WAL.
    """

    @abstractmethod
    def begin(self) -> int:
        """
        Begin a new transaction.

        Returns:
            The transaction ID
        """
        pass

    @abstractmethod
    def commit(self, txn_id: int) -> None:
        """
        Commit a transaction.

        Args:
            txn_id: The transaction to commit

        Raises:
            TransactionAbortedError: If transaction cannot commit
        """
        pass

    @abstractmethod
    def abort(self, txn_id: int) -> None:
        """
        Abort a transaction, rolling back all changes.

        Args:
            txn_id: The transaction to abort
        """
        pass

    @abstractmethod
    def is_active(self, txn_id: int) -> bool:
        """
        Check if a transaction is still active.

        Args:
            txn_id: The transaction to check

        Returns:
            True if transaction is active
        """
        pass
