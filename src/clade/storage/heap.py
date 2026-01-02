"""
Heap File implementation for row storage.

A heap file is an unordered collection of records stored across
multiple pages. Uses a free space map to efficiently find pages
with available space for new records.

Key Features:
- Free space tracking per page (avoids full scans)
- Variable-length record support
- RecordID-based access (page_id, slot_id)
"""

from typing import Iterator, Optional
from dataclasses import dataclass

from clade.storage.interfaces import IStorageEngine, RecordID
from clade.storage.buffer_manager import BufferPoolManager
from clade.storage.page import Page, HEADER_SIZE, SLOT_SIZE, PAGE_SIZE
from clade.utils.errors import PageFullError


# Minimum free space to consider a page for insertion
MIN_FREE_SPACE = SLOT_SIZE + 16  # At least slot + 16 bytes for record


@dataclass
class FreeSpaceEntry:
    """Entry in the free space map."""

    __slots__ = ("page_id", "free_space")

    page_id: int
    free_space: int


class HeapFile(IStorageEngine):
    """
    Heap file storage engine.

    Manages an unordered collection of records across multiple pages.
    Uses a free space map for efficient insertion.

    Thread Safety: Relies on BufferPoolManager for thread safety.
    """

    __slots__ = ("_buffer_pool", "_free_space_map", "_page_count")

    def __init__(self, buffer_pool: BufferPoolManager) -> None:
        """
        Initialize heap file.

        Args:
            buffer_pool: Buffer pool manager for page access
        """
        self._buffer_pool = buffer_pool
        self._free_space_map: dict[int, int] = {}  # page_id -> free_space
        self._page_count = 0

    # ─────────────────────────────────────────────────────────────────────────
    # IStorageEngine Interface
    # ─────────────────────────────────────────────────────────────────────────

    def insert(self, record: bytes) -> RecordID:
        """
        Insert a record into the heap.

        Finds a page with sufficient free space or allocates a new page.

        Args:
            record: The record data as bytes

        Returns:
            The RecordID of the inserted record
        """
        required_space = len(record) + SLOT_SIZE

        # Find page with enough space
        page_id = self._find_page_with_space(required_space)

        if page_id is None:
            # Allocate new page
            page_id, page = self._buffer_pool.new_page()
            self._page_count += 1
            self._free_space_map[page_id] = page.get_free_space()
        else:
            page = self._buffer_pool.fetch_page(page_id)

        try:
            slot_id = page.insert_record(record)
            self._free_space_map[page_id] = page.get_free_space()
            self._buffer_pool.unpin_page(page_id, is_dirty=True)
            return RecordID(page_id=page_id, slot_id=slot_id)
        except PageFullError:
            self._buffer_pool.unpin_page(page_id, is_dirty=False)
            raise

    def delete(self, rid: RecordID) -> None:
        """
        Delete a record.

        Args:
            rid: The record identifier

        Raises:
            ValueError: If record doesn't exist or already deleted
        """
        page = self._buffer_pool.fetch_page(rid.page_id)

        try:
            page.delete_record(rid.slot_id)
            self._free_space_map[rid.page_id] = page.get_free_space()
            self._buffer_pool.unpin_page(rid.page_id, is_dirty=True)
        except ValueError:
            self._buffer_pool.unpin_page(rid.page_id, is_dirty=False)
            raise

    def update(self, rid: RecordID, record: bytes) -> None:
        """
        Update a record in-place if possible.

        If the new record doesn't fit, raises an error.
        Caller should delete and re-insert as a new record.

        Args:
            rid: The record identifier
            record: The new record data

        Raises:
            ValueError: If record doesn't exist or update not possible
        """
        page = self._buffer_pool.fetch_page(rid.page_id)

        try:
            page.update_record(rid.slot_id, record)
            self._free_space_map[rid.page_id] = page.get_free_space()
            self._buffer_pool.unpin_page(rid.page_id, is_dirty=True)
        except ValueError:
            self._buffer_pool.unpin_page(rid.page_id, is_dirty=False)
            raise

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
        page = self._buffer_pool.fetch_page(rid.page_id)

        try:
            record = page.get_record(rid.slot_id)
            self._buffer_pool.unpin_page(rid.page_id, is_dirty=False)
            return record
        except ValueError:
            self._buffer_pool.unpin_page(rid.page_id, is_dirty=False)
            raise

    def scan(self) -> Iterator[tuple[RecordID, bytes]]:
        """
        Full table scan, yielding all records.

        Yields:
            Tuples of (RecordID, record_data)
        """
        for page_id in range(self._page_count):
            try:
                page = self._buffer_pool.fetch_page(page_id)
            except Exception:
                continue

            try:
                for slot_id, record in page.iter_records():
                    yield RecordID(page_id=page_id, slot_id=slot_id), record
            finally:
                self._buffer_pool.unpin_page(page_id, is_dirty=False)

    # ─────────────────────────────────────────────────────────────────────────
    # Free Space Management
    # ─────────────────────────────────────────────────────────────────────────

    def _find_page_with_space(self, required: int) -> Optional[int]:
        """
        Find a page with enough free space.

        Uses the free space map to avoid scanning all pages.

        Args:
            required: Required space in bytes

        Returns:
            Page ID with sufficient space, or None if not found
        """
        for page_id, free_space in self._free_space_map.items():
            if free_space >= required:
                return page_id
        return None

    def _update_free_space(self, page_id: int, page: Page) -> None:
        """Update free space map entry for a page."""
        self._free_space_map[page_id] = page.get_free_space()

    # ─────────────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def page_count(self) -> int:
        """Get the number of pages in the heap."""
        return self._page_count

    @property
    def record_count(self) -> int:
        """Get approximate record count (includes deleted)."""
        count = 0
        for page_id in range(self._page_count):
            try:
                page = self._buffer_pool.fetch_page(page_id)
                count += page.slot_count
                self._buffer_pool.unpin_page(page_id, is_dirty=False)
            except Exception:
                pass
        return count

    def get_free_space_stats(self) -> dict:
        """Get free space statistics."""
        total_free = sum(self._free_space_map.values())
        total_capacity = self._page_count * (PAGE_SIZE - HEADER_SIZE)
        return {
            "page_count": self._page_count,
            "total_free_bytes": total_free,
            "total_capacity_bytes": total_capacity,
            "utilization": 1.0 - (total_free / total_capacity) if total_capacity > 0 else 0.0,
        }

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"HeapFile(pages={self._page_count})"
