"""
Slotted page implementation for variable-length records.

Page Layout (8KB default):
┌──────────────────────────────┐
│ Header (64 bytes)            │
│   page_id, lsn, type,        │
│   free_offset, slot_count,   │
│   checksum                   │
├──────────────────────────────┤
│ Slot Directory (grows ↓)     │
│   [offset, length] × n       │
├──────────────────────────────┤
│ Free Space                   │
├──────────────────────────────┤
│ Records (grows ↑)            │
└──────────────────────────────┘

Slots grow downward from header, records grow upward from page end.
This allows efficient space utilization for variable-length records.
"""

import struct
import zlib
from dataclasses import dataclass
from typing import Optional, Iterator

from clade.storage.interfaces import IPage, PageType
from clade.utils.errors import PageFullError, PageCorruptionError


# Constants
PAGE_SIZE = 8192  # 8KB
HEADER_SIZE = 64
SLOT_SIZE = 8  # (offset: uint32, length: uint32)


@dataclass(frozen=True)
class PageHeader:
    """
    Page header structure (64 bytes total).

    Layout:
    - page_id: 8 bytes (uint64)
    - lsn: 8 bytes (uint64) - Log Sequence Number for WAL
    - page_type: 4 bytes (uint32)
    - free_offset: 4 bytes (uint32) - End of records area
    - slot_count: 4 bytes (uint32)
    - checksum: 4 bytes (uint32) - CRC32 of page data
    - reserved: 32 bytes (future use)
    """

    page_id: int
    lsn: int
    page_type: PageType
    free_offset: int
    slot_count: int
    checksum: int

    # struct format: little-endian, 2 uint64, 4 uint32, 32 bytes padding
    _FORMAT = "<QQIIII32x"

    def to_bytes(self) -> bytes:
        """Serialize header to bytes."""
        return struct.pack(
            self._FORMAT,
            self.page_id,
            self.lsn,
            self.page_type.value,
            self.free_offset,
            self.slot_count,
            self.checksum,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "PageHeader":
        """Deserialize header from bytes."""
        values = struct.unpack(cls._FORMAT, data[:HEADER_SIZE])
        return cls(
            page_id=values[0],
            lsn=values[1],
            page_type=PageType(values[2]),
            free_offset=values[3],
            slot_count=values[4],
            checksum=values[5],
        )


class Page(IPage):
    """
    Slotted page for variable-length record storage.

    Uses __slots__ for 60% memory reduction on frequently-allocated objects.
    Uses bytearray for efficient in-place modifications.

    Thread Safety: Not thread-safe. Use external locking (buffer manager).
    """

    __slots__ = ("_page_id", "_page_type", "_lsn", "_data", "_dirty")

    def __init__(self, page_id: int, page_type: PageType = PageType.DATA) -> None:
        """
        Initialize a new empty page.

        Args:
            page_id: Unique page identifier
            page_type: Type of page (DATA, INDEX, etc.)
        """
        self._page_id = page_id
        self._page_type = page_type
        self._lsn = 0
        self._data = bytearray(PAGE_SIZE)
        self._dirty = False

        # Initialize with header, records start at end of page
        self._write_header(free_offset=PAGE_SIZE, slot_count=0)

    # ─────────────────────────────────────────────────────────────────────────
    # Public Interface (IPage)
    # ─────────────────────────────────────────────────────────────────────────

    def insert_record(self, record: bytes) -> int:
        """
        Insert a record into the page.

        Args:
            record: Record data as bytes

        Returns:
            Slot ID of the inserted record

        Raises:
            PageFullError: If insufficient space for record
        """
        record_len = len(record)
        required_space = record_len + SLOT_SIZE

        if self.get_free_space() < required_space:
            raise PageFullError(
                self._page_id,
                required=required_space,
                available=self.get_free_space(),
            )

        header = self._read_header()
        slot_id = header.slot_count

        # Record grows upward from free_offset
        record_offset = header.free_offset - record_len
        self._data[record_offset:record_offset + record_len] = record

        # Slot grows downward from header
        slot_offset = HEADER_SIZE + slot_id * SLOT_SIZE
        struct.pack_into("<II", self._data, slot_offset, record_offset, record_len)

        # Update header
        self._write_header(
            free_offset=record_offset,
            slot_count=slot_id + 1,
        )

        self._dirty = True
        return slot_id

    def delete_record(self, slot_id: int) -> None:
        """
        Mark a record as deleted (tombstone).

        Space is not immediately reclaimed - use compaction for that.

        Args:
            slot_id: Slot to delete

        Raises:
            ValueError: If slot_id is invalid
        """
        self._validate_slot_id(slot_id)

        # Mark slot as deleted by setting length to 0
        slot_offset = HEADER_SIZE + slot_id * SLOT_SIZE
        struct.pack_into("<II", self._data, slot_offset, 0, 0)

        self._dirty = True

    def update_record(self, slot_id: int, record: bytes) -> None:
        """
        Update a record in-place if it fits.

        Args:
            slot_id: Slot to update
            record: New record data

        Raises:
            ValueError: If slot_id is invalid or record doesn't fit
        """
        self._validate_slot_id(slot_id)

        offset, length = self._get_slot(slot_id)
        if length == 0:
            raise ValueError(f"Cannot update deleted record at slot {slot_id}")

        if len(record) > length:
            raise ValueError(
                f"New record ({len(record)} bytes) exceeds slot capacity ({length} bytes). "
                "Delete and re-insert instead."
            )

        # In-place update (may leave unused space at end of slot)
        self._data[offset:offset + len(record)] = record

        # Update slot length if record is smaller
        slot_offset = HEADER_SIZE + slot_id * SLOT_SIZE
        struct.pack_into("<II", self._data, slot_offset, offset, len(record))

        self._dirty = True

    def get_record(self, slot_id: int) -> bytes:
        """
        Retrieve a record by slot ID.

        Args:
            slot_id: Slot to retrieve

        Returns:
            Record data as bytes

        Raises:
            ValueError: If slot_id is invalid or record is deleted
        """
        self._validate_slot_id(slot_id)

        offset, length = self._get_slot(slot_id)
        if length == 0:
            raise ValueError(f"Record at slot {slot_id} is deleted")

        return bytes(self._data[offset:offset + length])

    def get_free_space(self) -> int:
        """
        Get available free space in bytes.

        Returns:
            Number of bytes available for new records (including slot overhead)
        """
        header = self._read_header()
        slot_directory_end = HEADER_SIZE + header.slot_count * SLOT_SIZE
        return header.free_offset - slot_directory_end

    # ─────────────────────────────────────────────────────────────────────────
    # Serialization
    # ─────────────────────────────────────────────────────────────────────────

    def to_bytes(self) -> bytes:
        """
        Serialize page to bytes for disk storage.

        Updates checksum before serialization.

        Returns:
            Page data as bytes (PAGE_SIZE length)
        """
        # Update checksum before serialization
        checksum = self._compute_checksum()
        header = self._read_header()
        self._write_header(
            free_offset=header.free_offset,
            slot_count=header.slot_count,
            checksum=checksum,
        )
        return bytes(self._data)

    @classmethod
    def from_bytes(cls, data: bytes, verify_checksum: bool = True) -> "Page":
        """
        Deserialize page from bytes.

        Args:
            data: Page data (must be PAGE_SIZE length)
            verify_checksum: Whether to verify CRC32 checksum

        Returns:
            Page instance

        Raises:
            PageCorruptionError: If checksum validation fails
            ValueError: If data length is invalid
        """
        if len(data) != PAGE_SIZE:
            raise ValueError(f"Invalid page data length: {len(data)}, expected {PAGE_SIZE}")

        header = PageHeader.from_bytes(data)

        page = cls.__new__(cls)
        page._page_id = header.page_id
        page._page_type = header.page_type
        page._lsn = header.lsn
        page._data = bytearray(data)
        page._dirty = False

        if verify_checksum:
            actual_checksum = page._compute_checksum()
            if header.checksum != 0 and header.checksum != actual_checksum:
                raise PageCorruptionError(
                    page_id=header.page_id,
                    expected_checksum=header.checksum,
                    actual_checksum=actual_checksum,
                )

        return page

    # ─────────────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def page_id(self) -> int:
        """Get page ID."""
        return self._page_id

    @property
    def page_type(self) -> PageType:
        """Get page type."""
        return self._page_type

    @property
    def lsn(self) -> int:
        """Get Log Sequence Number."""
        return self._lsn

    @lsn.setter
    def lsn(self, value: int) -> None:
        """Set Log Sequence Number."""
        self._lsn = value
        self._dirty = True

    @property
    def is_dirty(self) -> bool:
        """Check if page has been modified."""
        return self._dirty

    @property
    def slot_count(self) -> int:
        """Get number of slots (including deleted)."""
        return self._read_header().slot_count

    def mark_clean(self) -> None:
        """Mark page as clean (called after flush to disk)."""
        self._dirty = False

    # ─────────────────────────────────────────────────────────────────────────
    # Iteration
    # ─────────────────────────────────────────────────────────────────────────

    def iter_records(self) -> Iterator[tuple[int, bytes]]:
        """
        Iterate over all non-deleted records.

        Yields:
            Tuples of (slot_id, record_data)
        """
        header = self._read_header()
        for slot_id in range(header.slot_count):
            offset, length = self._get_slot(slot_id)
            if length > 0:  # Skip deleted records
                yield slot_id, bytes(self._data[offset:offset + length])

    # ─────────────────────────────────────────────────────────────────────────
    # Private Methods
    # ─────────────────────────────────────────────────────────────────────────

    def _read_header(self) -> PageHeader:
        """Read header from page data."""
        return PageHeader.from_bytes(self._data)

    def _write_header(
        self,
        free_offset: int,
        slot_count: int,
        checksum: int = 0,
    ) -> None:
        """Write header to page data."""
        header = PageHeader(
            page_id=self._page_id,
            lsn=self._lsn,
            page_type=self._page_type,
            free_offset=free_offset,
            slot_count=slot_count,
            checksum=checksum,
        )
        self._data[:HEADER_SIZE] = header.to_bytes()

    def _get_slot(self, slot_id: int) -> tuple[int, int]:
        """Get (offset, length) for a slot."""
        slot_offset = HEADER_SIZE + slot_id * SLOT_SIZE
        return struct.unpack_from("<II", self._data, slot_offset)

    def _validate_slot_id(self, slot_id: int) -> None:
        """Validate slot ID is within bounds."""
        header = self._read_header()
        if slot_id < 0 or slot_id >= header.slot_count:
            raise ValueError(
                f"Invalid slot_id: {slot_id}. "
                f"Valid range: 0 to {header.slot_count - 1}"
            )

    def _compute_checksum(self) -> int:
        """Compute CRC32 checksum of page data (excluding checksum field)."""
        # Checksum covers everything except the checksum field itself (bytes 28-32)
        data_before = self._data[:28]
        data_after = self._data[32:]
        return zlib.crc32(data_before + data_after) & 0xFFFFFFFF

    def __repr__(self) -> str:
        """String representation for debugging."""
        header = self._read_header()
        return (
            f"Page(id={self._page_id}, type={self._page_type.name}, "
            f"slots={header.slot_count}, free={self.get_free_space()}, "
            f"dirty={self._dirty})"
        )
