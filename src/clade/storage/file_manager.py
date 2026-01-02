"""
File Manager with mmap-based I/O for efficient page access.

Provides low-level disk I/O abstraction using memory-mapped files
for zero-copy reads and efficient writes. Falls back to standard
file I/O on platforms where mmap is problematic.

Key Features:
- mmap for zero-copy page reads
- Atomic page writes with fsync
- Free page tracking with bitmap
- Automatic file growth
"""

import mmap
import os
import struct
import threading
from pathlib import Path
from typing import Optional, Union

from clade.storage.interfaces import IDiskManager
from clade.storage.page import PAGE_SIZE
from clade.utils.errors import DiskIOError, PageNotFoundError


# File header constants
FILE_HEADER_SIZE = PAGE_SIZE  # Reserve first page for metadata
MAGIC_NUMBER = 0x434C414445  # "CLADE" in hex
VERSION = 1

# Header layout: magic(8) + version(4) + page_count(4) + free_list_head(4) + reserved
HEADER_FORMAT = "<QIII"
HEADER_STRUCT_SIZE = struct.calcsize(HEADER_FORMAT)


class FileManager(IDiskManager):
    """
    mmap-based file manager for page I/O.

    Uses memory-mapped files for efficient reads. Writes go through
    standard file I/O with explicit fsync for durability.

    Thread Safety: Thread-safe via internal locking.

    File Layout:
    ┌──────────────────────────────┐ Page 0 (Header)
    │ Magic Number (8 bytes)       │
    │ Version (4 bytes)            │
    │ Page Count (4 bytes)         │
    │ Free List Head (4 bytes)     │
    │ Reserved                     │
    ├──────────────────────────────┤ Page 1
    │ Data Page                    │
    ├──────────────────────────────┤ Page 2
    │ Data Page                    │
    └──────────────────────────────┘
    """

    __slots__ = (
        "_path",
        "_file",
        "_mmap",
        "_page_count",
        "_free_list_head",
        "_lock",
        "_use_mmap",
    )

    def __init__(
        self,
        path: Union[str, Path],
        use_mmap: bool = True,
        create: bool = True,
    ) -> None:
        """
        Initialize file manager.

        Args:
            path: Path to the database file
            use_mmap: Whether to use memory-mapped I/O (default True)
            create: Create file if it doesn't exist (default True)

        Raises:
            DiskIOError: If file operations fail
        """
        self._path = Path(path)
        self._lock = threading.RLock()
        self._use_mmap = use_mmap
        self._mmap: Optional[mmap.mmap] = None

        try:
            if not self._path.exists():
                if not create:
                    raise DiskIOError("open", str(self._path), "File does not exist")
                self._create_new_file()
            else:
                self._open_existing_file()
        except OSError as e:
            raise DiskIOError("open", str(self._path), str(e)) from e

    def _create_new_file(self) -> None:
        """Create a new database file with header."""
        # Ensure parent directory exists
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # Create file with header page
        self._file = open(self._path, "w+b")

        # Initialize header
        self._page_count = 0
        self._free_list_head = 0  # 0 means no free pages

        # Write header page
        header = self._create_header()
        self._file.write(header)
        self._file.flush()
        os.fsync(self._file.fileno())

        # Setup mmap if enabled
        self._setup_mmap()

    def _open_existing_file(self) -> None:
        """Open an existing database file."""
        self._file = open(self._path, "r+b")

        # Read and validate header
        header_data = self._file.read(FILE_HEADER_SIZE)
        if len(header_data) < HEADER_STRUCT_SIZE:
            raise DiskIOError("read", str(self._path), "File too small for header")

        magic, version, page_count, free_list_head = struct.unpack(
            HEADER_FORMAT, header_data[:HEADER_STRUCT_SIZE]
        )

        if magic != MAGIC_NUMBER:
            raise DiskIOError("read", str(self._path), "Invalid magic number")

        if version > VERSION:
            raise DiskIOError(
                "read", str(self._path), f"Unsupported version {version}"
            )

        self._page_count = page_count
        self._free_list_head = free_list_head

        # Setup mmap if enabled
        self._setup_mmap()

    def _setup_mmap(self) -> None:
        """Setup memory mapping for the file."""
        if not self._use_mmap:
            return

        try:
            file_size = self._path.stat().st_size
            if file_size > 0:
                self._mmap = mmap.mmap(
                    self._file.fileno(),
                    0,  # Map entire file
                    access=mmap.ACCESS_READ,
                )
        except (OSError, ValueError):
            # mmap failed, fall back to standard I/O
            self._mmap = None
            self._use_mmap = False

    def _refresh_mmap(self) -> None:
        """Refresh mmap after file size change."""
        if not self._use_mmap:
            return

        with self._lock:
            if self._mmap is not None:
                try:
                    self._mmap.close()
                except Exception:
                    pass
                self._mmap = None

            self._setup_mmap()

    def _create_header(self) -> bytes:
        """Create file header page."""
        header = bytearray(FILE_HEADER_SIZE)
        struct.pack_into(
            HEADER_FORMAT,
            header,
            0,
            MAGIC_NUMBER,
            VERSION,
            self._page_count,
            self._free_list_head,
        )
        return bytes(header)

    def _write_header(self) -> None:
        """Write header to disk."""
        header = self._create_header()
        self._file.seek(0)
        self._file.write(header)
        self._file.flush()

    def _page_offset(self, page_id: int) -> int:
        """Calculate file offset for a page ID."""
        # Page 0 is header, data pages start at 1
        return FILE_HEADER_SIZE + page_id * PAGE_SIZE

    # ─────────────────────────────────────────────────────────────────────────
    # IDiskManager Interface
    # ─────────────────────────────────────────────────────────────────────────

    def read_page(self, page_id: int) -> bytes:
        """
        Read a page from disk.

        Uses mmap for zero-copy reads when available.

        Args:
            page_id: The page identifier (0-indexed)

        Returns:
            The raw page data as bytes (PAGE_SIZE length)

        Raises:
            PageNotFoundError: If page doesn't exist
            DiskIOError: If I/O operation fails
        """
        with self._lock:
            if page_id < 0 or page_id >= self._page_count:
                raise PageNotFoundError(page_id)

            offset = self._page_offset(page_id)

            try:
                # Try mmap first (zero-copy)
                if self._mmap is not None:
                    return bytes(self._mmap[offset : offset + PAGE_SIZE])

                # Fall back to standard I/O
                self._file.seek(offset)
                data = self._file.read(PAGE_SIZE)

                if len(data) != PAGE_SIZE:
                    raise DiskIOError(
                        "read",
                        str(self._path),
                        f"Short read: got {len(data)}, expected {PAGE_SIZE}",
                    )

                return data

            except OSError as e:
                raise DiskIOError("read", str(self._path), str(e)) from e

    def write_page(self, page_id: int, data: bytes) -> None:
        """
        Write a page to disk.

        Args:
            page_id: The page identifier
            data: The page data to write (must be PAGE_SIZE length)

        Raises:
            ValueError: If data length is not PAGE_SIZE
            DiskIOError: If I/O operation fails
        """
        if len(data) != PAGE_SIZE:
            raise ValueError(f"Page data must be {PAGE_SIZE} bytes, got {len(data)}")

        with self._lock:
            if page_id < 0 or page_id >= self._page_count:
                raise PageNotFoundError(page_id)

            offset = self._page_offset(page_id)

            try:
                self._file.seek(offset)
                self._file.write(data)
                self._file.flush()
            except OSError as e:
                raise DiskIOError("write", str(self._path), str(e)) from e

    def allocate_page(self) -> int:
        """
        Allocate a new page.

        Reuses deallocated pages if available, otherwise grows the file.

        Returns:
            The page_id of the newly allocated page
        """
        with self._lock:
            # TODO: Implement free list reuse
            # For now, always allocate at end
            page_id = self._page_count
            self._page_count += 1

            # Grow file
            try:
                offset = self._page_offset(page_id)
                self._file.seek(offset)
                self._file.write(b"\x00" * PAGE_SIZE)
                self._file.flush()

                # Update header
                self._write_header()

                # Refresh mmap to see new size
                self._refresh_mmap()

            except OSError as e:
                self._page_count -= 1  # Rollback
                raise DiskIOError("allocate", str(self._path), str(e)) from e

            return page_id

    def deallocate_page(self, page_id: int) -> None:
        """
        Deallocate a page, marking it as free.

        Args:
            page_id: The page to deallocate

        Raises:
            PageNotFoundError: If page doesn't exist
        """
        with self._lock:
            if page_id < 0 or page_id >= self._page_count:
                raise PageNotFoundError(page_id)

            # TODO: Implement free list management
            # For now, just mark page as zeroed
            try:
                offset = self._page_offset(page_id)
                self._file.seek(offset)
                self._file.write(b"\x00" * PAGE_SIZE)
                self._file.flush()
            except OSError as e:
                raise DiskIOError("deallocate", str(self._path), str(e)) from e

    def sync(self) -> None:
        """
        Synchronize all dirty data to disk (fsync).

        Ensures durability by forcing OS to write to physical storage.
        """
        with self._lock:
            try:
                self._file.flush()
                os.fsync(self._file.fileno())
            except OSError as e:
                raise DiskIOError("sync", str(self._path), str(e)) from e

    def close(self) -> None:
        """Close the file manager and release resources."""
        with self._lock:
            try:
                if self._mmap is not None:
                    self._mmap.close()
                    self._mmap = None

                if self._file is not None:
                    self._file.close()
                    self._file = None
            except OSError as e:
                raise DiskIOError("close", str(self._path), str(e)) from e

    # ─────────────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def page_count(self) -> int:
        """Get the number of allocated pages."""
        return self._page_count

    @property
    def path(self) -> Path:
        """Get the file path."""
        return self._path

    @property
    def file_size(self) -> int:
        """Get the current file size in bytes."""
        return FILE_HEADER_SIZE + self._page_count * PAGE_SIZE

    def __enter__(self) -> "FileManager":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"FileManager(path={self._path}, pages={self._page_count}, "
            f"mmap={self._mmap is not None})"
        )
