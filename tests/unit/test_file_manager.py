"""
Unit tests for mmap-based File Manager.

Tests cover:
- File creation and opening
- Page read/write operations
- Page allocation/deallocation
- mmap functionality
- Error handling
"""

import os
import pytest
import tempfile
from pathlib import Path

from clade.storage.file_manager import (
    FileManager,
    FILE_HEADER_SIZE,
    MAGIC_NUMBER,
    VERSION,
)
from clade.storage.page import PAGE_SIZE
from clade.utils.errors import DiskIOError, PageNotFoundError


@pytest.fixture
def temp_db_path():
    """Create a temporary file path for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


@pytest.fixture
def file_manager(temp_db_path):
    """Create a file manager for testing."""
    fm = FileManager(temp_db_path)
    yield fm
    fm.close()


class TestFileManagerCreation:
    """Tests for file manager creation."""

    def test_create_new_file(self, temp_db_path):
        """Should create a new database file."""
        fm = FileManager(temp_db_path)

        assert temp_db_path.exists()
        assert fm.page_count == 0
        assert fm.path == temp_db_path

        fm.close()

    def test_create_with_nested_directory(self, temp_db_path):
        """Should create parent directories if needed."""
        nested_path = temp_db_path.parent / "nested" / "dir" / "test.db"

        fm = FileManager(nested_path)

        assert nested_path.exists()
        fm.close()

    def test_file_has_header(self, temp_db_path):
        """New file should have a header page."""
        fm = FileManager(temp_db_path)
        fm.close()

        # Check file size includes header
        assert temp_db_path.stat().st_size == FILE_HEADER_SIZE

    def test_open_existing_file(self, temp_db_path):
        """Should open an existing database file."""
        # Create file first
        fm1 = FileManager(temp_db_path)
        fm1.allocate_page()
        fm1.allocate_page()
        fm1.close()

        # Reopen
        fm2 = FileManager(temp_db_path)
        assert fm2.page_count == 2
        fm2.close()

    def test_reject_nonexistent_file_without_create(self, temp_db_path):
        """Should raise error if file doesn't exist and create=False."""
        with pytest.raises(DiskIOError, match="does not exist"):
            FileManager(temp_db_path, create=False)


class TestFileManagerReadWrite:
    """Tests for page read/write operations."""

    def test_write_and_read_page(self, file_manager):
        """Should write and read back page data."""
        page_id = file_manager.allocate_page()
        test_data = b"X" * PAGE_SIZE

        file_manager.write_page(page_id, test_data)
        read_data = file_manager.read_page(page_id)

        assert read_data == test_data

    def test_read_unallocated_page_fails(self, file_manager):
        """Should raise error when reading unallocated page."""
        with pytest.raises(PageNotFoundError):
            file_manager.read_page(0)

    def test_read_negative_page_fails(self, file_manager):
        """Should raise error for negative page ID."""
        with pytest.raises(PageNotFoundError):
            file_manager.read_page(-1)

    def test_write_unallocated_page_fails(self, file_manager):
        """Should raise error when writing to unallocated page."""
        with pytest.raises(PageNotFoundError):
            file_manager.write_page(0, b"\x00" * PAGE_SIZE)

    def test_write_wrong_size_fails(self, file_manager):
        """Should raise error when data is not PAGE_SIZE."""
        page_id = file_manager.allocate_page()

        with pytest.raises(ValueError, match=f"{PAGE_SIZE} bytes"):
            file_manager.write_page(page_id, b"Too short")

    def test_multiple_pages(self, file_manager):
        """Should handle multiple pages independently."""
        page_ids = [file_manager.allocate_page() for _ in range(5)]
        test_data = [f"Page{i}".encode().ljust(PAGE_SIZE, b"\x00") for i in range(5)]

        # Write all
        for pid, data in zip(page_ids, test_data):
            file_manager.write_page(pid, data)

        # Read all and verify
        for pid, expected in zip(page_ids, test_data):
            assert file_manager.read_page(pid) == expected

    def test_overwrite_page(self, file_manager):
        """Should allow overwriting existing page."""
        page_id = file_manager.allocate_page()

        data1 = b"A" * PAGE_SIZE
        data2 = b"B" * PAGE_SIZE

        file_manager.write_page(page_id, data1)
        assert file_manager.read_page(page_id) == data1

        file_manager.write_page(page_id, data2)
        assert file_manager.read_page(page_id) == data2


class TestFileManagerAllocation:
    """Tests for page allocation."""

    def test_allocate_page_returns_sequential_ids(self, file_manager):
        """Allocated pages should have sequential IDs."""
        ids = [file_manager.allocate_page() for _ in range(5)]

        assert ids == [0, 1, 2, 3, 4]

    def test_allocate_increases_page_count(self, file_manager):
        """Allocation should increase page count."""
        assert file_manager.page_count == 0

        file_manager.allocate_page()
        assert file_manager.page_count == 1

        file_manager.allocate_page()
        assert file_manager.page_count == 2

    def test_allocate_grows_file(self, temp_db_path):
        """Allocation should grow the file."""
        fm = FileManager(temp_db_path)
        initial_size = temp_db_path.stat().st_size

        fm.allocate_page()

        assert temp_db_path.stat().st_size == initial_size + PAGE_SIZE
        fm.close()

    def test_new_page_is_zeroed(self, file_manager):
        """Newly allocated page should contain zeros."""
        page_id = file_manager.allocate_page()

        data = file_manager.read_page(page_id)

        assert data == b"\x00" * PAGE_SIZE


class TestFileManagerDeallocation:
    """Tests for page deallocation."""

    def test_deallocate_page(self, file_manager):
        """Should deallocate a page."""
        page_id = file_manager.allocate_page()
        file_manager.write_page(page_id, b"X" * PAGE_SIZE)

        file_manager.deallocate_page(page_id)

        # Page is zeroed but still readable (not removed from file)
        data = file_manager.read_page(page_id)
        assert data == b"\x00" * PAGE_SIZE

    def test_deallocate_invalid_page_fails(self, file_manager):
        """Should raise error for invalid page ID."""
        with pytest.raises(PageNotFoundError):
            file_manager.deallocate_page(0)


class TestFileManagerSync:
    """Tests for sync operations."""

    def test_sync_completes(self, file_manager):
        """Sync should complete without error."""
        page_id = file_manager.allocate_page()
        file_manager.write_page(page_id, b"Data".ljust(PAGE_SIZE, b"\x00"))

        # Should not raise
        file_manager.sync()


class TestFileManagerPersistence:
    """Tests for data persistence across open/close cycles."""

    def test_data_persists_after_close(self, temp_db_path):
        """Data should persist after closing and reopening."""
        test_data = b"Persistent data".ljust(PAGE_SIZE, b"\x00")

        # Write data
        fm1 = FileManager(temp_db_path)
        page_id = fm1.allocate_page()
        fm1.write_page(page_id, test_data)
        fm1.close()

        # Reopen and verify
        fm2 = FileManager(temp_db_path)
        assert fm2.read_page(page_id) == test_data
        fm2.close()

    def test_page_count_persists(self, temp_db_path):
        """Page count should persist after close."""
        fm1 = FileManager(temp_db_path)
        for _ in range(10):
            fm1.allocate_page()
        fm1.close()

        fm2 = FileManager(temp_db_path)
        assert fm2.page_count == 10
        fm2.close()


class TestFileManagerMmap:
    """Tests for mmap functionality."""

    def test_mmap_enabled_by_default(self, temp_db_path):
        """mmap should be enabled by default."""
        fm = FileManager(temp_db_path, use_mmap=True)
        fm.allocate_page()  # Need at least one page for mmap

        # mmap may or may not be active depending on platform
        # Just ensure it doesn't crash
        fm.close()

    def test_mmap_disabled(self, temp_db_path):
        """Should work with mmap disabled."""
        fm = FileManager(temp_db_path, use_mmap=False)
        page_id = fm.allocate_page()

        test_data = b"No mmap".ljust(PAGE_SIZE, b"\x00")
        fm.write_page(page_id, test_data)

        assert fm.read_page(page_id) == test_data
        fm.close()

    def test_mmap_refresh_after_grow(self, temp_db_path):
        """mmap should refresh after file grows."""
        fm = FileManager(temp_db_path, use_mmap=True)

        # Allocate multiple pages to trigger mmap refresh
        for i in range(5):
            page_id = fm.allocate_page()
            data = f"Page{i}".encode().ljust(PAGE_SIZE, b"\x00")
            fm.write_page(page_id, data)

        # Should still read correctly
        for i in range(5):
            data = fm.read_page(i)
            assert data.startswith(f"Page{i}".encode())

        fm.close()


class TestFileManagerContextManager:
    """Tests for context manager usage."""

    def test_context_manager_closes(self, temp_db_path):
        """Context manager should close file."""
        with FileManager(temp_db_path) as fm:
            fm.allocate_page()

        # File should be closed, can open again
        with FileManager(temp_db_path) as fm2:
            assert fm2.page_count == 1


class TestFileManagerProperties:
    """Tests for file manager properties."""

    def test_file_size_property(self, file_manager):
        """file_size should return correct size."""
        assert file_manager.file_size == FILE_HEADER_SIZE

        file_manager.allocate_page()
        assert file_manager.file_size == FILE_HEADER_SIZE + PAGE_SIZE

        file_manager.allocate_page()
        assert file_manager.file_size == FILE_HEADER_SIZE + 2 * PAGE_SIZE

    def test_repr(self, file_manager):
        """__repr__ should return useful info."""
        repr_str = repr(file_manager)

        assert "FileManager" in repr_str
        assert "pages=" in repr_str


class TestFileManagerEdgeCases:
    """Tests for edge cases."""

    def test_binary_data_all_bytes(self, file_manager):
        """Should handle all byte values."""
        page_id = file_manager.allocate_page()

        # Create data with all byte values repeated
        data = (bytes(range(256)) * (PAGE_SIZE // 256 + 1))[:PAGE_SIZE]

        file_manager.write_page(page_id, data)
        assert file_manager.read_page(page_id) == data

    def test_many_pages(self, file_manager):
        """Should handle many pages."""
        num_pages = 100

        for i in range(num_pages):
            file_manager.allocate_page()

        assert file_manager.page_count == num_pages

        # Write and read back each page
        for i in range(num_pages):
            data = f"Page{i:03d}".encode().ljust(PAGE_SIZE, b"\x00")
            file_manager.write_page(i, data)

        for i in range(num_pages):
            data = file_manager.read_page(i)
            assert data.startswith(f"Page{i:03d}".encode())
