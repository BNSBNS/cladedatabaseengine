"""
Unit tests for Heap File storage engine.

Tests cover:
- Record insertion, deletion, update, retrieval
- Full table scan
- Free space management
- Multi-page storage
"""

import pytest
import tempfile
from pathlib import Path

from clade.storage.heap import HeapFile
from clade.storage.buffer_manager import BufferPoolManager
from clade.storage.file_manager import FileManager
from clade.storage.interfaces import RecordID
from clade.storage.page import PAGE_SIZE, HEADER_SIZE, SLOT_SIZE


@pytest.fixture
def temp_db_path():
    """Create a temporary file path for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


@pytest.fixture
def buffer_pool(temp_db_path):
    """Create a buffer pool for testing."""
    fm = FileManager(temp_db_path)
    bp = BufferPoolManager(fm, pool_size=100)
    yield bp
    bp.flush_all_dirty()
    fm.close()


@pytest.fixture
def heap(buffer_pool):
    """Create a heap file for testing."""
    return HeapFile(buffer_pool)


class TestHeapFileBasicOperations:
    """Tests for basic heap file CRUD operations."""

    def test_insert_record(self, heap):
        """Should insert a record and return RecordID."""
        record = b"Hello, World!"

        rid = heap.insert(record)

        assert isinstance(rid, RecordID)
        assert rid.page_id == 0
        assert rid.slot_id == 0

    def test_insert_multiple_records(self, heap):
        """Should insert multiple records."""
        records = [f"Record {i}".encode() for i in range(10)]

        rids = [heap.insert(r) for r in records]

        assert len(rids) == 10
        # All on same page initially
        assert all(rid.page_id == 0 for rid in rids)
        assert [rid.slot_id for rid in rids] == list(range(10))

    def test_get_record(self, heap):
        """Should retrieve a record by RecordID."""
        record = b"Test data"
        rid = heap.insert(record)

        result = heap.get(rid)

        assert result == record

    def test_get_multiple_records(self, heap):
        """Should retrieve multiple records correctly."""
        records = [f"Record {i}".encode() for i in range(5)]
        rids = [heap.insert(r) for r in records]

        for rid, expected in zip(rids, records):
            assert heap.get(rid) == expected

    def test_delete_record(self, heap):
        """Should delete a record."""
        rid = heap.insert(b"To be deleted")

        heap.delete(rid)

        with pytest.raises(ValueError, match="deleted"):
            heap.get(rid)

    def test_delete_nonexistent_fails(self, heap):
        """Should fail when deleting nonexistent record."""
        heap.insert(b"Data")

        with pytest.raises(ValueError):
            heap.delete(RecordID(page_id=0, slot_id=999))

    def test_update_record(self, heap):
        """Should update a record in-place."""
        rid = heap.insert(b"Original data here")

        heap.update(rid, b"Updated")

        assert heap.get(rid) == b"Updated"

    def test_update_larger_fails(self, heap):
        """Should fail when update data is too large."""
        rid = heap.insert(b"Small")

        with pytest.raises(ValueError, match="exceeds"):
            heap.update(rid, b"This is much larger data")

    def test_update_deleted_fails(self, heap):
        """Should fail when updating deleted record."""
        rid = heap.insert(b"Data")
        heap.delete(rid)

        with pytest.raises(ValueError, match="deleted"):
            heap.update(rid, b"New data")


class TestHeapFileScan:
    """Tests for full table scan."""

    def test_scan_empty_heap(self, heap):
        """Scan on empty heap should yield nothing."""
        result = list(heap.scan())

        assert result == []

    def test_scan_all_records(self, heap):
        """Scan should return all records."""
        records = [f"Record {i}".encode() for i in range(5)]
        rids = [heap.insert(r) for r in records]

        result = list(heap.scan())

        assert len(result) == 5
        for (rid, data), expected in zip(result, records):
            assert data == expected

    def test_scan_skips_deleted(self, heap):
        """Scan should skip deleted records."""
        records = [f"Record {i}".encode() for i in range(5)]
        rids = [heap.insert(r) for r in records]

        # Delete middle record
        heap.delete(rids[2])

        result = list(heap.scan())

        assert len(result) == 4
        result_data = [r[1] for r in result]
        assert b"Record 2" not in result_data


class TestHeapFileMultiPage:
    """Tests for multi-page storage."""

    def test_allocate_new_page_when_full(self, heap):
        """Should allocate new page when current page is full."""
        # Fill first page
        max_record_size = PAGE_SIZE - HEADER_SIZE - SLOT_SIZE - 100
        large_record = b"X" * max_record_size
        rid1 = heap.insert(large_record)

        assert rid1.page_id == 0
        assert heap.page_count == 1

        # Second large record should go to new page
        rid2 = heap.insert(large_record)

        assert rid2.page_id == 1
        assert heap.page_count == 2

    def test_scan_across_pages(self, heap):
        """Scan should work across multiple pages."""
        # Insert records that span multiple pages
        max_record_size = (PAGE_SIZE - HEADER_SIZE) // 3
        records = [f"R{i}".encode().ljust(max_record_size, b"\x00") for i in range(10)]

        rids = [heap.insert(r) for r in records]

        # Should span multiple pages
        assert heap.page_count > 1

        # Scan should find all
        result = list(heap.scan())
        assert len(result) == 10


class TestHeapFileFreeSpaceManagement:
    """Tests for free space map."""

    def test_delete_does_not_reclaim_space(self, heap):
        """Deletion marks slot as empty but doesn't reclaim space.

        Space reclamation requires compaction (not implemented yet).
        This is a known design decision for the slotted page format.
        """
        # Insert records
        rids = [heap.insert(f"Record {i}".encode()) for i in range(5)]
        initial_stats = heap.get_free_space_stats()

        # Delete one
        heap.delete(rids[2])
        after_delete_stats = heap.get_free_space_stats()

        # Free space stays the same (no compaction)
        assert after_delete_stats["total_free_bytes"] == initial_stats["total_free_bytes"]

    def test_find_page_with_space(self, heap):
        """Free space map should find suitable pages."""
        # Create first page with some records
        for i in range(3):
            heap.insert(f"Record {i}".encode())

        first_page_free = heap._free_space_map.get(0, 0)

        # Should find page 0 for small record
        page_id = heap._find_page_with_space(100)
        assert page_id == 0

        # Fill first page
        large_record = b"X" * (first_page_free - SLOT_SIZE - 10)
        if first_page_free > SLOT_SIZE + 10:
            heap.insert(large_record)

    def test_utilization_stats(self, heap):
        """Should track utilization statistics."""
        stats = heap.get_free_space_stats()
        assert stats["page_count"] == 0

        heap.insert(b"Data")
        stats = heap.get_free_space_stats()

        assert stats["page_count"] == 1
        assert stats["utilization"] > 0


class TestHeapFileEdgeCases:
    """Tests for edge cases."""

    def test_binary_data(self, heap):
        """Should handle binary data with all byte values."""
        binary_record = bytes(range(256))
        rid = heap.insert(binary_record)

        assert heap.get(rid) == binary_record

    def test_large_record(self, heap):
        """Should handle large records."""
        max_size = PAGE_SIZE - HEADER_SIZE - SLOT_SIZE - 10
        large_record = b"X" * max_size

        rid = heap.insert(large_record)

        assert heap.get(rid) == large_record

    def test_many_small_records(self, heap):
        """Should handle many small records."""
        records = [f"R{i:04d}".encode() for i in range(500)]

        rids = [heap.insert(r) for r in records]

        assert len(rids) == 500
        for rid, expected in zip(rids, records):
            assert heap.get(rid) == expected


class TestHeapFileProperties:
    """Tests for heap file properties."""

    def test_page_count(self, heap):
        """page_count should track allocated pages."""
        assert heap.page_count == 0

        heap.insert(b"Data")
        assert heap.page_count == 1

    def test_repr(self, heap):
        """__repr__ should return useful info."""
        heap.insert(b"Data")
        repr_str = repr(heap)

        assert "HeapFile" in repr_str
        assert "pages=" in repr_str
