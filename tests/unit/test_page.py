"""
Unit tests for slotted page implementation.

Tests cover:
- Record insertion, deletion, update, retrieval
- Serialization and checksum verification
- Edge cases and error conditions
- Memory efficiency with __slots__
"""

import pytest
from clade.storage.page import Page, PageHeader, PAGE_SIZE, HEADER_SIZE, SLOT_SIZE
from clade.storage.interfaces import PageType
from clade.utils.errors import PageFullError, PageCorruptionError


class TestPageHeader:
    """Tests for PageHeader serialization."""

    def test_header_serialization_roundtrip(self):
        """Header should serialize and deserialize correctly."""
        header = PageHeader(
            page_id=42,
            lsn=1000,
            page_type=PageType.DATA,
            free_offset=8000,
            slot_count=5,
            checksum=0xDEADBEEF,
        )

        data = header.to_bytes()
        assert len(data) == HEADER_SIZE

        restored = PageHeader.from_bytes(data)
        assert restored.page_id == 42
        assert restored.lsn == 1000
        assert restored.page_type == PageType.DATA
        assert restored.free_offset == 8000
        assert restored.slot_count == 5
        assert restored.checksum == 0xDEADBEEF

    def test_header_different_page_types(self):
        """Header should handle all page types."""
        for page_type in PageType:
            header = PageHeader(
                page_id=1,
                lsn=0,
                page_type=page_type,
                free_offset=PAGE_SIZE,
                slot_count=0,
                checksum=0,
            )
            data = header.to_bytes()
            restored = PageHeader.from_bytes(data)
            assert restored.page_type == page_type


class TestPageBasicOperations:
    """Tests for basic page CRUD operations."""

    def test_page_creation(self):
        """New page should be properly initialized."""
        page = Page(page_id=1)

        assert page.page_id == 1
        assert page.page_type == PageType.DATA
        assert page.lsn == 0
        assert page.slot_count == 0
        assert not page.is_dirty
        assert page.get_free_space() == PAGE_SIZE - HEADER_SIZE

    def test_page_with_custom_type(self):
        """Page should support different page types."""
        page = Page(page_id=1, page_type=PageType.INDEX)
        assert page.page_type == PageType.INDEX

    def test_insert_single_record(self):
        """Should insert a record and return slot ID."""
        page = Page(page_id=1)
        record = b"Hello, World!"

        slot_id = page.insert_record(record)

        assert slot_id == 0
        assert page.slot_count == 1
        assert page.is_dirty
        assert page.get_record(slot_id) == record

    def test_insert_multiple_records(self):
        """Should insert multiple records with sequential slot IDs."""
        page = Page(page_id=1)
        records = [b"Record 1", b"Record 2", b"Record 3"]

        slot_ids = [page.insert_record(r) for r in records]

        assert slot_ids == [0, 1, 2]
        assert page.slot_count == 3
        for slot_id, expected in zip(slot_ids, records):
            assert page.get_record(slot_id) == expected

    def test_insert_reduces_free_space(self):
        """Inserting records should reduce available space."""
        page = Page(page_id=1)
        initial_space = page.get_free_space()

        record = b"Test record"
        page.insert_record(record)

        expected_reduction = len(record) + SLOT_SIZE
        assert page.get_free_space() == initial_space - expected_reduction

    def test_get_record_returns_copy(self):
        """Retrieved record should be a copy, not the original."""
        page = Page(page_id=1)
        original = b"Original data"
        slot_id = page.insert_record(original)

        retrieved = page.get_record(slot_id)
        assert retrieved == original
        assert retrieved is not original


class TestPageDeletion:
    """Tests for record deletion."""

    def test_delete_record(self):
        """Deleted record should not be retrievable."""
        page = Page(page_id=1)
        slot_id = page.insert_record(b"To be deleted")

        page.delete_record(slot_id)

        with pytest.raises(ValueError, match="deleted"):
            page.get_record(slot_id)

    def test_delete_marks_dirty(self):
        """Deletion should mark page as dirty."""
        page = Page(page_id=1)
        slot_id = page.insert_record(b"Data")
        page.mark_clean()

        page.delete_record(slot_id)

        assert page.is_dirty

    def test_delete_invalid_slot(self):
        """Deleting invalid slot should raise error."""
        page = Page(page_id=1)

        with pytest.raises(ValueError, match="Invalid slot_id"):
            page.delete_record(0)

    def test_delete_negative_slot(self):
        """Deleting negative slot should raise error."""
        page = Page(page_id=1)
        page.insert_record(b"Data")

        with pytest.raises(ValueError, match="Invalid slot_id"):
            page.delete_record(-1)


class TestPageUpdate:
    """Tests for record updates."""

    def test_update_record_same_size(self):
        """Should update record with same-size data."""
        page = Page(page_id=1)
        slot_id = page.insert_record(b"AAAA")

        page.update_record(slot_id, b"BBBB")

        assert page.get_record(slot_id) == b"BBBB"

    def test_update_record_smaller(self):
        """Should update record with smaller data."""
        page = Page(page_id=1)
        slot_id = page.insert_record(b"Long record data")

        page.update_record(slot_id, b"Short")

        assert page.get_record(slot_id) == b"Short"

    def test_update_record_larger_fails(self):
        """Should fail when new record exceeds slot capacity."""
        page = Page(page_id=1)
        slot_id = page.insert_record(b"Small")

        with pytest.raises(ValueError, match="exceeds slot capacity"):
            page.update_record(slot_id, b"Much larger record data")

    def test_update_deleted_record_fails(self):
        """Should fail when updating deleted record."""
        page = Page(page_id=1)
        slot_id = page.insert_record(b"Data")
        page.delete_record(slot_id)

        with pytest.raises(ValueError, match="deleted"):
            page.update_record(slot_id, b"New data")

    def test_update_marks_dirty(self):
        """Update should mark page as dirty."""
        page = Page(page_id=1)
        slot_id = page.insert_record(b"Data")
        page.mark_clean()

        page.update_record(slot_id, b"New")

        assert page.is_dirty


class TestPageFreeSpace:
    """Tests for free space management."""

    def test_page_full_error(self):
        """Should raise error when page is full."""
        page = Page(page_id=1)

        # Fill the page
        large_record = b"X" * (page.get_free_space() - SLOT_SIZE)
        page.insert_record(large_record)

        with pytest.raises(PageFullError) as exc_info:
            page.insert_record(b"One more")

        assert exc_info.value.page_id == 1

    def test_maximum_record_size(self):
        """Should accept maximum-size record."""
        page = Page(page_id=1)
        max_size = page.get_free_space() - SLOT_SIZE

        large_record = b"X" * max_size
        slot_id = page.insert_record(large_record)

        assert page.get_record(slot_id) == large_record
        assert page.get_free_space() == 0

    def test_empty_record_treated_as_deleted(self):
        """Empty records are stored but treated as deleted (length=0).

        This is a known design limitation: length=0 is used as the
        deletion marker, so empty records cannot be distinguished
        from deleted slots. Use at least 1-byte records in practice.
        """
        page = Page(page_id=1)

        slot_id = page.insert_record(b"")

        # Empty record has length=0, same as deleted marker
        with pytest.raises(ValueError, match="deleted"):
            page.get_record(slot_id)


class TestPageSerialization:
    """Tests for page serialization and checksum."""

    def test_serialization_roundtrip(self):
        """Page should serialize and deserialize correctly."""
        page = Page(page_id=42)
        page.lsn = 1000
        records = [b"Record A", b"Record B", b"Record C"]
        for r in records:
            page.insert_record(r)

        data = page.to_bytes()
        assert len(data) == PAGE_SIZE

        restored = Page.from_bytes(data)
        assert restored.page_id == 42
        assert restored.lsn == 1000
        assert restored.slot_count == 3
        for slot_id, expected in enumerate(records):
            assert restored.get_record(slot_id) == expected

    def test_serialization_preserves_deletions(self):
        """Serialization should preserve deletion state."""
        page = Page(page_id=1)
        page.insert_record(b"Keep")
        slot_to_delete = page.insert_record(b"Delete me")
        page.insert_record(b"Also keep")
        page.delete_record(slot_to_delete)

        data = page.to_bytes()
        restored = Page.from_bytes(data)

        assert restored.get_record(0) == b"Keep"
        assert restored.get_record(2) == b"Also keep"
        with pytest.raises(ValueError):
            restored.get_record(slot_to_delete)

    def test_checksum_validation(self):
        """Should detect corrupted pages."""
        page = Page(page_id=1)
        page.insert_record(b"Important data")

        data = bytearray(page.to_bytes())
        # Corrupt the data
        data[100] ^= 0xFF

        with pytest.raises(PageCorruptionError) as exc_info:
            Page.from_bytes(bytes(data))

        assert exc_info.value.page_id == 1

    def test_skip_checksum_verification(self):
        """Should allow skipping checksum verification."""
        page = Page(page_id=1)
        page.insert_record(b"Data")

        data = bytearray(page.to_bytes())
        data[100] ^= 0xFF  # Corrupt

        # Should not raise with verify_checksum=False
        restored = Page.from_bytes(bytes(data), verify_checksum=False)
        assert restored.page_id == 1

    def test_invalid_data_length(self):
        """Should reject data with wrong length."""
        with pytest.raises(ValueError, match="Invalid page data length"):
            Page.from_bytes(b"Too short")

    def test_restored_page_not_dirty(self):
        """Restored page should not be marked dirty."""
        page = Page(page_id=1)
        page.insert_record(b"Data")
        assert page.is_dirty

        data = page.to_bytes()
        restored = Page.from_bytes(data)

        assert not restored.is_dirty


class TestPageIteration:
    """Tests for record iteration."""

    def test_iter_records(self):
        """Should iterate over all non-deleted records."""
        page = Page(page_id=1)
        records = [b"First", b"Second", b"Third"]
        for r in records:
            page.insert_record(r)

        result = list(page.iter_records())

        assert result == [(0, b"First"), (1, b"Second"), (2, b"Third")]

    def test_iter_skips_deleted(self):
        """Iteration should skip deleted records."""
        page = Page(page_id=1)
        page.insert_record(b"Keep")
        page.insert_record(b"Delete")
        page.insert_record(b"Also keep")
        page.delete_record(1)

        result = list(page.iter_records())

        assert result == [(0, b"Keep"), (2, b"Also keep")]

    def test_iter_empty_page(self):
        """Should handle empty page iteration."""
        page = Page(page_id=1)

        result = list(page.iter_records())

        assert result == []


class TestPageProperties:
    """Tests for page properties."""

    def test_lsn_property(self):
        """LSN should be readable and writable."""
        page = Page(page_id=1)
        assert page.lsn == 0

        page.lsn = 42
        assert page.lsn == 42
        assert page.is_dirty

    def test_mark_clean(self):
        """mark_clean should clear dirty flag."""
        page = Page(page_id=1)
        page.insert_record(b"Data")
        assert page.is_dirty

        page.mark_clean()

        assert not page.is_dirty

    def test_repr(self):
        """__repr__ should return useful debug info."""
        page = Page(page_id=42, page_type=PageType.INDEX)
        page.insert_record(b"Data")

        repr_str = repr(page)

        assert "42" in repr_str
        assert "INDEX" in repr_str
        assert "slots=1" in repr_str


class TestPageMemoryOptimization:
    """Tests for memory optimization features."""

    def test_uses_slots(self):
        """Page should use __slots__ for memory efficiency.

        Note: When inheriting from ABC, __dict__ may still exist,
        but __slots__ still provides memory benefits for declared
        attributes by avoiding per-instance dict entries for them.
        """
        assert hasattr(Page, "__slots__")
        expected_slots = {"_page_id", "_page_type", "_lsn", "_data", "_dirty"}
        assert set(Page.__slots__) == expected_slots

    def test_data_is_bytearray(self):
        """Internal data should be bytearray for in-place modification."""
        page = Page(page_id=1)
        # Access private attribute for testing
        assert isinstance(page._data, bytearray)


class TestPageEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_large_page_id(self):
        """Should handle large page IDs."""
        page = Page(page_id=2**63 - 1)
        data = page.to_bytes()
        restored = Page.from_bytes(data)
        assert restored.page_id == 2**63 - 1

    def test_binary_data(self):
        """Should handle binary data with all byte values."""
        page = Page(page_id=1)
        binary_record = bytes(range(256))

        slot_id = page.insert_record(binary_record)

        assert page.get_record(slot_id) == binary_record

    def test_many_small_records(self):
        """Should handle many small records."""
        page = Page(page_id=1)
        records = [f"R{i}".encode() for i in range(100)]

        slot_ids = []
        for r in records:
            if page.get_free_space() >= len(r) + SLOT_SIZE:
                slot_ids.append(page.insert_record(r))
            else:
                break

        for slot_id, expected in zip(slot_ids, records[:len(slot_ids)]):
            assert page.get_record(slot_id) == expected

    def test_fill_then_delete_then_insert(self):
        """Deletion doesn't reclaim space until compaction."""
        page = Page(page_id=1)

        # Fill page with one large record
        max_size = page.get_free_space() - SLOT_SIZE
        slot_id = page.insert_record(b"X" * max_size)

        # Delete it
        page.delete_record(slot_id)

        # Space still used (no compaction implemented yet)
        # New insert should fail if it needs the same space
        with pytest.raises(PageFullError):
            page.insert_record(b"Y" * max_size)
