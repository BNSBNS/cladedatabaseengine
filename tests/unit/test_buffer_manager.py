"""
Unit tests for Buffer Pool Manager with CLOCK eviction.

Tests cover:
- Page fetching and pinning
- Page creation
- Dirty page handling
- CLOCK eviction algorithm
- Flush operations
"""

import pytest
import tempfile
from pathlib import Path

from clade.storage.buffer_manager import BufferPoolManager, BufferFrame
from clade.storage.file_manager import FileManager
from clade.storage.page import PAGE_SIZE
from clade.utils.errors import BufferPoolFullError, PageNotFoundError


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


@pytest.fixture
def buffer_pool(file_manager):
    """Create a buffer pool for testing."""
    return BufferPoolManager(file_manager, pool_size=10)


class TestBufferFrame:
    """Tests for BufferFrame dataclass."""

    def test_empty_frame(self):
        """Empty frame should have correct defaults."""
        frame = BufferFrame.empty()

        assert frame.page is None
        assert frame.page_id == -1
        assert frame.pin_count == 0
        assert not frame.is_dirty
        assert not frame.ref_bit


class TestBufferPoolCreation:
    """Tests for buffer pool creation."""

    def test_create_buffer_pool(self, file_manager):
        """Should create buffer pool with specified size."""
        bp = BufferPoolManager(file_manager, pool_size=100)

        assert bp.pool_size == 100
        assert bp.num_pages == 0

    def test_invalid_pool_size(self, file_manager):
        """Should reject non-positive pool size."""
        with pytest.raises(ValueError, match="positive"):
            BufferPoolManager(file_manager, pool_size=0)

        with pytest.raises(ValueError, match="positive"):
            BufferPoolManager(file_manager, pool_size=-1)


class TestBufferPoolNewPage:
    """Tests for new page allocation."""

    def test_new_page_returns_page_and_id(self, buffer_pool):
        """new_page should return page_id and page."""
        page_id, page = buffer_pool.new_page()

        assert page_id == 0
        assert page is not None
        assert page.page_id == 0

    def test_new_page_is_pinned(self, buffer_pool):
        """New page should be pinned."""
        page_id, page = buffer_pool.new_page()

        assert buffer_pool.get_pin_count(page_id) == 1

    def test_new_page_is_dirty(self, buffer_pool):
        """New page should be marked dirty."""
        page_id, page = buffer_pool.new_page()

        assert buffer_pool.is_page_dirty(page_id)

    def test_new_page_is_in_pool(self, buffer_pool):
        """New page should be in the pool."""
        page_id, page = buffer_pool.new_page()

        assert buffer_pool.is_page_in_pool(page_id)

    def test_multiple_new_pages(self, buffer_pool):
        """Should allocate multiple new pages."""
        pages = [buffer_pool.new_page() for _ in range(5)]

        page_ids = [p[0] for p in pages]
        assert page_ids == [0, 1, 2, 3, 4]


class TestBufferPoolFetch:
    """Tests for page fetching."""

    def test_fetch_existing_page(self, buffer_pool):
        """Should fetch an existing page from disk."""
        # Create and write a page
        page_id, page = buffer_pool.new_page()
        page.insert_record(b"Test data")
        buffer_pool.unpin_page(page_id, is_dirty=True)
        buffer_pool.flush_page(page_id)

        # Fetch it
        fetched = buffer_pool.fetch_page(page_id)

        assert fetched.page_id == page_id
        assert fetched.get_record(0) == b"Test data"

    def test_fetch_increments_pin_count(self, buffer_pool):
        """Fetching should increment pin count."""
        page_id, page = buffer_pool.new_page()
        buffer_pool.unpin_page(page_id, is_dirty=False)

        assert buffer_pool.get_pin_count(page_id) == 0

        buffer_pool.fetch_page(page_id)
        assert buffer_pool.get_pin_count(page_id) == 1

        buffer_pool.fetch_page(page_id)
        assert buffer_pool.get_pin_count(page_id) == 2

    def test_fetch_nonexistent_page_fails(self, buffer_pool):
        """Should raise error for nonexistent page."""
        with pytest.raises(PageNotFoundError):
            buffer_pool.fetch_page(999)

    def test_fetch_sets_ref_bit(self, buffer_pool):
        """Fetching should set reference bit for CLOCK."""
        page_id, page = buffer_pool.new_page()
        buffer_pool.unpin_page(page_id, is_dirty=False)

        # Force ref_bit to False by accessing the frame directly
        frame_id = buffer_pool._page_table[page_id]
        buffer_pool._frames[frame_id].ref_bit = False

        # Fetch should set ref_bit
        buffer_pool.fetch_page(page_id)
        assert buffer_pool._frames[frame_id].ref_bit


class TestBufferPoolUnpin:
    """Tests for page unpinning."""

    def test_unpin_decrements_pin_count(self, buffer_pool):
        """Unpinning should decrement pin count."""
        page_id, page = buffer_pool.new_page()
        assert buffer_pool.get_pin_count(page_id) == 1

        buffer_pool.unpin_page(page_id, is_dirty=False)
        assert buffer_pool.get_pin_count(page_id) == 0

    def test_unpin_marks_dirty(self, buffer_pool):
        """Unpinning with dirty=True should mark page dirty."""
        page_id, page = buffer_pool.new_page()
        buffer_pool.unpin_page(page_id, is_dirty=False)
        buffer_pool.flush_page(page_id)

        assert not buffer_pool.is_page_dirty(page_id)

        buffer_pool.fetch_page(page_id)
        buffer_pool.unpin_page(page_id, is_dirty=True)

        assert buffer_pool.is_page_dirty(page_id)

    def test_unpin_nonexistent_page_ignored(self, buffer_pool):
        """Unpinning nonexistent page should be ignored."""
        # Should not raise
        buffer_pool.unpin_page(999, is_dirty=False)

    def test_multiple_pins_require_multiple_unpins(self, buffer_pool):
        """Multiple pins require equal unpins."""
        page_id, page = buffer_pool.new_page()
        buffer_pool.fetch_page(page_id)
        buffer_pool.fetch_page(page_id)

        assert buffer_pool.get_pin_count(page_id) == 3

        buffer_pool.unpin_page(page_id, is_dirty=False)
        assert buffer_pool.get_pin_count(page_id) == 2

        buffer_pool.unpin_page(page_id, is_dirty=False)
        assert buffer_pool.get_pin_count(page_id) == 1

        buffer_pool.unpin_page(page_id, is_dirty=False)
        assert buffer_pool.get_pin_count(page_id) == 0


class TestBufferPoolFlush:
    """Tests for page flushing."""

    def test_flush_page(self, buffer_pool, temp_db_path):
        """Flushing should write page to disk."""
        page_id, page = buffer_pool.new_page()
        page.insert_record(b"Flush test")
        buffer_pool.unpin_page(page_id, is_dirty=True)

        buffer_pool.flush_page(page_id)

        assert not buffer_pool.is_page_dirty(page_id)

    def test_flush_nonexistent_page_fails(self, buffer_pool):
        """Flushing nonexistent page should raise error."""
        with pytest.raises(PageNotFoundError):
            buffer_pool.flush_page(999)

    def test_flush_all_dirty(self, buffer_pool):
        """flush_all_dirty should flush all dirty pages."""
        # Create multiple dirty pages
        for _ in range(5):
            page_id, page = buffer_pool.new_page()
            page.insert_record(b"Data")
            buffer_pool.unpin_page(page_id, is_dirty=True)

        buffer_pool.flush_all_dirty()

        # All pages should be clean
        for page_id in range(5):
            assert not buffer_pool.is_page_dirty(page_id)


class TestBufferPoolEviction:
    """Tests for CLOCK eviction algorithm."""

    def test_eviction_when_pool_full(self, file_manager):
        """Should evict pages when pool is full."""
        bp = BufferPoolManager(file_manager, pool_size=3)

        # Create 3 pages and unpin them
        for i in range(3):
            page_id, page = bp.new_page()
            bp.unpin_page(page_id, is_dirty=False)

        assert bp.num_pages == 3

        # Create 4th page, should trigger eviction
        page_id, page = bp.new_page()

        assert bp.num_pages == 3  # Still 3 (one evicted)
        assert bp.is_page_in_pool(page_id)  # New page is in pool

    def test_pinned_pages_not_evicted(self, file_manager):
        """Pinned pages should not be evicted."""
        bp = BufferPoolManager(file_manager, pool_size=3)

        # Create 3 pages, keep 2 pinned
        page_ids = []
        for i in range(3):
            page_id, page = bp.new_page()
            page_ids.append(page_id)
            if i < 2:
                pass  # Keep pinned
            else:
                bp.unpin_page(page_id, is_dirty=False)

        # Create 4th page, only unpinned page can be evicted
        page_id, page = bp.new_page()

        assert bp.is_page_in_pool(page_ids[0])  # Still pinned
        assert bp.is_page_in_pool(page_ids[1])  # Still pinned
        assert not bp.is_page_in_pool(page_ids[2])  # Was evicted
        assert bp.is_page_in_pool(page_id)  # New page

    def test_all_pinned_raises_error(self, file_manager):
        """Should raise error when all pages are pinned."""
        bp = BufferPoolManager(file_manager, pool_size=3)

        # Create 3 pages and keep all pinned
        for i in range(3):
            bp.new_page()  # Don't unpin

        # Try to create 4th page
        with pytest.raises(BufferPoolFullError):
            bp.new_page()

    def test_clock_second_chance(self, file_manager):
        """CLOCK should give second chance to referenced pages."""
        bp = BufferPoolManager(file_manager, pool_size=3)

        # Create 3 pages
        page_ids = []
        for i in range(3):
            page_id, page = bp.new_page()
            page_ids.append(page_id)
            bp.unpin_page(page_id, is_dirty=False)

        # Clear all ref_bits to simulate time passing
        # (In real usage, CLOCK would clear them during failed eviction attempts)
        for frame in bp._frames:
            frame.ref_bit = False

        # Access page 1 (sets its ref_bit back to True)
        bp.fetch_page(page_ids[1])
        bp.unpin_page(page_ids[1], is_dirty=False)

        # Create new page - CLOCK starts at hand position and evicts first
        # unpinned page with ref_bit=False
        new_page_id, _ = bp.new_page()

        # Page 1 should still be in pool (had ref_bit set)
        assert bp.is_page_in_pool(page_ids[1])

    def test_eviction_flushes_dirty_pages(self, file_manager, temp_db_path):
        """Eviction should flush dirty pages to disk."""
        bp = BufferPoolManager(file_manager, pool_size=2)

        # Create a dirty page
        page_id1, page1 = bp.new_page()
        page1.insert_record(b"Important data")
        bp.unpin_page(page_id1, is_dirty=True)

        # Create another page
        page_id2, _ = bp.new_page()
        bp.unpin_page(page_id2, is_dirty=False)

        # Create third page, triggers eviction
        page_id3, _ = bp.new_page()

        # Close and reopen to verify data persisted
        bp.flush_all_dirty()

        # Fetch the first page (should load from disk)
        page = bp.fetch_page(page_id1)
        assert page.get_record(0) == b"Important data"


class TestBufferPoolPersistence:
    """Tests for data persistence."""

    def test_data_persists_through_eviction(self, file_manager):
        """Data should persist when page is evicted and re-fetched."""
        bp = BufferPoolManager(file_manager, pool_size=2)

        # Create and modify a page
        page_id, page = bp.new_page()
        page.insert_record(b"Persistent data")
        bp.unpin_page(page_id, is_dirty=True)

        # Create more pages to force eviction
        for _ in range(3):
            new_id, _ = bp.new_page()
            bp.unpin_page(new_id, is_dirty=False)

        # Fetch original page (may load from disk)
        fetched = bp.fetch_page(page_id)
        assert fetched.get_record(0) == b"Persistent data"


class TestBufferPoolProperties:
    """Tests for buffer pool properties."""

    def test_num_pages(self, buffer_pool):
        """num_pages should track pages in pool."""
        assert buffer_pool.num_pages == 0

        buffer_pool.new_page()
        assert buffer_pool.num_pages == 1

        buffer_pool.new_page()
        assert buffer_pool.num_pages == 2

    def test_repr(self, buffer_pool):
        """__repr__ should return useful info."""
        repr_str = repr(buffer_pool)

        assert "BufferPoolManager" in repr_str
        assert "size=" in repr_str
        assert "pages=" in repr_str


class TestBufferPoolThreadSafety:
    """Basic thread safety tests."""

    def test_concurrent_access(self, buffer_pool):
        """Basic concurrent access should not crash."""
        import threading

        errors = []

        def worker():
            try:
                for _ in range(10):
                    page_id, page = buffer_pool.new_page()
                    buffer_pool.unpin_page(page_id, is_dirty=False)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
