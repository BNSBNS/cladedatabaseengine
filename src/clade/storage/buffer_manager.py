"""
Buffer Pool Manager with CLOCK eviction policy.

Manages an in-memory cache of database pages, coordinating reads
and writes through the file manager. Uses the CLOCK algorithm for
efficient page eviction that's scan-resistant.

Key Features:
- CLOCK eviction (approximates LRU with O(1) operations)
- Pin counting for concurrent access
- Dirty page tracking
- Automatic flush on eviction
"""

import threading
from dataclasses import dataclass
from typing import Optional

from clade.storage.interfaces import IBufferPool, IPage, PageType
from clade.storage.file_manager import FileManager
from clade.storage.page import Page
from clade.utils.errors import BufferPoolFullError, PageNotFoundError


@dataclass
class BufferFrame:
    """
    A frame in the buffer pool holding a page.

    Tracks metadata for eviction and concurrency control.
    """

    __slots__ = ("page", "page_id", "pin_count", "is_dirty", "ref_bit")

    page: Optional[Page]
    page_id: int  # -1 if frame is empty
    pin_count: int
    is_dirty: bool
    ref_bit: bool  # For CLOCK algorithm

    @classmethod
    def empty(cls) -> "BufferFrame":
        """Create an empty buffer frame."""
        return cls(page=None, page_id=-1, pin_count=0, is_dirty=False, ref_bit=False)


class BufferPoolManager(IBufferPool):
    """
    Buffer pool with CLOCK eviction policy.

    CLOCK Algorithm:
    - Each frame has a "reference bit" set when accessed
    - Eviction scans frames in circular order
    - If ref_bit=1, set to 0 and skip (give second chance)
    - If ref_bit=0 and pin_count=0, evict this frame

    Thread Safety: Thread-safe via internal locking.
    """

    __slots__ = (
        "_file_manager",
        "_pool_size",
        "_frames",
        "_page_table",
        "_clock_hand",
        "_lock",
    )

    def __init__(self, file_manager: FileManager, pool_size: int = 1024) -> None:
        """
        Initialize buffer pool.

        Args:
            file_manager: File manager for disk I/O
            pool_size: Number of frames in the pool (default 1024)

        Raises:
            ValueError: If pool_size is not positive
        """
        if pool_size <= 0:
            raise ValueError(f"pool_size must be positive, got {pool_size}")

        self._file_manager = file_manager
        self._pool_size = pool_size
        self._frames = [BufferFrame.empty() for _ in range(pool_size)]
        self._page_table: dict[int, int] = {}  # page_id -> frame_id
        self._clock_hand = 0
        self._lock = threading.RLock()

    # ─────────────────────────────────────────────────────────────────────────
    # IBufferPool Interface
    # ─────────────────────────────────────────────────────────────────────────

    def fetch_page(self, page_id: int) -> IPage:
        """
        Fetch a page from the buffer pool or disk.

        If the page is in the pool, returns it and increments pin count.
        If not, loads from disk (evicting if necessary).

        Args:
            page_id: The page to fetch

        Returns:
            The requested page (pinned)

        Raises:
            PageNotFoundError: If page doesn't exist on disk
            BufferPoolFullError: If can't evict any pages
        """
        with self._lock:
            # Check if page is already in pool
            if page_id in self._page_table:
                frame_id = self._page_table[page_id]
                frame = self._frames[frame_id]
                frame.pin_count += 1
                frame.ref_bit = True
                return frame.page

            # Need to load from disk
            frame_id = self._find_victim_frame()
            if frame_id is None:
                raise BufferPoolFullError(self._pool_size)

            # Evict current page if needed
            frame = self._frames[frame_id]
            if frame.page_id != -1:
                self._evict_frame(frame_id)

            # Load new page
            try:
                data = self._file_manager.read_page(page_id)
                page = Page.from_bytes(data)
            except PageNotFoundError:
                raise

            # Install in frame
            frame.page = page
            frame.page_id = page_id
            frame.pin_count = 1
            frame.is_dirty = False
            frame.ref_bit = True

            self._page_table[page_id] = frame_id

            return page

    def unpin_page(self, page_id: int, is_dirty: bool) -> None:
        """
        Unpin a page, marking it as dirty if modified.

        Args:
            page_id: The page to unpin
            is_dirty: True if page was modified
        """
        with self._lock:
            if page_id not in self._page_table:
                return  # Page not in pool, ignore

            frame_id = self._page_table[page_id]
            frame = self._frames[frame_id]

            if frame.pin_count > 0:
                frame.pin_count -= 1

            if is_dirty:
                frame.is_dirty = True

    def flush_page(self, page_id: int) -> None:
        """
        Force write a specific page to disk.

        Args:
            page_id: The page to flush

        Raises:
            PageNotFoundError: If page is not in pool
        """
        with self._lock:
            if page_id not in self._page_table:
                raise PageNotFoundError(page_id)

            frame_id = self._page_table[page_id]
            frame = self._frames[frame_id]

            if frame.is_dirty and frame.page is not None:
                data = frame.page.to_bytes()
                self._file_manager.write_page(page_id, data)
                frame.is_dirty = False
                frame.page.mark_clean()

    def flush_all_dirty(self) -> None:
        """
        Flush all dirty pages to disk.

        Used during checkpoints and shutdown.
        """
        with self._lock:
            for frame in self._frames:
                if frame.is_dirty and frame.page is not None:
                    data = frame.page.to_bytes()
                    self._file_manager.write_page(frame.page_id, data)
                    frame.is_dirty = False
                    frame.page.mark_clean()

            self._file_manager.sync()

    def new_page(self) -> tuple[int, IPage]:
        """
        Allocate and return a new page.

        The page is automatically pinned. Caller must unpin when done.

        Returns:
            Tuple of (page_id, page)

        Raises:
            BufferPoolFullError: If can't evict any pages
        """
        with self._lock:
            # Find a frame for the new page
            frame_id = self._find_victim_frame()
            if frame_id is None:
                raise BufferPoolFullError(self._pool_size)

            # Evict current page if needed
            frame = self._frames[frame_id]
            if frame.page_id != -1:
                self._evict_frame(frame_id)

            # Allocate new page on disk
            page_id = self._file_manager.allocate_page()

            # Create new page object
            page = Page(page_id=page_id)

            # Install in frame
            frame.page = page
            frame.page_id = page_id
            frame.pin_count = 1
            frame.is_dirty = True  # New page is dirty
            frame.ref_bit = True

            self._page_table[page_id] = frame_id

            return page_id, page

    # ─────────────────────────────────────────────────────────────────────────
    # CLOCK Eviction
    # ─────────────────────────────────────────────────────────────────────────

    def _find_victim_frame(self) -> Optional[int]:
        """
        Find a frame to evict using CLOCK algorithm.

        Returns:
            Frame ID to use, or None if all frames are pinned
        """
        # First pass: look for empty frame
        for i, frame in enumerate(self._frames):
            if frame.page_id == -1:
                return i

        # CLOCK algorithm: scan up to 2 full rotations
        num_scanned = 0
        max_scans = 2 * self._pool_size

        while num_scanned < max_scans:
            frame = self._frames[self._clock_hand]

            if frame.pin_count == 0:
                if not frame.ref_bit:
                    # Found victim
                    victim = self._clock_hand
                    self._clock_hand = (self._clock_hand + 1) % self._pool_size
                    return victim
                else:
                    # Give second chance
                    frame.ref_bit = False

            self._clock_hand = (self._clock_hand + 1) % self._pool_size
            num_scanned += 1

        # All pages are pinned
        return None

    def _evict_frame(self, frame_id: int) -> None:
        """
        Evict the page in a frame, flushing if dirty.

        Args:
            frame_id: The frame to evict
        """
        frame = self._frames[frame_id]

        if frame.page_id == -1:
            return  # Already empty

        # Flush if dirty
        if frame.is_dirty and frame.page is not None:
            data = frame.page.to_bytes()
            self._file_manager.write_page(frame.page_id, data)

        # Remove from page table
        if frame.page_id in self._page_table:
            del self._page_table[frame.page_id]

        # Clear frame
        frame.page = None
        frame.page_id = -1
        frame.pin_count = 0
        frame.is_dirty = False
        frame.ref_bit = False

    # ─────────────────────────────────────────────────────────────────────────
    # Properties and Utilities
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def pool_size(self) -> int:
        """Get the buffer pool size."""
        return self._pool_size

    @property
    def num_pages(self) -> int:
        """Get the number of pages currently in the pool."""
        return len(self._page_table)

    def get_pin_count(self, page_id: int) -> int:
        """Get the pin count for a page (0 if not in pool)."""
        with self._lock:
            if page_id not in self._page_table:
                return 0
            frame_id = self._page_table[page_id]
            return self._frames[frame_id].pin_count

    def is_page_dirty(self, page_id: int) -> bool:
        """Check if a page is marked dirty."""
        with self._lock:
            if page_id not in self._page_table:
                return False
            frame_id = self._page_table[page_id]
            return self._frames[frame_id].is_dirty

    def is_page_in_pool(self, page_id: int) -> bool:
        """Check if a page is in the buffer pool."""
        with self._lock:
            return page_id in self._page_table

    def get_hit_ratio(self) -> float:
        """Get buffer pool hit ratio (placeholder for metrics)."""
        # TODO: Implement hit/miss counting
        return 0.0

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"BufferPoolManager(size={self._pool_size}, "
            f"pages={self.num_pages}, "
            f"clock_hand={self._clock_hand})"
        )
