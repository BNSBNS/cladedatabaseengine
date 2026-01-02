"""
Unit tests for B+Tree index implementation.

Tests cover:
- Search, insert, delete operations
- Range scans
- Tree balancing (splits)
- Edge cases
"""

import pytest
import random

from clade.index.btree import BPlusTree, BTreeNode
from clade.storage.interfaces import RecordID
from clade.utils.errors import DuplicateKeyError, KeyNotFoundError


@pytest.fixture
def btree():
    """Create a B+Tree with small order for testing."""
    return BPlusTree(order=4)


@pytest.fixture
def large_btree():
    """Create a B+Tree with default order."""
    return BPlusTree(order=128)


def rid(page: int, slot: int = 0) -> RecordID:
    """Helper to create RecordID."""
    return RecordID(page_id=page, slot_id=slot)


class TestBTreeNode:
    """Tests for BTreeNode dataclass."""

    def test_create_leaf(self):
        """Should create an empty leaf node."""
        node = BTreeNode.create_leaf()

        assert node.is_leaf
        assert node.keys == []
        assert node.values == []
        assert node.next_leaf is None

    def test_create_internal(self):
        """Should create an empty internal node."""
        node = BTreeNode.create_internal()

        assert not node.is_leaf
        assert node.keys == []
        assert node.children == []


class TestBTreeCreation:
    """Tests for B+Tree creation."""

    def test_create_tree(self):
        """Should create an empty tree."""
        tree = BPlusTree(order=4)

        assert tree.size == 0
        assert tree.height == 0

    def test_invalid_order(self):
        """Should reject order less than 3."""
        with pytest.raises(ValueError, match="Order must be at least 3"):
            BPlusTree(order=2)


class TestBTreeInsert:
    """Tests for B+Tree insertion."""

    def test_insert_single(self, btree):
        """Should insert a single key."""
        btree.insert(10, rid(1))

        assert btree.size == 1
        assert btree.search(10) == rid(1)

    def test_insert_multiple_sorted(self, btree):
        """Should insert multiple keys in sorted order."""
        for i in range(10):
            btree.insert(i, rid(i))

        assert btree.size == 10
        for i in range(10):
            assert btree.search(i) == rid(i)

    def test_insert_multiple_reverse(self, btree):
        """Should insert multiple keys in reverse order."""
        for i in range(9, -1, -1):
            btree.insert(i, rid(i))

        assert btree.size == 10
        for i in range(10):
            assert btree.search(i) == rid(i)

    def test_insert_multiple_random(self, btree):
        """Should insert keys in random order."""
        keys = list(range(20))
        random.shuffle(keys)

        for k in keys:
            btree.insert(k, rid(k))

        assert btree.size == 20
        for k in keys:
            assert btree.search(k) == rid(k)

    def test_insert_duplicate_fails(self, btree):
        """Should raise error for duplicate key in unique index."""
        btree.insert(10, rid(1))

        with pytest.raises(DuplicateKeyError):
            btree.insert(10, rid(2))

    def test_insert_duplicate_allowed_non_unique(self):
        """Non-unique index should allow duplicate keys."""
        tree = BPlusTree(order=4, unique=False)

        tree.insert(10, rid(1))
        tree.insert(10, rid(2))

        assert tree.size == 2

    def test_insert_triggers_split(self, btree):
        """Inserting beyond order should trigger split."""
        # Order 4 means split at 4 keys
        for i in range(10):
            btree.insert(i, rid(i))

        assert btree.size == 10
        assert btree.height >= 1  # At least one split occurred


class TestBTreeSearch:
    """Tests for B+Tree search."""

    def test_search_empty_tree(self, btree):
        """Searching empty tree should return None."""
        assert btree.search(10) is None

    def test_search_existing_key(self, btree):
        """Should find existing key."""
        btree.insert(10, rid(1))

        assert btree.search(10) == rid(1)

    def test_search_nonexistent_key(self, btree):
        """Should return None for nonexistent key."""
        btree.insert(10, rid(1))
        btree.insert(20, rid(2))

        assert btree.search(15) is None

    def test_search_after_splits(self, btree):
        """Should find keys after splits."""
        for i in range(100):
            btree.insert(i, rid(i))

        for i in range(100):
            assert btree.search(i) == rid(i)

    def test_contains_operator(self, btree):
        """Should support 'in' operator."""
        btree.insert(10, rid(1))

        assert 10 in btree
        assert 20 not in btree


class TestBTreeDelete:
    """Tests for B+Tree deletion."""

    def test_delete_single(self, btree):
        """Should delete a key."""
        btree.insert(10, rid(1))
        btree.delete(10)

        assert btree.size == 0
        assert btree.search(10) is None

    def test_delete_nonexistent_fails(self, btree):
        """Should raise error for nonexistent key."""
        btree.insert(10, rid(1))

        with pytest.raises(KeyNotFoundError):
            btree.delete(20)

    def test_delete_from_middle(self, btree):
        """Should delete key from middle of leaf."""
        for i in range(5):
            btree.insert(i, rid(i))

        btree.delete(2)

        assert btree.size == 4
        assert btree.search(2) is None
        assert btree.search(1) == rid(1)
        assert btree.search(3) == rid(3)

    def test_delete_all(self, btree):
        """Should handle deleting all keys."""
        for i in range(10):
            btree.insert(i, rid(i))

        for i in range(10):
            btree.delete(i)

        assert btree.size == 0


class TestBTreeRangeScan:
    """Tests for B+Tree range scans."""

    def test_range_scan_empty(self, btree):
        """Range scan on empty tree should yield nothing."""
        result = list(btree.range_scan(0, 100))

        assert result == []

    def test_range_scan_all(self, btree):
        """Range scan covering all keys."""
        for i in range(10):
            btree.insert(i, rid(i))

        result = list(btree.range_scan(0, 9))

        assert len(result) == 10
        assert [k for k, _ in result] == list(range(10))

    def test_range_scan_partial(self, btree):
        """Range scan covering partial range."""
        for i in range(20):
            btree.insert(i, rid(i))

        result = list(btree.range_scan(5, 14))

        assert len(result) == 10
        assert [k for k, _ in result] == list(range(5, 15))

    def test_range_scan_single_key(self, btree):
        """Range scan for single key."""
        for i in range(10):
            btree.insert(i, rid(i))

        result = list(btree.range_scan(5, 5))

        assert len(result) == 1
        assert result[0][0] == 5

    def test_range_scan_no_matches(self, btree):
        """Range scan with no matches."""
        for i in range(10):
            btree.insert(i * 10, rid(i))

        result = list(btree.range_scan(11, 19))

        assert result == []

    def test_range_scan_sorted_order(self, btree):
        """Range scan should return keys in sorted order."""
        keys = [5, 3, 8, 1, 9, 2, 7, 4, 6, 0]
        for k in keys:
            btree.insert(k, rid(k))

        result = list(btree.range_scan(0, 9))

        assert [k for k, _ in result] == list(range(10))


class TestBTreeScanAll:
    """Tests for full tree scan."""

    def test_scan_all_empty(self, btree):
        """Scan on empty tree should yield nothing."""
        result = list(btree.scan_all())

        assert result == []

    def test_scan_all(self, btree):
        """Should scan all entries in order."""
        keys = [5, 3, 8, 1, 9, 2, 7, 4, 6, 0]
        for k in keys:
            btree.insert(k, rid(k))

        result = list(btree.scan_all())

        assert [k for k, _ in result] == list(range(10))


class TestBTreeProperties:
    """Tests for B+Tree properties."""

    def test_size_property(self, btree):
        """size should track entry count."""
        assert btree.size == 0

        btree.insert(1, rid(1))
        assert btree.size == 1

        btree.insert(2, rid(2))
        assert btree.size == 2

        btree.delete(1)
        assert btree.size == 1

    def test_height_property(self, btree):
        """height should reflect tree depth."""
        assert btree.height == 0

        # Insert enough to cause splits
        for i in range(20):
            btree.insert(i, rid(i))

        assert btree.height >= 1

    def test_min_max_key(self, btree):
        """Should track min and max keys."""
        assert btree.min_key() is None
        assert btree.max_key() is None

        keys = [5, 3, 8, 1, 9, 2]
        for k in keys:
            btree.insert(k, rid(k))

        assert btree.min_key() == 1
        assert btree.max_key() == 9

    def test_len(self, btree):
        """Should support len()."""
        assert len(btree) == 0

        btree.insert(1, rid(1))
        assert len(btree) == 1

    def test_repr(self, btree):
        """__repr__ should return useful info."""
        btree.insert(1, rid(1))
        repr_str = repr(btree)

        assert "BPlusTree" in repr_str
        assert "order=" in repr_str
        assert "size=" in repr_str


class TestBTreeScalability:
    """Tests for B+Tree with larger datasets."""

    def test_many_entries(self, large_btree):
        """Should handle many entries."""
        n = 10000
        for i in range(n):
            large_btree.insert(i, rid(i))

        assert large_btree.size == n

        # Random lookups
        for _ in range(100):
            k = random.randint(0, n - 1)
            assert large_btree.search(k) == rid(k)

    def test_random_operations(self, large_btree):
        """Should handle mixed random operations."""
        inserted = set()

        for _ in range(1000):
            op = random.choice(["insert", "delete", "search"])
            key = random.randint(0, 500)

            if op == "insert":
                if key not in inserted:
                    large_btree.insert(key, rid(key))
                    inserted.add(key)
            elif op == "delete":
                if key in inserted:
                    large_btree.delete(key)
                    inserted.discard(key)
            else:
                result = large_btree.search(key)
                if key in inserted:
                    assert result == rid(key)
                else:
                    assert result is None

        assert large_btree.size == len(inserted)


class TestBTreeEdgeCases:
    """Tests for edge cases."""

    def test_string_keys(self):
        """Should work with string keys."""
        tree = BPlusTree(order=4)

        tree.insert("apple", rid(1))
        tree.insert("banana", rid(2))
        tree.insert("cherry", rid(3))

        assert tree.search("banana") == rid(2)
        assert list(tree.range_scan("a", "c")) == [
            ("apple", rid(1)),
            ("banana", rid(2)),
        ]

    def test_negative_keys(self, btree):
        """Should work with negative keys."""
        for i in range(-10, 10):
            btree.insert(i, rid(i + 10))

        assert btree.search(-5) == rid(5)
        assert btree.min_key() == -10
        assert btree.max_key() == 9

    def test_float_keys(self):
        """Should work with float keys."""
        tree = BPlusTree(order=4)

        tree.insert(1.5, rid(1))
        tree.insert(2.5, rid(2))
        tree.insert(0.5, rid(0))

        assert tree.search(1.5) == rid(1)
        assert tree.min_key() == 0.5
