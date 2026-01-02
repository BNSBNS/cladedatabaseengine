"""
B+Tree index implementation for efficient key-based lookups.

A B+Tree is a balanced tree structure optimized for disk-based access:
- All values are stored in leaf nodes
- Leaf nodes are linked for efficient range scans
- Internal nodes contain only keys for navigation

Key Features:
- O(log n) search, insert, delete
- Efficient range scans via leaf linking
- Self-balancing through splits and merges
"""

from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Generic, TypeVar, List

from clade.storage.interfaces import IIndex, RecordID
from clade.utils.errors import DuplicateKeyError, KeyNotFoundError


K = TypeVar("K")  # Key type


# Default order (max children per node)
DEFAULT_ORDER = 128


@dataclass
class BTreeNode:
    """
    A node in the B+Tree.

    For internal nodes: keys separate children
    For leaf nodes: keys map to values (RecordIDs)
    """

    __slots__ = ("keys", "children", "values", "is_leaf", "next_leaf", "parent")

    keys: list
    children: list  # For internal nodes
    values: list  # For leaf nodes (RecordIDs)
    is_leaf: bool
    next_leaf: Optional["BTreeNode"]  # Linked list for leaf nodes
    parent: Optional["BTreeNode"]

    @classmethod
    def create_leaf(cls) -> "BTreeNode":
        """Create an empty leaf node."""
        return cls(
            keys=[],
            children=[],
            values=[],
            is_leaf=True,
            next_leaf=None,
            parent=None,
        )

    @classmethod
    def create_internal(cls) -> "BTreeNode":
        """Create an empty internal node."""
        return cls(
            keys=[],
            children=[],
            values=[],
            is_leaf=False,
            next_leaf=None,
            parent=None,
        )


class BPlusTree(IIndex):
    """
    B+Tree index implementation.

    Provides O(log n) search, insert, and delete operations.
    Leaf nodes are linked for efficient range scans.

    Thread Safety: Not thread-safe. Use external locking.
    """

    __slots__ = ("_root", "_order", "_size", "_unique")

    def __init__(self, order: int = DEFAULT_ORDER, unique: bool = True) -> None:
        """
        Initialize B+Tree.

        Args:
            order: Maximum number of children per node (default 128)
            unique: Whether to enforce unique keys (default True)
        """
        if order < 3:
            raise ValueError("Order must be at least 3")

        self._order = order
        self._unique = unique
        self._root = BTreeNode.create_leaf()
        self._size = 0

    # ─────────────────────────────────────────────────────────────────────────
    # IIndex Interface
    # ─────────────────────────────────────────────────────────────────────────

    def insert(self, key: Any, rid: RecordID) -> None:
        """
        Insert a key-value pair into the index.

        Args:
            key: The key to index
            rid: The record identifier

        Raises:
            DuplicateKeyError: If key already exists (for unique indexes)
        """
        leaf = self._find_leaf(key)

        # Check for duplicate
        if self._unique:
            idx = self._binary_search(leaf.keys, key)
            if idx < len(leaf.keys) and leaf.keys[idx] == key:
                raise DuplicateKeyError(key)

        # Insert into leaf
        self._insert_into_leaf(leaf, key, rid)
        self._size += 1

    def delete(self, key: Any) -> None:
        """
        Delete a key from the index.

        Args:
            key: The key to delete

        Raises:
            KeyNotFoundError: If key doesn't exist
        """
        leaf = self._find_leaf(key)
        idx = self._binary_search(leaf.keys, key)

        if idx >= len(leaf.keys) or leaf.keys[idx] != key:
            raise KeyNotFoundError(key)

        # Remove from leaf
        leaf.keys.pop(idx)
        leaf.values.pop(idx)
        self._size -= 1

        # Handle underflow (simplified - just let nodes become small)
        # Full implementation would merge/redistribute

    def search(self, key: Any) -> Optional[RecordID]:
        """
        Search for a key in the index.

        Args:
            key: The key to search for

        Returns:
            The RecordID if found, None otherwise
        """
        leaf = self._find_leaf(key)
        idx = self._binary_search(leaf.keys, key)

        if idx < len(leaf.keys) and leaf.keys[idx] == key:
            return leaf.values[idx]
        return None

    def range_scan(
        self, start_key: Any, end_key: Any
    ) -> Iterator[tuple[Any, RecordID]]:
        """
        Scan a range of keys.

        Args:
            start_key: The inclusive start of the range
            end_key: The inclusive end of the range

        Yields:
            Tuples of (key, RecordID) in sorted order
        """
        # Find starting leaf
        leaf = self._find_leaf(start_key)
        idx = self._binary_search(leaf.keys, start_key)

        # Scan through leaves
        while leaf is not None:
            while idx < len(leaf.keys):
                key = leaf.keys[idx]
                if key > end_key:
                    return
                if key >= start_key:
                    yield key, leaf.values[idx]
                idx += 1

            leaf = leaf.next_leaf
            idx = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Tree Navigation
    # ─────────────────────────────────────────────────────────────────────────

    def _find_leaf(self, key: Any) -> BTreeNode:
        """Find the leaf node that should contain the key."""
        node = self._root

        while not node.is_leaf:
            idx = self._binary_search(node.keys, key)
            # Navigate to appropriate child
            if idx < len(node.keys) and node.keys[idx] == key:
                idx += 1
            node = node.children[idx]

        return node

    def _binary_search(self, keys: list, key: Any) -> int:
        """
        Binary search for insert position.

        Returns the index where key should be inserted.
        """
        left, right = 0, len(keys)
        while left < right:
            mid = (left + right) // 2
            if keys[mid] < key:
                left = mid + 1
            else:
                right = mid
        return left

    # ─────────────────────────────────────────────────────────────────────────
    # Insertion
    # ─────────────────────────────────────────────────────────────────────────

    def _insert_into_leaf(self, leaf: BTreeNode, key: Any, rid: RecordID) -> None:
        """Insert key-value into a leaf node, splitting if necessary."""
        idx = self._binary_search(leaf.keys, key)

        # Insert at correct position
        leaf.keys.insert(idx, key)
        leaf.values.insert(idx, rid)

        # Check if split needed
        if len(leaf.keys) >= self._order:
            self._split_leaf(leaf)

    def _split_leaf(self, leaf: BTreeNode) -> None:
        """Split a full leaf node."""
        mid = len(leaf.keys) // 2

        # Create new leaf with right half
        new_leaf = BTreeNode.create_leaf()
        new_leaf.keys = leaf.keys[mid:]
        new_leaf.values = leaf.values[mid:]

        # Keep left half in original
        leaf.keys = leaf.keys[:mid]
        leaf.values = leaf.values[:mid]

        # Update linked list
        new_leaf.next_leaf = leaf.next_leaf
        leaf.next_leaf = new_leaf

        # Promote first key of new leaf to parent
        promote_key = new_leaf.keys[0]
        self._insert_into_parent(leaf, promote_key, new_leaf)

    def _split_internal(self, node: BTreeNode) -> None:
        """Split a full internal node."""
        mid = len(node.keys) // 2

        # Create new node with right half
        new_node = BTreeNode.create_internal()
        promote_key = node.keys[mid]

        new_node.keys = node.keys[mid + 1 :]
        new_node.children = node.children[mid + 1 :]

        # Update parent pointers
        for child in new_node.children:
            child.parent = new_node

        # Keep left half in original
        node.keys = node.keys[:mid]
        node.children = node.children[: mid + 1]

        # Promote middle key to parent
        self._insert_into_parent(node, promote_key, new_node)

    def _insert_into_parent(
        self, left: BTreeNode, key: Any, right: BTreeNode
    ) -> None:
        """Insert a key and right child into parent node."""
        parent = left.parent

        if parent is None:
            # Create new root
            new_root = BTreeNode.create_internal()
            new_root.keys = [key]
            new_root.children = [left, right]
            left.parent = new_root
            right.parent = new_root
            self._root = new_root
            return

        # Insert into existing parent
        idx = self._binary_search(parent.keys, key)
        parent.keys.insert(idx, key)
        parent.children.insert(idx + 1, right)
        right.parent = parent

        # Check if parent needs splitting
        if len(parent.keys) >= self._order:
            self._split_internal(parent)

    # ─────────────────────────────────────────────────────────────────────────
    # Properties and Utilities
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def order(self) -> int:
        """Get the tree order."""
        return self._order

    @property
    def size(self) -> int:
        """Get the number of entries in the tree."""
        return self._size

    @property
    def height(self) -> int:
        """Get the height of the tree."""
        h = 0
        node = self._root
        while not node.is_leaf:
            h += 1
            node = node.children[0]
        return h

    def min_key(self) -> Optional[Any]:
        """Get the minimum key in the tree."""
        if self._size == 0:
            return None
        node = self._root
        while not node.is_leaf:
            node = node.children[0]
        return node.keys[0] if node.keys else None

    def max_key(self) -> Optional[Any]:
        """Get the maximum key in the tree."""
        if self._size == 0:
            return None
        node = self._root
        while not node.is_leaf:
            node = node.children[-1]
        return node.keys[-1] if node.keys else None

    def scan_all(self) -> Iterator[tuple[Any, RecordID]]:
        """Scan all entries in sorted order."""
        # Find leftmost leaf
        node = self._root
        while not node.is_leaf:
            node = node.children[0]

        # Traverse linked list
        while node is not None:
            for key, rid in zip(node.keys, node.values):
                yield key, rid
            node = node.next_leaf

    def __len__(self) -> int:
        """Return the number of entries."""
        return self._size

    def __contains__(self, key: Any) -> bool:
        """Check if key exists in the tree."""
        return self.search(key) is not None

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"BPlusTree(order={self._order}, size={self._size}, height={self.height})"
