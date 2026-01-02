"""
Global timestamp manager for transaction ordering.

Provides globally unique, monotonically increasing timestamps
for transaction ordering in MVCC.

Thread Safety: Thread-safe via atomic operations.
"""

import threading
from typing import Set


class TimestampManager:
    """
    Manages global timestamps for transaction ordering.

    Uses a simple counter-based approach for timestamp generation.
    In a distributed system, this would need to be a hybrid logical clock
    or similar mechanism.

    Thread Safety: Thread-safe.
    """

    __slots__ = ("_current_ts", "_lock")

    def __init__(self, initial_ts: int = 0) -> None:
        """
        Initialize timestamp manager.

        Args:
            initial_ts: Initial timestamp value (default 0)
        """
        self._current_ts = initial_ts
        self._lock = threading.Lock()

    def next_timestamp(self) -> int:
        """
        Get the next timestamp.

        Returns:
            A globally unique, monotonically increasing timestamp
        """
        with self._lock:
            self._current_ts += 1
            return self._current_ts

    def current_timestamp(self) -> int:
        """
        Get the current timestamp (without incrementing).

        Returns:
            The current timestamp value
        """
        return self._current_ts

    def __repr__(self) -> str:
        """String representation."""
        return f"TimestampManager(current={self._current_ts})"
