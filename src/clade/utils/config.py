"""
Configuration management for Clade Database Engine.

This module provides a centralized configuration system with
validation and sensible defaults. Uses dataclasses for type safety.
"""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from clade.utils.errors import ConfigurationError


@dataclass
class StorageConfig:
    """Configuration for storage engine."""

    # Page settings
    page_size: int = 8192  # 8KB pages
    max_page_size: int = 16384  # 16KB maximum

    # Buffer pool settings
    buffer_pool_size: int = 1024  # Number of pages in buffer pool
    buffer_pool_eviction_policy: str = "clock"  # clock, lru, arc

    # File I/O settings
    use_mmap: bool = True  # Use memory-mapped I/O
    use_direct_io: bool = False  # Use direct I/O (bypasses OS cache)
    sync_on_commit: bool = True  # fsync on commit

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.page_size <= 0 or self.page_size > self.max_page_size:
            raise ConfigurationError(
                "page_size",
                self.page_size,
                f"must be > 0 and <= {self.max_page_size}",
            )

        if self.buffer_pool_size <= 0:
            raise ConfigurationError(
                "buffer_pool_size",
                self.buffer_pool_size,
                "must be positive",
            )

        valid_policies = {"clock", "lru", "arc"}
        if self.buffer_pool_eviction_policy not in valid_policies:
            raise ConfigurationError(
                "buffer_pool_eviction_policy",
                self.buffer_pool_eviction_policy,
                f"must be one of {valid_policies}",
            )


@dataclass
class TransactionConfig:
    """Configuration for transaction management."""

    # MVCC settings
    enable_mvcc: bool = True
    isolation_level: str = "snapshot"  # snapshot, serializable, repeatable_read

    # Locking settings
    lock_timeout_ms: int = 5000  # 5 seconds
    deadlock_detection_enabled: bool = True

    # Vacuum settings
    auto_vacuum: bool = True
    vacuum_threshold: int = 1000  # Vacuum after this many deleted versions

    def __post_init__(self) -> None:
        """Validate configuration."""
        valid_isolation = {"snapshot", "serializable", "repeatable_read", "read_committed"}
        if self.isolation_level not in valid_isolation:
            raise ConfigurationError(
                "isolation_level",
                self.isolation_level,
                f"must be one of {valid_isolation}",
            )

        if self.lock_timeout_ms < 0:
            raise ConfigurationError(
                "lock_timeout_ms",
                self.lock_timeout_ms,
                "must be non-negative",
            )


@dataclass
class WALConfig:
    """Configuration for write-ahead logging."""

    # WAL file settings
    wal_segment_size: int = 16 * 1024 * 1024  # 16MB segments
    wal_buffer_size: int = 64 * 1024  # 64KB buffer

    # Checkpoint settings
    checkpoint_interval_seconds: int = 300  # 5 minutes
    checkpoint_on_shutdown: bool = True

    # Group commit settings
    group_commit_enabled: bool = True
    group_commit_max_wait_ms: int = 10  # Wait up to 10ms for batching

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.wal_segment_size <= 0:
            raise ConfigurationError(
                "wal_segment_size",
                self.wal_segment_size,
                "must be positive",
            )

        if self.checkpoint_interval_seconds < 0:
            raise ConfigurationError(
                "checkpoint_interval_seconds",
                self.checkpoint_interval_seconds,
                "must be non-negative",
            )


@dataclass
class QueryConfig:
    """Configuration for query processing."""

    # Parser settings
    max_query_length: int = 1024 * 1024  # 1MB

    # Optimizer settings
    enable_cost_based_optimization: bool = True
    join_reorder_limit: int = 8  # Dynamic programming for <= 8 tables

    # Execution settings
    use_vectorized_execution: bool = True  # For OLAP queries
    vectorization_batch_size: int = 1024
    enable_jit_compilation: bool = True  # Use Numba JIT

    # Memory limits
    max_memory_per_query_mb: int = 256  # 256MB per query
    spill_to_disk_enabled: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.max_query_length <= 0:
            raise ConfigurationError(
                "max_query_length",
                self.max_query_length,
                "must be positive",
            )

        if self.vectorization_batch_size <= 0:
            raise ConfigurationError(
                "vectorization_batch_size",
                self.vectorization_batch_size,
                "must be positive",
            )


@dataclass
class ObservabilityConfig:
    """Configuration for observability features."""

    # Metrics settings
    enable_metrics: bool = True
    metrics_port: int = 9090

    # Logging settings
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    log_format: str = "json"  # json, text
    log_file: Optional[str] = None  # None means stdout

    # Tracing settings
    enable_query_tracing: bool = True
    trace_slow_queries_ms: int = 1000  # Trace queries slower than 1s

    def __post_init__(self) -> None:
        """Validate configuration."""
        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level not in valid_log_levels:
            raise ConfigurationError(
                "log_level",
                self.log_level,
                f"must be one of {valid_log_levels}",
            )

        valid_formats = {"json", "text"}
        if self.log_format not in valid_formats:
            raise ConfigurationError(
                "log_format",
                self.log_format,
                f"must be one of {valid_formats}",
            )


@dataclass
class DatabaseConfig:
    """Master configuration for the database."""

    # Database path
    db_path: str = "./cladedb"

    # Component configurations
    storage: StorageConfig = field(default_factory=StorageConfig)
    transaction: TransactionConfig = field(default_factory=TransactionConfig)
    wal: WALConfig = field(default_factory=WALConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)

    def __post_init__(self) -> None:
        """Validate configuration and create db_path if needed."""
        # Validate db_path
        if not self.db_path:
            raise ConfigurationError("db_path", self.db_path, "cannot be empty")

        # Create directory if it doesn't exist
        Path(self.db_path).mkdir(parents=True, exist_ok=True)

    @classmethod
    def default(cls) -> "DatabaseConfig":
        """Create configuration with default values."""
        return cls()

    @classmethod
    def for_testing(cls) -> "DatabaseConfig":
        """Create configuration optimized for testing."""
        return cls(
            db_path="./test_cladedb",
            storage=StorageConfig(
                buffer_pool_size=64,  # Smaller for tests
                use_mmap=False,  # More predictable for tests
            ),
            transaction=TransactionConfig(
                lock_timeout_ms=1000,  # Faster timeout
            ),
            wal=WALConfig(
                checkpoint_interval_seconds=10,  # More frequent
                group_commit_enabled=False,  # Simpler for tests
            ),
            observability=ObservabilityConfig(
                log_level="DEBUG",
                enable_metrics=False,  # Less overhead
            ),
        )

    @classmethod
    def for_oltp(cls) -> "DatabaseConfig":
        """Create configuration optimized for OLTP workloads."""
        return cls(
            storage=StorageConfig(
                buffer_pool_size=2048,  # Larger cache
                buffer_pool_eviction_policy="lru",
            ),
            transaction=TransactionConfig(
                enable_mvcc=True,
                isolation_level="snapshot",
            ),
            query=QueryConfig(
                use_vectorized_execution=False,  # Volcano model for OLTP
                enable_jit_compilation=False,
            ),
        )

    @classmethod
    def for_olap(cls) -> "DatabaseConfig":
        """Create configuration optimized for OLAP workloads."""
        return cls(
            storage=StorageConfig(
                buffer_pool_size=4096,  # Very large cache
                buffer_pool_eviction_policy="arc",  # Better for scans
            ),
            query=QueryConfig(
                use_vectorized_execution=True,
                vectorization_batch_size=2048,  # Larger batches
                enable_jit_compilation=True,
                max_memory_per_query_mb=1024,  # More memory for aggregations
            ),
        )
