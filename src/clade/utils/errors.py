"""
Custom exception hierarchy for Clade Database Engine.

This module defines a comprehensive exception hierarchy following
the principle of failing fast with clear, actionable error messages.
All exceptions inherit from CladeDBError for easy catching.
"""


class CladeDBError(Exception):
    """Base exception for all Clade database errors."""

    pass


# Storage Layer Exceptions


class StorageError(CladeDBError):
    """Base exception for storage-related errors."""

    pass


class PageError(StorageError):
    """Exception related to page operations."""

    pass


class PageNotFoundError(PageError):
    """Raised when requested page does not exist."""

    def __init__(self, page_id: int):
        super().__init__(f"Page not found: page_id={page_id}")
        self.page_id = page_id


class PageFullError(PageError):
    """Raised when page has insufficient space for operation."""

    def __init__(self, page_id: int, required: int, available: int):
        super().__init__(
            f"Page {page_id} is full: required {required} bytes, "
            f"only {available} bytes available"
        )
        self.page_id = page_id
        self.required = required
        self.available = available


class PageCorruptionError(PageError):
    """Raised when page fails checksum validation."""

    def __init__(self, page_id: int, expected_checksum: int, actual_checksum: int):
        super().__init__(
            f"Page {page_id} is corrupted: expected checksum {expected_checksum}, "
            f"got {actual_checksum}"
        )
        self.page_id = page_id
        self.expected_checksum = expected_checksum
        self.actual_checksum = actual_checksum


class BufferPoolError(StorageError):
    """Exception related to buffer pool operations."""

    pass


class BufferPoolFullError(BufferPoolError):
    """Raised when buffer pool cannot evict pages."""

    def __init__(self, pool_size: int):
        super().__init__(f"Buffer pool is full (size={pool_size}) and all pages are pinned")
        self.pool_size = pool_size


class DiskIOError(StorageError):
    """Exception related to disk I/O operations."""

    def __init__(self, operation: str, path: str, reason: str):
        super().__init__(f"Disk I/O error during {operation} on {path}: {reason}")
        self.operation = operation
        self.path = path
        self.reason = reason


# Index Layer Exceptions


class IndexError(CladeDBError):
    """Base exception for index-related errors."""

    pass


class DuplicateKeyError(IndexError):
    """Raised when attempting to insert duplicate key in unique index."""

    def __init__(self, key: any):
        super().__init__(f"Duplicate key: {key}")
        self.key = key


class KeyNotFoundError(IndexError):
    """Raised when key is not found in index."""

    def __init__(self, key: any):
        super().__init__(f"Key not found: {key}")
        self.key = key


# Transaction Layer Exceptions


class TransactionError(CladeDBError):
    """Base exception for transaction-related errors."""

    pass


class TransactionAbortedError(TransactionError):
    """Raised when transaction is aborted."""

    def __init__(self, txn_id: int, reason: str):
        super().__init__(f"Transaction {txn_id} aborted: {reason}")
        self.txn_id = txn_id
        self.reason = reason


class DeadlockError(TransactionError):
    """Raised when deadlock is detected."""

    def __init__(self, txn_id: int, conflicting_txn_id: int):
        super().__init__(
            f"Deadlock detected: transaction {txn_id} "
            f"conflicts with transaction {conflicting_txn_id}"
        )
        self.txn_id = txn_id
        self.conflicting_txn_id = conflicting_txn_id


class WriteWriteConflictError(TransactionError):
    """Raised when write-write conflict is detected in MVCC."""

    def __init__(self, txn_id: int, conflicting_txn_id: int, record_id: str):
        super().__init__(
            f"Write-write conflict: transaction {txn_id} "
            f"conflicts with transaction {conflicting_txn_id} on record {record_id}"
        )
        self.txn_id = txn_id
        self.conflicting_txn_id = conflicting_txn_id
        self.record_id = record_id


# WAL Layer Exceptions


class WALError(CladeDBError):
    """Base exception for write-ahead log errors."""

    pass


class WALCorruptionError(WALError):
    """Raised when WAL is corrupted and cannot be recovered."""

    def __init__(self, lsn: int, reason: str):
        super().__init__(f"WAL corruption at LSN {lsn}: {reason}")
        self.lsn = lsn
        self.reason = reason


class RecoveryError(WALError):
    """Raised when recovery process fails."""

    def __init__(self, phase: str, reason: str):
        super().__init__(f"Recovery failed during {phase} phase: {reason}")
        self.phase = phase
        self.reason = reason


# Query Layer Exceptions


class QueryError(CladeDBError):
    """Base exception for query-related errors."""

    pass


class ParseError(QueryError):
    """Raised when SQL parsing fails."""

    def __init__(self, sql: str, position: int, reason: str):
        super().__init__(f"Parse error at position {position}: {reason}\nSQL: {sql}")
        self.sql = sql
        self.position = position
        self.reason = reason


class PlanError(QueryError):
    """Raised when query planning fails."""

    def __init__(self, reason: str):
        super().__init__(f"Query planning failed: {reason}")
        self.reason = reason


class ExecutionError(QueryError):
    """Raised when query execution fails."""

    def __init__(self, reason: str):
        super().__init__(f"Query execution failed: {reason}")
        self.reason = reason


# Catalog Layer Exceptions


class CatalogError(CladeDBError):
    """Base exception for catalog-related errors."""

    pass


class TableNotFoundError(CatalogError):
    """Raised when table does not exist."""

    def __init__(self, table_name: str):
        super().__init__(f"Table not found: {table_name}")
        self.table_name = table_name


class ColumnNotFoundError(CatalogError):
    """Raised when column does not exist."""

    def __init__(self, table_name: str, column_name: str):
        super().__init__(f"Column '{column_name}' not found in table '{table_name}'")
        self.table_name = table_name
        self.column_name = column_name


class SchemaError(CatalogError):
    """Raised when schema validation fails."""

    def __init__(self, reason: str):
        super().__init__(f"Schema error: {reason}")
        self.reason = reason


# Validation Exceptions


class ValidationError(CladeDBError):
    """Raised when input validation fails."""

    pass


class InvalidArgumentError(ValidationError):
    """Raised when function argument is invalid."""

    def __init__(self, argument_name: str, value: any, reason: str):
        super().__init__(f"Invalid argument '{argument_name}={value}': {reason}")
        self.argument_name = argument_name
        self.value = value
        self.reason = reason


class ConfigurationError(ValidationError):
    """Raised when configuration is invalid."""

    def __init__(self, key: str, value: any, reason: str):
        super().__init__(f"Invalid configuration '{key}={value}': {reason}")
        self.key = key
        self.value = value
        self.reason = reason
