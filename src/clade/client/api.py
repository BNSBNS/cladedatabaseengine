"""
Public API for Clade Database Engine.

Provides a high-level interface for database operations.

Usage:
    from clade.client.api import Database

    db = Database("my_database")
    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name VARCHAR(100))")
    db.execute("INSERT INTO users (id, name) VALUES (1, 'Alice')")

    for row in db.query("SELECT * FROM users"):
        print(row)

    db.close()
"""

from pathlib import Path
from typing import Iterator, Dict, Any, Optional, List
import threading
import tempfile
import shutil

from clade.catalog.schema import Catalog, Table, Column, DataType
from clade.storage.file_manager import FileManager
from clade.storage.buffer_manager import BufferPoolManager
from clade.storage.heap import HeapFile
from clade.query.parser import SQLParser
from clade.query.planner import QueryPlanner, PlanNodeType
from clade.query.optimizer import Optimizer
from clade.execution.volcano import ExecutionEngine
from clade.transaction.mvcc import MVCCManager
from clade.wal.logger import WALManager
from clade.wal.recovery import RecoveryManager
from clade.observability.metrics import MetricsCollector, DatabaseMetrics


class Connection:
    """
    A database connection.

    Manages a session with the database including transactions.
    """

    __slots__ = (
        "_db", "_txn_id", "_autocommit", "_closed", "_lock"
    )

    def __init__(self, db: "Database", autocommit: bool = True) -> None:
        self._db = db
        self._txn_id: Optional[int] = None
        self._autocommit = autocommit
        self._closed = False
        self._lock = threading.Lock()

    def execute(self, sql: str) -> int:
        """
        Execute a DML statement.

        Args:
            sql: SQL statement (INSERT, UPDATE, DELETE)

        Returns:
            Number of affected rows
        """
        self._check_open()
        return self._db._execute_dml(sql, self._txn_id)

    def query(self, sql: str) -> Iterator[Dict[str, Any]]:
        """
        Execute a query and return results.

        Args:
            sql: SQL SELECT statement

        Yields:
            Row dictionaries
        """
        self._check_open()
        yield from self._db._execute_query(sql, self._txn_id)

    def begin(self) -> None:
        """Begin a transaction."""
        self._check_open()
        with self._lock:
            if self._txn_id is not None:
                raise RuntimeError("Transaction already in progress")
            self._txn_id = self._db._begin_transaction()

    def commit(self) -> None:
        """Commit the current transaction."""
        self._check_open()
        with self._lock:
            if self._txn_id is None:
                raise RuntimeError("No transaction in progress")
            self._db._commit_transaction(self._txn_id)
            self._txn_id = None

    def rollback(self) -> None:
        """Rollback the current transaction."""
        self._check_open()
        with self._lock:
            if self._txn_id is None:
                raise RuntimeError("No transaction in progress")
            self._db._rollback_transaction(self._txn_id)
            self._txn_id = None

    def close(self) -> None:
        """Close the connection."""
        with self._lock:
            if self._txn_id is not None:
                self._db._rollback_transaction(self._txn_id)
                self._txn_id = None
            self._closed = True

    def _check_open(self) -> None:
        """Check that connection is open."""
        if self._closed:
            raise RuntimeError("Connection is closed")

    @property
    def in_transaction(self) -> bool:
        """Check if in a transaction."""
        return self._txn_id is not None

    def __enter__(self) -> "Connection":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class Database:
    """
    Main database interface.

    Manages tables, executes queries, and coordinates transactions.
    """

    __slots__ = (
        "_name", "_path", "_catalog", "_heaps", "_file_managers",
        "_buffer_managers", "_parser", "_planner", "_optimizer",
        "_engine", "_mvcc", "_wal", "_recovery", "_metrics_collector",
        "_metrics", "_lock", "_closed", "_temp_dir", "_pool_size"
    )

    def __init__(
        self,
        name: str,
        path: Optional[str] = None,
        pool_size: int = 100,
        enable_wal: bool = True,
        enable_metrics: bool = True,
    ) -> None:
        """
        Create or open a database.

        Args:
            name: Database name
            path: Directory path (default: temp directory)
            pool_size: Buffer pool size per table
            enable_wal: Enable write-ahead logging
            enable_metrics: Enable metrics collection
        """
        self._name = name
        self._temp_dir: Optional[str] = None

        if path is None:
            self._temp_dir = tempfile.mkdtemp(prefix=f"clade_{name}_")
            self._path = Path(self._temp_dir)
        else:
            self._path = Path(path)
            self._path.mkdir(parents=True, exist_ok=True)

        # Core components
        self._catalog = Catalog()
        self._heaps: Dict[str, HeapFile] = {}
        self._file_managers: Dict[str, FileManager] = {}
        self._buffer_managers: Dict[str, BufferPoolManager] = {}

        # Query processing
        self._parser = SQLParser()
        self._planner = QueryPlanner(self._catalog)
        self._optimizer = Optimizer()
        self._engine: Optional[ExecutionEngine] = None

        # Transaction management
        self._mvcc = MVCCManager()

        # WAL
        if enable_wal:
            wal_path = self._path / "wal"
            wal_path.mkdir(exist_ok=True)
            self._wal = WALManager(str(wal_path))
            self._recovery = RecoveryManager(self._wal)
        else:
            self._wal = None
            self._recovery = None

        # Metrics
        if enable_metrics:
            self._metrics_collector = MetricsCollector()
            self._metrics = DatabaseMetrics(self._metrics_collector)
        else:
            self._metrics_collector = None
            self._metrics = None

        self._lock = threading.RLock()
        self._closed = False

        # Create pool_size attribute for heap creation
        self._pool_size = pool_size

    def execute(self, sql: str) -> int:
        """
        Execute a DDL or DML statement.

        Args:
            sql: SQL statement

        Returns:
            Number of affected rows (0 for DDL)
        """
        self._check_open()

        stmt = self._parser.parse(sql)
        plan = self._planner.plan(stmt)

        if plan.node_type == PlanNodeType.CREATE_TABLE:
            return self._execute_create_table(plan)
        elif plan.node_type == PlanNodeType.DROP_TABLE:
            return self._execute_drop_table(plan)
        else:
            return self._execute_dml(sql, None)

    def query(self, sql: str) -> Iterator[Dict[str, Any]]:
        """
        Execute a query and return results.

        Args:
            sql: SQL SELECT statement

        Yields:
            Row dictionaries
        """
        self._check_open()
        yield from self._execute_query(sql, None)

    def connect(self, autocommit: bool = True) -> Connection:
        """
        Get a connection to the database.

        Args:
            autocommit: Auto-commit each statement

        Returns:
            Connection object
        """
        self._check_open()
        return Connection(self, autocommit)

    def _execute_create_table(self, plan) -> int:
        """Execute CREATE TABLE."""
        with self._lock:
            table_def = plan.table_def

            # Build columns - handle dict format from parser
            columns = []
            for col_def in table_def.columns:
                if isinstance(col_def, dict):
                    dtype = self._parse_data_type(col_def.get("type", "VARCHAR"))
                    col = Column(
                        name=col_def.get("name", ""),
                        data_type=dtype,
                        nullable=col_def.get("nullable", True),
                        primary_key=col_def.get("pk", False),
                        max_length=col_def.get("max_length"),
                    )
                else:
                    dtype = self._parse_data_type(col_def.data_type)
                    col = Column(
                        name=col_def.name,
                        data_type=dtype,
                        nullable=col_def.nullable,
                        primary_key=col_def.primary_key,
                        max_length=col_def.length,
                    )
                columns.append(col)

            # Determine primary key
            pk_cols = [c.name for c in columns if c.primary_key]

            table = Table(
                name=table_def.table,
                columns=columns,
                primary_key=pk_cols or None,
            )

            self._catalog.create_table(table)
            self._create_heap_for_table(table_def.table)

            return 0

    def _execute_drop_table(self, plan) -> int:
        """Execute DROP TABLE."""
        with self._lock:
            table_name = plan.table_name

            if plan.if_exists and table_name not in self._catalog._tables:
                return 0

            self._catalog.drop_table(table_name)
            self._close_heap_for_table(table_name)

            return 0

    def _execute_dml(self, sql: str, txn_id: Optional[int]) -> int:
        """Execute DML statement."""
        import time
        start = time.perf_counter()

        with self._lock:
            stmt = self._parser.parse(sql)
            plan = self._planner.plan(stmt)
            plan = self._optimizer.optimize(plan)

            self._ensure_engine()
            count = self._engine.execute_dml(plan)

            duration = time.perf_counter() - start
            if self._metrics:
                query_type = plan.node_type.name.lower()
                self._metrics.record_query(query_type, duration, count)

            return count

    def _execute_query(self, sql: str, txn_id: Optional[int]) -> Iterator[Dict[str, Any]]:
        """Execute query and return results."""
        import time
        start = time.perf_counter()

        with self._lock:
            stmt = self._parser.parse(sql)
            plan = self._planner.plan(stmt)
            plan = self._optimizer.optimize(plan)

            self._ensure_engine()
            rows = list(self._engine.execute(plan))

            duration = time.perf_counter() - start
            if self._metrics:
                self._metrics.record_query("select", duration, len(rows))

        # Yield outside lock to avoid holding it during iteration
        yield from rows

    def _begin_transaction(self) -> int:
        """Begin a new transaction."""
        return self._mvcc.begin_transaction()

    def _commit_transaction(self, txn_id: int) -> None:
        """Commit a transaction."""
        import time
        start = time.perf_counter()

        self._mvcc.commit_transaction(txn_id)

        if self._wal:
            self._wal.log_commit(txn_id)

        duration = time.perf_counter() - start
        if self._metrics:
            self._metrics.record_transaction("commit", duration)

    def _rollback_transaction(self, txn_id: int) -> None:
        """Rollback a transaction."""
        import time
        start = time.perf_counter()

        self._mvcc.abort_transaction(txn_id)

        if self._wal:
            self._wal.log_abort(txn_id)

        duration = time.perf_counter() - start
        if self._metrics:
            self._metrics.record_transaction("abort", duration)

    def _create_heap_for_table(self, table_name: str) -> None:
        """Create heap file for a table."""
        file_path = self._path / f"{table_name}.db"
        fm = FileManager(file_path)
        bp = BufferPoolManager(fm, pool_size=self._pool_size)
        heap = HeapFile(bp)

        self._file_managers[table_name] = fm
        self._buffer_managers[table_name] = bp
        self._heaps[table_name] = heap

    def _close_heap_for_table(self, table_name: str) -> None:
        """Close and remove heap file for a table."""
        if table_name in self._heaps:
            # HeapFile doesn't have close, just remove from dict
            del self._heaps[table_name]

        if table_name in self._buffer_managers:
            del self._buffer_managers[table_name]

        if table_name in self._file_managers:
            self._file_managers[table_name].close()
            del self._file_managers[table_name]

        # Remove file
        file_path = self._path / f"{table_name}.db"
        if file_path.exists():
            file_path.unlink()

    def _ensure_engine(self) -> None:
        """Ensure execution engine is initialized."""
        if self._engine is None or self._engine._heaps != self._heaps:
            self._engine = ExecutionEngine(self._catalog, self._heaps)

    def _parse_data_type(self, type_val: Any) -> DataType:
        """Parse data type string or enum to DataType enum."""
        # Already a DataType enum
        if isinstance(type_val, DataType):
            return type_val

        # String type
        type_upper = str(type_val).upper()
        if type_upper.startswith("VARCHAR"):
            return DataType.VARCHAR
        elif type_upper.startswith("CHAR"):
            return DataType.CHAR

        type_map = {
            "INTEGER": DataType.INTEGER,
            "INT": DataType.INTEGER,
            "BIGINT": DataType.BIGINT,
            "SMALLINT": DataType.SMALLINT,
            "FLOAT": DataType.FLOAT,
            "DOUBLE": DataType.FLOAT,
            "REAL": DataType.FLOAT,
            "BOOLEAN": DataType.BOOLEAN,
            "BOOL": DataType.BOOLEAN,
            "TEXT": DataType.TEXT,
            "BLOB": DataType.BLOB,
            "DATE": DataType.DATE,
            "TIMESTAMP": DataType.TIMESTAMP,
            "DECIMAL": DataType.DECIMAL,
        }
        return type_map.get(type_upper, DataType.VARCHAR)

    def _check_open(self) -> None:
        """Check that database is open."""
        if self._closed:
            raise RuntimeError("Database is closed")

    @property
    def tables(self) -> List[str]:
        """Get list of table names."""
        return list(self._catalog._tables.keys())

    @property
    def metrics(self) -> Optional[DatabaseMetrics]:
        """Get metrics collector."""
        return self._metrics

    def export_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        if self._metrics_collector:
            return self._metrics_collector.export_prometheus()
        return ""

    def checkpoint(self) -> None:
        """Create a checkpoint."""
        self._check_open()

        with self._lock:
            # Flush all dirty buffer pool pages
            for table_name, bp in self._buffer_managers.items():
                bp.flush_all_dirty()

            # Log checkpoint
            if self._wal:
                self._wal.checkpoint()

    def close(self) -> None:
        """Close the database."""
        with self._lock:
            if self._closed:
                return

            # Checkpoint before closing
            try:
                self.checkpoint()
            except Exception:
                pass

            # Clear heaps (HeapFile has no close method)
            self._heaps.clear()

            # Close file managers
            for fm in self._file_managers.values():
                try:
                    fm.close()
                except Exception:
                    pass
            self._file_managers.clear()
            self._buffer_managers.clear()

            # Close WAL
            if self._wal:
                try:
                    self._wal.close()
                except Exception:
                    pass

            # Clean up temp directory
            if self._temp_dir:
                try:
                    shutil.rmtree(self._temp_dir)
                except Exception:
                    pass

            self._closed = True

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"Database(name={self._name!r}, tables={len(self._catalog._tables)})"


# Convenience function
def open_database(
    name: str,
    path: Optional[str] = None,
    **kwargs
) -> Database:
    """
    Open or create a database.

    Args:
        name: Database name
        path: Directory path (default: temp directory)
        **kwargs: Additional options passed to Database

    Returns:
        Database instance
    """
    return Database(name, path, **kwargs)
