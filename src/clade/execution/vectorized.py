"""
Vectorized Execution Engine for OLAP workloads.

Processes data in batches (vectors) rather than one row at a time.
Uses NumPy for SIMD-friendly operations on columnar data.

Key Features:
- Batch processing (1024 rows default)
- NumPy-based operations
- Late materialization
- Predicate evaluation on compressed data
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Iterator, Tuple
import numpy as np

from clade.catalog.schema import DataType
from clade.storage.column_store import ColumnStore, CompressionType
from clade.query.parser import Expr


# Default batch size
VECTOR_SIZE = 1024


class Vector:
    """
    A batch of values for vectorized processing.

    Wraps a NumPy array with null tracking.
    """

    __slots__ = ("_data", "_nulls", "_dtype")

    def __init__(
        self,
        data: np.ndarray,
        nulls: Optional[np.ndarray] = None,
        dtype: DataType = DataType.NULL,
    ) -> None:
        self._data = data
        self._nulls = nulls if nulls is not None else np.zeros(len(data), dtype=bool)
        self._dtype = dtype

    @classmethod
    def from_list(cls, values: List[Any], dtype: DataType) -> "Vector":
        """Create vector from Python list."""
        if dtype == DataType.INTEGER:
            np_dtype = np.int32
        elif dtype == DataType.BIGINT:
            np_dtype = np.int64
        elif dtype == DataType.FLOAT:
            np_dtype = np.float64
        elif dtype == DataType.BOOLEAN:
            np_dtype = bool
        else:
            np_dtype = object

        nulls = np.array([v is None for v in values], dtype=bool)
        data = np.array([v if v is not None else 0 for v in values], dtype=np_dtype)

        return cls(data, nulls, dtype)

    @property
    def data(self) -> np.ndarray:
        """Get underlying data array."""
        return self._data

    @property
    def nulls(self) -> np.ndarray:
        """Get null mask."""
        return self._nulls

    @property
    def size(self) -> int:
        """Get vector size."""
        return len(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Any:
        if self._nulls[idx]:
            return None
        return self._data[idx]

    def filter(self, mask: np.ndarray) -> "Vector":
        """Filter values by boolean mask."""
        return Vector(
            self._data[mask],
            self._nulls[mask],
            self._dtype,
        )

    def to_list(self) -> List[Any]:
        """Convert to Python list."""
        return [None if self._nulls[i] else self._data[i] for i in range(len(self._data))]


class VectorBatch:
    """
    A batch of vectors (multiple columns).

    Represents a horizontal slice of data.
    """

    __slots__ = ("_columns", "_size")

    def __init__(self) -> None:
        self._columns: Dict[str, Vector] = {}
        self._size = 0

    def add_column(self, name: str, vector: Vector) -> None:
        """Add a column vector."""
        self._columns[name] = vector
        self._size = vector.size

    def get_column(self, name: str) -> Optional[Vector]:
        """Get column vector by name."""
        return self._columns.get(name)

    def filter(self, mask: np.ndarray) -> "VectorBatch":
        """Filter all columns by mask."""
        result = VectorBatch()
        for name, vector in self._columns.items():
            result.add_column(name, vector.filter(mask))
        return result

    @property
    def size(self) -> int:
        """Get batch size."""
        return self._size

    @property
    def column_names(self) -> List[str]:
        """Get column names."""
        return list(self._columns.keys())

    def to_dicts(self) -> Iterator[Dict[str, Any]]:
        """Convert to row dictionaries."""
        for i in range(self._size):
            yield {name: vec[i] for name, vec in self._columns.items()}


class VectorizedOperator(ABC):
    """Base class for vectorized operators."""

    __slots__ = ("_output_columns",)

    def __init__(self) -> None:
        self._output_columns: List[str] = []

    @abstractmethod
    def open(self) -> None:
        """Initialize the operator."""
        pass

    @abstractmethod
    def next_batch(self) -> Optional[VectorBatch]:
        """Return next batch of data."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up resources."""
        pass


class ColumnScanOperator(VectorizedOperator):
    """Scan columns from column store."""

    __slots__ = ("_store", "_columns", "_iterator", "_batch_size", "_data", "_offset", "_total")

    def __init__(
        self,
        store: ColumnStore,
        columns: List[str],
        batch_size: int = VECTOR_SIZE,
    ) -> None:
        super().__init__()
        self._store = store
        self._columns = columns
        self._batch_size = batch_size
        self._iterator = None
        self._data = None
        self._offset = 0
        self._total = 0

    def open(self) -> None:
        # Get all data as numpy arrays
        self._data = self._store.scan_columns_vectorized(self._columns)
        self._offset = 0
        self._total = self._store.row_count

    def next_batch(self) -> Optional[VectorBatch]:
        if self._offset >= self._total:
            return None

        batch = VectorBatch()
        end = min(self._offset + self._batch_size, self._total)

        for name in self._columns:
            if name in self._data:
                arr = self._data[name][self._offset:end]
                nulls = np.array([v is None if isinstance(v, type(None)) else False for v in arr])
                vector = Vector(arr, nulls)
                batch.add_column(name, vector)

        self._offset = end
        return batch

    def close(self) -> None:
        self._data = None


class FilterOperator(VectorizedOperator):
    """Vectorized filter operator."""

    __slots__ = ("_child", "_predicate")

    def __init__(self, child: VectorizedOperator, predicate: Expr) -> None:
        super().__init__()
        self._child = child
        self._predicate = predicate

    def open(self) -> None:
        self._child.open()

    def next_batch(self) -> Optional[VectorBatch]:
        batch = self._child.next_batch()
        if batch is None:
            return None

        # Evaluate predicate on batch
        mask = VectorizedEvaluator.evaluate(self._predicate, batch)

        # Filter batch
        return batch.filter(mask)

    def close(self) -> None:
        self._child.close()


class ProjectOperator(VectorizedOperator):
    """Vectorized projection operator."""

    __slots__ = ("_child", "_expressions", "_aliases")

    def __init__(
        self,
        child: VectorizedOperator,
        expressions: List[Expr],
        aliases: List[Optional[str]],
    ) -> None:
        super().__init__()
        self._child = child
        self._expressions = expressions
        self._aliases = aliases

    def open(self) -> None:
        self._child.open()

    def next_batch(self) -> Optional[VectorBatch]:
        batch = self._child.next_batch()
        if batch is None:
            return None

        result = VectorBatch()

        for expr, alias in zip(self._expressions, self._aliases):
            name = alias or self._get_expr_name(expr)
            vector = VectorizedEvaluator.evaluate_to_vector(expr, batch)
            result.add_column(name, vector)

        return result

    def _get_expr_name(self, expr: Expr) -> str:
        if expr.op == "column":
            return expr.value[1]
        elif expr.op == "function":
            return expr.value
        return "expr"

    def close(self) -> None:
        self._child.close()


class AggregateOperator(VectorizedOperator):
    """Vectorized aggregation operator."""

    __slots__ = ("_child", "_group_by", "_aggregates", "_result", "_returned")

    def __init__(
        self,
        child: VectorizedOperator,
        group_by: List[str],
        aggregates: List[Tuple[str, str, str]],  # (func, column, alias)
    ) -> None:
        super().__init__()
        self._child = child
        self._group_by = group_by
        self._aggregates = aggregates
        self._result = None

    def open(self) -> None:
        self._child.open()

        # Collect all data
        all_data: Dict[str, List] = {g: [] for g in self._group_by}
        for func, col, _ in self._aggregates:
            all_data[col] = []

        while True:
            batch = self._child.next_batch()
            if batch is None:
                break

            for name in all_data:
                vec = batch.get_column(name)
                if vec:
                    all_data[name].extend(vec.to_list())

        # Convert to numpy
        np_data = {k: np.array(v) for k, v in all_data.items()}

        # Compute aggregates
        if self._group_by:
            # Group by aggregation
            self._result = self._aggregate_grouped(np_data)
        else:
            # Global aggregation
            self._result = self._aggregate_global(np_data)

        self._returned = False

    def _aggregate_global(self, data: Dict[str, np.ndarray]) -> VectorBatch:
        """Compute global aggregates."""
        batch = VectorBatch()

        for func, col, alias in self._aggregates:
            arr = data.get(col, np.array([]))
            value = self._compute_aggregate(func, arr)
            vector = Vector(np.array([value]), np.array([False]))
            batch.add_column(alias or f"{func}({col})", vector)

        return batch

    def _aggregate_grouped(self, data: Dict[str, np.ndarray]) -> VectorBatch:
        """Compute grouped aggregates."""
        # Simple implementation - use numpy unique for grouping
        if not self._group_by:
            return self._aggregate_global(data)

        group_col = self._group_by[0]
        group_arr = data.get(group_col, np.array([]))
        unique_groups = np.unique(group_arr)

        batch = VectorBatch()

        # Add group column
        batch.add_column(group_col, Vector(unique_groups, np.zeros(len(unique_groups), dtype=bool)))

        # Compute aggregates for each group
        for func, col, alias in self._aggregates:
            arr = data.get(col, np.array([]))
            values = []
            for g in unique_groups:
                mask = group_arr == g
                values.append(self._compute_aggregate(func, arr[mask]))
            vector = Vector(np.array(values), np.zeros(len(values), dtype=bool))
            batch.add_column(alias or f"{func}({col})", vector)

        return batch

    def _compute_aggregate(self, func: str, arr: np.ndarray) -> Any:
        """Compute a single aggregate value."""
        func = func.upper()
        if len(arr) == 0:
            return None

        if func == "COUNT":
            return len(arr)
        elif func == "SUM":
            return np.sum(arr)
        elif func == "AVG":
            return np.mean(arr)
        elif func == "MIN":
            return np.min(arr)
        elif func == "MAX":
            return np.max(arr)
        else:
            return None

    def next_batch(self) -> Optional[VectorBatch]:
        if self._returned:
            return None
        self._returned = True
        return self._result

    def close(self) -> None:
        self._child.close()


class VectorizedEvaluator:
    """Evaluate expressions on vector batches."""

    @classmethod
    def evaluate(cls, expr: Expr, batch: VectorBatch) -> np.ndarray:
        """
        Evaluate expression to boolean mask.

        Args:
            expr: Expression to evaluate
            batch: Input batch

        Returns:
            Boolean mask array
        """
        if expr.op == "literal":
            return np.full(batch.size, bool(expr.value))

        elif expr.op == "column":
            _, col_name = expr.value
            vec = cls._find_column(batch, col_name)
            if vec is None:
                return np.full(batch.size, False)
            return vec.data.astype(bool) & ~vec.nulls

        elif expr.op == "binary":
            left = cls.evaluate_to_array(expr.operands[0], batch)
            right = cls.evaluate_to_array(expr.operands[1], batch)
            op = expr.value

            if op == "=":
                return left == right
            elif op == "!=":
                return left != right
            elif op == "<":
                return left < right
            elif op == ">":
                return left > right
            elif op == "<=":
                return left <= right
            elif op == ">=":
                return left >= right
            elif op == "AND":
                return cls.evaluate(expr.operands[0], batch) & cls.evaluate(expr.operands[1], batch)
            elif op == "OR":
                return cls.evaluate(expr.operands[0], batch) | cls.evaluate(expr.operands[1], batch)

        elif expr.op == "unary":
            if expr.value == "NOT":
                return ~cls.evaluate(expr.operands[0], batch)

        return np.full(batch.size, True)

    @classmethod
    def evaluate_to_array(cls, expr: Expr, batch: VectorBatch) -> np.ndarray:
        """Evaluate expression to value array."""
        if expr.op == "literal":
            return np.full(batch.size, expr.value)

        elif expr.op == "column":
            _, col_name = expr.value
            vec = cls._find_column(batch, col_name)
            if vec is None:
                return np.full(batch.size, None)
            return vec.data

        elif expr.op == "binary":
            left = cls.evaluate_to_array(expr.operands[0], batch)
            right = cls.evaluate_to_array(expr.operands[1], batch)
            op = expr.value

            if op == "+":
                return left + right
            elif op == "-":
                return left - right
            elif op == "*":
                return left * right
            elif op == "/":
                return np.divide(left, right, where=right != 0)

        return np.full(batch.size, None)

    @classmethod
    def evaluate_to_vector(cls, expr: Expr, batch: VectorBatch) -> Vector:
        """Evaluate expression to vector."""
        arr = cls.evaluate_to_array(expr, batch)
        nulls = np.array([v is None for v in arr]) if arr.dtype == object else np.zeros(len(arr), dtype=bool)
        return Vector(arr, nulls)

    @classmethod
    def _find_column(cls, batch: VectorBatch, name: str) -> Optional[Vector]:
        """Find column in batch."""
        vec = batch.get_column(name)
        if vec:
            return vec
        # Try without table prefix
        for col_name in batch.column_names:
            if col_name.endswith(f".{name}"):
                return batch.get_column(col_name)
        return None


class VectorizedExecutor:
    """
    Vectorized query executor.

    Executes queries using batch processing for OLAP workloads.
    """

    __slots__ = ("_column_store",)

    def __init__(self, column_store: ColumnStore) -> None:
        self._column_store = column_store

    def execute_scan(
        self,
        columns: List[str],
        filter_expr: Optional[Expr] = None,
    ) -> Iterator[VectorBatch]:
        """
        Execute a vectorized scan.

        Args:
            columns: Columns to scan
            filter_expr: Optional filter predicate

        Yields:
            VectorBatch instances
        """
        operator = ColumnScanOperator(self._column_store, columns)

        if filter_expr:
            operator = FilterOperator(operator, filter_expr)

        operator.open()
        try:
            while True:
                batch = operator.next_batch()
                if batch is None:
                    break
                if batch.size > 0:
                    yield batch
        finally:
            operator.close()

    def execute_aggregate(
        self,
        columns: List[str],
        group_by: List[str],
        aggregates: List[Tuple[str, str, str]],
    ) -> VectorBatch:
        """
        Execute a vectorized aggregation.

        Args:
            columns: All columns needed
            group_by: Group by columns
            aggregates: List of (function, column, alias)

        Returns:
            Aggregated results as VectorBatch
        """
        scan_op = ColumnScanOperator(self._column_store, columns)
        agg_op = AggregateOperator(scan_op, group_by, aggregates)

        agg_op.open()
        try:
            return agg_op.next_batch()
        finally:
            agg_op.close()

    def __repr__(self) -> str:
        return f"VectorizedExecutor(store={self._column_store})"
