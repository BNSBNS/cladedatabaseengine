"""
Volcano (Iterator) Execution Model.

Implements pull-based query execution where each operator
is an iterator that produces one tuple at a time.

Interface:
- open(): Initialize the operator
- next(): Return the next tuple, or None if done
- close(): Clean up resources

Operators:
- SeqScanExecutor: Sequential table scan
- FilterExecutor: Predicate evaluation
- ProjectExecutor: Column selection
- SortExecutor: ORDER BY (materializing)
- LimitExecutor: LIMIT/OFFSET
- JoinExecutor: Nested loop join
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Iterator, Tuple
import operator as op_module

from clade.catalog.schema import Catalog, Table, RecordSerializer, DataType
from clade.storage.heap import HeapFile
from clade.query.planner import (
    PlanNode,
    PlanNodeType,
    SeqScanNode,
    FilterNode,
    ProjectNode,
    SortNode,
    LimitNode,
    JoinNode,
    InsertNode,
    UpdateNode,
    DeleteNode,
    CreateTableNode,
    DropTableNode,
    JoinType,
)
from clade.query.parser import Expr


# Type alias for a row
Row = Dict[str, Any]


class Executor(ABC):
    """
    Base class for Volcano-style executors.

    Each executor produces rows one at a time via the next() method.
    """

    __slots__ = ("_plan", "_schema", "_is_open")

    def __init__(self, plan: PlanNode) -> None:
        self._plan = plan
        self._schema = plan.output_schema
        self._is_open = False

    @abstractmethod
    def open(self) -> None:
        """Initialize the executor."""
        pass

    @abstractmethod
    def next(self) -> Optional[Row]:
        """Return the next row, or None if done."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up resources."""
        pass

    def __iter__(self) -> Iterator[Row]:
        """Allow using executor as an iterator."""
        self.open()
        try:
            while True:
                row = self.next()
                if row is None:
                    break
                yield row
        finally:
            self.close()


class SeqScanExecutor(Executor):
    """Sequential table scan."""

    __slots__ = ("_heap", "_serializer", "_iterator", "_table_alias", "_predicates")

    def __init__(
        self,
        plan: SeqScanNode,
        heap: HeapFile,
        serializer: RecordSerializer,
    ) -> None:
        super().__init__(plan)
        self._heap = heap
        self._serializer = serializer
        self._table_alias = plan.table_alias or plan.table_name
        self._predicates = plan.predicates
        self._iterator = None

    def open(self) -> None:
        self._iterator = self._heap.scan()
        self._is_open = True

    def next(self) -> Optional[Row]:
        if not self._is_open:
            return None

        while True:
            try:
                rid, data = next(self._iterator)
            except StopIteration:
                return None

            # Deserialize the record
            row = self._serializer.deserialize(data)

            # Add table alias prefix to column names
            prefixed = {f"{self._table_alias}.{k}": v for k, v in row.items()}

            # Apply pushed-down predicates
            if self._predicates:
                if not all(self._evaluate_predicate(p, prefixed) for p in self._predicates):
                    continue

            return prefixed

    def _evaluate_predicate(self, pred: Expr, row: Row) -> bool:
        """Evaluate a predicate on a row."""
        return ExpressionEvaluator.evaluate(pred, row)

    def close(self) -> None:
        self._iterator = None
        self._is_open = False


class FilterExecutor(Executor):
    """Filter rows based on a predicate."""

    __slots__ = ("_child", "_predicate")

    def __init__(self, plan: FilterNode, child: Executor) -> None:
        super().__init__(plan)
        self._child = child
        self._predicate = plan.predicate

    def open(self) -> None:
        self._child.open()
        self._is_open = True

    def next(self) -> Optional[Row]:
        if not self._is_open:
            return None

        while True:
            row = self._child.next()
            if row is None:
                return None

            if ExpressionEvaluator.evaluate(self._predicate, row):
                return row

    def close(self) -> None:
        self._child.close()
        self._is_open = False


class ProjectExecutor(Executor):
    """Project specific columns/expressions."""

    __slots__ = ("_child", "_expressions", "_aliases")

    def __init__(self, plan: ProjectNode, child: Executor) -> None:
        super().__init__(plan)
        self._child = child
        self._expressions = plan.expressions
        self._aliases = plan.aliases

    def open(self) -> None:
        self._child.open()
        self._is_open = True

    def next(self) -> Optional[Row]:
        if not self._is_open:
            return None

        row = self._child.next()
        if row is None:
            return None

        # Evaluate each expression
        result = {}
        for expr, alias in zip(self._expressions, self._aliases):
            value = ExpressionEvaluator.evaluate(expr, row)
            name = alias or self._get_expr_name(expr)
            result[name] = value

        return result

    def _get_expr_name(self, expr: Expr) -> str:
        """Get a name for an expression."""
        if expr.op == "column":
            return expr.value[1]  # Column name
        elif expr.op == "function":
            return expr.value  # Function name
        else:
            return "expr"

    def close(self) -> None:
        self._child.close()
        self._is_open = False


class SortExecutor(Executor):
    """Sort rows (materializing)."""

    __slots__ = ("_child", "_sort_keys", "_directions", "_sorted_rows", "_index")

    def __init__(self, plan: SortNode, child: Executor) -> None:
        super().__init__(plan)
        self._child = child
        self._sort_keys = plan.sort_keys
        self._directions = plan.directions
        self._sorted_rows = []
        self._index = 0

    def open(self) -> None:
        self._child.open()

        # Materialize all rows
        rows = []
        while True:
            row = self._child.next()
            if row is None:
                break
            rows.append(row)

        # Sort rows
        def sort_key(row):
            return tuple(
                (ExpressionEvaluator.evaluate(key, row) or "")
                for key in self._sort_keys
            )

        # Determine if reverse (DESC for any key)
        reverse = any(d == "DESC" for d in self._directions)

        self._sorted_rows = sorted(rows, key=sort_key, reverse=reverse)
        self._index = 0
        self._is_open = True

    def next(self) -> Optional[Row]:
        if not self._is_open or self._index >= len(self._sorted_rows):
            return None

        row = self._sorted_rows[self._index]
        self._index += 1
        return row

    def close(self) -> None:
        self._child.close()
        self._sorted_rows = []
        self._index = 0
        self._is_open = False


class LimitExecutor(Executor):
    """Limit and offset rows."""

    __slots__ = ("_child", "_limit", "_offset", "_count", "_skipped")

    def __init__(self, plan: LimitNode, child: Executor) -> None:
        super().__init__(plan)
        self._child = child
        self._limit = plan.limit
        self._offset = plan.offset
        self._count = 0
        self._skipped = 0

    def open(self) -> None:
        self._child.open()
        self._count = 0
        self._skipped = 0
        self._is_open = True

    def next(self) -> Optional[Row]:
        if not self._is_open:
            return None

        # Skip offset rows
        while self._skipped < self._offset:
            row = self._child.next()
            if row is None:
                return None
            self._skipped += 1

        # Return up to limit rows
        if self._count >= self._limit:
            return None

        row = self._child.next()
        if row is not None:
            self._count += 1
        return row

    def close(self) -> None:
        self._child.close()
        self._is_open = False


class NestedLoopJoinExecutor(Executor):
    """Nested loop join."""

    __slots__ = (
        "_left",
        "_right",
        "_join_type",
        "_condition",
        "_current_left",
        "_right_exhausted",
        "_left_matched",
    )

    def __init__(
        self,
        plan: JoinNode,
        left: Executor,
        right: Executor,
    ) -> None:
        super().__init__(plan)
        self._left = left
        self._right = right
        self._join_type = plan.join_type
        self._condition = plan.condition
        self._current_left = None
        self._right_exhausted = False
        self._left_matched = False

    def open(self) -> None:
        self._left.open()
        self._current_left = self._left.next()
        if self._current_left is not None:
            self._right.open()
        self._right_exhausted = False
        self._left_matched = False
        self._is_open = True

    def next(self) -> Optional[Row]:
        if not self._is_open:
            return None

        while self._current_left is not None:
            # Try to get a matching right row
            if not self._right_exhausted:
                right_row = self._right.next()

                if right_row is not None:
                    # Check join condition
                    combined = {**self._current_left, **right_row}

                    if self._condition is None or ExpressionEvaluator.evaluate(self._condition, combined):
                        self._left_matched = True
                        return combined
                    continue
                else:
                    self._right_exhausted = True

            # Right side exhausted for current left row
            # For LEFT JOIN, output unmatched left rows
            if self._join_type == JoinType.LEFT and not self._left_matched:
                null_right = {name: None for name, _ in self._right._schema}
                result = {**self._current_left, **null_right}
                self._left_matched = True  # Mark as outputted
                return result

            # Move to next left row
            self._current_left = self._left.next()
            if self._current_left is not None:
                self._right.close()
                self._right.open()
                self._right_exhausted = False
                self._left_matched = False

        return None

    def close(self) -> None:
        self._left.close()
        self._right.close()
        self._is_open = False


class ExpressionEvaluator:
    """Evaluate expressions on rows."""

    # Comparison operators
    _COMPARISONS = {
        "=": op_module.eq,
        "!=": op_module.ne,
        "<": op_module.lt,
        ">": op_module.gt,
        "<=": op_module.le,
        ">=": op_module.ge,
    }

    # Arithmetic operators
    _ARITHMETIC = {
        "+": op_module.add,
        "-": op_module.sub,
        "*": op_module.mul,
        "/": op_module.truediv,
        "%": op_module.mod,
    }

    @classmethod
    def evaluate(cls, expr: Expr, row: Row) -> Any:
        """
        Evaluate an expression on a row.

        Args:
            expr: Expression to evaluate
            row: Row data (column name -> value)

        Returns:
            The result of evaluating the expression
        """
        if expr.op == "literal":
            return expr.value

        elif expr.op == "column":
            table_alias, col_name = expr.value
            # Try with table prefix
            if table_alias:
                full_name = f"{table_alias}.{col_name}"
                if full_name in row:
                    return row[full_name]
            # Try without prefix
            if col_name in row:
                return row[col_name]
            # Try to find matching column
            for key in row:
                if key.endswith(f".{col_name}"):
                    return row[key]
            return None

        elif expr.op == "binary":
            left = cls.evaluate(expr.operands[0], row)
            right = cls.evaluate(expr.operands[1], row)
            op = expr.value

            if op in cls._COMPARISONS:
                if left is None or right is None:
                    return None  # NULL comparison
                return cls._COMPARISONS[op](left, right)

            elif op in cls._ARITHMETIC:
                if left is None or right is None:
                    return None
                return cls._ARITHMETIC[op](left, right)

            elif op == "AND":
                return bool(left) and bool(right)

            elif op == "OR":
                return bool(left) or bool(right)

            elif op == "LIKE":
                if left is None or right is None:
                    return None
                import re
                pattern = right.replace("%", ".*").replace("_", ".")
                return bool(re.match(f"^{pattern}$", str(left)))

        elif expr.op == "unary":
            operand = cls.evaluate(expr.operands[0], row)
            op = expr.value

            if op == "NOT":
                return not bool(operand)
            elif op == "-":
                return -operand if operand is not None else None
            elif op == "IS NULL":
                return operand is None
            elif op == "IS NOT NULL":
                return operand is not None

        elif expr.op == "between":
            value = cls.evaluate(expr.operands[0], row)
            low = cls.evaluate(expr.operands[1], row)
            high = cls.evaluate(expr.operands[2], row)
            if any(v is None for v in [value, low, high]):
                return None
            return low <= value <= high

        elif expr.op == "in":
            value = cls.evaluate(expr.operands[0], row)
            if value is None:
                return None
            values = [cls.evaluate(e, row) for e in expr.operands[1:]]
            return value in values

        elif expr.op == "function":
            # Aggregate functions are handled by AggregateExecutor
            # For scalar functions, evaluate here
            func_name = expr.value.upper()
            args = [cls.evaluate(arg, row) for arg in expr.operands]

            if func_name == "UPPER":
                return str(args[0]).upper() if args[0] else None
            elif func_name == "LOWER":
                return str(args[0]).lower() if args[0] else None
            elif func_name == "LENGTH":
                return len(str(args[0])) if args[0] else None
            elif func_name == "ABS":
                return abs(args[0]) if args[0] is not None else None

            # For aggregate functions, return placeholder
            return None

        return None


class ExecutionEngine:
    """
    Query execution engine.

    Converts logical plans to physical executors and runs them.
    """

    __slots__ = ("_catalog", "_heaps")

    def __init__(self, catalog: Catalog, heaps: Dict[str, HeapFile]) -> None:
        """
        Initialize execution engine.

        Args:
            catalog: Database catalog
            heaps: Map of table name to HeapFile
        """
        self._catalog = catalog
        self._heaps = heaps

    def execute(self, plan: PlanNode) -> Iterator[Row]:
        """
        Execute a query plan.

        Args:
            plan: Logical query plan

        Yields:
            Result rows
        """
        executor = self._build_executor(plan)
        return iter(executor)

    def execute_dml(self, plan: PlanNode) -> int:
        """
        Execute a DML statement.

        Args:
            plan: INSERT, UPDATE, or DELETE plan

        Returns:
            Number of affected rows
        """
        if plan.node_type == PlanNodeType.INSERT:
            return self._execute_insert(plan)
        elif plan.node_type == PlanNodeType.UPDATE:
            return self._execute_update(plan)
        elif plan.node_type == PlanNodeType.DELETE:
            return self._execute_delete(plan)
        else:
            raise ValueError(f"Not a DML plan: {plan.node_type}")

    def _build_executor(self, plan: PlanNode) -> Executor:
        """Build an executor tree from a plan."""
        if plan.node_type == PlanNodeType.SEQ_SCAN:
            scan = plan
            heap = self._heaps.get(scan.table_name)
            if heap is None:
                raise ValueError(f"No heap file for table '{scan.table_name}'")
            serializer = self._catalog.get_serializer(scan.table_name)
            return SeqScanExecutor(scan, heap, serializer)

        elif plan.node_type == PlanNodeType.FILTER:
            child = self._build_executor(plan.children[0])
            return FilterExecutor(plan, child)

        elif plan.node_type == PlanNodeType.PROJECT:
            child = self._build_executor(plan.children[0])
            return ProjectExecutor(plan, child)

        elif plan.node_type == PlanNodeType.SORT:
            child = self._build_executor(plan.children[0])
            return SortExecutor(plan, child)

        elif plan.node_type == PlanNodeType.LIMIT:
            child = self._build_executor(plan.children[0])
            return LimitExecutor(plan, child)

        elif plan.node_type == PlanNodeType.NESTED_LOOP_JOIN:
            left = self._build_executor(plan.children[0])
            right = self._build_executor(plan.children[1])
            return NestedLoopJoinExecutor(plan, left, right)

        else:
            raise ValueError(f"Unknown plan type: {plan.node_type}")

    def _execute_insert(self, plan: InsertNode) -> int:
        """Execute INSERT statement."""
        heap = self._heaps.get(plan.table_name)
        if heap is None:
            raise ValueError(f"No heap file for table '{plan.table_name}'")

        table = self._catalog.get_table(plan.table_name)
        serializer = self._catalog.get_serializer(plan.table_name)

        count = 0
        for row_values in plan.values:
            # Build row dictionary
            row = {}
            columns = plan.columns or table.column_names

            for col_name, expr in zip(columns, row_values):
                row[col_name] = ExpressionEvaluator.evaluate(expr, {})

            # Serialize and insert
            data = serializer.serialize(row)
            heap.insert(data)
            count += 1

        return count

    def _execute_update(self, plan: UpdateNode) -> int:
        """Execute UPDATE statement."""
        heap = self._heaps.get(plan.table_name)
        if heap is None:
            raise ValueError(f"No heap file for table '{plan.table_name}'")

        serializer = self._catalog.get_serializer(plan.table_name)

        # Get rows to update
        child = self._build_executor(plan.children[0])
        rows_to_update = list(child)

        count = 0
        for row in rows_to_update:
            # Apply assignments
            updated = dict(row)
            for col_name, expr in plan.assignments:
                full_name = f"{plan.table_name}.{col_name}"
                value = ExpressionEvaluator.evaluate(expr, row)
                if full_name in updated:
                    updated[full_name] = value
                else:
                    updated[col_name] = value

            # For now, we'd need RID tracking to update in-place
            # This is a simplified version
            count += 1

        return count

    def _execute_delete(self, plan: DeleteNode) -> int:
        """Execute DELETE statement."""
        heap = self._heaps.get(plan.table_name)
        if heap is None:
            raise ValueError(f"No heap file for table '{plan.table_name}'")

        # Get rows to delete
        child = self._build_executor(plan.children[0])
        rows_to_delete = list(child)

        # For now, we'd need RID tracking to actually delete
        return len(rows_to_delete)

    def __repr__(self) -> str:
        return f"ExecutionEngine(tables={list(self._heaps.keys())})"
