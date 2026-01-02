"""
Query Planner.

Converts parsed SQL statements into logical query plans.
The logical plan is a tree of relational operators that can be
optimized and then executed.

Operators:
- SeqScan: Sequential table scan
- IndexScan: Index-based scan
- Filter: Predicate evaluation
- Project: Column selection
- Sort: ORDER BY
- Limit: LIMIT/OFFSET
- Join: INNER, LEFT, RIGHT, CROSS joins
- Aggregate: GROUP BY and aggregation functions
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Any, Dict, Tuple, Iterator

from clade.catalog.schema import Catalog, Table, Column, DataType
from clade.query.parser import (
    SelectStmt,
    InsertStmt,
    UpdateStmt,
    DeleteStmt,
    CreateTableStmt,
    DropTableStmt,
    Expr,
)


class PlanNodeType(Enum):
    """Types of plan nodes."""

    # Scan operators
    SEQ_SCAN = auto()
    INDEX_SCAN = auto()

    # Single-table operators
    FILTER = auto()
    PROJECT = auto()
    SORT = auto()
    LIMIT = auto()

    # Multi-table operators
    NESTED_LOOP_JOIN = auto()
    HASH_JOIN = auto()
    MERGE_JOIN = auto()

    # Aggregation
    AGGREGATE = auto()
    HASH_AGGREGATE = auto()

    # DML
    INSERT = auto()
    UPDATE = auto()
    DELETE = auto()

    # DDL
    CREATE_TABLE = auto()
    DROP_TABLE = auto()

    # Special
    VALUES = auto()
    RESULT = auto()


class JoinType(Enum):
    """Types of joins."""

    INNER = auto()
    LEFT = auto()
    RIGHT = auto()
    CROSS = auto()


class AggregateType(Enum):
    """Types of aggregation functions."""

    COUNT = auto()
    SUM = auto()
    AVG = auto()
    MIN = auto()
    MAX = auto()
    COUNT_ALL = auto()


class PlanNode(ABC):
    """
    Base class for plan nodes.

    Each plan node represents a relational operator in the query plan.
    """

    __slots__ = ("node_type", "output_schema", "children", "cost")

    def __init__(
        self,
        node_type: PlanNodeType,
        output_schema: List[Tuple[str, DataType]],
        children: Optional[List["PlanNode"]] = None,
    ) -> None:
        self.node_type = node_type
        self.output_schema = output_schema
        self.children = children or []
        self.cost = 0.0

    @abstractmethod
    def __repr__(self) -> str:
        pass


class SeqScanNode(PlanNode):
    """Sequential table scan."""

    __slots__ = ("table_name", "table_alias", "predicates")

    def __init__(
        self,
        table_name: str,
        output_schema: List[Tuple[str, DataType]],
        table_alias: Optional[str] = None,
        predicates: Optional[List[Expr]] = None,
    ) -> None:
        super().__init__(PlanNodeType.SEQ_SCAN, output_schema)
        self.table_name = table_name
        self.table_alias = table_alias or table_name
        self.predicates = predicates or []

    def __repr__(self) -> str:
        alias = f" AS {self.table_alias}" if self.table_alias != self.table_name else ""
        preds = f" WHERE {self.predicates}" if self.predicates else ""
        return f"SeqScan({self.table_name}{alias}{preds})"


class IndexScanNode(PlanNode):
    """Index-based table scan."""

    __slots__ = ("table_name", "index_name", "search_key", "predicates")

    def __init__(
        self,
        table_name: str,
        index_name: str,
        output_schema: List[Tuple[str, DataType]],
        search_key: Optional[Any] = None,
        predicates: Optional[List[Expr]] = None,
    ) -> None:
        super().__init__(PlanNodeType.INDEX_SCAN, output_schema)
        self.table_name = table_name
        self.index_name = index_name
        self.search_key = search_key
        self.predicates = predicates or []

    def __repr__(self) -> str:
        return f"IndexScan({self.table_name} USING {self.index_name})"


class FilterNode(PlanNode):
    """Filter rows based on predicate."""

    __slots__ = ("predicate",)

    def __init__(
        self,
        child: PlanNode,
        predicate: Expr,
    ) -> None:
        super().__init__(PlanNodeType.FILTER, child.output_schema, [child])
        self.predicate = predicate

    def __repr__(self) -> str:
        return f"Filter({self.predicate})"


class ProjectNode(PlanNode):
    """Project (select) specific columns."""

    __slots__ = ("expressions", "aliases")

    def __init__(
        self,
        child: PlanNode,
        expressions: List[Expr],
        aliases: List[Optional[str]],
        output_schema: List[Tuple[str, DataType]],
    ) -> None:
        super().__init__(PlanNodeType.PROJECT, output_schema, [child])
        self.expressions = expressions
        self.aliases = aliases

    def __repr__(self) -> str:
        return f"Project({len(self.expressions)} cols)"


class SortNode(PlanNode):
    """Sort rows by expressions."""

    __slots__ = ("sort_keys", "directions")

    def __init__(
        self,
        child: PlanNode,
        sort_keys: List[Expr],
        directions: List[str],  # 'ASC' or 'DESC'
    ) -> None:
        super().__init__(PlanNodeType.SORT, child.output_schema, [child])
        self.sort_keys = sort_keys
        self.directions = directions

    def __repr__(self) -> str:
        return f"Sort({len(self.sort_keys)} keys)"


class LimitNode(PlanNode):
    """Limit and offset rows."""

    __slots__ = ("limit", "offset")

    def __init__(
        self,
        child: PlanNode,
        limit: int,
        offset: int = 0,
    ) -> None:
        super().__init__(PlanNodeType.LIMIT, child.output_schema, [child])
        self.limit = limit
        self.offset = offset

    def __repr__(self) -> str:
        return f"Limit({self.limit} OFFSET {self.offset})"


class JoinNode(PlanNode):
    """Join two relations."""

    __slots__ = ("join_type", "condition")

    def __init__(
        self,
        left: PlanNode,
        right: PlanNode,
        join_type: JoinType,
        condition: Optional[Expr],
        output_schema: List[Tuple[str, DataType]],
    ) -> None:
        super().__init__(PlanNodeType.NESTED_LOOP_JOIN, output_schema, [left, right])
        self.join_type = join_type
        self.condition = condition

    def __repr__(self) -> str:
        return f"Join({self.join_type.name})"


class AggregateNode(PlanNode):
    """Aggregate with GROUP BY."""

    __slots__ = ("group_by", "aggregates")

    def __init__(
        self,
        child: PlanNode,
        group_by: List[Expr],
        aggregates: List[Tuple[AggregateType, Expr, str]],  # (type, expr, alias)
        output_schema: List[Tuple[str, DataType]],
    ) -> None:
        super().__init__(PlanNodeType.AGGREGATE, output_schema, [child])
        self.group_by = group_by
        self.aggregates = aggregates

    def __repr__(self) -> str:
        return f"Aggregate({len(self.aggregates)} aggs)"


class InsertNode(PlanNode):
    """Insert rows into a table."""

    __slots__ = ("table_name", "columns", "values")

    def __init__(
        self,
        table_name: str,
        columns: Optional[List[str]],
        values: List[List[Expr]],
    ) -> None:
        super().__init__(PlanNodeType.INSERT, [])
        self.table_name = table_name
        self.columns = columns
        self.values = values

    def __repr__(self) -> str:
        return f"Insert({self.table_name}, {len(self.values)} rows)"


class UpdateNode(PlanNode):
    """Update rows in a table."""

    __slots__ = ("table_name", "assignments", "predicate")

    def __init__(
        self,
        child: PlanNode,
        table_name: str,
        assignments: List[Tuple[str, Expr]],
        predicate: Optional[Expr],
    ) -> None:
        super().__init__(PlanNodeType.UPDATE, [], [child])
        self.table_name = table_name
        self.assignments = assignments
        self.predicate = predicate

    def __repr__(self) -> str:
        return f"Update({self.table_name})"


class DeleteNode(PlanNode):
    """Delete rows from a table."""

    __slots__ = ("table_name", "predicate")

    def __init__(
        self,
        child: PlanNode,
        table_name: str,
        predicate: Optional[Expr],
    ) -> None:
        super().__init__(PlanNodeType.DELETE, [], [child])
        self.table_name = table_name
        self.predicate = predicate

    def __repr__(self) -> str:
        return f"Delete({self.table_name})"


class CreateTableNode(PlanNode):
    """Create a new table."""

    __slots__ = ("table_def",)

    def __init__(self, table_def: CreateTableStmt) -> None:
        super().__init__(PlanNodeType.CREATE_TABLE, [])
        self.table_def = table_def

    def __repr__(self) -> str:
        return f"CreateTable({self.table_def.table})"


class DropTableNode(PlanNode):
    """Drop a table."""

    __slots__ = ("table_name", "if_exists")

    def __init__(self, table_name: str, if_exists: bool = False) -> None:
        super().__init__(PlanNodeType.DROP_TABLE, [])
        self.table_name = table_name
        self.if_exists = if_exists

    def __repr__(self) -> str:
        return f"DropTable({self.table_name})"


class QueryPlanner:
    """
    Query planner that converts parsed SQL to logical plans.

    The planner:
    1. Validates table and column references against the catalog
    2. Resolves column types
    3. Creates a tree of plan nodes

    Thread Safety: Thread-safe (read-only catalog access).
    """

    __slots__ = ("_catalog",)

    def __init__(self, catalog: Catalog) -> None:
        """Initialize planner with catalog."""
        self._catalog = catalog

    def plan(self, stmt: Any) -> PlanNode:
        """
        Create a logical plan for a SQL statement.

        Args:
            stmt: Parsed SQL statement

        Returns:
            Root of the logical plan tree

        Raises:
            ValueError: If table or column references are invalid
        """
        if isinstance(stmt, SelectStmt):
            return self._plan_select(stmt)
        elif isinstance(stmt, InsertStmt):
            return self._plan_insert(stmt)
        elif isinstance(stmt, UpdateStmt):
            return self._plan_update(stmt)
        elif isinstance(stmt, DeleteStmt):
            return self._plan_delete(stmt)
        elif isinstance(stmt, CreateTableStmt):
            return CreateTableNode(stmt)
        elif isinstance(stmt, DropTableStmt):
            return DropTableNode(stmt.table, stmt.if_exists)
        else:
            raise ValueError(f"Unknown statement type: {type(stmt)}")

    def _plan_select(self, stmt: SelectStmt) -> PlanNode:
        """Plan a SELECT statement."""
        # Get table schema
        table = self._catalog.get_table(stmt.table)
        if table is None:
            raise ValueError(f"Table '{stmt.table}' not found")

        # Build base scan
        output_schema = self._get_table_schema(table, stmt.table_alias)
        plan: PlanNode = SeqScanNode(
            table_name=stmt.table,
            output_schema=output_schema,
            table_alias=stmt.table_alias,
        )

        # Handle JOINs
        for join in stmt.joins:
            join_table = self._catalog.get_table(join["table"])
            if join_table is None:
                raise ValueError(f"Table '{join['table']}' not found")

            join_schema = self._get_table_schema(join_table, join.get("alias"))
            right_scan = SeqScanNode(
                table_name=join["table"],
                output_schema=join_schema,
                table_alias=join.get("alias"),
            )

            # Combine schemas
            combined_schema = plan.output_schema + right_scan.output_schema

            # Map join type
            join_type_map = {
                "INNER": JoinType.INNER,
                "LEFT": JoinType.LEFT,
                "RIGHT": JoinType.RIGHT,
                "CROSS": JoinType.CROSS,
            }
            join_type = join_type_map.get(join["type"], JoinType.INNER)

            plan = JoinNode(
                left=plan,
                right=right_scan,
                join_type=join_type,
                condition=join.get("condition"),
                output_schema=combined_schema,
            )

        # Add Filter for WHERE clause
        if stmt.where is not None:
            plan = FilterNode(plan, stmt.where)

        # Add Projection
        if not stmt.is_select_all:
            expressions = [col[0] for col in stmt.columns]
            aliases = [col[1] for col in stmt.columns]

            # Determine output schema from expressions
            proj_schema = self._infer_projection_schema(
                expressions, aliases, plan.output_schema
            )

            plan = ProjectNode(
                child=plan,
                expressions=expressions,
                aliases=aliases,
                output_schema=proj_schema,
            )

        # Add Sort for ORDER BY
        if stmt.order_by is not None:
            sort_keys = [item[0] for item in stmt.order_by]
            directions = [item[1] for item in stmt.order_by]
            plan = SortNode(plan, sort_keys, directions)

        # Add Limit
        if stmt.limit is not None:
            plan = LimitNode(plan, stmt.limit, stmt.offset)

        return plan

    def _plan_insert(self, stmt: InsertStmt) -> PlanNode:
        """Plan an INSERT statement."""
        table = self._catalog.get_table(stmt.table)
        if table is None:
            raise ValueError(f"Table '{stmt.table}' not found")

        # Validate columns if specified
        if stmt.columns:
            for col_name in stmt.columns:
                if table.get_column(col_name) is None:
                    raise ValueError(f"Column '{col_name}' not found in table '{stmt.table}'")

        return InsertNode(
            table_name=stmt.table,
            columns=stmt.columns,
            values=stmt.values,
        )

    def _plan_update(self, stmt: UpdateStmt) -> PlanNode:
        """Plan an UPDATE statement."""
        table = self._catalog.get_table(stmt.table)
        if table is None:
            raise ValueError(f"Table '{stmt.table}' not found")

        # Validate assignment columns
        for col_name, _ in stmt.assignments:
            if table.get_column(col_name) is None:
                raise ValueError(f"Column '{col_name}' not found in table '{stmt.table}'")

        # Build scan
        output_schema = self._get_table_schema(table)
        scan = SeqScanNode(stmt.table, output_schema)

        # Add filter if WHERE clause exists
        if stmt.where is not None:
            scan = FilterNode(scan, stmt.where)

        return UpdateNode(
            child=scan,
            table_name=stmt.table,
            assignments=stmt.assignments,
            predicate=stmt.where,
        )

    def _plan_delete(self, stmt: DeleteStmt) -> PlanNode:
        """Plan a DELETE statement."""
        table = self._catalog.get_table(stmt.table)
        if table is None:
            raise ValueError(f"Table '{stmt.table}' not found")

        # Build scan
        output_schema = self._get_table_schema(table)
        scan = SeqScanNode(stmt.table, output_schema)

        # Add filter if WHERE clause exists
        if stmt.where is not None:
            scan = FilterNode(scan, stmt.where)

        return DeleteNode(
            child=scan,
            table_name=stmt.table,
            predicate=stmt.where,
        )

    def _get_table_schema(
        self, table: Table, alias: Optional[str] = None
    ) -> List[Tuple[str, DataType]]:
        """Get output schema for a table."""
        prefix = alias or table.name
        return [(f"{prefix}.{col.name}", col.data_type) for col in table.columns]

    def _infer_projection_schema(
        self,
        expressions: List[Expr],
        aliases: List[Optional[str]],
        input_schema: List[Tuple[str, DataType]],
    ) -> List[Tuple[str, DataType]]:
        """Infer output schema from projection expressions."""
        schema = []

        for expr, alias in zip(expressions, aliases):
            if expr.op == "column":
                _, col_name = expr.value
                # Find column in input schema
                dtype = DataType.NULL  # Default
                for name, dt in input_schema:
                    if name.endswith(f".{col_name}") or name == col_name:
                        dtype = dt
                        break

                out_name = alias or col_name
                schema.append((out_name, dtype))
            elif expr.op == "function":
                # Aggregate functions
                out_name = alias or expr.value
                dtype = self._infer_function_type(expr.value)
                schema.append((out_name, dtype))
            elif expr.op == "literal":
                out_name = alias or str(expr.value)
                dtype = self._infer_literal_type(expr.value)
                schema.append((out_name, dtype))
            else:
                # Binary/unary expression - use alias or default
                out_name = alias or "expr"
                schema.append((out_name, DataType.NULL))

        return schema

    def _infer_function_type(self, func_name: str) -> DataType:
        """Infer return type of a function."""
        if func_name in ("COUNT", "COUNT_ALL"):
            return DataType.BIGINT
        elif func_name in ("SUM", "AVG"):
            return DataType.FLOAT
        elif func_name in ("MIN", "MAX"):
            return DataType.NULL  # Depends on input
        else:
            return DataType.NULL

    def _infer_literal_type(self, value: Any) -> DataType:
        """Infer type of a literal value."""
        if value is None:
            return DataType.NULL
        elif isinstance(value, bool):
            return DataType.BOOLEAN
        elif isinstance(value, int):
            return DataType.INTEGER
        elif isinstance(value, float):
            return DataType.FLOAT
        elif isinstance(value, str):
            return DataType.TEXT
        else:
            return DataType.NULL

    def __repr__(self) -> str:
        return f"QueryPlanner(catalog={self._catalog})"


def print_plan(node: PlanNode, indent: int = 0) -> str:
    """Pretty print a query plan."""
    lines = ["  " * indent + repr(node)]
    for child in node.children:
        lines.append(print_plan(child, indent + 1))
    return "\n".join(lines)
