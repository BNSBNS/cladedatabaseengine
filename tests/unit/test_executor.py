"""
Unit tests for Volcano executor.

Tests cover:
- Sequential scan
- Filter
- Projection
- Sort
- Limit
- Join
- Expression evaluation
"""

import pytest
import tempfile
from pathlib import Path

from clade.catalog.schema import Catalog, Table, Column, DataType
from clade.storage.file_manager import FileManager
from clade.storage.buffer_manager import BufferPoolManager
from clade.storage.heap import HeapFile
from clade.query.parser import SQLParser, Expr
from clade.query.planner import QueryPlanner
from clade.query.optimizer import Optimizer
from clade.execution.volcano import (
    ExecutionEngine,
    ExpressionEvaluator,
    SeqScanExecutor,
    FilterExecutor,
    ProjectExecutor,
    SortExecutor,
    LimitExecutor,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def catalog():
    """Create a test catalog."""
    cat = Catalog()

    users_cols = [
        Column(name="id", data_type=DataType.INTEGER, primary_key=True),
        Column(name="name", data_type=DataType.VARCHAR, max_length=100),
        Column(name="age", data_type=DataType.INTEGER),
        Column(name="active", data_type=DataType.BOOLEAN),
    ]
    cat.create_table(Table(name="users", columns=users_cols, primary_key=["id"]))

    orders_cols = [
        Column(name="id", data_type=DataType.INTEGER, primary_key=True),
        Column(name="user_id", data_type=DataType.INTEGER),
        Column(name="total", data_type=DataType.FLOAT),
    ]
    cat.create_table(Table(name="orders", columns=orders_cols, primary_key=["id"]))

    return cat


@pytest.fixture
def heaps(temp_dir, catalog):
    """Create heap files with sample data."""
    heaps = {}
    resources = []  # Track all resources for cleanup

    # Users heap
    fm_users = FileManager(temp_dir / "users.db")
    bp_users = BufferPoolManager(fm_users, pool_size=10)
    heap_users = HeapFile(bp_users)
    resources.append((heap_users, bp_users, fm_users))

    serializer = catalog.get_serializer("users")
    test_users = [
        {"id": 1, "name": "Alice", "age": 30, "active": True},
        {"id": 2, "name": "Bob", "age": 25, "active": True},
        {"id": 3, "name": "Charlie", "age": 35, "active": False},
        {"id": 4, "name": "Diana", "age": 28, "active": True},
        {"id": 5, "name": "Eve", "age": 22, "active": False},
    ]
    for user in test_users:
        data = serializer.serialize(user)
        heap_users.insert(data)

    heaps["users"] = heap_users

    # Orders heap
    fm_orders = FileManager(temp_dir / "orders.db")
    bp_orders = BufferPoolManager(fm_orders, pool_size=10)
    heap_orders = HeapFile(bp_orders)
    resources.append((heap_orders, bp_orders, fm_orders))

    serializer_orders = catalog.get_serializer("orders")
    test_orders = [
        {"id": 1, "user_id": 1, "total": 100.50},
        {"id": 2, "user_id": 1, "total": 200.00},
        {"id": 3, "user_id": 2, "total": 50.25},
        {"id": 4, "user_id": 3, "total": 75.00},
    ]
    for order in test_orders:
        data = serializer_orders.serialize(order)
        heap_orders.insert(data)

    heaps["orders"] = heap_orders

    yield heaps

    # Cleanup - close in reverse order
    for heap, bp, fm in resources:
        try:
            heap.close()
        except Exception:
            pass
        try:
            fm.close()
        except Exception:
            pass


@pytest.fixture
def engine(catalog, heaps):
    """Create an execution engine."""
    return ExecutionEngine(catalog, heaps)


@pytest.fixture
def parser():
    """Create a SQL parser."""
    return SQLParser()


@pytest.fixture
def planner(catalog):
    """Create a query planner."""
    return QueryPlanner(catalog)


class TestExpressionEvaluator:
    """Tests for expression evaluation."""

    def test_literal(self):
        """Should evaluate literal values."""
        expr = Expr("literal", [], 42)
        result = ExpressionEvaluator.evaluate(expr, {})
        assert result == 42

    def test_column_reference(self):
        """Should evaluate column references."""
        expr = Expr("column", [], (None, "name"))
        row = {"name": "Alice"}
        result = ExpressionEvaluator.evaluate(expr, row)
        assert result == "Alice"

    def test_qualified_column(self):
        """Should evaluate qualified column references."""
        expr = Expr("column", [], ("users", "name"))
        row = {"users.name": "Alice"}
        result = ExpressionEvaluator.evaluate(expr, row)
        assert result == "Alice"

    def test_comparison_operators(self):
        """Should evaluate comparison operators."""
        row = {"a": 5, "b": 3}

        # Equal
        expr = Expr("binary", [Expr("column", [], (None, "a")), Expr("literal", [], 5)], "=")
        assert ExpressionEvaluator.evaluate(expr, row) is True

        # Less than
        expr = Expr("binary", [Expr("column", [], (None, "b")), Expr("column", [], (None, "a"))], "<")
        assert ExpressionEvaluator.evaluate(expr, row) is True

    def test_arithmetic_operators(self):
        """Should evaluate arithmetic operators."""
        row = {"a": 10, "b": 3}

        expr = Expr("binary", [Expr("column", [], (None, "a")), Expr("column", [], (None, "b"))], "+")
        assert ExpressionEvaluator.evaluate(expr, row) == 13

        expr = Expr("binary", [Expr("column", [], (None, "a")), Expr("column", [], (None, "b"))], "*")
        assert ExpressionEvaluator.evaluate(expr, row) == 30

    def test_and_or(self):
        """Should evaluate AND/OR."""
        left = Expr("literal", [], True)
        right = Expr("literal", [], False)

        expr = Expr("binary", [left, right], "AND")
        assert ExpressionEvaluator.evaluate(expr, {}) is False

        expr = Expr("binary", [left, right], "OR")
        assert ExpressionEvaluator.evaluate(expr, {}) is True

    def test_is_null(self):
        """Should evaluate IS NULL."""
        row = {"a": None, "b": 5}

        expr = Expr("unary", [Expr("column", [], (None, "a"))], "IS NULL")
        assert ExpressionEvaluator.evaluate(expr, row) is True

        expr = Expr("unary", [Expr("column", [], (None, "b"))], "IS NULL")
        assert ExpressionEvaluator.evaluate(expr, row) is False

    def test_between(self):
        """Should evaluate BETWEEN."""
        row = {"age": 25}

        expr = Expr("between", [
            Expr("column", [], (None, "age")),
            Expr("literal", [], 20),
            Expr("literal", [], 30),
        ], "BETWEEN")
        assert ExpressionEvaluator.evaluate(expr, row) is True

    def test_like(self):
        """Should evaluate LIKE."""
        row = {"name": "Alice"}

        expr = Expr("binary", [
            Expr("column", [], (None, "name")),
            Expr("literal", [], "A%"),
        ], "LIKE")
        assert ExpressionEvaluator.evaluate(expr, row) is True

        expr = Expr("binary", [
            Expr("column", [], (None, "name")),
            Expr("literal", [], "B%"),
        ], "LIKE")
        assert ExpressionEvaluator.evaluate(expr, row) is False


class TestSeqScan:
    """Tests for sequential scan."""

    def test_scan_all_rows(self, engine, parser, planner):
        """Should scan all rows from table."""
        stmt = parser.parse("SELECT * FROM users")
        plan = planner.plan(stmt)

        rows = list(engine.execute(plan))

        assert len(rows) == 5

    def test_scan_with_filter(self, engine, parser, planner):
        """Should filter rows during scan."""
        stmt = parser.parse("SELECT * FROM users WHERE active = TRUE")
        plan = planner.plan(stmt)

        rows = list(engine.execute(plan))

        assert len(rows) == 3  # Alice, Bob, Diana


class TestFilter:
    """Tests for filter operator."""

    def test_filter_equality(self, engine, parser, planner):
        """Should filter by equality."""
        stmt = parser.parse("SELECT * FROM users WHERE id = 1")
        plan = planner.plan(stmt)

        rows = list(engine.execute(plan))

        assert len(rows) == 1
        assert rows[0]["users.name"] == "Alice"

    def test_filter_comparison(self, engine, parser, planner):
        """Should filter by comparison."""
        stmt = parser.parse("SELECT * FROM users WHERE age > 28")
        plan = planner.plan(stmt)

        rows = list(engine.execute(plan))

        assert len(rows) == 2  # Alice (30), Charlie (35)


class TestProjection:
    """Tests for projection operator."""

    def test_project_columns(self, engine, parser, planner):
        """Should project specific columns."""
        stmt = parser.parse("SELECT name, age FROM users")
        plan = planner.plan(stmt)

        rows = list(engine.execute(plan))

        assert len(rows) == 5
        assert "name" in rows[0]
        assert "age" in rows[0]


class TestSort:
    """Tests for sort operator."""

    def test_sort_asc(self, engine, parser, planner):
        """Should sort ascending."""
        stmt = parser.parse("SELECT * FROM users ORDER BY age ASC")
        plan = planner.plan(stmt)

        rows = list(engine.execute(plan))

        ages = [r["users.age"] for r in rows]
        assert ages == sorted(ages)

    def test_sort_desc(self, engine, parser, planner):
        """Should sort descending."""
        stmt = parser.parse("SELECT * FROM users ORDER BY age DESC")
        plan = planner.plan(stmt)

        rows = list(engine.execute(plan))

        ages = [r["users.age"] for r in rows]
        assert ages == sorted(ages, reverse=True)


class TestLimit:
    """Tests for limit operator."""

    def test_limit(self, engine, parser, planner):
        """Should limit rows."""
        stmt = parser.parse("SELECT * FROM users LIMIT 3")
        plan = planner.plan(stmt)

        rows = list(engine.execute(plan))

        assert len(rows) == 3

    def test_limit_offset(self, engine, parser, planner):
        """Should skip offset rows."""
        stmt = parser.parse("SELECT * FROM users ORDER BY id LIMIT 2 OFFSET 2")
        plan = planner.plan(stmt)

        rows = list(engine.execute(plan))

        assert len(rows) == 2
        # Should get users 3 and 4
        ids = [r["users.id"] for r in rows]
        assert ids == [3, 4]


class TestJoin:
    """Tests for join operator."""

    def test_inner_join(self, engine, parser, planner):
        """Should perform inner join."""
        stmt = parser.parse(
            "SELECT * FROM users u "
            "INNER JOIN orders o ON u.id = o.user_id"
        )
        plan = planner.plan(stmt)

        rows = list(engine.execute(plan))

        # Users 1, 2, 3 have orders
        assert len(rows) == 4  # 2 orders for user 1, 1 for user 2, 1 for user 3


class TestComplexQueries:
    """Tests for complex queries."""

    def test_filter_sort_limit(self, engine, parser, planner):
        """Should combine filter, sort, and limit."""
        stmt = parser.parse(
            "SELECT * FROM users "
            "WHERE active = TRUE "
            "ORDER BY age DESC "
            "LIMIT 2"
        )
        plan = planner.plan(stmt)

        rows = list(engine.execute(plan))

        assert len(rows) == 2
        # Oldest active users: Alice (30), Diana (28)
        names = [r["users.name"] for r in rows]
        assert names == ["Alice", "Diana"]


class TestDML:
    """Tests for DML operations."""

    def test_insert(self, engine, parser, planner, heaps, catalog):
        """Should insert rows."""
        stmt = parser.parse("INSERT INTO users (id, name, age, active) VALUES (6, 'Frank', 40, TRUE)")
        plan = planner.plan(stmt)

        count = engine.execute_dml(plan)

        assert count == 1

        # Verify insert
        serializer = catalog.get_serializer("users")
        found = False
        for rid, data in heaps["users"].scan():
            row = serializer.deserialize(data)
            if row["id"] == 6:
                found = True
                assert row["name"] == "Frank"
                assert row["age"] == 40

        assert found
