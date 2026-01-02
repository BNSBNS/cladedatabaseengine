"""
Unit tests for query planner and optimizer.

Tests cover:
- Logical plan generation
- Cost estimation
- Optimization rules
"""

import pytest
from clade.catalog.schema import Catalog, Table, Column, DataType
from clade.query.parser import SQLParser
from clade.query.planner import (
    QueryPlanner,
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
    PlanNodeType,
    JoinType,
    print_plan,
)
from clade.query.optimizer import Optimizer


@pytest.fixture
def catalog():
    """Create a test catalog with sample tables."""
    cat = Catalog()

    # Users table
    users_cols = [
        Column(name="id", data_type=DataType.INTEGER, primary_key=True),
        Column(name="name", data_type=DataType.VARCHAR, max_length=100),
        Column(name="email", data_type=DataType.VARCHAR, max_length=200),
        Column(name="active", data_type=DataType.BOOLEAN),
    ]
    cat.create_table(Table(name="users", columns=users_cols, primary_key=["id"]))

    # Orders table
    orders_cols = [
        Column(name="id", data_type=DataType.INTEGER, primary_key=True),
        Column(name="user_id", data_type=DataType.INTEGER),
        Column(name="total", data_type=DataType.FLOAT),
        Column(name="status", data_type=DataType.VARCHAR, max_length=50),
    ]
    cat.create_table(Table(name="orders", columns=orders_cols, primary_key=["id"]))

    return cat


@pytest.fixture
def parser():
    """Create a SQL parser."""
    return SQLParser()


@pytest.fixture
def planner(catalog):
    """Create a query planner."""
    return QueryPlanner(catalog)


@pytest.fixture
def optimizer():
    """Create an optimizer with sample statistics."""
    stats = {
        "users": {"num_pages": 10, "num_rows": 100},
        "orders": {"num_pages": 50, "num_rows": 1000},
    }
    return Optimizer(stats)


class TestSelectPlanning:
    """Tests for SELECT statement planning."""

    def test_select_all(self, parser, planner):
        """Should plan SELECT * FROM table."""
        stmt = parser.parse("SELECT * FROM users")
        plan = planner.plan(stmt)

        assert plan.node_type == PlanNodeType.SEQ_SCAN
        assert plan.table_name == "users"

    def test_select_with_columns(self, parser, planner):
        """Should plan SELECT with specific columns."""
        stmt = parser.parse("SELECT id, name FROM users")
        plan = planner.plan(stmt)

        assert plan.node_type == PlanNodeType.PROJECT
        assert plan.children[0].node_type == PlanNodeType.SEQ_SCAN

    def test_select_with_where(self, parser, planner):
        """Should plan SELECT with WHERE clause."""
        stmt = parser.parse("SELECT * FROM users WHERE id = 1")
        plan = planner.plan(stmt)

        assert plan.node_type == PlanNodeType.FILTER
        assert plan.children[0].node_type == PlanNodeType.SEQ_SCAN

    def test_select_with_order_by(self, parser, planner):
        """Should plan SELECT with ORDER BY."""
        stmt = parser.parse("SELECT * FROM users ORDER BY name ASC")
        plan = planner.plan(stmt)

        assert plan.node_type == PlanNodeType.SORT
        assert len(plan.sort_keys) == 1

    def test_select_with_limit(self, parser, planner):
        """Should plan SELECT with LIMIT."""
        stmt = parser.parse("SELECT * FROM users LIMIT 10")
        plan = planner.plan(stmt)

        assert plan.node_type == PlanNodeType.LIMIT
        assert plan.limit == 10

    def test_select_with_join(self, parser, planner):
        """Should plan SELECT with JOIN."""
        stmt = parser.parse(
            "SELECT * FROM users u "
            "INNER JOIN orders o ON u.id = o.user_id"
        )
        plan = planner.plan(stmt)

        assert plan.node_type == PlanNodeType.NESTED_LOOP_JOIN
        assert plan.join_type == JoinType.INNER

    def test_select_complex(self, parser, planner):
        """Should plan complex SELECT."""
        stmt = parser.parse(
            "SELECT id, name FROM users "
            "WHERE active = TRUE "
            "ORDER BY name DESC "
            "LIMIT 10 OFFSET 5"
        )
        plan = planner.plan(stmt)

        # Should be: Limit -> Sort -> Project -> Filter -> SeqScan
        assert plan.node_type == PlanNodeType.LIMIT
        assert plan.children[0].node_type == PlanNodeType.SORT
        assert plan.children[0].children[0].node_type == PlanNodeType.PROJECT


class TestDMLPlanning:
    """Tests for DML statement planning."""

    def test_insert(self, parser, planner):
        """Should plan INSERT statement."""
        stmt = parser.parse("INSERT INTO users (id, name) VALUES (1, 'Alice')")
        plan = planner.plan(stmt)

        assert plan.node_type == PlanNodeType.INSERT
        assert plan.table_name == "users"
        assert plan.columns == ["id", "name"]

    def test_update(self, parser, planner):
        """Should plan UPDATE statement."""
        stmt = parser.parse("UPDATE users SET name = 'Bob' WHERE id = 1")
        plan = planner.plan(stmt)

        assert plan.node_type == PlanNodeType.UPDATE
        assert plan.table_name == "users"
        assert len(plan.assignments) == 1

    def test_delete(self, parser, planner):
        """Should plan DELETE statement."""
        stmt = parser.parse("DELETE FROM users WHERE id = 1")
        plan = planner.plan(stmt)

        assert plan.node_type == PlanNodeType.DELETE
        assert plan.table_name == "users"


class TestDDLPlanning:
    """Tests for DDL statement planning."""

    def test_create_table(self, parser, planner):
        """Should plan CREATE TABLE statement."""
        stmt = parser.parse("CREATE TABLE test (id INTEGER, name VARCHAR(50))")
        plan = planner.plan(stmt)

        assert plan.node_type == PlanNodeType.CREATE_TABLE
        assert plan.table_def.table == "test"

    def test_drop_table(self, parser, planner):
        """Should plan DROP TABLE statement."""
        stmt = parser.parse("DROP TABLE users")
        plan = planner.plan(stmt)

        assert plan.node_type == PlanNodeType.DROP_TABLE
        assert plan.table_name == "users"

    def test_drop_table_if_exists(self, parser, planner):
        """Should plan DROP TABLE IF EXISTS."""
        stmt = parser.parse("DROP TABLE IF EXISTS users")
        plan = planner.plan(stmt)

        assert plan.node_type == PlanNodeType.DROP_TABLE
        assert plan.if_exists is True


class TestPlannerValidation:
    """Tests for planner validation."""

    def test_unknown_table_fails(self, parser, planner):
        """Should fail for unknown table."""
        stmt = parser.parse("SELECT * FROM nonexistent")

        with pytest.raises(ValueError, match="not found"):
            planner.plan(stmt)

    def test_unknown_column_in_insert_fails(self, parser, planner):
        """Should fail for unknown column in INSERT."""
        stmt = parser.parse("INSERT INTO users (nonexistent) VALUES (1)")

        with pytest.raises(ValueError, match="not found"):
            planner.plan(stmt)


class TestOptimizer:
    """Tests for query optimizer."""

    def test_predicate_pushdown_to_scan(self, parser, planner, optimizer):
        """Should push predicates down to scan."""
        stmt = parser.parse("SELECT * FROM users WHERE id = 1")
        plan = planner.plan(stmt)

        # Before optimization: Filter -> SeqScan
        assert plan.node_type == PlanNodeType.FILTER

        optimized = optimizer.optimize(plan)

        # After optimization: SeqScan with predicate
        assert optimized.node_type == PlanNodeType.SEQ_SCAN
        assert len(optimized.predicates) == 1

    def test_constant_folding(self, parser, planner, optimizer):
        """Should fold constant expressions."""
        stmt = parser.parse("SELECT * FROM users WHERE 1 = 1")
        plan = planner.plan(stmt)
        optimized = optimizer.optimize(plan)

        # Constant TRUE predicate should be eliminated
        # Filter should be removed
        assert optimized.node_type == PlanNodeType.SEQ_SCAN

    def test_cost_estimation(self, parser, planner, optimizer):
        """Should estimate costs for plan nodes."""
        stmt = parser.parse("SELECT * FROM users")
        plan = planner.plan(stmt)
        optimized = optimizer.optimize(plan)

        assert optimized.cost > 0

    def test_join_cost_estimation(self, parser, planner, optimizer):
        """Should estimate join costs."""
        stmt = parser.parse(
            "SELECT * FROM users u "
            "INNER JOIN orders o ON u.id = o.user_id"
        )
        plan = planner.plan(stmt)
        optimized = optimizer.optimize(plan)

        assert optimized.cost > 0


class TestPrintPlan:
    """Tests for plan printing."""

    def test_print_simple_plan(self, parser, planner):
        """Should print a simple plan."""
        stmt = parser.parse("SELECT * FROM users")
        plan = planner.plan(stmt)

        output = print_plan(plan)

        assert "SeqScan" in output
        assert "users" in output

    def test_print_complex_plan(self, parser, planner):
        """Should print a complex plan."""
        stmt = parser.parse(
            "SELECT id, name FROM users "
            "WHERE active = TRUE "
            "ORDER BY name "
            "LIMIT 10"
        )
        plan = planner.plan(stmt)

        output = print_plan(plan)

        assert "Limit" in output
        assert "Sort" in output
        assert "Project" in output
        assert "Filter" in output
        assert "SeqScan" in output
