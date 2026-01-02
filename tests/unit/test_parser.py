"""
Unit tests for SQL parser.

Tests cover:
- SELECT statements
- INSERT statements
- UPDATE statements
- DELETE statements
- CREATE TABLE statements
- DROP TABLE statements
- Expressions and operators
"""

import pytest
from clade.query.parser import (
    SQLParser,
    SelectStmt,
    InsertStmt,
    UpdateStmt,
    DeleteStmt,
    CreateTableStmt,
    DropTableStmt,
    Expr,
)
from clade.catalog.schema import DataType


@pytest.fixture
def parser():
    """Create a SQL parser."""
    return SQLParser()


class TestSelectStatements:
    """Tests for SELECT statement parsing."""

    def test_select_all(self, parser):
        """Should parse SELECT * FROM table."""
        stmt = parser.parse("SELECT * FROM users")

        assert isinstance(stmt, SelectStmt)
        assert stmt.is_select_all is True
        assert stmt.table == "users"
        assert stmt.where is None

    def test_select_columns(self, parser):
        """Should parse column list."""
        stmt = parser.parse("SELECT id, name, email FROM users")

        assert isinstance(stmt, SelectStmt)
        assert stmt.is_select_all is False
        assert len(stmt.columns) == 3

    def test_select_with_alias(self, parser):
        """Should parse column aliases."""
        stmt = parser.parse("SELECT id AS user_id, name AS user_name FROM users")

        assert len(stmt.columns) == 2
        assert stmt.columns[0][1] == "user_id"
        assert stmt.columns[1][1] == "user_name"

    def test_select_with_where(self, parser):
        """Should parse WHERE clause."""
        stmt = parser.parse("SELECT * FROM users WHERE id = 1")

        assert stmt.where is not None
        assert stmt.where.op == "binary"
        assert stmt.where.value == "="

    def test_select_with_and_or(self, parser):
        """Should parse AND/OR conditions."""
        stmt = parser.parse("SELECT * FROM users WHERE id > 1 AND name = 'Alice'")

        assert stmt.where is not None
        assert stmt.where.op == "binary"
        assert stmt.where.value == "AND"

    def test_select_with_order_by(self, parser):
        """Should parse ORDER BY clause."""
        stmt = parser.parse("SELECT * FROM users ORDER BY name ASC, id DESC")

        assert stmt.order_by is not None
        assert len(stmt.order_by) == 2
        assert stmt.order_by[0][1] == "ASC"
        assert stmt.order_by[1][1] == "DESC"

    def test_select_with_limit(self, parser):
        """Should parse LIMIT clause."""
        stmt = parser.parse("SELECT * FROM users LIMIT 10")

        assert stmt.limit == 10
        assert stmt.offset == 0

    def test_select_with_limit_offset(self, parser):
        """Should parse LIMIT with OFFSET."""
        stmt = parser.parse("SELECT * FROM users LIMIT 10 OFFSET 20")

        assert stmt.limit == 10
        assert stmt.offset == 20

    def test_select_with_table_alias(self, parser):
        """Should parse table alias."""
        stmt = parser.parse("SELECT u.id FROM users u WHERE u.id = 1")

        assert stmt.table == "users"
        assert stmt.table_alias == "u"

    def test_select_with_join(self, parser):
        """Should parse JOIN clause."""
        stmt = parser.parse(
            "SELECT * FROM users u "
            "INNER JOIN orders o ON u.id = o.user_id"
        )

        assert len(stmt.joins) == 1
        assert stmt.joins[0]["type"] == "INNER"
        assert stmt.joins[0]["table"] == "orders"
        assert stmt.joins[0]["alias"] == "o"

    def test_select_with_left_join(self, parser):
        """Should parse LEFT JOIN."""
        stmt = parser.parse(
            "SELECT * FROM users u "
            "LEFT JOIN orders o ON u.id = o.user_id"
        )

        assert stmt.joins[0]["type"] == "LEFT"


class TestInsertStatements:
    """Tests for INSERT statement parsing."""

    def test_insert_basic(self, parser):
        """Should parse basic INSERT."""
        stmt = parser.parse("INSERT INTO users VALUES (1, 'Alice', TRUE)")

        assert isinstance(stmt, InsertStmt)
        assert stmt.table == "users"
        assert stmt.columns is None
        assert len(stmt.values) == 1
        assert len(stmt.values[0]) == 3

    def test_insert_with_columns(self, parser):
        """Should parse INSERT with column list."""
        stmt = parser.parse(
            "INSERT INTO users (id, name) VALUES (1, 'Alice')"
        )

        assert stmt.columns == ["id", "name"]
        assert len(stmt.values[0]) == 2

    def test_insert_values(self, parser):
        """Should correctly parse literal values."""
        stmt = parser.parse(
            "INSERT INTO test VALUES (42, 3.14, 'hello', TRUE, NULL)"
        )

        values = stmt.values[0]
        assert values[0].value == 42
        assert values[1].value == 3.14
        assert values[2].value == "hello"
        assert values[3].value is True
        assert values[4].value is None


class TestUpdateStatements:
    """Tests for UPDATE statement parsing."""

    def test_update_basic(self, parser):
        """Should parse basic UPDATE."""
        stmt = parser.parse("UPDATE users SET name = 'Bob'")

        assert isinstance(stmt, UpdateStmt)
        assert stmt.table == "users"
        assert len(stmt.assignments) == 1
        assert stmt.assignments[0][0] == "name"
        assert stmt.where is None

    def test_update_multiple_columns(self, parser):
        """Should parse UPDATE with multiple columns."""
        stmt = parser.parse("UPDATE users SET name = 'Bob', age = 30")

        assert len(stmt.assignments) == 2

    def test_update_with_where(self, parser):
        """Should parse UPDATE with WHERE."""
        stmt = parser.parse("UPDATE users SET name = 'Bob' WHERE id = 1")

        assert stmt.where is not None
        assert stmt.where.value == "="


class TestDeleteStatements:
    """Tests for DELETE statement parsing."""

    def test_delete_basic(self, parser):
        """Should parse basic DELETE."""
        stmt = parser.parse("DELETE FROM users")

        assert isinstance(stmt, DeleteStmt)
        assert stmt.table == "users"
        assert stmt.where is None

    def test_delete_with_where(self, parser):
        """Should parse DELETE with WHERE."""
        stmt = parser.parse("DELETE FROM users WHERE id = 1")

        assert stmt.where is not None


class TestCreateTableStatements:
    """Tests for CREATE TABLE statement parsing."""

    def test_create_table_basic(self, parser):
        """Should parse basic CREATE TABLE."""
        stmt = parser.parse(
            "CREATE TABLE users (id INTEGER, name VARCHAR(100))"
        )

        assert isinstance(stmt, CreateTableStmt)
        assert stmt.table == "users"
        assert len(stmt.columns) == 2

    def test_create_table_with_types(self, parser):
        """Should parse various data types."""
        stmt = parser.parse(
            "CREATE TABLE test ("
            "  int_col INTEGER,"
            "  bigint_col BIGINT,"
            "  float_col FLOAT,"
            "  bool_col BOOLEAN,"
            "  varchar_col VARCHAR(50),"
            "  text_col TEXT"
            ")"
        )

        assert stmt.columns[0]["type"] == DataType.INTEGER
        assert stmt.columns[1]["type"] == DataType.BIGINT
        assert stmt.columns[2]["type"] == DataType.FLOAT
        assert stmt.columns[3]["type"] == DataType.BOOLEAN
        assert stmt.columns[4]["type"] == DataType.VARCHAR
        assert stmt.columns[4]["max_length"] == 50
        assert stmt.columns[5]["type"] == DataType.TEXT

    def test_create_table_with_constraints(self, parser):
        """Should parse column constraints."""
        stmt = parser.parse(
            "CREATE TABLE users ("
            "  id INTEGER PRIMARY KEY,"
            "  name VARCHAR(100) NOT NULL,"
            "  status INTEGER DEFAULT 0"
            ")"
        )

        assert stmt.columns[0]["pk"] is True
        assert stmt.columns[1]["nullable"] is False
        assert stmt.columns[2]["default"] is not None

    def test_create_table_with_table_constraint(self, parser):
        """Should parse table-level PRIMARY KEY."""
        stmt = parser.parse(
            "CREATE TABLE test ("
            "  a INTEGER,"
            "  b INTEGER,"
            "  PRIMARY KEY (a, b)"
            ")"
        )

        assert stmt.primary_key == ["a", "b"]


class TestDropTableStatements:
    """Tests for DROP TABLE statement parsing."""

    def test_drop_table(self, parser):
        """Should parse DROP TABLE."""
        stmt = parser.parse("DROP TABLE users")

        assert isinstance(stmt, DropTableStmt)
        assert stmt.table == "users"
        assert stmt.if_exists is False

    def test_drop_table_if_exists(self, parser):
        """Should parse DROP TABLE IF EXISTS."""
        stmt = parser.parse("DROP TABLE IF EXISTS users")

        assert stmt.if_exists is True


class TestExpressions:
    """Tests for expression parsing."""

    def test_arithmetic_operations(self, parser):
        """Should parse arithmetic expressions."""
        stmt = parser.parse("SELECT 1 + 2 * 3 FROM test")

        expr = stmt.columns[0][0]
        assert expr.op == "binary"
        assert expr.value == "+"

    def test_comparison_operators(self, parser):
        """Should parse comparison operators."""
        operators = ["=", "!=", "<", ">", "<=", ">="]

        for op in operators:
            stmt = parser.parse(f"SELECT * FROM test WHERE a {op} b")
            assert stmt.where is not None

    def test_like_operator(self, parser):
        """Should parse LIKE operator."""
        stmt = parser.parse("SELECT * FROM users WHERE name LIKE '%Alice%'")

        assert stmt.where.value == "LIKE"

    def test_is_null(self, parser):
        """Should parse IS NULL."""
        stmt = parser.parse("SELECT * FROM users WHERE name IS NULL")

        assert stmt.where.value == "IS NULL"

    def test_is_not_null(self, parser):
        """Should parse IS NOT NULL."""
        stmt = parser.parse("SELECT * FROM users WHERE name IS NOT NULL")

        assert stmt.where.value == "IS NOT NULL"

    def test_between(self, parser):
        """Should parse BETWEEN."""
        stmt = parser.parse("SELECT * FROM users WHERE age BETWEEN 18 AND 65")

        assert stmt.where.op == "between"

    def test_in_list(self, parser):
        """Should parse IN list."""
        stmt = parser.parse("SELECT * FROM users WHERE id IN (1, 2, 3)")

        assert stmt.where.op == "in"

    def test_function_call(self, parser):
        """Should parse function calls."""
        stmt = parser.parse("SELECT COUNT(*) FROM users")

        expr = stmt.columns[0][0]
        assert expr.op == "function"
        assert expr.value == "COUNT"

    def test_not_operator(self, parser):
        """Should parse NOT operator."""
        stmt = parser.parse("SELECT * FROM users WHERE NOT active")

        assert stmt.where.value == "NOT"


class TestCaseInsensitivity:
    """Tests for case insensitivity."""

    def test_keywords_case_insensitive(self, parser):
        """SQL keywords should be case insensitive."""
        # All variations should parse successfully
        parser.parse("select * from users")
        parser.parse("SELECT * FROM users")
        parser.parse("Select * From Users")
        parser.parse("sElEcT * fRoM uSeRs")

    def test_identifiers_case_sensitive(self, parser):
        """Table/column names should preserve case."""
        stmt = parser.parse("SELECT MyColumn FROM MyTable")

        assert stmt.table == "MyTable"


class TestComplexQueries:
    """Tests for complex query patterns."""

    def test_full_select(self, parser):
        """Should parse complex SELECT."""
        stmt = parser.parse(
            "SELECT u.id, u.name, COUNT(*) AS order_count "
            "FROM users u "
            "LEFT JOIN orders o ON u.id = o.user_id "
            "WHERE u.active = TRUE "
            "ORDER BY order_count DESC "
            "LIMIT 10 OFFSET 0"
        )

        assert stmt.table == "users"
        assert stmt.table_alias == "u"
        assert len(stmt.joins) == 1
        assert stmt.where is not None
        assert stmt.order_by is not None
        assert stmt.limit == 10

    def test_nested_expressions(self, parser):
        """Should parse nested expressions."""
        stmt = parser.parse(
            "SELECT * FROM test WHERE (a > 1 AND b < 2) OR (c = 3)"
        )

        assert stmt.where is not None
        assert stmt.where.value == "OR"
