"""
Unit tests for Client API.

Tests cover:
- Database creation and closing
- Table creation and dropping
- Insert, update, delete
- Queries
- Connections and transactions
- Metrics
"""

import pytest
import tempfile
from pathlib import Path

from clade.client.api import Database, Connection, open_database


@pytest.fixture
def db():
    """Create a test database."""
    database = Database("test_db", enable_wal=False)
    yield database
    database.close()


@pytest.fixture
def db_with_table(db):
    """Create a database with a test table."""
    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name VARCHAR(100), age INTEGER)")
    return db


class TestDatabaseLifecycle:
    """Tests for database lifecycle."""

    def test_create_database(self):
        """Should create a database."""
        db = Database("test")
        assert db._name == "test"
        db.close()

    def test_create_with_path(self):
        """Should create database at specified path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database("test", path=tmpdir)
            assert db._path == Path(tmpdir)
            db.close()

    def test_close_database(self):
        """Should close database."""
        db = Database("test")
        db.close()
        assert db._closed is True

    def test_context_manager(self):
        """Should work as context manager."""
        with Database("test") as db:
            assert db._closed is False
        assert db._closed is True

    def test_open_database_function(self):
        """Should create database via open_database."""
        db = open_database("test")
        assert db._name == "test"
        db.close()

    def test_repr(self):
        """Should return string representation."""
        with Database("test") as db:
            assert "test" in repr(db)


class TestDDL:
    """Tests for DDL operations."""

    def test_create_table(self, db):
        """Should create a table."""
        db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name VARCHAR(100))")
        assert "users" in db.tables

    def test_create_table_multiple_columns(self, db):
        """Should create table with multiple columns."""
        db.execute("""
            CREATE TABLE orders (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                total FLOAT,
                status VARCHAR(50)
            )
        """)
        assert "orders" in db.tables

    def test_drop_table(self, db):
        """Should drop a table."""
        db.execute("CREATE TABLE test (id INTEGER)")
        db.execute("DROP TABLE test")
        assert "test" not in db.tables

    def test_drop_table_if_exists(self, db):
        """Should not fail if table doesn't exist."""
        db.execute("DROP TABLE IF EXISTS nonexistent")
        # Should not raise


class TestDML:
    """Tests for DML operations."""

    def test_insert(self, db_with_table):
        """Should insert a row."""
        count = db_with_table.execute("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)")
        assert count == 1

    def test_insert_multiple(self, db_with_table):
        """Should insert multiple rows."""
        db_with_table.execute("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)")
        db_with_table.execute("INSERT INTO users (id, name, age) VALUES (2, 'Bob', 25)")

        rows = list(db_with_table.query("SELECT * FROM users"))
        assert len(rows) == 2


class TestQuery:
    """Tests for query operations."""

    def test_select_all(self, db_with_table):
        """Should select all rows."""
        db_with_table.execute("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)")
        db_with_table.execute("INSERT INTO users (id, name, age) VALUES (2, 'Bob', 25)")

        rows = list(db_with_table.query("SELECT * FROM users"))
        assert len(rows) == 2

    def test_select_with_where(self, db_with_table):
        """Should filter rows."""
        db_with_table.execute("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)")
        db_with_table.execute("INSERT INTO users (id, name, age) VALUES (2, 'Bob', 25)")

        rows = list(db_with_table.query("SELECT * FROM users WHERE age > 27"))
        assert len(rows) == 1

    def test_select_with_order_by(self, db_with_table):
        """Should order rows."""
        db_with_table.execute("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)")
        db_with_table.execute("INSERT INTO users (id, name, age) VALUES (2, 'Bob', 25)")

        rows = list(db_with_table.query("SELECT * FROM users ORDER BY age ASC"))
        ages = [r.get("users.age", r.get("age")) for r in rows]
        assert ages == sorted(ages)

    def test_select_with_limit(self, db_with_table):
        """Should limit rows."""
        db_with_table.execute("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)")
        db_with_table.execute("INSERT INTO users (id, name, age) VALUES (2, 'Bob', 25)")
        db_with_table.execute("INSERT INTO users (id, name, age) VALUES (3, 'Charlie', 35)")

        rows = list(db_with_table.query("SELECT * FROM users LIMIT 2"))
        assert len(rows) == 2


class TestConnection:
    """Tests for connection handling."""

    def test_connect(self, db_with_table):
        """Should create a connection."""
        conn = db_with_table.connect()
        assert conn is not None
        conn.close()

    def test_connection_context_manager(self, db_with_table):
        """Should work as context manager."""
        with db_with_table.connect() as conn:
            assert conn._closed is False
        assert conn._closed is True

    def test_connection_execute(self, db_with_table):
        """Should execute via connection."""
        with db_with_table.connect() as conn:
            conn.execute("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)")

        rows = list(db_with_table.query("SELECT * FROM users"))
        assert len(rows) == 1

    def test_connection_query(self, db_with_table):
        """Should query via connection."""
        db_with_table.execute("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)")

        with db_with_table.connect() as conn:
            rows = list(conn.query("SELECT * FROM users"))
            assert len(rows) == 1


class TestTransactions:
    """Tests for transaction handling."""

    def test_begin_transaction(self, db_with_table):
        """Should begin a transaction."""
        with db_with_table.connect() as conn:
            conn.begin()
            assert conn.in_transaction is True

    def test_commit_transaction(self, db_with_table):
        """Should commit a transaction."""
        with db_with_table.connect() as conn:
            conn.begin()
            conn.execute("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)")
            conn.commit()
            assert conn.in_transaction is False

        # Verify data persisted
        rows = list(db_with_table.query("SELECT * FROM users"))
        assert len(rows) == 1

    def test_rollback_transaction(self, db_with_table):
        """Should rollback a transaction."""
        with db_with_table.connect() as conn:
            conn.begin()
            conn.execute("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)")
            conn.rollback()
            assert conn.in_transaction is False

    def test_auto_rollback_on_close(self, db_with_table):
        """Should rollback on close if transaction open."""
        conn = db_with_table.connect()
        conn.begin()
        conn.execute("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)")
        conn.close()  # Should auto-rollback

    def test_begin_when_already_in_transaction(self, db_with_table):
        """Should fail if already in transaction."""
        with db_with_table.connect() as conn:
            conn.begin()
            with pytest.raises(RuntimeError, match="already in progress"):
                conn.begin()

    def test_commit_without_transaction(self, db_with_table):
        """Should fail if no transaction."""
        with db_with_table.connect() as conn:
            with pytest.raises(RuntimeError, match="No transaction"):
                conn.commit()


class TestMetrics:
    """Tests for metrics."""

    def test_metrics_enabled(self):
        """Should enable metrics by default."""
        with Database("test", enable_metrics=True) as db:
            assert db.metrics is not None

    def test_metrics_disabled(self):
        """Should allow disabling metrics."""
        with Database("test", enable_metrics=False) as db:
            assert db.metrics is None

    def test_export_metrics(self, db_with_table):
        """Should export metrics."""
        db_with_table.execute("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)")
        list(db_with_table.query("SELECT * FROM users"))

        output = db_with_table.export_metrics()
        assert "clade_queries_total" in output

    def test_query_metrics_recorded(self, db_with_table):
        """Should record query metrics."""
        db_with_table.execute("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)")

        metrics = db_with_table.metrics
        assert metrics.queries_total.get(labels={"type": "insert"}) >= 1


class TestCheckpoint:
    """Tests for checkpointing."""

    def test_checkpoint(self, db_with_table):
        """Should create checkpoint."""
        db_with_table.execute("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)")
        db_with_table.checkpoint()  # Should not raise


class TestErrorHandling:
    """Tests for error handling."""

    def test_closed_database_raises(self):
        """Should raise when using closed database."""
        db = Database("test")
        db.close()

        with pytest.raises(RuntimeError, match="closed"):
            db.execute("SELECT 1")

    def test_closed_connection_raises(self, db_with_table):
        """Should raise when using closed connection."""
        conn = db_with_table.connect()
        conn.close()

        with pytest.raises(RuntimeError, match="closed"):
            conn.execute("SELECT 1")


class TestTables:
    """Tests for table listing."""

    def test_list_tables(self, db):
        """Should list tables."""
        db.execute("CREATE TABLE a (id INTEGER)")
        db.execute("CREATE TABLE b (id INTEGER)")

        assert "a" in db.tables
        assert "b" in db.tables
