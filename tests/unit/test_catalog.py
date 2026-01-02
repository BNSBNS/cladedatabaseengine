"""
Unit tests for catalog schema.

Tests cover:
- Data types
- Column definitions
- Table schemas
- Record serialization
- Catalog operations
"""

import pytest
from clade.catalog.schema import (
    DataType,
    Column,
    Table,
    RecordSerializer,
    Catalog,
)


class TestColumn:
    """Tests for Column definition."""

    def test_create_integer_column(self):
        """Should create an integer column."""
        col = Column(name="id", data_type=DataType.INTEGER)

        assert col.name == "id"
        assert col.data_type == DataType.INTEGER
        assert col.nullable is True
        assert col.is_fixed_size()
        assert col.get_size() == 4

    def test_create_varchar_column(self):
        """Should create a VARCHAR column with max length."""
        col = Column(name="name", data_type=DataType.VARCHAR, max_length=100)

        assert col.name == "name"
        assert col.data_type == DataType.VARCHAR
        assert col.max_length == 100
        assert not col.is_fixed_size()
        assert col.get_size() is None

    def test_varchar_requires_max_length(self):
        """VARCHAR without max_length should raise error."""
        with pytest.raises(ValueError, match="max_length"):
            Column(name="name", data_type=DataType.VARCHAR)

    def test_primary_key_not_nullable(self):
        """Primary key columns should not be nullable."""
        col = Column(name="id", data_type=DataType.INTEGER, primary_key=True)

        assert col.primary_key is True
        assert col.nullable is False

    def test_column_with_default(self):
        """Should create column with default value."""
        col = Column(name="status", data_type=DataType.INTEGER, default=0)

        assert col.default == 0


class TestTable:
    """Tests for Table schema."""

    def test_create_simple_table(self):
        """Should create a simple table."""
        cols = [
            Column(name="id", data_type=DataType.INTEGER, primary_key=True),
            Column(name="name", data_type=DataType.VARCHAR, max_length=100),
        ]
        table = Table(name="users", columns=cols, primary_key=["id"])

        assert table.name == "users"
        assert table.column_count == 2
        assert table.column_names == ["id", "name"]

    def test_get_column_by_name(self):
        """Should retrieve column by name."""
        cols = [
            Column(name="id", data_type=DataType.INTEGER),
            Column(name="name", data_type=DataType.VARCHAR, max_length=100),
        ]
        table = Table(name="test", columns=cols)

        col = table.get_column("name")
        assert col is not None
        assert col.name == "name"

        assert table.get_column("nonexistent") is None

    def test_get_column_index(self):
        """Should get column position."""
        cols = [
            Column(name="a", data_type=DataType.INTEGER),
            Column(name="b", data_type=DataType.INTEGER),
            Column(name="c", data_type=DataType.INTEGER),
        ]
        table = Table(name="test", columns=cols)

        assert table.get_column_index("a") == 0
        assert table.get_column_index("b") == 1
        assert table.get_column_index("c") == 2
        assert table.get_column_index("d") is None

    def test_invalid_primary_key(self):
        """Should reject non-existent primary key column."""
        cols = [Column(name="id", data_type=DataType.INTEGER)]

        with pytest.raises(ValueError, match="not found"):
            Table(name="test", columns=cols, primary_key=["nonexistent"])


class TestRecordSerializer:
    """Tests for record serialization."""

    @pytest.fixture
    def simple_table(self):
        """Create a simple test table."""
        cols = [
            Column(name="id", data_type=DataType.INTEGER),
            Column(name="name", data_type=DataType.VARCHAR, max_length=100),
            Column(name="active", data_type=DataType.BOOLEAN),
        ]
        return Table(name="test", columns=cols)

    def test_serialize_deserialize_roundtrip(self, simple_table):
        """Serialization and deserialization should preserve data."""
        serializer = RecordSerializer(simple_table)

        original = {"id": 42, "name": "Alice", "active": True}
        data = serializer.serialize(original)
        restored = serializer.deserialize(data)

        assert restored == original

    def test_serialize_with_null(self, simple_table):
        """Should handle NULL values."""
        serializer = RecordSerializer(simple_table)

        original = {"id": 42, "name": None, "active": True}
        data = serializer.serialize(original)
        restored = serializer.deserialize(data)

        assert restored == original

    def test_serialize_all_types(self):
        """Should handle all data types."""
        cols = [
            Column(name="int_col", data_type=DataType.INTEGER),
            Column(name="bigint_col", data_type=DataType.BIGINT),
            Column(name="float_col", data_type=DataType.FLOAT),
            Column(name="bool_col", data_type=DataType.BOOLEAN),
            Column(name="varchar_col", data_type=DataType.VARCHAR, max_length=50),
            Column(name="text_col", data_type=DataType.TEXT),
        ]
        table = Table(name="all_types", columns=cols)
        serializer = RecordSerializer(table)

        original = {
            "int_col": -12345,
            "bigint_col": 9223372036854775807,
            "float_col": 3.14159,
            "bool_col": False,
            "varchar_col": "Hello",
            "text_col": "World",
        }
        data = serializer.serialize(original)
        restored = serializer.deserialize(data)

        assert restored["int_col"] == original["int_col"]
        assert restored["bigint_col"] == original["bigint_col"]
        assert abs(restored["float_col"] - original["float_col"]) < 0.00001
        assert restored["bool_col"] == original["bool_col"]
        assert restored["varchar_col"] == original["varchar_col"]
        assert restored["text_col"] == original["text_col"]

    def test_varchar_max_length_enforced(self, simple_table):
        """Should reject values exceeding max_length."""
        serializer = RecordSerializer(simple_table)

        with pytest.raises(ValueError, match="too long"):
            serializer.serialize({"id": 1, "name": "x" * 200, "active": True})

    def test_not_null_enforced(self):
        """Should reject NULL for NOT NULL columns."""
        cols = [Column(name="id", data_type=DataType.INTEGER, nullable=False)]
        table = Table(name="test", columns=cols)
        serializer = RecordSerializer(table)

        with pytest.raises(ValueError, match="cannot be NULL"):
            serializer.serialize({"id": None})


class TestCatalog:
    """Tests for Catalog operations."""

    def test_create_table(self):
        """Should register a new table."""
        catalog = Catalog()
        cols = [Column(name="id", data_type=DataType.INTEGER)]
        table = Table(name="test", columns=cols)

        catalog.create_table(table)

        assert catalog.table_exists("test")
        assert catalog.get_table("test") is table

    def test_create_duplicate_table_fails(self):
        """Should reject duplicate table names."""
        catalog = Catalog()
        cols = [Column(name="id", data_type=DataType.INTEGER)]
        table = Table(name="test", columns=cols)

        catalog.create_table(table)

        with pytest.raises(ValueError, match="already exists"):
            catalog.create_table(table)

    def test_drop_table(self):
        """Should remove a table."""
        catalog = Catalog()
        cols = [Column(name="id", data_type=DataType.INTEGER)]
        table = Table(name="test", columns=cols)
        catalog.create_table(table)

        catalog.drop_table("test")

        assert not catalog.table_exists("test")

    def test_drop_nonexistent_table_fails(self):
        """Should fail when dropping nonexistent table."""
        catalog = Catalog()

        with pytest.raises(ValueError, match="does not exist"):
            catalog.drop_table("nonexistent")

    def test_get_serializer(self):
        """Should provide serializer for table."""
        catalog = Catalog()
        cols = [Column(name="id", data_type=DataType.INTEGER)]
        table = Table(name="test", columns=cols)
        catalog.create_table(table)

        serializer = catalog.get_serializer("test")

        assert serializer is not None
        assert isinstance(serializer, RecordSerializer)

    def test_table_names(self):
        """Should list all table names."""
        catalog = Catalog()

        for name in ["users", "orders", "products"]:
            cols = [Column(name="id", data_type=DataType.INTEGER)]
            catalog.create_table(Table(name=name, columns=cols))

        names = catalog.table_names

        assert set(names) == {"users", "orders", "products"}

    def test_repr(self):
        """Should have useful string representation."""
        catalog = Catalog()
        cols = [Column(name="id", data_type=DataType.INTEGER)]
        catalog.create_table(Table(name="test", columns=cols))

        repr_str = repr(catalog)

        assert "Catalog" in repr_str
        assert "test" in repr_str
