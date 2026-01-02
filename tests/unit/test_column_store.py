"""
Unit tests for columnar storage and vectorized execution.

Tests cover:
- Column segment compression
- Zone maps
- Column store operations
- Vectorized operators
"""

import pytest
import numpy as np

from clade.catalog.schema import DataType
from clade.storage.column_store import (
    ColumnSegment,
    ColumnStore,
    CompressionType,
    ZoneMap,
)
from clade.execution.vectorized import (
    Vector,
    VectorBatch,
    ColumnScanOperator,
    FilterOperator,
    AggregateOperator,
    VectorizedExecutor,
    VectorizedEvaluator,
)
from clade.query.parser import Expr


class TestZoneMap:
    """Tests for zone map statistics."""

    def test_may_contain(self):
        """Zone map should correctly check value containment."""
        zm = ZoneMap(min_val=10, max_val=100, null_count=0, row_count=50)

        assert zm.may_contain(50)
        assert zm.may_contain(10)
        assert zm.may_contain(100)
        assert not zm.may_contain(5)
        assert not zm.may_contain(101)

    def test_may_contain_null(self):
        """Zone map should track null values."""
        zm_with_null = ZoneMap(min_val=1, max_val=10, null_count=5, row_count=50)
        zm_no_null = ZoneMap(min_val=1, max_val=10, null_count=0, row_count=50)

        assert zm_with_null.may_contain(None)
        assert not zm_no_null.may_contain(None)

    def test_may_match_range(self):
        """Zone map should correctly check range overlap."""
        zm = ZoneMap(min_val=10, max_val=50, null_count=0, row_count=50)

        assert zm.may_match_range(20, 40)  # Within range
        assert zm.may_match_range(0, 15)  # Overlaps start
        assert zm.may_match_range(45, 100)  # Overlaps end
        assert not zm.may_match_range(0, 5)  # Below range
        assert not zm.may_match_range(60, 100)  # Above range


class TestColumnSegment:
    """Tests for column segment compression."""

    def test_uncompressed_roundtrip(self):
        """Should encode and decode uncompressed data."""
        segment = ColumnSegment(DataType.INTEGER, CompressionType.NONE)
        values = [1, 2, 3, None, 5, 6, 7, 8, 9, 10]

        segment.encode(values)
        result = segment.decode()

        assert result == values

    def test_rle_compression(self):
        """RLE should compress repeated values."""
        segment = ColumnSegment(DataType.INTEGER, CompressionType.RLE)
        values = [1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3]

        segment.encode(values)
        result = segment.decode()

        assert result == values

    def test_dictionary_compression(self):
        """Dictionary encoding should work for string values."""
        segment = ColumnSegment(DataType.TEXT, CompressionType.DICTIONARY)
        values = ["apple", "banana", "apple", "cherry", "banana", "apple"]

        segment.encode(values)
        result = segment.decode()

        assert result == values

    def test_bitpack_compression(self):
        """Bit-packing should work for small integers."""
        segment = ColumnSegment(DataType.INTEGER, CompressionType.BITPACK)
        values = [10, 11, 12, 13, 14, 15, 10, 11, 12]

        segment.encode(values)
        result = segment.decode()

        assert result == values

    def test_zone_map_creation(self):
        """Zone map should be created during encoding."""
        segment = ColumnSegment(DataType.INTEGER, CompressionType.NONE)
        values = [5, 10, None, 20, 15]

        segment.encode(values)

        zm = segment.zone_map
        assert zm is not None
        assert zm.min_val == 5
        assert zm.max_val == 20
        assert zm.null_count == 1
        assert zm.row_count == 5

    def test_vectorized_decode(self):
        """Should decode to numpy array."""
        segment = ColumnSegment(DataType.FLOAT, CompressionType.NONE)
        values = [1.0, 2.5, 3.7, 4.2]

        segment.encode(values)
        arr = segment.decode_vectorized()

        assert isinstance(arr, np.ndarray)
        assert len(arr) == 4
        assert np.allclose(arr, values)


class TestColumnStore:
    """Tests for column store operations."""

    def test_insert_batch(self):
        """Should insert a batch of rows."""
        store = ColumnStore()

        rows = [
            {"id": 1, "name": "Alice", "age": 30},
            {"id": 2, "name": "Bob", "age": 25},
            {"id": 3, "name": "Charlie", "age": 35},
        ]

        store.insert_batch(rows)

        assert store.row_count == 3
        assert "id" in store.column_names
        assert "name" in store.column_names
        assert "age" in store.column_names

    def test_scan_column(self):
        """Should scan a single column."""
        store = ColumnStore()

        rows = [
            {"id": 1, "value": 10},
            {"id": 2, "value": 20},
            {"id": 3, "value": 30},
        ]
        store.insert_batch(rows)

        values = list(store.scan_column("value"))

        assert values == [10, 20, 30]

    def test_scan_with_predicate(self):
        """Zone map should enable segment skipping (not row filtering)."""
        store = ColumnStore()

        # Insert multiple batches to create segments
        rows = [{"id": i, "value": i * 10} for i in range(100)]
        store.insert_batch(rows)

        # Scan with predicate - zone map only skips whole segments
        # Since all values fit in one segment, it returns all values
        # (Zone map filtering happens at segment level, not row level)
        values = list(store.scan_column("value", predicate=(">=", 500)))

        # With single segment, zone map can't skip anything
        assert len(values) == 100

    def test_scan_columns_vectorized(self):
        """Should return numpy arrays for columns."""
        store = ColumnStore()

        rows = [
            {"a": 1, "b": 10.5},
            {"a": 2, "b": 20.5},
            {"a": 3, "b": 30.5},
        ]
        store.insert_batch(rows)

        result = store.scan_columns_vectorized(["a", "b"])

        assert "a" in result
        assert "b" in result
        assert isinstance(result["a"], np.ndarray)
        assert list(result["a"]) == [1, 2, 3]

    def test_compression_stats(self):
        """Should report compression statistics."""
        store = ColumnStore()

        # Insert with compression
        rows = [{"category": "A"} for _ in range(1000)]
        rows.extend([{"category": "B"} for _ in range(1000)])

        store.insert_batch(rows, compression={"category": CompressionType.RLE})

        stats = store.get_compression_stats()
        assert "category" in stats


class TestVector:
    """Tests for Vector class."""

    def test_from_list(self):
        """Should create vector from list."""
        values = [1, 2, None, 4]
        vec = Vector.from_list(values, DataType.INTEGER)

        assert vec.size == 4
        assert vec[0] == 1
        assert vec[2] is None
        assert vec[3] == 4

    def test_filter(self):
        """Should filter vector by mask."""
        vec = Vector.from_list([1, 2, 3, 4, 5], DataType.INTEGER)
        mask = np.array([True, False, True, False, True])

        filtered = vec.filter(mask)

        assert filtered.size == 3
        assert filtered.to_list() == [1, 3, 5]


class TestVectorBatch:
    """Tests for VectorBatch class."""

    def test_add_and_get_column(self):
        """Should add and retrieve columns."""
        batch = VectorBatch()
        vec = Vector.from_list([1, 2, 3], DataType.INTEGER)

        batch.add_column("id", vec)

        assert batch.size == 3
        assert batch.get_column("id") is not None
        assert "id" in batch.column_names

    def test_filter_batch(self):
        """Should filter all columns by mask."""
        batch = VectorBatch()
        batch.add_column("a", Vector.from_list([1, 2, 3, 4], DataType.INTEGER))
        batch.add_column("b", Vector.from_list([10, 20, 30, 40], DataType.INTEGER))

        mask = np.array([True, False, True, False])
        filtered = batch.filter(mask)

        assert filtered.size == 2
        assert filtered.get_column("a").to_list() == [1, 3]
        assert filtered.get_column("b").to_list() == [10, 30]

    def test_to_dicts(self):
        """Should convert batch to dictionaries."""
        batch = VectorBatch()
        batch.add_column("id", Vector.from_list([1, 2], DataType.INTEGER))
        batch.add_column("name", Vector.from_list(["a", "b"], DataType.TEXT))

        dicts = list(batch.to_dicts())

        assert len(dicts) == 2
        assert dicts[0]["id"] == 1
        assert dicts[0]["name"] == "a"


class TestVectorizedEvaluator:
    """Tests for vectorized expression evaluation."""

    def test_evaluate_literal(self):
        """Should evaluate literal to boolean mask."""
        batch = VectorBatch()
        batch.add_column("x", Vector.from_list([1, 2, 3], DataType.INTEGER))

        expr = Expr("literal", [], True)
        mask = VectorizedEvaluator.evaluate(expr, batch)

        assert all(mask)

    def test_evaluate_comparison(self):
        """Should evaluate comparison expressions."""
        batch = VectorBatch()
        batch.add_column("x", Vector.from_list([1, 5, 10, 15, 20], DataType.INTEGER))

        # x > 10
        expr = Expr("binary", [
            Expr("column", [], (None, "x")),
            Expr("literal", [], 10),
        ], ">")

        mask = VectorizedEvaluator.evaluate(expr, batch)

        assert list(mask) == [False, False, False, True, True]

    def test_evaluate_and(self):
        """Should evaluate AND expressions."""
        batch = VectorBatch()
        batch.add_column("x", Vector.from_list([5, 10, 15, 20, 25], DataType.INTEGER))

        # x > 5 AND x < 20
        expr = Expr("binary", [
            Expr("binary", [
                Expr("column", [], (None, "x")),
                Expr("literal", [], 5),
            ], ">"),
            Expr("binary", [
                Expr("column", [], (None, "x")),
                Expr("literal", [], 20),
            ], "<"),
        ], "AND")

        mask = VectorizedEvaluator.evaluate(expr, batch)

        assert list(mask) == [False, True, True, False, False]


class TestVectorizedExecutor:
    """Tests for vectorized query execution."""

    @pytest.fixture
    def test_store(self):
        """Create a test column store."""
        store = ColumnStore()

        rows = [
            {"id": i, "category": chr(65 + i % 3), "value": i * 10}
            for i in range(100)
        ]
        store.insert_batch(rows)

        return store

    def test_execute_scan(self, test_store):
        """Should execute vectorized scan."""
        executor = VectorizedExecutor(test_store)

        batches = list(executor.execute_scan(["id", "value"]))

        total_rows = sum(b.size for b in batches)
        assert total_rows == 100

    def test_execute_scan_with_filter(self, test_store):
        """Should execute filtered scan."""
        executor = VectorizedExecutor(test_store)

        # Filter: value > 500
        filter_expr = Expr("binary", [
            Expr("column", [], (None, "value")),
            Expr("literal", [], 500),
        ], ">")

        batches = list(executor.execute_scan(["id", "value"], filter_expr))

        total_rows = sum(b.size for b in batches)
        # Values 510, 520, ..., 990 = 49 rows (indices 51-99)
        assert total_rows == 49

    def test_execute_aggregate(self, test_store):
        """Should execute aggregation."""
        executor = VectorizedExecutor(test_store)

        result = executor.execute_aggregate(
            columns=["value"],
            group_by=[],
            aggregates=[
                ("COUNT", "value", "cnt"),
                ("SUM", "value", "total"),
                ("AVG", "value", "avg"),
            ],
        )

        assert result.size == 1
        assert result.get_column("cnt")[0] == 100
        assert result.get_column("total")[0] == sum(i * 10 for i in range(100))

    def test_execute_grouped_aggregate(self, test_store):
        """Should execute grouped aggregation."""
        executor = VectorizedExecutor(test_store)

        result = executor.execute_aggregate(
            columns=["category", "value"],
            group_by=["category"],
            aggregates=[
                ("COUNT", "value", "cnt"),
            ],
        )

        # 3 groups: A, B, C
        assert result.size == 3
