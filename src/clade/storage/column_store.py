"""
Columnar Storage Engine for OLAP workloads.

Stores data in column-major format for efficient analytical queries.
Supports various compression schemes optimized for different data types.

Key Features:
- Column-major storage (one column per segment)
- Compression: RLE, Dictionary, Bit-packing
- Zone maps for predicate pushdown
- Batch (vectorized) reads

Performance Characteristics:
- Sequential reads for column scans
- Skip non-relevant columns
- Compression reduces I/O
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Dict, Any, Iterator, Tuple
import struct
import numpy as np

from clade.catalog.schema import DataType


class CompressionType(Enum):
    """Column compression types."""

    NONE = auto()  # Uncompressed
    RLE = auto()  # Run-length encoding
    DICTIONARY = auto()  # Dictionary encoding
    BITPACK = auto()  # Bit-packing for integers
    DELTA = auto()  # Delta encoding


@dataclass
class ZoneMap:
    """
    Min/max statistics for a column segment.

    Enables predicate pushdown by skipping segments that
    definitely don't contain matching values.
    """

    __slots__ = ("min_val", "max_val", "null_count", "row_count")

    min_val: Any
    max_val: Any
    null_count: int
    row_count: int

    def may_contain(self, value: Any) -> bool:
        """Check if segment may contain value."""
        if value is None:
            return self.null_count > 0
        if self.min_val is None or self.max_val is None:
            return True
        return self.min_val <= value <= self.max_val

    def may_match_range(self, low: Any, high: Any) -> bool:
        """Check if segment may have values in range."""
        if self.min_val is None or self.max_val is None:
            return True
        return not (self.max_val < low or self.min_val > high)


class ColumnSegment:
    """
    A compressed column segment.

    Stores values for a single column in a contiguous block.
    Supports multiple compression schemes.
    """

    __slots__ = (
        "_data_type",
        "_compression",
        "_raw_data",
        "_dictionary",
        "_zone_map",
        "_row_count",
    )

    SEGMENT_SIZE = 65536  # 64K rows per segment

    def __init__(
        self,
        data_type: DataType,
        compression: CompressionType = CompressionType.NONE,
    ) -> None:
        self._data_type = data_type
        self._compression = compression
        self._raw_data: bytes = b""
        self._dictionary: Optional[Dict[Any, int]] = None
        self._zone_map: Optional[ZoneMap] = None
        self._row_count = 0

    def encode(self, values: List[Any]) -> None:
        """
        Encode values into the segment.

        Args:
            values: List of values to encode
        """
        if not values:
            return

        self._row_count = len(values)

        # Build zone map
        non_null = [v for v in values if v is not None]
        self._zone_map = ZoneMap(
            min_val=min(non_null) if non_null else None,
            max_val=max(non_null) if non_null else None,
            null_count=values.count(None),
            row_count=len(values),
        )

        # Select compression based on data
        if self._compression == CompressionType.NONE:
            self._raw_data = self._encode_uncompressed(values)
        elif self._compression == CompressionType.RLE:
            self._raw_data = self._encode_rle(values)
        elif self._compression == CompressionType.DICTIONARY:
            self._raw_data = self._encode_dictionary(values)
        elif self._compression == CompressionType.BITPACK:
            self._raw_data = self._encode_bitpack(values)

    def decode(self) -> List[Any]:
        """
        Decode all values from the segment.

        Returns:
            List of decoded values
        """
        if not self._raw_data:
            return []

        if self._compression == CompressionType.NONE:
            return self._decode_uncompressed()
        elif self._compression == CompressionType.RLE:
            return self._decode_rle()
        elif self._compression == CompressionType.DICTIONARY:
            return self._decode_dictionary()
        elif self._compression == CompressionType.BITPACK:
            return self._decode_bitpack()

        return []

    def decode_vectorized(self) -> np.ndarray:
        """
        Decode to numpy array for vectorized processing.

        Returns:
            NumPy array of decoded values
        """
        values = self.decode()

        # Convert to appropriate numpy dtype
        if self._data_type == DataType.INTEGER:
            return np.array(values, dtype=np.int32)
        elif self._data_type == DataType.BIGINT:
            return np.array(values, dtype=np.int64)
        elif self._data_type == DataType.FLOAT:
            return np.array(values, dtype=np.float64)
        elif self._data_type == DataType.BOOLEAN:
            return np.array(values, dtype=np.bool_)
        else:
            return np.array(values, dtype=object)

    # ─────────────────────────────────────────────────────────────────────────
    # Compression Encoders
    # ─────────────────────────────────────────────────────────────────────────

    def _encode_uncompressed(self, values: List[Any]) -> bytes:
        """Encode values without compression."""
        result = bytearray()

        # Null bitmap
        null_bitmap = self._build_null_bitmap(values)
        result.extend(struct.pack("<I", len(null_bitmap)))
        result.extend(null_bitmap)

        # Values
        for v in values:
            if v is not None:
                result.extend(self._encode_value(v))

        return bytes(result)

    def _decode_uncompressed(self) -> List[Any]:
        """Decode uncompressed values."""
        offset = 0

        # Read null bitmap
        bitmap_len = struct.unpack_from("<I", self._raw_data, offset)[0]
        offset += 4
        null_bitmap = self._raw_data[offset : offset + bitmap_len]
        offset += bitmap_len

        # Read values
        values = []
        for i in range(self._row_count):
            if self._is_null_in_bitmap(null_bitmap, i):
                values.append(None)
            else:
                val, size = self._decode_value(self._raw_data, offset)
                values.append(val)
                offset += size

        return values

    def _encode_rle(self, values: List[Any]) -> bytes:
        """Encode using run-length encoding."""
        result = bytearray()

        # Null bitmap first
        null_bitmap = self._build_null_bitmap(values)
        result.extend(struct.pack("<I", len(null_bitmap)))
        result.extend(null_bitmap)

        # RLE encode runs of identical values
        if not values:
            result.extend(struct.pack("<I", 0))  # 0 runs
            return bytes(result)

        runs = []
        current_val = values[0]
        current_count = 1

        for v in values[1:]:
            if v == current_val:
                current_count += 1
            else:
                runs.append((current_val, current_count))
                current_val = v
                current_count = 1
        runs.append((current_val, current_count))

        # Write runs
        result.extend(struct.pack("<I", len(runs)))
        for val, count in runs:
            result.extend(struct.pack("<I", count))
            if val is None:
                result.extend(struct.pack("<?", True))
            else:
                result.extend(struct.pack("<?", False))
                result.extend(self._encode_value(val))

        return bytes(result)

    def _decode_rle(self) -> List[Any]:
        """Decode RLE-encoded values."""
        offset = 0

        # Skip null bitmap
        bitmap_len = struct.unpack_from("<I", self._raw_data, offset)[0]
        offset += 4 + bitmap_len

        # Read runs
        num_runs = struct.unpack_from("<I", self._raw_data, offset)[0]
        offset += 4

        values = []
        for _ in range(num_runs):
            count = struct.unpack_from("<I", self._raw_data, offset)[0]
            offset += 4
            is_null = struct.unpack_from("<?", self._raw_data, offset)[0]
            offset += 1

            if is_null:
                values.extend([None] * count)
            else:
                val, size = self._decode_value(self._raw_data, offset)
                offset += size
                values.extend([val] * count)

        return values

    def _encode_dictionary(self, values: List[Any]) -> bytes:
        """Encode using dictionary encoding."""
        result = bytearray()

        # Build dictionary
        unique_vals = list(set(v for v in values if v is not None))
        self._dictionary = {v: i for i, v in enumerate(unique_vals)}

        # Null bitmap
        null_bitmap = self._build_null_bitmap(values)
        result.extend(struct.pack("<I", len(null_bitmap)))
        result.extend(null_bitmap)

        # Write dictionary
        result.extend(struct.pack("<I", len(unique_vals)))
        for v in unique_vals:
            encoded = self._encode_value(v)
            result.extend(struct.pack("<I", len(encoded)))
            result.extend(encoded)

        # Write indices
        for v in values:
            if v is None:
                result.extend(struct.pack("<I", 0xFFFFFFFF))
            else:
                result.extend(struct.pack("<I", self._dictionary[v]))

        return bytes(result)

    def _decode_dictionary(self) -> List[Any]:
        """Decode dictionary-encoded values."""
        offset = 0

        # Read null bitmap
        bitmap_len = struct.unpack_from("<I", self._raw_data, offset)[0]
        offset += 4 + bitmap_len

        # Read dictionary
        dict_size = struct.unpack_from("<I", self._raw_data, offset)[0]
        offset += 4

        dictionary = []
        for _ in range(dict_size):
            val_len = struct.unpack_from("<I", self._raw_data, offset)[0]
            offset += 4
            val, _ = self._decode_value(self._raw_data, offset)
            dictionary.append(val)
            offset += val_len

        # Read indices
        values = []
        for _ in range(self._row_count):
            idx = struct.unpack_from("<I", self._raw_data, offset)[0]
            offset += 4
            if idx == 0xFFFFFFFF:
                values.append(None)
            else:
                values.append(dictionary[idx])

        return values

    def _encode_bitpack(self, values: List[Any]) -> bytes:
        """Encode integers using bit-packing."""
        if self._data_type not in (DataType.INTEGER, DataType.BIGINT):
            return self._encode_uncompressed(values)

        result = bytearray()

        # Null bitmap
        null_bitmap = self._build_null_bitmap(values)
        result.extend(struct.pack("<I", len(null_bitmap)))
        result.extend(null_bitmap)

        # Find min and bit width needed
        non_null = [v for v in values if v is not None]
        if not non_null:
            result.extend(struct.pack("<q", 0))  # min
            result.extend(struct.pack("<B", 0))  # bits
            return bytes(result)

        min_val = min(non_null)
        max_val = max(non_null)
        range_val = max_val - min_val

        bits_needed = range_val.bit_length() if range_val > 0 else 1

        result.extend(struct.pack("<q", min_val))
        result.extend(struct.pack("<B", bits_needed))

        # Pack values
        buffer = 0
        buffer_bits = 0
        packed = bytearray()

        for v in values:
            if v is not None:
                delta = v - min_val
                buffer |= delta << buffer_bits
                buffer_bits += bits_needed

                while buffer_bits >= 8:
                    packed.append(buffer & 0xFF)
                    buffer >>= 8
                    buffer_bits -= 8

        if buffer_bits > 0:
            packed.append(buffer & 0xFF)

        result.extend(packed)
        return bytes(result)

    def _decode_bitpack(self) -> List[Any]:
        """Decode bit-packed integers."""
        offset = 0

        # Read null bitmap
        bitmap_len = struct.unpack_from("<I", self._raw_data, offset)[0]
        offset += 4
        null_bitmap = self._raw_data[offset : offset + bitmap_len]
        offset += bitmap_len

        # Read min and bit width
        min_val = struct.unpack_from("<q", self._raw_data, offset)[0]
        offset += 8
        bits_needed = struct.unpack_from("<B", self._raw_data, offset)[0]
        offset += 1

        if bits_needed == 0:
            return [None if self._is_null_in_bitmap(null_bitmap, i) else min_val
                    for i in range(self._row_count)]

        # Unpack values
        packed = self._raw_data[offset:]
        values = []
        buffer = 0
        buffer_bits = 0
        byte_idx = 0
        mask = (1 << bits_needed) - 1

        for i in range(self._row_count):
            if self._is_null_in_bitmap(null_bitmap, i):
                values.append(None)
                continue

            while buffer_bits < bits_needed and byte_idx < len(packed):
                buffer |= packed[byte_idx] << buffer_bits
                buffer_bits += 8
                byte_idx += 1

            delta = buffer & mask
            buffer >>= bits_needed
            buffer_bits -= bits_needed
            values.append(min_val + delta)

        return values

    # ─────────────────────────────────────────────────────────────────────────
    # Helper Methods
    # ─────────────────────────────────────────────────────────────────────────

    def _build_null_bitmap(self, values: List[Any]) -> bytes:
        """Build null bitmap from values."""
        bitmap = bytearray((len(values) + 7) // 8)
        for i, v in enumerate(values):
            if v is None:
                bitmap[i // 8] |= 1 << (i % 8)
        return bytes(bitmap)

    def _is_null_in_bitmap(self, bitmap: bytes, idx: int) -> bool:
        """Check if value at index is null."""
        return (bitmap[idx // 8] >> (idx % 8)) & 1 == 1

    def _encode_value(self, value: Any) -> bytes:
        """Encode a single value."""
        if self._data_type == DataType.INTEGER:
            return struct.pack("<i", value)
        elif self._data_type == DataType.BIGINT:
            return struct.pack("<q", value)
        elif self._data_type == DataType.FLOAT:
            return struct.pack("<d", value)
        elif self._data_type == DataType.BOOLEAN:
            return struct.pack("<?", value)
        elif self._data_type in (DataType.VARCHAR, DataType.TEXT):
            encoded = value.encode("utf-8")
            return struct.pack("<I", len(encoded)) + encoded
        else:
            return b""

    def _decode_value(self, data: bytes, offset: int) -> Tuple[Any, int]:
        """Decode a single value, return (value, bytes_consumed)."""
        if self._data_type == DataType.INTEGER:
            return struct.unpack_from("<i", data, offset)[0], 4
        elif self._data_type == DataType.BIGINT:
            return struct.unpack_from("<q", data, offset)[0], 8
        elif self._data_type == DataType.FLOAT:
            return struct.unpack_from("<d", data, offset)[0], 8
        elif self._data_type == DataType.BOOLEAN:
            return struct.unpack_from("<?", data, offset)[0], 1
        elif self._data_type in (DataType.VARCHAR, DataType.TEXT):
            length = struct.unpack_from("<I", data, offset)[0]
            text = data[offset + 4 : offset + 4 + length].decode("utf-8")
            return text, 4 + length
        else:
            return None, 0

    @property
    def zone_map(self) -> Optional[ZoneMap]:
        """Get zone map for this segment."""
        return self._zone_map

    @property
    def compression_ratio(self) -> float:
        """Get compression ratio."""
        if not self._row_count:
            return 1.0
        # Estimate uncompressed size
        if self._data_type == DataType.INTEGER:
            uncompressed = self._row_count * 4
        elif self._data_type in (DataType.BIGINT, DataType.FLOAT):
            uncompressed = self._row_count * 8
        else:
            uncompressed = len(self._raw_data)

        return uncompressed / max(1, len(self._raw_data))

    def __repr__(self) -> str:
        return (
            f"ColumnSegment({self._data_type.name}, "
            f"{self._compression.name}, {self._row_count} rows)"
        )


class ColumnStore:
    """
    Columnar storage engine.

    Stores table data in column-major format with compression.
    Optimized for analytical workloads with large scans.
    """

    __slots__ = ("_columns", "_row_count", "_segment_size")

    def __init__(self, segment_size: int = ColumnSegment.SEGMENT_SIZE) -> None:
        """Initialize column store."""
        self._columns: Dict[str, List[ColumnSegment]] = {}
        self._row_count = 0
        self._segment_size = segment_size

    def add_column(
        self,
        name: str,
        data_type: DataType,
        compression: CompressionType = CompressionType.NONE,
    ) -> None:
        """
        Add a column to the store.

        Args:
            name: Column name
            data_type: Column data type
            compression: Compression type to use
        """
        self._columns[name] = []

    def insert_batch(
        self,
        rows: List[Dict[str, Any]],
        compression: Optional[Dict[str, CompressionType]] = None,
    ) -> None:
        """
        Insert a batch of rows.

        Args:
            rows: List of row dictionaries
            compression: Optional per-column compression settings
        """
        if not rows:
            return

        compression = compression or {}

        # Pivot rows to columns
        for col_name in rows[0].keys():
            if col_name not in self._columns:
                # Infer data type from first non-null value
                values = [r.get(col_name) for r in rows]
                dtype = self._infer_type(values)
                self._columns[col_name] = []

            values = [r.get(col_name) for r in rows]

            # Create segment(s)
            for i in range(0, len(values), self._segment_size):
                segment_values = values[i : i + self._segment_size]
                dtype = self._infer_type(segment_values)
                comp = compression.get(col_name, CompressionType.NONE)
                segment = ColumnSegment(dtype, comp)
                segment.encode(segment_values)
                self._columns[col_name].append(segment)

        self._row_count += len(rows)

    def scan_column(
        self,
        name: str,
        predicate: Optional[Tuple[str, Any]] = None,
    ) -> Iterator[Any]:
        """
        Scan a single column.

        Args:
            name: Column name
            predicate: Optional (operator, value) for zone map filtering

        Yields:
            Column values
        """
        if name not in self._columns:
            return

        for segment in self._columns[name]:
            # Check zone map
            if predicate and segment.zone_map:
                op, val = predicate
                zm = segment.zone_map

                if op == "=" and not zm.may_contain(val):
                    # Skip this segment
                    continue
                elif op in ("<", "<=") and zm.min_val is not None and zm.min_val > val:
                    continue
                elif op in (">", ">=") and zm.max_val is not None and zm.max_val < val:
                    continue

            # Decode segment
            for value in segment.decode():
                yield value

    def scan_columns_vectorized(
        self,
        names: List[str],
    ) -> Dict[str, np.ndarray]:
        """
        Scan multiple columns as numpy arrays.

        Args:
            names: List of column names

        Returns:
            Dictionary of column name to numpy array
        """
        result = {}
        for name in names:
            if name not in self._columns:
                continue

            arrays = []
            for segment in self._columns[name]:
                arrays.append(segment.decode_vectorized())

            if arrays:
                result[name] = np.concatenate(arrays)
            else:
                result[name] = np.array([])

        return result

    def _infer_type(self, values: List[Any]) -> DataType:
        """Infer data type from values."""
        for v in values:
            if v is None:
                continue
            if isinstance(v, bool):
                return DataType.BOOLEAN
            elif isinstance(v, int):
                return DataType.INTEGER
            elif isinstance(v, float):
                return DataType.FLOAT
            elif isinstance(v, str):
                return DataType.TEXT
        return DataType.NULL

    @property
    def row_count(self) -> int:
        """Get total row count."""
        return self._row_count

    @property
    def column_names(self) -> List[str]:
        """Get list of column names."""
        return list(self._columns.keys())

    def get_compression_stats(self) -> Dict[str, float]:
        """Get compression ratios for all columns."""
        stats = {}
        for name, segments in self._columns.items():
            if segments:
                ratios = [s.compression_ratio for s in segments]
                stats[name] = sum(ratios) / len(ratios)
        return stats

    def __repr__(self) -> str:
        return f"ColumnStore({len(self._columns)} columns, {self._row_count} rows)"
