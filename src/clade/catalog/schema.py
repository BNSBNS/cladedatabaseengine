"""
Schema definitions for the database catalog.

Provides data types, column definitions, and table schemas.
The catalog is the source of truth for database metadata.

Key Components:
- DataType: Supported data types (INT, VARCHAR, etc.)
- Column: Column definition with name, type, constraints
- Table: Table schema with columns and indexes
- Catalog: In-memory catalog manager
"""

from enum import Enum, auto
from typing import Optional, Dict, List, Any
import struct


class DataType(Enum):
    """Supported data types."""

    INTEGER = auto()  # 4-byte signed integer
    BIGINT = auto()  # 8-byte signed integer
    FLOAT = auto()  # 8-byte double precision
    BOOLEAN = auto()  # 1-byte boolean
    VARCHAR = auto()  # Variable-length string (with max length)
    TEXT = auto()  # Unlimited length string
    BLOB = auto()  # Binary data
    NULL = auto()  # NULL type (for expressions)


# Type size mapping (for fixed-size types)
TYPE_SIZES = {
    DataType.INTEGER: 4,
    DataType.BIGINT: 8,
    DataType.FLOAT: 8,
    DataType.BOOLEAN: 1,
}


class Column:
    """
    Column definition in a table schema.

    Attributes:
        name: Column name
        data_type: Data type
        nullable: Whether NULL values are allowed
        max_length: Maximum length for VARCHAR
        default: Default value
        primary_key: Is this column a primary key
    """

    __slots__ = ("name", "data_type", "nullable", "max_length", "default", "primary_key")

    def __init__(
        self,
        name: str,
        data_type: DataType,
        nullable: bool = True,
        max_length: Optional[int] = None,
        default: Optional[Any] = None,
        primary_key: bool = False,
    ) -> None:
        self.name = name
        self.data_type = data_type
        self.nullable = nullable
        self.max_length = max_length
        self.default = default
        self.primary_key = primary_key

        # Validation
        if self.data_type == DataType.VARCHAR and self.max_length is None:
            raise ValueError("VARCHAR requires max_length")
        if self.primary_key:
            self.nullable = False

    def is_fixed_size(self) -> bool:
        """Check if this column has a fixed size."""
        return self.data_type in TYPE_SIZES

    def get_size(self) -> Optional[int]:
        """Get fixed size or None for variable-length types."""
        return TYPE_SIZES.get(self.data_type)

    def __repr__(self) -> str:
        return f"Column({self.name}, {self.data_type.name})"


class Table:
    """
    Table schema definition.

    Attributes:
        name: Table name
        columns: Ordered list of columns
        primary_key: List of primary key column names
        indexes: Map of index name to indexed columns
    """

    __slots__ = ("name", "columns", "primary_key", "indexes", "_column_map", "heap_file_id")

    def __init__(
        self,
        name: str,
        columns: List[Column],
        primary_key: Optional[List[str]] = None,
        indexes: Optional[Dict[str, List[str]]] = None,
        heap_file_id: Optional[int] = None,
    ) -> None:
        self.name = name
        self.columns = columns
        self.primary_key = primary_key or []
        self.indexes = indexes or {}
        self.heap_file_id = heap_file_id

        # Build column lookup map
        self._column_map = {col.name: (i, col) for i, col in enumerate(self.columns)}

        # Validate primary key columns exist
        for pk_col in self.primary_key:
            if pk_col not in self._column_map:
                raise ValueError(f"Primary key column '{pk_col}' not found")

    def get_column(self, name: str) -> Optional[Column]:
        """Get column by name."""
        entry = self._column_map.get(name)
        return entry[1] if entry else None

    def get_column_index(self, name: str) -> Optional[int]:
        """Get column position by name."""
        entry = self._column_map.get(name)
        return entry[0] if entry else None

    @property
    def column_count(self) -> int:
        """Get number of columns."""
        return len(self.columns)

    @property
    def column_names(self) -> List[str]:
        """Get list of column names."""
        return [col.name for col in self.columns]

    def __repr__(self) -> str:
        return f"Table({self.name}, {len(self.columns)} columns)"


class RecordSerializer:
    """
    Serializes and deserializes records for a table schema.

    Record Format:
    - Null bitmap: ceil(num_columns / 8) bytes
    - Fixed fields: concatenated in column order
    - Variable fields: [4-byte offset, data]
    """

    __slots__ = ("_table", "_null_bitmap_size", "_fixed_size")

    def __init__(self, table: Table) -> None:
        """Initialize serializer for a table schema."""
        self._table = table
        self._null_bitmap_size = (len(table.columns) + 7) // 8
        self._fixed_size = self._calculate_fixed_size()

    def _calculate_fixed_size(self) -> int:
        """Calculate size of fixed-length portion."""
        size = 0
        for col in self._table.columns:
            if col.is_fixed_size():
                size += col.get_size()
        return size

    def serialize(self, values: Dict[str, Any]) -> bytes:
        """
        Serialize a record from column values.

        Args:
            values: Map of column name to value

        Returns:
            Serialized bytes
        """
        null_bitmap = bytearray(self._null_bitmap_size)
        fixed_parts = []
        variable_parts = []

        for i, col in enumerate(self._table.columns):
            value = values.get(col.name)

            # Handle NULL
            if value is None:
                if not col.nullable:
                    raise ValueError(f"Column '{col.name}' cannot be NULL")
                null_bitmap[i // 8] |= 1 << (i % 8)
                if col.is_fixed_size():
                    fixed_parts.append(b"\x00" * col.get_size())
                continue

            # Serialize value by type
            if col.data_type == DataType.INTEGER:
                fixed_parts.append(struct.pack("<i", value))
            elif col.data_type == DataType.BIGINT:
                fixed_parts.append(struct.pack("<q", value))
            elif col.data_type == DataType.FLOAT:
                fixed_parts.append(struct.pack("<d", value))
            elif col.data_type == DataType.BOOLEAN:
                fixed_parts.append(struct.pack("<?", value))
            elif col.data_type in (DataType.VARCHAR, DataType.TEXT):
                encoded = value.encode("utf-8") if isinstance(value, str) else value
                if col.max_length and len(encoded) > col.max_length:
                    raise ValueError(
                        f"Value too long for column '{col.name}': "
                        f"{len(encoded)} > {col.max_length}"
                    )
                variable_parts.append(encoded)
            elif col.data_type == DataType.BLOB:
                variable_parts.append(value if isinstance(value, bytes) else bytes(value))

        # Build record: null_bitmap + fixed + variable_lengths + variable_data
        result = bytearray(null_bitmap)
        result.extend(b"".join(fixed_parts))

        # Variable-length fields: store length prefixes then data
        for var_data in variable_parts:
            result.extend(struct.pack("<I", len(var_data)))
        for var_data in variable_parts:
            result.extend(var_data)

        return bytes(result)

    def deserialize(self, data: bytes) -> Dict[str, Any]:
        """
        Deserialize a record to column values.

        Args:
            data: Serialized record bytes

        Returns:
            Map of column name to value
        """
        values = {}
        offset = 0

        # Read null bitmap
        null_bitmap = data[:self._null_bitmap_size]
        offset = self._null_bitmap_size

        # Track variable-length field positions
        var_columns = []

        # Read fixed-length fields
        for i, col in enumerate(self._table.columns):
            is_null = (null_bitmap[i // 8] >> (i % 8)) & 1

            if is_null:
                values[col.name] = None
                if col.is_fixed_size():
                    offset += col.get_size()
                else:
                    var_columns.append((col, True))
                continue

            if col.data_type == DataType.INTEGER:
                values[col.name] = struct.unpack_from("<i", data, offset)[0]
                offset += 4
            elif col.data_type == DataType.BIGINT:
                values[col.name] = struct.unpack_from("<q", data, offset)[0]
                offset += 8
            elif col.data_type == DataType.FLOAT:
                values[col.name] = struct.unpack_from("<d", data, offset)[0]
                offset += 8
            elif col.data_type == DataType.BOOLEAN:
                values[col.name] = struct.unpack_from("<?", data, offset)[0]
                offset += 1
            elif col.data_type in (DataType.VARCHAR, DataType.TEXT, DataType.BLOB):
                var_columns.append((col, False))

        # Read variable-length field lengths
        var_lengths = []
        for col, is_null in var_columns:
            if is_null:
                var_lengths.append(0)
            else:
                length = struct.unpack_from("<I", data, offset)[0]
                var_lengths.append(length)
                offset += 4

        # Read variable-length field data
        for (col, is_null), length in zip(var_columns, var_lengths):
            if is_null:
                continue
            var_data = data[offset : offset + length]
            offset += length

            if col.data_type == DataType.BLOB:
                values[col.name] = var_data
            else:
                values[col.name] = var_data.decode("utf-8")

        return values


class Catalog:
    """
    In-memory database catalog.

    Manages table schemas and provides metadata lookup.

    Thread Safety: Not thread-safe. Caller must provide synchronization.
    """

    __slots__ = ("_tables", "_serializers")

    def __init__(self) -> None:
        """Initialize empty catalog."""
        self._tables: Dict[str, Table] = {}
        self._serializers: Dict[str, RecordSerializer] = {}

    def create_table(self, table: Table) -> None:
        """
        Register a new table.

        Args:
            table: Table schema to register

        Raises:
            ValueError: If table already exists
        """
        if table.name in self._tables:
            raise ValueError(f"Table '{table.name}' already exists")
        self._tables[table.name] = table
        self._serializers[table.name] = RecordSerializer(table)

    def drop_table(self, name: str) -> None:
        """
        Remove a table from the catalog.

        Args:
            name: Table name to drop

        Raises:
            ValueError: If table doesn't exist
        """
        if name not in self._tables:
            raise ValueError(f"Table '{name}' does not exist")
        del self._tables[name]
        del self._serializers[name]

    def get_table(self, name: str) -> Optional[Table]:
        """Get table schema by name."""
        return self._tables.get(name)

    def get_serializer(self, table_name: str) -> Optional[RecordSerializer]:
        """Get record serializer for a table."""
        return self._serializers.get(table_name)

    def table_exists(self, name: str) -> bool:
        """Check if a table exists."""
        return name in self._tables

    @property
    def table_names(self) -> List[str]:
        """Get list of all table names."""
        return list(self._tables.keys())

    def __repr__(self) -> str:
        """String representation."""
        return f"Catalog(tables={list(self._tables.keys())})"
