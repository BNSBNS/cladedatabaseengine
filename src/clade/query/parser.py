"""
SQL Parser using Lark grammar.

Parses SQL statements into an Abstract Syntax Tree (AST).

Supported Statements:
- SELECT (with WHERE, ORDER BY, LIMIT, JOIN)
- INSERT
- UPDATE
- DELETE
- CREATE TABLE
- DROP TABLE

Grammar is designed to be ANSI SQL compatible with common extensions.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Any, Union
from lark import Lark, Transformer, v_args

from clade.catalog.schema import DataType


# SQL Grammar definition
SQL_GRAMMAR = r"""
    ?start: statement

    ?statement: select_stmt
              | insert_stmt
              | update_stmt
              | delete_stmt
              | create_table_stmt
              | drop_table_stmt

    // SELECT
    select_stmt: "SELECT"i select_list "FROM"i table_ref where_clause? order_by_clause? limit_clause?

    select_list: "*" -> select_all
               | select_item ("," select_item)* -> select_columns

    select_item: expr alias? -> select_expr

    alias: "AS"i? NAME

    table_ref: NAME alias? join_clause*

    join_clause: join_type "JOIN"i NAME alias? "ON"i expr
    join_type: "INNER"i? -> inner_join
             | "LEFT"i "OUTER"i? -> left_join
             | "RIGHT"i "OUTER"i? -> right_join
             | "CROSS"i -> cross_join

    where_clause: "WHERE"i expr

    order_by_clause: "ORDER"i "BY"i order_item ("," order_item)*
    order_item: expr order_direction?
    order_direction: "ASC"i -> asc
                   | "DESC"i -> desc

    limit_clause: "LIMIT"i NUMBER ("OFFSET"i NUMBER)?

    // INSERT
    insert_stmt: "INSERT"i "INTO"i NAME column_list? "VALUES"i values_list

    column_list: "(" NAME ("," NAME)* ")"

    values_list: "(" expr ("," expr)* ")" ("," "(" expr ("," expr)* ")")*

    // UPDATE
    update_stmt: "UPDATE"i NAME "SET"i set_clause ("," set_clause)* where_clause?

    set_clause: NAME "=" expr

    // DELETE
    delete_stmt: "DELETE"i "FROM"i NAME where_clause?

    // CREATE TABLE
    create_table_stmt: "CREATE"i "TABLE"i NAME "(" column_def ("," column_def)* ("," table_constraint)* ")"

    column_def: NAME data_type column_constraint*

    data_type: "INTEGER"i -> type_int
             | "INT"i -> type_int
             | "BIGINT"i -> type_bigint
             | "FLOAT"i -> type_float
             | "DOUBLE"i -> type_float
             | "BOOLEAN"i -> type_bool
             | "BOOL"i -> type_bool
             | "VARCHAR"i "(" NUMBER ")" -> type_varchar
             | "TEXT"i -> type_text
             | "BLOB"i -> type_blob

    column_constraint: "NOT"i "NULL"i -> not_null
                     | "NULL"i -> nullable
                     | "PRIMARY"i "KEY"i -> primary_key
                     | "DEFAULT"i expr -> default_value

    table_constraint: "PRIMARY"i "KEY"i "(" NAME ("," NAME)* ")" -> pk_constraint

    // DROP TABLE
    drop_table_stmt: "DROP"i "TABLE"i "IF"i "EXISTS"i NAME -> drop_if_exists
                   | "DROP"i "TABLE"i NAME -> drop_table

    // Expressions
    ?expr: or_expr

    ?or_expr: and_expr
            | or_expr "OR"i and_expr -> or_op

    ?and_expr: not_expr
             | and_expr "AND"i not_expr -> and_op

    ?not_expr: "NOT"i not_expr -> not_op
             | comparison

    ?comparison: add_expr
               | comparison "=" add_expr -> eq
               | comparison "!=" add_expr -> ne
               | comparison "<>" add_expr -> ne
               | comparison "<" add_expr -> lt
               | comparison ">" add_expr -> gt
               | comparison "<=" add_expr -> le
               | comparison ">=" add_expr -> ge
               | comparison "IS"i "NULL"i -> is_null
               | comparison "IS"i "NOT"i "NULL"i -> is_not_null
               | comparison "LIKE"i add_expr -> like
               | comparison "IN"i "(" expr ("," expr)* ")" -> in_list
               | comparison "BETWEEN"i add_expr "AND"i add_expr -> between

    ?add_expr: mul_expr
             | add_expr "+" mul_expr -> add
             | add_expr "-" mul_expr -> sub

    ?mul_expr: unary_expr
             | mul_expr "*" unary_expr -> mul
             | mul_expr "/" unary_expr -> div
             | mul_expr "%" unary_expr -> mod

    ?unary_expr: "-" unary_expr -> neg
              | atom

    ?atom: literal
         | column_ref
         | func_call
         | "(" expr ")"

    column_ref: NAME "." NAME -> qualified_column
              | NAME -> simple_column

    func_call: NAME "(" func_args? ")"

    func_args: "*" -> count_all
             | "DISTINCT"i expr -> distinct_arg
             | expr ("," expr)* -> arg_list

    ?literal: NUMBER -> number
            | SIGNED_NUMBER -> number
            | ESCAPED_STRING -> string
            | "TRUE"i -> true
            | "FALSE"i -> false
            | "NULL"i -> null

    NAME: /[a-zA-Z_][a-zA-Z0-9_]*/
    NUMBER: /[0-9]+(\.[0-9]+)?/
    SIGNED_NUMBER: /[-+]?[0-9]+(\.[0-9]+)?/
    ESCAPED_STRING: "'" /([^'\\]|\\.)*/ "'"

    %import common.WS
    %ignore WS
"""


# AST Node Types
class ASTNodeType(Enum):
    """Types of AST nodes."""

    # Statements
    SELECT = auto()
    INSERT = auto()
    UPDATE = auto()
    DELETE = auto()
    CREATE_TABLE = auto()
    DROP_TABLE = auto()

    # Expressions
    LITERAL = auto()
    COLUMN_REF = auto()
    BINARY_OP = auto()
    UNARY_OP = auto()
    FUNCTION_CALL = auto()
    SUBQUERY = auto()

    # Clauses
    SELECT_LIST = auto()
    FROM_CLAUSE = auto()
    WHERE_CLAUSE = auto()
    ORDER_BY = auto()
    LIMIT = auto()
    JOIN = auto()


@dataclass
class ASTNode:
    """Base AST node."""

    __slots__ = ("node_type", "value", "children")

    node_type: ASTNodeType
    value: Any
    children: List["ASTNode"]

    def __repr__(self) -> str:
        if self.children:
            return f"{self.node_type.name}({self.value}, {self.children})"
        return f"{self.node_type.name}({self.value})"


@dataclass
class SelectStmt:
    """SELECT statement representation."""

    __slots__ = (
        "columns",
        "table",
        "table_alias",
        "joins",
        "where",
        "order_by",
        "limit",
        "offset",
        "is_select_all",
    )

    columns: List[tuple]  # [(expr, alias), ...]
    table: str
    table_alias: Optional[str]
    joins: List[dict]
    where: Optional[Any]
    order_by: Optional[List[tuple]]  # [(expr, 'ASC'|'DESC'), ...]
    limit: Optional[int]
    offset: Optional[int]
    is_select_all: bool


@dataclass
class InsertStmt:
    """INSERT statement representation."""

    __slots__ = ("table", "columns", "values")

    table: str
    columns: Optional[List[str]]
    values: List[List[Any]]


@dataclass
class UpdateStmt:
    """UPDATE statement representation."""

    __slots__ = ("table", "assignments", "where")

    table: str
    assignments: List[tuple]  # [(column, expr), ...]
    where: Optional[Any]


@dataclass
class DeleteStmt:
    """DELETE statement representation."""

    __slots__ = ("table", "where")

    table: str
    where: Optional[Any]


@dataclass
class CreateTableStmt:
    """CREATE TABLE statement representation."""

    __slots__ = ("table", "columns", "primary_key")

    table: str
    columns: List[dict]  # [{name, type, nullable, default, pk}, ...]
    primary_key: List[str]


@dataclass
class DropTableStmt:
    """DROP TABLE statement representation."""

    __slots__ = ("table", "if_exists")

    table: str
    if_exists: bool


@dataclass
class Expr:
    """Expression node."""

    __slots__ = ("op", "operands", "value")

    op: str  # 'literal', 'column', 'binary', 'unary', 'function'
    operands: List[Any]
    value: Any


class SQLTransformer(Transformer):
    """Transform Lark parse tree to AST."""

    # Literals
    def number(self, items):
        val = items[0].value
        return Expr("literal", [], float(val) if "." in val else int(val))

    def string(self, items):
        # Remove quotes and handle escapes
        val = items[0].value[1:-1].replace("\\'", "'").replace("\\\\", "\\")
        return Expr("literal", [], val)

    def true(self, _):
        return Expr("literal", [], True)

    def false(self, _):
        return Expr("literal", [], False)

    def null(self, _):
        return Expr("literal", [], None)

    # Column references
    def simple_column(self, items):
        return Expr("column", [], (None, items[0].value))

    def qualified_column(self, items):
        return Expr("column", [], (items[0].value, items[1].value))

    # Binary operations
    def eq(self, items):
        return Expr("binary", [items[0], items[1]], "=")

    def ne(self, items):
        return Expr("binary", [items[0], items[1]], "!=")

    def lt(self, items):
        return Expr("binary", [items[0], items[1]], "<")

    def gt(self, items):
        return Expr("binary", [items[0], items[1]], ">")

    def le(self, items):
        return Expr("binary", [items[0], items[1]], "<=")

    def ge(self, items):
        return Expr("binary", [items[0], items[1]], ">=")

    def add(self, items):
        return Expr("binary", [items[0], items[1]], "+")

    def sub(self, items):
        return Expr("binary", [items[0], items[1]], "-")

    def mul(self, items):
        return Expr("binary", [items[0], items[1]], "*")

    def div(self, items):
        return Expr("binary", [items[0], items[1]], "/")

    def mod(self, items):
        return Expr("binary", [items[0], items[1]], "%")

    def and_op(self, items):
        return Expr("binary", [items[0], items[1]], "AND")

    def or_op(self, items):
        return Expr("binary", [items[0], items[1]], "OR")

    def like(self, items):
        return Expr("binary", [items[0], items[1]], "LIKE")

    def between(self, items):
        return Expr("between", [items[0], items[1], items[2]], "BETWEEN")

    def in_list(self, items):
        return Expr("in", list(items), "IN")

    # Unary operations
    def not_op(self, items):
        return Expr("unary", [items[0]], "NOT")

    def neg(self, items):
        return Expr("unary", [items[0]], "-")

    def is_null(self, items):
        return Expr("unary", [items[0]], "IS NULL")

    def is_not_null(self, items):
        return Expr("unary", [items[0]], "IS NOT NULL")

    # Function calls
    def func_call(self, items):
        name = items[0].value.upper()
        args = items[1] if len(items) > 1 else []
        return Expr("function", args if isinstance(args, list) else [args], name)

    def count_all(self, _):
        return [Expr("literal", [], "*")]

    def distinct_arg(self, items):
        return [("DISTINCT", items[0])]

    def arg_list(self, items):
        return list(items)

    # SELECT
    def select_all(self, _):
        return ("*", None)

    def select_columns(self, items):
        return list(items)

    def alias(self, items):
        return items[0].value

    def select_expr(self, items):
        expr = items[0]
        alias = items[1] if len(items) > 1 else None
        return (expr, alias)

    def table_ref(self, items):
        table = items[0].value
        alias = None
        joins = []
        for item in items[1:]:
            if isinstance(item, str):
                alias = item  # Transformed alias
            elif hasattr(item, "value"):
                alias = item.value  # Token
            elif isinstance(item, dict):
                joins.append(item)
        return {"table": table, "alias": alias, "joins": joins}

    # Joins
    def inner_join(self, _):
        return "INNER"

    def left_join(self, _):
        return "LEFT"

    def right_join(self, _):
        return "RIGHT"

    def cross_join(self, _):
        return "CROSS"

    def join_clause(self, items):
        join_type = items[0]
        table = items[1].value
        alias = None
        condition = None
        for item in items[2:]:
            if isinstance(item, str) and not item.startswith(("=", "!", "<", ">")):
                alias = item  # Transformed alias (not an operator)
            elif isinstance(item, Expr):
                condition = item
            elif hasattr(item, "type") and item.type == "NAME":
                alias = item.value  # NAME token
        return {"type": join_type, "table": table, "alias": alias, "condition": condition}

    # WHERE
    def where_clause(self, items):
        return items[0]

    # ORDER BY
    def order_by_clause(self, items):
        return list(items)

    def order_item(self, items):
        expr = items[0]
        direction = items[1] if len(items) > 1 else "ASC"
        return (expr, direction)

    def asc(self, _):
        return "ASC"

    def desc(self, _):
        return "DESC"

    # LIMIT
    def limit_clause(self, items):
        limit = int(items[0].value)
        offset = int(items[1].value) if len(items) > 1 else 0
        return (limit, offset)

    # SELECT statement
    def select_stmt(self, items):
        select_list = items[0]
        table_ref = items[1]
        where = None
        order_by = None
        limit = None
        offset = 0

        for item in items[2:]:
            if isinstance(item, Expr):
                where = item
            elif isinstance(item, list):
                order_by = item
            elif isinstance(item, tuple) and len(item) == 2:
                limit, offset = item

        is_select_all = select_list == ("*", None)
        columns = [] if is_select_all else select_list

        return SelectStmt(
            columns=columns,
            table=table_ref["table"],
            table_alias=table_ref.get("alias"),
            joins=table_ref.get("joins", []),
            where=where,
            order_by=order_by,
            limit=limit,
            offset=offset,
            is_select_all=is_select_all,
        )

    # INSERT
    def column_list(self, items):
        return [item.value for item in items]

    def values_list(self, items):
        # Group expressions into rows
        rows = []
        current_row = []
        for item in items:
            if isinstance(item, Expr):
                current_row.append(item)
        if current_row:
            rows.append(current_row)
        return rows if rows else [[item for item in items if isinstance(item, Expr)]]

    def insert_stmt(self, items):
        table = items[0].value
        columns = None
        values = []

        for item in items[1:]:
            if isinstance(item, list):
                if item and isinstance(item[0], str):
                    columns = item
                else:
                    values = item if item and isinstance(item[0], list) else [item]

        return InsertStmt(table=table, columns=columns, values=values)

    # UPDATE
    def set_clause(self, items):
        return (items[0].value, items[1])

    def update_stmt(self, items):
        table = items[0].value
        assignments = []
        where = None

        for item in items[1:]:
            if isinstance(item, tuple):
                assignments.append(item)
            elif isinstance(item, Expr):
                where = item

        return UpdateStmt(table=table, assignments=assignments, where=where)

    # DELETE
    def delete_stmt(self, items):
        table = items[0].value
        where = items[1] if len(items) > 1 else None
        return DeleteStmt(table=table, where=where)

    # CREATE TABLE
    def type_int(self, _):
        return DataType.INTEGER

    def type_bigint(self, _):
        return DataType.BIGINT

    def type_float(self, _):
        return DataType.FLOAT

    def type_bool(self, _):
        return DataType.BOOLEAN

    def type_varchar(self, items):
        return (DataType.VARCHAR, int(items[0].value))

    def type_text(self, _):
        return DataType.TEXT

    def type_blob(self, _):
        return DataType.BLOB

    def not_null(self, _):
        return ("nullable", False)

    def nullable(self, _):
        return ("nullable", True)

    def primary_key(self, _):
        return ("pk", True)

    def default_value(self, items):
        return ("default", items[0])

    def column_def(self, items):
        name = items[0].value
        dtype = items[1]
        nullable = True
        default = None
        pk = False
        max_length = None

        if isinstance(dtype, tuple):
            dtype, max_length = dtype

        for item in items[2:]:
            if isinstance(item, tuple):
                key, val = item
                if key == "nullable":
                    nullable = val
                elif key == "pk":
                    pk = val
                    nullable = False
                elif key == "default":
                    default = val

        return {
            "name": name,
            "type": dtype,
            "nullable": nullable,
            "default": default,
            "pk": pk,
            "max_length": max_length,
        }

    def pk_constraint(self, items):
        return ("pk_constraint", [item.value for item in items])

    def create_table_stmt(self, items):
        table = items[0].value
        columns = []
        primary_key = []

        for item in items[1:]:
            if isinstance(item, dict):
                columns.append(item)
                if item.get("pk"):
                    primary_key.append(item["name"])
            elif isinstance(item, tuple) and item[0] == "pk_constraint":
                primary_key = item[1]

        return CreateTableStmt(table=table, columns=columns, primary_key=primary_key)

    # DROP TABLE
    def drop_table(self, items):
        return DropTableStmt(table=items[0].value, if_exists=False)

    def drop_if_exists(self, items):
        return DropTableStmt(table=items[0].value, if_exists=True)


class SQLParser:
    """
    SQL Parser.

    Parses SQL statements into AST representations.

    Thread Safety: Thread-safe (parser is stateless).
    """

    __slots__ = ("_parser", "_transformer")

    def __init__(self) -> None:
        """Initialize parser."""
        self._parser = Lark(SQL_GRAMMAR, start="start", parser="lalr")
        self._transformer = SQLTransformer()

    def parse(self, sql: str) -> Union[SelectStmt, InsertStmt, UpdateStmt, DeleteStmt, CreateTableStmt, DropTableStmt]:
        """
        Parse a SQL statement.

        Args:
            sql: SQL statement string

        Returns:
            Parsed statement object

        Raises:
            ParseError: If SQL is invalid
        """
        tree = self._parser.parse(sql)
        return self._transformer.transform(tree)

    def __repr__(self) -> str:
        return "SQLParser()"
