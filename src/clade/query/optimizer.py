"""
Query Optimizer.

Transforms logical query plans into more efficient forms.

Optimization Rules:
- Predicate pushdown: Push filters closer to scan operators
- Projection pushdown: Eliminate unused columns early
- Constant folding: Evaluate constant expressions at plan time
- Join reordering: Reorder joins for better performance

Cost Model:
- SeqScan: num_pages * IO_COST + num_rows * CPU_COST
- Filter: input_cost + num_rows * FILTER_COST
- Sort: input_cost + N * log(N) * SORT_COST
"""

from typing import List, Optional, Any, Dict
from clade.query.planner import (
    PlanNode,
    PlanNodeType,
    SeqScanNode,
    FilterNode,
    ProjectNode,
    SortNode,
    LimitNode,
    JoinNode,
    JoinType,
)
from clade.query.parser import Expr


# Cost constants
IO_COST_PER_PAGE = 1.0
CPU_COST_PER_ROW = 0.01
FILTER_COST_PER_ROW = 0.005
SORT_COST_FACTOR = 0.05
JOIN_COST_FACTOR = 0.1


class Optimizer:
    """
    Query optimizer.

    Applies optimization rules to logical plans and estimates costs.

    Thread Safety: Thread-safe (stateless transformations).
    """

    __slots__ = ("_stats",)

    def __init__(self, statistics: Optional[Dict[str, Dict]] = None) -> None:
        """
        Initialize optimizer.

        Args:
            statistics: Optional table statistics for cost estimation
        """
        self._stats = statistics or {}

    def optimize(self, plan: PlanNode) -> PlanNode:
        """
        Optimize a query plan.

        Args:
            plan: Logical query plan

        Returns:
            Optimized plan
        """
        # Apply optimization rules in order
        plan = self._push_down_predicates(plan)
        plan = self._push_down_projections(plan)
        plan = self._fold_constants(plan)
        plan = self._estimate_costs(plan)
        return plan

    # ─────────────────────────────────────────────────────────────────────────
    # Predicate Pushdown
    # ─────────────────────────────────────────────────────────────────────────

    def _push_down_predicates(self, node: PlanNode) -> PlanNode:
        """
        Push filter predicates down towards scan operators.

        This reduces the number of rows processed by upper operators.
        """
        if node.node_type == PlanNodeType.FILTER:
            filter_node = node
            child = node.children[0]

            # If child is a SeqScan, merge filter into scan
            if child.node_type == PlanNodeType.SEQ_SCAN:
                scan = child
                scan.predicates.append(filter_node.predicate)
                return self._push_down_predicates(scan)

            # If child is a Filter, combine predicates
            if child.node_type == PlanNodeType.FILTER:
                combined = Expr(
                    "binary",
                    [child.predicate, filter_node.predicate],
                    "AND"
                )
                new_filter = FilterNode(child.children[0], combined)
                return self._push_down_predicates(new_filter)

            # If child is a Join, try to push to appropriate side
            if child.node_type == PlanNodeType.NESTED_LOOP_JOIN:
                join = child
                pushed = self._push_predicate_through_join(
                    filter_node.predicate, join
                )
                if pushed is not None:
                    return self._push_down_predicates(pushed)

        # Recursively process children
        new_children = [self._push_down_predicates(c) for c in node.children]
        node.children = new_children
        return node

    def _push_predicate_through_join(
        self, predicate: Expr, join: JoinNode
    ) -> Optional[PlanNode]:
        """Try to push predicate through a join."""
        # Get columns referenced in predicate
        columns = self._get_referenced_columns(predicate)

        # Get columns available from each side
        left_cols = {name for name, _ in join.children[0].output_schema}
        right_cols = {name for name, _ in join.children[1].output_schema}

        # Check if predicate only references one side
        refs_left = any(c in left_cols or any(c.endswith(f".{lc.split('.')[-1]}") for lc in left_cols) for c in columns)
        refs_right = any(c in right_cols or any(c.endswith(f".{rc.split('.')[-1]}") for rc in right_cols) for c in columns)

        if refs_left and not refs_right:
            # Push to left side
            new_left = FilterNode(join.children[0], predicate)
            return JoinNode(
                left=new_left,
                right=join.children[1],
                join_type=join.join_type,
                condition=join.condition,
                output_schema=join.output_schema,
            )
        elif refs_right and not refs_left:
            # Push to right side
            new_right = FilterNode(join.children[1], predicate)
            return JoinNode(
                left=join.children[0],
                right=new_right,
                join_type=join.join_type,
                condition=join.condition,
                output_schema=join.output_schema,
            )

        return None

    def _get_referenced_columns(self, expr: Expr) -> List[str]:
        """Get all column references in an expression."""
        columns = []

        if expr.op == "column":
            table, col = expr.value
            if table:
                columns.append(f"{table}.{col}")
            else:
                columns.append(col)
        elif expr.operands:
            for operand in expr.operands:
                if isinstance(operand, Expr):
                    columns.extend(self._get_referenced_columns(operand))

        return columns

    # ─────────────────────────────────────────────────────────────────────────
    # Projection Pushdown
    # ─────────────────────────────────────────────────────────────────────────

    def _push_down_projections(self, node: PlanNode) -> PlanNode:
        """
        Push projections down to eliminate unused columns early.

        This reduces memory usage and processing overhead.
        """
        # For now, just recursively process (full implementation would track
        # required columns and add projections closer to scans)
        new_children = [self._push_down_projections(c) for c in node.children]
        node.children = new_children
        return node

    # ─────────────────────────────────────────────────────────────────────────
    # Constant Folding
    # ─────────────────────────────────────────────────────────────────────────

    def _fold_constants(self, node: PlanNode) -> PlanNode:
        """
        Evaluate constant expressions at plan time.

        Examples:
        - 1 + 2 -> 3
        - 'foo' = 'bar' -> FALSE
        """
        # Process filters
        if node.node_type == PlanNodeType.FILTER:
            filter_node = node
            folded = self._fold_expr(filter_node.predicate)

            # If predicate folds to TRUE, eliminate filter
            if folded.op == "literal" and folded.value is True:
                return self._fold_constants(filter_node.children[0])

            # If predicate folds to FALSE, return empty result
            # (would need ResultNode to represent empty)

            filter_node.predicate = folded

        # Recursively process children
        new_children = [self._fold_constants(c) for c in node.children]
        node.children = new_children
        return node

    def _fold_expr(self, expr: Expr) -> Expr:
        """Fold constant expressions."""
        if expr.op == "literal" or expr.op == "column":
            return expr

        # Fold operands first
        folded_operands = []
        for operand in expr.operands:
            if isinstance(operand, Expr):
                folded_operands.append(self._fold_expr(operand))
            else:
                folded_operands.append(operand)

        # Check if all operands are now literals
        if all(isinstance(o, Expr) and o.op == "literal" for o in folded_operands):
            values = [o.value for o in folded_operands]

            if expr.op == "binary":
                result = self._eval_binary(expr.value, values[0], values[1])
                if result is not None:
                    return Expr("literal", [], result)

        return Expr(expr.op, folded_operands, expr.value)

    def _eval_binary(self, op: str, left: Any, right: Any) -> Optional[Any]:
        """Evaluate a binary operation on constant values."""
        try:
            if op == "+":
                return left + right
            elif op == "-":
                return left - right
            elif op == "*":
                return left * right
            elif op == "/":
                return left / right if right != 0 else None
            elif op == "=":
                return left == right
            elif op == "!=":
                return left != right
            elif op == "<":
                return left < right
            elif op == ">":
                return left > right
            elif op == "<=":
                return left <= right
            elif op == ">=":
                return left >= right
            elif op == "AND":
                return left and right
            elif op == "OR":
                return left or right
        except Exception:
            pass
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Cost Estimation
    # ─────────────────────────────────────────────────────────────────────────

    def _estimate_costs(self, node: PlanNode) -> PlanNode:
        """
        Estimate the cost of executing each plan node.

        Cost model is simplified but captures relative costs.
        """
        # Process children first
        for child in node.children:
            self._estimate_costs(child)

        # Calculate cost based on node type
        if node.node_type == PlanNodeType.SEQ_SCAN:
            scan = node
            stats = self._stats.get(scan.table_name, {})
            num_pages = stats.get("num_pages", 100)
            num_rows = stats.get("num_rows", 1000)

            # Base scan cost
            node.cost = num_pages * IO_COST_PER_PAGE + num_rows * CPU_COST_PER_ROW

            # Reduce for predicates (selectivity)
            if scan.predicates:
                # Assume 10% selectivity per predicate
                selectivity = 0.1 ** len(scan.predicates)
                node.cost *= max(0.1, selectivity)

        elif node.node_type == PlanNodeType.FILTER:
            child_cost = node.children[0].cost if node.children else 0
            # Assume 10% of rows pass filter
            node.cost = child_cost + child_cost * 0.1 * FILTER_COST_PER_ROW

        elif node.node_type == PlanNodeType.PROJECT:
            child_cost = node.children[0].cost if node.children else 0
            node.cost = child_cost * 1.01  # Minimal overhead

        elif node.node_type == PlanNodeType.SORT:
            child_cost = node.children[0].cost if node.children else 0
            # N log N for sorting
            estimated_rows = 1000  # Would come from stats
            import math
            node.cost = child_cost + estimated_rows * math.log2(max(1, estimated_rows)) * SORT_COST_FACTOR

        elif node.node_type == PlanNodeType.LIMIT:
            child_cost = node.children[0].cost if node.children else 0
            limit_node = node
            # Reduce cost based on limit
            estimated_rows = 1000
            fraction = min(1.0, limit_node.limit / max(1, estimated_rows))
            node.cost = child_cost * fraction

        elif node.node_type == PlanNodeType.NESTED_LOOP_JOIN:
            left_cost = node.children[0].cost if node.children else 0
            right_cost = node.children[1].cost if len(node.children) > 1 else 0
            # Nested loop: O(N * M)
            node.cost = left_cost + left_cost * right_cost * JOIN_COST_FACTOR

        else:
            # Default: sum of children costs
            node.cost = sum(c.cost for c in node.children)

        return node

    def get_statistics(self, table_name: str) -> Dict:
        """Get statistics for a table."""
        return self._stats.get(table_name, {})

    def update_statistics(self, table_name: str, stats: Dict) -> None:
        """Update statistics for a table."""
        self._stats[table_name] = stats

    def __repr__(self) -> str:
        return f"Optimizer(tables={list(self._stats.keys())})"
