"""
Metrics collection for database observability.

Provides Prometheus-style metrics for monitoring database performance.

Metrics Types:
- Counter: Monotonically increasing value (e.g., queries executed)
- Gauge: Point-in-time value (e.g., active connections)
- Histogram: Distribution of values (e.g., query latency)

Usage:
    metrics = MetricsCollector()
    metrics.increment("queries_total", labels={"type": "select"})
    with metrics.timer("query_duration"):
        execute_query()
"""

from abc import ABC
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Callable
import threading
import time
import math


class MetricType(Enum):
    """Types of metrics."""

    COUNTER = auto()
    GAUGE = auto()
    HISTOGRAM = auto()


@dataclass
class MetricValue:
    """A single metric value."""

    __slots__ = ("value", "labels", "timestamp")

    value: float
    labels: Dict[str, str]
    timestamp: float


class Counter:
    """
    A monotonically increasing counter.

    Use for: total queries, bytes written, errors, etc.
    """

    __slots__ = ("_name", "_description", "_values", "_lock")

    def __init__(self, name: str, description: str = "") -> None:
        self._name = name
        self._description = description
        self._values: Dict[str, float] = {}  # label_key -> value
        self._lock = threading.Lock()

    def inc(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment the counter."""
        key = self._labels_key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0) + value

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current counter value."""
        key = self._labels_key(labels)
        with self._lock:
            return self._values.get(key, 0)

    def reset(self) -> None:
        """Reset all values."""
        with self._lock:
            self._values.clear()

    def _labels_key(self, labels: Optional[Dict[str, str]]) -> str:
        """Create a string key from labels."""
        if not labels:
            return ""
        return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))

    def collect(self) -> List[MetricValue]:
        """Collect all values."""
        with self._lock:
            now = time.time()
            return [
                MetricValue(value=v, labels=self._parse_labels(k), timestamp=now)
                for k, v in self._values.items()
            ]

    def _parse_labels(self, key: str) -> Dict[str, str]:
        """Parse labels from key string."""
        if not key:
            return {}
        labels = {}
        for pair in key.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                labels[k] = v
        return labels

    @property
    def name(self) -> str:
        return self._name

    @property
    def type(self) -> MetricType:
        return MetricType.COUNTER


class Gauge:
    """
    A point-in-time value that can go up or down.

    Use for: active connections, queue size, memory usage, etc.
    """

    __slots__ = ("_name", "_description", "_values", "_lock")

    def __init__(self, name: str, description: str = "") -> None:
        self._name = name
        self._description = description
        self._values: Dict[str, float] = {}
        self._lock = threading.Lock()

    def set(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set the gauge value."""
        key = self._labels_key(labels)
        with self._lock:
            self._values[key] = value

    def inc(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment the gauge."""
        key = self._labels_key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0) + value

    def dec(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Decrement the gauge."""
        self.inc(-value, labels)

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current gauge value."""
        key = self._labels_key(labels)
        with self._lock:
            return self._values.get(key, 0)

    def _labels_key(self, labels: Optional[Dict[str, str]]) -> str:
        if not labels:
            return ""
        return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))

    def collect(self) -> List[MetricValue]:
        """Collect all values."""
        with self._lock:
            now = time.time()
            return [
                MetricValue(
                    value=v,
                    labels=self._parse_labels(k) if k else {},
                    timestamp=now
                )
                for k, v in self._values.items()
            ]

    def _parse_labels(self, key: str) -> Dict[str, str]:
        if not key:
            return {}
        labels = {}
        for pair in key.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                labels[k] = v
        return labels

    @property
    def name(self) -> str:
        return self._name

    @property
    def type(self) -> MetricType:
        return MetricType.GAUGE


class Histogram:
    """
    A distribution of values with configurable buckets.

    Use for: query latency, request size, etc.
    """

    __slots__ = ("_name", "_description", "_buckets", "_values", "_lock")

    # Default buckets for latency in seconds
    DEFAULT_BUCKETS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

    def __init__(
        self,
        name: str,
        description: str = "",
        buckets: Optional[tuple] = None,
    ) -> None:
        self._name = name
        self._description = description
        self._buckets = buckets or self.DEFAULT_BUCKETS
        self._values: Dict[str, Dict[str, float]] = {}  # labels -> {bucket: count, sum, count}
        self._lock = threading.Lock()

    def observe(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record an observation."""
        key = self._labels_key(labels)
        with self._lock:
            if key not in self._values:
                self._values[key] = {
                    "_sum": 0.0,
                    "_count": 0,
                    **{str(b): 0 for b in self._buckets},
                    "+Inf": 0,
                }

            data = self._values[key]
            data["_sum"] += value
            data["_count"] += 1

            # Update buckets
            for bucket in self._buckets:
                if value <= bucket:
                    data[str(bucket)] += 1
            data["+Inf"] += 1

    def get_percentile(self, percentile: float, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Estimate percentile value from histogram."""
        key = self._labels_key(labels)
        with self._lock:
            if key not in self._values:
                return None

            data = self._values[key]
            total = data["_count"]
            if total == 0:
                return None

            target = total * (percentile / 100.0)
            cumulative = 0

            prev_bucket = 0.0
            for bucket in self._buckets:
                bucket_count = data[str(bucket)]
                if cumulative + bucket_count >= target:
                    # Linear interpolation within bucket
                    fraction = (target - cumulative) / max(1, bucket_count)
                    return prev_bucket + (bucket - prev_bucket) * fraction
                cumulative += bucket_count
                prev_bucket = bucket

            return self._buckets[-1] if self._buckets else None

    def _labels_key(self, labels: Optional[Dict[str, str]]) -> str:
        if not labels:
            return ""
        return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))

    def collect(self) -> List[MetricValue]:
        """Collect histogram values."""
        with self._lock:
            now = time.time()
            result = []

            for key, data in self._values.items():
                labels = self._parse_labels(key) if key else {}

                # Sum and count
                result.append(MetricValue(
                    value=data["_sum"],
                    labels={**labels, "le": "sum"},
                    timestamp=now
                ))
                result.append(MetricValue(
                    value=data["_count"],
                    labels={**labels, "le": "count"},
                    timestamp=now
                ))

                # Buckets
                for bucket in self._buckets:
                    result.append(MetricValue(
                        value=data[str(bucket)],
                        labels={**labels, "le": str(bucket)},
                        timestamp=now
                    ))
                result.append(MetricValue(
                    value=data["+Inf"],
                    labels={**labels, "le": "+Inf"},
                    timestamp=now
                ))

            return result

    def _parse_labels(self, key: str) -> Dict[str, str]:
        if not key:
            return {}
        labels = {}
        for pair in key.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                labels[k] = v
        return labels

    @property
    def name(self) -> str:
        return self._name

    @property
    def type(self) -> MetricType:
        return MetricType.HISTOGRAM


class Timer:
    """Context manager for timing operations."""

    __slots__ = ("_histogram", "_labels", "_start")

    def __init__(
        self,
        histogram: Histogram,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        self._histogram = histogram
        self._labels = labels
        self._start = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        duration = time.perf_counter() - self._start
        self._histogram.observe(duration, self._labels)


class MetricsCollector:
    """
    Central metrics collector.

    Manages all metrics and provides collection/export.
    """

    __slots__ = ("_metrics", "_lock")

    def __init__(self) -> None:
        self._metrics: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def counter(self, name: str, description: str = "") -> Counter:
        """Get or create a counter."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Counter(name, description)
            return self._metrics[name]

    def gauge(self, name: str, description: str = "") -> Gauge:
        """Get or create a gauge."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Gauge(name, description)
            return self._metrics[name]

    def histogram(
        self,
        name: str,
        description: str = "",
        buckets: Optional[tuple] = None,
    ) -> Histogram:
        """Get or create a histogram."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Histogram(name, description, buckets)
            return self._metrics[name]

    def timer(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> Timer:
        """Get a timer for the named histogram."""
        hist = self.histogram(name)
        return Timer(hist, labels)

    def collect_all(self) -> Dict[str, List[MetricValue]]:
        """Collect all metrics."""
        with self._lock:
            return {name: m.collect() for name, m in self._metrics.items()}

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []

        for name, values in self.collect_all().items():
            metric = self._metrics.get(name)
            if metric is None:
                continue

            # Add type hint
            metric_type = "counter" if metric.type == MetricType.COUNTER else (
                "gauge" if metric.type == MetricType.GAUGE else "histogram"
            )
            lines.append(f"# TYPE {name} {metric_type}")

            for val in values:
                label_str = ""
                if val.labels:
                    labels = ",".join(f'{k}="{v}"' for k, v in val.labels.items())
                    label_str = f"{{{labels}}}"
                lines.append(f"{name}{label_str} {val.value}")

        return "\n".join(lines)

    def reset_all(self) -> None:
        """Reset all metrics."""
        with self._lock:
            for m in self._metrics.values():
                if hasattr(m, "reset"):
                    m.reset()

    def __repr__(self) -> str:
        return f"MetricsCollector({len(self._metrics)} metrics)"


# Global default metrics collector
_default_collector: Optional[MetricsCollector] = None


def get_default_collector() -> MetricsCollector:
    """Get the global default metrics collector."""
    global _default_collector
    if _default_collector is None:
        _default_collector = MetricsCollector()
    return _default_collector


# Pre-defined database metrics
class DatabaseMetrics:
    """Pre-defined metrics for database monitoring."""

    def __init__(self, collector: Optional[MetricsCollector] = None) -> None:
        self._collector = collector or get_default_collector()

        # Counters
        self.queries_total = self._collector.counter(
            "clade_queries_total",
            "Total number of queries executed"
        )
        self.rows_read = self._collector.counter(
            "clade_rows_read_total",
            "Total rows read"
        )
        self.rows_written = self._collector.counter(
            "clade_rows_written_total",
            "Total rows written"
        )
        self.buffer_hits = self._collector.counter(
            "clade_buffer_hits_total",
            "Buffer pool cache hits"
        )
        self.buffer_misses = self._collector.counter(
            "clade_buffer_misses_total",
            "Buffer pool cache misses"
        )
        self.transactions_total = self._collector.counter(
            "clade_transactions_total",
            "Total transactions"
        )

        # Gauges
        self.active_connections = self._collector.gauge(
            "clade_active_connections",
            "Number of active connections"
        )
        self.buffer_pool_pages = self._collector.gauge(
            "clade_buffer_pool_pages",
            "Pages in buffer pool"
        )
        self.active_transactions = self._collector.gauge(
            "clade_active_transactions",
            "Active transactions"
        )

        # Histograms
        self.query_duration = self._collector.histogram(
            "clade_query_duration_seconds",
            "Query execution time in seconds"
        )
        self.transaction_duration = self._collector.histogram(
            "clade_transaction_duration_seconds",
            "Transaction duration in seconds"
        )
        self.page_io_duration = self._collector.histogram(
            "clade_page_io_duration_seconds",
            "Page I/O duration in seconds"
        )

    def record_query(
        self,
        query_type: str,
        duration: float,
        rows_affected: int = 0,
    ) -> None:
        """Record a completed query."""
        self.queries_total.inc(labels={"type": query_type})
        self.query_duration.observe(duration, labels={"type": query_type})
        if rows_affected > 0:
            if query_type in ("select", "scan"):
                self.rows_read.inc(rows_affected)
            else:
                self.rows_written.inc(rows_affected)

    def record_buffer_access(self, hit: bool) -> None:
        """Record a buffer pool access."""
        if hit:
            self.buffer_hits.inc()
        else:
            self.buffer_misses.inc()

    def record_transaction(self, status: str, duration: float) -> None:
        """Record a completed transaction."""
        self.transactions_total.inc(labels={"status": status})
        self.transaction_duration.observe(duration, labels={"status": status})

    @property
    def buffer_hit_ratio(self) -> float:
        """Calculate buffer pool hit ratio."""
        hits = self.buffer_hits.get()
        misses = self.buffer_misses.get()
        total = hits + misses
        return hits / total if total > 0 else 0.0
