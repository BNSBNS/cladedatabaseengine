"""
Unit tests for metrics collection.

Tests cover:
- Counter metrics
- Gauge metrics
- Histogram metrics
- Timer context manager
- MetricsCollector
- DatabaseMetrics
- Prometheus export
"""

import pytest
import time
import threading

from clade.observability.metrics import (
    MetricType,
    MetricValue,
    Counter,
    Gauge,
    Histogram,
    Timer,
    MetricsCollector,
    DatabaseMetrics,
    get_default_collector,
)


class TestCounter:
    """Tests for Counter metric."""

    def test_increment_default(self):
        """Should increment by 1 by default."""
        counter = Counter("test_counter")
        counter.inc()
        assert counter.get() == 1

    def test_increment_value(self):
        """Should increment by specified value."""
        counter = Counter("test_counter")
        counter.inc(5)
        assert counter.get() == 5

    def test_increment_with_labels(self):
        """Should track separate values per label set."""
        counter = Counter("test_counter")
        counter.inc(labels={"type": "select"})
        counter.inc(labels={"type": "insert"})
        counter.inc(labels={"type": "select"})

        assert counter.get(labels={"type": "select"}) == 2
        assert counter.get(labels={"type": "insert"}) == 1

    def test_reset(self):
        """Should reset all values."""
        counter = Counter("test_counter")
        counter.inc(10)
        counter.inc(5, labels={"type": "a"})
        counter.reset()

        assert counter.get() == 0
        assert counter.get(labels={"type": "a"}) == 0

    def test_collect(self):
        """Should collect all values."""
        counter = Counter("test_counter")
        counter.inc(10)
        counter.inc(5, labels={"type": "a"})

        values = counter.collect()
        assert len(values) == 2

    def test_name_and_type(self):
        """Should return name and type."""
        counter = Counter("test_counter", "A test counter")
        assert counter.name == "test_counter"
        assert counter.type == MetricType.COUNTER

    def test_thread_safety(self):
        """Should be thread-safe."""
        counter = Counter("test_counter")
        threads = []

        def increment():
            for _ in range(1000):
                counter.inc()

        for _ in range(10):
            t = threading.Thread(target=increment)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert counter.get() == 10000


class TestGauge:
    """Tests for Gauge metric."""

    def test_set(self):
        """Should set value."""
        gauge = Gauge("test_gauge")
        gauge.set(42)
        assert gauge.get() == 42

    def test_increment(self):
        """Should increment value."""
        gauge = Gauge("test_gauge")
        gauge.set(10)
        gauge.inc(5)
        assert gauge.get() == 15

    def test_decrement(self):
        """Should decrement value."""
        gauge = Gauge("test_gauge")
        gauge.set(10)
        gauge.dec(3)
        assert gauge.get() == 7

    def test_with_labels(self):
        """Should track separate values per label set."""
        gauge = Gauge("test_gauge")
        gauge.set(10, labels={"host": "a"})
        gauge.set(20, labels={"host": "b"})

        assert gauge.get(labels={"host": "a"}) == 10
        assert gauge.get(labels={"host": "b"}) == 20

    def test_collect(self):
        """Should collect all values."""
        gauge = Gauge("test_gauge")
        gauge.set(10)
        gauge.set(20, labels={"host": "a"})

        values = gauge.collect()
        assert len(values) == 2

    def test_name_and_type(self):
        """Should return name and type."""
        gauge = Gauge("test_gauge", "A test gauge")
        assert gauge.name == "test_gauge"
        assert gauge.type == MetricType.GAUGE


class TestHistogram:
    """Tests for Histogram metric."""

    def test_observe(self):
        """Should observe values."""
        histogram = Histogram("test_histogram")
        histogram.observe(0.1)
        histogram.observe(0.2)
        histogram.observe(0.3)

        values = histogram.collect()
        # Should have sum, count, and bucket values
        assert len(values) > 0

    def test_percentile(self):
        """Should estimate percentiles."""
        histogram = Histogram("test_histogram", buckets=(0.1, 0.5, 1.0, 5.0))

        # Add values
        for _ in range(100):
            histogram.observe(0.05)  # Below 0.1 bucket
        for _ in range(100):
            histogram.observe(0.3)  # In 0.1-0.5 bucket

        p50 = histogram.get_percentile(50)
        assert p50 is not None
        assert 0 <= p50 <= 1.0

    def test_with_labels(self):
        """Should track separate values per label set."""
        histogram = Histogram("test_histogram")
        histogram.observe(0.1, labels={"type": "a"})
        histogram.observe(0.2, labels={"type": "b"})

        p50_a = histogram.get_percentile(50, labels={"type": "a"})
        p50_b = histogram.get_percentile(50, labels={"type": "b"})
        assert p50_a is not None
        assert p50_b is not None

    def test_collect_buckets(self):
        """Should collect bucket values."""
        histogram = Histogram("test_histogram", buckets=(0.1, 0.5, 1.0))
        histogram.observe(0.05)
        histogram.observe(0.3)
        histogram.observe(2.0)

        values = histogram.collect()
        # Should have: sum, count, 3 buckets, +Inf
        assert len(values) == 6

    def test_name_and_type(self):
        """Should return name and type."""
        histogram = Histogram("test_histogram", "A test histogram")
        assert histogram.name == "test_histogram"
        assert histogram.type == MetricType.HISTOGRAM


class TestTimer:
    """Tests for Timer context manager."""

    def test_timer_records_duration(self):
        """Should record elapsed time."""
        histogram = Histogram("test_duration")

        with Timer(histogram):
            time.sleep(0.01)  # 10ms

        p50 = histogram.get_percentile(50)
        assert p50 is not None
        assert p50 >= 0.01  # At least 10ms

    def test_timer_with_labels(self):
        """Should record with labels."""
        histogram = Histogram("test_duration")

        with Timer(histogram, labels={"op": "query"}):
            time.sleep(0.01)

        p50 = histogram.get_percentile(50, labels={"op": "query"})
        assert p50 is not None


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_create_counter(self):
        """Should create and return counters."""
        collector = MetricsCollector()
        counter = collector.counter("test_counter")
        assert counter is not None
        assert counter.name == "test_counter"

    def test_get_existing_counter(self):
        """Should return existing counter."""
        collector = MetricsCollector()
        counter1 = collector.counter("test_counter")
        counter2 = collector.counter("test_counter")
        assert counter1 is counter2

    def test_create_gauge(self):
        """Should create and return gauges."""
        collector = MetricsCollector()
        gauge = collector.gauge("test_gauge")
        assert gauge is not None
        assert gauge.name == "test_gauge"

    def test_create_histogram(self):
        """Should create and return histograms."""
        collector = MetricsCollector()
        histogram = collector.histogram("test_histogram")
        assert histogram is not None
        assert histogram.name == "test_histogram"

    def test_timer(self):
        """Should return timer for histogram."""
        collector = MetricsCollector()

        with collector.timer("test_duration"):
            time.sleep(0.01)

        histogram = collector.histogram("test_duration")
        p50 = histogram.get_percentile(50)
        assert p50 is not None

    def test_collect_all(self):
        """Should collect all metrics."""
        collector = MetricsCollector()
        collector.counter("counter1").inc()
        collector.gauge("gauge1").set(10)
        collector.histogram("histogram1").observe(0.1)

        all_metrics = collector.collect_all()
        assert "counter1" in all_metrics
        assert "gauge1" in all_metrics
        assert "histogram1" in all_metrics

    def test_export_prometheus(self):
        """Should export in Prometheus format."""
        collector = MetricsCollector()
        collector.counter("test_counter").inc(5)
        collector.gauge("test_gauge").set(42)

        output = collector.export_prometheus()
        assert "# TYPE test_counter counter" in output
        assert "test_counter" in output
        assert "5" in output
        assert "# TYPE test_gauge gauge" in output
        assert "test_gauge" in output
        assert "42" in output

    def test_export_prometheus_with_labels(self):
        """Should export labels in Prometheus format."""
        collector = MetricsCollector()
        collector.counter("requests").inc(labels={"method": "GET", "status": "200"})

        output = collector.export_prometheus()
        assert 'method="GET"' in output
        assert 'status="200"' in output

    def test_reset_all(self):
        """Should reset all metrics."""
        collector = MetricsCollector()
        collector.counter("counter1").inc(10)

        collector.reset_all()

        assert collector.counter("counter1").get() == 0

    def test_repr(self):
        """Should return string representation."""
        collector = MetricsCollector()
        collector.counter("c1")
        collector.gauge("g1")

        assert "2 metrics" in repr(collector)


class TestDatabaseMetrics:
    """Tests for DatabaseMetrics."""

    def test_record_query(self):
        """Should record query metrics."""
        collector = MetricsCollector()
        metrics = DatabaseMetrics(collector)

        metrics.record_query("select", 0.05, rows_affected=10)

        assert metrics.queries_total.get(labels={"type": "select"}) == 1
        assert metrics.rows_read.get() == 10

    def test_record_write_query(self):
        """Should record write query metrics."""
        collector = MetricsCollector()
        metrics = DatabaseMetrics(collector)

        metrics.record_query("insert", 0.02, rows_affected=5)

        assert metrics.queries_total.get(labels={"type": "insert"}) == 1
        assert metrics.rows_written.get() == 5

    def test_record_buffer_access(self):
        """Should record buffer accesses."""
        collector = MetricsCollector()
        metrics = DatabaseMetrics(collector)

        metrics.record_buffer_access(hit=True)
        metrics.record_buffer_access(hit=True)
        metrics.record_buffer_access(hit=False)

        assert metrics.buffer_hits.get() == 2
        assert metrics.buffer_misses.get() == 1

    def test_record_transaction(self):
        """Should record transaction metrics."""
        collector = MetricsCollector()
        metrics = DatabaseMetrics(collector)

        metrics.record_transaction("commit", 0.1)
        metrics.record_transaction("abort", 0.05)

        assert metrics.transactions_total.get(labels={"status": "commit"}) == 1
        assert metrics.transactions_total.get(labels={"status": "abort"}) == 1

    def test_buffer_hit_ratio(self):
        """Should calculate buffer hit ratio."""
        collector = MetricsCollector()
        metrics = DatabaseMetrics(collector)

        # 80% hit rate
        for _ in range(80):
            metrics.record_buffer_access(hit=True)
        for _ in range(20):
            metrics.record_buffer_access(hit=False)

        assert abs(metrics.buffer_hit_ratio - 0.8) < 0.01

    def test_buffer_hit_ratio_no_accesses(self):
        """Should return 0 when no accesses."""
        collector = MetricsCollector()
        metrics = DatabaseMetrics(collector)

        assert metrics.buffer_hit_ratio == 0.0


class TestDefaultCollector:
    """Tests for default collector."""

    def test_get_default_collector(self):
        """Should return singleton collector."""
        collector1 = get_default_collector()
        collector2 = get_default_collector()
        assert collector1 is collector2

    def test_database_metrics_uses_default(self):
        """Should use default collector when none provided."""
        metrics = DatabaseMetrics()
        assert metrics._collector is get_default_collector()
