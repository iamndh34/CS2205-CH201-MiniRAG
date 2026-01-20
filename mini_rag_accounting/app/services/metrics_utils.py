"""
Utility functions for measuring real system resources.
Uses psutil for CPU/RAM monitoring.
"""

import os
import psutil
import time
from dataclasses import dataclass
from typing import Optional
from contextlib import contextmanager


@dataclass
class ResourceMetrics:
    """Container for resource measurements"""
    time_seconds: float = 0.0
    ram_mb: float = 0.0
    cpu_percent: float = 0.0
    ram_delta_mb: float = 0.0  # RAM change during operation


class ResourceMonitor:
    """
    Monitor system resources during an operation.

    Usage:
        monitor = ResourceMonitor()
        monitor.start()
        # ... do work ...
        metrics = monitor.stop()
    """

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self._start_time: Optional[float] = None
        self._start_ram: Optional[float] = None
        self._cpu_samples: list = []

    def start(self):
        """Start monitoring"""
        # Get initial measurements
        self._start_time = time.perf_counter()
        self._start_ram = self.process.memory_info().rss / (1024 * 1024)  # MB

        # Reset CPU percent counter (first call returns 0)
        self.process.cpu_percent()
        self._cpu_samples = []

    def sample_cpu(self):
        """Sample CPU usage during operation"""
        if self._start_time is not None:
            cpu = self.process.cpu_percent()
            if cpu > 0:
                self._cpu_samples.append(cpu)

    def stop(self) -> ResourceMetrics:
        """Stop monitoring and return metrics"""
        if self._start_time is None:
            return ResourceMetrics()

        # Time
        elapsed = time.perf_counter() - self._start_time

        # RAM
        current_ram = self.process.memory_info().rss / (1024 * 1024)  # MB
        ram_delta = current_ram - self._start_ram

        # CPU - get final sample and average
        final_cpu = self.process.cpu_percent()
        if final_cpu > 0:
            self._cpu_samples.append(final_cpu)

        avg_cpu = sum(self._cpu_samples) / len(self._cpu_samples) if self._cpu_samples else 0

        return ResourceMetrics(
            time_seconds=elapsed,
            ram_mb=current_ram,
            cpu_percent=avg_cpu,
            ram_delta_mb=ram_delta
        )


@contextmanager
def measure_resources():
    """
    Context manager for measuring resources.

    Usage:
        with measure_resources() as monitor:
            # ... do work ...
        metrics = monitor.metrics
    """
    monitor = ResourceMonitor()
    monitor.start()

    class MetricsHolder:
        metrics: Optional[ResourceMetrics] = None

    holder = MetricsHolder()

    try:
        yield holder
    finally:
        holder.metrics = monitor.stop()


def get_current_memory_mb() -> float:
    """Get current process memory in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_system_memory_info() -> dict:
    """Get system-wide memory info"""
    mem = psutil.virtual_memory()
    return {
        "total_mb": mem.total / (1024 * 1024),
        "available_mb": mem.available / (1024 * 1024),
        "percent_used": mem.percent
    }
