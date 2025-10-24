# Owner: Samya
# Responsibility: Benchmark inference latency, memory, and throughput on Raspberry Pi 5 with AI Hat.
# Goals:
# - Provide simple microbenchmarks for per-request latency and batched throughput
# - Measure model size, memory usage, and CPU/GPU utilization
# Integration points:
# - Uses loaded model from `inference_engine`
# Testing requirements:
# - Repeatable runs and summary outputs that can be logged to disk for comparison

import time
import statistics
import typing as t


class PerformanceBenchmarks:
    """
    Utilities to measure latency and basic throughput.

    Methods:
        - measure_latency(callable_fn, inputs, runs=10) -> dict(mean, p95, max)
        - measure_throughput(callable_fn, inputs, duration_s=10) -> items_per_sec
    """

    @staticmethod
    def measure_latency(fn, inputs: t.List, runs: int = 10) -> dict:
        times = []
        for i in range(runs):
            start = time.perf_counter()
            fn(*inputs)
            times.append(time.perf_counter() - start)
        return {'mean': statistics.mean(times), 'p95': statistics.quantiles(times, n=20)[-1], 'max': max(times)}

    @staticmethod
    def measure_throughput(fn, inputs: t.List, duration_s: int = 10) -> float:
        end_time = time.time() + duration_s
        count = 0
        while time.time() < end_time:
            fn(*inputs)
            count += 1
        return count / float(duration_s)


if __name__ == '__main__':
    print('PerformanceBenchmarks skeleton - call measure_latency with real engine functions')
