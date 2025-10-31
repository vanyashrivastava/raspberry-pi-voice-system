# Owner: Samya
# Responsibility: Benchmark inference latency, memory, and throughput on Raspberry Pi 5 with AI Hat.
# Goals:
# - Provide simple microbenchmarks for per-request latency and batched throughput
# - Measure model size, memory usage, and CPU/GPU utilization
# Integration points:
# - Uses loaded model from `inference_engine`
# Testing requirements:
# - Repeatable runs and summary outputs that can be logged to disk for comparison


from __future__ import annotations

import time
import statistics
import tracemalloc
import json
import cProfile
import pstats
import io
import logging
from typing import Any, Callable, Dict, List, Optional

# Try to import psutil for resident memory (recommended but optional)
try:
    import psutil
except Exception:
    psutil = None

# Import inference helpers from your engine
from .inference_engine import (
    load_model,
    run_inference,
    batch_inference,
    get_model_status,
    InferenceEngine,
)

logger = logging.getLogger(__name__)


class PerformanceBenchmarks:
    """
    Utilities to measure latency and basic throughput for the inference engine.

    Methods you can call:
      - measure_latency(fn, inputs, runs=10) -> dict(mean_ms, p95_ms, max_ms, samples_ms)
      - measure_throughput(fn, inputs, duration_s=10) -> dict(items_per_sec, total_items, total_time_s)
      - benchmark_inference_speed(num_samples=100, sample_text=...) -> dict
      - benchmark_throughput_batch(batch_sizes=[1,4,8], num_batches=50) -> dict
      - benchmark_memory_usage(sample_text=...) -> dict
      - profile_bottlenecks(sample_text=...) -> str (cProfile output)
      - compare_model_versions(model_v1, model_v2, sample_text=...) -> dict
    """

    @staticmethod
    def _now() -> float:
        return time.perf_counter()

    @staticmethod
    def measure_latency(fn: Callable[..., Any], inputs: List[Any], runs: int = 10) -> Dict[str, Any]:
        """
        Measure per-call latency of fn(*inputs) over `runs` iterations.
        Returns ms statistics.
        """
        if runs <= 0:
            raise ValueError("runs must be > 0")

        times_ms: List[float] = []
        # small warm-up
        try:
            fn(*inputs)
        except Exception:
            # ignore warm-up errors
            pass

        for _ in range(runs):
            t0 = PerformanceBenchmarks._now()
            fn(*inputs)
            t1 = PerformanceBenchmarks._now()
            times_ms.append((t1 - t0) * 1000.0)

        times_sorted = sorted(times_ms)
        p95_idx = max(0, int(0.95 * len(times_sorted)) - 1)
        result = {
            "count": runs,
            "mean_ms": statistics.mean(times_ms),
            "median_ms": statistics.median(times_ms),
            "min_ms": min(times_ms),
            "max_ms": max(times_ms),
            "p95_ms": times_sorted[p95_idx],
            "samples_ms": times_ms,
        }
        return result

    @staticmethod
    def measure_throughput(fn: Callable[..., Any], inputs: List[Any], duration_s: int = 10) -> Dict[str, Any]:
        """
        Run fn(*inputs) repeatedly for duration_s seconds and report items/sec.
        Returns total_time_s and items_per_sec.
        """
        if duration_s <= 0:
            raise ValueError("duration_s must be > 0")

        # warm-up
        try:
            fn(*inputs)
        except Exception:
            pass

        end_ts = time.time() + duration_s
        count = 0
        start = time.time()
        while time.time() < end_ts:
            fn(*inputs)
            count += 1
        total_time = time.time() - start
        items_per_sec = count / total_time if total_time > 0 else 0.0

        return {"duration_s": total_time, "items_processed": count, "items_per_sec": items_per_sec}

    # ---------------- higher-level convenience benchmarks ----------------

    @staticmethod
    def benchmark_inference_speed(num_samples: int = 100, sample_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Benchmarks single-item inference using run_inference.
        """
        sample_text = sample_text or "Benchmark sample: quick synthetic input for latency measurement."
        # Ensure model loaded / warmed
        load_model(None, warmup_examples=1)

        def fn(x):
            return run_inference(x)

        res = PerformanceBenchmarks.measure_latency(fn, [sample_text], runs=num_samples)
        logger.info("Inference speed benchmark: %s", res)
        return res

    @staticmethod
    def benchmark_throughput_batch(batch_sizes: List[int] = [1, 4, 8], num_batches: int = 50) -> Dict[int, Dict[str, Any]]:
        """
        Measure throughput for different batch sizes using batch_inference.
        Returns a dict keyed by batch size with items_per_sec.
        """
        load_model(None, warmup_examples=1)
        out: Dict[int, Dict[str, Any]] = {}
        for b in batch_sizes:
            total_items = 0
            total_time = 0.0
            for _ in range(num_batches):
                batch = [f"throughput-sample-{_}-{i}" for i in range(b)]
                t0 = PerformanceBenchmarks._now()
                batch_inference(batch)
                t1 = PerformanceBenchmarks._now()
                total_time += (t1 - t0)
                total_items += b
            items_per_sec = total_items / total_time if total_time > 0 else 0.0
            out[b] = {"items_processed": total_items, "total_time_s": total_time, "items_per_sec": items_per_sec}
            logger.info("Batch size %d -> %s items/sec", b, out[b]["items_per_sec"])
        return out

    @staticmethod
    def benchmark_memory_usage(sample_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Report memory usage before/after load + a single inference. Uses psutil if available,
        otherwise returns tracemalloc diffs (top lines).
        """
        sample_text = sample_text or "memory-bench-sample"
        if psutil:
            proc = psutil.Process()
            rss_before = proc.memory_info().rss / (1024 * 1024)
            load_model(None, warmup_examples=1)
            rss_after_load = proc.memory_info().rss / (1024 * 1024)
            run_inference(sample_text)
            rss_after_infer = proc.memory_info().rss / (1024 * 1024)
            return {
                "rss_mb_before": rss_before,
                "rss_mb_after_load": rss_after_load,
                "rss_mb_after_infer": rss_after_infer,
                "delta_after_load_mb": rss_after_load - rss_before,
                "delta_after_infer_mb": rss_after_infer - rss_after_load,
            }
        else:
            tracemalloc.start()
            load_model(None, warmup_examples=1)
            snap1 = tracemalloc.take_snapshot()
            run_inference(sample_text)
            snap2 = tracemalloc.take_snapshot()
            stats = snap2.compare_to(snap1, "lineno")
            top5 = [str(s) for s in stats[:5]]
            tracemalloc.stop()
            return {"tracemalloc_top5": top5}

    @staticmethod
    def profile_bottlenecks(sample_text: Optional[str] = None, print_output: bool = True) -> str:
        """
        Run a single inference under cProfile and return formatted profiling string.
        """
        sample_text = sample_text or "profile-benchmark-sample"
        load_model(None, warmup_examples=1)
        pr = cProfile.Profile()
        pr.enable()
        run_inference(sample_text)
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
        ps.print_stats(40)
        out = s.getvalue()
        if print_output:
            print(out)
        return out

    @staticmethod
    def compare_model_versions(model_v1: str, model_v2: str, sample_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Load model_v1 then model_v2 and measure basic speed + a sample inference.
        WARNING: loading multiple HF models is slow and memory heavy; use small checkpoints.
        """
        sample_text = sample_text or "compare-model-sample"
        results: Dict[str, Any] = {}

        for v in (model_v1, model_v2):
            load_model(v, warmup_examples=1)
            speed = PerformanceBenchmarks.benchmark_inference_speed(num_samples=20, sample_text=sample_text)
            last_result = run_inference(sample_text)
            results[v] = {"speed": speed, "last_result": last_result, "status": get_model_status()}
            # free caches where possible (best-effort)
        return results


# CLI convenience
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pb = PerformanceBenchmarks()
    print("Running quick inference speed (20 samples)...")
    print(json.dumps(pb.benchmark_inference_speed(num_samples=20), indent=2))
    print("Running throughput test (1,4,8, num_batches=20)...")
    print(json.dumps(pb.benchmark_throughput_batch(batch_sizes=[1, 4, 8], num_batches=20), indent=2))
    print("Memory usage:")
    print(json.dumps(pb.benchmark_memory_usage(), indent=2))
    print("Profile (single run):")
    pb.profile_bottlenecks()