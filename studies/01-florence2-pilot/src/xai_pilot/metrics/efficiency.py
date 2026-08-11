"""Efficiency: per-call cost and a linear extrapolation to full-study scale.

Only Day 3's baseline_predictions.csv logs inference_ms -- Days 5-7 never
instrumented their own rerun calls. Days 3 and 5 reuse the exact same call
type (answer_rule, default num_beams=3 beam search), so Day 5's per-sample
cost is recovered exactly from Day 3's per-call timing times the known
number of reruns per sample (1, or 2 if a second region existed), with no
new model calls. Days 6 and 7 use two decoding configs never timed before
(output_attentions=True forces eager attention in Florence-2's attention
classes -- slower than the SDPA path; do_sample=True/num_beams=1 has no
beam multiplicity -- plausibly faster), so the script that uses this module
runs a small fresh calibration for those two call types rather than silently
assuming they cost the same as Day 3's calls.
"""

from time import perf_counter

import numpy as np


def time_call_cuda(fn):
    """Run `fn`, returning (result, gpu_ms, wall_ms) with a standardized boundary.

    Separates GPU-compute time from end-to-end wall-clock (Scale-up Phase 2.6):
    `gpu_ms` is the GPU-busy time between two CUDA events (0 while the GPU is
    idle waiting on CPU work), and `wall_ms` is the total call time. Their ratio
    is the GPU-bound-ness -- gpu_ms approximately equal to wall_ms means the call
    is compute-bound, not I/O- or Python-bound. The boundary is made explicit
    with torch.cuda.synchronize() so asynchronous kernel launches are fully
    accounted rather than under-counted. On CPU, gpu_ms is NaN.
    """
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall0 = perf_counter()
        start.record()
        result = fn()
        end.record()
        torch.cuda.synchronize()
        wall_ms = (perf_counter() - wall0) * 1000.0
        gpu_ms = float(start.elapsed_time(end))
    else:
        wall0 = perf_counter()
        result = fn()
        wall_ms = (perf_counter() - wall0) * 1000.0
        gpu_ms = float("nan")
    return result, gpu_ms, wall_ms


def aggregate_timings(timings_ms: list[float]) -> dict:
    """Mean/median/std/max of a list of per-call timings in milliseconds."""
    # NumPy provides a consistent population standard deviation (`ddof=0`)
    # and handles the scalar conversion needed for CSV serialization.
    arr = np.asarray(timings_ms, dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean_ms": float("nan"), "median_ms": float("nan"), "std_ms": float("nan"), "max_ms": float("nan")}
    return {
        "n": int(arr.size),
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "std_ms": float(arr.std()),
        "max_ms": float(arr.max()),
    }


def summarize_timings(timings_ms: list[float], warmup: int = 0) -> dict:
    """Per-sample timing stats after discarding the first `warmup` measurements
    (Scale-up Phase 2.6).

    Retires the pilot's 15-sample calibration in favor of real per-sample
    timing: the warm-up iterations (cold caches, autotuning) are dropped, and a
    p90 is added so tail cost is visible, not just the mean.
    """
    measured = timings_ms[warmup:]
    arr = np.asarray(measured, dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean_ms": float("nan"), "median_ms": float("nan"),
                "p90_ms": float("nan"), "std_ms": float("nan"), "max_ms": float("nan")}
    return {
        "n": int(arr.size),
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p90_ms": float(np.percentile(arr, 90)),
        "std_ms": float(arr.std()),
        "max_ms": float(arr.max()),
    }


def extrapolate(per_sample_ms: float, n_samples: int) -> dict:
    """Linear scaling of a per-sample cost to a larger sample count."""
    total_ms = per_sample_ms * n_samples
    return {
        "n_samples": n_samples,
        "total_seconds": total_ms / 1000,
        "total_minutes": total_ms / 60_000,
        "total_hours": total_ms / 3_600_000,
    }
