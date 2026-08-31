#!/usr/bin/env python3
"""Complexity benchmark: spectral mixing vs. self-attention.

Measures wall-clock time for spectral mixing (AFNO) vs. standard O(n²)
self-attention at various sequence lengths, prints a comparison table, and
estimates the scaling exponents.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --seq-lens 128 256 512 1024
    python scripts/benchmark.py --channels 64 --repeats 10
"""

import argparse
import sys
import time

import numpy as np

# Ensure the project root is on sys.path so `spectral_silicon` is importable
# regardless of the caller's working directory or PYTHONPATH.
import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Error: torch is required. Install with: pip install torch", file=sys.stderr)
    sys.exit(1)

from spectral_silicon.afno import AFNOLayer
from spectral_silicon.transformer import SpectralTransformerBlock


# ---------------------------------------------------------------------------
# Baseline self-attention (O(n²))
# ---------------------------------------------------------------------------

class SelfAttention:
    """Simple O(n²) self-attention for benchmarking."""

    def __init__(self, channels):
        self.channels = channels
        self.q = torch.nn.Linear(channels, channels, bias=False)
        self.k = torch.nn.Linear(channels, channels, bias=False)
        self.v = torch.nn.Linear(channels, channels, bias=False)

    def __call__(self, x):
        d = self.channels
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d)
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v)


# ---------------------------------------------------------------------------
# Timing utility
# ---------------------------------------------------------------------------

def measure_time(fn, x, warmup=3, repeats=10):
    """Measure median wall-clock time of fn(x)."""
    for _ in range(warmup):
        _ = fn(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = fn(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return np.median(times)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_benchmark(seq_lens, channels, repeats, warmup):
    """Run benchmark and return results table."""
    torch.manual_seed(42)
    results = []

    print(f"{'Seq Len':>8} {'Spectral (ms)':>15} {'Attention (ms)':>16} {'Speedup':>10}")
    print("-" * 55)

    for n in seq_lens:
        x = torch.randn(1, n, channels)
        modes = min(n // 2, 32)

        # Spectral (AFNO)
        spectral = AFNOLayer(channels=channels, n_modes=modes, block_size=8)
        t_spectral = measure_time(spectral, x, warmup, repeats)

        # Self-attention
        attention = SelfAttention(channels)
        t_attention = measure_time(attention, x, warmup, repeats)

        speedup = t_attention / t_spectral if t_spectral > 0 else float("inf")
        results.append((n, t_spectral, t_attention, speedup))
        print(f"{n:>8} {t_spectral * 1000:>15.3f} {t_attention * 1000:>16.3f} {speedup:>10.2f}x")

    # Estimate scaling exponents
    print("\n--- Scaling Exponents (power-law fit: time ~ n^alpha) ---")
    log_n = np.log(np.array(seq_lens, dtype=float))
    log_s = np.log(np.array([r[1] for r in results]) + 1e-12)
    log_a = np.log(np.array([r[2] for r in results]) + 1e-12)

    alpha_spectral = np.polyfit(log_n, log_s, 1)[0]
    alpha_attention = np.polyfit(log_n, log_a, 1)[0]

    print(f"  Spectral (AFNO):  alpha = {alpha_spectral:.3f}  (theory: O(n log n) ≈ 1.x)")
    print(f"  Self-attention:   alpha = {alpha_attention:.3f}  (theory: O(n²) ≈ 2.0)")

    if alpha_spectral < alpha_attention:
        print(f"\n  ✓ Spectral scales better (Δalpha = {alpha_attention - alpha_spectral:.3f})")
    else:
        print(f"\n  ✗ Attention scales better? (unexpected)")

    return results, alpha_spectral, alpha_attention


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark spectral mixing vs. self-attention complexity."
    )
    parser.add_argument(
        "--seq-lens", type=int, nargs="+",
        default=[128, 256, 512, 1024, 2048, 4096],
        help="Sequence lengths to benchmark. Default: 128 256 512 1024 2048 4096"
    )
    parser.add_argument(
        "--channels", type=int, default=32,
        help="Number of channels/hidden dim. Default: 32"
    )
    parser.add_argument(
        "--repeats", type=int, default=10,
        help="Number of timed repeats per measurement. Default: 10"
    )
    parser.add_argument(
        "--warmup", type=int, default=3,
        help="Warmup iterations before timing. Default: 3"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Spectral Silicon — Complexity Benchmark")
    print(f"Channels: {args.channels}  Repeats: {args.repeats}  Device: "
          f"{'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 60)

    results, alpha_s, alpha_a = run_benchmark(
        args.seq_lens, args.channels, args.repeats, args.warmup
    )

    print("\n" + "=" * 60)
    print("Benchmark complete.")
    print("=" * 60)

    # Exit code: 0 if spectral scales better, 1 otherwise
    sys.exit(0 if alpha_s < alpha_a else 1)


if __name__ == "__main__":
    main()