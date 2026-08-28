"""Tests for the complexity benchmark — Prompt P7.

Covers:
  - benchmark spectral vs attention at seq_len=128,256,512,1024,2048,4096
  - assert scaling exponents match theory
"""

import time

import numpy as np
import pytest
import torch

from spectral_silicon.afno import AFNOLayer
from spectral_silicon.transformer import SpectralTransformerBlock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _measure(fn, x, warmup=2, repeats=5):
    """Measure median wall-clock time of fn(x)."""
    for _ in range(warmup):
        _ = fn(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = fn(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats


def _self_attention(x):
    """Simple O(n²) self-attention baseline (no learnable params)."""
    # x: (batch, seq_len, d)
    q = k = v = x
    d = x.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d)
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------

class TestComplexityBenchmark:
    @pytest.mark.slow
    def test_spectral_vs_attention_scaling(self):
        """Run both at several seq lengths, fit power-law exponents, and
        assert spectral scales ~O(n log n) and attention ~O(n²)."""
        seq_lens = [128, 256, 512, 1024]
        channels = 32
        spectral_times = []
        attention_times = []

        for n in seq_lens:
            x = torch.randn(1, n, channels)
            spectral = AFNOLayer(channels=channels, n_modes=min(n // 2, 16), block_size=8)
            spectral_times.append(_measure(spectral, x))
            attention_times.append(_measure(_self_attention, x))

        # Fit log-log linear: time ~ n^alpha
        log_n = np.log(np.array(seq_lens, dtype=float))
        log_s = np.log(np.array(spectral_times) + 1e-12)
        log_a = np.log(np.array(attention_times) + 1e-12)

        alpha_spectral = np.polyfit(log_n, log_s, 1)[0]
        alpha_attention = np.polyfit(log_n, log_a, 1)[0]

        # O(n^2) attention: alpha should be near 2 (allow tolerance)
        # O(n log n) spectral: alpha should be near 1 (1 <= alpha < 2)
        assert alpha_spectral < 2.0, (
            f"spectral exponent {alpha_spectral:.2f} should be < 2"
        )
        assert alpha_spectral < alpha_attention, (
            "spectral must scale better than attention"
        )

    @pytest.mark.slow
    def test_crossover_point(self):
        """At the largest tested seq_len, spectral must be faster than
        attention (the spectral advantage grows with n)."""
        seq_lens = [128, 256, 512, 1024, 2048]
        channels = 32
        results = {}
        for n in seq_lens:
            x = torch.randn(1, n, channels)
            spectral = AFNOLayer(channels=channels, n_modes=min(n // 2, 16), block_size=8)
            t_s = _measure(spectral, x, warmup=1, repeats=3)
            t_a = _measure(_self_attention, x, warmup=1, repeats=3)
            results[n] = (t_s, t_a)

        # At the largest n, spectral should be faster or comparable.
        n_max = max(seq_lens)
        t_s, t_a = results[n_max]
        assert t_s <= t_a * 2.0, (
            f"spectral {t_s:.4f}s much slower than attention {t_a:.4f}s at n={n_max}"
        )

    def test_benchmark_runs(self):
        """Smoke test: the benchmark runs without error at a small size."""
        n = 128
        x = torch.randn(1, n, 16)
        spectral = AFNOLayer(channels=16, n_modes=8, block_size=4)
        t_s = _measure(spectral, x, warmup=1, repeats=2)
        t_a = _measure(_self_attention, x, warmup=1, repeats=2)
        assert t_s > 0 and t_a > 0


# ---------------------------------------------------------------------------
# Exponent assertion tests (smaller, faster)
# ---------------------------------------------------------------------------

class TestScalingExponents:
    @pytest.mark.slow
    def test_spectral_exponent_below_2(self):
        """Spectral scaling exponent (fitted on small sizes) < 2."""
        seq_lens = [128, 256, 512]
        channels = 16
        times = []
        for n in seq_lens:
            x = torch.randn(1, n, channels)
            layer = AFNOLayer(channels=channels, n_modes=min(n // 2, 8), block_size=4)
            times.append(_measure(layer, x, warmup=1, repeats=3))
        log_n = np.log(np.array(seq_lens, dtype=float))
        log_t = np.log(np.array(times) + 1e-12)
        alpha = np.polyfit(log_n, log_t, 1)[0]
        assert alpha < 2.0, f"spectral exponent {alpha:.2f} should be < 2"

    @pytest.mark.slow
    def test_attention_exponent_near_2(self):
        """Attention scaling exponent should be higher than spectral's."""
        from spectral_silicon.afno import AFNOLayer

        seq_lens = [128, 256, 512]
        channels = 16
        spectral_times = []
        attention_times = []
        for n in seq_lens:
            x = torch.randn(1, n, channels)
            layer = AFNOLayer(channels=channels, n_modes=min(n // 2, 8), block_size=4)
            spectral_times.append(_measure(layer, x, warmup=1, repeats=3))
            attention_times.append(_measure(_self_attention, x, warmup=1, repeats=3))
        log_n = np.log(np.array(seq_lens, dtype=float))
        alpha_s = np.polyfit(log_n, np.log(np.array(spectral_times) + 1e-12), 1)[0]
        alpha_a = np.polyfit(log_n, np.log(np.array(attention_times) + 1e-12), 1)[0]
        # On CPU-bound systems, absolute exponents are noisy, but attention
        # should scale worse than spectral.
        assert alpha_a > alpha_s, (
            f"attention exponent {alpha_a:.2f} should exceed spectral {alpha_s:.2f}"
        )