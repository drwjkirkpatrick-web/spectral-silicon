"""Tests for the quantization module — Prompt P9.

Covers:
  - int8 quantization error < threshold
  - complex weight quantization
  - dequantize round-trip
"""

import numpy as np
import pytest
import torch

from spectral_silicon.quantize import (
    quantize_weight,
    quantize_complex_weights,
    dequantize,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _set_seed():
    torch.manual_seed(42)
    np.random.seed(42)


# ---------------------------------------------------------------------------
# Real (int8) quantization tests
# ---------------------------------------------------------------------------

class TestQuantizeReal:
    def test_quantize_range(self):
        w = torch.randn(100) * 5
        q, scale = quantize_weight(w)
        assert q.min() >= -128
        assert q.max() <= 127

    def test_quantize_error(self):
        """int8 quantization error must be < 1/128 of the weight range."""
        w = torch.randn(500) * 3
        q, scale = quantize_weight(w)
        dq = dequantize(q, scale, 0)
        error = float(np.abs(dq - w.numpy()).mean())
        weight_range = (w.max() - w.min()).item()
        assert error < weight_range / 128.0, (
            f"quant error {error} >= {weight_range/128}"
        )

    def test_quantize_zero_weight(self):
        w = torch.zeros(10)
        q, scale = quantize_weight(w)
        assert np.allclose(q, np.zeros_like(q))

    def test_quantize_preserves_shape(self):
        for shape in [(10,), (4, 8), (2, 3, 4), (16, 16)]:
            w = torch.randn(*shape)
            q, scale = quantize_weight(w)
            assert q.shape == tuple(w.shape)

    def test_quantize_int_dtype(self):
        w = torch.randn(20)
        q, scale = quantize_weight(w)
        assert q.dtype in (np.int8, np.int16, np.float32)
        # values should be integer-valued
        assert np.allclose(q.astype(np.float32), np.round(q.astype(np.float32)))


# ---------------------------------------------------------------------------
# Complex weight quantization tests
# ---------------------------------------------------------------------------

class TestQuantizeComplex:
    def test_quantize_complex_shape(self):
        w = torch.randn(8, 16, dtype=torch.complex64) * 2
        qw = quantize_complex_weights(w.real, w.imag)
        q_re = qw.real_int
        q_im = qw.imag_int
        assert q_re.shape == w.shape
        assert q_im.shape == w.shape

    def test_quantize_complex_range(self):
        w = torch.randn(100, dtype=torch.complex64) * 3
        qw = quantize_complex_weights(w.real, w.imag)
        q_re = qw.real_int
        q_im = qw.imag_int
        assert q_re.min() >= -128
        assert q_re.max() <= 127
        assert q_im.min() >= -128
        assert q_im.max() <= 127

    def test_quantize_complex_error(self):
        w = torch.randn(200, dtype=torch.complex64) * 2
        qw = quantize_complex_weights(w.real, w.imag)
        dq_re, dq_im = qw.dequantize()
        dq = dq_re + 1j * dq_im
        w_np = w.numpy()
        error = float(np.abs(dq - w_np).mean())
        ref_mag = float(np.abs(w_np).mean())
        assert error < ref_mag / 64.0

    def test_quantize_complex_zero(self):
        w = torch.zeros(10, dtype=torch.complex64)
        qw = quantize_complex_weights(w.real, w.imag)
        q_re = qw.real_int
        q_im = qw.imag_int
        assert np.allclose(q_re, np.zeros_like(q_re))
        assert np.allclose(q_im, np.zeros_like(q_im))


# ---------------------------------------------------------------------------
# Dequantize round-trip tests
# ---------------------------------------------------------------------------

class TestDequantRoundTrip:
    def test_real_round_trip(self):
        w = torch.randn(100) * 2
        q, scale = quantize_weight(w)
        dq = dequantize(q, scale, 0)
        w_np = w.numpy()
        # Round-trip error should be small relative to range
        range_val = (w.max() - w.min()).item()
        rel_err = float(np.abs(dq - w_np).max()) / range_val
        assert rel_err < 0.02, f"round-trip rel error {rel_err} too high"

    def test_complex_round_trip(self):
        w = torch.randn(50, dtype=torch.complex64) * 2
        qw = quantize_complex_weights(w.real, w.imag)
        dq_re, dq_im = qw.dequantize()
        dq = dq_re + 1j * dq_im
        w_np = w.numpy()
        rel_err = float(np.abs(dq - w_np).max()) / (float(np.abs(w_np).max()) + 1e-9)
        assert rel_err < 0.05, f"complex round-trip rel error {rel_err} too high"

    def test_round_trip_preserves_sign(self):
        w = torch.randn(50) * 5
        q, scale = quantize_weight(w)
        dq = dequantize(q, scale, 0)
        w_np = w.numpy()
        # signs should mostly match
        sign_match = float(np.mean(np.sign(dq) == np.sign(w_np)))
        assert sign_match > 0.9