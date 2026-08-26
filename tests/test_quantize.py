"""Tests for the quantization module — Prompt P9.

Covers:
  - int8 quantization error < threshold
  - complex weight quantization
  - dequantize round-trip
"""

import numpy as np
import pytest
import torch

from spectral_silicon.quantize import quantize_weight, quantize_complex_weights


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
        q = quantize_weight(w)
        assert q.min() >= -128
        assert q.max() <= 127

    def test_quantize_error(self):
        """int8 quantization error must be < 1/128 of the weight range."""
        w = torch.randn(500) * 3
        q = quantize_weight(w)
        # dequantize back
        if hasattr(quantize_weight, "dequantize"):
            dq = quantize_weight.dequantize(q, w.min(), w.max())
        else:
            scale = (w.max() - w.min()) / 255.0
            zero = (-128.0 - w.min() / scale) if scale > 0 else 0
            dq = (q.float() - zero) * scale
        error = (dq - w).abs().mean().item()
        weight_range = (w.max() - w.min()).item()
        assert error < weight_range / 128.0, (
            f"quant error {error} >= {weight_range/128}"
        )

    def test_quantize_zero_weight(self):
        w = torch.zeros(10)
        q = quantize_weight(w)
        assert torch.allclose(q, torch.zeros_like(q))

    def test_quantize_preserves_shape(self):
        for shape in [(10,), (4, 8), (2, 3, 4), (16, 16)]:
            w = torch.randn(*shape)
            q = quantize_weight(w)
            assert q.shape == w.shape

    def test_quantize_int_dtype(self):
        w = torch.randn(20)
        q = quantize_weight(w)
        assert q.dtype in (torch.int8, torch.int32, torch.float32)
        # values should be integer-valued
        assert torch.allclose(q.float(), q.float().round())


# ---------------------------------------------------------------------------
# Complex weight quantization tests
# ---------------------------------------------------------------------------

class TestQuantizeComplex:
    def test_quantize_complex_shape(self):
        w = torch.randn(8, 16, dtype=torch.complex64) * 2
        q_re, q_im = quantize_complex_weights(w)
        assert q_re.shape == w.shape
        assert q_im.shape == w.shape

    def test_quantize_complex_range(self):
        w = torch.randn(100, dtype=torch.complex64) * 3
        q_re, q_im = quantize_complex_weights(w)
        assert q_re.min() >= -128
        assert q_re.max() <= 127
        assert q_im.min() >= -128
        assert q_im.max() <= 127

    def test_quantize_complex_error(self):
        w = torch.randn(200, dtype=torch.complex64) * 2
        q_re, q_im = quantize_complex_weights(w)
        # Reconstruct and compare
        if hasattr(quantize_complex_weights, "dequantize"):
            dq = quantize_complex_weights.dequantize(q_re, q_im, w)
        else:
            # Manual dequantization: scale by real/imag ranges
            re = w.real
            im = w.imag
            scale_re = (re.max() - re.min()) / 255.0 if re.max() > re.min() else 1.0
            scale_im = (im.max() - im.min()) / 255.0 if im.max() > im.min() else 1.0
            dq_re = q_re.float() * scale_re + re.min()
            dq_im = q_im.float() * scale_im + im.min()
            dq = torch.complex(dq_re, dq_im)
        error = (dq - w).abs().mean().item()
        ref_mag = w.abs().mean().item()
        assert error < ref_mag / 64.0

    def test_quantize_complex_zero(self):
        w = torch.zeros(10, dtype=torch.complex64)
        q_re, q_im = quantize_complex_weights(w)
        assert torch.allclose(q_re, torch.zeros_like(q_re))
        assert torch.allclose(q_im, torch.zeros_like(q_im))


# ---------------------------------------------------------------------------
# Dequantize round-trip tests
# ---------------------------------------------------------------------------

class TestDequantRoundTrip:
    def test_real_round_trip(self):
        w = torch.randn(100) * 2
        q = quantize_weight(w)
        if hasattr(quantize_weight, "dequantize"):
            dq = quantize_weight.dequantize(q, w.min(), w.max())
        else:
            scale = (w.max() - w.min()) / 255.0 if w.max() > w.min() else 1.0
            zero = (-128.0 - w.min() / scale) if scale > 0 else 0
            dq = (q.float() - zero) * scale
        # Round-trip error should be small relative to range
        range_val = (w.max() - w.min()).item()
        rel_err = (dq - w).abs().max().item() / range_val
        assert rel_err < 0.02, f"round-trip rel error {rel_err} too high"

    def test_complex_round_trip(self):
        w = torch.randn(50, dtype=torch.complex64) * 2
        q_re, q_im = quantize_complex_weights(w)
        if hasattr(quantize_complex_weights, "dequantize"):
            dq = quantize_complex_weights.dequantize(q_re, q_im, w)
        else:
            re, im = w.real, w.imag
            sr = (re.max() - re.min()) / 255.0 if re.max() > re.min() else 1.0
            si = (im.max() - im.min()) / 255.0 if im.max() > im.min() else 1.0
            dq = torch.complex(q_re.float() * sr + re.min(), q_im.float() * si + im.min())
        rel_err = (dq - w).abs().max().item() / (w.abs().max().item() + 1e-9)
        assert rel_err < 0.05, f"complex round-trip rel error {rel_err} too high"

    def test_round_trip_preserves_sign(self):
        w = torch.randn(50) * 5
        q = quantize_weight(w)
        if hasattr(quantize_weight, "dequantize"):
            dq = quantize_weight.dequantize(q, w.min(), w.max())
        else:
            scale = (w.max() - w.min()) / 255.0
            zero = (-128.0 - w.min() / scale) if scale > 0 else 0
            dq = (q.float() - zero) * scale
        # signs should mostly match
        sign_match = (dq.sign() == w.sign()).float().mean()
        assert sign_match > 0.9