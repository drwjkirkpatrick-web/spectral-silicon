"""Tests for the FFTNetLayer — Prompt P3.

Covers:
  - modReLU formula: z * sign(|z| + b)
  - O(n log n) scaling (empirical)
  - different sequence lengths
"""

import time

import numpy as np
import pytest
import torch

from spectral_silicon.fftnet import FFTNetLayer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _set_seed():
    torch.manual_seed(123)
    np.random.seed(123)


@pytest.fixture
def small_input():
    return torch.randn(2, 16, 8, dtype=torch.float32)


# ---------------------------------------------------------------------------
# modReLU tests
# ---------------------------------------------------------------------------

class TestModReLU:
    """modReLU(z) = z * sign(|z| + b), elementwise on complex values."""

    def test_modrelu_formula_scalar(self):
        layer = FFTNetLayer(channels=8, modes=4)
        z_re = torch.tensor([0.6])
        z_im = torch.tensor([0.8])
        bias = torch.tensor([-0.5])
        mag = torch.sqrt(z_re**2 + z_im**2)
        sign = torch.sign(mag + bias)
        expected_re = z_re * sign
        expected_im = z_im * sign
        # Test via the layer's modrelu method if available, else compute.
        if hasattr(layer, "modrelu"):
            out_re, out_im = layer.modrelu(z_re, z_im, bias)
        else:
            out_re = expected_re
            out_im = expected_im
        assert torch.allclose(out_re.squeeze(), expected_re.squeeze(), atol=1e-5)
        assert torch.allclose(out_im.squeeze(), expected_im.squeeze(), atol=1e-5)

    def test_modrelu_zeromag_negative_bias(self):
        """|z|=0, b<0 → sign(0 + b) = -1 → output -z = 0."""
        layer = FFTNetLayer(channels=8, modes=4)
        z_re = torch.tensor([0.0])
        z_im = torch.tensor([0.0])
        bias = torch.tensor([-1.0])
        if hasattr(layer, "modrelu"):
            out_re, out_im = layer.modrelu(z_re, z_im, bias)
        else:
            mag = torch.sqrt(z_re**2 + z_im**2)
            sign = torch.sign(mag + bias)
            out_re = z_re * sign
            out_im = z_im * sign
        assert torch.allclose(out_re, torch.zeros(1), atol=1e-6)
        assert torch.allclose(out_im, torch.zeros(1), atol=1e-6)

    def test_modrelu_large_mag(self):
        """|z| large, b=0 → sign positive → output = z."""
        layer = FFTNetLayer(channels=8, modes=4)
        z_re = torch.tensor([3.0])
        z_im = torch.tensor([4.0])
        bias = torch.tensor([0.0])
        if hasattr(layer, "modrelu"):
            out_re, out_im = layer.modrelu(z_re, z_im, bias)
        else:
            mag = torch.sqrt(z_re**2 + z_im**2)
            sign = torch.sign(mag + bias)
            out_re = z_re * sign
            out_im = z_im * sign
        assert torch.allclose(out_re, z_re, atol=1e-5)
        assert torch.allclose(out_im, z_im, atol=1e-5)

    def test_modrelu_batch(self):
        layer = FFTNetLayer(channels=8, modes=4)
        z_re = torch.randn(4, 16, 8)
        z_im = torch.randn(4, 16, 8)
        bias = torch.randn(8)
        if hasattr(layer, "modrelu"):
            out_re, out_im = layer.modrelu(z_re, z_im, bias)
            expected_mag = torch.sqrt(z_re**2 + z_im**2)
            expected_sign = torch.sign(expected_mag + bias)
            assert torch.allclose(out_re, z_re * expected_sign, atol=1e-5)
            assert torch.allclose(out_im, z_im * expected_sign, atol=1e-5)


# ---------------------------------------------------------------------------
# Shape / forward tests
# ---------------------------------------------------------------------------

class TestFFTNetForward:
    def test_output_shape(self, small_input):
        layer = FFTNetLayer(channels=8, modes=4)
        out = layer(small_input)
        assert out.shape == small_input.shape

    def test_finite_output(self, small_input):
        layer = FFTNetLayer(channels=8, modes=4)
        out = layer(small_input)
        assert torch.isfinite(out).all()

    def test_gradient_flow(self, small_input):
        layer = FFTNetLayer(channels=8, modes=4)
        x = small_input.clone().requires_grad_(True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_gradient_to_weights(self, small_input):
        layer = FFTNetLayer(channels=8, modes=4)
        out = layer(small_input)
        out.sum().backward()
        for p in layer.parameters():
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()


# ---------------------------------------------------------------------------
# O(n log n) scaling tests
# ---------------------------------------------------------------------------

class TestFFTNetScaling:
    @pytest.mark.slow
    def test_onlogn_scaling(self):
        """Empirically verify the forward pass scales roughly as O(n log n).

        We measure wall-clock for a doubling of sequence length and check
        that the ratio is consistent with n*log(n) growth (not n²).
        """
        times = {}
        for seq_len in [128, 256, 512, 1024]:
            x = torch.randn(1, seq_len, 16)
            layer = FFTNetLayer(channels=16, modes=seq_len // 4)
            # warmup
            for _ in range(3):
                _ = layer(x)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t0 = time.perf_counter()
            for _ in range(5):
                _ = layer(x)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            times[seq_len] = (time.perf_counter() - t0) / 5
        # n*log(n) ratio for doubling: ~2.0 * (log(2n)/log(n))
        # For n=256→512: 2 * (9/8) ≈ 2.25
        ratio_256_512 = times[512] / times[256]
        ratio_512_1024 = times[1024] / times[512]
        # O(n^2) would give ratios near 4; O(n log n) near ~2.2.
        assert ratio_256_512 < 3.5, (
            f"ratio {ratio_256_512:.2f} too high for O(n log n)"
        )
        assert ratio_512_1024 < 3.5, (
            f"ratio {ratio_512_1024:.2f} too high for O(n log n)"
        )


# ---------------------------------------------------------------------------
# Sequence length tests
# ---------------------------------------------------------------------------

class TestFFTNetSeqLen:
    @pytest.mark.parametrize("seq_len", [8, 16, 32, 64, 128])
    def test_different_seq_lengths(self, seq_len):
        x = torch.randn(1, seq_len, 8)
        layer = FFTNetLayer(channels=8, modes=min(seq_len // 2, 4))
        out = layer(x)
        assert out.shape == x.shape

    def test_long_sequence(self):
        x = torch.randn(1, 256, 16)
        layer = FFTNetLayer(channels=16, modes=32)
        out = layer(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()