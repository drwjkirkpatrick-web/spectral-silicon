"""Tests for the FourierNeuralOperator (FNO) layer — Prompt P1.

Covers:
  - zero modes (k=0) → identity
  - output shape matches input shape
  - gradient flow through the layer
  - different k (mode truncation) values
"""

import numpy as np
import pytest
import torch

from spectral_silicon.fno import FourierNeuralOperator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _set_seed():
    torch.manual_seed(42)
    np.random.seed(42)


@pytest.fixture
def small_input():
    """A small (batch=2, seq_len=16, channels=8) float32 tensor."""
    return torch.randn(2, 16, 8, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Shape / identity tests
# ---------------------------------------------------------------------------

class TestFNOShape:
    def test_output_shape_matches_input(self, small_input):
        fno = FourierNeuralOperator(channels=8, modes=4)
        out = fno(small_input)
        assert out.shape == small_input.shape

    def test_output_shape_various(self):
        for batch, seq_len, ch in [(1, 8, 4), (3, 32, 16), (2, 64, 8)]:
            x = torch.randn(batch, seq_len, ch)
            fno = FourierNeuralOperator(channels=ch, modes=min(seq_len // 2, 4))
            out = fno(x)
            assert out.shape == x.shape, f"shape mismatch for {x.shape}"

    def test_dtype_preserved(self, small_input):
        fno = FourierNeuralOperator(channels=8, modes=4)
        out = fno(small_input)
        assert out.dtype == small_input.dtype


class TestFNOZeroModes:
    """With k=0 (zero modes retained) the spectral path should be zero,
    so the output equals the (possible) residual/identity path."""

    def test_zero_modes_identity(self, small_input):
        fno = FourierNeuralOperator(channels=8, modes=0)
        out = fno(small_input)
        # With zero modes the spectral branch contributes nothing; if the
        # layer has a skip/identity it should return the input unchanged.
        assert torch.allclose(out, small_input, atol=1e-5), (
            "k=0 should produce identity (within float tolerance)"
        )

    def test_zero_modes_shape(self, small_input):
        fno = FourierNeuralOperator(channels=8, modes=0)
        out = fno(small_input)
        assert out.shape == small_input.shape


# ---------------------------------------------------------------------------
# Gradient tests
# ---------------------------------------------------------------------------

class TestFNOGradient:
    def test_gradient_flow(self, small_input):
        fno = FourierNeuralOperator(channels=8, modes=4)
        x = small_input.clone().requires_grad_(True)
        out = fno(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, "gradient must flow to input"
        assert x.grad.shape == x.shape
        assert torch.isfinite(x.grad).all()

    def test_gradient_flows_to_weights(self, small_input):
        fno = FourierNeuralOperator(channels=8, modes=4)
        out = fno(small_input)
        out.sum().backward()
        params = list(fno.parameters())
        assert len(params) > 0, "FNO must have learnable parameters"
        for p in params:
            assert p.grad is not None, "all parameters must receive gradients"
            assert torch.isfinite(p.grad).all()


# ---------------------------------------------------------------------------
# Mode truncation tests
# ---------------------------------------------------------------------------

class TestFNOOptions:
    @pytest.mark.parametrize("modes", [1, 2, 4, 8])
    def test_different_k_values(self, modes):
        x = torch.randn(2, 16, 8)
        fno = FourierNeuralOperator(channels=8, modes=modes)
        out = fno(x)
        assert out.shape == x.shape

    def test_more_modes_changes_output(self):
        x = torch.randn(2, 16, 8)
        fno_low = FourierNeuralOperator(channels=8, modes=1)
        fno_high = FourierNeuralOperator(channels=8, modes=8)
        # Initialize identically (same seed), then they differ only in modes.
        torch.manual_seed(0)
        fno_low = FourierNeuralOperator(channels=8, modes=1)
        torch.manual_seed(0)
        fno_high = FourierNeuralOperator(channels=8, modes=8)
        out_low = fno_low(x)
        out_high = fno_high(x)
        assert not torch.allclose(out_low, out_high, atol=1e-4), (
            "more modes should change the output"
        )

    def test_modes_exceeding_half_seq(self):
        # modes > seq_len//2 should not crash — the layer clamps internally.
        x = torch.randn(1, 8, 4)
        fno = FourierNeuralOperator(channels=4, modes=10)
        out = fno(x)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Slow / integration tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestFNOSlow:
    def test_long_sequence(self):
        x = torch.randn(1, 512, 32)
        fno = FourierNeuralOperator(channels=32, modes=16)
        out = fno(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()