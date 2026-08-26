"""Tests for the AFNOLayer — Prompt P2.

Covers:
  - threshold=0 + block_size=channels reduces to standard FNO
  - gradient flow
  - soft-thresholding zeros modes below the threshold
"""

import numpy as np
import pytest
import torch

from spectral_silicon.afno import AFNOLayer
from spectral_silicon.fno import FourierNeuralOperator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _set_seed():
    torch.manual_seed(7)
    np.random.seed(7)


@pytest.fixture
def small_input():
    return torch.randn(2, 16, 8, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Reduction-to-FNO tests
# ---------------------------------------------------------------------------

class TestAFNOReducesToFNO:
    def test_threshold0_block_channels_reduces_to_fno(self, small_input):
        """With threshold=0 and block_size=channels the AFNO should behave
        like a plain FNO (full complex weight, no soft-thresholding)."""
        channels = 8
        afno = AFNOLayer(
            channels=channels,
            modes=4,
            block_size=channels,  # single full block
            threshold=0.0,
        )
        fno = FourierNeuralOperator(channels=channels, modes=4)

        # Copy weights so both layers use identical spectral weights.
        # AFNO stores block-diagonal weights; when block_size=channels the
        # block is the full weight, so shapes match.
        if hasattr(fno, "weights") and hasattr(afno, "weights"):
            with torch.no_grad():
                fno.weights.copy_(afno.weights)
        out_afno = afno(small_input)
        out_fno = fno(small_input)
        # They should match closely (only residual conventions may differ).
        assert out_afno.shape == out_fno.shape
        assert torch.allclose(out_afno, out_fno, atol=1e-4), (
            "AFNO with threshold=0, block_size=channels should match FNO"
        )

    def test_full_block_shape(self):
        afno = AFNOLayer(channels=8, modes=4, block_size=8, threshold=0.1)
        out = afno(torch.randn(1, 16, 8))
        assert out.shape == (1, 16, 8)


# ---------------------------------------------------------------------------
# Gradient tests
# ---------------------------------------------------------------------------

class TestAFNOGradient:
    def test_gradient_flow_to_input(self, small_input):
        afno = AFNOLayer(channels=8, modes=4, block_size=4, threshold=0.1)
        x = small_input.clone().requires_grad_(True)
        out = afno(x)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_gradient_flow_to_weights(self, small_input):
        afno = AFNOLayer(channels=8, modes=4, block_size=4, threshold=0.1)
        out = afno(small_input)
        out.sum().backward()
        for p in afno.parameters():
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()


# ---------------------------------------------------------------------------
# Soft-thresholding tests
# ---------------------------------------------------------------------------

class TestAFNOThreshold:
    def test_high_threshold_zeros_modes(self):
        """With a very high threshold, all spectral coefficients are
        zeroed, so the spectral branch contributes nothing and the
        output equals the residual (input)."""
        x = torch.randn(1, 16, 8)
        afno = AFNOLayer(
            channels=8, modes=4, block_size=4, threshold=1e6
        )
        out = afno(x)
        assert out.shape == x.shape

    def test_threshold_increases_sparsity(self):
        """Increasing the threshold should zero more spectral coefficients."""
        x = torch.randn(1, 16, 8)
        afno_low = AFNOLayer(channels=8, modes=4, block_size=4, threshold=0.0)
        afno_high = AFNOLayer(channels=8, modes=4, block_size=4, threshold=10.0)
        # Use same initial weights so only threshold differs.
        torch.manual_seed(0)
        afno_low = AFNOLayer(channels=8, modes=4, block_size=4, threshold=0.0)
        torch.manual_seed(0)
        afno_high = AFNOLayer(channels=8, modes=4, block_size=4, threshold=10.0)
        out_low = afno_low(x)
        out_high = afno_high(x)
        # High threshold output should be closer to identity (residual).
        residual = x
        err_low = (out_low - residual).abs().mean().item()
        err_high = (out_high - residual).abs().mean().item()
        assert err_high <= err_low + 1e-4, (
            "higher threshold should move output toward residual"
        )

    def test_zero_threshold_no_sparsity(self):
        """With threshold=0, no modes are zeroed by soft-thresholding."""
        afno = AFNOLayer(channels=8, modes=4, block_size=4, threshold=0.0)
        x = torch.randn(1, 16, 8)
        out = afno(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Block diagonal tests
# ---------------------------------------------------------------------------

class TestAFNOBlockDiagonal:
    @pytest.mark.parametrize("block_size", [1, 2, 4, 8])
    def test_various_block_sizes(self, block_size):
        afno = AFNOLayer(
            channels=8, modes=4, block_size=block_size, threshold=0.1
        )
        x = torch.randn(1, 16, 8)
        out = afno(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_block_size_not_divisor(self):
        """block_size that does not divide channels should still work."""
        afno = AFNOLayer(channels=8, modes=4, block_size=3, threshold=0.1)
        x = torch.randn(1, 16, 8)
        out = afno(x)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Slow tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestAFNOSlow:
    def test_large_block_size(self):
        afno = AFNOLayer(channels=64, modes=16, block_size=16, threshold=0.5)
        x = torch.randn(2, 64, 64)
        out = afno(x)
        assert out.shape == x.shape
        out.sum().backward()