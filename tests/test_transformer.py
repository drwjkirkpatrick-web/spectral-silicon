"""Tests for the SpectralTransformerBlock — Prompt P4.

Covers:
  - forward pass on random input
  - resolution invariance (same params for seq_len 512 vs 2048)
  - parameter count independent of seq_len
"""

import numpy as np
import pytest
import torch

from spectral_silicon.transformer import SpectralTransformerBlock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _set_seed():
    torch.manual_seed(99)
    np.random.seed(99)


@pytest.fixture
def block():
    return SpectralTransformerBlock(d_model=32, num_modes=16, block_size=16)


# ---------------------------------------------------------------------------
# Forward pass tests
# ---------------------------------------------------------------------------

class TestTransformerForward:
    def test_forward_random_input(self, block):
        x = torch.randn(2, 64, 32)
        out = block(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_forward_dtype(self, block):
        x = torch.randn(1, 32, 32, dtype=torch.float32)
        out = block(x)
        assert out.dtype == torch.float32

    def test_gradient_flow(self, block):
        x = torch.randn(2, 64, 32, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_gradient_to_weights(self, block):
        x = torch.randn(2, 64, 32)
        out = block(x)
        out.sum().backward()
        for p in block.parameters():
            assert p.grad is not None


# ---------------------------------------------------------------------------
# Resolution invariance tests
# ---------------------------------------------------------------------------

class TestResolutionInvariance:
    def test_forward_different_seq_lens_same_params(self):
        """The same block (same parameter set) must process both a short
        and a long sequence — this is the core resolution-invariance
        property of spectral mixing."""
        block = SpectralTransformerBlock(d_model=16, num_modes=8, block_size=16)
        x_short = torch.randn(1, 128, 16)
        x_long = torch.randn(1, 512, 16)
        out_short = block(x_short)
        out_long = block(x_long)
        assert out_short.shape == x_short.shape
        assert out_long.shape == x_long.shape
        assert torch.isfinite(out_short).all()
        assert torch.isfinite(out_long).all()

    @pytest.mark.slow
    def test_resolution_512_vs_2048(self):
        block = SpectralTransformerBlock(d_model=16, num_modes=8, block_size=16)
        x_512 = torch.randn(1, 512, 16)
        x_2048 = torch.randn(1, 2048, 16)
        out_512 = block(x_512)
        out_2048 = block(x_2048)
        assert out_512.shape == (1, 512, 16)
        assert out_2048.shape == (1, 2048, 16)
        assert torch.isfinite(out_512).all()
        assert torch.isfinite(out_2048).all()

    def test_param_count_independent_of_seq_len(self):
        """Parameter count must NOT change when we instantiate the block
        and then run it on different sequence lengths."""
        block = SpectralTransformerBlock(d_model=16, num_modes=8, block_size=16)
        n_params_short = sum(p.numel() for p in block.parameters())

        _ = block(torch.randn(1, 128, 16))
        n_params_after = sum(p.numel() for p in block.parameters())
        assert n_params_short == n_params_after, (
            "param count must not change after forward on a sequence"
        )

        _ = block(torch.randn(1, 256, 16))
        n_params_final = sum(p.numel() for p in block.parameters())
        assert n_params_final == n_params_short

    def test_param_count_same_for_two_instantiations(self):
        """Two blocks with identical hyperparams have the same param count
        regardless of the seq_len they'll be used on."""
        block_a = SpectralTransformerBlock(d_model=16, num_modes=8, block_size=16)
        block_b = SpectralTransformerBlock(d_model=16, num_modes=8, block_size=16)
        n_a = sum(p.numel() for p in block_a.parameters())
        n_b = sum(p.numel() for p in block_b.parameters())
        assert n_a == n_b

    def test_output_stats_stable_across_resolutions(self):
        block = SpectralTransformerBlock(d_model=16, num_modes=8, block_size=16)
        stds = []
        for seq_len in [128, 256, 512]:
            x = torch.randn(1, seq_len, 16)
            out = block(x)
            stds.append(out.std().item())
        # std should not explode across resolutions
        assert max(stds) / (min(stds) + 1e-8) < 10.0


# ---------------------------------------------------------------------------
# Slow integration tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestTransformerSlow:
    def test_large_channels(self):
        block = SpectralTransformerBlock(d_model=128, num_modes=32, block_size=32)
        x = torch.randn(2, 256, 128)
        out = block(x)
        assert out.shape == x.shape
        out.sum().backward()
        assert torch.isfinite(out).all()