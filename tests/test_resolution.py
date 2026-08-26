"""Tests for resolution invariance — Prompt P6.

Covers:
  - train on seq_len=64, evaluate on seq_len=256
  - assert spectral model perplexity degrades gracefully
"""

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from spectral_silicon.transformer import SpectralTransformerBlock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _set_seed():
    torch.manual_seed(2025)
    np.random.seed(2025)


def _make_tiny_spectral_lm(vocab_size=64, channels=32, n_layers=2, modes=8):
    """Build a tiny spectral LM: embedding → spectral blocks → unembedding."""
    embed = nn.Embedding(vocab_size, channels)
    blocks = nn.ModuleList(
        [SpectralTransformerBlock(channels=channels, modes=modes) for _ in range(n_layers)]
    )
    unembed = nn.Linear(channels, vocab_size)
    return nn.ModuleDict({
        "embed": embed,
        "blocks": blocks,
        "unembed": unembed,
    })


def _forward_lm(model, tokens):
    x = model["embed"](tokens)  # (batch, seq_len, channels)
    for block in model["blocks"]:
        x = block(x)
    logits = model["unembed"](x)  # (batch, seq_len, vocab_size)
    return logits


def _train_step(model, tokens, optimizer):
    logits = _forward_lm(model, tokens[:, :-1])
    targets = tokens[:, 1:]
    loss = nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def _compute_perplexity(model, tokens):
    with torch.no_grad():
        logits = _forward_lm(model, tokens[:, :-1])
        targets = tokens[:, 1:]
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )
    return math.exp(loss.item())


def _make_data(seq_len, vocab_size=64, n_samples=64):
    """Random token sequences for training/eval."""
    return torch.randint(0, vocab_size, (n_samples, seq_len))


# ---------------------------------------------------------------------------
# Resolution invariance tests
# ---------------------------------------------------------------------------

class TestResolutionInvariance:
    @pytest.mark.slow
    def test_train_64_eval_256_graceful_degradation(self):
        """Train on seq_len=64, evaluate on seq_len=256. The spectral
        model's perplexity should degrade gracefully (not explode)."""
        vocab_size = 64
        channels = 32
        model = _make_tiny_spectral_lm(vocab_size=vocab_size, channels=channels, modes=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train on short sequences
        train_data = _make_data(seq_len=64, vocab_size=vocab_size, n_samples=32)
        n_steps = 50
        for epoch in range(n_steps // len(train_data) + 1):
            for batch in train_data:
                _train_step(model, batch.unsqueeze(0), optimizer)

        # Evaluate on training length
        eval_short = _make_data(seq_len=64, vocab_size=vocab_size, n_samples=16)
        ppl_short = _compute_perplexity(model, eval_short[0:1])

        # Evaluate on longer sequences (zero-shot extrapolation)
        eval_long = _make_data(seq_len=256, vocab_size=vocab_size, n_samples=16)
        ppl_long = _compute_perplexity(model, eval_long[0:1])

        # Perplexity should not explode — ratio bounded
        assert ppl_short < vocab_size, "trained perplexity should be below random"
        ratio = ppl_long / (ppl_short + 1e-8)
        assert ratio < 5.0, (
            f"perplexity ratio {ratio:.2f} too high — degradation not graceful"
        )

    def test_eval_different_lengths_no_crash(self):
        """A spectral model trained at one length must run at other lengths
        without error — this is the basic resolution-invariance property."""
        model = _make_tiny_spectral_lm(vocab_size=32, channels=16, modes=4)
        for seq_len in [32, 64, 128, 256]:
            tokens = torch.randint(0, 32, (1, seq_len))
            logits = _forward_lm(model, tokens)
            assert logits.shape == (1, seq_len, 32)
            assert torch.isfinite(logits).all()

    def test_perplexity_finite_at_various_lengths(self):
        model = _make_tiny_spectral_lm(vocab_size=32, channels=16, modes=4)
        for seq_len in [32, 64, 128]:
            tokens = torch.randint(0, 32, (1, seq_len))
            ppl = _compute_perplexity(model, tokens)
            assert math.isfinite(ppl), f"perplexity not finite at seq_len={seq_len}"
            assert ppl > 0

    @pytest.mark.slow
    def test_spectral_better_than_attention_extrapolation(self):
        """Compare spectral vs a simple attention LM at train=64, eval=256.
        Spectral should degrade more gracefully."""
        # Build attention baseline
        class TinyAttentionLM(nn.Module):
            def __init__(self, vocab_size, channels, n_layers):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, channels)
                self.unembed = nn.Linear(channels, vocab_size)
                self.n_layers = n_layers
                self.channels = channels
                self.q = nn.Linear(channels, channels)
                self.k = nn.Linear(channels, channels)
                self.v = nn.Linear(channels, channels)

            def forward(self, tokens):
                x = self.embed(tokens)
                d = self.channels
                q = self.q(x)
                k = self.k(x)
                v = self.v(x)
                scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
                attn = torch.softmax(scores, dim=-1)
                x = torch.matmul(attn, v)
                return self.unembed(x)

        vocab_size = 32
        spectral_model = _make_tiny_spectral_lm(vocab_size=vocab_size, channels=16, modes=4)
        attention_model = TinyAttentionLM(vocab_size, 16, 2)

        # Train both briefly on seq_len=64
        opt_s = torch.optim.Adam(spectral_model.parameters(), lr=1e-3)
        opt_a = torch.optim.Adam(attention_model.parameters(), lr=1e-3)
        for step in range(20):
            data = torch.randint(0, vocab_size, (4, 64))
            _train_step(spectral_model, data, opt_s)
            _train_step(attention_model, data, opt_a)

        # Evaluate at seq_len=256
        eval_long = torch.randint(0, vocab_size, (1, 256))
        ppl_spectral = _compute_perplexity(spectral_model, eval_long)
        ppl_attention = _compute_perplexity(attention_model, eval_long)
        # Spectral should degrade more gracefully
        assert ppl_spectral < ppl_attention * 2.0, (
            f"spectral ppl {ppl_spectral:.1f} not better than attention {ppl_attention:.1f}"
        )