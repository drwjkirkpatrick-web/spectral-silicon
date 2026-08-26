"""Spectral Transformer Block (Prompt 4).

Assembles a transformer block that replaces classical self-attention with a
*spectral mixing* layer — either an AFNO (Adaptive Fourier Neural Operator)
layer or an FFTNet adaptive spectral filter — followed by a SwiGLU
feed-forward network, two LayerNorms, and two residual connections.

The defining property of the spectral mixer is **resolution invariance**:
because the learnable parameters live in the frequency domain (complex weight
tensors and thresholds), the same block can ingest a sequence of *any* length
without changing its parameter count.  ``SpectralTransformerBlock`` therefore
processes ``seq_len=512`` and ``seq_len=2048`` identically — the only thing
that changes is the size of the FFT, not the number of trainable weights.

Layout::

        ┌──────────┐
   x ──►│ LayerNorm │──► spectral_mix ──┬──► (+) ──┬────────────────────┐
        └──────────┘                     │         │                    │
                                         └─ resid ─┘                    │
                                                                     ┌──▼────────┐
                                                                     │ LayerNorm │──► ffn ──┬──► (+) ──► out
                                                                     └──────────┘          │
                                                                                           └─ resid ─┘
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Sister modules built in parallel.  We import lazily-ish but at module level
# so that the names are available; the actual symbols are provided by the
# parallel workstreams on fno.py / afno.py / fftnet.py.
from spectral_silicon.afno import AFNOLayer
from spectral_silicon.fftnet import FFTNetLayer


__all__ = ["SpectralTransformerBlock", "SwiGLU"]


class SwiGLU(nn.Module):
    """SwiGLU feed-forward sub-layer.

    SwiGLU computes ``W2 (silu(W1 x) ⊙ W3 x)`` where ``W1`` and ``W3`` project
    from ``d_model`` to ``2*d_model`` (the gate and the value) and ``W2``
    projects back down to ``d_model``.  The SiLU gate provides a smooth,
    non-linear gating that empirically outperforms plain ReLU/GELU FFNs.

    Parameters
    ----------
    d_model : int
        Input/output width.
    expansion : int, default 2
        The hidden width is ``expansion * d_model`` (the gate/value tensors
        each have this width before the multiply).
    bias : bool, default False
        Whether to include bias terms in the linear projections.
    """

    def __init__(self, d_model: int, expansion: int = 2, bias: bool = False) -> None:
        super().__init__()
        hidden = expansion * d_model
        # Gate + value packed into a single Linear for efficiency: input -> 2*hidden.
        self.w_gate_value = nn.Linear(d_model, 2 * hidden, bias=bias)
        # Down-projection.
        self.w_down = nn.Linear(hidden, d_model, bias=bias)
        self.d_model = d_model
        self.hidden = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: ``W_down (silu(gate) ⊙ value)``."""
        gv = self.w_gate_value(x)               # (B, S, 2*hidden)
        gate, value = gv.chunk(2, dim=-1)        # each (B, S, hidden)
        gated = F.silu(gate) * value             # element-wise gate
        return self.w_down(gated)               # (B, S, d_model)


class SpectralTransformerBlock(nn.Module):
    """A transformer block whose attention is replaced by spectral mixing.

    The block follows the canonical post-norm-residual transformer pattern::

        x → norm → spectral_mix → +residual → norm → ffn → +residual

    The spectral mixing layer (AFNO or FFTNet) operates entirely in the
    frequency domain, which means its learnable parameters do *not* depend on
    the input sequence length.  This grants **resolution invariance**: a block
    trained at ``seq_len=512`` can be evaluated at ``seq_len=2048`` (or any
    other length) with zero parameter changes.

    Parameters
    ----------
    d_model : int
        Hidden width of the block.
    n_heads : int
        Number of attention heads.  **Unused** for the spectral path (spectral
        mixing has no notion of heads), but kept in the API so the block is a
        drop-in replacement for a standard ``nn.TransformerEncoderLayer`` in
        mixed architectures.
    seq_len : int
        Nominal sequence length the block is constructed for.  This only
        affects the *default* number of Fourier modes retained; the block
        remains usable at other lengths.
    mixer_type : {"afno", "fftnet"}
        Which spectral mixing layer to use.
    num_modes : int
        Number of low-frequency Fourier modes to keep (the rest are
        zeroed/truncated).  Drives the parameter count of the mixer.
    block_size : int
        Block-diagonal width for the AFNO weight tensor.  Ignored by FFTNet.
    threshold : float
        Adaptive soft-thresholding cut-off applied to spectral coefficients.
        Modes with magnitude below ``threshold`` are suppressed toward zero,
        inducing sparsity.
    ffn_expansion : int, default 2
        SwiGLU expansion factor.
    dropout : float, default 0.0
        Dropout applied to the output of each sub-layer (before the residual
        add).

    Attributes
    ----------
    spectral_mix : nn.Module
        The spectral mixing layer (AFNO or FFTNet).
    ffn : SwiGLU
        The feed-forward sub-layer.
    norm1, norm2 : nn.LayerNorm
        Pre-norm layer normalizations.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        seq_len: int = 512,
        mixer_type: str = "afno",
        num_modes: int = 13,
        block_size: int = 64,
        threshold: float = 0.02,
        ffn_expansion: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if mixer_type not in ("afno", "fftnet"):
            raise ValueError(
                f"mixer_type must be 'afno' or 'fftnet', got {mixer_type!r}"
            )

        self.d_model = d_model
        self.n_heads = n_heads          # kept for API compatibility, unused
        self.seq_len = seq_len
        self.mixer_type = mixer_type
        self.num_modes = num_modes
        self.block_size = block_size
        self.threshold = threshold

        # --- Spectral mixing layer ---------------------------------------
        if mixer_type == "afno":
            self.spectral_mix: nn.Module = AFNOLayer(
                channels=d_model,
                n_modes=num_modes,
                block_size=block_size,
                threshold=threshold,
            )
        else:
            self.spectral_mix = FFTNetLayer(
                channels=d_model,
                n_modes=num_modes,
            )

        # --- Feed-forward (SwiGLU) ---------------------------------------
        self.ffn = SwiGLU(
            d_model=d_model,
            expansion=ffn_expansion,
            bias=False,
        )

        # --- Norms + residuals -------------------------------------------
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input activations of shape ``(batch, seq_len, d_model)``.

        Returns
        -------
        torch.Tensor
            Output activations, same shape as input.

        Notes
        -----
        Because the spectral mixer operates in the frequency domain, the
        sequence dimension is handled by an FFT whose size is determined by
        ``x.shape[1]`` at run time — so the *same* parameters process
        ``seq_len=512`` and ``seq_len=2048`` without any change.  This is the
        resolution-invariance guarantee.
        """
        # Sub-layer 1: spectral mixing with residual.
        residual = x
        h = self.norm1(x)
        h = self.spectral_mix(h)
        h = self.dropout(h)
        x = residual + h

        # Sub-layer 2: SwiGLU FFN with residual.
        residual = x
        h = self.norm2(x)
        h = self.ffn(h)
        h = self.dropout(h)
        x = residual + h
        return x

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        """Return the number of trainable parameters in this block."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def resolution_invariant(self) -> bool:
        """Return ``True`` — by construction the block is resolution invariant.

        The learnable parameters live in the spectral domain and do not scale
        with sequence length; only the (non-parametric) FFT size changes.
        """
        return True