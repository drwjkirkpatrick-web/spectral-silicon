"""Fourier Neural Operator (FNO) layer.

Implements the core spectral-mixing primitive described in
Li et al., "Fourier Neural Operator for Parametric PDEs" (ICLR 2021,
arXiv:2010.08895).

The layer maps a 1-D token sequence ``x`` of shape
``(batch, seq_len, channels)`` through:

1. ``torch.fft.fft`` along the sequence dimension,
2. truncation to the first ``k`` Fourier modes,
3. multiplication by a learnable complex weight tensor,
4. ``torch.fft.ifft`` back to the spatial domain,
5. a residual connection ``y = x + spectral_mix(x)``.

With ``k = 0`` (no modes retained) the spectral branch is the identity
transform ``FFT -> IFFT`` so the layer collapses to a pure residual
``y = x``.

Resolution invariance
---------------------
The weight tensor is defined on *mode indices*, not on absolute sequence
length, so a layer trained at one ``seq_len`` can be evaluated at a
different ``seq_len`` with zero retraining — a core property of neural
operators exploited by the Spectral Silicon chip.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["FourierNeuralOperator"]


class FourierNeuralOperator(nn.Module):
    """A single Fourier Neural Operator spectral-mixing layer.

    Parameters
    ----------
    channels : int
        Number of feature channels ``d`` in the input ``(B, N, d)``.
    n_modes : int
        Number of low Fourier modes ``k`` to keep and transform. The
        remaining ``N - k`` modes are zeroed before the inverse transform.
        ``k = 0`` disables spectral mixing (identity / pure residual).
    block_size : int, default 1
        Block-diagonal width for the complex weight matrix. The default
        ``1`` produces a fully diagonal (per-channel) weight — the plain
        FNO. ``block_size = d`` gives a single full ``(d, d)`` block.
        ``channels`` must be divisible by ``block_size``.
    resolution_invariant : bool, default True
        When ``True`` (the neural-operator default) the weight is stored
        as a parameter of shape ``(k, channels, block_size)`` and is
        reused for any ``seq_len >= k``. Set to ``False`` only if you
        want to tie the weight to a fixed sequence length.

    Attributes
    ----------
    weight : nn.Parameter
        Learnable complex weight of shape ``(k, channels, block_size)``.
        The first axis indexes Fourier mode (``0..k-1``), the second the
        input channel, the third the output column within the channel
        block.
    """

    def __init__(
        self,
        channels: int,
        n_modes: int,
        block_size: int = 1,
        resolution_invariant: bool = True,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if n_modes < 0:
            raise ValueError(f"n_modes must be non-negative, got {n_modes}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if channels % block_size != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by "
                f"block_size ({block_size})"
            )

        self.channels = channels
        self.n_modes = n_modes
        self.block_size = block_size
        self.n_blocks = channels // block_size
        self.resolution_invariant = resolution_invariant

        # Learnable complex weight.  Glorot-style initialisation scaled for
        # complex tensors so the spectral branch starts near the identity.
        if n_modes > 0:
            bound = 1.0 / max(n_modes * block_size, 1)
            real = torch.empty(n_modes, channels, block_size).uniform_(-bound, bound)
            imag = torch.empty(n_modes, channels, block_size).uniform_(-bound, bound)
            self.weight = nn.Parameter(torch.complex(real, imag))
        else:
            # No learnable parameters when k=0 — register an empty buffer so
            # ``list(self.parameters())`` stays clean and downstream code
            # that touches ``self.weight`` short-circuits on shape (0,...).
            self.register_buffer(
                "weight",
                torch.zeros(0, channels, block_size, dtype=torch.complex64),
                persistent=False,
            )

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply FNO spectral mixing with a residual connection.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, seq_len, channels)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(batch, seq_len, channels)`` — the
            same shape as ``x``.
        """
        self._validate_input(x)

        # Fast path: zero modes → FFT→IFFT is identity, residual is just x.
        if self.n_modes == 0:
            return x

        residual = x

        # 1) FFT along the sequence (last spatial) dimension.
        x_ft = torch.fft.fft(x, dim=1, norm="ortho")

        # 2) Truncate to first k modes and apply block-diagonal weight.
        out_ft = torch.zeros_like(x_ft)
        k = min(self.n_modes, x.shape[1])
        if k > 0:
            modes = x_ft[:, :k, :]                      # (B, k, C)
            out_ft[:, :k, :] = self._apply_weight(modes)

        # 3) Inverse FFT back to the spatial domain.
        x_spatial = torch.fft.ifft(out_ft, dim=1, norm="ortho").real

        # 4) Residual connection.
        return residual + x_spatial

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _apply_weight(self, modes: torch.Tensor) -> torch.Tensor:
        """Block-diagonal complex multiply of ``modes`` by ``self.weight``.

        Parameters
        ----------
        modes : torch.Tensor
            Truncated spectral coefficients of shape ``(B, k, C)``.

        Returns
        -------
        torch.Tensor
            Transformed coefficients of the same shape ``(B, k, C)``.
        """
        B, k, C = modes.shape
        bs = self.block_size
        nb = self.n_blocks

        w = self.weight[:k]                            # (k, C, bs)
        # Reshape modes into blocks: (B, k, nb, bs)
        modes_blk = modes.reshape(B, k, nb, bs)
        # ``w`` already has channel axis split into (nb, bs) when reshaped:
        # (k, nb, bs, bs) -> permute for batched matmul.
        w_blk = w.reshape(k, nb, bs, bs)               # (k, nb, bs, bs)

        # Batched complex matmul: modes_blk @ w_blk over the last two dims.
        # Use einsum: B batch, M mode, N block, I input col, O output col.
        out = torch.einsum("bkni,mnoi->bkno", modes_blk, w_blk)
        return out.reshape(B, k, C)

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.dim() != 3:
            raise ValueError(
                f"expected input of shape (batch, seq_len, channels), "
                f"got shape {tuple(x.shape)}"
            )
        if x.shape[-1] != self.channels:
            raise ValueError(
                f"input has {x.shape[-1]} channels but layer was built for "
                f"{self.channels}"
            )

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, n_modes={self.n_modes}, "
            f"block_size={self.block_size}, "
            f"resolution_invariant={self.resolution_invariant}"
        )