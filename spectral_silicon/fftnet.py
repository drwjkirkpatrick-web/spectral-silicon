"""FFTNet — input-dependent adaptive spectral filter layer.

Implements the token mixer from Fein-Ashley, "The FFT Strikes Back: An
Efficient Alternative to Self-Attention" (2025, arXiv:2502.18394).

Unlike the FNO/AFNO, where the spectral weight is a static learned
tensor, FFTNet computes the spectral filter *conditioned on a global
summary of the input sequence*: the mean over the token axis is passed
through a small MLP that outputs the complex filter coefficients for
the first ``k`` modes. The filtered spectrum then passes through a
modReLU activation before the inverse FFT.

Forward pass
------------
1. Compute a global context vector ``g = mean(x, dim=1)``  — shape
   ``(B, C)``.
2. Map ``g`` through a two-layer MLP to ``2·k`` real scalars, reshape
   into ``k`` complex filter weights ``w ∈ ℂ^k`` (shared across channels
   by default, or per-block when ``block_size > 1``).
3. FFT the input along the sequence axis, multiply the first ``k`` modes
   by the per-batch complex filter, apply modReLU, IFFT.
4. Add the residual connection.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["FFTNetLayer", "modReLU"]


def modReLU(z: torch.Tensor, b: torch.Tensor | float) -> torch.Tensor:
    """modReLU activation for complex tensors.

    For a complex input ``z`` and real bias ``b``::

        modReLU(z) = z * sign(|z| + b)

    The bias shifts the magnitude threshold at which the activation
    switches on/off. A positive ``b`` makes the activation sparser
    (more outputs zeroed); a negative ``b`` keeps small magnitudes
    alive.

    Parameters
    ----------
    z : torch.Tensor
        Complex tensor of any shape.
    b : torch.Tensor or float
        Real-valued bias, broadcastable to ``z``.

    Returns
    -------
    torch.Tensor
        Complex tensor of the same shape and dtype as ``z``.
    """
    mag = z.abs()
    # sign(0) → 0, which is the desired behaviour (a zero coefficient
    # stays zero). For magnitudes where |z| + b == 0 exactly, sign(0)=0
    # zeros the output; otherwise the phase is preserved.
    gate = torch.sign(mag + b)
    return z * gate


class FFTNetLayer(nn.Module):
    """FFTNet adaptive spectral-filter token mixer.

    Parameters
    ----------
    channels : int
        Feature dimension ``d`` of the input ``(B, N, d)``.
    n_modes : int
        Number of low Fourier modes ``k`` whose complex filter weights
        are predicted by the MLP.
    hidden_dim : int, default 128
        Width of the context-MLP hidden layer.
    block_size : int, default 1
        Channel grouping for the predicted filter. ``block_size = 1``
        gives one shared complex scalar per mode (per batch); larger
        values predict a per-block filter of length ``k`` for each
        ``block_size`` channel group. Must divide ``channels``.
    modrelu_bias_init : float, default 0.0
        Initial value of the learnable modReLU bias.
    mlp_dropout : float, default 0.0
        Dropout applied to the hidden layer of the context MLP.
    resolution_invariant : bool, default True
        When ``True`` the predicted filter is indexed by mode, not by
        position, so the layer works at any ``seq_len >= n_modes``.

    Attributes
    ----------
    context_mlp : nn.Sequential
        The MLP that maps the global context vector to ``2 * k *
        n_blocks`` real scalars (real/imaginary parts of the per-block,
        per-mode filter weights).
    modrelu_bias : nn.Parameter
        Learnable real bias for the modReLU activation.
    """

    def __init__(
        self,
        channels: int,
        n_modes: int,
        hidden_dim: int = 128,
        block_size: int = 1,
        modrelu_bias_init: float = 0.0,
        mlp_dropout: float = 0.0,
        resolution_invariant: bool = True,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if n_modes <= 0:
            raise ValueError(
                f"n_modes must be positive for FFTNet (got {n_modes}); "
                f"use FNO if you need the k=0 identity path."
            )
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if channels % block_size != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by "
                f"block_size ({block_size})"
            )
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        self.channels = channels
        self.n_modes = n_modes
        self.hidden_dim = hidden_dim
        self.block_size = block_size
        self.n_blocks = channels // block_size
        self.resolution_invariant = resolution_invariant

        # Output of the MLP: 2 * k * n_blocks real scalars per batch.
        self.filter_dim = 2 * n_modes * self.n_blocks
        self.context_mlp = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(hidden_dim, self.filter_dim),
        )

        # Learnable real bias for modReLU.
        self.modrelu_bias = nn.Parameter(torch.tensor(float(modrelu_bias_init)))

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply FFTNet adaptive spectral filtering + residual.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, seq_len, channels)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(batch, seq_len, channels)``.
        """
        self._validate_input(x)

        residual = x
        B, N, C = x.shape

        # 1) Global context: mean over the sequence axis → (B, C).
        context = x.mean(dim=1)

        # 2) Predict per-batch complex filter weights.
        raw = self.context_mlp(context)                # (B, 2*k*nb)
        filt = self._reshape_filter(raw, B)           # (B, k, nb)

        # 3) FFT along the sequence dimension.
        x_ft = torch.fft.fft(x, dim=1, norm="ortho")  # (B, N, C)

        # 4) Broadcast-multiply the first k modes by the predicted filter.
        out_ft = torch.zeros_like(x_ft)
        k = min(self.n_modes, N)
        if k > 0:
            modes = x_ft[:, :k, :]                     # (B, k, C)
            weighted = self._apply_filter(modes, filt, k)
            out_ft[:, :k, :] = weighted

        # 5) modReLU on the (complex) filtered spectrum.
        activated = modReLU(out_ft, self.modrelu_bias)

        # 6) IFFT back to the spatial domain.
        x_spatial = torch.fft.ifft(activated, dim=1, norm="ortho").real

        # 7) Residual connection.
        return residual + x_spatial

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _reshape_filter(self, raw: torch.Tensor, batch: int) -> torch.Tensor:
        """Turn the MLP's real output into a complex filter tensor.

        Parameters
        ----------
        raw : torch.Tensor
            MLP output of shape ``(batch, 2 * k * n_blocks)``.
        batch : int
            Batch size (redundant with ``raw.shape[0]``; kept for
            clarity).

        Returns
        -------
        torch.Tensor
            Complex filter of shape ``(batch, k, n_blocks)``.
        """
        k = self.n_modes
        nb = self.n_blocks
        reshaped = raw.reshape(batch, k, nb, 2)
        real, imag = reshaped[..., 0], reshaped[..., 1]
        return torch.complex(real, imag)

    def _apply_filter(
        self,
        modes: torch.Tensor,
        filt: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """Multiply the first ``k`` modes by the per-block complex filter.

        Parameters
        ----------
        modes : torch.Tensor
            Spectral coefficients of shape ``(B, k, C)``.
        filt : torch.Tensor
            Predicted complex filter of shape ``(B, k, n_blocks)``.
        k : int
            Number of active modes (may be < ``self.n_modes`` when
            ``seq_len < n_modes``).

        Returns
        -------
        torch.Tensor
            Filtered coefficients of shape ``(B, k, C)``.
        """
        B, _, C = modes.shape
        bs = self.block_size
        nb = self.n_blocks

        # Split channel axis into blocks: (B, k, nb, bs)
        modes_blk = modes.reshape(B, k, nb, bs)
        # filt: (B, k, nb) → broadcast over the last (bs) channel dim.
        weighted = modes_blk * filt.unsqueeze(-1)
        return weighted.reshape(B, k, C)

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
            f"hidden_dim={self.hidden_dim}, block_size={self.block_size}, "
            f"resolution_invariant={self.resolution_invariant}"
        )