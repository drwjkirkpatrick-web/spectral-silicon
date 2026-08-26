"""Adaptive Fourier Neural Operator (AFNO) layer.

Extends the FNO (:mod:`spectral_silicon.fno`) with the three mechanisms
introduced by Guibas et al., "Adaptive Fourier Neural Operators:
Efficient Token Mixers for Transformers" (2021, arXiv:2111.13587):

1. **Block-diagonal complex weights** — a ``block_size`` parameter
   trades expressivity for parameter/area efficiency by splitting the
   channel dimension into independent ``block_size × block_size``
   complex sub-matrices.
2. **Adaptive soft-thresholding** — every spectral coefficient whose
   magnitude falls below a learnable ``threshold`` is shrunk toward
   zero, sparsifying the frequency response and saving multiply
   bandwidth in hardware.
3. **Weight sharing across tokens** — the weight tensor is indexed by
   *mode* not by *position*, so the same parameters process any
   sequence length (resolution invariance).

With ``threshold = 0`` and ``block_size = channels`` the layer reduces
exactly to a standard FNO.
"""

from __future__ import annotations

import torch
from torch import nn

from .fno import FourierNeuralOperator

__all__ = ["AFNOLayer"]


class AFNOLayer(nn.Module):
    """Adaptive FNO spectral-mixing block with soft-thresholding.

    Parameters
    ----------
    channels : int
        Feature dimension ``d`` of the input ``(B, N, d)``.
    n_modes : int
        Number of low Fourier modes ``k`` to transform.
    block_size : int, default 16
        Width of each block-diagonal complex weight. Must divide
        ``channels``. ``block_size = channels`` reproduces the standard
        FNO (single full block); ``block_size = 1`` is fully diagonal.
    threshold : float, default 0.0
        Initial magnitude threshold for soft-thresholding. Coefficients
        with ``|w| <= threshold`` are driven to zero and those above are
        shrunk by ``threshold``. ``0.0`` disables shrinkage.
    learnable_threshold : bool, default True
        When ``True`` the threshold is a learnable scalar (clamped to
        ``>= 0`` during the forward pass so soft-thresholding stays
        well-defined); when ``False`` it is a fixed buffer.
    threshold_max : float, default 10.0
        Upper clamp applied to a learnable threshold so it cannot grow
        unboundedly and zero out the whole spectrum.

    Notes
    -----
    The forward pass implements, for each retained mode ``m`` and each
    channel block ``b``::

        z'  = W_b · z                     # block-diagonal complex multiply
        |z'|-th  if |z'| > th else 0
        z'' =  sign(|z'|) * z'            # soft-thresholded coefficient

    followed by IFFT and a residual connection.
    """

    def __init__(
        self,
        channels: int,
        n_modes: int,
        block_size: int = 16,
        threshold: float = 0.0,
        learnable_threshold: bool = True,
        threshold_max: float = 10.0,
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
        if threshold < 0.0:
            raise ValueError(f"threshold must be non-negative, got {threshold}")

        self.channels = channels
        self.n_modes = n_modes
        self.block_size = block_size
        self.n_blocks = channels // block_size
        self.learnable_threshold = learnable_threshold
        self.threshold_max = threshold_max

        # Reuse the FNO weight layout / init for the block-diagonal tensor.
        self._fno = FourierNeuralOperator(
            channels=channels,
            n_modes=n_modes,
            block_size=block_size,
            resolution_invariant=True,
        )

        # Soft-threshold parameter.
        if learnable_threshold:
            init_val = float(threshold)
            if init_val > threshold_max:
                init_val = threshold_max
            self.threshold = nn.Parameter(torch.tensor(init_val))
        else:
            self.register_buffer(
                "threshold", torch.tensor(float(threshold)), persistent=True
            )

    # ------------------------------------------------------------------ #
    # convenience: expose the weight like FNO does
    # ------------------------------------------------------------------ #
    @property
    def weight(self) -> torch.Tensor:
        """Learnable complex spectral weight ``(k, C, block_size)``."""
        return self._fno.weight

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply AFNO spectral mixing with soft-threshold + residual.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, seq_len, channels)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(batch, seq_len, channels)``.
        """
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

        # Fast path: zero modes → identity, return residual only.
        if self.n_modes == 0:
            return x

        residual = x

        # 1) FFT along the sequence dimension.
        x_ft = torch.fft.fft(x, dim=1, norm="ortho")

        # 2) Truncate to first k modes and apply block-diagonal weight.
        out_ft = torch.zeros_like(x_ft)
        k = min(self.n_modes, x.shape[1])
        if k > 0:
            modes = x_ft[:, :k, :]                      # (B, k, C)
            weighted = self._apply_weight(modes)
            out_ft[:, :k, :] = self._soft_threshold(weighted)

        # 3) Inverse FFT back to the spatial domain.
        x_spatial = torch.fft.ifft(out_ft, dim=1, norm="ortho").real

        # 4) Residual connection.
        return residual + x_spatial

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _apply_weight(self, modes: torch.Tensor) -> torch.Tensor:
        """Block-diagonal complex multiply (delegated to the FNO core)."""
        return self._fno._apply_weight(modes)

    def _soft_threshold(self, z: torch.Tensor) -> torch.Tensor:
        """Adaptive complex soft-thresholding (shrinkage).

        For a complex coefficient ``z`` and non-negative threshold ``t``::

            |z| <= t  →  0
            |z| >  t  →  z * (|z| - t) / |z|

        Equivalent to soft-thresholding applied to the magnitude while
        preserving the phase. With ``t = 0`` this is the identity, so the
        AFNO reduces to a plain block-diagonal FNO.
        """
        t = self.threshold
        if self.learnable_threshold:
            t = torch.clamp(t, min=0.0, max=self.threshold_max)

        if t == 0:
            return z

        mag = z.abs()
        # Avoid division by zero; entries with mag <= t are zeroed anyway.
        scale = torch.clamp(mag - t, min=0.0) / (mag + 1e-12)
        return z * scale

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, n_modes={self.n_modes}, "
            f"block_size={self.block_size}, "
            f"threshold={float(self.threshold):.4g}, "
            f"learnable_threshold={self.learnable_threshold}"
        )