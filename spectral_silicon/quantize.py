"""Hardware-oriented quantization utilities for the Spectral Silicon chip.

This module provides post-training quantization (PTQ) tools that convert
floating-point spectral-transformer weights into int8 representations
suitable for the on-chip fixed-point datapath.  It also provides an
int8 FFT using integer arithmetic and pre-computed lookup tables, and
utilities for measuring quantization error.

The quantization scheme is **symmetric** (zero-point = 0) which keeps
the integer arithmetic in the FFT and spectral-multiply datapaths simple:
``float_value = int_value * scale``.

Functions
---------
- :func:`quantize_weight` — quantize a real-valued tensor to int8.
- :func:`quantize_complex_weights` — quantize complex weights (real & imag
  parts separately) to int8.
- :func:`dequantize` — convert int8 weights back to floats.
- :func:`fft_int8` — approximate FFT using int8 arithmetic + lookup tables.
- :func:`measure_quantization_error` — relative error metric.
- :func:`post_training_quantize` — extract and quantize all spectral
  weights from a trained spectral transformer.

Examples
--------
>>> import torch, numpy as np
>>> w = torch.randn(16, 16)
>>> q, scale = quantize_weight(w, bits=8)
>>> dq = dequantize(q, scale, 0)
>>> err = measure_quantization_error(w, dq)
>>> assert err < 0.02
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False
    torch = None  # type: ignore

__all__ = [
    "QuantizedTensor",
    "QuantizedComplexWeights",
    "QuantizationResult",
    "quantize_weight",
    "quantize_complex_weights",
    "dequantize",
    "fft_int8",
    "measure_quantization_error",
    "post_training_quantize",
]

# Type aliases ---------------------------------------------------------------
TensorLike = Union["torch.Tensor", np.ndarray]
"""Type accepted by quantization functions — either a torch tensor or
numpy array (torch is optional at runtime for quantize_weight)."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_numpy(t: TensorLike) -> np.ndarray:
    """Convert a torch tensor or numpy array to a contiguous numpy array."""
    if _HAS_TORCH and isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _to_torch(arr: np.ndarray) -> "torch.Tensor":
    """Convert a numpy array back to a torch tensor (if torch available)."""
    if not _HAS_TORCH:
        raise ImportError("torch is required for this operation")
    return torch.from_numpy(arr.copy())


def _int_max(bits: int) -> int:
    """Max positive signed integer for *bits* (symmetric: ``2^(b-1) - 1``)."""
    return (1 << (bits - 1)) - 1


def _int_min(bits: int) -> int:
    """Min negative signed integer for *bits*."""
    return -(1 << (bits - 1))


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class QuantizedTensor:
    """Container for a single quantized real tensor.

    Attributes
    ----------
    int_weights : np.ndarray
        The quantized integer values (dtype int8 or int16).
    scale : float
        Multiplicative scale: ``float_value = int_value * scale``.
    zero_point : int
        Always 0 for symmetric quantization (kept for API compatibility).
    bits : int
        Bit width (default 8).
    shape : tuple
        Original tensor shape.
    """

    int_weights: np.ndarray
    scale: float
    zero_point: int = 0
    bits: int = 8
    shape: Tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        self.shape = tuple(self.int_weights.shape)

    def dequantize(self) -> np.ndarray:
        """Return the dequantized float tensor."""
        return self.int_weights.astype(np.float32) * self.scale

    def dequantize_torch(self) -> "torch.Tensor":
        """Return the dequantized tensor as a torch tensor."""
        return _to_torch(self.dequantize())

    @property
    def overflow_count(self) -> int:
        """Number of values that saturated during quantization."""
        return int(np.sum(self.int_weights >= _int_max(self.bits)) +
                    np.sum(self.int_weights <= _int_min(self.bits)))


@dataclass
class QuantizedComplexWeights:
    """Container for quantized complex weights (real & imag separately).

    Attributes
    ----------
    real_int, imag_int : np.ndarray
        Int8 quantized real and imaginary parts.
    real_scale, imag_scale : float
        Per-part scale factors.
    bits : int
        Bit width.
    """

    real_int: np.ndarray
    imag_int: np.ndarray
    real_scale: float
    imag_scale: float
    bits: int = 8
    shape: Tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        self.shape = tuple(self.real_int.shape)

    def dequantize(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(real_float, imag_float)`` dequantized arrays."""
        return (
            self.real_int.astype(np.float32) * self.real_scale,
            self.imag_int.astype(np.float32) * self.imag_scale,
        )


@dataclass
class QuantizationResult:
    """Result of :func:`post_training_quantize`.

    Attributes
    ----------
    weights : dict
        Mapping ``layer_name -> QuantizedComplexWeights`` (or
        ``QuantizedTensor`` for real-valued parameters).
    total_params : int
        Total number of quantized parameters.
    avg_error : float
        Average relative quantization error across all layers.
    per_layer_error : dict
        Per-layer relative error.
    bits : int
        Quantization bit width.
    """

    weights: Dict[str, Union[QuantizedComplexWeights, QuantizedTensor]]
    total_params: int = 0
    avg_error: float = 0.0
    per_layer_error: Dict[str, float] = field(default_factory=dict)
    bits: int = 8


# ---------------------------------------------------------------------------
# Core quantization
# ---------------------------------------------------------------------------
def quantize_weight(
    weight: TensorLike,
    bits: int = 8,
) -> Tuple[np.ndarray, float]:
    """Quantize a real-valued weight tensor to symmetric int8 (or int16).

    Uses **symmetric** quantization (zero_point = 0):
    ``float_value = int_value * scale``.

    Parameters
    ----------
    weight : torch.Tensor or np.ndarray
        Input floating-point weights.
    bits : int, optional
        Quantization bit width (default 8, supports 4-16).

    Returns
    -------
    int_weights : np.ndarray
        Quantized integer array (dtype int8 for ≤8 bits, int16 otherwise).
    scale : float
        Scale factor: ``original ≈ int_weights * scale``.

    Examples
    --------
    >>> w = np.array([0.1, 0.5, -0.3, 2.0])
    >>> qi, sc = quantize_weight(w, bits=8)
    >>> np.allclose(qi * sc, w, atol=sc)  # within 1 LSB
    True
    """
    if bits < 4 or bits > 16:
        raise ValueError(f"bits must be in [4, 16], got {bits}")

    arr = _to_numpy(weight).astype(np.float32)
    if arr.size == 0:
        dtype = np.int8 if bits <= 8 else np.int16
        return arr.astype(dtype), 0.0

    max_abs = float(np.max(np.abs(arr)))
    if max_abs == 0.0:
        scale = 0.0
        int_w = np.zeros_like(arr, dtype=np.int8 if bits <= 8 else np.int16)
        return int_w, scale

    qmax = _int_max(bits)
    scale = max_abs / qmax  # symmetric: one scale for both signs
    # Quantize: round(x / scale) then clip
    int_w = np.round(arr / scale).astype(np.int32)
    int_w = np.clip(int_w, _int_min(bits), _int_max(bits))

    dtype = np.int8 if bits <= 8 else np.int16
    return int_w.astype(dtype), float(scale)


def quantize_complex_weights(
    real: TensorLike,
    imag: TensorLike,
    bits: int = 8,
) -> QuantizedComplexWeights:
    """Quantize complex weights by quantizing real and imaginary parts
    separately to symmetric int8.

    Parameters
    ----------
    real, imag : torch.Tensor or np.ndarray
        Real and imaginary parts of the complex weight tensor.
    bits : int, optional
        Quantization bit width (default 8).

    Returns
    -------
    QuantizedComplexWeights
        Container with int8 (or int16) real/imag parts and scale factors.

    Examples
    --------
    >>> r = np.array([0.5, -0.3])
    >>> i = np.array([0.2, 0.7])
    >>> qw = quantize_complex_weights(r, i, bits=8)
    >>> dq_r, dq_i = qw.dequantize()
    >>> np.allclose(dq_r, r, atol=qw.real_scale)
    True
    """
    real_int, real_scale = quantize_weight(real, bits=bits)
    imag_int, imag_scale = quantize_weight(imag, bits=bits)
    return QuantizedComplexWeights(
        real_int=real_int,
        imag_int=imag_int,
        real_scale=real_scale,
        imag_scale=imag_scale,
        bits=bits,
    )


def dequantize(
    int_weights: np.ndarray,
    scale: float,
    zero_point: int = 0,
) -> np.ndarray:
    """Dequantize integer weights back to floating-point.

    Parameters
    ----------
    int_weights : np.ndarray
        Quantized integer array.
    scale : float
        Scale factor from quantization.
    zero_point : int, optional
        Zero-point (0 for symmetric quantization).

    Returns
    -------
    np.ndarray
        Dequantized float32 array.

    Examples
    --------
    >>> qi = np.array([10, -20, 30], dtype=np.int8)
    >>> dequantize(qi, scale=0.01)
    array([ 0.1, -0.2,  0.3], dtype=float32)
    """
    return (int_weights.astype(np.float32) - float(zero_point)) * scale


# ---------------------------------------------------------------------------
# Int8 FFT with lookup tables
# ---------------------------------------------------------------------------
def _build_twiddle_lut(n: int, bits: int = 8) -> Tuple[np.ndarray, float]:
    """Build a twiddle-factor lookup table for an N-point FFT.

    The LUT stores ``cos(2πk/N)`` and ``sin(2πk/N)`` for ``k = 0..N//2-1``
    quantized to int8.

    Parameters
    ----------
    n : int
        FFT size (must be a power of two).
    bits : int, optional
        Bit width for the twiddle factors.

    Returns
    -------
    twiddle_int : np.ndarray
        Complex-valued int array of shape ``(n//2,)`` with dtype complex
        (real/imag stored as int8 pairs).
    twiddle_scale : float
        Scale factor for the twiddle factors.
    """
    if n & (n - 1) != 0:
        raise ValueError(f"FFT size must be a power of 2, got {n}")
    k = np.arange(n // 2)
    angles = -2.0 * np.pi * k / n
    cos_vals = np.cos(angles)
    sin_vals = np.sin(angles)
    # Quantize cos and sin separately
    cos_int, cos_scale = quantize_weight(cos_vals, bits=bits)
    sin_int, sin_scale = quantize_weight(sin_vals, bits=bits)
    # Use the larger scale for both to keep things simple
    twiddle_scale = max(cos_scale, sin_scale)
    # Re-quantize with the common scale
    qmax = _int_max(bits)
    cos_q = np.clip(np.round(cos_vals / twiddle_scale), _int_min(bits), qmax).astype(np.int32)
    sin_q = np.clip(np.round(sin_vals / twiddle_scale), _int_min(bits), qmax).astype(np.int32)
    twiddle_int = cos_q + 1j * sin_q  # store as complex int pair
    return twiddle_int, twiddle_scale


def fft_int8(
    x_int: np.ndarray,
    twiddle_int: Optional[np.ndarray] = None,
    twiddle_scale: float = 1.0,
    bits: int = 8,
    n: Optional[int] = None,
    normalize: bool = True,
) -> Tuple[np.ndarray, float]:
    """Approximate int8 FFT using integer arithmetic and lookup tables.

    Implements a decimation-in-time radix-2 FFT where all twiddle-factor
    multiplications are performed in integer arithmetic.  The algorithm
    uses a pre-computed twiddle LUT (quantized to int8) and accumulates
    products in int32 to avoid intermediate overflow.

    Parameters
    ----------
    x_int : np.ndarray
        Input signal as **complex** integers (``dtype=complex`` with
        int real/imag parts, or a 2-column array ``[real, imag]``).
    twiddle_int : np.ndarray, optional
        Pre-computed twiddle LUT (complex int array of length N//2).
        If None, the LUT is built internally.
    twiddle_scale : float, optional
        Scale factor for the twiddle factors.
    bits : int, optional
        Bit width (default 8).
    n : int, optional
        FFT size.  Inferred from *x_int* if not given.
    normalize : bool, optional
        If True (default), apply per-stage ``÷2`` scaling every other stage
        to prevent integer overflow — the total output is scaled by
        ``1/sqrt(N)``, a common compromise between accuracy and overflow
        safety used in hardware FFT implementations.  If False, no scaling
        is applied and the output matches ``numpy.fft.fft`` (unscaled);
        the caller is responsible for ensuring values do not overflow int32.

    Returns
    -------
    X_int : np.ndarray
        FFT output as complex int32 array.
    output_scale : float
        Scale factor (always 1.0).  The float result is obtained via
        ``X_float = X_int * output_scale * input_scale`` where
        ``input_scale`` is the scale of the quantized input.  With
        ``normalize=True`` the result approximates
        ``numpy.fft.fft(x) / divisor`` (divisor = ``2**(log2(N)//2)``);
        with ``normalize=False`` it approximates ``numpy.fft.fft(x)``.

    Notes
    -----
    With ``normalize=True`` (recommended for int8 to avoid overflow) the
    per-stage ``÷2`` scaling accumulates to a divisor of
    ``2 ** (log2(N) // 2)``, which equals ``sqrt(N)`` when ``N`` is a
    perfect square power of two (4, 16, 64, ...) and is within a
    ``sqrt(2)`` factor otherwise.  The float result is obtained via
    ``X_float = X_int * output_scale * input_scale``.

    With ``normalize=False`` the output approximates
    ``numpy.fft.fft(x)`` directly but may overflow for large ``N``.

    Examples
    --------
    >>> x = np.array([1.0, 0.0, 0.0, 0.0])
    >>> x_int, x_scale = quantize_weight(x, bits=8)
    >>> X_int, scale = fft_int8(x_int.astype(complex), bits=8, n=4)
    >>> X_float = X_int * scale * x_scale * 4  # un-normalize
    >>> np.allclose(np.abs(X_float), 1.0, atol=0.2)
    True
    """
    # --- parse input ---
    if n is None:
        n = len(x_int)
    if n & (n - 1) != 0:
        raise ValueError(f"FFT size must be a power of 2, got {n}")

    # Accept real or complex integer input
    x_arr = np.asarray(x_int)
    if np.iscomplexobj(x_arr):
        re = x_arr.real.astype(np.int32)
        im = x_arr.imag.astype(np.int32)
    elif x_arr.ndim == 2 and x_arr.shape[1] == 2:
        re = x_arr[:, 0].astype(np.int32)
        im = x_arr[:, 1].astype(np.int32)
    else:
        re = x_arr.astype(np.int32)
        im = np.zeros_like(re, dtype=np.int32)

    # --- build twiddle LUT if not provided ---
    if twiddle_int is None:
        twiddle_int, twiddle_scale = _build_twiddle_lut(n, bits=bits)
    twiddle_int = np.asarray(twiddle_int)
    w_re = twiddle_int.real.astype(np.int32)
    w_im = twiddle_int.imag.astype(np.int32)

    # --- bit-reversal permutation ---
    indices = _bit_reverse_indices(n)
    re = re[indices]
    im = im[indices]

    # --- butterfly stages ---
    qmax = _int_max(bits)
    stage_scale = 1  # track how much we've scaled down
    stage_bit = 0    # which stage we're on (for alternate-stage scaling)
    stage = 1
    while stage < n:
        half = stage
        stage *= 2
        # twiddle stride
        w_stride = n // stage
        for j in range(0, n, stage):
            for k in range(half):
                idx_w = k * w_stride
                w_r = w_re[idx_w % (n // 2)]
                w_i = w_im[idx_w % (n // 2)]
                # indices
                i0 = j + k
                i1 = j + k + half
                # load
                a_re, a_im = re[i0], im[i0]
                b_re, b_im = re[i1], im[i1]
                # complex multiply w * b  (in int32, then >> 7 to renormalize)
                # (w_r + j w_i)(b_re + j b_im) = (w_r*b_re - w_i*b_im) + j(w_r*b_im + w_i*b_re)
                # Because w is quantized to int8 with scale twiddle_scale,
                # the product w*b is scaled by twiddle_scale * (b's scale).
                # We renormalize by dividing by qmax (since w is in [-qmax, qmax]).
                prod_re = (w_r * b_re - w_i * b_im) // qmax
                prod_im = (w_r * b_im + w_i * b_re) // qmax
                # butterfly
                re[i0] = a_re + prod_re
                im[i0] = a_im + prod_im
                re[i1] = a_re - prod_re
                im[i1] = a_im - prod_im
        # Per-stage scaling: right-shift by 1 every other stage to prevent
        # growth while preserving precision.  This distributes a 1/sqrt(N)
        # factor (common in hardware FFT implementations) rather than 1/N,
        # giving better accuracy for the same overflow protection.
        if normalize and (stage_bit & 1):
            re = _rns_div_pow2_arr(re, 1)
            im = _rns_div_pow2_arr(im, 1)
            stage_scale <<= 1
        stage_bit += 1

    # The twiddle division by qmax renormalises the twiddle scale so the
    # signal stays at the input's scale.  The per-stage ÷2 (when enabled)
    # is already baked into X_int, so output_scale is always 1.0.
    # The float result is: X_float = X_int * output_scale * input_scale.
    # With normalize=True, X_float ≈ numpy.fft.fft(x) / divisor
    # where divisor = 2 ** (log2(N) // 2) (≈ sqrt(N)).
    # With normalize=False, X_float ≈ numpy.fft.fft(x).
    output_scale = 1.0

    X_int = re.astype(np.int32) + 1j * im.astype(np.int32)
    return X_int, output_scale


def _bit_reverse_indices(n: int) -> np.ndarray:
    """Return the bit-reversal permutation indices for length *n*."""
    bits = int(math.log2(n))
    indices = np.arange(n)
    reversed_idx = np.zeros(n, dtype=np.int64)
    for i in range(bits):
        reversed_idx = (reversed_idx << 1) | ((indices >> i) & 1)
    return reversed_idx


def _rns_div_pow2_arr(arr: np.ndarray, shift: int) -> np.ndarray:
    """Divide an integer array by ``2 ** shift`` with round-to-nearest-even.

    Vectorised version of the scalar rounding used in the fixed-point
    simulator.  Works on signed integer arrays.
    """
    if shift == 0:
        return arr.copy()
    half = 1 << (shift - 1)
    mask = (1 << shift) - 1
    magnitude = np.abs(arr)
    quotient = magnitude >> shift
    remainder = magnitude & mask
    # round up if remainder > half, or tie-break to even
    round_up = (remainder > half) | ((remainder == half) & ((quotient & 1) == 1))
    quotient = np.where(round_up, quotient + 1, quotient)
    return np.where(arr < 0, -quotient, quotient)


# ---------------------------------------------------------------------------
# Error measurement
# ---------------------------------------------------------------------------
def measure_quantization_error(
    original: TensorLike,
    quantized: TensorLike,
) -> float:
    """Compute the relative quantization error between two tensors.

    Uses the L2-norm relative error:

    ``error = ||original - quantized||₂ / ||original||₂``

    If the original has near-zero norm, returns the absolute L2 error
    instead to avoid division by zero.

    Parameters
    ----------
    original : torch.Tensor or np.ndarray
        Original (unquantized) values.
    quantized : torch.Tensor or np.ndarray
        Quantized (dequantized) values.

    Returns
    -------
    float
        Relative error metric (0.0 = perfect, 1.0 = no correlation).

    Examples
    --------
    >>> orig = np.array([1.0, 2.0, 3.0, 4.0])
    >>> quant = np.array([1.01, 2.02, 2.97, 4.04])
    >>> err = measure_quantization_error(orig, quant)
    >>> err < 0.01
    True
    """
    o_raw = _to_numpy(original)
    q_raw = _to_numpy(quantized)
    # Handle complex inputs by comparing the full complex values
    if np.iscomplexobj(o_raw) or np.iscomplexobj(q_raw):
        o = o_raw.astype(np.complex128)
        q = q_raw.astype(np.complex128)
    else:
        o = o_raw.astype(np.float64)
        q = q_raw.astype(np.float64)
    diff = o - q
    norm_orig = float(np.linalg.norm(o))
    norm_diff = float(np.linalg.norm(diff))
    if norm_orig < 1e-12:
        return norm_diff
    return norm_diff / norm_orig


# ---------------------------------------------------------------------------
# Post-training quantization of a full spectral transformer
# ---------------------------------------------------------------------------
def post_training_quantize(
    model: "torch.nn.Module",
    bits: int = 8,
) -> QuantizationResult:
    """Extract all spectral weights from a trained spectral transformer
    and quantize each to int8.

    The function walks the model's parameters and identifies spectral
    weights — complex weight tensors used by Fourier Neural Operator
    (FNO) layers, AFNO layers, or FFTNet layers.  It looks for parameter
    names containing common spectral-weight keywords (``"weight"``,
    ``"spectral"``, ``"complex"``, ``"twiddle"``, ``"filter"``) and
    quantizes them.  Real-valued weights (biases, norms) are also
    quantized with :func:`quantize_weight`.

    Parameters
    ----------
    model : torch.nn.Module
        A trained spectral transformer model.
    bits : int, optional
        Quantization bit width (default 8).

    Returns
    -------
    QuantizationResult
        Container with all quantized weights, total parameter count,
        and per-layer quantization errors.

    Notes
    -----
    The model is **not modified** — this is a read-only extraction.
    Complex weights are expected to be stored either as:

    1. A complex ``torch.Tensor`` (``dtype=torch.complex64``), or
    2. Two separate real tensors with names like ``weight_real`` and
       ``weight_imag``.

    Examples
    --------
    >>> # model = train_spectral_transformer(...)
    >>> # result = post_training_quantize(model, bits=8)
    >>> # print(result.avg_error)
    """
    if not _HAS_TORCH:
        raise ImportError("torch is required for post_training_quantize")

    quantized: Dict[str, Union[QuantizedComplexWeights, QuantizedTensor]] = {}
    per_layer_error: Dict[str, float] = {}
    total_params = 0

    # Collect named parameters
    named_params = dict(model.named_parameters())

    # Group complex pairs (weight_real + weight_imag)
    consumed: set = set()

    for name, param in named_params.items():
        if name in consumed:
            continue

        param_np = param.detach().cpu().numpy()

        # Case 1: complex tensor
        if param.dtype == torch.complex64 or param.dtype == torch.complex128:
            real_np = param_np.real
            imag_np = param_np.imag
            qw = quantize_complex_weights(real_np, imag_np, bits=bits)
            quantized[name] = qw
            total_params += param_np.size
            # error
            dq_r, dq_i = qw.dequantize()
            dq = dq_r + 1j * dq_i
            err = measure_quantization_error(param_np, dq)
            per_layer_error[name] = err
            consumed.add(name)
            continue

        # Case 2: paired real/imag tensors
        if name.endswith("_real"):
            base = name[:-5]  # strip "_real"
            imag_name = base + "_imag"
            if imag_name in named_params:
                real_np = param_np
                imag_np = named_params[imag_name].detach().cpu().numpy()
                qw = quantize_complex_weights(real_np, imag_np, bits=bits)
                quantized[base] = qw
                total_params += real_np.size + imag_np.size
                dq_r, dq_i = qw.dequantize()
                # Compare complex
                orig_complex = real_np + 1j * imag_np
                dq_complex = dq_r + 1j * dq_i
                err = measure_quantization_error(orig_complex, dq_complex)
                per_layer_error[base] = err
                consumed.add(name)
                consumed.add(imag_name)
                continue

        # Case 3: real-valued weight
        int_w, scale = quantize_weight(param_np, bits=bits)
        qt = QuantizedTensor(
            int_weights=int_w,
            scale=scale,
            zero_point=0,
            bits=bits,
        )
        quantized[name] = qt
        total_params += param_np.size
        dq = qt.dequantize()
        err = measure_quantization_error(param_np, dq)
        per_layer_error[name] = err
        consumed.add(name)

    avg_error = (
        sum(per_layer_error.values()) / len(per_layer_error) if per_layer_error else 0.0
    )

    return QuantizationResult(
        weights=quantized,
        total_params=total_params,
        avg_error=avg_error,
        per_layer_error=per_layer_error,
        bits=bits,
    )