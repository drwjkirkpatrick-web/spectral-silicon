#!/usr/bin/env python3
"""
SpectralChipCompiler — PyTorch → spectral-chip binary blob compiler.

This module implements Prompt 29 of the Spectral Silicon project: it takes a
trained PyTorch spectral transformer model (built from AFNO / FFTNet layers),
extracts the learnable complex spectral weights from every spectral-mixing
sub-layer, quantizes the real and imaginary parts independently to int8, and
emits a compact binary blob suitable for loading onto the custom chip's
weight memory.

The binary format is intentionally simple and self-describing so that the
host driver (``spectral_driver.py``, Prompt 28) and the on-chip Wishbone
loader (Prompt 20) can parse it with minimal code.

Binary layout (all integers little-endian, struct format documented in
``HEADER_FORMAT``)::

    ┌──────────────────────────────────────────────────────────┐
    │ Header (64 bytes, padded)                                 │
    │   magic        : 4s   b"SPLR"                             │
    │   version      : I    1                                   │
    │   n_layers     : I    number of spectral layers            │
    │   d_model      : I    model hidden width                   │
    │   seq_len      : I    nominal sequence length              │
    │   n_modes      : I    number of retained Fourier modes     │
    │   block_size   : I    AFNO block-diagonal block size       │
    │   reserved     : 32s  zero padding (future use)           │
    ├──────────────────────────────────────────────────────────┤
    │ Per-layer blocks (repeated n_layers times)                │
    │   layer_type   : I    0 = AFNO, 1 = FFTNet                  │
    │   weight_len   : I    number of complex weights in layer   │
    │   scale_real   : f    float32 scale factor (max abs real)  │
    │   scale_imag   : f    float32 scale factor (max abs imag)  │
    │   threshold   : f    float32 soft-threshold value         │
    │   reserved     : I    0 (alignment)                       │
    │   weights_real : <n>i  int8 quantized real parts           │
    │   weights_imag : <n>i  int8 quantized imaginary parts      │
    └──────────────────────────────────────────────────────────┘

Quantization scheme
-------------------
For every spectral layer the complex weight tensor ``w ∈ C^{N}`` is split
into ``w.real`` and ``w.imag``.  Each half is linearly quantized to int8
using a symmetric scale factor ``s = max(|x|) / 127.0`` (with a tiny epsilon
to avoid division by zero).  The dequantization is ``x ≈ q * s``.

The chip's software simulation (``_simulate_layer``) dequantizes the weights
and re-applies the spectral mixing (FFT → multiply → soft-threshold → IFFT)
so that ``verify_compilation`` can compare the quantized path against the
full-precision PyTorch path.

Usage
-----
    from spectral_silicon.compiler import SpectralChipCompiler, compile_model

    blob = compile_model(model)
    SpectralChipCompiler.save_blob(blob, "chip_blob.bin")
    parsed = SpectralChipCompiler.load_blob("chip_blob.bin")
    SpectralChipCompiler.verify_compilation(model, blob)

    # CLI
    # python -m spectral_silicon.compiler --model weights.pt --output chip_blob.bin

Dependencies: ``struct``, ``numpy``, ``torch``.
"""

from __future__ import annotations

# Standard library
import argparse
import struct
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Third-party
import numpy as np

# torch is required for model introspection and the verification path.
# We import it lazily where possible so that pure blob round-trip helpers
# (``save_blob`` / ``load_blob``) remain usable in lightweight contexts.
try:
    import torch
except ImportError:  # pragma: no cover - torch is a declared dependency
    torch = None  # type: ignore[assignment]


# Re-export the quantizer / model symbols that this module is specified to
# import.  These imports are written against the sibling modules that are
# being built in parallel (``quantize.py`` and ``model.py``); if those have
# not landed yet the compiler degrades gracefully (see ``_lazy_import``).
from spectral_silicon.quantize import (  # noqa: E402
    quantize_complex_weights,
)

# ``TinySpectralLM`` is only needed for the CLI / default model construction;
# import it lazily to keep the module importable when model.py is absent.
_TinySpectralLM = None  # populated by ``_ensure_model_cls``


# ---------------------------------------------------------------------------
# Binary format constants
# ---------------------------------------------------------------------------

MAGIC = b"SPLR"
VERSION = 1

# Header struct (little-endian):
#   4s magic
#   I   version
#   I   n_layers
#   I   d_model
#   I   seq_len
# I   n_modes
# I   block_size
# 32s reserved
HEADER_FORMAT = "<4sIIIIII32s"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 4 + 6*4 + 32 = 60

# Per-layer header:
#   I layer_type   (0 = AFNO, 1 = FFTNet)
#   I weight_len   (number of complex weights)
#   f scale_real
#   f scale_imag
#   f threshold
#   I reserved
LAYER_HEADER_FORMAT = "<IIfffI"
LAYER_HEADER_SIZE = struct.calcsize(LAYER_HEADER_FORMAT)  # 6 × 4 = 24

LAYER_TYPE_AFNO = 0
LAYER_TYPE_FFTNET = 1


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class LayerSpec:
    """Metadata + quantized weights for one spectral layer."""

    layer_type: int
    weight_len: int
    scale_real: float
    scale_imag: float
    threshold: float
    weights_real: np.ndarray  # int8, shape (weight_len,)
    weights_imag: np.ndarray  # int8, shape (weight_len,)

    def dequantize(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (real, imag) float32 arrays reconstructed from int8."""
        real = self.weights_real.astype(np.float32) * self.scale_real
        imag = self.weights_imag.astype(np.float32) * self.scale_imag
        return real, imag

    def complex_weights(self) -> np.ndarray:
        """Return a complex64 array reconstructed from the quantized blobs."""
        real, imag = self.dequantize()
        return (real + 1j * imag).astype(np.complex64)


@dataclass
class CompiledBlob:
    """Parsed representation of a chip binary blob."""

    magic: bytes
    version: int
    n_layers: int
    d_model: int
    seq_len: int
    n_modes: int
    block_size: int
    layers: List[LayerSpec] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dict (useful for testing / serialization)."""
        return {
            "magic": self.magic,
            "version": self.version,
            "n_layers": self.n_layers,
            "d_model": self.d_model,
            "seq_len": self.seq_len,
            "n_modes": self.n_modes,
            "block_size": self.block_size,
            "layers": [
                {
                    "layer_type": l.layer_type,
                    "weight_len": l.weight_len,
                    "scale_real": l.scale_real,
                    "scale_imag": l.scale_imag,
                    "threshold": l.threshold,
                    "weights_real": l.weights_real.tolist(),
                    "weights_imag": l.weights_imag.tolist(),
                }
                for l in self.layers
            ],
        }


# ---------------------------------------------------------------------------
# Helpers for locating spectral weights inside a PyTorch model
# ---------------------------------------------------------------------------


def _to_numpy(t: Any) -> np.ndarray:
    """Best-effort conversion of a tensor/array/scalar to a numpy array."""
    if t is None:
        return np.array([], dtype=np.float32)
    if torch is not None and isinstance(t, torch.Tensor):
        t = t.detach().cpu()
        if t.is_complex():
            return t.numpy()
        return t.float().numpy()
    if isinstance(t, np.ndarray):
        return t
    return np.asarray(t, dtype=np.float32)


def _iter_spectral_layers(model: Any) -> List[Tuple[int, str, Any]]:
    """
    Walk ``model`` and yield (index, layer_type_name, module) tuples for every
    spectral-mixing sub-layer we know how to compile.

    Detection is name/attribute based so it works for both the canonical
    ``TinySpectralLM`` (``model.py``) and for ad-hoc models that follow the
    same naming conventions (``afno`` / ``fftnet`` / ``spectral``).
    """
    found: List[Tuple[int, str, Any]] = []
    idx = 0

    # Pattern 1: explicit ``spectral_layers`` / ``layers`` list on the model.
    for attr in ("spectral_layers", "spectral_blocks"):
        seq = getattr(model, attr, None)
        if seq is None:
            continue
        for mod in seq:
            name = _classify_layer(mod)
            if name is not None:
                found.append((idx, name, mod))
                idx += 1
        if found:
            return found

    # Pattern 2: walk all sub-modules, pick the spectral ones.
    if torch is not None and hasattr(model, "named_modules"):
        for _name, mod in model.named_modules():
            name = _classify_layer(mod)
            if name is not None:
                found.append((idx, name, mod))
                idx += 1
        if found:
            return found

    return found


def _classify_layer(mod: Any) -> Optional[str]:
    """Return ``"afno"`` / ``"fftnet"`` if *mod* looks like a spectral layer."""
    cls_name = type(mod).__name__.lower()

    # AFNO layers: have ``weight`` (directly or via _fno) and ``block_size``.
    if "afno" in cls_name:
        if hasattr(mod, "complex_weights") or hasattr(mod, "weight") or hasattr(mod, "_fno"):
            return "afno"
    # FFTNet layers: have ``spectral_filter`` or ``complex_weights`` or ``weight``.
    if "fftnet" in cls_name:
        if hasattr(mod, "complex_weights") or hasattr(mod, "spectral_filter") or hasattr(mod, "weight"):
            return "fftnet"
    # Generic fallback: anything that explicitly calls itself spectral.
    if "spectral" in cls_name and (hasattr(mod, "complex_weights") or hasattr(mod, "weight") or hasattr(mod, "_fno")):
        return "afno"  # default to AFNO semantics

    return None


def _extract_complex_weights(layer: Any, layer_type: str) -> np.ndarray:
    """
    Extract the complex weight tensor from a spectral layer as a flat
    complex128 numpy array.
    """
    w: Any = None
    if hasattr(layer, "complex_weights"):
        w = layer.complex_weights
    elif hasattr(layer, "weight") and getattr(layer, "is_complex", False):
        w = layer.weight
    elif hasattr(layer, "spectral_filter"):
        w = layer.spectral_filter
    elif hasattr(layer, "_fno") and hasattr(layer._fno, "weight"):
        w = layer._fno.weight
    if w is None:
        raise ValueError(
            f"Spectral layer {layer!r} has no discoverable complex weight tensor"
        )

    arr = _to_numpy(w)
    if np.iscomplexobj(arr):
        return arr.astype(np.complex128).ravel()

    # Some implementations store real/imag as a trailing 2-vector or as two
    # separate tensors.  Try a couple of common conventions.
    if arr.ndim >= 1 and arr.shape[-1] == 2:
        real = arr[..., 0]
        imag = arr[..., 1]
        return (real.astype(np.complex128) + 1j * imag.astype(np.complex128)).ravel()

    # Last resort: treat the array as the real part only.
    return arr.astype(np.complex128).ravel()


def _extract_threshold(layer: Any) -> float:
    """Best-effort extraction of the soft-threshold value from a layer."""
    for attr in ("threshold", "soft_threshold", "thresh"):
        v = getattr(layer, attr, None)
        if v is None:
            continue
        return float(_to_numpy(v))
    return 0.0


def _extract_model_meta(model: Any) -> Dict[str, int]:
    """
    Pull (d_model, seq_len, n_modes, block_size) from a model using a variety
    of naming conventions.
    """
    def _get(*names: str, default: int = 0) -> int:
        for n in names:
            v = getattr(model, n, None)
            if v is None:
                continue
            # Skip nn.Module / non-numeric attributes (e.g. ModuleDict keys)
            if hasattr(v, "parameters") or hasattr(v, "forward"):
                continue
            try:
                return int(v)
            except (TypeError, ValueError):
                # Could be a 1-D tensor / array.
                try:
                    return int(_to_numpy(v))
                except (TypeError, ValueError):
                    continue
        return default

    d_model = _get("d_model", "hidden_size", "dim", "embed_dim")
    seq_len = _get("seq_len", "max_seq_len", "block_len", "context_len")
    n_modes = _get("n_modes", "num_modes", "modes", "k", default=0)
    block_size = _get("block_size", "block", default=0)

    # Fall back to inferring from the first spectral layer if the model did
    # not carry the attributes.
    layers = _iter_spectral_layers(model)
    if d_model == 0 and layers:
        mod = layers[0][2]
        # Try the spectral layer's own attributes first
        d_model = int(getattr(mod, "channels", 0) or getattr(mod, "d_model", 0) or 0)
        if d_model == 0:
            w = _extract_complex_weights(mod, layers[0][1])
            raw = _to_numpy(getattr(mod, "complex_weights", np.array([])))
            if raw.size:
                d_model = int(raw.shape[-1]) if raw.ndim >= 1 else 0
    if n_modes == 0 and layers:
        mod = layers[0][2]
        n_modes = int(getattr(mod, "n_modes", 0) or getattr(mod, "num_modes", 0) or 0)
    if block_size == 0 and layers:
        mod = layers[0][2]
        block_size = int(getattr(mod, "block_size", 0) or 0)
    return {
        "d_model": d_model,
        "seq_len": seq_len,
        "n_modes": n_modes,
        "block_size": block_size,
    }


# ---------------------------------------------------------------------------
# Quantization (uses the imported ``quantize_complex_weights``)
# ---------------------------------------------------------------------------


def _quantize_layer_weights(
    complex_w: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Quantize a flat complex array to int8 real / int8 imag with symmetric
    per-component scaling.

    Returns (q_real, q_imag, scale_real, scale_imag).
    """
    real = np.real(complex_w).astype(np.float32)
    imag = np.imag(complex_w).astype(np.float32)

    # Prefer the shared ``quantize_complex_weights`` helper from quantize.py.
    try:
        result = quantize_complex_weights(complex_w)
        # Accept several possible return shapes from the helper:
        #  (q_real, q_imag, scale_real, scale_imag)
        #  dict(q_real=..., q_imag=..., scale_real=..., scale_imag=...)
        #  (q_real, q_imag) with scale == max/127
        if isinstance(result, dict):
            q_real = np.asarray(result["q_real"], dtype=np.int8)
            q_imag = np.asarray(result["q_imag"], dtype=np.int8)
            scale_real = float(result.get("scale_real", _default_scale(real)))
            scale_imag = float(result.get("scale_imag", _default_scale(imag)))
            return q_real, q_imag, scale_real, scale_imag
        if isinstance(result, (tuple, list)) and len(result) >= 4:
            q_real = np.asarray(result[0], dtype=np.int8)
            q_imag = np.asarray(result[1], dtype=np.int8)
            scale_real = float(result[2])
            scale_imag = float(result[3])
            return q_real, q_imag, scale_real, scale_imag
        if isinstance(result, (tuple, list)) and len(result) == 2:
            q_real = np.asarray(result[0], dtype=np.int8)
            q_imag = np.asarray(result[1], dtype=np.int8)
            return q_real, q_imag, _default_scale(real), _default_scale(imag)
    except Exception:
        # Fall through to the local implementation.
        pass

    # Local fallback implementation (matches the spec exactly).
    scale_real = _default_scale(real)
    scale_imag = _default_scale(imag)
    q_real = _quantize_symmetric(real, scale_real)
    q_imag = _quantize_symmetric(imag, scale_imag)
    return q_real, q_imag, scale_real, scale_imag


def _default_scale(x: np.ndarray) -> float:
    """Symmetric int8 scale factor ``max(|x|) / 127`` (with epsilon)."""
    m = float(np.max(np.abs(x))) if x.size else 0.0
    return m / 127.0 if m > 0 else 1.0


def _quantize_symmetric(x: np.ndarray, scale: float) -> np.ndarray:
    """Symmetric int8 quantization: ``q = round(x / scale)`` clipped to [-128,127]."""
    if scale == 0:
        return np.zeros_like(x, dtype=np.int8)
    q = np.round(x / scale)
    q = np.clip(q, -128, 127)
    return q.astype(np.int8)


# ---------------------------------------------------------------------------
# SpectralChipCompiler
# ---------------------------------------------------------------------------


class SpectralChipCompiler:
    """
    Compile a trained PyTorch spectral transformer into a binary blob for the
    Spectral Silicon chip.

    The compiler is deliberately tolerant: it works with any model that
    exposes spectral layers whose complex weight tensors can be discovered
    through the standard attribute conventions (``complex_weights``,
    ``spectral_filter``, or ``weight`` with ``is_complex=True``).
    """

    MAGIC = MAGIC
    VERSION = VERSION
    HEADER_FORMAT = HEADER_FORMAT
    LAYER_HEADER_FORMAT = LAYER_HEADER_FORMAT

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compile_model(self, model: Any) -> bytes:
        """
        Compile *model* into a binary blob.

        Parameters
        ----------
        model : torch.nn.Module
            A trained spectral transformer (e.g. ``TinySpectralLM``).

        Returns
        -------
        bytes
            The chip-ready binary blob.
        """
        meta = _extract_model_meta(model)
        spectral_layers = _iter_spectral_layers(model)
        if not spectral_layers:
            raise ValueError(
                "compile_model: no spectral layers found in model "
                "(expected AFNO/FFTNet layers with complex_weights)"
            )

        layer_blobs: List[bytes] = []
        for idx, layer_type, mod in spectral_layers:
            complex_w = _extract_complex_weights(mod, layer_type)
            threshold = _extract_threshold(mod)
            q_real, q_imag, s_real, s_imag = _quantize_layer_weights(complex_w)

            lt_code = (
                LAYER_TYPE_FFTNET if layer_type == "fftnet" else LAYER_TYPE_AFNO
            )
            layer_blobs.append(self._pack_layer(
                lt_code, q_real, q_imag, s_real, s_imag, threshold
            ))

        header = struct.pack(
            HEADER_FORMAT,
            MAGIC,
            VERSION,
            len(spectral_layers),
            meta["d_model"],
            meta["seq_len"],
            meta["n_modes"],
            meta["block_size"],
            b"\x00" * 32,
        )
        return header + b"".join(layer_blobs)

    @staticmethod
    def save_blob(blob: bytes, path: str) -> None:
        """
        Write a binary blob to *path*.

        Parameters
        ----------
        blob : bytes
            Binary blob returned by :meth:`compile_model` / :func:`compile_model`.
        path : str
            Destination file path.
        """
        with open(path, "wb") as f:
            f.write(blob)

    @staticmethod
    def load_blob(path_or_bytes) -> Dict[str, Any]:
        """
        Read and parse a blob from either a file path or raw bytes.

        Parameters
        ----------
        path_or_bytes : str or bytes
            Path to a ``chip_blob.bin`` file, or the raw blob bytes.

        Returns
        -------
        dict
            Parsed blob as a dictionary with keys: ``magic``, ``version``,
            ``n_layers``, ``d_model``, ``seq_len``, ``n_modes``,
            ``block_size``, and ``layers`` (each layer is itself a dict with
            quantized weights + scale factors).
        """
        if isinstance(path_or_bytes, (bytes, bytearray)):
            raw = bytes(path_or_bytes)
        else:
            with open(path_or_bytes, "rb") as f:
                raw = f.read()
        return SpectralChipCompiler._parse_blob(raw).to_dict()

    @staticmethod
    def parse_blob(raw: bytes) -> CompiledBlob:
        """Parse a raw ``bytes`` blob into a :class:`CompiledBlob`."""
        return SpectralChipCompiler._parse_blob(raw)

    @staticmethod
    def verify_compilation(
        model: Any,
        blob: bytes,
        test_input: Optional[Any] = None,
        tolerance: float = 0.05,
    ) -> bool:
        """
        Verify that the compiled blob reproduces the PyTorch model output.

        The blob is loaded into a lightweight software simulation of the
        chip's spectral-mix path (FFT → dequantized complex multiply →
        soft-threshold → IFFT) and the output is compared against the
        PyTorch model's spectral-layer outputs for a random test input.

        Parameters
        ----------
        model : torch.nn.Module
            The original trained model.
        blob : bytes
            The compiled binary blob.
        test_input : torch.Tensor, optional
            Explicit input tensor ``(batch, seq_len, d_model)``.  If
            ``None`` a small random tensor is generated from the blob metadata.
        tolerance : float
            Maximum acceptable mean relative error (default 5%).

        Returns
        -------
        bool
            ``True`` if the mean relative error across all spectral layers is
            below *tolerance*.
        """
        if torch is None:
            raise RuntimeError(
                "verify_compilation requires torch, but torch is not installed"
            )

        parsed = SpectralChipCompiler._parse_blob(blob)
        spectral_layers = _iter_spectral_layers(model)
        if len(spectral_layers) != parsed.n_layers:
            raise ValueError(
                f"verify_compilation: layer count mismatch "
                f"(model={len(spectral_layers)}, blob={parsed.n_layers})"
            )

        # Construct a test input if none was supplied.
        if test_input is None:
            test_input = torch.randn(
                1,
                max(parsed.seq_len, 1) or 16,
                max(parsed.d_model, 1) or 16,
            )

        model.eval()
        model_device = next(model.parameters()).device
        test_input = test_input.to(model_device)

        errors: List[float] = []
        with torch.no_grad():
            x = test_input
            for (idx, ltype, mod), spec in zip(spectral_layers, parsed.layers):
                # Reference output from the real PyTorch layer.
                ref_out = mod(x)
                if isinstance(ref_out, tuple):
                    ref_out = ref_out[0]
                # Simulated output from the quantized weights in the blob.
                sim_out = _simulate_layer(x, spec)
                # Compare in float (compare on CPU/numpy).
                ref_np = _to_numpy(ref_out)
                sim_np = _to_numpy(sim_out)
                if ref_np.shape != sim_np.shape:
                    # Reshape broadcast: trim or pad sim to match.
                    sim_np = _align_shape(sim_np, ref_np.shape)
                err = _relative_error(ref_np, sim_np)
                errors.append(err)
                # Feed the reference output forward (as the model would).
                x = ref_out if not isinstance(ref_out, tuple) else ref_out[0]

        mean_err = float(np.mean(errors)) if errors else 0.0
        max_err = float(np.max(errors)) if errors else 0.0
        return mean_err <= tolerance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pack_layer(
        layer_type: int,
        q_real: np.ndarray,
        q_imag: np.ndarray,
        scale_real: float,
        scale_imag: float,
        threshold: float,
    ) -> bytes:
        """Pack one layer's header + quantized weights into bytes."""
        q_real = np.asarray(q_real, dtype=np.int8)
        q_imag = np.asarray(q_imag, dtype=np.int8)
        if q_real.shape != q_imag.shape:
            raise ValueError(
                f"_pack_layer: real/imag shape mismatch "
                f"{q_real.shape} vs {q_imag.shape}"
            )
        n = q_real.size
        header = struct.pack(
            LAYER_HEADER_FORMAT,
            int(layer_type),
            int(n),
            float(scale_real),
            float(scale_imag),
            float(threshold),
            0,
        )
        return header + q_real.tobytes() + q_imag.tobytes()

    @staticmethod
    def _parse_blob(raw: bytes) -> CompiledBlob:
        """Parse raw bytes into a :class:`CompiledBlob`."""
        if len(raw) < HEADER_SIZE:
            raise ValueError(
                f"load_blob: blob too short ({len(raw)} < {HEADER_SIZE} bytes)"
            )
        (
            magic,
            version,
            n_layers,
            d_model,
            seq_len,
            n_modes,
            block_size,
            _reserved,
        ) = struct.unpack(HEADER_FORMAT, raw[:HEADER_SIZE])
        if magic != MAGIC:
            raise ValueError(
                f"load_blob: bad magic {magic!r} (expected {MAGIC!r})"
            )
        if version != VERSION:
            raise ValueError(
                f"load_blob: unsupported version {version} (expected {VERSION})"
            )

        offset = HEADER_SIZE
        layers: List[LayerSpec] = []
        for _ in range(n_layers):
            if offset + LAYER_HEADER_SIZE > len(raw):
                raise ValueError("load_blob: truncated layer header")
            (
                layer_type,
                weight_len,
                scale_real,
                scale_imag,
                threshold,
                _layer_reserved,
            ) = struct.unpack(
                LAYER_HEADER_FORMAT,
                raw[offset : offset + LAYER_HEADER_SIZE],
            )
            offset += LAYER_HEADER_SIZE
            w_bytes = weight_len * 1  # int8 = 1 byte each
            if offset + 2 * w_bytes > len(raw):
                raise ValueError("load_blob: truncated layer weights")
            q_real = np.frombuffer(
                raw[offset : offset + w_bytes], dtype=np.int8
            ).copy()
            offset += w_bytes
            q_imag = np.frombuffer(
                raw[offset : offset + w_bytes], dtype=np.int8
            ).copy()
            offset += w_bytes
            layers.append(
                LayerSpec(
                    layer_type=layer_type,
                    weight_len=weight_len,
                    scale_real=scale_real,
                    scale_imag=scale_imag,
                    threshold=threshold,
                    weights_real=q_real,
                    weights_imag=q_imag,
                )
            )

        return CompiledBlob(
            magic=magic,
            version=version,
            n_layers=n_layers,
            d_model=d_model,
            seq_len=seq_len,
            n_modes=n_modes,
            block_size=block_size,
            layers=layers,
        )


# ---------------------------------------------------------------------------
# Software simulation of the on-chip spectral-mix path (for verification)
# ---------------------------------------------------------------------------


def _simulate_layer(x: Any, spec: LayerSpec) -> np.ndarray:
    """
    Software simulation of one spectral-mix layer using the dequantized
    weights from a :class:`LayerSpec`.

    The simulation mirrors the on-chip dataflow:
        1. FFT along the sequence axis.
        2. Truncate to the first ``n_modes`` modes.
        3. Multiply by the dequantized complex weights.
        4. Soft-threshold: ``|z| < threshold → 0``.
        5. Zero-pad back to the original length.
        6. IFFT to return to the time domain.

    This is deliberately a NumPy-only implementation so it can run on the
    host without requiring a torch install.
    """
    x_np = _to_numpy(x)
    if x_np.ndim == 2:
        x_np = x_np[None, ...]
    if x_np.ndim != 3:
        raise ValueError(
            f"_simulate_layer: expected (batch, seq, d) input, got {x_np.shape}"
        )

    batch, seq_len, d_model = x_np.shape
    # FFT along the sequence axis.
    X = np.fft.fft(x_np, axis=1)

    w = spec.complex_weights()
    # Determine the weight layout: if weight_len == n_modes * d_model, the
    # weights are a flattened (n_modes, d_model) matrix.  Otherwise, they
    # are a per-mode 1-D array of length n_modes that broadcasts across d_model.
    if w.size >= d_model and w.size % d_model == 0:
        n_modes = w.size // d_model
        w2d = w.reshape(n_modes, d_model)
    elif w.size >= seq_len and w.size % seq_len == 0:
        # Alternative: (n_modes, seq_len) — unlikely but handle it.
        n_modes = w.size // seq_len
        w2d = w.reshape(n_modes, seq_len)[:, :d_model]
        w2d = np.tile(w2d, (1, int(np.ceil(d_model / w2d.shape[1]))))[:, :d_model]
    else:
        # 1-D per-mode weights.
        n_modes = w.size
        w2d = np.tile(w[:, None], (1, d_model))

    # Truncate to min(n_modes, seq_len).
    k = min(n_modes, seq_len)
    modes = X[:, :k, :]
    w2d = w2d[:k, :]  # (k, d_model)

    # Complex multiply.
    modes = modes * w2d

    # Soft-threshold: |z| < threshold → 0.
    if spec.threshold > 0:
        mag = np.abs(modes)
        mask = mag >= spec.threshold
        modes = np.where(mask, modes, 0.0 + 0.0j)

    # Re-insert the modified modes and IFFT.
    X[:, :k, :] = modes
    y = np.fft.ifft(X, axis=1).real

    # If the input had no batch dim, squeeze it back out.
    if x_np.shape[0] == 1 and (np.asarray(x).ndim == 2):
        y = y[0]
    return y.astype(np.float32)


def _broadcast_weights(w: np.ndarray, k: int, d_model: int) -> np.ndarray:
    """
    Make a 1-D complex weight array broadcastable against (batch, k, d_model).

    The chip stores weights block-diagonally; if the flattened length divides
    evenly into ``k`` we reshape to ``(k, blocks)`` and broadcast across
    ``d_model / blocks``.  Otherwise we tile the flat weights to length ``k``
    and then broadcast across ``d_model``.
    """
    if w.size == 0:
        return np.ones((k, d_model), dtype=np.complex64)

    # If the weights already have a 2-D shape (k, block_size), use it directly.
    if w.ndim == 2 and w.shape[0] == k:
        block = w.shape[1]
        if d_model % block == 0:
            reps = d_model // block
            return np.tile(w[:, :, None], (1, 1, reps)).reshape(k, d_model)
        return np.tile(w, (1, d_model // w.shape[1] + 1))[:, :d_model]

    # 1-D weights: tile to (k, d_model).
    flat = w.ravel()
    if flat.size >= k:
        w_k = flat[:k]
    else:
        reps = int(np.ceil(k / flat.size))
        w_k = np.tile(flat, reps)[:k]
    return np.tile(w_k[:, None], (1, d_model))


def _align_shape(sim: np.ndarray, target: Tuple[int, ...]) -> np.ndarray:
    """Trim or pad *sim* so that its shape matches *target*."""
    if sim.shape == target:
        return sim
    # Trim trailing dimensions.
    slices = tuple(slice(0, t) for t in target)
    # Broadcast-pad leading dims if needed.
    if sim.ndim < len(target):
        sim = sim[(None,) * (len(target) - sim.ndim)]
    return sim[slices]


def _relative_error(ref: np.ndarray, sim: np.ndarray) -> float:
    """Mean relative error between two arrays (L2 norm of diff / ref)."""
    denom = np.linalg.norm(ref) + 1e-8
    return float(np.linalg.norm(ref - sim) / denom)


# ---------------------------------------------------------------------------
# Module-level convenience functions (spec API)
# ---------------------------------------------------------------------------


def compile_model(model: Any) -> bytes:
    """
    Compile a trained PyTorch spectral model into a chip binary blob.

    Thin wrapper around :meth:`SpectralChipCompiler.compile_model`.
    """
    return SpectralChipCompiler().compile_model(model)


def save_blob(blob: bytes, path: str) -> None:
    """Thin wrapper around :meth:`SpectralChipCompiler.save_blob`."""
    SpectralChipCompiler.save_blob(blob, path)


def load_blob(path: str) -> Dict[str, Any]:
    """Thin wrapper around :meth:`SpectralChipCompiler.load_blob`."""
    return SpectralChipCompiler.load_blob(path)


def parse_blob(raw: bytes) -> "CompiledBlob":
    """Thin wrapper around :meth:`SpectralChipCompiler.parse_blob`."""
    return SpectralChipCompiler.parse_blob(raw)


def verify_compilation(
    model: Any,
    blob: bytes,
    test_input: Optional[Any] = None,
    tolerance: float = 0.05,
) -> bool:
    """Thin wrapper around :meth:`SpectralChipCompiler.verify_compilation`."""
    return SpectralChipCompiler.verify_compilation(
        model, blob, test_input=test_input, tolerance=tolerance
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _ensure_model_cls():
    """Lazily import ``TinySpectralLM`` from ``model.py``."""
    global _TinySpectralLM
    if _TinySpectralLM is not None:
        return _TinySpectralLM
    from spectral_silicon.model import TinySpectralLM  # noqa: WPS433
    _TinySpectralLM = TinySpectralLM
    return _TinySpectralLM


def _load_model(path: str, **kwargs: Any) -> Any:
    """
    Load a PyTorch model from *path*.

    Tries (in order):
      1. ``torch.load(path)`` as a state-dict, applied to a fresh
         ``TinySpectralLM``.
      2. ``torch.load(path)`` as a full model object.
    """
    if torch is None:
        raise RuntimeError("torch is required to load a model checkpoint")
    # Prefer weights_only=True (safe: only tensors/primitives are unpickled).
    # Fall back to weights_only=False for checkpoints that store a full model
    # object, but only after the safe path fails — this limits arbitrary-code
    # execution risk to user-supplied checkpoints that genuinely require it.
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Older torch without the kwarg.
        obj = torch.load(path, map_location="cpu")
    except Exception:
        # Checkpoint contains non-tensor objects (e.g. a full nn.Module).
        obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and any(
        k.endswith("complex_weights") for k in obj.keys()
    ):
        # Looks like a state-dict; build the default model and load it.
        ModelCls = _ensure_model_cls()
        model = ModelCls()
        try:
            model.load_state_dict(obj, strict=False)
        except Exception:
            # Fall back to loading whatever keys match.
            model.load_state_dict(
                {k: v for k, v in obj.items() if k in model.state_dict()},
                strict=False,
            )
        return model
    if isinstance(obj, dict):
        # Generic state-dict: try building the default model.
        ModelCls = _ensure_model_cls()
        model = ModelCls()
        model.load_state_dict(obj, strict=False)
        return model
    # Assume it's a full model.
    return obj


def _build_default_model() -> Any:
    """Construct a tiny default model (used by ``--demo``)."""
    ModelCls = _ensure_model_cls()
    try:
        return ModelCls()
    except TypeError:
        # ``TinySpectralLM.__init__`` may require kwargs; try sensible defaults.
        return ModelCls(
            vocab_size=128,
            d_model=64,
            seq_len=64,
            n_layers=2,
            n_modes=16,
            block_size=8,
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m spectral_silicon.compiler",
        description=(
            "Compile a trained PyTorch spectral transformer into a binary "
            "blob for the Spectral Silicon chip."
        ),
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to a PyTorch model checkpoint (.pt) to compile.",
    )
    p.add_argument(
        "--output",
        "--out",
        dest="output",
        type=str,
        default="chip_blob.bin",
        help="Output path for the binary blob (default: chip_blob.bin).",
    )
    p.add_argument(
        "--demo",
        action="store_true",
        help="Compile a tiny randomly-initialized demo model (no --model needed).",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="After compiling, verify the blob reproduces the model output.",
    )
    p.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Relative error tolerance for --verify (default: 0.05 = 5 percent).",
    )
    p.add_argument(
        "--print-summary",
        action="store_true",
        help="Print a human-readable summary of the compiled blob.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    args = _build_arg_parser().parse_args(argv)

    if args.demo:
        if torch is None:
            print("error: --demo requires torch", file=sys.stderr)
            return 1
        model = _build_default_model()
        print(f"[demo] built {type(model).__name__}")
    elif args.model:
        model = _load_model(args.model)
        print(f"[load] loaded model from {args.model}")
    else:
        _build_arg_parser().print_help()
        return 1

    compiler = SpectralChipCompiler()
    t0 = time.time()
    blob = compiler.compile_model(model)
    dt = time.time() - t0

    SpectralChipCompiler.save_blob(blob, args.output)
    print(
        f"[compile] wrote {len(blob)} bytes to {args.output} "
        f"({dt*1000:.1f} ms)"
    )

    if args.print_summary:
        parsed = SpectralChipCompiler.parse_blob(blob)
        _print_summary(parsed)

    if args.verify:
        ok = SpectralChipCompiler.verify_compilation(
            model, blob, tolerance=args.tolerance
        )
        if ok:
            print(f"[verify] PASS (within {args.tolerance:.0%} relative error)")
        else:
            print(
                f"[verify] FAIL (exceeds {args.tolerance:.0%} relative error)",
                file=sys.stderr,
            )
            return 2

    return 0


def _print_summary(parsed: CompiledBlob) -> None:
    """Print a human-readable summary of a parsed blob."""
    print("── Spectral Chip Blob ──")
    print(f"  magic      : {parsed.magic!r}")
    print(f"  version    : {parsed.version}")
    print(f"  n_layers   : {parsed.n_layers}")
    print(f"  d_model    : {parsed.d_model}")
    print(f"  seq_len    : {parsed.seq_len}")
    print(f"  n_modes    : {parsed.n_modes}")
    print(f"  block_size : {parsed.block_size}")
    print("  layers:")
    for i, l in enumerate(parsed.layers):
        ltype = "AFNO" if l.layer_type == LAYER_TYPE_AFNO else "FFTNet"
        print(
            f"    [{i}] {ltype:6s}  weights={l.weight_len:6d}  "
            f"scale=({l.scale_real:.4g}, {l.scale_imag:.4g})  "
            f"thresh={l.threshold:.4g}"
        )


if __name__ == "__main__":
    sys.exit(main())