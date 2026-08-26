"""Fixed-point arithmetic simulator for the Spectral Silicon chip.

This module provides :class:`FixedPoint` and :class:`FixedPointComplex`
classes that emulate the arithmetic used by the on-chip datapath.  The
simulator supports three standard Qm.n formats — Q4.4, Q8.8 and Q2.6 — and
implements:

* saturation arithmetic (clamping on overflow / underflow)
* round-to-nearest-even (banker's rounding) rounding
* addition and multiplication
* complex addition, complex multiplication, magnitude estimation
  (shift-and-add and a lightweight CORDIC-lite approximation)
* radix-2 FFT butterfly operation

Every arithmetic operation updates per-instance ``overflow`` / ``underflow``
flags so callers can measure error rates for a given format.

Examples
--------
>>> a = FixedPoint(1.5, fmt="Q4.4")
>>> b = FixedPoint(2.25, fmt="Q4.4")
>>> c = a + b
>>> c.to_float()
3.75
>>> c.to_int()
60

>>> z1 = FixedPointComplex(1.0, 2.0, fmt="Q8.8")
>>> z2 = FixedPointComplex(0.5, -0.5, fmt="Q8.8")
>>> z3 = z1 * z2
>>> abs(round(z3.real.to_float(), 3))
1.5
>>> abs(round(z3.imag.to_float(), 3))
0.5
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar, Optional, Tuple

__all__ = [
    "FixedPoint",
    "FixedPointComplex",
    "Q_FORMATS",
    "fft_butterfly",
]


# ---------------------------------------------------------------------------
# Q-format definitions
# ---------------------------------------------------------------------------
#: Pre-defined fixed-point formats supported by the simulator.
#: Each entry maps a format name to ``(int_bits, frac_bits)`` where the
#: total bit width is ``int_bits + frac_bits`` (the integer field includes
#: the sign bit, standard two's-complement Qm.n convention).
Q_FORMATS: dict[str, Tuple[int, int]] = {
    "Q4.4": (4, 4),
    "Q8.8": (8, 8),
    "Q2.6": (2, 6),
}


def _fmt_to_ints(fmt) -> Tuple[int, int]:
    """Resolve a format spec (name string or ``(int_bits, frac_bits)`` tuple)
    into a ``(int_bits, frac_bits)`` integer pair."""
    if isinstance(fmt, str):
        if fmt not in Q_FORMATS:
            raise ValueError(
                f"Unknown Q-format '{fmt}'. Supported: {list(Q_FORMATS)}"
            )
        return Q_FORMATS[fmt]
    if isinstance(fmt, (tuple, list)) and len(fmt) == 2:
        return int(fmt[0]), int(fmt[1])
    raise ValueError(f"Invalid format spec: {fmt!r}")


# ---------------------------------------------------------------------------
# FixedPoint
# ---------------------------------------------------------------------------
@dataclass
class FixedPoint:
    """A signed fixed-point number with saturation and round-to-nearest-even.

    Parameters
    ----------
    value : float or int
        Initial value.  If *raw_int* is True this is treated as the **raw
        integer** representation; otherwise it is a real-world (float)
        value that is immediately quantised.
    fmt : str or tuple
        Format name (``"Q4.4"``, ``"Q8.8"``, ``"Q2.6"``) or an explicit
        ``(int_bits, frac_bits)`` tuple.
    raw_int : bool, optional
        When True, *value* is interpreted as the pre-quantised integer
        representation (useful for reading hardware registers).

    Attributes
    ----------
    int_bits, frac_bits : int
        Resolved format parameters.
    overflow, underflow : bool
        Sticky flags set whenever a quantisation or arithmetic operation
        saturated at the positive or negative rail respectively.
    """

    # --- dataclass fields ---------------------------------------------------
    value: float = 0.0
    fmt: str | Tuple[int, int] = "Q8.8"
    raw_int: bool = False

    int_bits: int = field(init=False, default=0)
    frac_bits: int = field(init=False, default=0)
    _raw: int = field(init=False, default=0, repr=False)
    overflow: bool = field(init=False, default=False)
    underflow: bool = field(init=False, default=False)

    # --- construction --------------------------------------------------------
    def __post_init__(self) -> None:
        self.int_bits, self.frac_bits = _fmt_to_ints(self.fmt)
        if self.raw_int:
            self._raw = self._clamp_int(int(self.value))
        else:
            self._raw = self._float_to_raw(float(self.value))

    # --- bounds -------------------------------------------------------------
    @property
    def total_bits(self) -> int:
        """Total bit width (int_bits includes sign, two's-complement Qm.n)."""
        return self.int_bits + self.frac_bits

    @property
    def scale(self) -> int:
        """Integer scale factor ``2 ** frac_bits``."""
        return 1 << self.frac_bits

    @property
    def max_val(self) -> float:
        """Largest representable real-world value (positive rail)."""
        return float(self.max_int) / self.scale

    @property
    def min_val(self) -> float:
        """Smallest representable real-world value (negative rail)."""
        return float(self.min_int) / self.scale

    @property
    def max_int(self) -> int:
        """Largest representable raw integer (positive rail)."""
        return (1 << (self.int_bits + self.frac_bits - 1)) - 1

    @property
    def min_int(self) -> int:
        """Smallest representable raw integer (negative rail)."""
        return -(1 << (self.int_bits + self.frac_bits - 1))

    @property
    def raw(self) -> int:
        """The underlying two's-complement integer representation."""
        return self._raw

    @raw.setter
    def raw(self, v: int) -> None:
        self._raw = self._clamp_int(v)

    # --- conversion helpers -------------------------------------------------
    def _clamp_int(self, raw: int) -> int:
        """Clamp a raw integer into range, setting overflow/underflow flags."""
        if raw > self.max_int:
            self.overflow = True
            return self.max_int
        if raw < self.min_int:
            self.underflow = True
            return self.min_int
        return raw

    def _float_to_raw(self, f: float) -> int:
        """Convert a float to a raw integer with round-to-nearest-even."""
        scaled = f * self.scale
        rounded = _round_to_nearest_even(scaled)
        return self._clamp_int(rounded)

    @classmethod
    def from_float(cls, f: float, fmt: str | Tuple[int, int] = "Q8.8") -> "FixedPoint":
        """Create a :class:`FixedPoint` from a float value."""
        return cls(f, fmt=fmt)

    @classmethod
    def from_int(cls, raw: int, fmt: str | Tuple[int, int] = "Q8.8") -> "FixedPoint":
        """Create a :class:`FixedPoint` from a raw integer representation."""
        return cls(raw, fmt=fmt, raw_int=True)

    def to_float(self) -> float:
        """Return the real-world floating-point approximation."""
        return self._raw / self.scale

    def to_int(self) -> int:
        """Return the raw integer (two's-complement) representation."""
        return self._raw

    # --- arithmetic ----------------------------------------------------------
    def _align(self, other: "FixedPoint") -> "FixedPoint":
        """Return *other* converted to *self*'s format (result format = lhs)."""
        if other.int_bits == self.int_bits and other.frac_bits == self.frac_bits:
            return other
        # Convert via float — precision loss is inherent when narrowing.
        return FixedPoint(other.to_float(), fmt=(self.int_bits, self.frac_bits))

    def _new(self, raw: int) -> "FixedPoint":
        """Create a new FixedPoint in this format, applying clamping."""
        result = FixedPoint(0, fmt=(self.int_bits, self.frac_bits), raw_int=True)
        result._raw = self._clamp_int(raw)
        return result

    def __add__(self, other: "FixedPoint") -> "FixedPoint":
        other = self._align(other)
        result = self._new(self._raw + other._raw)
        # propagate flags
        result.overflow = self.overflow or other.overflow or result.overflow
        result.underflow = self.underflow or other.underflow or result.underflow
        return result

    add = __add__

    def __sub__(self, other: "FixedPoint") -> "FixedPoint":
        other = self._align(other)
        result = self._new(self._raw - other._raw)
        result.overflow = self.overflow or other.overflow or result.overflow
        result.underflow = self.underflow or other.underflow or result.underflow
        return result

    sub = __sub__

    def __mul__(self, other: "FixedPoint") -> "FixedPoint":
        other = self._align(other)
        # Full-precision product, then shift right by frac_bits with RNE.
        product = self._raw * other._raw
        shifted_int = _rns_div_pow2(product, self.frac_bits)
        result = self._new(shifted_int)
        result.overflow = self.overflow or other.overflow or result.overflow
        result.underflow = self.underflow or other.underflow or result.underflow
        return result

    multiply = __mul__

    def __neg__(self) -> "FixedPoint":
        result = self._new(-self._raw)
        result.overflow = self.overflow or result.overflow
        result.underflow = self.underflow or result.underflow
        return result

    # --- comparisons ---------------------------------------------------------
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FixedPoint):
            return NotImplemented
        other = self._align(other)
        return self._raw == other._raw

    def __lt__(self, other: "FixedPoint") -> bool:
        other = self._align(other)
        return self._raw < other._raw

    def __le__(self, other: "FixedPoint") -> bool:
        return self == other or self < other

    def __gt__(self, other: "FixedPoint") -> bool:
        return not (self <= other)

    def __ge__(self, other: "FixedPoint") -> bool:
        return not (self < other)

    def __hash__(self) -> int:
        return hash((self.int_bits, self.frac_bits, self._raw))

    # --- misc ----------------------------------------------------------------
    def __repr__(self) -> str:
        fmt_name = f"Q{self.int_bits}.{self.frac_bits}"
        return f"FixedPoint({self.to_float()}, fmt={fmt_name}, raw={self._raw})"

    def __float__(self) -> float:
        return self.to_float()

    def __int__(self) -> int:
        return self.to_int()

    def reset_flags(self) -> None:
        """Clear the overflow/underflow sticky flags."""
        self.overflow = False
        self.underflow = False

    def flags(self) -> Tuple[bool, bool]:
        """Return ``(overflow, underflow)`` sticky flags."""
        return self.overflow, self.underflow


# ---------------------------------------------------------------------------
# Rounding utilities (round-to-nearest-even / banker's rounding)
# ---------------------------------------------------------------------------
def _round_to_nearest_even(x: float) -> int:
    """Round *x* to nearest integer using round-half-to-even (banker's).

    Examples
    --------
    >>> _round_to_nearest_even(0.5)
    0
    >>> _round_to_nearest_even(1.5)
    2
    >>> _round_to_nearest_even(2.5)
    2
    >>> _round_to_nearest_even(3.5)
    4
    """
    return int(math.floor(x + 0.5)) if (x - math.floor(x)) != 0.5 else (
        int(math.floor(x)) if (int(math.floor(x)) % 2 == 0) else int(math.ceil(x))
    )


def _rns_div_pow2(value: int, shift: int) -> int:
    """Divide *value* by ``2 ** shift`` with round-to-nearest-even.

    This is the integer-arithmetic equivalent of rounding a fixed-point
    product back to the target precision.  It avoids intermediate floats
    entirely so it behaves identically on any platform.
    """
    if shift == 0:
        return value
    half = 1 << (shift - 1)
    sign = -1 if value < 0 else 1
    magnitude = abs(value)
    quotient = magnitude >> shift
    remainder = magnitude & ((1 << shift) - 1)
    if remainder > half:
        quotient += 1
    elif remainder == half:
        # tie → round to even
        if quotient & 1:
            quotient += 1
    return sign * quotient


# ---------------------------------------------------------------------------
# FixedPointComplex
# ---------------------------------------------------------------------------
@dataclass
class FixedPointComplex:
    """A complex fixed-point number with separate real / imaginary parts.

    Parameters
    ----------
    real, imag : float
        Real and imaginary components (real-world float values).  If
        *raw_int* is True these are treated as raw integer representations.
    fmt : str or tuple
        Q-format for both real and imaginary parts.
    """

    real: float = 0.0
    imag: float = 0.0
    fmt: str | Tuple[int, int] = "Q8.8"
    raw_int: bool = False

    _re: Optional[FixedPoint] = field(init=False, default=None, repr=False)
    _im: Optional[FixedPoint] = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self._re = FixedPoint(self.real, fmt=self.fmt, raw_int=self.raw_int)
        self._im = FixedPoint(self.imag, fmt=self.fmt, raw_int=self.raw_int)

    # --- accessors ----------------------------------------------------------
    @property
    def re(self) -> FixedPoint:
        """Real component as a :class:`FixedPoint`."""
        return self._re

    @property
    def im(self) -> FixedPoint:
        """Imaginary component as a :class:`FixedPoint`."""
        return self._im

    def to_complex(self) -> complex:
        """Return the floating-point complex approximation."""
        return complex(self._re.to_float(), self._im.to_float())

    @classmethod
    def from_complex(
        cls, z: complex, fmt: str | Tuple[int, int] = "Q8.8"
    ) -> "FixedPointComplex":
        """Create from a Python :class:`complex`."""
        return cls(z.real, z.imag, fmt=fmt)

    @classmethod
    def from_ints(
        cls,
        re_raw: int,
        im_raw: int,
        fmt: str | Tuple[int, int] = "Q8.8",
    ) -> "FixedPointComplex":
        """Create from raw integer representations of real / imag parts."""
        return cls(re_raw, im_raw, fmt=fmt, raw_int=True)

    # --- flags --------------------------------------------------------------
    @property
    def overflow(self) -> bool:
        return self._re.overflow or self._im.overflow

    @property
    def underflow(self) -> bool:
        return self._re.underflow or self._im.underflow

    def flags(self) -> Tuple[bool, bool]:
        """Return combined ``(overflow, underflow)`` flags for both parts."""
        return self.overflow, self.underflow

    def reset_flags(self) -> None:
        self._re.reset_flags()
        self._im.reset_flags()

    # --- arithmetic ---------------------------------------------------------
    def complex_add(self, other: "FixedPointComplex") -> "FixedPointComplex":
        """Complex addition: ``(a + b)`` in fixed point."""
        fmt = (self._re.int_bits, self._re.frac_bits)
        result = FixedPointComplex(0, 0, fmt=fmt, raw_int=True)
        result._re = self._re + other._re
        result._im = self._im + other._im
        return result

    def __add__(self, other: "FixedPointComplex") -> "FixedPointComplex":
        return self.complex_add(other)

    def __sub__(self, other: "FixedPointComplex") -> "FixedPointComplex":
        fmt = (self._re.int_bits, self._re.frac_bits)
        result = FixedPointComplex(0, 0, fmt=fmt, raw_int=True)
        result._re = self._re - other._re
        result._im = self._im - other._im
        return result

    def complex_multiply(self, other: "FixedPointComplex") -> "FixedPointComplex":
        """Complex multiply: ``(a + ja)(c + jd) = (ac - bd) + j(ad + bc)``.

        Uses 4 real fixed-point multiplications.
        """
        fmt = (self._re.int_bits, self._re.frac_bits)
        result = FixedPointComplex(0, 0, fmt=fmt, raw_int=True)
        ac = self._re * other._re
        bd = self._im * other._im
        ad = self._re * other._im
        bc = self._im * other._re
        result._re = ac - bd
        result._im = ad + bc
        # Sync the public float fields with the internal FixedPoint values
        result.real = float(result._re)
        result.imag = float(result._im)
        # propagate flags
        result._re.overflow = ac.overflow or bd.overflow or result._re.overflow
        result._re.underflow = ac.underflow or bd.underflow or result._re.underflow
        result._im.overflow = ad.overflow or bc.overflow or result._im.overflow
        result._im.underflow = ad.underflow or bc.underflow or result._im.underflow
        return result

    def __mul__(self, other: "FixedPointComplex") -> "FixedPointComplex":
        return self.complex_multiply(other)

    def __neg__(self) -> "FixedPointComplex":
        fmt = (self._re.int_bits, self._re.frac_bits)
        result = FixedPointComplex(0, 0, fmt=fmt, raw_int=True)
        result._re = -self._re
        result._im = -self._im
        return result

    # --- magnitude ----------------------------------------------------------
    def magnitude_shift(self) -> FixedPoint:
        """Approximate ``|z|`` via a shift-and-add approximation.

        ``|z| ≈ max(|re|, |im|) + min(|re|, |im|)/2``  — a one-multiply-free
        magnitude estimator accurate to ~8% worst case.  The result is in
        the same Q-format as the inputs.
        """
        re_abs = self._re if self._re._raw >= 0 else -self._re
        im_abs = self._im if self._im._raw >= 0 else -self._im
        if re_abs._raw >= im_abs._raw:
            big, small = re_abs, im_abs
        else:
            big, small = im_abs, re_abs
        # small/2  via right shift with RNE
        half_small_raw = _rns_div_pow2(small._raw, 1)
        half_small = FixedPoint(
            half_small_raw, fmt=(self._re.int_bits, self._re.frac_bits), raw_int=True
        )
        return big + half_small

    def magnitude_cordic(self, iterations: int = 8) -> FixedPoint:
        """Approximate ``|z|`` via a CORDIC-lite (magnitude mode) algorithm.

        CORDIC vectoring mode rotates ``(re, im)`` toward the real axis,
        accumulating a magnitude.  After *iterations* micro-rotations the
        real part approximates ``|z|``.

        Parameters
        ----------
        iterations : int
            Number of CORDIC micro-rotation steps.  More iterations give
            better accuracy; 8 is a good default for Q8.8.

        Returns
        -------
        FixedPoint
            Approximate magnitude in the same Q-format.
        """
        fmt = (self._re.int_bits, self._re.frac_bits)
        re = FixedPoint(self._re.to_float(), fmt=fmt)
        im = FixedPoint(self._im.to_float(), fmt=fmt)
        re.reset_flags()
        im.reset_flags()
        # Pre-computed 2^-i as floats for the tangent approximations.
        # tan(atan(2^-i)) = 2^-i, so each step uses a shift.
        for i in range(iterations):
            if im._raw >= 0:
                # rotate clockwise: re' = re + im*2^-i, im' = im - re*2^-i
                shift = _rns_div_pow2(im._raw, i)
                new_re = re._raw + _rns_div_pow2(im._raw, i)
                new_im = im._raw - _rns_div_pow2(re._raw, i)
            else:
                new_re = re._raw - _rns_div_pow2(im._raw, i)
                new_im = im._raw + _rns_div_pow2(re._raw, i)
            re = FixedPoint(new_re, fmt=fmt, raw_int=True)
            im = FixedPoint(new_im, fmt=fmt, raw_int=True)
        # CORDIC gain ≈ product of 1/sqrt(1+2^-2i) ≈ 0.6073 for many iterations.
        # For a compact implementation we use a fixed correction factor.
        # 0.6073 in Q-format:
        correction = FixedPoint(0.6072529350, fmt=fmt)
        result = re * correction
        result.overflow = re.overflow or result.overflow
        result.underflow = re.underflow or result.underflow
        return result

    magnitude = magnitude_shift  # default alias

    # --- misc ---------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"FixedPointComplex({self._re.to_float():.4f}"
            f"+{self._im.to_float():.4f}j, "
            f"fmt=Q{self._re.int_bits}.{self._re.frac_bits})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FixedPointComplex):
            return NotImplemented
        return self._re == other._re and self._im == other._im

    def __hash__(self) -> int:
        return hash((self._re, self._im))


# ---------------------------------------------------------------------------
# FFT butterfly
# ---------------------------------------------------------------------------
def fft_butterfly(
    a: FixedPointComplex,
    b: FixedPointComplex,
    w: FixedPointComplex,
) -> Tuple[FixedPointComplex, FixedPointComplex]:
    """Radix-2 decimation-in-time FFT butterfly.

    Computes::

        out_upper = a + w * b
        out_lower = a - w * b

    Parameters
    ----------
    a, b : FixedPointComplex
        Input complex samples.
    w : FixedPointComplex
        Twiddle factor (complex exponential ``e^{-j 2πk/N}``).

    Returns
    -------
    (FixedPointComplex, FixedPointComplex)
        Upper and lower butterfly outputs.
    """
    wb = w * b  # complex multiply
    upper = a + wb
    lower = a - wb
    # propagate flags
    upper._re.overflow = wb.overflow or upper._re.overflow
    upper._re.underflow = wb.underflow or upper._re.underflow
    upper._im.overflow = wb.overflow or upper._im.overflow
    upper._im.underflow = wb.underflow or upper._im.underflow
    lower._re.overflow = wb.overflow or lower._re.overflow
    lower._re.underflow = wb.underflow or lower._re.underflow
    lower._im.overflow = wb.overflow or lower._im.overflow
    lower._im.underflow = wb.underflow or lower._im.underflow
    return upper, lower


# ---------------------------------------------------------------------------
# Convenience: build a twiddle factor in fixed point
# ---------------------------------------------------------------------------
def make_twiddle(angle: float, fmt: str | Tuple[int, int] = "Q8.8") -> FixedPointComplex:
    """Create a twiddle factor ``e^{-j*angle}`` in fixed point.

    Parameters
    ----------
    angle : float
        Angle in radians (typically ``-2π * k / N``).
    fmt : str or tuple
        Q-format for both real and imaginary parts.

    Returns
    -------
    FixedPointComplex
        Twiddle factor with ``real = cos(angle)``, ``imag = sin(angle)``.
    """
    return FixedPointComplex(math.cos(angle), math.sin(angle), fmt=fmt)