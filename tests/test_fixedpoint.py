"""Tests for the FixedPoint and FixedPointComplex arithmetic library — P10.

Covers:
  - Q4.4, Q8.8, Q2.6 formats
  - add / multiply against float reference
  - complex multiply
  - overflow saturation
  - butterfly operation
"""

import numpy as np
import pytest

from spectral_silicon.fixedpoint import FixedPoint, FixedPointComplex, fft_butterfly


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def approx_float(fp, expected, atol_frac=0.05):
    """Compare a FixedPoint to a float within a fractional tolerance."""
    return abs(float(fp) - expected) <= atol_frac * (abs(expected) + 1e-9)


# ---------------------------------------------------------------------------
# Format instantiation / quantization tests
# ---------------------------------------------------------------------------

class TestFormats:
    @pytest.mark.parametrize("int_bits,frac_bits", [(4, 4), (8, 8), (2, 6)])
    def test_zero(self, int_bits, frac_bits):
        fp = FixedPoint(0.0, fmt=(int_bits, frac_bits))
        assert float(fp) == 0.0

    def test_q44_range(self):
        fp = FixedPoint(0.0, fmt=(4, 4))
        # Q4.4 range is [-8, 7.9375]
        assert FixedPoint(7.9375, fmt=(4, 4)).value > 0
        assert FixedPoint(-8.0, fmt=(4, 4)).value < 0

    def test_q88_range(self):
        fp = FixedPoint(127.996, fmt=(8, 8))
        assert abs(float(fp) - 127.996) < 0.01

    def test_q26_range(self):
        fp = FixedPoint(1.984375, fmt=(2, 6))
        assert abs(float(fp) - 1.984375) < 0.02

    @pytest.mark.parametrize("val", [0.5, 1.0, -0.5, -1.0, 2.25])
    def test_q44_quantization(self, val):
        fp = FixedPoint(val, fmt=(4, 4))
        # Q4.4 has 1/16 resolution
        assert abs(float(fp) - val) <= 1.0 / 16

    @pytest.mark.parametrize("val", [0.5, 1.0, -0.5, 3.5, -2.25])
    def test_q88_quantization(self, val):
        fp = FixedPoint(val, fmt=(8, 8))
        assert abs(float(fp) - val) <= 1.0 / 256

    @pytest.mark.parametrize("val", [0.5, 0.25, -0.75, 1.5])
    def test_q26_quantization(self, val):
        fp = FixedPoint(val, fmt=(2, 6))
        assert abs(float(fp) - val) <= 1.0 / 64


# ---------------------------------------------------------------------------
# Addition tests
# ---------------------------------------------------------------------------

class TestAdd:
    @pytest.mark.parametrize("a,b", [(1.5, 2.0), (0.25, 0.75), (-1.0, 1.5), (3.0, -2.5)])
    def test_add_q44(self, a, b):
        fa = FixedPoint(a, fmt=(4, 4))
        fb = FixedPoint(b, fmt=(4, 4))
        result = fa + fb
        assert abs(float(result) - (a + b)) <= 2.0 / 16

    @pytest.mark.parametrize("a,b", [(1.5, 2.0), (0.25, 0.75), (-1.0, 1.5), (50.0, -25.5)])
    def test_add_q88(self, a, b):
        fa = FixedPoint(a, fmt=(8, 8))
        fb = FixedPoint(b, fmt=(8, 8))
        result = fa + fb
        assert abs(float(result) - (a + b)) <= 2.0 / 256

    @pytest.mark.parametrize("a,b", [(0.5, 0.25), (-0.5, 0.75), (1.0, -1.0)])
    def test_add_q26(self, a, b):
        fa = FixedPoint(a, fmt=(2, 6))
        fb = FixedPoint(b, fmt=(2, 6))
        result = fa + fb
        assert abs(float(result) - (a + b)) <= 2.0 / 64


# ---------------------------------------------------------------------------
# Multiplication tests
# ---------------------------------------------------------------------------

class TestMultiply:
    @pytest.mark.parametrize("a,b", [(1.5, 2.0), (0.25, 4.0), (-1.0, 2.0), (2.5, -1.5)])
    def test_mul_q44(self, a, b):
        fa = FixedPoint(a, fmt=(4, 4))
        fb = FixedPoint(b, fmt=(4, 4))
        result = fa * fb
        assert abs(float(result) - (a * b)) <= 4.0 / 16

    @pytest.mark.parametrize("a,b", [(1.5, 2.0), (0.25, 4.0), (-1.0, 2.0), (10.5, 3.0)])
    def test_mul_q88(self, a, b):
        fa = FixedPoint(a, fmt=(8, 8))
        fb = FixedPoint(b, fmt=(8, 8))
        result = fa * fb
        assert abs(float(result) - (a * b)) <= 4.0 / 256

    @pytest.mark.parametrize("a,b", [(0.5, 0.25), (-0.5, 0.75), (1.0, -1.0)])
    def test_mul_q26(self, a, b):
        fa = FixedPoint(a, fmt=(2, 6))
        fb = FixedPoint(b, fmt=(2, 6))
        result = fa * fb
        assert abs(float(result) - (a * b)) <= 4.0 / 64


# ---------------------------------------------------------------------------
# Overflow saturation tests
# ---------------------------------------------------------------------------

class TestOverflow:
    def test_overflow_q44_saturates(self):
        # Q4.4 max ≈ 7.9375; adding beyond should saturate
        fa = FixedPoint(7.0, fmt=(4, 4))
        fb = FixedPoint(5.0, fmt=(4, 4))
        result = fa + fb
        assert float(result) <= 8.0  # saturates at max
        assert float(result) >= 7.0

    def test_underflow_q44_saturates(self):
        fa = FixedPoint(-7.0, fmt=(4, 4))
        fb = FixedPoint(-5.0, fmt=(4, 4))
        result = fa + fb
        assert float(result) >= -8.0
        assert float(result) <= -7.0

    def test_overflow_q88_saturates(self):
        # Q8.8 max ≈ 127.996; use representable values whose sum overflows
        fa = FixedPoint(100.0, fmt=(8, 8))
        fb = FixedPoint(100.0, fmt=(8, 8))
        result = fa + fb
        assert float(result) <= 128.0  # saturates at max
        assert float(result) >= 100.0

    def test_mul_overflow_saturates(self):
        fa = FixedPoint(100.0, fmt=(8, 8))
        fb = FixedPoint(100.0, fmt=(8, 8))
        result = fa * fb
        # Should saturate, not wrap around
        assert float(result) >= 0  # no negative from overflow of positive


# ---------------------------------------------------------------------------
# Complex fixed-point tests
# ---------------------------------------------------------------------------

class TestFixedPointComplex:
    def test_construct(self):
        c = FixedPointComplex(1.0, 2.0, fmt=(8, 8))
        assert abs(float(c.re) - 1.0) <= 1.0 / 256
        assert abs(float(c.im) - 2.0) <= 1.0 / 256

    def test_complex_add(self):
        a = FixedPointComplex(1.5, 2.0, fmt=(8, 8))
        b = FixedPointComplex(0.5, 1.0, fmt=(8, 8))
        c = a + b
        assert abs(float(c.re) - 2.0) <= 2.0 / 256
        assert abs(float(c.im) - 3.0) <= 2.0 / 256

    def test_complex_multiply(self):
        a = FixedPointComplex(1.0, 1.0, fmt=(8, 8))
        b = FixedPointComplex(1.0, 1.0, fmt=(8, 8))
        # (1+j)*(1+j) = 2j
        c = a * b
        assert abs(float(c.re) - 0.0) <= 4.0 / 256
        assert abs(float(c.im) - 2.0) <= 4.0 / 256

    def test_complex_multiply_general(self):
        ar, ai, br, bi = 2.0, 3.0, 1.0, -1.0
        a = FixedPointComplex(ar, ai, fmt=(8, 8))
        b = FixedPointComplex(br, bi, fmt=(8, 8))
        c = a * b
        exp_re = ar * br - ai * bi
        exp_im = ar * bi + ai * br
        assert abs(float(c.re) - exp_re) <= 8.0 / 256
        assert abs(float(c.im) - exp_im) <= 8.0 / 256

    def test_complex_magnitude(self):
        c = FixedPointComplex(3.0, 4.0, fmt=(8, 8))
        if hasattr(c, "magnitude"):
            mag = c.magnitude()
            assert abs(float(mag) - 5.0) <= 0.5
        elif hasattr(c, "mag"):
            mag = c.mag()
            assert abs(float(mag) - 5.0) <= 0.5

    def test_complex_conjugate(self):
        c = FixedPointComplex(1.0, 2.0, fmt=(8, 8))
        if hasattr(c, "conjugate"):
            conj = c.conjugate()
            assert abs(float(conj.re) - 1.0) <= 1.0 / 256
            assert abs(float(conj.im) - (-2.0)) <= 1.0 / 256


# ---------------------------------------------------------------------------
# Butterfly operation tests
# ---------------------------------------------------------------------------

class TestButterfly:
    """Radix-2 butterfly: (a, b) → (a + W*b, a - W*b)."""

    def test_butterfly_identity_twiddle(self):
        """W = 1+0j → output (a+b, a-b)."""
        a = FixedPointComplex(2.0, 0.0, fmt=(8, 8))
        b = FixedPointComplex(1.0, 0.0, fmt=(8, 8))
        w = FixedPointComplex(1.0, 0.0, fmt=(8, 8))
        out_upper, out_lower = fft_butterfly(a, b, w)
        assert abs(float(out_upper.re) - 3.0) < 0.1
        assert abs(float(out_lower.re) - 1.0) < 0.1

    def test_butterfly_with_twiddle(self):
        """W = 0.707+0.707j, a=3+4j, b=1+2j — compare to float reference."""
        a = FixedPointComplex(3.0, 4.0, fmt=(8, 8))
        b = FixedPointComplex(1.0, 2.0, fmt=(8, 8))
        w = FixedPointComplex(0.707, 0.707, fmt=(8, 8))
        out_upper, out_lower = fft_butterfly(a, b, w)
        ref_a = complex(3, 4)
        ref_b = complex(1, 2)
        ref_w = complex(0.707, 0.707)
        exp_upper = ref_a + ref_w * ref_b
        exp_lower = ref_a - ref_w * ref_b
        assert abs(complex(float(out_upper.re), float(out_upper.im)) - exp_upper) < 0.5
        assert abs(complex(float(out_lower.re), float(out_lower.im)) - exp_lower) < 0.5

    def test_butterfly_full_dft_2point(self):
        """A single butterfly computes a 2-point DFT when W=1."""
        x = [complex(1.0, 0.0), complex(2.0, 0.0)]
        ref = np.fft.fft(x)
        a = FixedPointComplex(x[0].real, x[0].imag, fmt=(8, 8))
        b = FixedPointComplex(x[1].real, x[1].imag, fmt=(8, 8))
        w = FixedPointComplex(1.0, 0.0, fmt=(8, 8))
        up, lo = fft_butterfly(a, b, w)
        assert abs(complex(float(up.re), float(up.im)) - ref[0]) < 0.2
        assert abs(complex(float(lo.re), float(lo.im)) - ref[1]) < 0.2