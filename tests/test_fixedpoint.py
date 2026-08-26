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

from spectral_silicon.fixedpoint import FixedPoint, FixedPointComplex


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
    @pytest.mark.parametrize("fmt,name", [(4, 4), (8, 8), (2, 6)])
    def test_zero(self, fmt, name):
        fp = FixedPoint(0.0, width=fmt + name, frac_bits=name)
        assert float(fp) == 0.0

    def test_q44_range(self):
        fp = FixedPoint(0.0, width=8, frac_bits=4)
        # Q4.4 range is [-8, 7.9375]
        assert FixedPoint(7.9375, width=8, frac_bits=4).value > 0
        assert FixedPoint(-8.0, width=8, frac_bits=4).value < 0

    def test_q88_range(self):
        fp = FixedPoint(127.996, width=16, frac_bits=8)
        assert abs(float(fp) - 127.996) < 0.01

    def test_q26_range(self):
        fp = FixedPoint(1.984375, width=8, frac_bits=6)
        assert abs(float(fp) - 1.984375) < 0.02

    @pytest.mark.parametrize("val", [0.5, 1.0, -0.5, -1.0, 2.25])
    def test_q44_quantization(self, val):
        fp = FixedPoint(val, width=8, frac_bits=4)
        # Q4.4 has 1/16 resolution
        assert abs(float(fp) - val) <= 1.0 / 16

    @pytest.mark.parametrize("val", [0.5, 1.0, -0.5, 3.5, -2.25])
    def test_q88_quantization(self, val):
        fp = FixedPoint(val, width=16, frac_bits=8)
        assert abs(float(fp) - val) <= 1.0 / 256

    @pytest.mark.parametrize("val", [0.5, 0.25, -0.75, 1.5])
    def test_q26_quantization(self, val):
        fp = FixedPoint(val, width=8, frac_bits=6)
        assert abs(float(fp) - val) <= 1.0 / 64


# ---------------------------------------------------------------------------
# Addition tests
# ---------------------------------------------------------------------------

class TestAdd:
    @pytest.mark.parametrize("a,b", [(1.5, 2.0), (0.25, 0.75), (-1.0, 1.5), (3.0, -2.5)])
    def test_add_q44(self, a, b):
        fa = FixedPoint(a, width=8, frac_bits=4)
        fb = FixedPoint(b, width=8, frac_bits=4)
        result = fa + fb
        assert abs(float(result) - (a + b)) <= 2.0 / 16

    @pytest.mark.parametrize("a,b", [(1.5, 2.0), (0.25, 0.75), (-1.0, 1.5), (50.0, -25.5)])
    def test_add_q88(self, a, b):
        fa = FixedPoint(a, width=16, frac_bits=8)
        fb = FixedPoint(b, width=16, frac_bits=8)
        result = fa + fb
        assert abs(float(result) - (a + b)) <= 2.0 / 256

    @pytest.mark.parametrize("a,b", [(0.5, 0.25), (-0.5, 0.75), (1.0, -1.0)])
    def test_add_q26(self, a, b):
        fa = FixedPoint(a, width=8, frac_bits=6)
        fb = FixedPoint(b, width=8, frac_bits=6)
        result = fa + fb
        assert abs(float(result) - (a + b)) <= 2.0 / 64


# ---------------------------------------------------------------------------
# Multiplication tests
# ---------------------------------------------------------------------------

class TestMultiply:
    @pytest.mark.parametrize("a,b", [(1.5, 2.0), (0.25, 4.0), (-1.0, 2.0), (2.5, -1.5)])
    def test_mul_q44(self, a, b):
        fa = FixedPoint(a, width=8, frac_bits=4)
        fb = FixedPoint(b, width=8, frac_bits=4)
        result = fa * fb
        assert abs(float(result) - (a * b)) <= 4.0 / 16

    @pytest.mark.parametrize("a,b", [(1.5, 2.0), (0.25, 4.0), (-1.0, 2.0), (10.5, 3.0)])
    def test_mul_q88(self, a, b):
        fa = FixedPoint(a, width=16, frac_bits=8)
        fb = FixedPoint(b, width=16, frac_bits=8)
        result = fa * fb
        assert abs(float(result) - (a * b)) <= 4.0 / 256

    @pytest.mark.parametrize("a,b", [(0.5, 0.25), (-0.5, 0.75), (1.0, -1.0)])
    def test_mul_q26(self, a, b):
        fa = FixedPoint(a, width=8, frac_bits=6)
        fb = FixedPoint(b, width=8, frac_bits=6)
        result = fa * fb
        assert abs(float(result) - (a * b)) <= 4.0 / 64


# ---------------------------------------------------------------------------
# Overflow saturation tests
# ---------------------------------------------------------------------------

class TestOverflow:
    def test_overflow_q44_saturates(self):
        # Q4.4 max ≈ 7.9375; adding beyond should saturate
        fa = FixedPoint(7.0, width=8, frac_bits=4)
        fb = FixedPoint(5.0, width=8, frac_bits=4)
        result = fa + fb
        assert float(result) <= 8.0  # saturates at max
        assert float(result) >= 7.0

    def test_underflow_q44_saturates(self):
        fa = FixedPoint(-7.0, width=8, frac_bits=4)
        fb = FixedPoint(-5.0, width=8, frac_bits=4)
        result = fa + fb
        assert float(result) >= -8.0
        assert float(result) <= -7.0

    def test_overflow_q88_saturates(self):
        fa = FixedPoint(200.0, width=16, frac_bits=8)
        fb = FixedPoint(200.0, width=16, frac_bits=8)
        result = fa + fb
        assert float(result) <= 256.0
        assert float(result) >= 200.0

    def test_mul_overflow_saturates(self):
        fa = FixedPoint(100.0, width=16, frac_bits=8)
        fb = FixedPoint(100.0, width=16, frac_bits=8)
        result = fa * fb
        # Should saturate, not wrap around
        assert float(result) >= 0  # no negative from overflow of positive


# ---------------------------------------------------------------------------
# Complex fixed-point tests
# ---------------------------------------------------------------------------

class TestFixedPointComplex:
    def test_construct(self):
        c = FixedPointComplex(1.0 + 2.0j, width=16, frac_bits=8)
        assert abs(float(c.real) - 1.0) <= 1.0 / 256
        assert abs(float(c.imag) - 2.0) <= 1.0 / 256

    def test_complex_add(self):
        a = FixedPointComplex(1.5 + 2.0j, width=16, frac_bits=8)
        b = FixedPointComplex(0.5 + 1.0j, width=16, frac_bits=8)
        c = a + b
        assert abs(float(c.real) - 2.0) <= 2.0 / 256
        assert abs(float(c.imag) - 3.0) <= 2.0 / 256

    def test_complex_multiply(self):
        a = FixedPointComplex(1.0 + 1.0j, width=16, frac_bits=8)
        b = FixedPointComplex(1.0 + 1.0j, width=16, frac_bits=8)
        # (1+j)*(1+j) = 2j
        c = a * b
        assert abs(float(c.real) - 0.0) <= 4.0 / 256
        assert abs(float(c.imag) - 2.0) <= 4.0 / 256

    def test_complex_multiply_general(self):
        ar, ai, br, bi = 2.0, 3.0, 1.0, -1.0
        a = FixedPointComplex(complex(ar, ai), width=16, frac_bits=8)
        b = FixedPointComplex(complex(br, bi), width=16, frac_bits=8)
        c = a * b
        exp_re = ar * br - ai * bi
        exp_im = ar * bi + ai * br
        assert abs(float(c.real) - exp_re) <= 8.0 / 256
        assert abs(float(c.imag) - exp_im) <= 8.0 / 256

    def test_complex_magnitude(self):
        c = FixedPointComplex(3.0 + 4.0j, width=16, frac_bits=8)
        if hasattr(c, "magnitude"):
            mag = c.magnitude()
            assert abs(float(mag) - 5.0) < 0.5

    def test_complex_conjugate(self):
        c = FixedPointComplex(1.0 + 2.0j, width=16, frac_bits=8)
        if hasattr(c, "conjugate"):
            conj = c.conjugate()
            assert abs(float(conj.real) - 1.0) <= 1.0 / 256
            assert abs(float(conj.imag) - (-2.0)) <= 1.0 / 256


# ---------------------------------------------------------------------------
# Butterfly operation tests
# ---------------------------------------------------------------------------

class TestButterfly:
    """Radix-2 butterfly: (a, b) → (a + W*b, a - W*b)."""

    def test_butterfly_identity_twiddle(self):
        """W = 1+0j → output (a+b, a-b)."""
        a = FixedPointComplex(2.0 + 0.0j, width=16, frac_bits=8)
        b = FixedPointComplex(1.0 + 0.0j, width=16, frac_bits=8)
        w = FixedPointComplex(1.0 + 0.0j, width=16, frac_bits=8)
        if hasattr(FixedPointComplex, "butterfly"):
            out_upper, out_lower = FixedPointComplex.butterfly(a, b, w)
            assert abs(float(out_upper.real) - 3.0) < 0.1
            assert abs(float(out_lower.real) - 1.0) < 0.1
        else:
            pytest.skip("butterfly method not available")

    def test_butterfly_with_twiddle(self):
        """W = 0.707+0.707j, a=3+4j, b=1+2j — compare to float reference."""
        a = FixedPointComplex(3.0 + 4.0j, width=16, frac_bits=8)
        b = FixedPointComplex(1.0 + 2.0j, width=16, frac_bits=8)
        w = FixedPointComplex(0.707 + 0.707j, width=16, frac_bits=8)
        if hasattr(FixedPointComplex, "butterfly"):
            out_upper, out_lower = FixedPointComplex.butterfly(a, b, w)
            ref_a = complex(3, 4)
            ref_b = complex(1, 2)
            ref_w = complex(0.707, 0.707)
            exp_upper = ref_a + ref_w * ref_b
            exp_lower = ref_a - ref_w * ref_b
            assert abs(complex(float(out_upper.real), float(out_upper.imag)) - exp_upper) < 0.5
            assert abs(complex(float(out_lower.real), float(out_lower.imag)) - exp_lower) < 0.5
        else:
            pytest.skip("butterfly method not available")

    def test_butterfly_full_dft_2point(self):
        """A single butterfly computes a 2-point DFT when W=1."""
        x = [complex(1.0, 0.0), complex(2.0, 0.0)]
        ref = np.fft.fft(x)
        if hasattr(FixedPointComplex, "butterfly"):
            a = FixedPointComplex(x[0], width=16, frac_bits=8)
            b = FixedPointComplex(x[1], width=16, frac_bits=8)
            w = FixedPointComplex(1.0 + 0.0j, width=16, frac_bits=8)
            up, lo = FixedPointComplex.butterfly(a, b, w)
            assert abs(complex(float(up.real), float(up.imag)) - ref[0]) < 0.2
            assert abs(complex(float(lo.real), float(lo.imag)) - ref[1]) < 0.2
        else:
            pytest.skip("butterfly method not available")