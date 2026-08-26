"""Cocotb testbench for butterfly2.v and butterfly4.v — Prompt P11/P12.

Tests the radix-2 and radix-4 FFT butterfly modules by driving complex
fixed-point inputs and comparing outputs against numpy reference.

Run with:
    make SIM=icarus TOPLEVEL= butterfly2
    make SIM=icarus TOPLEVEL= butterfly4

The DUT interface (assumed from the RTL spec):

butterfly2 (radix-2):
    inputs:  a_re, a_im, b_re, b_im, w_re, w_im  (signed, DATA_WIDTH bits)
    outputs: y0_re, y0_im, y1_re, y1_im          (signed, DATA_WIDTH bits)
    Computes: (a + W*b, a - W*b)

butterfly4 (radix-4):
    inputs:  x0_re, x0_im, x1_re, x1_im, x2_re, x2_im, x3_re, x3_im
             w0_re, w0_im, w1_re, w1_im, w2_re, w2_im
    outputs: y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im
    Computes a 4-point DFT with 3 twiddle multiplications.
"""

import random
import struct

import cocotb
from cocotb.clock import Clock
from cocotb.regression import TestFactory
from cocotb.triggers import ClockCycles, RisingEdge, Timer

import numpy as np


# ---------------------------------------------------------------------------
# Fixed-point helpers (Q8.8 — 16-bit total, 8 fractional bits)
# ---------------------------------------------------------------------------

DATA_WIDTH = 16
FRAC_BITS = 8
SCALE = 1 << FRAC_BITS


def to_fixed(val, width=DATA_WIDTH, frac=FRAC_BITS):
    """Convert a float to a signed fixed-point integer."""
    raw = int(round(val * (1 << frac)))
    max_val = (1 << (width - 1)) - 1
    min_val = -(1 << (width - 1))
    raw = max(min_val, min(max_val, raw))
    return raw


def from_fixed(raw, width=DATA_WIDTH, frac=FRAC_BITS):
    """Convert a signed fixed-point integer back to float."""
    if raw >= (1 << (width - 1)):
        raw -= (1 << width)
    return raw / (1 << frac)


def to_unsigned(raw, width=DATA_WIDTH):
    """Convert signed to unsigned representation for driving DUT."""
    if raw < 0:
        raw += (1 << width)
    return raw


# ---------------------------------------------------------------------------
# Clock
# ---------------------------------------------------------------------------

def start_clock(dut, period_ns=10):
    """Start a clock on the DUT's clk signal if it exists."""
    clk_sig = getattr(dut, "clk", None)
    if clk_sig is not None:
        cocotb.start_soon(Clock(clk_sig, period_ns, units="ns").start())


async def reset_dut(dut):
    """Assert reset if the DUT has a rst_n signal."""
    rst = getattr(dut, "rst_n", getattr(dut, "rst", None))
    if rst is not None:
        rst.value = 0
        await ClockCycles(dut.clk, 5) if hasattr(dut, "clk") else await Timer(50, "ns")
        rst.value = 1
        await ClockCycles(dut.clk, 2) if hasattr(dut, "clk") else await Timer(20, "ns")


# ---------------------------------------------------------------------------
# Radix-2 butterfly tests
# ---------------------------------------------------------------------------

async def drive_butterfly2(dut, a, b, w):
    """Drive butterfly2 inputs with complex values a, b, twiddle w."""
    # Set inputs
    if hasattr(dut, "a_re"):
        dut.a_re.value = to_unsigned(to_fixed(a.real))
        dut.a_im.value = to_unsigned(to_fixed(a.imag))
        dut.b_re.value = to_unsigned(to_fixed(b.real))
        dut.b_im.value = to_unsigned(to_fixed(b.imag))
        dut.w_re.value = to_unsigned(to_fixed(w.real))
        dut.w_im.value = to_unsigned(to_fixed(w.imag))
    elif hasattr(dut, "a"):
        # Packed format
        dut.a.value = to_unsigned(to_fixed(a.real))
        dut.b.value = to_unsigned(to_fixed(b.real))
        dut.w.value = to_unsigned(to_fixed(w.real))

    await Timer(10, "ns")


def read_butterfly2(dut):
    """Read butterfly2 outputs."""
    y0_re = from_fixed(dut.y0_re.value.signed_integer) if hasattr(dut, "y0_re") else 0.0
    y0_im = from_fixed(dut.y0_im.value.signed_integer) if hasattr(dut, "y0_im") else 0.0
    y1_re = from_fixed(dut.y1_re.value.signed_integer) if hasattr(dut, "y1_re") else 0.0
    y1_im = from_fixed(dut.y1_im.value.signed_integer) if hasattr(dut, "y1_im") else 0.0
    return complex(y0_re, y0_im), complex(y1_re, y1_im)


@cocotb.test()
async def test_butterfly2_known(dut):
    """Test butterfly2 with known values from Prompt 11.

    Input: a=(3+4j), b=(1+2j), W=(0.707+0.707j)
    Expected: (a+W*b, a-W*b)
    """
    start_clock(dut)
    await reset_dut(dut)

    a = complex(3.0, 4.0)
    b = complex(1.0, 2.0)
    w = complex(0.707, 0.707)

    await drive_butterfly2(dut, a, b, w)
    y0, y1 = read_butterfly2(dut)

    expected_upper = a + w * b
    expected_lower = a - w * b

    dut._log.info(f"y0={y0:.4f}, expected={expected_upper:.4f}")
    dut._log.info(f"y1={y1:.4f}, expected={expected_lower:.4f}")

    assert abs(y0 - expected_upper) < 0.1, f"upper: {y0} vs {expected_upper}"
    assert abs(y1 - expected_lower) < 0.1, f"lower: {y1} vs {expected_lower}"


@cocotb.test()
async def test_butterfly2_identity_twiddle(dut):
    """Test butterfly2 with W=1 (identity twiddle → simple add/subtract)."""
    start_clock(dut)
    await reset_dut(dut)

    a = complex(2.0, 0.0)
    b = complex(1.0, 0.0)
    w = complex(1.0, 0.0)

    await drive_butterfly2(dut, a, b, w)
    y0, y1 = read_butterfly2(dut)

    assert abs(y0 - complex(3.0, 0.0)) < 0.05, f"upper: {y0}"
    assert abs(y1 - complex(1.0, 0.0)) < 0.05, f"lower: {y1}"


@cocotb.test()
async def test_butterfly2_random(dut):
    """Test butterfly2 with 50 random complex inputs."""
    start_clock(dut)
    await reset_dut(dut)

    random.seed(42)
    np.random.seed(42)
    max_err = 0.0

    for i in range(50):
        a = complex(np.random.uniform(-3, 3), np.random.uniform(-3, 3))
        b = complex(np.random.uniform(-3, 3), np.random.uniform(-3, 3))
        w = complex(
            np.random.uniform(-1, 1),
            np.random.uniform(-1, 1),
        )

        await drive_butterfly2(dut, a, b, w)
        y0, y1 = read_butterfly2(dut)

        exp_upper = a + w * b
        exp_lower = a - w * b

        err_upper = abs(y0 - exp_upper)
        err_lower = abs(y1 - exp_lower)
        max_err = max(max_err, err_upper, err_lower)

        assert err_upper < 0.2, f"[{i}] upper err {err_upper:.4f}"
        assert err_lower < 0.2, f"[{i}] lower err {err_lower:.4f}"

    dut._log.info(f"50 random tests passed, max error: {max_err:.4f}")


@cocotb.test()
async def test_butterfly2_2point_dft(dut):
    """A single radix-2 butterfly with W=1 computes a 2-point DFT."""
    start_clock(dut)
    await reset_dut(dut)

    x = [complex(1.0, 0.0), complex(2.0, 0.0)]
    ref = np.fft.fft(x)

    await drive_butterfly2(dut, x[0], x[1], complex(1.0, 0.0))
    y0, y1 = read_butterfly2(dut)

    assert abs(y0 - ref[0]) < 0.1, f"X[0]: {y0} vs {ref[0]}"
    assert abs(y1 - ref[1]) < 0.1, f"X[1]: {y1} vs {ref[1]}"


# ---------------------------------------------------------------------------
# Radix-4 butterfly tests
# ---------------------------------------------------------------------------

async def drive_butterfly4(dut, x0, x1, x2, x3, w0, w1, w2):
    """Drive butterfly4 inputs."""
    if hasattr(dut, "x0_re"):
        for name, val in [
            ("x0", x0), ("x1", x1), ("x2", x2), ("x3", x3),
            ("w0", w0), ("w1", w1), ("w2", w2),
        ]:
            getattr(dut, f"{name}_re").value = to_unsigned(to_fixed(val.real))
            getattr(dut, f"{name}_im").value = to_unsigned(to_fixed(val.imag))
    await Timer(10, "ns")


def read_butterfly4(dut):
    """Read butterfly4 outputs."""
    result = []
    for i in range(4):
        re = from_fixed(getattr(dut, f"y{i}_re").value.signed_integer) if hasattr(dut, f"y{i}_re") else 0.0
        im = from_fixed(getattr(dut, f"y{i}_im").value.signed_integer) if hasattr(dut, f"y{i}_im") else 0.0
        result.append(complex(re, im))
    return result


@cocotb.test()
async def test_butterfly4_4point_dft(dut):
    """Test butterfly4 against a 4-point DFT (numpy.fft.fft).

    The radix-4 butterfly computes a 4-point DFT with twiddle factors
    W0=1, W1=exp(-j*2π/4)=-j, W2=exp(-j*4π/4)=-1.
    """
    start_clock(dut)
    await reset_dut(dut)

    x = [complex(1.0, 0.0), complex(2.0, 0.0), complex(3.0, 0.0), complex(4.0, 0.0)]
    ref = np.fft.fft(x)

    # Twiddle factors for 4-point DFT
    w0 = complex(1.0, 0.0)        # W^0
    w1 = complex(0.0, -1.0)       # W^1 = -j
    w2 = complex(-1.0, 0.0)       # W^2 = -1

    await drive_butterfly4(dut, x[0], x[1], x[2], x[3], w0, w1, w2)
    outputs = read_butterfly4(dut)

    for i in range(4):
        dut._log.info(f"y[{i}]={outputs[i]:.4f}, ref={ref[i]:.4f}")
        assert abs(outputs[i] - ref[i]) < 0.3, f"y[{i}]: {outputs[i]} vs {ref[i]}"


@cocotb.test()
async def test_butterfly4_random(dut):
    """Test butterfly4 with random inputs against numpy 4-point DFT."""
    start_clock(dut)
    await reset_dut(dut)

    random.seed(99)
    np.random.seed(99)
    max_err = 0.0

    for trial in range(20):
        x = [complex(np.random.uniform(-2, 2), np.random.uniform(-2, 2)) for _ in range(4)]
        ref = np.fft.fft(x)

        w0 = complex(1.0, 0.0)
        w1 = complex(0.0, -1.0)
        w2 = complex(-1.0, 0.0)

        await drive_butterfly4(dut, x[0], x[1], x[2], x[3], w0, w1, w2)
        outputs = read_butterfly4(dut)

        for i in range(4):
            err = abs(outputs[i] - ref[i])
            max_err = max(max_err, err)
            assert err < 0.5, f"trial {trial} y[{i}] err {err:.4f}"

    dut._log.info(f"20 random butterfly4 tests passed, max error: {max_err:.4f}")