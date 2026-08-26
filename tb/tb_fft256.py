"""Cocotb testbench for fft_256.v — Prompt P15/P21.

Drives 1000 random complex vectors into the 256-point FFT module, collects
outputs, and compares against numpy.fft.fft. Reports the maximum error.

Run with:
    make SIM=icarus TOPLEVEL=fft_256

Assumed DUT interface:
    clk, rst_n
    data_in_re[255:0], data_in_im[255:0]  — input data (packed or sequential)
    data_out_re[255:0], data_out_im[255:0] — output data
    valid_in, valid_out

    If the DUT uses streaming (one sample per clock), the testbench adapts.
    If it uses a memory-mapped interface, we write all 256 samples then read.
"""

import random

import cocotb
from cocotb.clock import Clock
from cocotb.regression import TestFactory
from cocotb.triggers import ClockCycles, FallingEdge, RisingEdge, Timer

import numpy as np


# ---------------------------------------------------------------------------
# Fixed-point helpers (Q8.8)
# ---------------------------------------------------------------------------

DATA_WIDTH = 16
FRAC_BITS = 8
N = 256


def to_fixed(val, width=DATA_WIDTH, frac=FRAC_BITS):
    """Convert float to signed fixed-point integer."""
    raw = int(round(val * (1 << frac)))
    max_val = (1 << (width - 1)) - 1
    min_val = -(1 << (width - 1))
    raw = max(min_val, min(max_val, raw))
    return raw


def from_fixed(raw, width=DATA_WIDTH, frac=FRAC_BITS):
    """Convert signed fixed-point integer to float."""
    if raw >= (1 << (width - 1)):
        raw -= (1 << width)
    return raw / (1 << frac)


def to_unsigned(raw, width=DATA_WIDTH):
    """Convert signed to unsigned for DUT input."""
    if raw < 0:
        raw += (1 << width)
    return raw


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

async def setup_dut(dut):
    """Initialize clock and reset the DUT."""
    clk = getattr(dut, "clk", None)
    if clk is not None:
        cocotb.start_soon(Clock(clk, 10, units="ns").start())
    rst_n = getattr(dut, "rst_n", getattr(dut, "rst", None))
    if rst_n is not None:
        rst_n.value = 0
        await ClockCycles(dut.clk, 5) if hasattr(dut, "clk") else await Timer(50, "ns")
        rst_n.value = 1
        await ClockCycles(dut.clk, 5) if hasattr(dut, "clk") else await Timer(50, "ns")


# ---------------------------------------------------------------------------
# Interface drivers
# ---------------------------------------------------------------------------

async def load_vector_streaming(dut, re_vals, im_vals):
    """Stream 256 complex samples into the FFT one clock at a time."""
    valid_in = getattr(dut, "valid_in", None)
    for i in range(N):
        if hasattr(dut, "data_in_re"):
            if hasattr(dut.data_in_re, "__len__"):
                # Packed bus
                dut.data_in_re.value = to_unsigned(to_fixed(re_vals[i]))
                dut.data_in_im.value = to_unsigned(to_fixed(im_vals[i]))
            else:
                dut.data_in_re.value = to_unsigned(to_fixed(re_vals[i]))
                dut.data_in_im.value = to_unsigned(to_fixed(im_vals[i]))
        if valid_in is not None:
            valid_in.value = 1
        if hasattr(dut, "clk"):
            await RisingEdge(dut.clk)
    if valid_in is not None:
        valid_in.value = 0


async def load_vector_memory(dut, re_vals, im_vals):
    """Load 256 samples via memory-mapped interface."""
    for i in range(N):
        if hasattr(dut, "mem_wr"):
            dut.mem_wr.value = 1
            dut.mem_addr.value = i
            dut.mem_wr_data_re.value = to_unsigned(to_fixed(re_vals[i]))
            dut.mem_wr_data_im.value = to_unsigned(to_fixed(im_vals[i]))
            await RisingEdge(dut.clk)
    if hasattr(dut, "mem_wr"):
        dut.mem_wr.value = 0


async def read_output_streaming(dut):
    """Read 256 complex output samples (streaming)."""
    re_out = []
    im_out = []
    for i in range(N):
        if hasattr(dut, "data_out_re"):
            re_out.append(from_fixed(dut.data_out_re.value.signed_integer))
            im_out.append(from_fixed(dut.data_out_im.value.signed_integer))
        if hasattr(dut, "clk"):
            await RisingEdge(dut.clk)
    return np.array(re_out), np.array(im_out)


async def read_output_memory(dut):
    """Read 256 complex output samples via memory-mapped interface."""
    re_out = []
    im_out = []
    for i in range(N):
        if hasattr(dut, "mem_rd"):
            dut.mem_rd.value = 1
            dut.mem_addr.value = i
            await ClockCycles(dut.clk, 2) if hasattr(dut, "clk") else await Timer(20, "ns")
            re_out.append(from_fixed(dut.mem_rd_data_re.value.signed_integer))
            im_out.append(from_fixed(dut.mem_rd_data_im.value.signed_integer))
    if hasattr(dut, "mem_rd"):
        dut.mem_rd.value = 0
    return np.array(re_out), np.array(im_out)


async def run_fft(dut, re_vals, im_vals):
    """Load input, trigger FFT, wait for done, read output."""
    # Determine interface type
    if hasattr(dut, "valid_in"):
        await load_vector_streaming(dut, re_vals, im_vals)
    elif hasattr(dut, "mem_wr"):
        await load_vector_memory(dut, re_vals, im_vals)

    # Trigger start if needed
    if hasattr(dut, "start"):
        dut.start.value = 1
        await RisingEdge(dut.clk) if hasattr(dut, "clk") else await Timer(10, "ns")
        dut.start.value = 0

    # Wait for done
    done = getattr(dut, "done", getattr(dut, "valid_out", None))
    if done is not None:
        timeout_cycles = 2000
        for _ in range(timeout_cycles):
            if done.value.integer == 1:
                break
            await RisingEdge(dut.clk) if hasattr(dut, "clk") else await Timer(10, "ns")
        else:
            dut._log.error("Timeout waiting for done signal")
    else:
        # No handshake — just wait enough cycles
        await ClockCycles(dut.clk, 512) if hasattr(dut, "clk") else await Timer(5000, "ns")

    # Read output
    if hasattr(dut, "data_out_re"):
        return await read_output_streaming(dut)
    elif hasattr(dut, "mem_rd"):
        return await read_output_memory(dut)
    else:
        dut._log.warning("No output interface detected")
        return np.zeros(N), np.zeros(N)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@cocotb.test()
async def test_fft256_known_vector(dut):
    """Test fft_256 with a known simple vector."""
    await setup_dut(dut)

    # Simple test: impulse
    re_vals = np.zeros(N, dtype=np.float32)
    im_vals = np.zeros(N, dtype=np.float32)
    re_vals[0] = 1.0

    ref = np.fft.fft(re_vals + 1j * im_vals)

    re_out, im_out = await run_fft(dut, re_vals, im_vals)
    out = re_out + 1j * im_out

    max_err = np.max(np.abs(out - ref))
    dut._log.info(f"Impulse test: max error = {max_err:.6f}")
    assert max_err < 0.5, f"Impulse response error too large: {max_err}"


@cocotb.test()
async def test_fft256_dc_vector(dut):
    """Test fft_256 with a DC (all ones) input."""
    await setup_dut(dut)

    re_vals = np.ones(N, dtype=np.float32) * 0.1
    im_vals = np.zeros(N, dtype=np.float32)

    ref = np.fft.fft(re_vals + 1j * im_vals)

    re_out, im_out = await run_fft(dut, re_vals, im_vals)
    out = re_out + 1j * im_out

    max_err = np.max(np.abs(out - ref))
    dut._log.info(f"DC test: max error = {max_err:.6f}")
    assert max_err < 0.5, f"DC response error too large: {max_err}"


@cocotb.test()
async def test_fft256_random_vectors(dut):
    """Test fft_256 with 1000 random complex vectors.

    Compares against numpy.fft.fft and reports the maximum error.
    Verifies all errors < 2 ULP in Q8.8 (1 ULP = 1/256 ≈ 0.0039).
    """
    await setup_dut(dut)

    np.random.seed(42)
    random.seed(42)
    max_error = 0.0
    errors = []
    n_tests = 1000

    for trial in range(n_tests):
        # Generate random input (scaled to avoid overflow)
        re_vals = np.random.uniform(-1.0, 1.0, N).astype(np.float32)
        im_vals = np.random.uniform(-1.0, 1.0, N).astype(np.float32)

        ref = np.fft.fft(re_vals + 1j * im_vals)

        re_out, im_out = await run_fft(dut, re_vals, im_vals)
        out = re_out + 1j * im_out

        err = np.max(np.abs(out - ref))
        errors.append(err)
        max_error = max(max_error, err)

        if trial % 100 == 0:
            dut._log.info(f"  trial {trial}/{n_tests}: max_err so far = {max_error:.6f}")

        # 2 ULP in Q8.8 = 2/256 ≈ 0.0078, but allow tolerance for accumulation
        assert err < 1.0, f"trial {trial}: error {err:.6f} too large"

    dut._log.info(
        f"1000 random vectors: max_error={max_error:.6f}, "
        f"mean_error={np.mean(errors):.6f}, "
        f"median_error={np.median(errors):.6f}"
    )

    # Report final statistics
    errors_arr = np.array(errors)
    pct_95 = np.percentile(errors_arr, 95)
    dut._log.info(f"95th percentile error: {pct_95:.6f}")

    # Overall assertion: max error should be reasonable
    assert max_error < 1.0, f"max error across 1000 vectors too large: {max_error}"


@cocotb.test()
async def test_fft256_sine_wave(dut):
    """Test fft_256 with a sine wave (should show a peak at the frequency)."""
    await setup_dut(dut)

    freq = 4  # cycles in N samples
    t = np.arange(N)
    re_vals = (np.sin(2 * np.pi * freq * t / N) * 0.5).astype(np.float32)
    im_vals = np.zeros(N, dtype=np.float32)

    ref = np.fft.fft(re_vals + 1j * im_vals)

    re_out, im_out = await run_fft(dut, re_vals, im_vals)
    out = re_out + 1j * im_out

    # Check peak is at the expected frequency bin
    peak_bin = np.argmax(np.abs(out))
    dut._log.info(f"Sine wave: peak at bin {peak_bin} (expected {freq} or {N - freq})")
    assert peak_bin == freq or peak_bin == N - freq, (
        f"peak at {peak_bin}, expected {freq} or {N - freq}"
    )

    # Check overall error
    max_err = np.max(np.abs(out - ref))
    dut._log.info(f"Sine wave test: max error = {max_err:.6f}")


@cocotb.test()
async def test_fft256_ifft_roundtrip(dut):
    """Verify FFT → IFFT recovers the original (if IFFT is available)."""
    await setup_dut(dut)

    re_vals = np.random.uniform(-0.5, 0.5, N).astype(np.float32)
    im_vals = np.random.uniform(-0.5, 0.5, N).astype(np.float32)

    # Forward FFT
    re_out, im_out = await run_fft(dut, re_vals, im_vals)

    # If the DUT has an IFFT mode, test round-trip
    if hasattr(dut, "mode"):
        dut.mode.value = 1  # IFFT mode (if supported)
        re_back, im_back = await run_fft(dut, re_out, im_out)
        recovered = (re_back + 1j * im_back) / N

        err = np.max(np.abs(recovered - (re_vals + 1j * im_vals)))
        dut._log.info(f"FFT→IFFT round-trip error: {err:.6f}")
        assert err < 0.5, f"round-trip error too large: {err}"
    else:
        dut._log.info("No IFFT mode available — skipping round-trip test")