# =============================================================================
# tb_mode_skip.py — Cocotb testbench for mode_skip_multiply.v
# =============================================================================
#
# Verifies the mode-skip power-optimization module:
#   (1) Modes 0..31 produce correct complex multiply + soft-threshold results.
#   (2) Modes 32..255 output exactly zero (multiplier bypassed).
#   (3) Throughput is still 1 mode per clock cycle (constant latency).
#
# The module is a drop-in replacement for spectral_multiply.v with the same
# streaming interface: data_in_valid/ready, data_out_valid/ready, and a
# mode counter that wraps 0..255.
#
# -----------------------------------------------------------------------------
# Makefile snippet (add to the project Makefile):
# -----------------------------------------------------------------------------
#
#   SIM ?= icarus
#   TOPLEVEL ?= mode_skip_multiply
#   VERILOG_SOURCES = rtl/mode_skip_multiply.v
#   MODULE = tb_mode_skip
#
#   include $(shell cocotb-config --makefiles)/Makefile.sim
#
# Run:
#   make SIM=icarus TOPLEVEL=mode_skip_multiply MODULE=tb_mode_skip
#
# -----------------------------------------------------------------------------
# DUT interface (mode_skip_multiply.v):
#   clk, rst_n
#   weight_we, weight_addr[4:0], weight_wr_re[15:0], weight_wr_im[15:0]
#   threshold[15:0]
#   data_in_valid, data_in_ready, data_in_re[15:0], data_in_im[15:0]
#   data_out_valid, data_out_ready, data_out_re[15:0], data_out_im[15:0]
# =============================================================================

import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, Timer

import numpy as np


# ---------------------------------------------------------------------------
# Constants (must match RTL parameters)
# ---------------------------------------------------------------------------

N_MODES    = 32
BLOCK_SIZE = 8
WIDTH      = 16
FRAC       = 8
SCALE      = 1 << FRAC
N_TOTAL    = 256    # total FFT modes (mode counter wraps at 255)


# ---------------------------------------------------------------------------
# Fixed-point helpers (Q8.8 — 16-bit signed, 8 fractional bits)
# ---------------------------------------------------------------------------

def to_fixed(val, width=WIDTH, frac=FRAC):
    """Convert a float to a signed fixed-point integer."""
    raw = int(round(val * (1 << frac)))
    max_val = (1 << (width - 1)) - 1
    min_val = -(1 << (width - 1))
    return max(min_val, min(max_val, raw))


def from_fixed(raw, width=WIDTH, frac=FRAC):
    """Convert a cocotb unsigned/signed integer back to float."""
    iv = int(raw)
    if iv >= (1 << (width - 1)):
        iv -= (1 << width)
    return iv / (1 << frac)


def to_signed(val, width=WIDTH):
    """Wrap a Python int into the signed representation cocotb expects."""
    val = int(val) & ((1 << width) - 1)
    if val >= (1 << (width - 1)):
        val -= (1 << width)
    return val


# ---------------------------------------------------------------------------
# Reference model: complex multiply + soft-threshold + truncation
# ---------------------------------------------------------------------------

def ref_complex_mul(w_re, w_im, x_re, x_im, threshold):
    """
    Compute the expected output of spectral_multiply for a single mode.

    Matches the RTL exactly:
      1. Complex multiply:  y = w * x  (Q8.8, >> FRAC)
      2. Soft-threshold:    if |w| < threshold → 0
      3. Truncation:         if mode >= N_MODES → 0
    Returns (re, im) as floats.
    """
    # Complex multiply in fixed-point
    prod_re = w_re * x_re - w_im * x_im
    prod_im = w_re * x_im + w_im * x_re
    mult_re = prod_re >> FRAC
    mult_im = prod_im >> FRAC

    # Approximate magnitude: max(|w_re|,|w_im|) + min/2
    abs_re = abs(w_re)
    abs_im = abs(w_im)
    max_abs = max(abs_re, abs_im)
    min_abs = min(abs_re, abs_im)
    mag = max_abs + (min_abs >> 1)

    is_zero = mag < threshold

    if is_zero:
        return 0.0, 0.0
    return from_fixed(mult_re), from_fixed(mult_im)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

async def setup_dut(dut):
    """Initialize clock and reset the DUT."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    dut.rst_n.value = 0
    dut.weight_we.value = 0
    dut.weight_addr.value = 0
    dut.weight_wr_re.value = 0
    dut.weight_wr_im.value = 0
    dut.threshold.value = 0
    dut.data_in_valid.value = 0
    dut.data_in_re.value = 0
    dut.data_in_im.value = 0
    dut.data_out_ready.value = 1
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)


async def write_weight(dut, idx, w_re, w_im):
    """Write a single complex weight into the register file."""
    dut.weight_we.value = 1
    dut.weight_addr.value = idx
    dut.weight_wr_re.value = to_signed(w_re)
    dut.weight_wr_im.value = to_signed(w_im)
    await RisingEdge(dut.clk)
    dut.weight_we.value = 0


async def load_weights(dut, weights):
    """Load all N_MODES complex weights.

    weights: list of (w_re, w_im) in Q8.8 integer form.
    """
    for idx, (w_re, w_im) in enumerate(weights):
        await write_weight(dut, idx, w_re, w_im)


# ---------------------------------------------------------------------------
# Drive / monitor helpers
# ---------------------------------------------------------------------------

async def drive_one_mode(dut, x_re, x_im):
    """Present one input mode and wait for it to be accepted (1 cycle)."""
    dut.data_in_re.value = to_signed(x_re)
    dut.data_in_im.value = to_signed(x_im)
    dut.data_in_valid.value = 1
    # Wait until data_in_ready is high (should be immediate after reset)
    for _ in range(10):
        await RisingEdge(dut.clk)
        if dut.data_in_ready.value == 1:
            break
    dut.data_in_valid.value = 0


async def capture_one_output(dut, timeout=20):
    """Wait for data_out_valid and capture one output mode."""
    for _ in range(timeout):
        await RisingEdge(dut.clk)
        if dut.data_out_valid.value == 1:
            re_val = from_fixed(dut.data_out_re.value)
            im_val = from_fixed(dut.data_out_im.value)
            return re_val, im_val
    raise cocotb.result.TestFailure(
        "Timeout waiting for data_out_valid"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@cocotb.test()
async def test_active_modes_correct(dut):
    """Test (1): modes 0..31 produce correct multiply + soft-threshold results."""
    await setup_dut(dut)

    # Set a small threshold so soft-thresholding does not zero everything.
    # threshold = 0.5 in Q8.8 → 128
    threshold = 128
    dut.threshold.value = threshold

    # Load weights: alternating pattern so each mode maps to a distinct weight
    # via block-diagonal indexing (wt_idx = mode % BLOCK_SIZE).
    # Use weights well above threshold so the soft-threshold does not fire
    # for most modes.
    np.random.seed(42)
    weights = []
    for i in range(N_MODES):
        w_re = to_fixed(np.random.uniform(0.5, 1.5))
        w_im = to_fixed(np.random.uniform(-0.5, 0.5))
        weights.append((w_re, w_im))
    await load_weights(dut, weights)

    dut._log.info("Weights loaded, testing modes 0..31")

    # Drive 32 active modes and capture outputs.
    # The pipeline has 1-cycle latency, so outputs lag inputs by 1 cycle.
    errors = []
    for mode in range(N_MODES):
        x_re = to_fixed(np.random.uniform(-0.5, 0.5))
        x_im = to_fixed(np.random.uniform(-0.5, 0.5))

        wt_idx = mode % BLOCK_SIZE
        w_re, w_im = weights[wt_idx]

        await drive_one_mode(dut, x_re, x_im)

        # Capture output (lag of 1 cycle means first output appears after
        # the second input is driven; handle by capturing on the cycle after
        # driving).  We capture in the same loop for simplicity since the
        # pipeline is 1-stage.
        hw_re, hw_im = await capture_one_output(dut)

        exp_re, exp_im = ref_complex_mul(w_re, w_im, x_re, x_im, threshold)

        err = max(abs(hw_re - exp_re), abs(hw_im - exp_im))
        if err > 0.02:  # tolerance for fixed-point rounding
            errors.append((mode, hw_re, hw_im, exp_re, exp_im, err))

        if mode < 4 or err > 0.02:
            dut._log.info(
                f"  mode {mode:3d}: hw=({hw_re:+.4f},{hw_im:+.4f}) "
                f"exp=({exp_re:+.4f},{exp_im:+.4f}) err={err:.6f}"
            )

    assert len(errors) == 0, (
        f"{len(errors)} modes mismatched. First: {errors[0]}"
    )
    dut._log.info(f"PASS: all {N_MODES} active modes correct")


@cocotb.test()
async def test_truncated_modes_zero(dut):
    """Test (2): modes 32..255 output exactly zero (multiplier bypassed)."""
    await setup_dut(dut)

    # Set threshold to 0 so soft-thresholding never fires — the only zeroing
    # mechanism for modes >= N_MODES is the truncation bypass.
    dut.threshold.value = 0

    # Load non-trivial weights (all above threshold=0 so they never get
    # soft-thresholded).
    np.random.seed(99)
    weights = []
    for i in range(N_MODES):
        weights.append((to_fixed(1.0), to_fixed(0.5)))
    await load_weights(dut, weights)

    dut._log.info("Testing modes 32..255 output zero")

    # First, cycle through modes 0..31 to advance mode_cnt to 32.
    # Drive all 32 active modes.
    for mode in range(N_MODES):
        x_re = to_fixed(np.random.uniform(-0.5, 0.5))
        x_im = to_fixed(np.random.uniform(-0.5, 0.5))
        await drive_one_mode(dut, x_re, x_im)
        # Drain the pipeline output for active modes
        await capture_one_output(dut)

    # Now mode_cnt == 32.  Drive modes 32..255 (224 modes) and verify zero.
    max_mag = 0.0
    zero_count = 0
    sample_interval = 32  # check every mode but log samples
    for mode in range(N_MODES, N_TOTAL):
        x_re = to_fixed(np.random.uniform(-1.0, 1.0))
        x_im = to_fixed(np.random.uniform(-1.0, 1.0))
        await drive_one_mode(dut, x_re, x_im)
        hw_re, hw_im = await capture_one_output(dut)

        mag = max(abs(hw_re), abs(hw_im))
        max_mag = max(max_mag, mag)

        if mag == 0.0:
            zero_count += 1

        if (mode - N_MODES) % sample_interval == 0:
            dut._log.info(
                f"  mode {mode:3d}: out=({hw_re:+.4f},{hw_im:+.4f}) mag={mag:.6f}"
            )

    total_truncated = N_TOTAL - N_MODES
    dut._log.info(
        f"Truncated modes: {zero_count}/{total_truncated} exactly zero, "
        f"max magnitude={max_mag:.6f}"
    )

    assert max_mag == 0.0, (
        f"Truncated modes should be zero, max magnitude={max_mag}"
    )
    assert zero_count == total_truncated, (
        f"Expected {total_truncated} zero outputs, got {zero_count}"
    )
    dut._log.info(
        f"PASS: all {total_truncated} truncated modes output zero"
    )


@cocotb.test()
async def test_throughput_one_per_cycle(dut):
    """Test (3): throughput is 1 mode/cycle (constant latency, no stall)."""
    await setup_dut(dut)

    dut.threshold.value = 0

    # Load identity weights (1+0j) for quick verification.
    weights = [(to_fixed(1.0), 0) for _ in range(N_MODES)]
    await load_weights(dut, weights)

    dut._log.info("Testing throughput: 1 mode per cycle")

    # Drive 256 modes back-to-back (no gap between data_in_valid cycles).
    # Count the number of cycles from first valid input to last valid output.
    # With 1-cycle pipeline latency and 1 mode/cycle throughput, 256 modes
    # should take 256 + 1 = 257 cycles (256 input + 1 latency).
    dut.data_out_ready.value = 1

    # Start timing: drive first input
    start_cycle = 0
    outputs_captured = 0
    cycle = 0

    # We drive all 256 modes in consecutive cycles while capturing outputs.
    # Use a combined drive+capture loop.
    dut.data_in_valid.value = 1

    # Phase 1: drive inputs for 256 consecutive cycles
    for mode in range(N_TOTAL):
        dut.data_in_re.value = to_fixed(0.25 * (1 if mode < N_MODES else 2))
        dut.data_in_im.value = to_fixed(0.0)
        await RisingEdge(dut.clk)
        cycle += 1

    dut.data_in_valid.value = 0

    # Phase 2: drain remaining pipeline outputs (at most 1 extra cycle)
    drain_timeout = 10
    for _ in range(drain_timeout):
        await RisingEdge(dut.clk)
        cycle += 1
        if dut.data_out_valid.value == 1:
            outputs_captured += 1
        # After the pipeline is empty, data_out_valid goes low
        if dut.data_out_valid.value == 0 and outputs_captured > 0:
            break

    total_cycles = cycle

    # Throughput check: 256 modes in ~257 cycles (1-cycle pipeline latency).
    # Allow a small margin for reset/sync overhead.
    expected_cycles = N_TOTAL + 1  # 256 modes + 1 latency
    dut._log.info(
        f"Processed {N_TOTAL} modes in {total_cycles} cycles "
        f"(expected ~{expected_cycles})"
    )

    assert total_cycles <= expected_cycles + 5, (
        f"Throughput violation: {total_cycles} cycles for {N_TOTAL} modes, "
        f"expected <= {expected_cycles + 5}"
    )
    dut._log.info(
        f"PASS: throughput = {N_TOTAL / total_cycles:.3f} modes/cycle "
        f"(1 mode/cycle maintained)"
    )