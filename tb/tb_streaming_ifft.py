"""Cocotb testbench for streaming_ifft_loader.v — Streaming IFFT Input Loader.

Verifies the ping/pong dual-buffer streaming IFFT loader:
  1. Modes are correctly buffered and forwarded to the IFFT interface.
  2. The IFFT output does not start until n_modes are buffered.
  3. Data integrity: the first n_modes values sent to IFFT match the
     spectral-multiply output exactly (Q8.8 fixed-point).

Makefile snippet (place in tb/Makefile or run directly):

    #-------------------------------------------------------------------------
    # tb_streaming_ifft — streaming IFFT loader
    #-------------------------------------------------------------------------
    SIM ?= icarus
    TOPLEVEL ?= streaming_ifft_loader
    VERILOG = ../rtl/streaming_ifft_loader.v
    MODULE = tb_streaming_ifft

    include $(shell cocotb-config --makefiles)/Makefile.sim

    # Or run directly:
    #   make SIM=icarus TOPLEVEL=streaming_ifft_loader MODULE=tb_streaming_ifft

Run:
    cd tb && make SIM=icarus TOPLEVEL=streaming_ifft_loader MODULE=tb_streaming_ifft
"""

import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, ClockCycles, Timer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_WIDTH = 16
FRAC_BITS = 8
N_TOTAL = 256
DEFAULT_N_MODES = 32


# ---------------------------------------------------------------------------
# Fixed-point helpers (Q8.8)
# ---------------------------------------------------------------------------

def to_fixed(val, width=DATA_WIDTH, frac=FRAC_BITS):
    """Convert float to signed fixed-point integer."""
    raw = int(round(val * (1 << frac)))
    max_val = (1 << (width - 1)) - 1
    min_val = -(1 << (width - 1))
    raw = max(min_val, min(max_val, raw))
    return raw


def to_unsigned(raw, width=DATA_WIDTH):
    """Convert signed to unsigned for DUT input."""
    if raw < 0:
        raw += (1 << width)
    return raw


def from_unsigned(raw, width=DATA_WIDTH):
    """Convert unsigned DUT output to signed integer."""
    if raw >= (1 << (width - 1)):
        raw -= (1 << width)
    return raw


# ---------------------------------------------------------------------------
# Reset / clock setup
# ---------------------------------------------------------------------------

async def setup_dut(dut, n_modes=DEFAULT_N_MODES):
    """Initialize clock, drive reset, and set n_modes config."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    dut.n_modes.value = n_modes
    dut.start.value = 0
    dut.sm_data_valid.value = 0
    dut.sm_data_re.value = 0
    dut.sm_data_im.value = 0
    dut.ifft_data_ready.value = 0

    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)


# ---------------------------------------------------------------------------
# Stream drivers
# ---------------------------------------------------------------------------

async def drive_sm_output(dut, re_vals, im_vals, start_cycle=None):
    """Drive spectral-multiply output into the loader, one sample per cycle.

    Generates a ready/valid handshake.  Returns a list of (re, im) tuples
    that were actually accepted by the loader.
    """
    sent = []
    idx = 0
    while idx < len(re_vals):
        dut.sm_data_valid.value = 1
        dut.sm_data_re.value = to_unsigned(to_fixed(re_vals[idx]))
        dut.sm_data_im.value = to_unsigned(to_fixed(im_vals[idx]))
        await RisingEdge(dut.clk)
        # Check if accepted (sm_data_ready was high)
        if dut.sm_data_ready.value.integer == 1:
            sent.append((re_vals[idx], im_vals[idx]))
            idx += 1
        # If not ready, hold and try again next cycle
    dut.sm_data_valid.value = 0
    dut.sm_data_re.value = 0
    dut.sm_data_im.value = 0
    return sent


async def collect_ifft_output(dut, expected_count):
    """Collect IFFT output samples with a ready/valid handshake.

    IFFT is always ready (ifft_data_ready=1).  Returns list of (re_signed, im_signed).
    """
    dut.ifft_data_ready.value = 1
    collected = []
    timeout = 0
    max_timeout = expected_count * 10 + 500

    while len(collected) < expected_count:
        await RisingEdge(dut.clk)
        timeout += 1
        if timeout > max_timeout:
            dut._log.error(
                f"Timeout collecting IFFT output: got {len(collected)}"
                f"/{expected_count}"
            )
            break
        if dut.ifft_data_valid.value.integer == 1:
            re_raw = dut.ifft_data_re.value.integer
            im_raw = dut.ifft_data_im.value.integer
            re_signed = from_unsigned(re_raw)
            im_signed = from_unsigned(im_raw)
            collected.append((re_signed, im_signed))

    dut.ifft_data_ready.value = 0
    return collected


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@cocotb.test()
async def test_basic_buffer_and_forward(dut):
    """Test (1): modes are correctly buffered and forwarded.

    Send n_modes=32 distinct values, verify the IFFT receives all 32
    with exact data integrity.
    """
    n_modes = 32
    await setup_dut(dut, n_modes=n_modes)

    # Generate distinct test values: mode i has re=i*10+1, im=-(i*10+1)
    re_vals = [float(i * 10 + 1) / 256.0 for i in range(N_TOTAL)]
    im_vals = [float(-(i * 10 + 1)) / 256.0 for i in range(N_TOTAL)]

    # Expected fixed-point values for first n_modes
    expected = []
    for i in range(n_modes):
        expected.append((to_fixed(re_vals[i]), to_fixed(im_vals[i])))

    # Start the loader
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Drive SM output and collect IFFT output concurrently
    sm_task = cocotb.start_soon(drive_sm_output(dut, re_vals, im_vals))
    ifft_data = await collect_ifft_output(dut, expected_count=n_modes)
    await sm_task

    # Verify all n_modes were forwarded
    assert len(ifft_data) == n_modes, (
        f"Expected {n_modes} IFFT outputs, got {len(ifft_data)}"
    )

    # Verify data integrity: each forwarded value matches the SM input
    for i in range(n_modes):
        exp_re, exp_im = expected[i]
        got_re, got_im = ifft_data[i]
        assert got_re == exp_re, (
            f"Mode {i}: re mismatch — expected {exp_re}, got {got_re}"
        )
        assert got_im == exp_im, (
            f"Mode {i}: im mismatch — expected {exp_im}, got {got_im}"
        )

    dut._log.info(
        f"PASS: {n_modes} modes buffered and forwarded with exact integrity"
    )


@cocotb.test()
async def test_ifft_starts_after_n_modes(dut):
    """Test (2): IFFT output does not start until n_modes are buffered.

    Monitor ifft_data_valid; it must stay low until at least n_modes
    SM samples have been accepted.
    """
    n_modes = 32
    await setup_dut(dut, n_modes=n_modes)

    # Track when IFFT valid first asserts
    ifft_valid_first_cycle = None
    sm_accepted_count = 0

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Make IFFT always ready
    dut.ifft_data_ready.value = 1

    cycle = 0
    re_vals = [float(i + 1) / 256.0 for i in range(N_TOTAL)]
    im_vals = [float(-(i + 1)) / 256.0 for i in range(N_TOTAL)]

    # Drive SM data one cycle at a time, checking ifft_data_valid
    idx = 0
    while idx < N_TOTAL:
        dut.sm_data_valid.value = 1
        dut.sm_data_re.value = to_unsigned(to_fixed(re_vals[idx]))
        dut.sm_data_im.value = to_unsigned(to_fixed(im_vals[idx]))
        await RisingEdge(dut.clk)
        cycle += 1

        if dut.sm_data_ready.value.integer == 1:
            sm_accepted_count += 1
            idx += 1

        # Check: ifft_data_valid should NOT be high before n_modes accepted
        if dut.ifft_data_valid.value.integer == 1:
            if ifft_valid_first_cycle is None:
                ifft_valid_first_cycle = cycle
                dut._log.info(
                    f"ifft_data_valid first asserted at cycle {cycle}, "
                    f"sm_accepted={sm_accepted_count}"
                )

        # Before n_modes are accepted, ifft_data_valid must be 0
        if sm_accepted_count < n_modes:
            assert dut.ifft_data_valid.value.integer == 0, (
                f"ifft_data_valid asserted at cycle {cycle} before "
                f"{n_modes} modes buffered (only {sm_accepted_count} accepted)"
            )

    dut.sm_data_valid.value = 0

    # Now ifft_data_valid should have asserted
    assert ifft_valid_first_cycle is not None, (
        "ifft_data_valid never asserted"
    )

    # The IFFT valid should assert only after n_modes are accepted
    # (it asserts on the cycle AFTER the n_modes-th sample is written,
    #  which is the same cycle the state transitions to S_FEED)
    dut._log.info(
        f"PASS: IFFT started at cycle {ifft_valid_first_cycle} "
        f"after {n_modes} modes buffered"
    )


@cocotb.test()
async def test_data_integrity_distinct(dut):
    """Test (3): data integrity with distinct, non-trivial values.

    Uses values that exercise the full Q8.8 range to ensure no data
    corruption through the buffer path.
    """
    n_modes = 32
    await setup_dut(dut, n_modes=n_modes)

    random.seed(12345)

    # Generate random Q8.8 values across the full signed range
    re_fixed = [random.randint(-32768, 32767) for _ in range(N_TOTAL)]
    im_fixed = [random.randint(-32768, 32767) for _ in range(N_TOTAL)]

    # Convert to float for the driver (which re-quantizes)
    re_vals = [v / 256.0 for v in re_fixed]
    im_vals = [v / 256.0 for v in im_fixed]

    expected = []
    for i in range(n_modes):
        expected.append((to_fixed(re_vals[i]), to_fixed(im_vals[i])))

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    sm_task = cocotb.start_soon(drive_sm_output(dut, re_vals, im_vals))
    ifft_data = await collect_ifft_output(dut, expected_count=n_modes)
    await sm_task

    assert len(ifft_data) == n_modes, (
        f"Expected {n_modes}, got {len(ifft_data)}"
    )

    mismatches = 0
    for i in range(n_modes):
        exp_re, exp_im = expected[i]
        got_re, got_im = ifft_data[i]
        if got_re != exp_re or got_im != exp_im:
            mismatches += 1
            dut._log.error(
                f"Mode {i}: expected ({exp_re}, {exp_im}), "
                f"got ({got_re}, {got_im})"
            )

    assert mismatches == 0, f"Data integrity failed: {mismatches} mismatches"
    dut._log.info(f"PASS: all {n_modes} modes forwarded with exact integrity")


@cocotb.test()
async def test_default_n_modes_when_config_zero(dut):
    """Test: when n_modes=0, the loader defaults to 32 modes."""
    await setup_dut(dut, n_modes=0)  # Config = 0 → should default to 32

    re_vals = [float(i + 1) / 256.0 for i in range(N_TOTAL)]
    im_vals = [float(-(i + 1)) / 256.0 for i in range(N_TOTAL)]
    expected = [
        (to_fixed(re_vals[i]), to_fixed(im_vals[i]))
        for i in range(DEFAULT_N_MODES)
    ]

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    sm_task = cocotb.start_soon(drive_sm_output(dut, re_vals, im_vals))
    ifft_data = await collect_ifft_output(dut, expected_count=DEFAULT_N_MODES)
    await sm_task

    assert len(ifft_data) == DEFAULT_N_MODES, (
        f"Expected {DEFAULT_N_MODES} (default), got {len(ifft_data)}"
    )

    for i in range(DEFAULT_N_MODES):
        exp_re, exp_im = expected[i]
        got_re, got_im = ifft_data[i]
        assert got_re == exp_re, f"Mode {i} re: expected {exp_re}, got {got_re}"
        assert got_im == exp_im, f"Mode {i} im: expected {exp_im}, got {got_im}"

    dut._log.info(f"PASS: default n_modes=32 when config=0")


@cocotb.test()
async def test_small_n_modes(dut):
    """Test: with a small n_modes=8, IFFT starts after 8 modes."""
    n_modes = 8
    await setup_dut(dut, n_modes=n_modes)

    re_vals = [float(i * 100 + 1) / 256.0 for i in range(N_TOTAL)]
    im_vals = [float(-(i * 100 + 1)) / 256.0 for i in range(N_TOTAL)]
    expected = [
        (to_fixed(re_vals[i]), to_fixed(im_vals[i]))
        for i in range(n_modes)
    ]

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    sm_task = cocotb.start_soon(drive_sm_output(dut, re_vals, im_vals))
    ifft_data = await collect_ifft_output(dut, expected_count=n_modes)
    await sm_task

    assert len(ifft_data) == n_modes, (
        f"Expected {n_modes}, got {len(ifft_data)}"
    )

    for i in range(n_modes):
        exp_re, exp_im = expected[i]
        got_re, got_im = ifft_data[i]
        assert got_re == exp_re, f"Mode {i} re: expected {exp_re}, got {got_re}"
        assert got_im == exp_im, f"Mode {i} im: expected {exp_im}, got {got_im}"

    dut._log.info(f"PASS: n_modes={n_modes} correctly buffered and forwarded")


@cocotb.test()
async def test_ifft_backpressure(dut):
    """Test: IFFT with backpressure (not always ready).

    The loader should hold ifft_data_valid until the IFFT accepts.
    """
    n_modes = 16
    await setup_dut(dut, n_modes=n_modes)

    re_vals = [float(i + 1) / 256.0 for i in range(N_TOTAL)]
    im_vals = [float(-(i + 1)) / 256.0 for i in range(N_TOTAL)]
    expected = [
        (to_fixed(re_vals[i]), to_fixed(im_vals[i]))
        for i in range(n_modes)
    ]

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Start SM driver
    sm_task = cocotb.start_soon(drive_sm_output(dut, re_vals, im_vals))

    # Collect IFFT output with intermittent backpressure
    dut.ifft_data_ready.value = 1
    collected = []
    timeout = 0
    max_timeout = n_modes * 50 + 500

    while len(collected) < n_modes:
        await RisingEdge(dut.clk)
        timeout += 1
        if timeout > max_timeout:
            dut._log.error(f"Timeout: got {len(collected)}/{n_modes}")
            break

        # Toggle ready every other cycle to create backpressure
        if len(collected) % 3 == 2:
            dut.ifft_data_ready.value = 0
        else:
            dut.ifft_data_ready.value = 1

        if dut.ifft_data_valid.value.integer == 1 and \
           dut.ifft_data_ready.value.integer == 1:
            re_raw = dut.ifft_data_re.value.integer
            im_raw = dut.ifft_data_im.value.integer
            collected.append((from_unsigned(re_raw), from_unsigned(im_raw)))

    dut.ifft_data_ready.value = 0
    await sm_task

    assert len(collected) == n_modes, (
        f"Expected {n_modes}, got {len(collected)}"
    )

    for i in range(n_modes):
        exp_re, exp_im = expected[i]
        got_re, got_im = collected[i]
        assert got_re == exp_re, f"Mode {i} re: expected {exp_re}, got {got_re}"
        assert got_im == exp_im, f"Mode {i} im: expected {exp_im}, got {got_im}"

    dut._log.info(f"PASS: backpressure handled, {n_modes} modes intact")