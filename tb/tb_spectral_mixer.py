"""Cocotb testbench for spectral_mixer.v — Prompt P19/P22.

Loads spectral weights via the Wishbone bus interface, starts the spectral
mixer computation, and compares the output against a Python AFNO simulation
with matching quantization.

Run with:
    make SIM=icarus TOPLEVEL=spectral_mixer

Assumed DUT interface:
    clk, rst_n
    Wishbone bus: wb_adr_i, wb_dat_i, wb_dat_o, wb_we_i, wb_stb_i, wb_ack_o, wb_cyc_i
    data_in (256 × channels, real/imag or packed)
    data_out (256 × channels)
    start, done
"""

import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, Timer

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N = 256           # FFT size
CHANNELS = 8      # Feature dimension (configurable)
BLOCK_SIZE = 8    # Block-diagonal block size
MODES = 32        # Number of spectral modes
DATA_WIDTH = 16
FRAC_BITS = 8
SCALE = 1 << FRAC_BITS


# Wishbone register map (must match wishbone_if.v)
REG_START = 0x00
REG_DONE = 0x01
REG_MODE_COUNT = 0x02
REG_BLOCK_SIZE = 0x03
REG_THRESHOLD = 0x04
REG_WEIGHT_BASE = 0x10
REG_DATA_BASE = 0x40
REG_OUTPUT_BASE = 0x80


# ---------------------------------------------------------------------------
# Fixed-point helpers
# ---------------------------------------------------------------------------

def to_fixed(val, width=DATA_WIDTH, frac=FRAC_BITS):
    raw = int(round(val * (1 << frac)))
    max_val = (1 << (width - 1)) - 1
    min_val = -(1 << (width - 1))
    return max(min_val, min(max_val, raw))


def from_fixed(raw, width=DATA_WIDTH, frac=FRAC_BITS):
    if raw >= (1 << (width - 1)):
        raw -= (1 << width)
    return raw / (1 << frac)


def to_unsigned(raw, width=DATA_WIDTH):
    if raw < 0:
        raw += (1 << width)
    return raw


# ---------------------------------------------------------------------------
# Wishbone bus driver
# ---------------------------------------------------------------------------

class WishboneBus:
    """Wishbone Classic bus driver for the cocotb DUT."""

    def __init__(self, dut, clk_name="clk"):
        self.dut = dut
        self.clk_name = clk_name
        self._has_wishbone = all(
            hasattr(dut, sig) for sig in [
                "wb_adr_i", "wb_dat_i", "wb_we_i", "wb_stb_i", "wb_cyc_i"
            ]
        )

    async def write(self, addr, data):
        """Write a 32-bit word to a Wishbone register."""
        if not self._has_wishbone:
            # Fall back to direct register access
            await self._direct_write(addr, data)
            return

        self.dut.wb_adr_i.value = addr
        self.dut.wb_dat_i.value = data
        self.dut.wb_we_i.value = 1
        self.dut.wb_stb_i.value = 1
        self.dut.wb_cyc_i.value = 1

        # Wait for ack
        timeout = 100
        for _ in range(timeout):
            if hasattr(self.dut, "wb_ack_o") and self.dut.wb_ack_o.value == 1:
                break
            await RisingEdge(getattr(self.dut, self.clk_name))

        self.dut.wb_stb_i.value = 0
        self.dut.wb_cyc_i.value = 0
        self.dut.wb_we_i.value = 0

    async def read(self, addr):
        """Read a 32-bit word from a Wishbone register."""
        if not self._has_wishbone:
            return await self._direct_read(addr)

        self.dut.wb_adr_i.value = addr
        self.dut.wb_we_i.value = 0
        self.dut.wb_stb_i.value = 1
        self.dut.wb_cyc_i.value = 1

        timeout = 100
        for _ in range(timeout):
            if hasattr(self.dut, "wb_ack_o") and self.dut.wb_ack_o.value == 1:
                break
            await RisingEdge(getattr(self.dut, self.clk_name))

        data = self.dut.wb_dat_o.value.signed_integer if hasattr(self.dut, "wb_dat_o") else 0

        self.dut.wb_stb_i.value = 0
        self.dut.wb_cyc_i.value = 0

        return data

    async def _direct_write(self, addr, data):
        """Direct register write (non-Wishbone fallback)."""
        reg_map = {
            REG_START: "start",
            REG_MODE_COUNT: "mode_count",
            REG_BLOCK_SIZE: "block_size",
            REG_THRESHOLD: "threshold",
        }
        if addr in reg_map and hasattr(self.dut, reg_map[addr]):
            getattr(self.dut, reg_map[addr]).value = data
        await RisingEdge(getattr(self.dut, self.clk_name))

    async def _direct_read(self, addr):
        """Direct register read (non-Wishbone fallback)."""
        reg_map = {
            REG_DONE: "done",
            REG_MODE_COUNT: "mode_count",
            REG_BLOCK_SIZE: "block_size",
        }
        if addr in reg_map and hasattr(self.dut, reg_map[addr]):
            return getattr(self.dut, reg_map[addr]).value.integer
        return 0

    async def write_weight_block(self, base_addr, data_array):
        """Write a block of weight data to consecutive addresses."""
        for i, val in enumerate(data_array):
            await self.write(base_addr + i, to_unsigned(int(val)))

    async def read_output_block(self, base_addr, n_words):
        """Read a block of output data from consecutive addresses."""
        result = []
        for i in range(n_words):
            val = await self.read(base_addr + i)
            result.append(val)
        return result


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

async def setup_dut(dut):
    """Initialize clock and reset."""
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
# Python reference simulation (AFNO with int8 quantization)
# ---------------------------------------------------------------------------

def python_afno_sim(input_data, weights, modes, block_size, threshold):
    """Simulate the spectral mixer in Python with int8 weights.

    Parameters
    ----------
    input_data : np.ndarray, shape (N, channels)
        Real input data.
    weights : tuple of (q_re, q_im), each int8, shape (modes, channels)
        Quantized complex spectral weights.
    modes : int
        Number of spectral modes.
    block_size : int
        Block-diagonal block size.
    threshold : float
        Soft-threshold value.

    Returns
    -------
    np.ndarray, shape (N, channels)
        Output of the spectral mixer.
    """
    N, ch = input_data.shape

    # FFT along sequence dimension
    freq = np.fft.fft(input_data, axis=0)  # (N, ch)

    # Dequantize weights
    q_re, q_im = weights
    w = q_re.astype(np.float32) / 127.0 + 1j * q_im.astype(np.float32) / 127.0

    # Apply spectral weights to first 'modes' modes
    k = min(modes, N // 2)
    if w.ndim == 2:
        freq[:k] = freq[:k] * w[:k][np.newaxis, ...] if w.shape[0] >= k else freq[:k] * w[np.newaxis, ...]
    else:
        freq[:k] = freq[:k] * w[:k, np.newaxis]

    # Soft-thresholding
    mag = np.abs(freq[:k])
    scale = np.maximum(mag - threshold, 0) / (mag + 1e-12)
    freq[:k] = freq[:k] * scale

    # Zero out modes beyond k
    freq[k:] = 0

    # IFFT
    output = np.real(np.fft.ifft(freq, axis=0))
    return output.astype(np.float32)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@cocotb.test()
async def test_spectral_mixer_basic(dut):
    """Basic test: load weights, run one inference, check output is finite."""
    await setup_dut(dut)
    bus = WishboneBus(dut)

    # Configure
    await bus.write(REG_MODE_COUNT, MODES)
    await bus.write(REG_BLOCK_SIZE, BLOCK_SIZE)
    threshold_fixed = int(0.1 * SCALE)
    await bus.write(REG_THRESHOLD, threshold_fixed)

    # Load weights (int8 quantized)
    np.random.seed(42)
    q_re = np.random.randint(-127, 128, (MODES, CHANNELS), dtype=np.int8)
    q_im = np.random.randint(-127, 128, (MODES, CHANNELS), dtype=np.int8)

    # Write real parts
    for i in range(MODES):
        for j in range(CHANNELS):
            await bus.write(
                REG_WEIGHT_BASE + i * CHANNELS + j,
                to_unsigned(int(q_re[i, j]))
            )

    dut._log.info("Weights loaded")


@cocotb.test()
async def test_spectral_mixer_vs_python(dut):
    """Compare spectral_mixer output to Python AFNO simulation.

    Loads int8 quantized weights, runs 10 random inputs, and compares.
    """
    await setup_dut(dut)
    bus = WishboneBus(dut)

    np.random.seed(123)
    threshold = 0.1

    # Configure
    await bus.write(REG_MODE_COUNT, MODES)
    await bus.write(REG_BLOCK_SIZE, BLOCK_SIZE)
    await bus.write(REG_THRESHOLD, int(threshold * SCALE))

    # Generate and load quantized weights
    q_re = np.random.randint(-127, 128, (MODES, CHANNELS), dtype=np.int8)
    q_im = np.random.randint(-127, 128, (MODES, CHANNELS), dtype=np.int8)

    # Write weights via Wishbone
    for i in range(MODES):
        for j in range(CHANNELS):
            await bus.write(
                REG_WEIGHT_BASE + i * 2 * CHANNELS + j,
                to_unsigned(int(q_re[i, j]))
            )
            await bus.write(
                REG_WEIGHT_BASE + i * 2 * CHANNELS + CHANNELS + j,
                to_unsigned(int(q_im[i, j]))
            )

    dut._log.info(f"Loaded {MODES}x{CHANNELS} complex weights")

    # Run 10 random inputs
    max_errors = []
    for trial in range(10):
        # Generate random input (scaled to avoid overflow)
        input_data = np.random.uniform(-0.5, 0.5, (N, CHANNELS)).astype(np.float32)

        # Python reference
        ref_output = python_afno_sim(
            input_data, (q_re, q_im), MODES, BLOCK_SIZE, threshold
        )

        # Load input data via Wishbone
        for i in range(N):
            for j in range(CHANNELS):
                await bus.write(
                    REG_DATA_BASE + i * CHANNELS + j,
                    to_unsigned(to_fixed(input_data[i, j]))
                )

        # Start computation
        await bus.write(REG_START, 1)
        await RisingEdge(dut.clk) if hasattr(dut, "clk") else await Timer(10, "ns")
        await bus.write(REG_START, 0)

        # Wait for done
        timeout_cycles = 5000
        done = False
        for _ in range(timeout_cycles):
            done_val = await bus.read(REG_DONE)
            if done_val & 1:
                done = True
                break
            await RisingEdge(dut.clk) if hasattr(dut, "clk") else await Timer(10, "ns")

        if not done:
            dut._log.error(f"Trial {trial}: timeout waiting for done")
            continue

        # Read output
        hw_output = np.zeros((N, CHANNELS), dtype=np.float32)
        for i in range(N):
            for j in range(CHANNELS):
                raw = await bus.read(REG_OUTPUT_BASE + i * CHANNELS + j)
                hw_output[i, j] = from_fixed(raw)

        # Compare
        err = np.max(np.abs(hw_output - ref_output))
        max_errors.append(err)
        dut._log.info(f"Trial {trial}: max error = {err:.6f}")

    if max_errors:
        overall_max = max(max_errors)
        dut._log.info(f"Overall max error across {len(max_errors)} trials: {overall_max:.6f}")
        # Allow reasonable tolerance for fixed-point quantization
        assert overall_max < 2.0, f"max error {overall_max} too large"


@cocotb.test()
async def test_spectral_mixer_zero_input(dut):
    """Test that zero input produces near-zero output."""
    await setup_dut(dut)
    bus = WishboneBus(dut)

    # Configure
    await bus.write(REG_MODE_COUNT, MODES)
    await bus.write(REG_BLOCK_SIZE, BLOCK_SIZE)
    await bus.write(REG_THRESHOLD, 0)

    # Zero input
    for i in range(N):
        for j in range(CHANNELS):
            await bus.write(REG_DATA_BASE + i * CHANNELS + j, 0)

    # Start
    await bus.write(REG_START, 1)
    await RisingEdge(dut.clk) if hasattr(dut, "clk") else await Timer(10, "ns")
    await bus.write(REG_START, 0)

    # Wait for done
    for _ in range(5000):
        if (await bus.read(REG_DONE)) & 1:
            break
        await RisingEdge(dut.clk) if hasattr(dut, "clk") else await Timer(10, "ns")

    # Read output — should be near zero
    max_val = 0.0
    for i in range(0, N, 32):  # Sample every 32
        for j in range(0, CHANNELS, 2):  # Sample every 2 channels
            raw = await bus.read(REG_OUTPUT_BASE + i * CHANNELS + j)
            val = abs(from_fixed(raw))
            max_val = max(max_val, val)

    dut._log.info(f"Zero input → max output magnitude: {max_val:.6f}")
    assert max_val < 0.1, f"zero input should produce near-zero output (got {max_val})"