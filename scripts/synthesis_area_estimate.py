#!/usr/bin/env python3
"""Synthesis area estimation for Spectral Silicon (v1/v2/v3).

Estimates gate counts for all RTL modules using analytical formulas
based on the architectural parameters of the chip.  The estimates are
first-order analytical models — not synthesis results — but they are
suitable for architectural comparison and tile-size planning.

Gate-Count Formulas
-------------------

The key formulas used (all gate counts in gate-equivalent units, GE):

**FFT (N-point, radix-4, S stages)**::

    stages = log4(N) = log2(N) / 2
    butterfly_gates = stages * radix4_butterfly_gates
    radix4_butterfly_gates = 3 * complex_mult_gates + 8 * adder_gates + reg_gates
    complex_mult_gates = 4 * (WIDTH/2) * (pp_gates + csa_gates)   # Booth
    FFT_total = N/4 * butterfly_gates + twiddle_gates + memory_gates

**Booth Multiplier (WIDTH-bit)**::

    partial_products = WIDTH / 2
    booth_gates = partial_products * (pp_gates + csa_gates)
    pp_gates     ≈ 20  (Booth encoder + MUX per partial product)
    csa_gates    ≈ 14  (carry-save adder per bit-slice)

**Block Floating-Point (BFP, N samples, block_size B)**::

    comparator_gates ≈ 12 per bit
    shifter_gates   ≈ 8 per bit
    BFP_total = (N / B) * (log2(B) * comparator_gates + B * shifter_gates) + exp_regs

**Memory (bits, dual-port SRAM)**::

    bits_per_cell     = 1  (1T1R cell)
    per_cell_overhead = 4  (sense amp + write circuitry amortized)
    Memory_total = bits / bits_per_cell * per_cell_overhead

**Spectral Multiply (K modes, BLOCK_SIZE block-diagonal)**::

    real_mult_gates = WIDTH * 8  (per real multiplier)
    SM_total = K * 2 * real_mult_gates / BLOCK_SIZE + accumulator_gates

**modReLU + soft-threshold**::

    magnitude_gates = 2 * WIDTH  (square + sqrt approximation)
    comparator_gates = WIDTH
    modrelu_total = magnitude_gates + comparator_gates + bias_adder

Usage
-----

    # Print a formatted table
    python scripts/synthesis_area_estimate.py

    # Output JSON
    python scripts/synthesis_area_estimate.py --json

    # Custom parameters
    python scripts/synthesis_area_estimate.py --width 16 --n-fft 256 --n-modes 32
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Ensure project root is on the path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

__all__ = [
    "estimate_fft",
    "estimate_booth_multiplier",
    "estimate_bfp",
    "estimate_memory",
    "estimate_spectral_multiply",
    "estimate_modrelu",
    "estimate_all",
    "ChipParams",
]


# ──────────────────────────────────────────────────────────────────────────
# Chip parameters (must match rtl/spectral_mixer.v / spectral_mixer_v2.v)
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class ChipParams:
    """Parameters governing the area estimation formulas.

    Parameters
    ----------
    n_fft : int
        FFT size (sequence length), default 256.
    d_channels : int
        Number of channels, default 64.
    n_modes : int
        Number of retained spectral modes, default 32.
    block_size : int
        Block-diagonal block size, default 8.
    width : int
        Data path width in bits (Q8.8 → 16), default 16.
    """

    n_fft: int = 256
    d_channels: int = 64
    n_modes: int = 32
    block_size: int = 8
    width: int = 16


# ──────────────────────────────────────────────────────────────────────────
# Base gate-count constants (sky130A gate-equivalent units)
# ──────────────────────────────────────────────────────────────────────────

# Per-gate costs (GE) — calibrated to sky130_fd_sc_hd typical cell
GATE_PER_LUT2 = 1.0       # 2-input LUT equivalent
ADDER_GATE = 3.0           # full adder (sum + carry)
REG_GATE = 4.0             # flip-flop (DFF)
MUX2_GATE = 1.5            # 2:1 mux
XOR_GATE = 1.0
COMPARATOR_BIT = 2.0       # per-bit comparator
SHIFTER_BIT = 2.5          # per-bit barrel shifter
MULT_BIT = 8.0             # per-bit multiplier slice (array)

# Booth radix-4 partial product: encoder + selector mux per radix-4 group
BOOTH_PP_GATE = 20.0       # gates per partial product
BOOTH_CSA_GATE = 14.0      # carry-save adder per bit-slice


# ──────────────────────────────────────────────────────────────────────────
# Estimation formulas
# ──────────────────────────────────────────────────────────────────────────


def estimate_booth_multiplier(width: int) -> Dict[str, float]:
    """Estimate gate count for a Booth-encoded radix-4 multiplier.

    Formula::

        partial_products = WIDTH / 2
        booth_gates = partial_products * (pp_gates + csa_gates)

    Parameters
    ----------
    width : int
        Operand width in bits.

    Returns
    -------
    dict
        Breakdown of gate counts.
    """
    partial_products = width // 2
    pp_gates = partial_products * BOOTH_PP_GATE
    csa_gates = partial_products * BOOTH_CSA_GATE
    # Final carry-propagate adder (ripple carry)
    final_adder = width * ADDER_GATE
    total = pp_gates + csa_gates + final_adder
    return {
        "partial_products": partial_products,
        "pp_gates": pp_gates,
        "csa_gates": csa_gates,
        "final_adder": final_adder,
        "total": total,
    }


def estimate_complex_mult(width: int, use_booth: bool = True) -> Dict[str, float]:
    """Estimate gate count for a complex multiplier (a+bi)*(c+di).

    A complex multiply needs 4 real multiplies + 2 add/subtract,
    or 3 real multiplies + 5 add/subtract (Karatsuba).
    We use the 4-multiply form with Booth multipliers.

    Parameters
    ----------
    width : int
        Operand width in bits.
    use_booth : bool
        If True, use Booth-encoded multipliers; else standard array multipliers.

    Returns
    -------
    dict
        Breakdown of gate counts.
    """
    if use_booth:
        real_mult = estimate_booth_multiplier(width)
        mult_gates = 4 * real_mult["total"]
    else:
        mult_gates = 4 * width * MULT_BIT
    # 2 add/subtract operations (real and imaginary parts)
    adder_gates = 2 * width * ADDER_GATE
    total = mult_gates + adder_gates
    return {
        "mult_gates": mult_gates,
        "adder_gates": adder_gates,
        "total": total,
    }


def estimate_fft(
    n: int,
    width: int,
    use_booth: bool = True,
    use_bfp: bool = False,
    pipeline_stages: int = 4,
) -> Dict[str, float]:
    """Estimate gate count for an N-point radix-4 FFT engine.

    Formula::

        stages = log4(N) = log2(N) / 2
        butterfly = 3 complex mults + 8 adds + registers
        FFT_total = stages * butterfly + twiddle + memory

    Parameters
    ----------
    n : int
        FFT size (must be power of 4 for radix-4).
    width : int
        Data path width in bits.
    use_booth : bool
        Use Booth-encoded complex multipliers.
    use_bfp : bool
        Include BFP scaling logic.
    pipeline_stages : int
        Number of pipeline stages (4 for v1/v2, 8 for v3 deep pipeline).

    Returns
    -------
    dict
        Breakdown of gate counts.
    """
    stages = int(math.log2(n) / 2)  # log4(N)

    # Radix-4 butterfly: 3 complex multiplies + 8 add/subtract + registers
    cmplx_mult = estimate_complex_mult(width, use_booth)
    butterfly_mult = 3 * cmplx_mult["total"]
    butterfly_adds = 8 * width * ADDER_GATE
    butterfly_regs = 8 * width * 2 * REG_GATE  # real+imag, 8 outputs
    butterfly_total = butterfly_mult + butterfly_adds + butterfly_regs

    # Twiddle factors
    if use_booth:
        # CORDIC + symmetry: only 16 entries for 256-pt FFT
        twiddle_entries = max(n // 16, 16)
        twiddle_gates = twiddle_entries * width * 2 * REG_GATE * 0.25  # compressed
    else:
        # Full ROM: N/4 entries × 2 (sin+cos) × width bits
        twiddle_entries = n // 4
        twiddle_gates = twiddle_entries * width * 2 * REG_GATE

    # Memory: N complex samples × 2 (real+imag) × width bits
    # Dual-port SRAM model: bits * per_cell_overhead
    memory_bits = n * width * 2 * 2  # input + output buffers
    per_cell_overhead = 4.0
    memory_gates = memory_bits / 1 * per_cell_overhead

    # Pipeline registers (extra for deeper pipeline)
    pipeline_reg_gates = pipeline_stages * width * 2 * REG_GATE * 2

    # BFP scaling logic
    bfp_gates = 0.0
    if use_bfp:
        bfp = estimate_bfp(n, width, block_size=64)
        bfp_gates = bfp["total"]

    total = (
        stages * butterfly_total
        + twiddle_gates
        + memory_gates
        + pipeline_reg_gates
        + bfp_gates
    )

    return {
        "n": n,
        "stages": stages,
        "pipeline_stages": pipeline_stages,
        "butterfly_total": butterfly_total,
        "butterfly_mult": butterfly_mult,
        "butterfly_adds": butterfly_adds,
        "butterfly_regs": butterfly_regs,
        "twiddle_gates": twiddle_gates,
        "memory_gates": memory_gates,
        "pipeline_reg_gates": pipeline_reg_gates,
        "bfp_gates": bfp_gates,
        "total": total,
    }


def estimate_bfp(n: int, width: int, block_size: int = 64) -> Dict[str, float]:
    """Estimate gate count for Block Floating-Point scaling logic.

    Formula::

        comparator_gates = log2(block_size) * COMPARATOR_BIT * width
        shifter_gates = block_size * SHIFTER_BIT * width
        BFP_total = (N / block_size) * (comparator_gates + shifter_gates) + exp_regs

    Parameters
    ----------
    n : int
        Number of samples.
    width : int
        Mantissa width in bits.
    block_size : int
        BFP block size (samples sharing an exponent).

    Returns
    -------
    dict
        Breakdown of gate counts.
    """
    n_blocks = max(n // block_size, 1)
    log_block = int(math.log2(max(block_size, 2)))

    # OR-tree comparator to find max magnitude in block
    comparator_gates = log_block * COMPARATOR_BIT * width
    # Barrel shifters to normalize each sample in the block
    shifter_gates = block_size * SHIFTER_BIT * width
    # Exponent registers (4-bit per block)
    exp_regs = n_blocks * 4 * REG_GATE

    per_block = comparator_gates + shifter_gates
    total = n_blocks * per_block + exp_regs

    return {
        "n_blocks": n_blocks,
        "log_block": log_block,
        "comparator_gates": comparator_gates,
        "shifter_gates": shifter_gates,
        "exp_regs": exp_regs,
        "per_block": per_block,
        "total": total,
    }


def estimate_memory(bits: int, bits_per_cell: int = 1, per_cell_overhead: float = 4.0) -> Dict[str, float]:
    """Estimate gate count for a memory block.

    Formula::

        Memory_total = bits / bits_per_cell * per_cell_overhead

    Parameters
    ----------
    bits : int
        Total memory capacity in bits.
    bits_per_cell : int
        Bits per memory cell (1 for 1T1R SRAM).
    per_cell_overhead : float
        Overhead gates per cell (sense amp + write circuitry, amortized).

    Returns
    -------
    dict
        Breakdown of gate counts.
    """
    cells = bits / bits_per_cell
    total = cells * per_cell_overhead
    return {
        "bits": bits,
        "cells": cells,
        "per_cell_overhead": per_cell_overhead,
        "total": total,
    }


def estimate_spectral_multiply(
    n_modes: int,
    block_size: int,
    width: int,
    serialized: bool = True,
    use_booth: bool = True,
    n_channels: int = 1,
    interleaved: bool = False,
) -> Dict[str, float]:
    """Estimate gate count for the spectral weight multiply unit.

    Parameters
    ----------
    n_modes : int
        Number of spectral modes (K).
    block_size : int
        Block-diagonal block size.
    width : int
        Data path width in bits.
    serialized : bool
        If True, channels are serialized (1 MAC unit). If False, all
        D channels are processed in parallel.
    use_booth : bool
        Use Booth-encoded multipliers.
    n_channels : int
        Number of channels (for parallel mode).
    interleaved : bool
        If True, mode interleaving doubles throughput (2 pipeline stages).

    Returns
    -------
    dict
        Breakdown of gate counts.
    """
    # Per-mode complex multiply: 2 real multipliers (real and imag parts)
    if use_booth:
        real_mult = estimate_booth_multiplier(width)
        per_mode_mult = 2 * real_mult["total"]
    else:
        per_mode_mult = 2 * width * MULT_BIT

    # Accumulator (carry-save for v3, carry-propagate for v1/v2)
    accumulator_gates = n_modes * width * REG_GATE

    # Weight registers: K modes × 2 (re+im) × width
    weight_regs = n_modes * 2 * width * REG_GATE

    # Block-diagonal: divide by block_size (shared weights)
    block_factor = max(block_size, 1)

    base_mult = (n_modes * per_mode_mult) / block_factor + accumulator_gates + weight_regs

    if interleaved:
        # Mode interleaving: 2 pipeline stages, but only extra accumulator regs
        base_mult = base_mult + n_modes * width * REG_GATE  # extra accumulator set

    if serialized:
        total = base_mult
    else:
        total = base_mult * n_channels

    return {
        "n_modes": n_modes,
        "block_size": block_size,
        "per_mode_mult": per_mode_mult,
        "accumulator_gates": accumulator_gates,
        "weight_regs": weight_regs,
        "serialized": serialized,
        "interleaved": interleaved,
        "base_mult": base_mult,
        "total": total,
    }


def estimate_modrelu(width: int, merged: bool = False) -> Dict[str, float]:
    """Estimate gate count for modReLU + soft-threshold activation.

    Parameters
    ----------
    width : int
        Data path width in bits.
    merged : bool
        If True, soft-threshold and modReLU are fused (v2/v3).

    Returns
    -------
    dict
        Breakdown of gate counts.
    """
    # Magnitude computation: |z| = sqrt(re² + im²) ≈ approximation
    magnitude_gates = 2 * width * MULT_BIT * 0.5  # simplified magnitude
    magnitude_gates += width * ADDER_GATE  # re² + im² add

    # Comparator for threshold
    comparator_gates = width * COMPARATOR_BIT

    # Bias adder
    bias_adder = width * ADDER_GATE

    # Soft-threshold logic (subtract threshold or zero)
    soft_threshold_gates = width * MUX2_GATE + width * ADDER_GATE

    if merged:
        # Fused: saves one magnitude computation and one register stage
        total = magnitude_gates + comparator_gates + bias_adder + soft_threshold_gates
        total *= 0.7  # ~30% savings from fusion
    else:
        # Separate: magnitude + compare + modReLU + separate soft-threshold
        modrelu_logic = width * MUX2_GATE + width * ADDER_GATE
        total = magnitude_gates + comparator_gates + bias_adder + soft_threshold_gates + modrelu_logic

    return {
        "magnitude_gates": magnitude_gates,
        "comparator_gates": comparator_gates,
        "bias_adder": bias_adder,
        "soft_threshold_gates": soft_threshold_gates,
        "merged": merged,
        "total": total,
    }


def estimate_carry_save_acc(width: int, n_modes: int) -> Dict[str, float]:
    """Estimate gate count for a carry-save accumulator (v3 improvement #3).

    Parameters
    ----------
    width : int
        Data path width in bits.
    n_modes : int
        Number of accumulation steps (modes).

    Returns
    -------
    dict
        Breakdown of gate counts.
    """
    # Carry-save: 2 registers per bit (sum + carry) + CSA per accumulation
    csa_per_bit = 2 * ADDER_GATE  # full adder as CSA
    csa_total = width * csa_per_bit
    # Registers for sum and carry vectors
    reg_gates = 2 * width * REG_GATE
    # Final carry-propagate adder (ripple carry)
    final_adder = width * ADDER_GATE

    total = csa_total + reg_gates + final_adder
    return {
        "csa_total": csa_total,
        "reg_gates": reg_gates,
        "final_adder": final_adder,
        "total": total,
    }


def estimate_fma_butterfly(width: int) -> Dict[str, float]:
    """Estimate gate count for a FMA butterfly (v3 improvement #4).

    Fused multiply-add: (a + W*b) and (a - W*b) as fused operations.

    Parameters
    ----------
    width : int
        Data path width in bits.

    Returns
    -------
    dict
        Breakdown of gate counts.
    """
    # 2 FMA units (for + and - butterfly outputs)
    real_mult = estimate_booth_multiplier(width)
    # FMA saves the intermediate register between mult and add
    fma_unit = real_mult["total"] + width * ADDER_GATE  # mult + add
    # 2 complex FMAs = 4 real FMAs (2 for + branch, 2 for - branch)
    # But - branch reuses the multiplier, just negates the add
    total = 3 * real_mult["total"] + 4 * width * ADDER_GATE
    # Savings: no intermediate register stage
    saved_regs = 2 * width * REG_GATE
    total -= saved_regs

    return {
        "fma_unit": fma_unit,
        "saved_regs": saved_regs,
        "total": max(total, 0),
    }


def estimate_dual_channel(base_datapath_gates: float) -> Dict[str, float]:
    """Estimate additional gate count for dual-channel datapath (v3 #17).

    Parameters
    ----------
    base_datapath_gates : float
        Gate count of a single-channel datapath.

    Returns
    -------
    dict
        Breakdown of gate counts.
    """
    # Second channel: full datapath duplicate, but shared weight storage
    second_channel = base_datapath_gates
    # Shared weight registers (no duplication)
    shared_weights_saved = 0  # already not counted in datapath

    return {
        "base_datapath": base_datapath_gates,
        "second_channel": second_channel,
        "shared_weights_saved": shared_weights_saved,
        "total": second_channel,
    }


def estimate_security_modules() -> Dict[str, Dict[str, Any]]:
    """Estimate gate counts for all 10 security modules.

    Returns
    -------
    dict
        Per-module gate count breakdown (each entry has 'description' and 'total').
    """
    return {
        "weight_crypto": {
            "description": "Trivium stream cipher for weight encryption",
            "total": 1500.0,
        },
        "logic_lock": {
            "description": "32-bit key-gated twiddle MUX",
            "total": 800.0,
        },
        "constant_time_mac": {
            "description": "Fixed-cycle MAC control",
            "total": 400.0,
        },
        "power_flattening": {
            "description": "Decoy MAC + LFSR",
            "total": 2000.0,
        },
        "scan_lockout": {
            "description": "Poly-fuse scan chain disconnect",
            "total": 200.0,
        },
        "netlist_obfuscation": {
            "description": "50 dummy filler cells",
            "total": 300.0,
        },
        "integrity_hash": {
            "description": "SHA-256 hash of weight bitstream",
            "total": 2300.0,
        },
        "em_shield": {
            "description": "Metal 6 ground shield (metal fill, ~0 logic gates)",
            "total": 0.0,
        },
        "reproducible_build": {
            "description": "Build manifest (software, no gates)",
            "total": 0.0,
        },
        "split_manufacturing": {
            "description": "Process-level, no RTL gates",
            "total": 0.0,
        },
    }


def estimate_dma_controller(width: int = 32) -> Dict[str, float]:
    """Estimate gate count for DMA burst controller (v3 #8).

    Parameters
    ----------
    width : int
        Bus width in bits.

    Returns
    -------
    dict
        Breakdown of gate counts.
    """
    # Address counter, burst counter, FIFO buffer
    addr_counter = 16 * REG_GATE
    burst_counter = 8 * REG_GATE
    fifo = 4 * width * REG_GATE  # 4-word burst buffer
    control_logic = 200.0  # state machine

    total = addr_counter + burst_counter + fifo + control_logic
    return {
        "addr_counter": addr_counter,
        "burst_counter": burst_counter,
        "fifo": fifo,
        "control_logic": control_logic,
        "total": total,
    }


def estimate_bit_reversal_router(n: int, width: int) -> Dict[str, float]:
    """Estimate gate count for bit-reversal permutation router (v3 #10).

    Parameters
    ----------
    n : int
        FFT size.
    width : int
        Data path width in bits.

    Returns
    -------
    dict
        Breakdown of gate counts.
    """
    log_n = int(math.log2(max(n, 2)))
    # Crossbar: log_n address bits × width data bits × MUX2
    mux_gates = log_n * width * MUX2_GATE
    # Address permutation logic (just wiring, ~0 gates)
    total = mux_gates
    return {
        "log_n": log_n,
        "mux_gates": mux_gates,
        "total": total,
    }


def estimate_dvfs_controller() -> Dict[str, float]:
    """Estimate gate count for secure DVFS controller (v3 #20).

    Returns
    -------
    dict
        Breakdown of gate counts.
    """
    # Voltage tracker state machine, voltage sensor comparator, safety timer
    state_machine = 150.0
    voltage_comparator = 8 * COMPARATOR_BIT
    safety_timer = 16 * REG_GATE
    clock_divider = 4 * REG_GATE + 2 * XOR_GATE

    total = state_machine + voltage_comparator + safety_timer + clock_divider
    return {
        "state_machine": state_machine,
        "voltage_comparator": voltage_comparator,
        "safety_timer": safety_timer,
        "clock_divider": clock_divider,
        "total": total,
    }


def estimate_shadow_weight_regs(n_modes: int, width: int) -> Dict[str, float]:
    """Estimate gate count for shadow weight register file (v3 #7).

    Parameters
    ----------
    n_modes : int
        Number of spectral modes.
    width : int
        Data path width in bits.

    Returns
    -------
    dict
        Breakdown of gate counts.
    """
    # Shadow regs: same size as weight regs, plus swap MUX
    shadow_regs = n_modes * 2 * width * REG_GATE
    swap_mux = n_modes * 2 * width * MUX2_GATE
    total = shadow_regs + swap_mux
    return {
        "shadow_regs": shadow_regs,
        "swap_mux": swap_mux,
        "total": total,
    }


# ──────────────────────────────────────────────────────────────────────────
# Full architecture estimation
# ──────────────────────────────────────────────────────────────────────────


def estimate_all(params: ChipParams) -> Dict[str, Any]:
    """Estimate gate counts for all RTL modules across v1, v2, and v3.

    Parameters
    ----------
    params : ChipParams
        Chip parameters.

    Returns
    -------
    dict
        Complete estimation results with per-module and per-version totals.
    """
    N = params.n_fft
    D = params.d_channels
    K = params.n_modes
    B = params.block_size
    W = params.width

    results: Dict[str, Any] = {
        "chip_params": {
            "N_FFT": N,
            "D_CHANNELS": D,
            "N_MODES": K,
            "BLOCK_SIZE": B,
            "WIDTH": W,
        },
    }

    # ── Common module estimates ──────────────────────────────────────────

    fft_v1 = estimate_fft(N, W, use_booth=False, use_bfp=False, pipeline_stages=4)
    fft_v2 = estimate_fft(N, W, use_booth=False, use_bfp=False, pipeline_stages=4)
    fft_v3 = estimate_fft(N, W, use_booth=True, use_bfp=True, pipeline_stages=8)

    sm_v1 = estimate_spectral_multiply(K, B, W, serialized=False, use_booth=False, n_channels=D)
    sm_v2 = estimate_spectral_multiply(K, B, W, serialized=True, use_booth=False)
    sm_v3 = estimate_spectral_multiply(K, B, W, serialized=True, use_booth=True, interleaved=True)

    modrelu_v1 = estimate_modrelu(W, merged=False)
    modrelu_v2 = estimate_modrelu(W, merged=True)
    modrelu_v3 = estimate_modrelu(W, merged=True)

    security = estimate_security_modules()
    security_total = sum(m["total"] for m in security.values())

    # v3-specific modules
    csa_acc = estimate_carry_save_acc(W, K)
    fma_butterfly = estimate_fma_butterfly(W)
    dma_ctrl = estimate_dma_controller()
    bit_rev = estimate_bit_reversal_router(N, W)
    dvfs = estimate_dvfs_controller()
    shadow_regs = estimate_shadow_weight_regs(K, W)
    booth_mult = estimate_booth_multiplier(W)

    # ── v1: Separate FFT/IFFT, parallel channels, no security ────────────

    v1_modules: Dict[str, float] = {
        "fft_256": fft_v1["total"],
        "ifft_256": fft_v1["total"],  # dedicated IFFT = same cost as FFT
        "spectral_multiply": sm_v1["total"],
        "modrelu_soft_threshold": modrelu_v1["total"],
        "twiddle_rom": fft_v1["twiddle_gates"],
        "memory_buffers": fft_v1["memory_gates"],
        "control_wishbone": 500.0,
        "tt_wrapper": 200.0,
    }
    v1_total = sum(v1_modules.values())

    # ── v2: Shared FFT/IFFT, serialized, 10 security modules ─────────────

    v2_modules: Dict[str, float] = {
        "fft_ifft_256 (shared)": fft_v2["total"],
        "spectral_multiply (serialized)": sm_v2["total"],
        "modrelu_soft_threshold (merged)": modrelu_v2["total"],
        "twiddle_rom": fft_v2["twiddle_gates"],
        "memory_buffers (in-place)": fft_v2["memory_gates"] * 0.5,  # in-place saves
        "control_wishbone": 500.0,
        "tt_wrapper_v2": 200.0,
        # Security modules (10)
        **{f"security/{k}": v["total"] for k, v in security.items()},
    }
    v2_total = sum(v2_modules.values())

    # ── v3: All v2 + 20 performance modules ──────────────────────────────

    v3_base: Dict[str, float] = {
        "fft_ifft_256 (shared, deep pipeline)": fft_v3["total"],
        "spectral_multiply (interleaved)": sm_v3["total"],
        "modrelu_soft_threshold (merged)": modrelu_v3["total"],
        "twiddle (CORDIC + symmetry)": fft_v3["twiddle_gates"],
        "memory (ping-pong dual-bank)": fft_v3["memory_gates"] * 0.75,  # 2 banks but smaller
        "control_wishbone": 500.0,
        "tt_wrapper_v2": 200.0,
        # Security modules (10, preserved)
        **{f"security/{k}": v["total"] for k, v in security.items()},
    }

    v3_perf_modules: Dict[str, float] = {
        # Datapath arithmetic (1-5)
        "booth_radix4_mult (#1)": booth_mult["total"],
        "bfp_stage (#2)": fft_v3["bfp_gates"],
        "carry_save_acc (#3)": csa_acc["total"],
        "fma_butterfly (#4)": fma_butterfly["total"],
        "truncated_booth_twiddle (#5)": booth_mult["total"] * 0.7,  # 30% smaller
        # Memory & data movement (6-10)
        "pingpong_buf (#6)": fft_v3["memory_gates"] * 0.25,  # extra bank
        "shadow_weight_reg (#7)": shadow_regs["total"],
        "wishbone_dma (#8)": dma_ctrl["total"],
        "conflict_free_addr (#9)": 150.0,  # address logic only
        "bit_reversal_router (#10)": bit_rev["total"],
        # Algorithmic (11-15)
        "rfft_256 (#11)": -fft_v3["total"] * 0.5,  # halves FFT (negative = savings)
        "twiddle_symmetry (#12)": -fft_v3["twiddle_gates"] * 0.75,  # 4× compression (savings)
        "zero_skip_dummy (#13)": 300.0,  # LFSR reuse, minimal extra logic
        "mode_interleave_mac (#14)": K * W * REG_GATE,  # extra accumulator set
        "adaptive_mode_cnt (#15)": 200.0,  # mode count register + logic
        # Pipeline & throughput (16-20)
        "deep_pipeline_fft8 (#16)": fft_v3["pipeline_reg_gates"] - fft_v1["pipeline_reg_gates"],  # extra stages
        "dual_channel_datapath (#17)": fft_v3["total"] + sm_v3["total"],  # second channel
        "early_ifft_overlap (#18)": 150.0,  # scheduling control logic
        "configurable_fft (#19)": 300.0,  # FFT size MUX + control
        "dvfs_secure (#20)": dvfs["total"],
    }

    v3_modules = {**v3_base, **v3_perf_modules}
    v3_total = sum(v3_modules.values())

    # ── Assemble results ─────────────────────────────────────────────────

    results["v1"] = {
        "description": "Separate FFT/IFFT, parallel channels, no security",
        "modules": v1_modules,
        "total_gates": v1_total,
        "estimated_area_mm2": v1_total / 40000,  # ~40K GE per mm² at sky130
    }
    results["v2"] = {
        "description": "Shared FFT/IFFT, serialized channels, 10 security modules",
        "modules": v2_modules,
        "total_gates": v2_total,
        "estimated_area_mm2": v2_total / 40000,
    }
    results["v3"] = {
        "description": "Shared FFT/IFFT + 20 performance modules, dual-channel, deep pipeline",
        "modules": v3_modules,
        "total_gates": v3_total,
        "estimated_area_mm2": v3_total / 40000,
    }

    # Comparison
    results["comparison"] = {
        "v2_vs_v1_area_reduction_pct": (1 - v2_total / v1_total) * 100 if v1_total else 0,
        "v3_vs_v2_area_increase_pct": (v3_total / v2_total - 1) * 100 if v2_total else 0,
        "v3_vs_v1_area_ratio": v3_total / v1_total if v1_total else 0,
        "v1_total_gates": v1_total,
        "v2_total_gates": v2_total,
        "v3_total_gates": v3_total,
    }

    return results


# ──────────────────────────────────────────────────────────────────────────
# Table printing
# ──────────────────────────────────────────────────────────────────────────


def print_table(results: Dict[str, Any]) -> None:
    """Print a formatted comparison table of v1/v2/v3 area estimates.

    Parameters
    ----------
    results : dict
        Results from :func:`estimate_all`.
    """
    params = results["chip_params"]

    print("=" * 80)
    print("  Spectral Silicon — Synthesis Area Estimation (v1/v2/v3)")
    print("=" * 80)
    print()
    print(f"  Chip parameters: N={params['N_FFT']}, D={params['D_CHANNELS']}, "
          f"K={params['N_MODES']}, block={params['BLOCK_SIZE']}, "
          f"W={params['WIDTH']}")
    print()
    print("-" * 80)
    print(f"  {'Module':<45} {'v1':>10} {'v2':>10} {'v3':>10}")
    print("-" * 80)

    v1_mods = results["v1"]["modules"]
    v2_mods = results["v2"]["modules"]
    v3_mods = results["v3"]["modules"]

    # Collect all unique module names
    all_modules = sorted(set(list(v1_mods.keys()) + list(v2_mods.keys()) + list(v3_mods.keys())))

    for mod in all_modules:
        v1_val = v1_mods.get(mod, 0)
        v2_val = v2_mods.get(mod, 0)
        v3_val = v3_mods.get(mod, 0)
        if v1_val == 0 and v2_val == 0 and v3_val == 0:
            continue
        print(f"  {mod:<45} {v1_val:>10.0f} {v2_val:>10.0f} {v3_val:>10.0f}")

    print("-" * 80)
    v1_total = results["v1"]["total_gates"]
    v2_total = results["v2"]["total_gates"]
    v3_total = results["v3"]["total_gates"]
    print(f"  {'TOTAL GATES':<45} {v1_total:>10.0f} {v2_total:>10.0f} {v3_total:>10.0f}")
    print(f"  {'ESTIMATED AREA (mm²)':<45} "
          f"{results['v1']['estimated_area_mm2']:>10.3f} "
          f"{results['v2']['estimated_area_mm2']:>10.3f} "
          f"{results['v3']['estimated_area_mm2']:>10.3f}")
    print("-" * 80)
    print()

    comp = results["comparison"]
    print("  COMPARISON:")
    print(f"    v2 vs v1 area reduction:  {comp['v2_vs_v1_area_reduction_pct']:.1f}%")
    print(f"    v3 vs v2 area increase:   {comp['v3_vs_v2_area_increase_pct']:.1f}%")
    print(f"    v3 vs v1 area ratio:      {comp['v3_vs_v1_area_ratio']:.2f}x")
    print()
    print("  NOTE: These are analytical estimates, not synthesis results.")
    print("  Gate counts are in gate-equivalent (GE) units for sky130A.")
    print("  Area assumes ~40K GE/mm² at sky130_fd_sc_hd typical density.")
    print("=" * 80)


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Estimate synthesis gate counts for Spectral Silicon v1/v2/v3.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Print formatted table
  python scripts/synthesis_area_estimate.py

  # Output JSON
  python scripts/synthesis_area_estimate.py --json

  # Custom parameters
  python scripts/synthesis_area_estimate.py --width 16 --n-fft 256 --n-modes 32
""",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of a formatted table.",
    )
    parser.add_argument(
        "--n-fft", type=int, default=256,
        help="FFT size (default: 256)",
    )
    parser.add_argument(
        "--d-channels", type=int, default=64,
        help="Number of channels (default: 64)",
    )
    parser.add_argument(
        "--n-modes", type=int, default=32,
        help="Number of spectral modes (default: 32)",
    )
    parser.add_argument(
        "--block-size", type=int, default=8,
        help="Block-diagonal block size (default: 8)",
    )
    parser.add_argument(
        "--width", type=int, default=16,
        help="Data path width in bits (default: 16)",
    )
    parser.add_argument(
        "--indent", type=int, default=2,
        help="JSON indentation level (default: 2)",
    )
    args = parser.parse_args()

    params = ChipParams(
        n_fft=args.n_fft,
        d_channels=args.d_channels,
        n_modes=args.n_modes,
        block_size=args.block_size,
        width=args.width,
    )

    results = estimate_all(params)

    if args.json:
        print(json.dumps(results, indent=args.indent))
    else:
        print_table(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())