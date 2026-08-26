#!/usr/bin/env python3
"""V2 architecture efficiency benchmark: v1 (separate FFT/IFFT) vs v2 (shared).

This script compares the estimated area and power of two spectral-mixer
architecture generations:

  **v1** — Separate FFT and IFFT engines:
    - Dedicated 256-point FFT block
    - Dedicated 256-point IFFT block
    - No clock gating
    - Parallel channel processing (all D channels at once)

  **v2** — Shared FFT/IFFT engine + efficiency improvements:
    - Single shared FFT engine (IFFT via conjugate-FFT, as in ifft_256.v)
    - Clock gating on inactive pipeline stages
    - Serialized channel processing (one channel at a time through the
      shared engine, as in spectral_mixer.v)

The estimates are based on the RTL parameters of the actual Spectral
Silicon chip (N=256, D=64, N_MODES=32, BLOCK_SIZE=8, WIDTH=16) and use
first-order ASIC area/power models for the major blocks (butterfly,
memory, twiddle ROM, spectral multiply, control logic).

Usage
-----
    PYTHONPATH=. python scripts/benchmark_v2.py
    PYTHONPATH=. python scripts/benchmark_v2.py --json
    PYTHONPATH=. python scripts/benchmark_v2.py --n-channels 128
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List

# Ensure project root is on the path when run as `PYTHONPATH=. python scripts/...`
import os

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

__all__ = ["benchmark_v2", "print_comparison_table", "ArchitectureSpec"]


# ──────────────────────────────────────────────────────────────────────────
# Chip parameters (must match rtl/spectral_mixer.v)
# ──────────────────────────────────────────────────────────────────────────

# FFT size (sequence length)
N_FFT = 256
# Number of channels
D_CHANNELS = 64
# Number of retained spectral modes
N_MODES = 32
# Block-diagonal block size
BLOCK_SIZE = 8
# Data width (Q8.8)
WIDTH = 16


# ──────────────────────────────────────────────────────────────────────────
# Area / Power estimation model
# ──────────────────────────────────────────────────────────────────────────
#
# These are first-order estimates suitable for architectural comparison.
# The absolute numbers are in arbitrary "gate-equivalent" (GE) units for
# area and mW for power.  The *ratios* between v1 and v2 are what matter.
#
# Base block estimates (GE for area, mW for power):
#   - Radix-4 butterfly (complex, Q8.8): ~400 GE, ~0.3 mW per stage
#   - 256x dual-port RAM (complex, 32-bit): ~2000 GE, ~0.5 mW
#   - Twiddle ROM (256 entries, 32-bit): ~1500 GE, ~0.1 mW
#   - Spectral multiply unit (32 modes, block-diag): ~1200 GE, ~0.4 mW
#   - modReLU: ~200 GE, ~0.05 mW
#   - Control / Wishbone: ~500 GE, ~0.1 mW

# ──────────────────────────────────────────────────────────────────────────


@dataclass
class ArchitectureSpec:
    """Specification of a spectral-mixer architecture generation.

    Parameters
    ----------
    name : str
        Architecture name (e.g. "v1", "v2").
    shared_fft : bool
        True if IFFT is implemented via the conjugate-FFT method (sharing
        the FFT engine).  False means a dedicated IFFT engine.
    clock_gating : bool
        True if inactive pipeline stages are clock-gated.
    serialized_channels : bool
        True if channels are serialized through a single pipeline (v2).
        False means all D channels are processed in parallel (v1).
    """

    name: str
    shared_fft: bool
    clock_gating: bool
    serialized_channels: bool

    # ── area estimation ─────────────────────────────────────────────────

    def estimate_area(self) -> Dict[str, float]:
        """Estimate the area (gate equivalents) of each major block.

        Returns
        -------
        dict
            Mapping of block name to area in GE.
        """
        n_radix4_stages = 4  # 256 = 4^4 → 4 radix-4 stages
        butterfly_ge = 400 * n_radix4_stages  # 1600
        ram_ge = 2000
        twiddle_rom_ge = 1500
        spectral_mult_ge = 1200
        modrelu_ge = 200
        control_ge = 500

        fft_engine = butterfly_ge + ram_ge + twiddle_rom_ge  # 5100

        area: Dict[str, float] = {}

        if self.shared_fft:
            # Single FFT engine reused for both FFT and IFFT
            area["fft_engine"] = fft_engine
            area["ifft_engine"] = 0.0  # no separate IFFT
        else:
            # Dedicated FFT and IFFT engines
            area["fft_engine"] = fft_engine
            area["ifft_engine"] = fft_engine  # same cost as FFT

        area["spectral_multiply"] = spectral_mult_ge
        area["modrelu"] = modrelu_ge
        area["control_wishbone"] = control_ge

        # Clock gating adds a small area overhead (~2% of total logic)
        if self.clock_gating:
            logic_area = (
                area["fft_engine"]
                + area["ifft_engine"]
                + area["spectral_multiply"]
                + area["modrelu"]
                + area["control_wishbone"]
            )
            area["clock_gating_overhead"] = logic_area * 0.02
        else:
            area["clock_gating_overhead"] = 0.0

        # Channel parallelism
        if self.serialized_channels:
            # Single pipeline instance shared across all channels
            area["channel_parallelism_factor"] = 1.0
        else:
            # D parallel pipeline instances
            area["channel_parallelism_factor"] = float(D_CHANNELS)

        total = sum(v for k, v in area.items() if k != "channel_parallelism_factor")
        if self.serialized_channels:
            total_pipeline = total
        else:
            # Multiply pipeline blocks by D channels (but not control)
            shared = area["control_wishbone"] + area["clock_gating_overhead"]
            pipeline = total - shared
            total_pipeline = pipeline * D_CHANNELS + shared

        area["total"] = total_pipeline
        return area

    # ── power estimation ────────────────────────────────────────────────

    def estimate_power(self) -> Dict[str, float]:
        """Estimate the power (mW) of each major block.

        Returns
        -------
        dict
            Mapping of block name to power in mW.
        """
        n_radix4_stages = 4
        butterfly_mw = 0.3 * n_radix4_stages  # 1.2
        ram_mw = 0.5
        twiddle_rom_mw = 0.1
        spectral_mult_mw = 0.4
        modrelu_mw = 0.05
        control_mw = 0.1

        fft_engine = butterfly_mw + ram_mw + twiddle_rom_mw  # 1.8

        power: Dict[str, float] = {}

        if self.shared_fft:
            power["fft_engine"] = fft_engine
            power["ifft_engine"] = 0.0
        else:
            power["fft_engine"] = fft_engine
            power["ifft_engine"] = fft_engine

        power["spectral_multiply"] = spectral_mult_mw
        power["modrelu"] = modrelu_mw
        power["control_wishbone"] = control_mw

        # Clock gating reduces dynamic power on inactive stages by ~30%
        if self.clock_gating:
            logic_power = (
                power["fft_engine"]
                + power["ifft_engine"]
                + power["spectral_multiply"]
                + power["modrelu"]
            )
            power["clock_gating_savings"] = -logic_power * 0.30
        else:
            power["clock_gating_savings"] = 0.0

        # Channel parallelism
        if self.serialized_channels:
            # Serialized: only one channel active at a time → lower power
            total = sum(power.values())
            # But serialized takes D× longer, so energy per inference is similar
            power["total_dynamic"] = total
            power["energy_per_inference"] = total * D_CHANNELS  # D× longer
        else:
            # Parallel: all D channels active simultaneously
            total = sum(power.values())
            shared = power["control_wishbone"]
            pipeline = total - shared
            power["total_dynamic"] = pipeline * D_CHANNELS + shared
            power["energy_per_inference"] = power["total_dynamic"]

        return power

    # ── latency estimation ─────────────────────────────────────────────

    def estimate_latency(self) -> Dict[str, float]:
        """Estimate the latency (clock cycles) for one full inference.

        Returns
        -------
        dict
            Latency metrics in clock cycles.
        """
        # FFT: 4 radix-4 stages, each ~64 cycles → 256 cycles
        fft_cycles = N_FFT  # 256
        ifft_cycles = N_FFT  # 256
        spectral_mult_cycles = N_MODES  # 32
        modrelu_cycles = N_FFT  # 256

        if self.shared_fft:
            # FFT and IFFT share the same engine → sequential
            transform_cycles = fft_cycles + ifft_cycles
        else:
            # Dedicated engines → could overlap, but conservatively sequential
            transform_cycles = fft_cycles + ifft_cycles

        per_channel_cycles = transform_cycles + spectral_mult_cycles + modrelu_cycles

        if self.serialized_channels:
            total_cycles = per_channel_cycles * D_CHANNELS
        else:
            # Parallel: all channels processed at once
            total_cycles = per_channel_cycles

        return {
            "fft_cycles": fft_cycles,
            "ifft_cycles": ifft_cycles,
            "spectral_mult_cycles": spectral_mult_cycles,
            "modrelu_cycles": modrelu_cycles,
            "per_channel_cycles": per_channel_cycles,
            "total_cycles": total_cycles,
        }


# ──────────────────────────────────────────────────────────────────────────
# Benchmark runner
# ──────────────────────────────────────────────────────────────────────────


def benchmark_v2() -> Dict[str, Any]:
    """Run the v1 vs v2 comparison benchmark.

    Returns
    -------
    dict
        Complete benchmark results including area, power, and latency
        for both architectures, plus improvement ratios.
    """
    v1 = ArchitectureSpec(
        name="v1",
        shared_fft=False,
        clock_gating=False,
        serialized_channels=False,
    )
    v2 = ArchitectureSpec(
        name="v2",
        shared_fft=True,
        clock_gating=True,
        serialized_channels=True,
    )

    v1_area = v1.estimate_area()
    v2_area = v2.estimate_area()
    v1_power = v1.estimate_power()
    v2_power = v2.estimate_power()
    v1_latency = v1.estimate_latency()
    v2_latency = v2.estimate_latency()

    # Improvement ratios
    area_reduction = 0.0
    if v1_area["total"] > 0:
        area_reduction = (1 - v2_area["total"] / v1_area["total"]) * 100

    power_reduction = 0.0
    if v1_power["total_dynamic"] > 0:
        power_reduction = (1 - v2_power["total_dynamic"] / v1_power["total_dynamic"]) * 100

    latency_increase = 0.0
    if v1_latency["total_cycles"] > 0:
        latency_increase = (v2_latency["total_cycles"] / v1_latency["total_cycles"] - 1) * 100

    return {
        "chip_params": {
            "N_FFT": N_FFT,
            "D_CHANNELS": D_CHANNELS,
            "N_MODES": N_MODES,
            "BLOCK_SIZE": BLOCK_SIZE,
            "WIDTH": WIDTH,
        },
        "v1": {
            "area": v1_area,
            "power": v1_power,
            "latency": v1_latency,
        },
        "v2": {
            "area": v2_area,
            "power": v2_power,
            "latency": v2_latency,
        },
        "comparison": {
            "area_reduction_pct": area_reduction,
            "power_reduction_pct": power_reduction,
            "latency_increase_pct": latency_increase,
            "area_ratio_v2_over_v1": v2_area["total"] / v1_area["total"] if v1_area["total"] else 0,
            "power_ratio_v2_over_v1": v2_power["total_dynamic"] / v1_power["total_dynamic"] if v1_power["total_dynamic"] else 0,
        },
    }


# ──────────────────────────────────────────────────────────────────────────
# Table printing
# ──────────────────────────────────────────────────────────────────────────


def print_comparison_table(results: Dict[str, Any]) -> None:
    """Print a formatted comparison table of v1 vs v2.

    Parameters
    ----------
    results : dict
        Results from :func:`benchmark_v2`.
    """
    v1 = results["v1"]
    v2 = results["v2"]
    comp = results["comparison"]
    params = results["chip_params"]

    # Header
    print("=" * 72)
    print("  Spectral Silicon — V2 Architecture Efficiency Benchmark")
    print("=" * 72)
    print()
    print(f"  Chip parameters: N={params['N_FFT']}, D={params['D_CHANNELS']}, "
          f"K={params['N_MODES']}, block={params['BLOCK_SIZE']}, "
          f"W={params['WIDTH']}")
    print()
    print("-" * 72)
    print(f"  {'Metric':<30} {'v1 (separate)':>15} {'v2 (shared)':>15} {'Change':>10}")
    print("-" * 72)

    # Area
    print()
    print("  AREA (gate equivalents):")
    for block in ["fft_engine", "ifft_engine", "spectral_multiply",
                   "modrelu", "control_wishbone", "clock_gating_overhead"]:
        v1_val = v1["area"].get(block, 0)
        v2_val = v2["area"].get(block, 0)
        if v1_val > 0 or v2_val > 0:
            change = ""
            if v1_val > 0:
                ratio = v2_val / v1_val
                change = f"{ratio:.2f}x"
            print(f"  {block:<30} {v1_val:>15.1f} {v2_val:>15.1f} {change:>10}")

    v1_total = v1["area"]["total"]
    v2_total = v2["area"]["total"]
    print(f"  {'TOTAL AREA':<30} {v1_total:>15.1f} {v2_total:>15.1f} "
          f"{v2_total/v1_total:.2f}x")
    print(f"  {'AREA REDUCTION':<30} {'':>15} {'':>15} "
          f"{comp['area_reduction_pct']:>9.1f}%")

    # Power
    print()
    print("  POWER (mW):")
    for block in ["fft_engine", "ifft_engine", "spectral_multiply",
                   "modrelu", "control_wishbone", "clock_gating_savings"]:
        v1_val = v1["power"].get(block, 0)
        v2_val = v2["power"].get(block, 0)
        if v1_val != 0 or v2_val != 0:
            change = ""
            if v1_val != 0:
                ratio = v2_val / v1_val
                change = f"{ratio:.2f}x"
            print(f"  {block:<30} {v1_val:>15.3f} {v2_val:>15.3f} {change:>10}")

    v1_pwr = v1["power"]["total_dynamic"]
    v2_pwr = v2["power"]["total_dynamic"]
    print(f"  {'TOTAL POWER':<30} {v1_pwr:>15.3f} {v2_pwr:>15.3f} "
          f"{v2_pwr/v1_pwr:.2f}x")
    print(f"  {'POWER REDUCTION':<30} {'':>15} {'':>15} "
          f"{comp['power_reduction_pct']:>9.1f}%")

    # Latency
    print()
    print("  LATENCY (clock cycles):")
    for key in ["fft_cycles", "ifft_cycles", "spectral_mult_cycles",
                "modrelu_cycles", "per_channel_cycles", "total_cycles"]:
        v1_val = v1["latency"][key]
        v2_val = v2["latency"][key]
        change = ""
        if v1_val > 0:
            ratio = v2_val / v1_val
            change = f"{ratio:.2f}x"
        print(f"  {key:<30} {v1_val:>15.0f} {v2_val:>15.0f} {change:>10}")

    # Summary
    print()
    print("-" * 72)
    print("  SUMMARY:")
    print(f"    Area reduction:     {comp['area_reduction_pct']:.1f}%  "
          f"({v1_total:.0f} → {v2_total:.0f} GE)")
    print(f"    Power reduction:   {comp['power_reduction_pct']:.1f}%  "
          f"({v1_pwr:.2f} → {v2_pwr:.2f} mW)")
    print(f"    Latency increase:  {comp['latency_increase_pct']:.1f}%  "
          f"({v1['latency']['total_cycles']:.0f} → "
          f"{v2['latency']['total_cycles']:.0f} cycles)")
    print()
    print("  v2 improvements: shared FFT/IFFT engine, clock gating, "
          "serialized channels")
    print("  Trade-off: lower area & power at the cost of increased latency")
    print("=" * 72)


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────


def main() -> int:
    """CLI entry point for the v2 benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark v1 vs v2 spectral-mixer architecture.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of a formatted table.",
    )
    args = parser.parse_args()

    results = benchmark_v2()

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_comparison_table(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())