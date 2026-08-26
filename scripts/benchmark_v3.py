#!/usr/bin/env python3
"""V3 architecture benchmark: v1 vs v2 vs v3 across area, power, throughput, latency.

This script compares three spectral-mixer architecture generations:

  **v1** — Basic (separate FFT/IFFT, parallel channels, no security):
    - Dedicated FFT and IFFT engines
    - All D channels processed in parallel
    - No clock gating, no security modules
    - 50 MHz target frequency

  **v2** — Efficiency (shared FFT/IFFT + security):
    - Single shared FFT engine (IFFT via conjugate-FFT)
    - Clock gating on inactive pipeline stages
    - Serialized channel processing
    - All 10 security measures from IMPROVEMENTS.md
    - 50 MHz target frequency

  **v3** — Performance (all 20 improvements):
    - All v2 efficiency + security improvements
    - Booth radix-4 multipliers, BFP, carry-save, FMA butterfly
    - Ping-pong buffers, shadow prefetch, conflict-free addressing
    - RFFT, twiddle symmetry, mode interleaving, early IFFT
    - Deep 8-stage pipeline at 80 MHz
    - Dual-channel parallel processing, configurable FFT, DVFS
    - Zero-skip MAC with constant timing + power reduction

Usage
-----
    PYTHONPATH=. python scripts/benchmark_v3.py
    PYTHONPATH=. python scripts/benchmark_v3.py --json
    PYTHONPATH=. python scripts/benchmark_v3.py --json > results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict

import os

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from spectral_silicon.perf_sim import PerfChipV3

__all__ = ["benchmark_v3", "print_comparison_table", "main"]


# ──────────────────────────────────────────────────────────────────────────
# Benchmark runner
# ──────────────────────────────────────────────────────────────────────────


def benchmark_v3() -> Dict[str, Any]:
    """Run the full v1/v2/v3 comparison benchmark.

    Returns
    -------
    dict
        Complete benchmark results including area, power, throughput,
        and latency for all three architectures, plus improvement ratios.
    """
    chip = PerfChipV3()

    area = chip.estimate_area()
    power = chip.estimate_power()
    throughput = chip.estimate_throughput()

    # Latency estimation (cycles for one token at default k=32, N=256)
    latency: Dict[str, Dict[str, float]] = {}
    for ver in ["v1", "v2", "v3"]:
        if ver == "v1":
            # Separate FFT+IFFT, D channels parallel, no RFFT
            fft = 256
            spectral = 32
            ifft = 256
            total = fft + spectral + ifft
        elif ver == "v2":
            # Shared FFT (sequential), serialized channels
            fft = 256
            spectral = 32
            ifft = 256
            per_channel = fft + spectral + ifft
            total = per_channel * 64  # 64 channels serialized
        else:  # v3
            # RFFT (129 modes), interleaved spectral (16), early IFFT overlap
            fft = 129
            spectral = 16
            ifft = 256
            overlap = max(0, ifft - (fft - 32))
            per_channel = fft + spectral + ifft - overlap
            total = per_channel * 64 / 2  # dual channel
        latency[ver] = {
            "fft_cycles": fft,
            "spectral_cycles": spectral,
            "ifft_cycles": ifft,
            "total_cycles": total,
        }

    # Improvement ratios
    v1_area = area["v1"]["total"]
    v2_area = area["v2"]["total"]
    v3_area = area["v3"]["total"]

    v1_pwr = power["v1"]["total"]
    v2_pwr = power["v2"]["total"]
    v3_pwr = power["v3"]["total"]

    v1_tp = throughput["v1"]["max"]
    v2_tp = throughput["v2"]["max"]
    v3_tp = throughput["v3"]["max"]

    v1_lat = latency["v1"]["total_cycles"]
    v2_lat = latency["v2"]["total_cycles"]
    v3_lat = latency["v3"]["total_cycles"]

    # Note: v1 has D=64 channels in parallel → very high throughput at huge
    # area/power.  v3 improves throughput *relative to v2* (same serialized
    # architecture + all 20 perf improvements).

    comparison = {
        "area": {
            "v2_vs_v1_reduction_pct": (1 - v2_area / v1_area) * 100 if v1_area else 0,
            "v3_vs_v1_reduction_pct": (1 - v3_area / v1_area) * 100 if v1_area else 0,
            "v3_vs_v2_reduction_pct": (1 - v3_area / v2_area) * 100 if v2_area else 0,
        },
        "power": {
            "v2_vs_v1_reduction_pct": (1 - v2_pwr / v1_pwr) * 100 if v1_pwr else 0,
            "v3_vs_v1_reduction_pct": (1 - v3_pwr / v1_pwr) * 100 if v1_pwr else 0,
            "v3_vs_v2_reduction_pct": (1 - v3_pwr / v2_pwr) * 100 if v2_pwr else 0,
        },
        "throughput": {
            "v3_vs_v1_improvement": v3_tp / v1_tp if v1_tp else 0,
            "v3_vs_v2_improvement": v3_tp / v2_tp if v2_tp else 0,
        },
        "latency": {
            "v2_vs_v1_change_pct": (v2_lat / v1_lat - 1) * 100 if v1_lat else 0,
            "v3_vs_v1_change_pct": (v3_lat / v1_lat - 1) * 100 if v1_lat else 0,
            "v3_vs_v2_change_pct": (v3_lat / v2_lat - 1) * 100 if v2_lat else 0,
        },
    }

    return {
        "chip_params": {
            "N_FFT": 256,
            "D_CHANNELS": 64,
            "N_MODES": 32,
            "BLOCK_SIZE": 8,
            "WIDTH": 16,
            "FREQ_V1_MHZ": 50,
            "FREQ_V2_MHZ": 50,
            "FREQ_V3_MHZ": 80,
        },
        "area": area,
        "power": power,
        "throughput": throughput,
        "latency": latency,
        "comparison": comparison,
        "security": chip.verify_security_preserved(),
    }


# ──────────────────────────────────────────────────────────────────────────
# Table printing
# ──────────────────────────────────────────────────────────────────────────


def print_comparison_table(results: Dict[str, Any]) -> None:
    """Print a formatted comparison table of v1/v2/v3.

    Parameters
    ----------
    results : dict
        Results from :func:`benchmark_v3`.
    """
    params = results["chip_params"]
    comp = results["comparison"]

    print("=" * 88)
    print("  Spectral Silicon — V3 Architecture Performance Benchmark")
    print("=" * 88)
    print()
    print(f"  Chip: N={params['N_FFT']}, D={params['D_CHANNELS']}, "
          f"K={params['N_MODES']}, block={params['BLOCK_SIZE']}, "
          f"W={params['WIDTH']}")
    print(f"  Freq: v1={params['FREQ_V1_MHZ']}MHz, "
          f"v2={params['FREQ_V2_MHZ']}MHz, "
          f"v3={params['FREQ_V3_MHZ']}MHz")
    print()
    print("-" * 88)
    print(f"  {'Metric':<32} {'v1':>15} {'v2':>15} {'v3':>15}")
    print("-" * 88)

    # ── Area ──
    print()
    print("  AREA (gate equivalents):")
    area = results["area"]
    all_blocks = set()
    for v in area.values():
        all_blocks.update(k for k in v if k != "total")
    for block in sorted(all_blocks):
        v1v = area["v1"].get(block, 0)
        v2v = area["v2"].get(block, 0)
        v3v = area["v3"].get(block, 0)
        if v1v == 0 and v2v == 0 and v3v == 0:
            continue
        print(f"  {block:<32} {v1v:>15.1f} {v2v:>15.1f} {v3v:>15.1f}")

    v1a = area["v1"]["total"]
    v2a = area["v2"]["total"]
    v3a = area["v3"]["total"]
    print(f"  {'─' * 32} {'─' * 15} {'─' * 15} {'─' * 15}")
    print(f"  {'TOTAL AREA':<32} {v1a:>15.1f} {v2a:>15.1f} {v3a:>15.1f}")
    print(f"  {'Reduction vs v1':<32} {'—':>15} "
          f"{comp['area']['v2_vs_v1_reduction_pct']:>14.1f}% "
          f"{comp['area']['v3_vs_v1_reduction_pct']:>14.1f}%")

    # ── Power ──
    print()
    print("  POWER (mW):")
    power = results["power"]
    all_pwr = set()
    for v in power.values():
        all_pwr.update(k for k in v if k != "total")
    for block in sorted(all_pwr):
        v1v = power["v1"].get(block, 0)
        v2v = power["v2"].get(block, 0)
        v3v = power["v3"].get(block, 0)
        if v1v == 0 and v2v == 0 and v3v == 0:
            continue
        print(f"  {block:<32} {v1v:>15.3f} {v2v:>15.3f} {v3v:>15.3f}")

    v1p = power["v1"]["total"]
    v2p = power["v2"]["total"]
    v3p = power["v3"]["total"]
    print(f"  {'─' * 32} {'─' * 15} {'─' * 15} {'─' * 15}")
    print(f"  {'TOTAL POWER':<32} {v1p:>15.3f} {v2p:>15.3f} {v3p:>15.3f}")
    print(f"  {'Reduction vs v1':<32} {'—':>15} "
          f"{comp['power']['v2_vs_v1_reduction_pct']:>14.1f}% "
          f"{comp['power']['v3_vs_v1_reduction_pct']:>14.1f}%")

    # ── Throughput ──
    print()
    print("  THROUGHPUT (tokens/sec):")
    tp = results["throughput"]
    configs = [k for k in tp["v1"] if k.startswith("k")]
    for cfg in sorted(configs):
        v1t = tp["v1"].get(cfg, 0)
        v2t = tp["v2"].get(cfg, 0)
        v3t = tp["v3"].get(cfg, 0)
        print(f"  {cfg:<32} {v1t:>15.1f} {v2t:>15.1f} {v3t:>15.1f}")
    print(f"  {'─' * 32} {'─' * 15} {'─' * 15} {'─' * 15}")
    print(f"  {'MAX THROUGHPUT':<32} "
          f"{tp['v1']['max']:>15.1f} {tp['v2']['max']:>15.1f} {tp['v3']['max']:>15.1f}")
    print(f"  {'v3 improvement vs v2':<32} {'—':>15} {'—':>15} "
          f"{comp['throughput']['v3_vs_v2_improvement']:>14.2f}x")
    print(f"  {'v3 improvement vs v1':<32} {'—':>15} {'—':>15} "
          f"{comp['throughput']['v3_vs_v1_improvement']:>14.2f}x "
          f"(v1 has 64× parallel channels)")

    # ── Latency ──
    print()
    print("  LATENCY (clock cycles, k=32, N=256):")
    lat = results["latency"]
    for key in ["fft_cycles", "spectral_cycles", "ifft_cycles", "total_cycles"]:
        v1l = lat["v1"][key]
        v2l = lat["v2"][key]
        v3l = lat["v3"][key]
        print(f"  {key:<32} {v1l:>15.0f} {v2l:>15.0f} {v3l:>15.0f}")
    print(f"  {'v3 latency vs v1':<32} {'—':>15} {'—':>15} "
          f"{comp['latency']['v3_vs_v1_change_pct']:>14.1f}%")

    # ── Security ──
    print()
    print("  SECURITY (all 10 measures):")
    sec = results["security"]
    for measure, ok in sec.items():
        status = "✓ PRESERVED" if ok else "✗ VIOLATED"
        print(f"    {measure:<34} {status}")
    all_ok = all(sec.values())
    print(f"  {'All security preserved':<32} {'✓ YES' if all_ok else '✗ NO'}")

    # ── Summary ──
    print()
    print("-" * 88)
    print("  SUMMARY:")
    print(f"    v1 → v2 area reduction:   {comp['area']['v2_vs_v1_reduction_pct']:.1f}%  "
          f"({v1a:.0f} → {v2a:.0f} GE)")
    print(f"    v2 → v3 area change:       {comp['area']['v3_vs_v2_reduction_pct']:.1f}%  "
          f"({v2a:.0f} → {v3a:.0f} GE)")
    print(f"    v1 → v3 area reduction:    {comp['area']['v3_vs_v1_reduction_pct']:.1f}%  "
          f"({v1a:.0f} → {v3a:.0f} GE)")
    print(f"    v1 → v3 power reduction:   {comp['power']['v3_vs_v1_reduction_pct']:.1f}%  "
          f"({v1p:.2f} → {v3p:.2f} mW)")
    print(f"    v2 → v3 throughput:         {comp['throughput']['v3_vs_v2_improvement']:.2f}x  "
          f"({tp['v2']['max']:.0f} → {tp['v3']['max']:.0f} tokens/sec)")
    print(f"    v1 → v3 latency change:     {comp['latency']['v3_vs_v1_change_pct']:.1f}%  "
          f"({lat['v1']['total_cycles']:.0f} → {lat['v3']['total_cycles']:.0f} cycles)")
    print(f"    Security preserved:         {'ALL 10 ✓' if all_ok else 'VIOLATIONS ✗'}")
    print()
    print("  v3 improvements: Booth radix-4, BFP, carry-save, FMA butterfly,")
    print("    truncated Booth, ping-pong buffers, shadow prefetch, zero-skip MAC,")
    print("    conflict-free addressing, bit-reversal router, RFFT, twiddle symmetry,")
    print("    mode interleaving, adaptive k, early IFFT, configurable FFT, DVFS,")
    print("    dual channel, deep pipeline, DMA burst")
    print("=" * 88)


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────


def main() -> int:
    """CLI entry point for the v3 benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark v1/v2/v3 spectral-mixer architecture.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of a formatted table.",
    )
    args = parser.parse_args()

    results = benchmark_v3()

    if args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        print_comparison_table(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())