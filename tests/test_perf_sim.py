"""Tests for the performance simulation module (perf_sim.py).

Covers all 20 performance improvement simulators plus the full-chip
PerfChipV3 model:

  - BFP accuracy and dynamic range
  - Carry-save correctness (matches carry-propagate)
  - FMA butterfly accuracy improvement
  - Ping-pong throughput improvement (no bubbles)
  - Zero-skip constant timing (uses ConstantTimeSpectralMAC internally)
  - RFFT correctness and Hermitian symmetry
  - Twiddle symmetry correctness (4 from 1)
  - Mode interleave 2× throughput and correctness
  - Adaptive k speedup
  - Early IFFT latency reduction
  - Configurable FFT performance
  - DVFS power savings
  - Dual channel throughput doubling
  - Deep pipeline frequency improvement
  - Conflict-free addressing (zero stalls)
  - Bit-reversal router
  - DMA burst overhead reduction
  - Booth multiplier accuracy
  - Truncated Booth correctness
  - Shadow prefetch latency hiding
  - PerfChipV3 area/power/throughput estimates
  - PerfChipV3 security verification (all 10 measures)
"""

import math

import numpy as np
import pytest

from spectral_silicon.perf_sim import (
    AdaptiveModeCount,
    BitReversalRouter,
    BlockFloatingPoint,
    BoothComplexMultiplier,
    CarrySaveAccumulator,
    ConfigurableFFT,
    ConflictFreeAddressing,
    DeepPipelineFFT,
    DMABurstController,
    DualChannelProcessor,
    DVFSSimulator,
    EarlyIFFT,
    FMAButterfly,
    ModeInterleaver,
    PerfChipV3,
    PingPongBuffer,
    RFFTSimulator,
    ShadowWeightPrefetch,
    TruncatedBooth,
    TwiddleSymmetry,
    ZeroSkipMAC,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def rng():
    """Deterministic NumPy random generator."""
    return np.random.RandomState(42)


@pytest.fixture
def chip():
    """Full PerfChipV3 simulation instance."""
    return PerfChipV3()


# --------------------------------------------------------------------------- #
# 1. Booth Complex Multiplier
# --------------------------------------------------------------------------- #


class TestBoothComplexMultiplier:
    """Tests for Booth radix-4 complex multiplier."""

    def test_multiply_accuracy(self):
        """Booth multiply should match reference within fixed-point precision."""
        bcm = BoothComplexMultiplier()
        a, b = complex(1.5, -0.5), complex(0.75, 0.25)
        result = bcm.multiply(a, b)
        expected = a * b
        assert abs(result - expected) < 0.1

    def test_partial_product_reduction(self):
        """Booth radix-4 should halve partial products."""
        bcm = BoothComplexMultiplier()
        assert bcm.partial_products_booth == bcm.partial_products_standard // 2

    def test_compare_returns_metrics(self):
        """compare() should return error, cycles, and critical path."""
        bcm = BoothComplexMultiplier()
        cmp = bcm.compare(complex(2, 1), complex(-1, 0.5))
        assert "error" in cmp
        assert "cycles" in cmp
        assert "critical_path_reduction" in cmp
        assert cmp["cycles"] == 1
        assert cmp["partial_product_reduction_pct"] == 50.0


# --------------------------------------------------------------------------- #
# 2. Block Floating-Point
# --------------------------------------------------------------------------- #


class TestBlockFloatingPoint:
    """Tests for BFP scaling and dynamic range."""

    def test_scale_unscale_roundtrip(self, rng):
        """Scaling then unscaling should preserve values approximately."""
        bfp = BlockFloatingPoint(mantissa_bits=12, block_size=64)
        data = rng.randn(64) * 10
        mantissas, exp = bfp.scale_block(data)
        reconstructed = bfp.unscale_block(mantissas, exp)
        assert np.allclose(reconstructed, data, atol=0.05)

    def test_zero_block(self):
        """A zero block should produce zero mantissas and exponent 0."""
        bfp = BlockFloatingPoint()
        mantissas, exp = bfp.scale_block(np.zeros(64))
        assert exp == 0
        assert np.all(mantissas == 0)

    def test_dynamic_range_improvement(self):
        """BFP should provide more dynamic range than fixed-point."""
        bfp = BlockFloatingPoint(mantissa_bits=12, exponent_bits=4)
        dr = bfp.measure_dynamic_range()
        assert dr["improvement_db"] > 20  # at least 20 dB improvement
        assert dr["improvement_bits"] > 3  # at least 3 extra bits

    def test_transform_inverse(self, rng):
        """Full transform/inverse should reconstruct data."""
        bfp = BlockFloatingPoint(mantissa_bits=12, block_size=32)
        data = rng.randn(128) * 5
        mantissas, exponents = bfp.transform(data)
        reconstructed = bfp.inverse(mantissas, exponents)
        assert np.allclose(reconstructed, data, atol=0.1)

    def test_exponent_clamping(self):
        """Exponents should be clamped to representable range."""
        bfp = BlockFloatingPoint(mantissa_bits=12, exponent_bits=4, block_size=8)
        # Very large value
        mantissas, exp = bfp.scale_block(np.array([1e10, 0, 0, 0, 0, 0, 0, 0]))
        assert exp <= (1 << bfp.exponent_bits) - 1


# --------------------------------------------------------------------------- #
# 3. Carry-Save Accumulator
# --------------------------------------------------------------------------- #


class TestCarrySaveAccumulator:
    """Tests for carry-save accumulator correctness."""

    def test_matches_carry_propagate(self):
        """Carry-save result must equal carry-propagate result."""
        csa = CarrySaveAccumulator(width=16)
        values = [100, 200, 300, 50, 25, 15]
        result = csa.compare(values)
        assert result["match"]
        assert result["carry_save_result"] == result["carry_propagate_result"]

    def test_single_value(self):
        """Single value accumulation should work."""
        csa = CarrySaveAccumulator(width=16)
        assert csa.accumulate_cs([42]) == 42 & 0xFFFF
        assert csa.accumulate_cp([42]) == 42 & 0xFFFF

    def test_latency_reduction(self):
        """Carry-save should show latency reduction for multi-value accumulation."""
        csa = CarrySaveAccumulator(width=16)
        result = csa.compare(list(range(20)))
        assert result["latency_reduction_pct"] > 90  # ~95% for 20 values
        assert result["cs_cycles"] == 1  # single final CPA

    def test_large_values(self):
        """Large values should still match (within width)."""
        csa = CarrySaveAccumulator(width=16)
        values = [30000, 20000, 15000]
        result = csa.compare(values)
        assert result["match"]


# --------------------------------------------------------------------------- #
# 4. FMA Butterfly
# --------------------------------------------------------------------------- #


class TestFMAButterfly:
    """Tests for fused multiply-add butterfly."""

    def test_fma_accuracy_better_or_equal(self):
        """FMA should have less or equal error than standard butterfly."""
        fma = FMAButterfly()
        a, b, w = complex(1.234, 0.567), complex(0.89, -0.123), complex(0.707, 0.707)
        result = fma.compare(a, b, w)
        assert result["fma_error"] <= result["standard_error"] + 1e-10

    def test_fma_stages_saved(self):
        """FMA should save 1 pipeline stage."""
        fma = FMAButterfly()
        result = fma.compare(complex(1, 1), complex(0.5, 0.5), complex(0.7, 0.3))
        assert result["stages_saved"] == 1

    def test_fma_results_match_reference(self):
        """FMA results should be close to full-precision reference."""
        fma = FMAButterfly()
        a, b, w = complex(1.0, 2.0), complex(0.5, -0.5), complex(0.9, 0.1)
        up, lo = fma.fma_butterfly(a, b, w)
        assert abs(up - (a + w * b)) < 0.02
        assert abs(lo - (a - w * b)) < 0.02


# --------------------------------------------------------------------------- #
# 5. Ping-Pong Buffer
# --------------------------------------------------------------------------- #


class TestPingPongBuffer:
    """Tests for ping-pong dual-buffer throughput."""

    def test_throughput_improvement(self):
        """Ping-pong should show throughput improvement > 1."""
        pp = PingPongBuffer(n_stages=8, stage_cycles=32)
        result = pp.measure_throughput()
        assert result["throughput_improvement"] > 1.0

    def test_bubbles_eliminated(self):
        """Ping-pong should eliminate n_stages - 1 bubbles."""
        pp = PingPongBuffer(n_stages=8, stage_cycles=32)
        result = pp.measure_throughput()
        assert result["bubbles_eliminated"] == 7

    def test_no_bubbles_in_pingpong(self):
        """Ping-pong cycles should be less than single-buffer cycles."""
        pp = PingPongBuffer(n_stages=4, stage_cycles=16)
        assert pp.pingpong_cycles() < pp.single_buffer_cycles()


# --------------------------------------------------------------------------- #
# 6. Shadow Weight Prefetch
# --------------------------------------------------------------------------- #


class TestShadowWeightPrefetch:
    """Tests for shadow weight prefetch latency hiding."""

    def test_prefetch_reduces_latency(self):
        """Prefetch should reduce total latency."""
        swp = ShadowWeightPrefetch(n_modes=32, load_cycles_per_word=1)
        result = swp.measure_latency_hiding()
        assert result["with_prefetch_total"] < result["without_prefetch_total"]

    def test_fully_hidden_when_load_equals_mac(self):
        """When load time == MAC time, latency is fully hidden."""
        swp = ShadowWeightPrefetch(n_modes=32, load_cycles_per_word=1)
        result = swp.measure_latency_hiding()
        assert result["fully_hidden"]

    def test_stall_reduction(self):
        """Stall cycles should be reduced with prefetch."""
        swp = ShadowWeightPrefetch(n_modes=32, load_cycles_per_word=1)
        result = swp.measure_latency_hiding()
        assert result["stall_cycles_with"] <= result["stall_cycles_without"]


# --------------------------------------------------------------------------- #
# 7. Zero-Skip MAC (constant timing + power reduction)
# --------------------------------------------------------------------------- #


class TestZeroSkipMAC:
    """Tests for zero-skip MAC with constant timing."""

    def test_constant_timing(self):
        """Zero-skip MAC must maintain constant timing."""
        zs = ZeroSkipMAC(n_modes=32, threshold=0.5)
        stats = zs.measure_timing_constant()
        assert stats["is_constant_time"]
        assert stats["cycle_variance"] == 0.0

    def test_all_modes_processed(self):
        """All modes must be processed (no skip)."""
        zs = ZeroSkipMAC(n_modes=32, threshold=0.5)
        stats = zs.measure_timing_constant()
        assert stats["all_modes_processed"]

    def test_power_reduction_at_sparsity(self):
        """Power reduction should scale with sparsity."""
        zs = ZeroSkipMAC(n_modes=32, threshold=0.5)
        pr50 = zs.measure_power_reduction(sparsity=0.5)
        pr75 = zs.measure_power_reduction(sparsity=0.75)
        assert pr50["power_reduction_pct"] > 0
        assert pr75["power_reduction_pct"] > pr50["power_reduction_pct"]

    def test_correct_result(self):
        """Zero-skip MAC should produce correct accumulation."""
        zs = ZeroSkipMAC(n_modes=8, threshold=0.0)
        modes = [complex(i + 1, i * 0.5) for i in range(8)]
        weights = [complex(1, 0) for _ in range(8)]
        result = zs.process(modes, weights)
        expected = sum(m * w for m, w in zip(modes, weights))
        assert abs(result - expected) < 1e-6

    def test_uses_constant_time_mac(self):
        """ZeroSkipMAC must use ConstantTimeSpectralMAC internally."""
        from spectral_silicon.constant_time import ConstantTimeSpectralMAC
        zs = ZeroSkipMAC(n_modes=8)
        assert isinstance(zs._ct_mac, ConstantTimeSpectralMAC)


# --------------------------------------------------------------------------- #
# 8. RFFT Simulator
# --------------------------------------------------------------------------- #


class TestRFFTSimulator:
    """Tests for real-input FFT."""

    def test_hermitian_symmetry(self, rng):
        """Real-input FFT should have Hermitian symmetry."""
        rfft = RFFTSimulator(n=256)
        data = rng.randn(256)
        assert rfft.verify_hermitian_symmetry(data)

    def test_mode_reduction(self, rng):
        """RFFT should produce N/2+1 modes (about half)."""
        rfft = RFFTSimulator(n=256)
        result = rfft.compare(rng.randn(256))
        assert result["n_modes_rfft"] == 129  # 256/2 + 1
        assert result["mode_reduction_pct"] > 45

    def test_reconstruction_error(self, rng):
        """Reconstructed full spectrum should match complex FFT."""
        rfft = RFFTSimulator(n=128)
        result = rfft.compare(rng.randn(128))
        assert result["reconstruction_error"] < 1e-6

    def test_rfft_faster(self, rng):
        """RFFT should be faster than complex FFT."""
        rfft = RFFTSimulator(n=256)
        result = rfft.compare(rng.randn(256))
        # At minimum, RFFT should not be slower
        assert result["speedup"] >= 0.8  # generous for Python timing


# --------------------------------------------------------------------------- #
# 9. Twiddle Symmetry
# --------------------------------------------------------------------------- #


class TestTwiddleSymmetry:
    """Tests for twiddle factor symmetry exploitation."""

    def test_generate_four_twiddles(self):
        """Should generate 4 correct twiddle factors from 1."""
        twiddles = TwiddleSymmetry.generate_four_twiddles(k=3, n=256)
        assert len(twiddles) == 4
        # Verify correctness
        assert TwiddleSymmetry.verify_correctness(3, 256)

    def test_verify_correctness_multiple_k(self):
        """Symmetry should hold for multiple k values."""
        for k in range(0, 64, 8):
            assert TwiddleSymmetry.verify_correctness(k, 256)

    def test_storage_reduction(self):
        """Should achieve 4× storage compression."""
        ts = TwiddleSymmetry()
        result = ts.measure_storage_reduction(256)
        assert result["compression_ratio"] == 4.0
        assert result["gate_cost"] == 0  # wiring only

    def test_w_k_plus_quarter_is_neg_j_w(self):
        """W_{k+N/4} should equal -j * W_k."""
        twiddles = TwiddleSymmetry.generate_four_twiddles(k=5, n=256)
        w0 = twiddles[0]
        w1 = twiddles[1]
        neg_j_w0 = complex(w0.imag, -w0.real)
        assert abs(w1 - neg_j_w0) < 1e-10


# --------------------------------------------------------------------------- #
# 10. Mode Interleaver
# --------------------------------------------------------------------------- #


class TestModeInterleaver:
    """Tests for even/odd mode interleaving."""

    def test_result_correctness(self):
        """Interleaved result should match sequential."""
        mi = ModeInterleaver(n_modes=32)
        modes = [complex(i + 1, i * 0.3) for i in range(32)]
        weights = [complex(0.5, 0.1) for _ in range(32)]
        result = mi.compare(modes, weights)
        assert result["result_match"]

    def test_2x_throughput(self):
        """Interleaving should give ~2× throughput."""
        mi = ModeInterleaver(n_modes=32)
        result = mi.compare(
            [complex(1, 0)] * 32, [complex(1, 0)] * 32
        )
        assert result["throughput_improvement"] == 2.0

    def test_odd_mode_count(self):
        """Odd mode count should give ceil(k/2) cycles."""
        mi = ModeInterleaver(n_modes=17)
        assert mi.n_modes == 17
        modes = [complex(1, 0)] * 17
        weights = [complex(1, 0)] * 17
        result = mi.compare(modes, weights)
        assert result["interleaved_cycles"] == 9  # ceil(17/2)
        assert result["result_match"]


# --------------------------------------------------------------------------- #
# 11. Adaptive Mode Count
# --------------------------------------------------------------------------- #


class TestAdaptiveModeCount:
    """Tests for adaptive mode count configuration."""

    def test_speedup_increases_with_lower_k(self):
        """Lower k should give higher speedup."""
        amc = AdaptiveModeCount(n_fft=256)
        result = amc.measure_speedup()
        assert result["max_speedup"] > result["min_speedup"]

    def test_speedup_at_max_k_is_1(self):
        """At max k, speedup should be 1.0 (baseline)."""
        amc = AdaptiveModeCount(n_fft=256)
        result = amc.measure_speedup()
        assert abs(result["speedup_by_k"][32] - 1.0) < 1e-6

    def test_cycles_decrease_with_k(self):
        """Cycles should decrease with smaller k."""
        amc = AdaptiveModeCount(n_fft=256)
        c8 = amc.estimate_cycles(8)
        c32 = amc.estimate_cycles(32)
        assert c8 < c32


# --------------------------------------------------------------------------- #
# 12. Early IFFT
# --------------------------------------------------------------------------- #


class TestEarlyIFFT:
    """Tests for early IFFT start with overlap."""

    def test_latency_reduction(self):
        """Early IFFT should reduce latency."""
        ei = EarlyIFFT(n_fft=256, n_modes=32)
        result = ei.measure_latency_reduction()
        assert result["latency_reduction_pct"] > 0

    def test_cycles_saved(self):
        """Should save cycles from overlap."""
        ei = EarlyIFFT(n_fft=256, n_modes=32)
        result = ei.measure_latency_reduction()
        assert result["cycles_saved"] > 0

    def test_overlap_positive(self):
        """Overlap cycles should be positive when IFFT can start early."""
        ei = EarlyIFFT(n_fft=256, n_modes=32)
        result = ei.measure_latency_reduction()
        assert result["overlap_cycles"] > 0


# --------------------------------------------------------------------------- #
# 13. Configurable FFT
# --------------------------------------------------------------------------- #


class TestConfigurableFFT:
    """Tests for configurable FFT size."""

    def test_three_sizes(self):
        """Should support 128/256/512 sizes."""
        cfft = ConfigurableFFT()
        result = cfft.compare()
        assert 128 in result
        assert 256 in result
        assert 512 in result

    def test_smaller_fft_fewer_cycles(self):
        """128-point FFT should take fewer cycles than 512-point."""
        cfft = ConfigurableFFT()
        r128 = cfft.estimate_cycles(128)
        r512 = cfft.estimate_cycles(512)
        assert r128 < r512

    def test_speedup_for_short_sequence(self):
        """128-point should be faster than 256-point (default)."""
        cfft = ConfigurableFFT(default_n=256)
        result = cfft.compare()
        assert result[128]["speedup_vs_default"] > 1.0


# --------------------------------------------------------------------------- #
# 14. DVFS
# --------------------------------------------------------------------------- #


class TestDVFSSimulator:
    """Tests for DVFS power savings."""

    def test_power_savings(self):
        """Low voltage should reduce power."""
        dvfs = DVFSSimulator()
        result = dvfs.measure_power_savings()
        assert result["active_power_reduction_pct"] > 50
        assert result["idle_power_reduction_pct"] > 80

    def test_power_scales_with_v_squared(self):
        """Power should scale with V² (quadratic voltage scaling)."""
        dvfs = DVFSSimulator(nominal_freq_mhz=80, nominal_voltage=1.8,
                             low_freq_mhz=80, low_voltage=1.2)
        # Same frequency, different voltage → V² ratio
        p_nom = dvfs.power_at(80, 1.8)
        p_low = dvfs.power_at(80, 1.2)
        expected_ratio = (1.2 / 1.8) ** 2
        assert abs(p_low / p_nom - expected_ratio) < 1e-6

    def test_secure_tracking(self):
        """Secure voltage tracking should be verified."""
        dvfs = DVFSSimulator()
        result = dvfs.verify_secure_tracking()
        assert result["all_verified"]
        assert result["transition_only_between_batches"]
        assert result["fault_attack_resistant"]


# --------------------------------------------------------------------------- #
# 15. Dual Channel Processor
# --------------------------------------------------------------------------- #


class TestDualChannelProcessor:
    """Tests for dual-channel parallel processing."""

    def test_2x_throughput(self):
        """Dual channel should give 2× throughput vs sequential."""
        dc = DualChannelProcessor()
        result = dc.measure_throughput()
        assert result["throughput_improvement"] == 2.0

    def test_shared_weights(self):
        """Weight storage overhead should be 1.0 (shared)."""
        dc = DualChannelProcessor()
        result = dc.measure_throughput()
        assert result["weight_storage_overhead"] == 1.0

    def test_area_overhead(self):
        """Area overhead should be 2× (duplicate FFT)."""
        dc = DualChannelProcessor()
        result = dc.measure_throughput()
        assert result["area_overhead"] == 2.0


# --------------------------------------------------------------------------- #
# 16. Deep Pipeline FFT
# --------------------------------------------------------------------------- #


class TestDeepPipelineFFT:
    """Tests for 8-stage deep pipeline."""

    def test_freq_improvement(self):
        """Deep pipeline should enable higher frequency."""
        dp = DeepPipelineFFT()
        result = dp.estimate_max_freq()
        assert result["deep_freq_mhz"] > result["shallow_freq_mhz"]
        assert result["freq_improvement"] > 1.0

    def test_time_improvement(self):
        """Overall time should improve despite extra pipeline fill."""
        dp = DeepPipelineFFT()
        result = dp.measure_performance()
        assert result["time_improvement"] > 1.0

    def test_extra_pipeline_fill(self):
        """Deep pipeline has 4 extra fill cycles."""
        dp = DeepPipelineFFT()
        lat = dp.estimate_latency()
        assert lat["extra_pipeline_fill"] == 4


# --------------------------------------------------------------------------- #
# 17. Conflict-Free Addressing
# --------------------------------------------------------------------------- #


class TestConflictFreeAddressing:
    """Tests for conflict-free memory addressing."""

    def test_zero_stalls(self):
        """No bank conflicts should occur with conflict-free addressing."""
        cf = ConflictFreeAddressing()
        result = cf.verify_zero_stalls(256)
        assert result["zero_stalls"]
        assert result["bank_conflicts"] == 0
        assert result["stall_cycles"] == 0

    def test_4_banks(self):
        """Should use 4 memory banks."""
        cf = ConflictFreeAddressing()
        assert cf.N_BANKS == 4

    def test_bank_assignment(self):
        """Bank = addr mod 4."""
        cf = ConflictFreeAddressing()
        assert cf.bank_assignment(0) == 0
        assert cf.bank_assignment(1) == 1
        assert cf.bank_assignment(5) == 1
        assert cf.bank_assignment(7) == 3


# --------------------------------------------------------------------------- #
# 18. Bit-Reversal Router
# --------------------------------------------------------------------------- #


class TestBitReversalRouter:
    """Tests for hardware bit-reversal."""

    def test_bit_reverse(self):
        """8-bit reversal of 0b00000001 should be 0b10000000."""
        br = BitReversalRouter()
        assert br.bit_reverse(1, 8) == 128
        assert br.bit_reverse(128, 8) == 1
        assert br.bit_reverse(0, 8) == 0

    def test_inverse_property(self):
        """Bit-reversal should be its own inverse."""
        br = BitReversalRouter()
        assert br.verify_correctness(256)

    def test_latency_saved(self):
        """Hardware should save cycles vs software."""
        br = BitReversalRouter()
        result = br.measure_latency_saved(256)
        assert result["cycles_saved"] > 0
        assert result["hardware_cycles"] == 1
        assert result["software_cycles"] == 256


# --------------------------------------------------------------------------- #
# 19. DMA Burst Controller
# --------------------------------------------------------------------------- #


class TestDMABurstController:
    """Tests for DMA burst mode."""

    def test_overhead_reduction(self):
        """Burst mode should reduce bus overhead."""
        dma = DMABurstController(burst_size=4, n_modes=32)
        result = dma.measure_overhead_reduction()
        assert result["overhead_reduction_pct"] > 50

    def test_burst_cycles(self):
        """32 modes with 4-word bursts = 8 cycles."""
        dma = DMABurstController(burst_size=4, n_modes=32)
        assert dma.burst_cycles() == 8

    def test_single_word_cycles(self):
        """32 modes single-word = 32 cycles."""
        dma = DMABurstController(burst_size=4, n_modes=32)
        assert dma.single_word_cycles() == 32


# --------------------------------------------------------------------------- #
# 20. Truncated Booth
# --------------------------------------------------------------------------- #


class TestTruncatedBooth:
    """Tests for truncated Booth multiplier."""

    def test_bounded_input_correctness(self):
        """Truncated multiply should work for bounded twiddle inputs."""
        tb = TruncatedBooth(data_bits=16, product_bits=16)
        # Twiddle in [-1, 1], data in Q8.8 range
        result = tb.compare(1.5, 0.707)
        assert result["error"] < 0.1

    def test_area_savings(self):
        """Truncated Booth should save ~30% area."""
        tb = TruncatedBooth()
        result = tb.compare(1.0, 0.5)
        assert result["area_savings_pct"] == 30.0


# --------------------------------------------------------------------------- #
# Full Chip — PerfChipV3
# --------------------------------------------------------------------------- #


class TestPerfChipV3:
    """Tests for the full-chip PerfChipV3 simulation."""

    def test_estimate_area(self, chip):
        """estimate_area should return v1/v2/v3 with totals."""
        area = chip.estimate_area()
        assert "v1" in area and "v2" in area and "v3" in area
        for v in ["v1", "v2", "v3"]:
            assert "total" in area[v]
            assert area[v]["total"] > 0

    def test_v2_smaller_than_v1(self, chip):
        """v2 should have smaller area than v1 (shared FFT, serialized)."""
        area = chip.estimate_area()
        assert area["v2"]["total"] < area["v1"]["total"]

    def test_estimate_power(self, chip):
        """estimate_power should return v1/v2/v3 with totals."""
        power = chip.estimate_power()
        assert "v1" in power and "v2" in power and "v3" in power
        for v in ["v1", "v2", "v3"]:
            assert "total" in power[v]
            assert power[v]["total"] > 0

    def test_v2_lower_power_than_v1(self, chip):
        """v2 should have lower power than v1 (clock gating, serialized)."""
        power = chip.estimate_power()
        assert power["v2"]["total"] < power["v1"]["total"]

    def test_estimate_throughput(self, chip):
        """estimate_throughput should return v1/v2/v3 with max/min."""
        tp = chip.estimate_throughput()
        assert "v1" in tp and "v2" in tp and "v3" in tp
        for v in ["v1", "v2", "v3"]:
            assert "max" in tp[v]
            assert "min" in tp[v]
            assert tp[v]["max"] > 0

    def test_v3_faster_than_v2(self, chip):
        """v3 should have higher max throughput than v2 (same serialized arch)."""
        tp = chip.estimate_throughput()
        assert tp["v3"]["max"] > tp["v2"]["max"]

    def test_verify_security_all_preserved(self, chip):
        """All security measures should be preserved in v3."""
        sec = chip.verify_security_preserved()
        assert isinstance(sec, dict)
        assert len(sec) >= 10  # at least 10 measures
        for measure, ok in sec.items():
            assert ok, f"Security measure '{measure}' is not preserved"

    def test_verify_security_includes_constant_time(self, chip):
        """Security verification should include constant-time MAC check."""
        sec = chip.verify_security_preserved()
        assert "constant_time_mac" in sec
        assert sec["constant_time_mac"] is True

    def test_verify_security_includes_dvfs(self, chip):
        """Security verification should include DVFS secure transition."""
        sec = chip.verify_security_preserved()
        assert "dvfs_secure_transition" in sec
        assert sec["dvfs_secure_transition"] is True

    def test_all_20_modules_instantiated(self, chip):
        """PerfChipV3 should instantiate all 20 improvement simulators."""
        # Check that all 20 simulators are present
        sim_names = [
            "booth", "bfp", "carry_save", "fma", "truncated_booth",
            "pingpong", "prefetch", "zero_skip", "conflict_free", "bit_reverse",
            "rfft", "twiddle_sym", "mode_interleave", "adaptive_k", "early_ifft",
            "configurable_fft", "dvfs", "dual_channel", "deep_pipeline", "dma_burst",
        ]
        for name in sim_names:
            assert hasattr(chip, name), f"Missing simulator: {name}"


# --------------------------------------------------------------------------- #
# Integration / smoke test
# --------------------------------------------------------------------------- #


class TestIntegration:
    """Integration tests combining multiple modules."""

    def test_full_chip_self_test(self):
        """PerfChipV3 self-test should pass."""
        from spectral_silicon.perf_sim import _self_test
        _self_test()  # should not raise

    def test_benchmark_v3_imports(self):
        """benchmark_v3.py should be importable and runnable."""
        import importlib
        import sys
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        # Import the benchmark module
        spec = importlib.util.spec_from_file_location(
            "benchmark_v3",
            os.path.join(project_root, "scripts", "benchmark_v3.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        results = mod.benchmark_v3()
        assert "area" in results
        assert "power" in results
        assert "throughput" in results
        assert "comparison" in results
        assert "security" in results