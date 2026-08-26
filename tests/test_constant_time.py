"""Tests for the constant-time spectral MAC simulation.

Covers:
  - Constant timing across different input sparsity levels
  - All modes are always processed (no early skip)
  - Cycle count is independent of weight pattern
  - Correct accumulation results
"""

import time

import pytest

from spectral_silicon.constant_time import ConstantTimeSpectralMAC


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def mac():
    """A 32-mode constant-time MAC."""
    return ConstantTimeSpectralMAC(n_modes=32, threshold=0.5)


@pytest.fixture
def dense_weights():
    """All 32 weights above threshold (dense, 0% sparse)."""
    return [complex(1.0 + i * 0.1, 0.5) for i in range(32)]


@pytest.fixture
def sparse_weights():
    """Half of the 32 weights below threshold (50% sparse)."""
    return [complex(0.01, 0.01)] * 16 + [complex(1.0 + i * 0.1, 0.5) for i in range(16)]


@pytest.fixture
def all_zero_weights():
    """All 32 weights below threshold (100% sparse)."""
    return [complex(0.01, 0.01)] * 32


@pytest.fixture
def sample_modes():
    """32 input spectral modes."""
    return [complex(i + 1, i * 0.5) for i in range(32)]


# --------------------------------------------------------------------------- #
# Constant timing tests
# --------------------------------------------------------------------------- #


class TestConstantTiming:
    """Verify that processing time does not depend on input sparsity."""

    def test_cycle_count_independent_of_sparsity(
        self, mac, sample_modes, dense_weights, sparse_weights, all_zero_weights
    ):
        """Cycle count should be the same regardless of weight sparsity."""
        # Dense
        mac.process(sample_modes, dense_weights)
        cycles_dense = mac.cycle_count

        # 50% sparse
        mac.process(sample_modes, sparse_weights)
        cycles_sparse = mac.cycle_count

        # 100% sparse (all zeroed)
        mac.process(sample_modes, all_zero_weights)
        cycles_zero = mac.cycle_count

        assert cycles_dense == cycles_sparse == cycles_zero
        assert cycles_dense == 32  # n_modes

    def test_cycle_count_matches_n_modes(self, mac, sample_modes, dense_weights):
        """Cycle count should always equal n_modes * n_cycles_per_mode."""
        mac.process(sample_modes, dense_weights)
        assert mac.cycle_count == mac.n_modes * mac.n_cycles_per_mode

    def test_all_modes_processed_flag(self, mac, sample_modes, dense_weights):
        """all_modes_processed() should return True after processing."""
        mac.process(sample_modes, dense_weights)
        assert mac.all_modes_processed() is True

    def test_no_early_termination_zero_weights(self, mac, sample_modes, all_zero_weights):
        """Processing all-zero weights should not skip any modes."""
        result = mac.process(sample_modes, all_zero_weights)
        assert mac.cycle_count == 32
        # Result should be zero (all modes thresholded)
        assert result == complex(0.0, 0.0)

    def test_measure_timing_returns_stats(self, mac):
        """measure_timing() should return a dict with expected keys."""
        stats = mac.measure_timing(n_trials=50)
        assert isinstance(stats, dict)
        assert "mean_cycles" in stats
        assert "cycle_variance" in stats
        assert "mean_time_s" in stats
        assert "max_time_s" in stats
        assert "min_time_s" in stats
        assert "time_variance" in stats
        assert "max_variance" in stats
        assert "is_constant_time" in stats
        assert "sparsity_timings" in stats

    def test_cycle_variance_is_zero(self, mac):
        """Cycle count variance across sparsity levels should be exactly 0."""
        stats = mac.measure_timing(n_trials=50)
        assert stats["cycle_variance"] == 0.0

    def test_is_constant_time(self, mac):
        """The MAC should be classified as constant-time."""
        stats = mac.measure_timing(n_trials=100)
        assert stats["is_constant_time"] is True

    def test_measure_timing_multiple_sparsity_levels(self, mac):
        """measure_timing should test multiple sparsity levels."""
        stats = mac.measure_timing(n_trials=20)
        sparsity = stats["sparsity_timings"]
        # Default levels: 0.0, 0.25, 0.5, 0.75, 1.0
        assert len(sparsity) >= 3
        assert 0.0 in sparsity
        assert 1.0 in sparsity

    def test_custom_sparsity_levels(self, mac):
        """Custom sparsity levels should be respected."""
        custom = [0.0, 0.5, 1.0]
        stats = mac.measure_timing(n_trials=20, sparsity_levels=custom)
        assert set(stats["sparsity_timings"].keys()) == set(custom)


# --------------------------------------------------------------------------- #
# Correctness tests
# --------------------------------------------------------------------------- #


class TestCorrectness:
    """Verify the MAC produces correct accumulation results."""

    def test_dense_result(self, mac, sample_modes, dense_weights):
        """Dense processing should accumulate all mode*weight products."""
        result = mac.process(sample_modes, dense_weights)
        expected = sum(m * w for m, w in zip(sample_modes, dense_weights))
        assert abs(result - expected) < 1e-6

    def test_zero_threshold_no_filtering(self, sample_modes, dense_weights):
        """With threshold=0, all modes should contribute regardless of magnitude."""
        mac = ConstantTimeSpectralMAC(n_modes=32, threshold=0.0)
        result = mac.process(sample_modes, dense_weights)
        expected = sum(m * w for m, w in zip(sample_modes, dense_weights))
        assert abs(result - expected) < 1e-6

    def test_thresholded_modes_are_zeroed(self, sample_modes, sparse_weights):
        """Modes below threshold should contribute zero."""
        mac = ConstantTimeSpectralMAC(n_modes=32, threshold=0.5)
        result = mac.process(sample_modes, sparse_weights)
        # Only modes 16..31 have weights above threshold
        expected = sum(
            sample_modes[i] * sparse_weights[i] for i in range(16, 32)
        )
        assert abs(result - expected) < 1e-6

    def test_all_zero_result(self, mac, sample_modes, all_zero_weights):
        """All-thresholded weights should produce zero output."""
        result = mac.process(sample_modes, all_zero_weights)
        assert result == complex(0.0, 0.0)

    def test_single_mode(self):
        """Single-mode MAC should work."""
        mac = ConstantTimeSpectralMAC(n_modes=1, threshold=0.0)
        result = mac.process([complex(2, 3)], [complex(1, 1)])
        assert abs(result - complex(2 + 3j + 2j - 3)) < 1e-6  # (2+3j)*(1+1j) = -1+5j

    def test_negative_threshold_disables_filtering(self, sample_modes):
        """A negative threshold means nothing is thresholded (all contribute)."""
        mac = ConstantTimeSpectralMAC(n_modes=32, threshold=-1.0)
        tiny_weights = [complex(0.001, 0.001)] * 32
        result = mac.process(sample_modes, tiny_weights)
        expected = sum(m * w for m, w in zip(sample_modes, tiny_weights))
        assert abs(result - expected) < 1e-6


# --------------------------------------------------------------------------- #
# Error handling tests
# --------------------------------------------------------------------------- #


class TestErrorHandling:
    """Test input validation and error conditions."""

    def test_wrong_mode_count(self, mac, dense_weights):
        """Wrong number of modes should raise ValueError."""
        with pytest.raises(ValueError):
            mac.process([complex(1, 0)], dense_weights)

    def test_wrong_weight_count(self, mac, sample_modes):
        """Wrong number of weights should raise ValueError."""
        with pytest.raises(ValueError):
            mac.process(sample_modes, [complex(1, 0)])

    def test_invalid_n_modes(self):
        """n_modes <= 0 should raise ValueError."""
        with pytest.raises(ValueError):
            ConstantTimeSpectralMAC(n_modes=0)
        with pytest.raises(ValueError):
            ConstantTimeSpectralMAC(n_modes=-1)

    def test_invalid_cycles_per_mode(self):
        """n_cycles_per_mode <= 0 should raise ValueError."""
        with pytest.raises(ValueError):
            ConstantTimeSpectralMAC(n_modes=32, n_cycles_per_mode=0)

    def test_empty_inputs(self, mac):
        """Empty mode and weight lists should raise ValueError."""
        with pytest.raises(ValueError):
            mac.process([], [])


# --------------------------------------------------------------------------- #
# Cycle count parameterization tests
# --------------------------------------------------------------------------- #


class TestCyclesPerMode:
    """Test that n_cycles_per_mode affects the total cycle count correctly."""

    @pytest.mark.parametrize("n_cycles", [1, 2, 3, 5])
    def test_cycle_count_scales(self, n_cycles):
        """Total cycles should be n_modes * n_cycles_per_mode."""
        mac = ConstantTimeSpectralMAC(n_modes=8, n_cycles_per_mode=n_cycles)
        modes = [complex(1, 0)] * 8
        weights = [complex(1, 0)] * 8
        mac.process(modes, weights)
        assert mac.cycle_count == 8 * n_cycles

    @pytest.mark.parametrize("n_modes", [1, 4, 16, 32, 64])
    def test_cycle_count_with_n_modes(self, n_modes):
        """Total cycles should scale with n_modes."""
        mac = ConstantTimeSpectralMAC(n_modes=n_modes)
        modes = [complex(1, 0)] * n_modes
        weights = [complex(1, 0)] * n_modes
        mac.process(modes, weights)
        assert mac.cycle_count == n_modes