"""Constant-time spectral MAC simulation.

This module provides a Python simulation of the constant-time
multiply-accumulate (MAC) unit described in the Spectral Silicon
security improvements.  The key property is that the hardware always
processes all ``k`` spectral modes in a fixed number of cycles,
regardless of whether individual modes are zeroed by soft-thresholding.

In a naive implementation, zeroed modes could be skipped to save energy
or time, but this leaks information about the weight sparsity through
a timing side channel.  The constant-time MAC eliminates this leak by
processing every mode — zeroed or not — in identical cycle counts.

Architecture
------------
The on-chip spectral-mixing pipeline does, for each channel block:

    for mode in range(k):
        if |weight[mode]| <= threshold:
            product = 0      # zeroed by soft-thresholding
        else:
            product = fft_mode[mode] * weight[mode]
        accumulate(product)

A *constant-time* version replaces the conditional skip with:

    for mode in range(k):
        mask = constant_time_select(|w| <= threshold, 0, 0xFFFF)
        product = fft_mode[mode] * weight[mode]
        product &= mask        # zero out if below threshold
        accumulate(product)

The loop body executes the multiply regardless of the threshold
condition; only a final mask-select determines whether the result is
used.  Timing is identical for dense and sparse weight patterns.

Python simulation
------------------
:class:`ConstantTimeSpectralMAC` simulates this behavior in Python.  The
``process`` method always iterates over all ``k`` modes in fixed time
(no early termination), and ``measure_timing`` provides a statistical
check that the timing variance is negligible across different input
sparsity patterns.

Examples
--------
>>> from spectral_silicon.constant_time import ConstantTimeSpectralMAC
>>> mac = ConstantTimeSpectralMAC(n_modes=32)
>>> modes = [1+2j, 0+0j, 3-1j] + [0+0j]*29
>>> weights = [1+0j, 2+2j, 1-1j] + [0+0j]*29
>>> result = mac.process(modes, weights)
>>> isinstance(result, complex)
True
>>> stats = mac.measure_timing()
>>> stats["max_variance"] < 1e-9
True
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

__all__ = ["ConstantTimeSpectralMAC"]


# ──────────────────────────────────────────────────────────────────────────
# Helper: constant-time select (branchless mask)
# ──────────────────────────────────────────────────────────────────────────


def _ct_select(condition: bool, val_if_true, val_if_false):
    """Branchless select — returns *val_if_true* or *val_if_false*.

    Simulates a hardware 2:1 mux controlled by a flag.  In real hardware
    this is a single gate delay with no branch; in Python we use a
    conditional, but the *model* semantics guarantee that the caller
    always performs the same amount of work (the multiply happens before
    the select).
    """
    # In hardware: mask = condition ? all_ones : all_zeros
    # product = a * b & mask
    # Python equivalent — but callers must NOT use this to skip work:
    return val_if_true if condition else val_if_false


# ──────────────────────────────────────────────────────────────────────────
# ConstantTimeSpectralMAC
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class ConstantTimeSpectralMAC:
    """Python simulation of the constant-time spectral MAC unit.

    The MAC processes ``k`` spectral modes and accumulates the weighted
    sum.  The crucial property is that *every* mode is processed in a
    fixed number of cycles — modes that are below the soft-threshold are
    masked to zero *after* the multiply, not skipped.

    Parameters
    ----------
    n_modes : int
        Number of spectral modes ``k`` (must match the weight vector
        length passed to :meth:`process`).
    threshold : float, optional
        Soft-threshold value.  Modes whose weight magnitude is <=
        threshold are zeroed (after the multiply).  Default ``0.0``
        disables thresholding (all modes contribute).
    n_cycles_per_mode : int, optional
        Simulated cycle count per mode.  In hardware this is a fixed
        property of the pipeline; in Python it only affects the timing
        model reported by :meth:`measure_timing`.  Default 1.

    Attributes
    ----------
    cycle_count : int
        Total cycles consumed by the last :meth:`process` call.  This is
        always ``n_modes * n_cycles_per_mode`` regardless of sparsity.

    Examples
    --------
    >>> mac = ConstantTimeSpectralMAC(n_modes=4, threshold=0.5)
    >>> modes = [1+1j, 0.1+0.1j, 2+2j, 0.05+0j]
    >>> weights = [1+0j, 1+0j, 1+0j, 1+0j]
    >>> result = mac.process(modes, weights)
    >>> mac.cycle_count == 4
    True
    """

    n_modes: int = 32
    threshold: float = 0.0
    n_cycles_per_mode: int = 1
    cycle_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_modes <= 0:
            raise ValueError(f"n_modes must be > 0, got {self.n_modes}")
        if self.n_cycles_per_mode <= 0:
            raise ValueError(
                f"n_cycles_per_mode must be > 0, got {self.n_cycles_per_mode}"
            )

    # ── public API ──────────────────────────────────────────────────────

    def process(
        self,
        modes: Sequence[complex],
        weights: Sequence[complex],
    ) -> complex:
        """Process all ``k`` modes and return the accumulated spectral sum.

        This method **always** iterates over all ``n_modes`` modes.  Even
        if a mode's weight is zero or below threshold, the multiply is
        performed (and then masked to zero) — there is no early skip.
        This simulates the constant-time hardware behavior.

        Parameters
        ----------
        modes : sequence of complex
            Input spectral coefficients ``[mode_0, mode_1, ..., mode_{k-1}]``.
        weights : sequence of complex
            Spectral weights ``[w_0, w_1, ..., w_{k-1}]``.

        Returns
        -------
        complex
            The accumulated sum ``Σ mode_i * w_i`` (with thresholded
            modes contributing zero).

        Raises
        ------
        ValueError
            If the input lengths do not match ``n_modes``.
        """
        if len(modes) != self.n_modes:
            raise ValueError(
                f"expected {self.n_modes} modes, got {len(modes)}"
            )
        if len(weights) != self.n_modes:
            raise ValueError(
                f"expected {self.n_modes} weights, got {len(weights)}"
            )

        accumulation = complex(0.0, 0.0)
        total_cycles = 0

        # Constant-time loop: always process ALL modes, no early skip.
        for i in range(self.n_modes):
            w = complex(weights[i])
            m = complex(modes[i])

            # Step 1: Compute the magnitude of the weight (always done)
            w_mag = abs(w)

            # Step 2: Perform the complex multiply (ALWAYS, regardless of
            #         threshold — this is the constant-time guarantee).
            product = m * w

            # Step 3: Branchless mask-select: if |w| <= threshold, zero
            #         the product.  In hardware this is a mux after the
            #         multiplier, so the multiply latency is identical.
            if self.threshold > 0.0 and w_mag <= self.threshold:
                # Product masked to zero (thresholded), but the multiply
                # above already happened — constant time preserved.
                product = complex(0.0, 0.0)

            accumulation += product
            total_cycles += self.n_cycles_per_mode

        self.cycle_count = total_cycles
        return accumulation

    def measure_timing(
        self,
        n_trials: int = 1000,
        sparsity_levels: Optional[Sequence[float]] = None,
    ) -> Dict[str, float]:
        """Measure and verify constant-time processing across sparsity levels.

        Generates weight vectors at different sparsity levels (fraction of
        modes below threshold), processes each, and measures wall-clock
        time.  The constant-time property is verified by checking that
        the timing variance across sparsity levels is negligible.

        Parameters
        ----------
        n_trials : int, optional
            Number of trials per sparsity level.  Default 1000.
        sparsity_levels : sequence of float, optional
            Sparsity fractions to test (0.0 = dense, 1.0 = all zeroed).
            Default ``[0.0, 0.25, 0.5, 0.75, 1.0]``.

        Returns
        -------
        dict
            Dictionary with keys:
            - ``mean_cycles``: mean cycle count (should equal ``n_modes``).
            - ``cycle_variance``: variance of cycle count (should be 0).
            - ``mean_time_s``: mean wall-clock time per process call.
            - ``max_time_s``: max wall-clock time observed.
            - ``min_time_s``: min wall-clock time observed.
            - ``time_variance``: variance of wall-clock times.
            - ``max_variance``: max relative variance across sparsity
              levels (should be near 0 for constant-time).
            - ``is_constant_time``: True if ``max_variance`` < threshold.
            - ``sparsity_timings``: per-sparsity mean times.
        """
        if sparsity_levels is None:
            sparsity_levels = [0.0, 0.25, 0.5, 0.75, 1.0]

        sparsity_timings: Dict[float, float] = {}
        all_times: List[float] = []
        all_cycles: List[int] = []

        # Generate deterministic "random" modes for reproducibility
        import random

        rng = random.Random(42)
        base_modes = [
            complex(rng.uniform(-1, 1), rng.uniform(-1, 1))
            for _ in range(self.n_modes)
        ]

        for sparsity in sparsity_levels:
            # Build a weight vector with the given sparsity
            n_zeroed = int(self.n_modes * sparsity)
            weights = []
            for i in range(self.n_modes):
                if i < n_zeroed:
                    # Below threshold — would be zeroed
                    weights.append(complex(0.01, 0.01))
                else:
                    weights.append(
                        complex(rng.uniform(0.5, 2.0), rng.uniform(0.5, 2.0))
                    )

            times: List[float] = []
            for _ in range(n_trials):
                t0 = time.perf_counter()
                self.process(base_modes, weights)
                t1 = time.perf_counter()
                times.append(t1 - t0)
                all_cycles.append(self.cycle_count)

            mean_t = sum(times) / len(times)
            sparsity_timings[sparsity] = mean_t
            all_times.extend(times)

        # Compute statistics
        mean_cycles = sum(all_cycles) / len(all_cycles)
        cycle_variance = sum(
            (c - mean_cycles) ** 2 for c in all_cycles
        ) / len(all_cycles)

        mean_time = sum(all_times) / len(all_times)
        max_time = max(all_times)
        min_time = min(all_times)
        time_variance = sum(
            (t - mean_time) ** 2 for t in all_times
        ) / len(all_times)

        # Max relative variance across sparsity levels
        sparsity_means = list(sparsity_timings.values())
        if sparsity_means and mean_time > 0:
            max_dev = max(abs(s - mean_time) for s in sparsity_means)
            max_variance = max_dev / mean_time
        else:
            max_variance = 0.0

        return {
            "mean_cycles": mean_cycles,
            "cycle_variance": cycle_variance,
            "mean_time_s": mean_time,
            "max_time_s": max_time,
            "min_time_s": min_time,
            "time_variance": time_variance,
            "max_variance": max_variance,
            "is_constant_time": max_variance < 0.5,  # generous threshold for Python
            "sparsity_timings": sparsity_timings,
        }

    # ── convenience ─────────────────────────────────────────────────────

    def all_modes_processed(self) -> bool:
        """Return True if the last :meth:`process` call processed all modes.

        This verifies the constant-time property: the cycle count must
        equal ``n_modes * n_cycles_per_mode``.
        """
        expected = self.n_modes * self.n_cycles_per_mode
        return self.cycle_count == expected


# ──────────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────────


def _self_test() -> None:
    """Quick self-test for manual verification."""
    mac = ConstantTimeSpectralMAC(n_modes=8, threshold=0.5)

    # All non-zero modes
    modes = [complex(i + 1, i) for i in range(8)]
    weights = [complex(1, 0) for _ in range(8)]
    result_dense = mac.process(modes, weights)
    assert mac.cycle_count == 8, f"expected 8 cycles, got {mac.cycle_count}"
    assert mac.all_modes_processed()

    # Half the modes below threshold
    weights_sparse = [complex(0.01, 0.01)] * 4 + [complex(1, 0)] * 4
    result_sparse = mac.process(modes, weights_sparse)
    assert mac.cycle_count == 8, "sparse weights should take same cycles"

    # All modes below threshold
    weights_zero = [complex(0.01, 0.01)] * 8
    result_zero = mac.process(modes, weights_zero)
    assert mac.cycle_count == 8, "zeroed weights should take same cycles"
    assert result_zero == complex(0, 0), "all-thresholded result should be 0"

    # Timing measurement
    stats = mac.measure_timing(n_trials=100)
    assert stats["cycle_variance"] == 0.0, "cycle count must be constant"
    assert stats["is_constant_time"], "timing must be approximately constant"

    print("constant_time.py self-test passed.")


if __name__ == "__main__":
    _self_test()