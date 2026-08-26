#!/usr/bin/env python3
"""SpectralChip — Host driver for the Spectral Silicon inference chip.

Provides a Python interface to communicate with the fabricated spectral
mixer chip via SPI. Supports two modes:

  1. **Hardware mode** (default): Uses spidev to communicate with the
     physical chip over SPI. Requires a Raspberry Pi or similar SPI host.

  2. **Simulation mode** (sim=True): Runs the full spectral mixing pipeline
     in software (numpy/torch) to verify correctness without hardware.

SPI Protocol (Wishbone-over-SPI):
  - 32-bit words, MSB first
  - Command format: [R/W (1 bit)] [addr (15 bits)] [data (16 bits)]
  - Register map: see REG_* constants below

Usage:
    from spectral_driver import SpectralChip

    # Simulation mode (no hardware needed)
    chip = SpectralChip(sim=True)
    chip.load_weights(weights)
    output = chip.run_inference(input_data)
    result = chip.read_output()

    # Hardware mode (requires spidev)
    chip = SpectralChip(spi_bus=0, spi_device=0, sim=False)
    chip.load_weights(weights)
    chip.run_inference(input_data)
    result = chip.read_output()
"""

import struct
import sys
import time

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ---------------------------------------------------------------------------
# Register map (Wishbone addresses, 16-bit each)
# ---------------------------------------------------------------------------

REG_START = 0x00           # Write 1 to start computation
REG_DONE = 0x01            # Read: 1 when computation complete
REG_MODE_COUNT = 0x02      # Number of spectral modes (k)
REG_BLOCK_SIZE = 0x03      # Block-diagonal block size
REG_THRESHOLD = 0x04       # Soft-threshold value (Q8.8)
REG_WEIGHT_BASE = 0x10     # Base address for weight memory
REG_DATA_BASE = 0x40       # Base address for input data buffer
REG_OUTPUT_BASE = 0x80     # Base address for output data buffer
REG_STATUS = 0x05          # Status register
REG_CONFIG = 0x06          # Configuration register
REG_VERSION = 0xFF         # Chip version / ID

# Status register bits
STATUS_IDLE = 0x01
STATUS_RUNNING = 0x02
STATUS_DONE = 0x04
STATUS_ERROR = 0x08


class SpectralChip:
    """Host driver for the Spectral Silicon inference chip.

    Parameters
    ----------
    sim : bool
        If True, run in software simulation mode (no hardware needed).
        If False, communicate with the physical chip via SPI.
    spi_bus : int
        SPI bus number (hardware mode only).
    spi_device : int
        SPI device/CS number (hardware mode only).
    spi_speed : int
        SPI clock speed in Hz (hardware mode only).
    fft_size : int
        FFT size (N). Default 256.
    channels : int
        Number of channels (feature dimension). Default 64.
    modes : int
        Number of spectral modes (k). Default 32.
    block_size : int
        Block-diagonal block size for AFNO weights. Default 8.
    threshold : float
        Soft-threshold value. Default 0.1.
    """

    def __init__(
        self,
        sim=True,
        spi_bus=0,
        spi_device=0,
        spi_speed=10_000_000,
        fft_size=256,
        channels=64,
        modes=32,
        block_size=8,
        threshold=0.1,
    ):
        self.sim = sim
        self.fft_size = fft_size
        self.channels = channels
        self.modes = modes
        self.block_size = block_size
        self.threshold = threshold
        self.spi_bus = spi_bus
        self.spi_device = spi_device
        self.spi_speed = spi_speed

        # State
        self._weights_loaded = False
        self._weights = None
        self._input_data = None
        self._output_data = None
        self._spi = None

        if not sim:
            self._init_spi()

    # -----------------------------------------------------------------
    # SPI initialization
    # -----------------------------------------------------------------

    def _init_spi(self):
        """Initialize the SPI device using spidev."""
        try:
            import spidev
        except ImportError:
            raise ImportError(
                "spidev is required for hardware mode. "
                "Install with: pip install spidev  (or apt-get install python3-spidev)"
            )
        self._spi = spidev.SpiDev()
        self._spi.open(self.spi_bus, self.spi_device)
        self._spi.max_speed_hz = self.spi_speed
        self._spi.mode = 0
        self._spi.bits_per_word = 8

    # -----------------------------------------------------------------
    # Low-level register access (hardware mode)
    # -----------------------------------------------------------------

    def _spi_write_reg(self, addr, data):
        """Write a 16-bit value to a Wishbone register via SPI."""
        # Command: [1=write][addr:15][data:16]
        cmd = 0x8000 | (addr & 0x7FFF)
        tx = struct.pack(">HH", cmd, data & 0xFFFF)
        self._spi.xfer2(list(tx))

    def _spi_read_reg(self, addr):
        """Read a 16-bit Wishbone register via SPI."""
        cmd = addr & 0x7FFF  # R/W=0 for read
        tx = struct.pack(">HH", cmd, 0x0000)
        rx = self._spi.xfer2(list(tx))
        # Second word is the response
        return struct.unpack(">H", bytes(rx[2:4]))[0]

    def _spi_write_block(self, base_addr, data_bytes):
        """Write a block of data to contiguous addresses."""
        for i in range(0, len(data_bytes), 2):
            word = struct.unpack(">H", data_bytes[i:i + 2])[0]
            self._spi_write_reg(base_addr + i // 2, word)

    def _spi_read_block(self, base_addr, num_words):
        """Read a block of data from contiguous addresses."""
        result = bytearray()
        for i in range(num_words):
            word = self._spi_read_reg(base_addr + i)
            result.extend(struct.pack(">H", word))
        return bytes(result)

    # -----------------------------------------------------------------
    # Bitbang SPI (fallback if spidev unavailable)
    # -----------------------------------------------------------------

    def _bitbang_write(self, data):
        """Software SPI bitbang for debugging on non-SPI platforms."""
        # Placeholder: implement GPIO bitbang if needed
        raise NotImplementedError("Bitbang SPI not implemented — use sim=True or install spidev")

    # -----------------------------------------------------------------
    # Public API: weight loading
    # -----------------------------------------------------------------

    def load_weights(self, weights):
        """Load spectral weights into the chip.

        Parameters
        ----------
        weights : np.ndarray or torch.Tensor
            Complex weight tensor of shape (modes, channels) or
            (modes, block_size, block_size) for block-diagonal weights.
            Real and imaginary parts are loaded separately.

        For int8 quantized weights, pass (q_re, q_im) tuple.
        """
        if isinstance(weights, tuple) and len(weights) == 2:
            # Already quantized (q_re, q_im)
            q_re, q_im = weights
            if HAS_TORCH and isinstance(q_re, torch.Tensor):
                q_re = q_re.numpy()
                q_im = q_im.numpy()
            self._weights = (np.asarray(q_re), np.asarray(q_im))
        else:
            # Float weights — store for simulation; quantize for hardware
            if HAS_TORCH and isinstance(weights, torch.Tensor):
                if weights.is_complex():
                    weights = weights.numpy()
                else:
                    weights = weights.detach().numpy()
            weights = np.asarray(weights)
            self._weights = weights

        if self.sim:
            self._weights_loaded = True
            return

        # Hardware: quantize and send via SPI
        if isinstance(self._weights, tuple):
            q_re, q_im = self._weights
        else:
            from spectral_silicon.quantize import quantize_complex_weights
            if HAS_TORCH:
                w_tensor = torch.from_numpy(self._weights.astype(np.complex64))
                q_re, q_im = quantize_complex_weights(w_tensor)
                q_re = q_re.numpy()
                q_im = q_im.numpy()
            else:
                # Manual quantization fallback
                q_re, q_im = self._manual_quantize(self._weights)

        # Send real parts
        re_bytes = q_re.astype(np.int8).tobytes()
        self._spi_write_block(REG_WEIGHT_BASE, re_bytes)
        # Send imaginary parts
        im_bytes = q_im.astype(np.int8).tobytes()
        weight_im_base = REG_WEIGHT_BASE + len(re_bytes) // 2
        self._spi_write_block(weight_im_base, im_bytes)

        # Configure registers
        self._spi_write_reg(REG_MODE_COUNT, self.modes)
        self._spi_write_reg(REG_BLOCK_SIZE, self.block_size)
        threshold_fixed = int(self.threshold * 256)  # Q8.8
        self._spi_write_reg(REG_THRESHOLD, threshold_fixed & 0xFFFF)

        self._weights_loaded = True

    def _manual_quantize(self, weights):
        """Manual int8 quantization when torch is unavailable."""
        if np.iscomplexobj(weights):
            re, im = weights.real, weights.imag
        else:
            re, im = weights, np.zeros_like(weights)
        def quant(x):
            x_min, x_max = x.min(), x.max()
            scale = (x_max - x_min) / 255.0 if x_max > x_min else 1.0
            return np.round((x - x_min) / scale - 128).astype(np.int8)
        return quant(re), quant(im)

    # -----------------------------------------------------------------
    # Public API: inference
    # -----------------------------------------------------------------

    def run_inference(self, input_data):
        """Run spectral mixing inference on the input.

        Parameters
        ----------
        input_data : np.ndarray or torch.Tensor
            Input tensor of shape (batch, seq_len, channels) or
            (seq_len, channels).

        Returns
        -------
        None — call read_output() to retrieve results.
        """
        if not self._weights_loaded:
            raise RuntimeError("Call load_weights() before run_inference()")

        if HAS_TORCH and isinstance(input_data, torch.Tensor):
            input_data = input_data.detach().numpy()
        input_data = np.asarray(input_data, dtype=np.float32)

        if input_data.ndim == 2:
            input_data = input_data[np.newaxis, ...]

        self._input_data = input_data

        if self.sim:
            self._output_data = self._run_sim(input_data)
            return

        # Hardware mode: send data, start, wait, read is in read_output()
        data_bytes = input_data.astype(np.float16).tobytes()
        self._spi_write_block(REG_DATA_BASE, data_bytes)
        self._spi_write_reg(REG_START, 1)

        # Poll for completion
        timeout = 5.0  # seconds
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self._spi_read_reg(REG_DONE) & 1:
                break
            time.sleep(0.001)
        else:
            raise TimeoutError("Chip did not complete inference within timeout")

    def _run_sim(self, input_data):
        """Run the spectral mixing pipeline in software simulation.

        This mirrors the hardware pipeline:
        FFT → spectral weight multiply → soft-threshold → modReLU → IFFT
        """
        batch, seq_len, ch = input_data.shape
        orig_seq_len = seq_len

        # Pad/truncate to fft_size if needed
        if seq_len < self.fft_size:
            pad = np.zeros((batch, self.fft_size - seq_len, ch), dtype=np.float32)
            input_data = np.concatenate([input_data, pad], axis=1)
            seq_len = self.fft_size
        elif seq_len > self.fft_size:
            input_data = input_data[:, :self.fft_size, :]
            seq_len = self.fft_size

        output = np.zeros_like(input_data)

        for b in range(batch):
            # FFT along sequence dimension
            freq = np.fft.fft(input_data[b], axis=0)  # (seq_len, ch)

            # Spectral weight multiply (only first 'modes' modes)
            k = min(self.modes, seq_len // 2)
            weights = self._weights
            if weights is None:
                w_complex = np.ones((k, 1), dtype=np.complex64)
            elif isinstance(weights, tuple):
                q_re, q_im = weights
                w_complex = q_re[:k].astype(np.float32) + 1j * q_im[:k].astype(np.float32)
            elif np.iscomplexobj(weights):
                w = weights[:k]
                if w.ndim == 1:
                    w_complex = w[:, np.newaxis]
                elif w.ndim == 2:
                    w_complex = w
                else:
                    w_complex = w
            else:
                w_complex = weights[:k].astype(np.complex64)

            # Apply weights to first k modes
            if w_complex.ndim == 2:
                freq[:k] = freq[:k] * w_complex[np.newaxis, ...] if w_complex.shape[0] == k else freq[:k] * w_complex.T[np.newaxis, ...]
            elif w_complex.ndim == 1:
                freq[:k] = freq[:k] * w_complex[:, np.newaxis]

            # Soft-thresholding
            mag = np.abs(freq[:k])
            threshold = self.threshold
            scale = np.maximum(mag - threshold, 0) / (mag + 1e-12)
            freq[:k] = freq[:k] * scale

            # Zero out modes beyond k
            freq[k:] = 0

            # IFFT
            output[b] = np.real(np.fft.ifft(freq, axis=0))

        # Remove padding (compare to original input length stored on instance)
        orig_input = self._input_data
        if orig_input is not None and output.shape[1] != orig_input.shape[1]:
            output = output[:, :orig_input.shape[1], :]

        return output.astype(np.float32)

    # -----------------------------------------------------------------
    # Public API: read output
    # -----------------------------------------------------------------

    def read_output(self):
        """Read the inference result.

        Returns
        -------
        np.ndarray
            Output tensor of shape (batch, seq_len, channels).
        """
        if self.sim:
            if self._output_data is None:
                raise RuntimeError("Call run_inference() before read_output()")
            return self._output_data.copy()

        # Hardware: read from output buffer
        n_words = self._input_data.size // 2  # float16 = 2 bytes
        out_bytes = self._spi_read_block(REG_OUTPUT_BASE, n_words)
        output = np.frombuffer(out_bytes, dtype=np.float16).astype(np.float32)
        output = output.reshape(self._input_data.shape)
        return output

    # -----------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------

    def reset(self):
        """Reset the chip to idle state."""
        if self.sim:
            self._output_data = None
            self._input_data = None
        else:
            self._spi_write_reg(REG_START, 0)
            self._spi_write_reg(REG_CONFIG, 0)

    def get_status(self):
        """Read the chip status register.

        Returns
        -------
        dict with keys: idle, running, done, error
        """
        if self.sim:
            if self._output_data is not None:
                return {"idle": False, "running": False, "done": True, "error": False}
            return {"idle": True, "running": False, "done": False, "error": False}

        status = self._spi_read_reg(REG_STATUS)
        return {
            "idle": bool(status & STATUS_IDLE),
            "running": bool(status & STATUS_RUNNING),
            "done": bool(status & STATUS_DONE),
            "error": bool(status & STATUS_ERROR),
        }

    def get_version(self):
        """Read the chip version/ID."""
        if self.sim:
            return "SIM-v0.1.0"
        return self._spi_read_reg(REG_VERSION)

    def close(self):
        """Release SPI resources."""
        if self._spi is not None:
            self._spi.close()
            self._spi = None

    def __del__(self):
        self.close()

    def __repr__(self):
        mode = "SIM" if self.sim else f"HW(spi{self.spi_bus}.{self.spi_device})"
        return (
            f"SpectralChip({mode}, N={self.fft_size}, "
            f"ch={self.channels}, k={self.modes}, "
            f"block={self.block_size}, thresh={self.threshold})"
        )