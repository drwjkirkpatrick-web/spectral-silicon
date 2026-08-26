# Spectral Silicon — 30 Testable Prompts

Each prompt is a spec for a module, test, or deliverable. They are ordered so that each builds on the prior, and each can be verified independently.

---

## Phase 1: Foundational Simulation (Python)

### Prompt 1 — FNO Layer in Pure Python
Write a `FourierNeuralOperator` class in Python that takes a 1D sequence of shape (batch, seq_len, channels), applies FFT along the sequence dimension, multiplies by a learnable complex weight tensor (truncated to k modes), applies IFFT, and returns the output. Verify that zero modes (k=0) produces the identity. Verify that output shape matches input shape.

### Prompt 2 — AFNO Block-Diagonal Layer
Extend the FNO to an `AFNOLayer` that uses block-diagonal complex weights (block_size parameter), adds adaptive soft-thresholding on the spectral coefficients, and includes a residual connection. Verify that with threshold=0 and block_size=channels, it reduces to standard FNO. Verify gradient flow with torch.autograd.

### Prompt 3 — FFTNet Adaptive Spectral Filter
Implement `FFTNetLayer` with a learnable spectral filter that is input-dependent (computed from a global context vector via a small MLP), modReLU activation on real/imaginary parts, and IFFT. Verify modReLU(z) = z * sign(|z| + b) for scalar test cases. Verify O(n log n) compute scaling empirically.

### Prompt 4 — Spectral Transformer Block
Assemble a `SpectralTransformerBlock` combining: spectral mixing (AFNO or FFTNet), a feed-forward network (SwiGLU), layer normalization, and residual connections. Verify forward pass on random input. Verify the block processes seq_len=512 and seq_len=2048 without changing parameter count (resolution invariance).

### Prompt 5 — Tiny Language Model with Spectral Attention
Build a 2-layer spectral transformer LM (embedding → 2× spectral blocks → unembedding) with ~100K parameters. Train on a tiny Shakespeare character dataset for 1000 steps. Verify loss decreases. Compare perplexity against an equivalent self-attention model with the same parameter count.

### Prompt 6 — Resolution Invariance Test
Train a spectral transformer on seq_len=64, then evaluate on seq_len=256 without retraining. Measure perplexity. Run the same test with standard attention (zero-shot extrapolation). Verify spectral model degrades more gracefully.

### Prompt 7 — Complexity Benchmark
Benchmark wall-clock time for spectral mixing vs. self-attention at seq_len = 128, 256, 512, 1024, 2048, 4096. Plot O(n log n) vs O(n²). Verify the crossover point and the scaling exponent matches theory.

### Prompt 8 — Spectral Weight Analysis
After training the tiny LM, extract the learned spectral weight matrices. Visualize magnitude vs. mode index. Verify that higher modes have lower magnitude (spectral decay). Verify soft-thresholding zeros out a fraction of modes (sparsity).

### Prompt 9 — Hardware-Oriented Quantization
Implement post-training quantization of the spectral weights to int8 (real and imaginary parts separately). Implement int8 FFT using the Gold FFT algorithm or lookup tables. Verify quantized model perplexity is within 5% of float32. Measure bit-level representation of each weight.

### Prompt 10 — Fixed-Point Arithmetic Simulator
Build a fixed-point arithmetic library (Q4.4, Q8.8, Q2.6 formats) with add, multiply, complex multiply, and FFT butterfly operations. Verify against float reference with known tolerance. Report overflow/underflow rates per format.

---

## Phase 2: Hardware Design (Verilog / SystemVerilog)

### Prompt 11 — Radix-2 Butterfly Core
Write a synthesizable Verilog module `butterfly2` that computes the radix-2 FFT butterfly: (a, b) → (a + W·b, a - W·b) where W is a complex twiddle factor. Parameterize data width. Verify with testbench: input (3+4j, 1+2j), W=(0.707+0.707j), check output against Python.

### Prompt 12 — Radix-4 Butterfly Core
Write `butterfly4` for radix-4 FFT: 4 inputs → 4 outputs with 3 complex twiddle multiplications. Verify against Python numpy.fft for a 4-point DFT. Measure gate count after Yosys synthesis with SKY130.

### Prompt 13 — Twiddle Factor ROM
Generate a twiddle factor ROM module initialized with sin/cos values for N=256 point FFT, Q8.8 fixed-point. Use `$readmemh` to load a generated hex file. Verify that ROM output matches numpy for indices 0..127.

### Prompt 14 — Pipelined FFT Stage
Build a single pipeline stage of a decimation-in-time FFT: input buffer → butterfly → twiddle multiply → output buffer, with valid/stall handshaking. Verify pipeline throughput = 1 sample/clock after fill. Verify 256-point FFT via 8 cascaded stages.

### Prompt 15 — 256-Point FFT Module
Assemble a complete `fft_256` module using 4 radix-4 stages (256 = 4⁴). Include input bit-reversal, pipelined stages, and output reordering. Verify against Python numpy.fft.fft for a known test vector. Synthesize with Yosys + SKY130; report area and max frequency.

### Prompt 16 — Spectral Weight Memory & Multiply
Design a module `spectral_multiply` that stores k=32 complex spectral weights (Q8.8) in a register file, reads them sequentially, multiplies each by the corresponding FFT mode, and applies soft-thresholding (|w| < threshold → 0). Verify the multiply-accumulate matches Python for a 32-mode example.

### Prompt 17 — IFFT Module
Build `ifft_256` as the conjugate-FFT approach: conjugate input → FFT → conjugate output → scale by 1/N. Verify against Python numpy.fft.ifft. Verify that FFT followed by IFFT recovers the original (within fixed-point tolerance).

### Prompt 18 — modReLU Activation Module
Implement `modReLU` in hardware: given complex (re, im) and bias b, compute output = z * sign(|z| + b). Use CORDIC for magnitude. Verify against Python for 16 test vectors. Synthesize and report area.

### Prompt 19 — Spectral Mixer Top Module
Assemble `spectral_mixer` top-level: input_seq (256 × d) → fft_256 → spectral_multiply (k=32, block_size=8) → ifft_256 → modReLU → output. Include a Wishbone bus interface for loading weights and reading results. Verify end-to-end against the Python AFNO simulation with int8 weights.

### Prompt 20 — Wishbone Bus Interface
Write a Wishbone-compatible register interface: 16 control/status registers accessible via a 32-bit Wishbone bus. Registers for: start, done, mode_count, block_size, threshold, base addresses for weight memory and data buffer. Write a Verilog testbench that drives the bus to load weights and start computation. Verify register read/write.

---

## Phase 3: Verification & Tapeout Preparation

### Prompt 21 — Cocotb Testbench for FFT
Write a cocotb testbench in Python that drives random complex inputs into `fft_256`, collects outputs, and compares against numpy. Run 1000 random vectors and report max error. Verify all errors < 2 ULP in Q8.8.

### Prompt 22 — Cocotb Testbench for Spectral Mixer
Write a cocotb testbench that loads spectral weights via Wishbone, starts the spectral_mixer, and compares the output against the Python simulation (with matching quantization). Verify end-to-end functional correctness over 100 random inputs.

### Prompt 23 — Gate-Level Simulation
After Yosys synthesis of `spectral_mixer`, run gate-level simulation with the cocotb testbench from Prompt 22. Verify timing (no setup/hold violations at 50 MHz). Report total gate count, total SRAM bits, and critical path.

### Prompt 24 — Power Estimation
Use OpenSTA to estimate power at 50 MHz, 1.8V VDD. Report static and dynamic power. Compare estimated throughput (tokens/sec) and energy/token against a simulated O(n²) attention accelerator of equivalent area.

### Prompt 25 — GDSII Generation via OpenLane
Create an OpenLane configuration for the `spectral_mixer` design targeting SKY130. Run the full flow (synthesis → floorplan → placement → CTS → routing → GDSII). Verify DRC clean and LVS clean. Report die area, utilization, and routing congestion.

### Prompt 26 — Tiny Tapeout Wrapper
Wrap the `spectral_mixer` in the Tiny Tapeout Verilog template (TT macro: `tt_um_spectral_silicon`). Ensure the design fits within the tile area constraint. Generate the GDSII compatible with Tiny Tapeout submission format. Verify the wrapper passes tt_tools checks.

### Prompt 27 — PCB / Breakout Board Spec
Write a spec for the carrier PCB: chip pinout mapping, power sequencing (1.8V core, 3.3V I/O), clock distribution, SPI/UART interface to host, and LED status indicators. Include a schematic netlist for the carrier board.

---

## Phase 4: Software Stack & System Integration

### Prompt 28 — Host Driver Library
Write a Python host driver (`spectral_driver.py`) that communicates with the fabricated chip via SPI: load spectral weights, write input data, trigger computation, read back results. Include a `SpectralChip` class with methods `load_weights()`, `run_inference()`, `read_output()`. Verify against a software simulation mode (`sim=True`).

### Prompt 29 — Compiler: PyTorch → Spectral Chip
Write a compiler module that takes a trained PyTorch spectral transformer, extracts the spectral weights, quantizes them to int8, and emits a binary blob for the chip. Verify that the emitted blob, when loaded into the software simulation, produces the same output as the PyTorch model (within quantization error).

### Prompt 30 — End-to-End Demo
Assemble a complete demo: train the tiny spectral LM (Prompt 5), compile it (Prompt 29), run inference on the software simulation of the chip (Prompt 19), and display the generated text. Write a CLI tool `spectral_silicon_demo` that runs the full pipeline. Verify that the demo output is coherent text and that the spectral chip simulation matches the PyTorch model output within 5% perplexity.