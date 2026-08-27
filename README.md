🔥 # Spectral Silicon — A Custom Chip That Makes LLM Inference Faster by Ditching Attention

## What If We Stopped Doing Attention in Hardware?

Here's the thing: every large language model today is bottlenecked by **self-attention** — the O(n²) operation where every token has to look at every other token. It's powerful, yes. But in silicon, it's a monster. A 256×256 attention matrix at 130nm needs 256KB of SRAM and 65,000 multiply-accumulators. That doesn't fit on a small chip.

So we asked a simple question: **what if we replaced attention with something that scales better — and is actually easier to build in hardware?**

That's Spectral Silicon. A custom ASIC that replaces O(n²) attention with O(n log n) **spectral mixing** — Fourier transforms, diagonal weight multiplies, and a couple of activation functions — all implemented directly in silicon on the open-source SkyWater SKY130 process. The whole chip fits in ~40,000 gates, about 1 square millimeter.

You can fabricate it for $50–$500 through Tiny Tapeout.

---

## The Big Idea (In Plain English)

Instead of computing how much every token relates to every other token (that's attention), we do something different:

1. **Transform** the token sequence into the frequency domain using an FFT (fast — O(n log n))
2. **Scale** each frequency by a learned weight (cheap — diagonal multiply, not a full matrix)
3. **Transform back** to the time domain with an inverse FFT (also O(n log n))
4. **Activate** with modReLU — a magnitude-gated ReLU that works on complex numbers

That's it. The whole pipeline is: **FFT → multiply → IFFT → activate**. No attention matrix. No O(n²) bottleneck.

The math comes from **neural operators** — specifically the Fourier Neural Operator (FNO, Li et al. 2021) and its adaptive variant AFNO (Guibas et al. 2021, NVIDIA). These were originally built for solving partial differential equations, then adapted as token mixers for transformers. We took the next step: putting them in hardware.

A bonus: spectral mixing is **resolution-invariant**. Train on 512-token sequences, run on 4096-token sequences with zero retraining. The weights don't depend on sequence length — they depend on frequency modes.

---

## How the Chip Works

```
Token Embeddings
       │
       ▼
┌──────────────┐
│  FFT Engine   │   256-point radix-4 butterfly network
│  (O(n log n)) │   Transforms tokens → frequency domain
└──────┬───────┘
       ▼
┌──────────────┐
│  Spectral     │   32 learnable complex weights (block-diagonal)
│  Multiply     │   Soft-threshold: zero out small modes (sparsity)
└──────┬───────┘
       ▼
┌──────────────┐
│  IFFT Engine  │   Same butterfly network, conjugate method
│  (O(n log n)) │   Transforms frequency → time domain
└──────┬───────┘
       ▼
┌──────────────┐
│  modReLU +    │   Magnitude-gated activation
│  Activation   │   Keeps or zeros each output element
└──────┬───────┘
       ▼
   Output Tokens
```

| Part | What It Does | Gate Count |
|------|-------------|------------|
| FFT engine (radix-4, 256-point) | Token sequence → frequency domain | ~15K |
| Spectral weight multiply (32 modes) | Scale each frequency by a learned weight | ~8K |
| IFFT engine (shared with FFT via conjugate method) | Frequency → time domain | ~15K (shared) |
| modReLU + soft-threshold | Activation + sparsity | ~2K |
| **Total estimated** | | **~40K gates, ~1mm²** |

For comparison: a 256×256 attention matrix at 130nm would need 256KB SRAM + 65K MACs — far beyond what fits on a Tiny Tapeout tile.

---

## The Full Workflow: From Idea to Silicon

This project isn't just a concept — it's a complete, buildable design with a clear path from Python simulation to physical chips. Here's every stage, end to end.

### Stage 1: Simulate in Python

Before building hardware, we prove the math works. The `spectral_silicon/` package contains a full ML simulation of the chip in PyTorch:

- **`fno.py`** — Fourier Neural Operator layer (the core math)
- **`afno.py`** — Adaptive FNO with block-diagonal weights and soft-thresholding
- **`fftnet.py`** — FFTNet adaptive spectral filter with modReLU activation
- **`transformer.py`** — A spectral transformer block that drops in as an attention replacement
- **`fixedpoint.py`** — Simulates the Q8.8 fixed-point arithmetic the chip actually uses
- **`quantize.py`** — Quantizes spectral weights to int8 for the chip
- **`model.py`** — A tiny character-level language model (~100K params) to prove end-to-end training
- **`compiler.py`** — Compiles a trained PyTorch model into a binary blob the chip can load

```bash
# Install dependencies
pip install torch numpy pytest

# Run the ML simulation tests (12 test files, 285 test cases)
pytest tests/ -v

# Train the tiny spectral language model (uses a built-in Shakespeare excerpt)
python -m spectral_silicon.model --train --steps 1000

# Generate text from the trained model
python -m spectral_silicon.model --prompt "ROMEO:"
```

The training is standard PyTorch — Adam optimizer, cross-entropy loss, gradient clipping. The default corpus is a Shakespeare excerpt built into the repo so it runs with no external data, but you can train on any text by passing `text="your corpus"`.

### Stage 2: Build the Hardware in Verilog

Once the math is validated, we translate it into RTL. The `rtl/` directory contains 44 Verilog modules — the complete hardware design in Verilog-2005:

**Core datapath** (the chip's main pipeline):
- `fft_256.v` — 256-point FFT (4 radix-4 stages, in-place RAM)
- `ifft_256.v` — 256-point IFFT (conjugate method: `IFFT(x) = conj(FFT(conj(x)))/N`)
- `spectral_multiply.v` — Complex weight multiply + soft-thresholding
- `modrelu.v` — modReLU activation (magnitude-gated ReLU for complex numbers)
- `spectral_mixer.v` — Top-level module wiring the full pipeline together
- `wishbone_if.v` — Wishbone bus interface (how the host talks to the chip)

**Butterfly cores** (the computational heart of every FFT stage):
- `butterfly2.v` — Radix-2 butterfly (for reference)
- `butterfly4.v` — Radix-4 butterfly (combinational, the workhorse)
- `pipelined_butterfly4.v` — 2-stage pipelined version (v4 improvement)
- `triple_twiddle_rom.v` — 3-port parallel twiddle ROM (v4, eliminates serial reads)
- `twiddle_rom.v` — Single-port twiddle factor ROM

**Efficiency improvements** (V2, 20 modules):
- `fft_ifft_256.v` — Shared FFT/IFFT engine (saves ~15K gates)
- `booth_mult.v`, `truncated_booth.v` — Booth-encoded multipliers (faster, smaller)
- `bfp_unit.v` — Block floating-point for wider dynamic range
- `carry_save_acc.v`, `fma_butterfly.v` — Faster accumulators
- `pingpong_ram.v` — Ping-pong dual-buffer memory (zero pipeline bubbles)
- `shadow_weights.v` — Weight prefetch with shadow registers
- `dma_burst.v` — Burst-mode DMA for weight loading
- `conflict_free_addr.v` — Conflict-free memory banking
- `bitreversal_router.v` — Hardware bit-reversal (eliminates software pre-processing)
- `rfft_256.v` — Real-input FFT (halves computation for real-valued inputs)
- `twiddle_symmetry.v` — Exploit twiddle symmetry (4× ROM compression)
- `zero_skip_mac.v` — Zero-skipping MAC with dummy cycles
- `mode_interleave.v` — Interleaved mode processing
- `early_ifft.v` — Early IFFT start with overlap
- `configurable_fft.v` — Configurable FFT size (128/256/512)
- `deep_pipeline_fft.v` — 8-stage deep FFT pipeline
- `dual_channel.v` — Two-channel parallel datapath
- `dvfs_tracker.v` — Dynamic voltage-frequency scaling

**Security modules** (V2, 10 modules):
- `constant_time_mac.v` — Constant-time spectral multiply (no timing leaks)
- `power_flattening.v` — Decoy MAC for power-trace obfuscation
- `weight_crypto.v` — Weight bitstream encryption
- `integrity_hash.v` — SHA-256 weight integrity verification
- `em_shield.v` — EM shielding via top metal layer
- `adaptive_k.v` — Adaptive mode count (host-configurable)

**Speed improvements** (V4, 5 new modules):
- `streaming_ifft_loader.v` — Overlaps IFFT loading with spectral multiply output (~30% latency reduction)
- `mode_skip_multiply.v` — Bypasses the multiplier for truncated modes (87.5% power savings)
- `batch_channel_controller.v` — Auto-sequences all 64 channels in one command (~6,300 cycles saved)
- Plus `pipelined_butterfly4.v` and `triple_twiddle_rom.v` (listed above)

**LLM component modules** (V5, 9 new modules — full transformer on chip):
- `residual_add.v` — Residual connection adder (mixer_out + input, with saturation)
- `layernorm.v` — LayerNorm for d_model=64 (two-phase: accumulate → normalize)
- `gelu_silu.v` — GELU/SiLU activation for the FFN (piecewise linear, mode-selectable)
- `softmax.v` — Softmax over vocab=128 (exp LUT + reciprocal, two-phase)
- `topk_sampler.v` — Top-k token sampling with LFSR randomness (k=1..8)
- `token_embedding.v` — Token embedding lookup (128×64×16-bit = 16KB)
- `unembedding.v` — Logits projection (d_model=64 → vocab=128, serialized MAC)
- `ffn.v` — Feed-forward network (d_model=64, d_ffn=128, GELU, serialized MAC)
- `weight_cache.v` — Multi-layer weight cache SRAM (4 layers × 32 modes = 4KB)

**Tapeout wrappers**:
- `tt_wrapper.v` / `tt_wrapper_v2.v` — Tiny Tapeout package wrappers
- `tt_wrapper_v2.v` — Updated for V2+ module set

Every RTL module is paired with a cocotb testbench in `tb/` (9 testbench files) and verified with Icarus Verilog (`iverilog -g2005`).

```bash
# Generate twiddle factor hex files (sin/cos lookup tables for the FFT)
python scripts/gen_twiddles.py --n 256 --format Q8.8

# Verify all RTL compiles
for f in rtl/*.v; do iverilog -g2005 -o /dev/null "$f" || echo "FAIL: $f"; done

# Run cocotb testbenches (requires iverilog + cocotb)
cd tb && make  # each testbench has a Makefile snippet
```

### Stage 3: Compile the Model to a Chip Binary

Once the model is trained and the hardware is built, the compiler bridges the two:

```bash
# Quantize the trained weights to int8 and compile to a binary blob
python -m spectral_silicon.compiler --model weights.pt --output chip_blob.bin
```

The compiler:
1. Extracts spectral weight tensors from the PyTorch model
2. Quantizes them to Q8.8 fixed-point (matching the hardware datapath)
3. Packs them into a flat binary blob the host loads via the Wishbone bus
4. Embeds a SHA-256 integrity hash so the chip can verify weights at boot

### Stage 4: Synthesize to GDSII (RTL → Silicon Layout)

The open-source EDA toolchain turns Verilog into a physical chip layout:

```
Verilog RTL
  → Yosys (synthesis: Verilog → gate-level netlist)
  → OpenROAD (floorplan → placement → clock tree → routing)
  → OpenSTA (timing analysis)
  → Magic (DRC — design rule check)
  → Netgen (LVS — layout vs. schematic)
  → GDSII (final layout file sent to foundry)
```

The `openlane/config.json` file configures the OpenLane automated flow for SKY130:

```json
{
    "DESIGN_NAME": "spectral_mixer",
    "PDK": "sky130A",
    "CLOCK_PERIOD": "20",          // 50 MHz
    "DIE_AREA": "0 0 1000 1000",   // 1mm × 1mm
    "VERILOG_FILES": ["rtl/butterfly4.v", "rtl/fft_256.v", ...]
}
```

```bash
# Run the full OpenLane flow (RTL → GDSII, typically <24 hours)
openlane openlane/config.json

# View the resulting layout
klayout -e gds/spectral_mixer.gds
```

### Stage 5: Submit for Fabrication

The `tapeout/info.yaml` file contains the Tiny Tapeout submission metadata:

```yaml
project_name: "Spectral Silicon"
description: "Neural-operator-based spectral mixing accelerator for LLM inference."
language: "Verilog"
clock_hz: 50_000_000   # 50 MHz
tile_size: 1            # 1 Tiny Tapeout tile
```

**Fabrication options:**

| Program | Cost | PDK | Notes |
|---------|------|-----|-------|
| **Tiny Tapeout** | ~$50–$500/tile | SKY130, GF180, IHP SG13G2 | Cheapest entry. Shared shuttle. |
| **Google Open MPW** | Free (open-source designs) | SKY130 | Intermittent availability. |
| **Efabless ChipIgnite** | ~$14,950/tapeout | SKY130 | Full custom tile, more area. |

Submit the GDSII + `info.yaml` to Tiny Tapeout. They handle the multi-project wafer shuttle, packaging, and PCB. You receive physical chips in a few months.

### Stage 6: Talk to the Fabricated Chip

Once you have physical chips, the `host/` directory has everything you need:

- **`spectral_driver.py`** — SPI driver for the fabricated chip (loads weights, sends data, reads results)
- **`demo.py`** — End-to-end CLI demo: load a compiled model → run inference → display output

```bash
# Run the end-to-end demo with a fabricated chip
python host/demo.py
```

The host computer connects to the chip over SPI (or the Wishbone bus on Tiny Tapeout's PCB), loads the compiled weight blob, sends token embeddings, and reads back the spectral mixer's output.

---

## Project Structure

```
spectral-silicon/
├── RESEARCH.md              # Full research notes: PDKs, neural operators, fabrication
├── IMPROVEMENTS.md           # 20 V2 efficiency + security improvements
├── PERFORMANCE.md           # 25 performance improvements (V2–V4)
├── PROMPTS.md               # 30 testable prompts / specifications
├── README.md                # This file
│
├── spectral_silicon/        # Python ML simulation & compiler (12 modules, 6,621 lines)
│   ├── fno.py               # Fourier Neural Operator layer
│   ├── afno.py              # Adaptive FNO (block-diagonal, soft-threshold)
│   ├── fftnet.py            # FFTNet adaptive spectral filter
│   ├── transformer.py       # Spectral transformer block & tiny LM
│   ├── model.py             # Tiny spectral LM (training + generation)
│   ├── fixedpoint.py        # Fixed-point arithmetic simulator
│   ├── quantize.py          # int8 quantization for spectral weights
│   ├── compiler.py          # PyTorch → chip binary blob compiler
│   ├── perf_sim.py          # Performance simulator
│   ├── security.py          # Security analysis (constant-time, logic locking)
│   ├── constant_time.py     # Constant-time operation verification
│   └── __init__.py
│
├── tests/                   # Python test suite (12 files, pytest)
│   ├── test_fno.py          # FNO layer correctness
│   ├── test_afno.py         # AFNO block-diagonal + soft-threshold
│   ├── test_fftnet.py       # FFTNet filter + modReLU
│   ├── test_transformer.py  # Spectral transformer block
│   ├── test_fixedpoint.py   # Q8.8 fixed-point arithmetic
│   ├── test_quantize.py     # int8 quantization
│   ├── test_compiler.py     # Model → binary compilation
│   ├── test_complexity.py   # O(n log n) scaling verification
│   ├── test_resolution.py   # Resolution invariance (train 512, run 4096)
│   ├── test_constant_time.py # Constant-time operation
│   ├── test_security.py     # Security properties
│   └── test_perf_sim.py     # Performance simulation
│
├── rtl/                     # Verilog hardware modules (53 files, ~10,000+ lines)
│   ├── (see Stage 2 above for full module listing)
│   └── twiddle_data/        # Generated twiddle factor hex files
│
├── tb/                      # cocotb testbenches (18 files)
│   ├── tb_butterfly.py      # Radix-4 butterfly tests
│   ├── tb_fft256.py         # 256-point FFT tests
│   ├── tb_spectral_mixer.py # Full pipeline tests
│   ├── tb_wishbone.py       # Bus interface tests
│   ├── tb_triple_twiddle.py # 3-port twiddle ROM (v4)
│   ├── tb_streaming_ifft.py # Overlapped IFFT loader (v4)
│   ├── tb_mode_skip.py      # Mode-skip multiply (v4)
│   ├── tb_pipelined_bf.py   # Pipelined butterfly (v4)
│   ├── tb_batch_channel.py # Batch channel controller (v4)
│   ├── tb_residual_add.py   # Residual connections (v5)
│   ├── tb_layernorm.py      # LayerNorm (v5)
│   ├── tb_gelu_silu.py      # GELU/SiLU activation (v5)
│   ├── tb_softmax.py        # Softmax (v5)
│   ├── tb_topk_sampler.py   # Top-k sampling (v5)
│   ├── tb_token_embedding.py # Token embeddings (v5)
│   ├── tb_unembedding.py   # Unembedding / logits (v5)
│   ├── tb_ffn.py            # Feed-forward network (v5)
│   └── tb_weight_cache.py  # Weight cache (v5)
│
├── openlane/                # OpenLane configuration for SKY130
│   └── config.json
├── tapeout/                 # Tiny Tapeout submission
│   └── info.yaml
├── host/                    # Host driver & demo
│   ├── spectral_driver.py   # SPI driver for fabricated chip
│   └── demo.py              # End-to-end CLI demo
├── scripts/                 # Build & utility scripts (7 files)
│   ├── gen_twiddles.py      # Generate twiddle factor hex
│   ├── run_tests.sh         # Run all tests
│   ├── benchmark.py         # Complexity benchmark
│   ├── benchmark_v2.py     # V2 benchmark
│   ├── benchmark_v3.py     # V3 benchmark
│   ├── perf_report.py      # Performance report generator
│   ├── gen_manifest.py     # Build manifest generator
│   └── synthesis_area_estimate.py  # Gate count estimator
└── data/                    # Training data & weight blobs
```

---

## The 25 Performance Improvements

The design evolved through 4 versions, each adding speed and efficiency improvements:

| Version | Improvements | Focus |
|---------|-------------|-------|
| **V1** | Initial build | Core architecture: FFT → multiply → IFFT → modReLU |
| **V2** | 1–20 (IMPROVEMENTS.md) | Efficiency (shared FFT/IFFT, Booth multipliers, ping-pong RAM) + security (constant-time, power flattening, weight encryption) |
| **V3** | 1–20 (PERFORMANCE.md) | Performance (BFP arithmetic, FMA butterfly, DMA, RFFT, deep pipeline, DVFS) |
| **V4** | 21–25 (PERFORMANCE.md) | Speed: triple twiddle ROM, streaming IFFT, mode-skip multiply, pipelined butterfly, batch channel controller |

See [IMPROVEMENTS.md](IMPROVEMENTS.md) for V1→V2 details and [PERFORMANCE.md](PERFORMANCE.md) for V3→V4 details.

---

## Why 130nm Is Enough

People hear "130nm" and think it's ancient. But the spectral approach changes what you need:

| What Attention Needs | What Spectral Mixing Needs |
|-----------------------|---------------------------|
| 256KB SRAM (for the attention matrix) | ~8KB (32 complex weights) |
| 65K MAC units (for pairwise products) | 3 complex multipliers (for 3 FFT twiddles) |
| Content-addressed memory | Fixed butterfly topology (regular wiring) |
| O(n²) compute | O(n log n) compute |

At 130nm, the FFT butterfly network (~15K gates) + diagonal spectral multiply (~8K gates) + modReLU (~2K gates) = ~40K gates, fitting in 1mm². An attention matrix engine at the same node wouldn't fit on a Tiny Tapeout tile at all.

---

## Key Research Behind the Design

- **FNO** (Li et al., ICLR 2021) — Fourier Neural Operator: O(n log n) spectral mixing, resolution-invariant
- **AFNO** (Guibas et al., 2021, NVIDIA) — Adaptive FNO for transformers: block-diagonal weights, soft-thresholding, replaces self-attention
- **FFTNet** (Fein-Ashley, 2025) — Learnable spectral filter + modReLU, O(n log n) alternative to attention
- **FNet** (Lee-Thorp et al., NeurIPS 2021) — Fixed FFT replacing attention achieves 92–97% of BERT accuracy, 80% faster training

Full citations and analysis in [RESEARCH.md](RESEARCH.md).

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/drwjkirkpatrick-web/spectral-silicon.git
cd spectral-silicon

# Set up the Python environment
python -m venv .venv && source .venv/bin/activate
pip install torch numpy pytest

# Run the ML simulation tests
pytest tests/ -v

# Train the tiny spectral LM (built-in corpus, no external data needed)
python -m spectral_silicon.model --train --steps 1000

# Generate text
python -m spectral_silicon.model --prompt "To be"

# Run the complexity benchmark
python scripts/benchmark.py

# Generate twiddle factors for hardware
python scripts/gen_twiddles.py --n 256 --format Q8.8

# Compile a trained model to a chip binary
python -m spectral_silicon.compiler --model weights.pt --output chip_blob.bin

# Verify all RTL compiles
for f in rtl/*.v; do iverilog -g2005 -o /dev/null "$f" && echo "OK: $f"; done

# Synthesize to GDSII (requires OpenLane installed)
openlane openlane/config.json
```

---

## License

MIT — open-source, required for Google Open MPW and Tiny Tapeout submission.

## Author

Walker — chip design enthusiast

## References

- [RESEARCH.md](RESEARCH.md) — Full research notes, PDK comparisons, neural operator theory, and citations
- [IMPROVEMENTS.md](IMPROVEMENTS.md) — V1→V2: 20 efficiency and security improvements
- [PERFORMANCE.md](PERFORMANCE.md) — V3→V4: 25 performance improvements
- [PROMPTS.md](PROMPTS.md) — 30 testable prompts that drove the initial build