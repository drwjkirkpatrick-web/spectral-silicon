# Spectral Silicon — A Neural-Operator-Based Custom Chip for LLM Inference

## Overview

Spectral Silicon is a custom ASIC design that replaces O(n²) self-attention with O(n log n) spectral mixing (Fourier Neural Operator / AFNO) implemented directly in silicon. The project targets the SkyWater SKY130 130nm open-source PDK and is designed for fabrication via Tiny Tapeout or Efabless Open MPW.

## The Novel Idea

Standard LLM attention computes pairwise token interactions in O(n²) time and memory — the dominant bottleneck for long-context inference. **Neural operators** (FNO, AFNO, FFTNet) replace this with:

1. **FFT** the token sequence → frequency domain (O(n log n))
2. **Multiply** by learnable spectral weights (diagonal, O(k·d) parameters)
3. **IFFT** back → spatial domain (O(n log n))

This is **resolution-invariant**: train at seq_len=512, run at seq_len=4096 with zero retraining. And it maps beautifully to hardware — an FFT butterfly network is regular, pipelineable, and needs a fraction of the area of an attention matrix engine at 130nm.

## Architecture

```
Token Sequence → FFT Engine (256-point, radix-4) → Spectral Weight Multiply
(block-diagonal, 32 modes) → Soft Threshold + modReLU → IFFT Engine → Output
```

| Component | Complexity | HW Cost @ 130nm |
|-----------|-----------|-----------------|
| 256-pt FFT (radix-4 pipeline) | O(n log n) | ~15K gates |
| Spectral weight multiply (32 modes) | O(k·d) | ~8K gates + weight regs |
| IFFT (conjugate-FFT) | O(n log n) | ~15K gates (shared) |
| modReLU + soft-threshold | O(n) | ~2K gates |
| **Total (estimated)** | **O(n log n)** | **~40K gates, ~1mm²** |

Compare: a 256×256 attention matrix at 130nm would need ~256KB SRAM + 65K MACs — far exceeding Tiny Tapeout tile area.

## Project Structure

```
spectral-silicon/
├── RESEARCH.md              # Research notes: PDKs, neural operators, fabrication
├── PROMPTS.md               # 30 testable prompts / specifications
├── README.md                # This file
├── spectral_silicon/        # Python ML simulation & compiler
│   ├── __init__.py
│   ├── fno.py               # Fourier Neural Operator layer
│   ├── afno.py              # Adaptive FNO (block-diagonal, soft-threshold)
│   ├── fftnet.py            # FFTNet adaptive spectral filter
│   ├── transformer.py       # Spectral transformer block & tiny LM
│   ├── fixedpoint.py        # Fixed-point arithmetic simulator
│   ├── quantize.py          # int8 quantization for spectral weights
│   ├── compiler.py          # PyTorch → chip binary blob compiler
│   └── model.py             # Tiny spectral LM definition
├── tests/                   # Python test suite (pytest)
│   ├── test_fno.py
│   ├── test_afno.py
│   ├── test_fftnet.py
│   ├── test_transformer.py
│   ├── test_fixedpoint.py
│   ├── test_quantize.py
│   ├── test_complexity.py
│   ├── test_resolution.py
│   └── test_compiler.py
├── rtl/                     # Verilog hardware modules
│   ├── butterfly2.v         # Radix-2 butterfly
│   ├── butterfly4.v         # Radix-4 butterfly (combinational)
│   ├── pipelined_butterfly4.v  # 2-stage pipelined radix-4 butterfly (v4)
│   ├── triple_twiddle_rom.v    # 3-port parallel twiddle ROM (v4)
│   ├── twiddle_rom.v        # Twiddle factor ROM
│   ├── fft_stage.v          # Pipelined FFT stage
│   ├── fft_256.v            # 256-point FFT module
│   ├── ifft_256.v           # 256-point IFFT (conjugate method)
│   ├── spectral_multiply.v # Spectral weight multiply + soft-threshold
│   ├── mode_skip_multiply.v   # Multiplier bypass for truncated modes (v4)
│   ├── modrelu.v            # modReLU activation
│   ├── spectral_mixer.v    # Top-level spectral mixer
│   ├── streaming_ifft_loader.v  # Overlapped IFFT loader (v4)
│   ├── batch_channel_controller.v  # Auto-sequence all D channels (v4)
│   ├── wishbone_if.v        # Wishbone bus interface
│   ├── tt_wrapper.v         # Tiny Tapeout wrapper
│   └── twiddle_data/        # Generated twiddle hex files
├── tb/                      # Verilog testbenches & cocotb
│   ├── tb_butterfly.py
│   ├── tb_fft256.py
│   ├── tb_spectral_mixer.py
│   ├── tb_wishbone.py
│   ├── tb_triple_twiddle.py    # v4
│   ├── tb_streaming_ifft.py   # v4
│   ├── tb_mode_skip.py        # v4
│   ├── tb_pipelined_bf.py     # v4
│   └── tb_batch_channel.py    # v4
├── openlane/                # OpenLane configuration for SKY130
│   └── config.json
├── tapeout/                 # Tapeout submission files
│   └── info.yaml
├── host/                    # Host driver & demo
│   ├── spectral_driver.py   # SPI driver for fabricated chip
│   └── demo.py              # End-to-end CLI demo
├── scripts/                 # Build & utility scripts
│   ├── gen_twiddles.py      # Generate twiddle factor hex
│   ├── run_tests.sh         # Run all tests
│   └── benchmark.py         # Complexity benchmark
└── data/                    # Training data & weight blobs
```

## Getting Started

```bash
# Install Python dependencies
pip install torch numpy pytest

# Run the ML simulation tests
pytest tests/ -v

# Run a complexity benchmark
python scripts/benchmark.py

# Train the tiny spectral LM
python -m spectral_silicon.model --train --steps 1000

# Generate twiddle factors for hardware
python scripts/gen_twiddles.py --n 256 --format Q8.8

# Compile a trained model to chip binary
python -m spectral_silicon.compiler --model weights.pt --output chip_blob.bin

# Run the end-to-end demo
python host/demo.py
```

## Key Findings from Research

### Fabrication Path
- **SkyWater SKY130** (130nm) is the most mature open-source PDK — production-proven, free PDK, active community
- **Tiny Tapeout**: cheapest entry (~$50–$500 per tile), supports SKY130, GF180, and IHP SG13G2
- **Google Open MPW**: free fabrication for open-source designs (intermittent availability)
- **OpenLane + OpenROAD**: full RTL→GDSII flow in <24 hours, no proprietary tools needed

### Neural Operator Theory
- **FNO** (Li et al. 2021): O(n log n) spectral mixing, resolution-invariant, originally for PDEs
- **AFNO** (Guibas et al. 2021, NVIDIA): adapts FNO for transformers — block-diagonal weights, soft-thresholding, replaces self-attention
- **FFTNet** (Fein-Ashley 2025): learnable spectral filter + modReLU, O(n log n) alternative to attention
- **FNet** (Google 2021): fixed FFT replacing attention achieves 92–97% of BERT accuracy, 80% faster training

### Why 130nm Is Sufficient
At 130nm we cannot fit a large attention matrix (needs ~256KB SRAM + 65K MACs). But the spectral approach only needs:
- A pipelined FFT butterfly network (~15K gates) — fixed topology, no content-addressed memory
- Diagonal/block-diagonal complex weight multiply (~8K gates) — O(k·d) not O(n²)
- Simple activation (modReLU, soft-threshold) — ~2K gates

Total ~40K gates fits comfortably in a Tiny Tapeout tile (~1mm² at 130nm).

## License

MIT (open-source, required for Google Open MPW / Tiny Tapeout submission)

## Author

Walker — chip design enthusiast

## References

See [RESEARCH.md](RESEARCH.md) for the full research notes and citations.