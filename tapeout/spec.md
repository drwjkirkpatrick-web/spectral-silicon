# Specification & Design

## Spectral Silicon — Neural-Operator-Based Custom Chip for LLM Inference

### The Problem

Standard LLM attention uses O(n²) matrix-matrix multiplies — the dominant bottleneck for long-context inference. At 130nm, a 256×256 attention matrix would need ~256KB SRAM + 65K MACs, far exceeding Tiny Tapeout tile area.

### The Solution

Replace self-attention with **spectral mixing** using Fourier Neural Operators (FNO/AFNO/FFTNet):

1. FFT the token sequence → frequency domain (O(n log n))
2. Multiply by learnable spectral weights (diagonal, O(k·d) parameters)
3. IFFT back → spatial domain (O(n log n))
4. Soft-threshold + modReLU activation

### Why This Works at 130nm

| Operation | HW Cost |
|-----------|---------|
| 256-pt FFT (radix-4 pipeline) | ~15K gates |
| Spectral weight multiply (32 modes) | ~8K gates + weight regs |
| IFFT (conjugate-FFT) | ~15K gates (shared) |
| modReLU + soft-threshold | ~2K gates |
| **Total** | **~40K gates, ~1mm²** |

### Resolution Invariance

Spectral weights operate on frequency modes, not spatial positions. Train at seq_len=64, evaluate at seq_len=256 with zero retraining. Standard attention cannot do this — it needs n×n weight matrices.

### Complexity

| Method | Time | Memory | Params |
|--------|------|--------|--------|
| Self-attention | O(n²) | O(n²) | O(n²) |
| FNO/AFNO | O(n log n) | O(n) | O(k·d) |
| FFTNet | O(n log n) | O(n) | O(k·d) |

### Chip Architecture

```
Input Buffer (SRAM) → FFT Engine (256-pt radix-4) → Spectral Weight Multiply
(block-diagonal, 32 modes, soft-thresholded) → IFFT Engine → modReLU → Output Buffer (SRAM)
```

Control via Wishbone bus: load weights, start computation, read results.

### Target Process: SKY130

- Node: 130nm, 1P6M
- Die area: ~1mm × 1mm (Tiny Tapeout tile)
- Clock: 50 MHz
- Core voltage: 1.8V, I/O: 3.3V
- Weights loaded at init via bus (no on-chip NVM)