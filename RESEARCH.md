# Spectral Silicon — Research Notes

## 1. Chip Fabrication for Individuals & Small Teams

### 1.1 Multi-Project Wafer (MPW) Shuttles

Individuals can get chips fabricated by sharing a wafer with other designers through MPW shuttle programs. Multiple designs are placed on a single reticle, splitting mask costs across participants.

### 1.2 Available Open-Source PDKs (Process Design Kits)

| PDK | Node | Foundry | Key Features | Status |
|-----|------|---------|-------------|--------|
| **SKY130** | 130nm | SkyWater Technology | 1P6M, mixed-signal CMOS, SRAM, analog | ✅ Production-proven, most mature open PDK |
| **GF180MCU** | 180nm | GlobalFoundries | 1P6M, 5V/3.3V devices, 7/9-track std cells | ✅ Open, growing community |
| **IHP SG13G2** | 130nm | IHP Microelectronics | SiGe BiCMOS, HBTs + CMOS, RF-capable | ✅ Early access, Tiny Tapeout support |
| **ICSprout55** | 55nm | Open ecosystem | Early-stage open PDK | 🔶 Experimental, not yet production-ready |

### 1.3 Fabrication Programs

| Program | Cost | PDK | Notes |
|---------|------|-----|-------|
| **Google/Efabless Open MPW** | **Free** (open-source designs) | SKY130 | Google-sponsored; open-source designs funded |
| **Tiny Tapeout** | ~$50–$500+ per tile (varies) | SKY130, GF180, IHP SG13G2 | Cheapest entry; Wokwi or Verilog; shared shuttle |
| **ChipIgnite (Efabless)** | ~$14,950/tapeout | SKY130 | Full custom tile; more area than Tiny Tapeout |
| **Cadence/SkyWater MPW** | Cost-shared | SKY130 | Commercial + open-source tools |
| **IHP MPW** | Subsidized | SG13G2 | BiCMOS/RF capability |

**Best path for a first chip:** Tiny Tapeout on SKY130 (digital) or IHP SG13G2 (if analog/RF needed). Google Open MPW offers free fabrication for open-source designs but has been intermittent.

### 1.4 Open-Source EDA Toolchain

| Tool | Role |
|------|------|
| **Yosys** | RTL synthesis (Verilog → netlist) |
| **OpenROAD** | Place & route, CTS, timing (RTL → GDSII in 24h) |
| **OpenLane** | Automated flow wrapper around OpenROAD for SKY130/GF180 |
| **Magic** | Analog layout, DRC |
| **Xschem** | Schematic capture |
| **Netgen** | LVS |
| **OpenSTA** | Static timing analysis |
| **KLayout** | GDSII viewer, DRC, visualization |
| **Verilator** | Verilog simulation |
| **ngspice** | SPICE simulation (analog) |

### 1.5 Design Flow (RTL → Silicon)

```
RTL (Verilog/Chisel)
  → Yosys synthesis → gate-level netlist
  → OpenROAD floorplan → placement → CTS → routing
  → OpenSTA timing analysis
  → DRC/LVS checks (Magic + Netgen)
  → GDSII export
  → Submit to foundry (Tiny Tapeout / MPW)
  → Receive packaged chips + PCB
```

---

## 2. Neural Operators — Theory & Architecture

### 2.1 What Are Neural Operators?

Neural operators generalize neural networks from learning mappings between finite-dimensional vectors to learning **operators between infinite-dimensional function spaces**. The key insight: instead of learning a mapping f: ℝⁿ → ℝᵐ, they learn **G: U → V** where U, V are function spaces.

This makes them **resolution-invariant** — train at one resolution, evaluate at another with zero retraining.

### 2.2 Fourier Neural Operator (FNO)

The FNO (Li et al., 2021) parameterizes the kernel integral directly in Fourier space:

```
v_{t+1} = σ(W·v_t + K(v_t))
```

where K is computed as:
1. FFT the input → frequency domain
2. Multiply by learnable weight tensor R (truncated to first k modes)
3. Inverse FFT → spatial domain

**Complexity:** O(n log n) instead of O(n²) for self-attention.

**Key property:** The spectral weight R operates on **global** information through every mode — no local windowing needed. Resolution-invariant.

### 2.3 DeepONet

Branch-trunk architecture: two networks approximate the operator by learning basis functions separately for the input function space (branch) and output function space (trunk). Their dot product gives the output.

### 2.4 Adaptive Fourier Neural Operator (AFNO) — *Critical for our chip*

Guibas et al. (NVIDIA/Caltech, 2021) adapted FNO for **vision transformers** as a token mixer replacing self-attention:

- **Block-diagonal channel mixing:** Reduces parameters, adds locality
- **Adaptive weight sharing:** Weights shared across tokens (resolution-invariant)
- **Soft-thresholding / shrinkage:** Sparsifies frequency modes → memory efficient
- **Quasi-linear O(n log n) complexity**, linear memory in sequence size
- Outperforms self-attention on few-shot segmentation at 65k sequence length

### 2.5 FFTNet — *Adaptive Spectral Filtering*

Fein-Ashley (USC, 2025): Replaces self-attention with:
1. FFT the token sequence
2. Apply learnable spectral filter (input-dependent, context-aware)
3. modReLU activation on real/imaginary parts
4. Inverse FFT
5. O(n log n) complexity, Parseval energy preservation

---

## 3. Novel Architecture: Spectral Attention Chip

### 3.1 Core Idea

Replace O(n²) self-attention with O(n log n) spectral mixing (FNO/AFNO/FFTNet) **in hardware**. The chip implements:

```
Token sequence → FFT (hardware radix-2/4) → Spectral weight multiply (analog/digital) 
→ IFFT → Activation → Output
```

This replaces the matrix-matrix attention multiply with:
- An O(n log n) FFT (highly regular, pipelineable)
- A diagonal/block-diagonal complex weight multiply (low parameter count)
- An O(n log n) IFFT

### 3.2 Why This Works in Hardware at 130nm

| Operation | HW Complexity at 130nm |
|-----------|------------------------|
| FFT/IFFT | Butterfly network — regular, pipelineable, well-understood IP |
| Spectral weight multiply | Diagonal complex multiply — massively parallel, low area |
| Soft-thresholding | Simple comparator + shrinkage — trivial in HW |
| Activation (modReLU) | Two comparators — trivial |
| Self-attention (baseline) | O(n²) MACs — would need huge SRAM + compute arrays |

At 130nm we **cannot fit a large attention matrix**, but we **can fit** an FFT butterfly + diagonal spectral mixer because:
- FFT is O(n log n) not O(n²)
- Spectral weights are diagonal (k modes × d channels) not full (n × n)
- The butterfly network is a fixed topology (no content-addressed memory needed)

### 3.3 Proposed Chip Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Spectral Silicon Die                │
│                                                      │
│  ┌──────────┐   ┌──────────────┐   ┌──────────┐    │
│  │ Input    │──→│  FFT Engine   │──→│ Spectral │    │
│  │ Buffer   │   │ (Radix-4)     │   │ Weight   │    │
│  │ (SRAM)   │   │  Pipeline     │   │ Multiply │    │
│  └──────────┘   └──────────────┘   │ (Block   │    │
│                                     │  Diagonal)│   │
│                                     └─────┬────┘   │
│                                           │        │
│  ┌──────────┐   ┌──────────────┐   ┌─────▼────┐   │
│  │ Output   │←──│  IFFT Engine  │←──│ Soft-    │   │
│  │ Buffer   │   │  (Radix-4)    │   │ Threshold│   │
│  │ (SRAM)   │   │  Pipeline    │   │ + modReLU│   │
│  └──────────┘   └──────────────┘   └──────────┘   │
│                                                      │
│  ┌─────────────────────────────────────────────┐    │
│  │          Config Register Interface            │    │
│  │  (modes k, block size, activation threshold)  │    │
│  └─────────────────────────────────────────────┘    │
│                                                      │
│  ┌─────────────────────────────────────────────┐    │
│  │          Wishbone / AXI-Lite Bus IF          │    │
│  └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

### 3.4 Target Process: SKY130

- **Node:** 130nm, 1P6M
- **Die area target:** ~1mm × 1mm (Tiny Tapeout tile) or larger (ChipIgnite)
- **Clock:** 10–50 MHz (sufficient for FFT pipeline)
- **I/O:** Digital serial (UART/SPI) or parallel bus (Wishbone)
- **Power:** 1.8V core, 3.3V I/O
- **Key constraint:** No on-chip NVM for weights → weights loaded via bus at init

### 3.5 Analog Extension (Future)

On SG13G2 or SKY130 analog tiles:
- Use charge-domain FFT (switched-capacitor) for ultra-low-power spectral mixing
- Analog crossbar arrays for the spectral weight multiply (compute-in-memory)
- ADC/DAC at the boundary for digital interface

---

## 4. References

1. Li, Z. et al. "Fourier Neural Operator for Parametric PDEs." ICLR 2021. arXiv:2010.08895
2. Guibas, J. et al. "Adaptive Fourier Neural Operators: Efficient Token Mixers for Transformers." 2021. arXiv:2111.13587
3. Fein-Ashley, J. "The FFT Strikes Back: An Efficient Alternative to Self-Attention." 2025. arXiv:2502.18394
4. Lee-Thorp, J. et al. "FNet: Mixing Tokens with Fourier Transforms." NeurIPS 2021.
5. Lu, L. et al. "Learning Nonlinear Operators via DeepONet." Nature Machine Intelligence, 2021.
6. OpenROAD Project — https://theopenroadproject.org/
7. SkyWater SKY130 PDK — https://github.com/google/skywater-pdk
8. Tiny Tapeout — https://tinytapeout.com/
9. Efabless ChipIgnite — https://chipfoundry.io/
10. GlobalFoundries GF180MCU PDK — https://github.com/fossi-foundation/open-pdks
11. IHP SG13G2 PDK — https://www.ihp-microelectronics.com/services/research-and-prototyping-service/fast-design-enablement/open-source-pdk
12. Kovachki, N. et al. "Neural Operator: Learning Maps Between Complex Function Spaces." 2021. arXiv:2108.08481
13. NVIDIA Research. "Neural Operators with Localized Integral and Differential Kernels." 2024.
14. "Memory Is All You Need: Compute-in-Memory for LLM Inference." arXiv:2406.08413, 2024.