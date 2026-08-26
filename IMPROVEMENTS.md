# Spectral Silicon — 20 Efficiency & Security Improvements

## Efficiency Improvements (1–10)

### 1. Shared FFT/IFFT Engine
Replace separate `fft_256` and `ifft_256` modules with a single `fft_ifft_256` that uses the conjugate method for IFFT: `IFFT(x) = conj(FFT(conj(x))) / N`. A mode bit selects FFT vs IFFT. **Saves ~15K gates** — nearly halves the FFT area.

### 2. Single-Path Delay Feedback (SDF) Pipeline
Replace the current 4-stage cascaded pipeline with a radix-4 SDF architecture. SDF uses a single butterfly unit that processes data sequentially through delay-line feedback, requiring only **1 complex multiplier** instead of 3 per stage. Reduces multiplier count from 12 to 3 for the full 256-point FFT.

### 3. In-Place Computation with Dual-Port SRAM
Replace register-based input/output buffers with a single dual-port SRAM block. The FFT operates in-place: butterfly results written back to the same memory locations. Eliminates the separate output buffer SRAM, **halving buffer area**.

### 4. Clock Gating for Idle Stages
Add clock gating cells (ICG — integrated clock gating) to each pipeline stage. When a stage is not actively processing (between FFT computations), its clock is gated off. At 130nm, this provides **20–40% dynamic power reduction** for typical inference workloads with idle gaps.

### 5. CORDIC Twiddle Generation
Replace the twiddle factor ROM (256 entries × 16 bits × 2 = 8KB) with a CORDIC rotator that computes sin/cos on-the-fly. CORDIC uses only shift-add operations — no multipliers, no ROM. **Saves ~8KB of ROM area** at the cost of 16 clock cycles per twiddle pair. For a pipelined design, this is amortized.

### 6. Quantized Twiddle Factors to Q6.10
Reduce twiddle factor precision from Q8.8 (16-bit) to Q6.10 — the extra fractional bits improve FFT accuracy while the reduced integer range (±32) is sufficient for sin/cos. This allows using **14-bit data paths** instead of 16-bit, reducing multiplier area by ~20%.

### 7. Block-Diagonal Weight Compression
The spectral weight matrix is already block-diagonal, but we can further compress by sharing weights across mode groups. Store only k/2 unique weight blocks and use a permutation to generate the other half (Hermitian symmetry of real-input FFT). **Halves weight storage** from 32×8×16-bit to 16×8×16-bit.

### 8. Merged Soft-Threshold + modReLU
Combine the soft-thresholding and modReLU activation into a single fused operation. Both operate on complex spectral coefficients and both involve magnitude comparison. Fusing them eliminates one magnitude computation and one register stage, **saving ~1K gates and 1 pipeline stage**.

### 9. Serialized Channel Processing
Instead of processing all d=64 channels in parallel (requiring 64 parallel spectral multipliers), serialize channels through a single multiply-accumulate unit. Trades throughput for area: **8× smaller spectral multiply block** at 8× longer latency per token. For LLM inference (not training), this is acceptable.

### 10. Operand Isolation for Leakage Reduction
During idle periods, force all butterfly inputs to zero using AND gates controlled by an enable signal. This eliminates **glitching power** in the combinational logic of the FFT datapath when the chip is not actively computing. A standard low-power technique that costs ~200 gates but reduces idle power by ~90%.

---

## Security Improvements (11–20)

### 11. Bitstream Encryption for Weight Loading
Encrypt the spectral weight bitstream loaded via the Wishbone bus using a lightweight stream cipher (Trivium or a 32-bit LFSR-based scheme). The decryption key is stored in on-chip fuses (SKY130 supports poly-fuse). Without the key, an attacker who probes the bus sees only ciphertext. **Prevents weight IP extraction** via bus snooping.

### 12. ~~Logic Locking with Spectral Mode Key~~ (REMOVED — open-source design)
Removed: This is an open-source, open-design chip. Anti-reverse-engineering measures are not needed.

### 13. Constant-Time Spectral Multiply
Make the spectral weight multiply take the same number of clock cycles regardless of how many modes are zeroed by soft-thresholding. Currently, zeroed modes could be skipped for efficiency, but this creates a timing side-channel that leaks which modes are active. Force all k modes to be processed (including zeroed ones) with a constant cycle count. **Prevents timing-based weight extraction**.

### 14. Power Flattening via Decoy MAC
Add a parallel decoy multiply-accumulate unit that operates simultaneously with the real spectral multiply. The decoy MAC computes on random data but draws the same power as the real MAC. An attacker measuring power traces cannot distinguish real computation from decoy. **Increases traces needed for CPA attack by ~100×** at the cost of ~2K extra gates.

### 15. ~~Scan Chain Lockout~~ (REMOVED — open-source design)
Removed: This is an open-source, open-design chip. Scan access is available for community testing and debugging.

### 16. ~~Layout-Level Netlist Obfuscation~~ (REMOVED — open-source design)
Removed: The full netlist is public. No need to obscure the layout.

### 17. Supply Chain Integrity Hash
Compute a SHA-256 hash of the compiled weight bitstream at compile time and embed it in the chip's read-only register. At boot, the chip recomputes the hash of loaded weights and compares to the stored hash. If mismatched, the chip sets a tamper flag and refuses to operate. **Detects weight tampering** in transit or during loading.

### 18. EM Shielding via Top Metal Layer
Use SKY130's top metal layer (Metal 6) to create a grounded shield over the spectral multiply core. This acts as a Faraday cage that **reduces electromagnetic emanations** by ~20–30dB, making EM side-channel attacks significantly harder. No functional impact — just a metal fill pattern connected to ground.

### 19. Reproducible Build Verification
Pin exact versions of all EDA tools (Yosys, OpenROAD, OpenLane, Magic) with SHA-256 hashes. Record the full build environment in a `tapeout/build_manifest.json` file. Before tapeout, verify that a clean rebuild produces a bit-identical GDSII. **Prevents supply chain injection of hardware trojans** via compromised EDA tools.

### 20. ~~Split Manufacturing Mask Set~~ (REMOVED — open-source design)
Removed: The complete design is public. No need to split masks across foundries for IP protection.