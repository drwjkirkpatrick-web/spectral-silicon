# Spectral Silicon — 20 Performance Improvements (Security-Preserving)

Each improvement explicitly notes how security is maintained or enhanced. No improvement weakens any of the 10 security measures from IMPROVEMENTS.md.

---

## Datapath Arithmetic Optimizations (1–5)

### 1. Booth-Encoded Radix-4 Complex Multiplier
Replace the standard complex multiplier (4 real multiplies + 2 adds) with a Booth-encoded radix-4 multiplier. Booth encoding halves the number of partial products, and carry-save compression (Wallace tree) eliminates the carry-propagate adder from the critical path. The complex multiply still takes 1 clock cycle but the critical path is ~30% shorter, enabling **higher clock frequency (50→65 MHz)**.

**Security preserved:** No change to data flow. Constant-time MAC still processes all modes in fixed cycles. Power flattening decoy MAC uses the same multiplier type, so power signatures remain indistinguishable.

### 2. Block Floating-Point (BFP) for FFT Stages
Replace fixed Q8.8 arithmetic in the FFT with block floating-point: each block of 64 samples shares a common 4-bit exponent, with 12-bit mantissas. The exponent tracks the largest magnitude in each block and scales accordingly. This gives **8 extra bits of dynamic range** at the same datapath width, reducing FFT rounding error by ~20×. Critical for multi-layer spectral transformer inference where errors compound.

**Security preserved:** BFP scaling is data-dependent but does not change the constant-time guarantee — all blocks are processed in the same number of cycles. The exponent is computed via a parallel OR-tree, which is a fixed-latency combinational path. Power traces are slightly more variable, but the decoy MAC also uses BFP so power flattening still holds.

### 3. Carry-Save Accumulator for Spectral MAC
Replace the carry-propagate accumulator in the spectral multiply-accumulate with a carry-save accumulator. Partial products are kept in carry-save format throughout the k-mode accumulation, with a single carry-propagate add at the end. This eliminates the carry-propagate delay from every accumulation step, **reducing MAC latency by ~40%** and enabling higher throughput.

**Security preserved:** The final carry-propagate add is always performed regardless of whether modes are zeroed (constant-time). No timing information leaks.

### 4. Fused Multiply-Add (FMA) Butterfly
Merge the twiddle multiply and the butterfly add/subtract into a single FMA operation: `(a + W*b)` and `(a - W*b)` computed as fused multiply-adds. This eliminates the intermediate register between the multiplier and adder, **saving 1 pipeline stage and ~500 gates** per butterfly. The FMA also improves numerical accuracy by avoiding intermediate rounding.

**Security preserved:** FMA is a purely arithmetic optimization. It doesn't change what data is processed or when — only how the hardware computes the same result.

### 5. Truncated Booth Multiplier for Twiddle Multiplication
Twiddle factors are sin/cos values that are always in [-1, 1], so the integer part of the product is bounded. Use a truncated Booth multiplier that computes only the lower 16 bits of the 16×16 product (the upper bits are guaranteed zero for bounded inputs). This **saves ~30% of the multiplier area** for twiddle-specific multiplications.

**Security preserved:** Truncation only affects the computation path, not what data is accessible. Logic locking still gates the twiddle selection. No security-relevant information is leaked by the truncation.

---

## Memory & Data Movement Optimizations (6–10)

### 6. Ping-Pong Dual-Buffer Memory Banking
Replace the single in-place SRAM with two banks operating in ping-pong fashion: while the FFT reads from bank A and writes to bank B in stage n, the next stage reads from bank B and writes to bank A. This **eliminates the pipeline bubble between FFT stages**, enabling continuous data flow at 1 sample/clock sustained throughput.

**Security preserved:** Memory banking changes the physical organization but not the logical data. Integrity hash still covers both banks. No additional observable side-channels — the same data is accessed, just in alternating banks.

### 7. Weight Prefetch with Shadow Register File
Add a shadow register file that prefetches the next block of spectral weights while the current block is being processed. When the current block completes, a single-cycle swap switches to the shadow. This **hides weight-loading latency entirely** — the Wishbone bus loads weights in the background without stalling the MAC pipeline.

**Security preserved:** Weight crypto decrypts into the shadow register, maintaining encrypted-in-transit guarantee. The integrity hash is computed on the decrypted data in the shadow before the swap. No plaintext weights are exposed on the bus.

### 8. DMA Burst Mode for Weight Loading
Add a burst-mode DMA controller to the Wishbone interface: the host writes a base address and length, and the DMA fetches an entire weight block in a single burst transaction. This **reduces bus overhead from 1 cycle/word to 0.25 cycles/word** (4-word bursts with pipelined Wishbone B3). Weight load time drops from 128 cycles to 32 cycles for a 32-mode block.

**Security preserved:** DMA burst mode is a bus protocol optimization. Weight crypto still decrypts each word as it arrives. Logic locking is unaffected. The DMA address counter is not security-sensitive (weights are positionally indexed, not addressed by content).

### 9. Conflict-Free Memory Addressing for FFT
Use a conflict-free addressing scheme that guarantees the two data points needed by each butterfly are always in different memory banks. For radix-4 with 4 banks, use a modulo-4 address mapping: `bank = addr[1:0]`, `row = addr >> 2`. This **eliminates all memory bank conflicts**, enabling single-cycle butterfly execution without stall.

**Security preserved:** Addressing scheme is a fixed mathematical mapping with no data dependency. No timing side-channel. The same number of memory accesses occurs regardless of data values.

### 10. Bit-Reversal Permutation via Hardware Router
Replace the current bit-reversal permutation (done in software before FFT) with a hardware bit-reversal router that reorders data as it enters the FFT. The router is a simple crossbar that permutes address bits: bit[k] → bit[log2(N)-1-k]. This **eliminates the bit-reversal pre-processing step** from the host CPU, reducing end-to-end latency by ~256 clock cycles for a 256-point FFT.

**Security preserved:** Bit-reversal is a fixed permutation, not data-dependent. It doesn't change what data is processed, only the order of processing. No side-channel introduced.

---

## Algorithmic & Architectural Optimizations (11–15)

### 11. Real-Input FFT Exploitation (RFFT)
LLM token embeddings are real-valued, so the FFT input is real. A real-input FFT (RFFT) computes only the first N/2+1 output bins (Hermitian symmetry gives the rest for free). This **halves the FFT computation and memory** — only 129 modes computed instead of 256 for N=256. The spectral multiply operates on 129 modes, and the IFFT reconstructs the full real output.

**Security preserved:** RFFT is a mathematical optimization that produces the same result. The integrity hash covers the reduced mode set. Constant-time MAC still processes all k modes in fixed cycles. No information about the computation is leaked — the same modes are always computed.

### 12. Twiddle Factor Symmetry Exploitation
Exploit the property W_N^(k+N/4) = -j * W_N^k to generate 4 twiddle factors from 1 stored value. For a 256-point FFT, this **reduces twiddle storage from 64 to 16 entries** (4× compression). The three derived factors are obtained by swapping real/imaginary parts and sign flipping — trivial in hardware (zero gate cost, just wiring).

**Security preserved:** Twiddle generation is deterministic from the stored seed value. Logic locking still gates whether real or decoy twiddles are used. The derivation is fixed arithmetic, not data-dependent.

### 13. Zero-Skipping Spectral Multiply with Dummy Cycle Injection
When a spectral mode has been zeroed by soft-thresholding, the multiply result is zero. Instead of computing the full multiply, inject a dummy cycle that performs a decoy multiply on random data (reusing the power-flattening LFSR). The cycle count remains constant (security preserved), but the **real multiplier is idle for zeroed modes**, reducing switching activity by ~30% for typical 50% sparsity.

**Security preserved:** This is the key insight — we maintain constant timing AND reduce power. The dummy cycle draws the same power as a real multiply (decoy data has the same statistics). An attacker cannot distinguish "real multiply on zeroed mode" from "dummy multiply on random data" via timing or power. This actually *enhances* security by making the decoy MAC (Improvement 14) double as both a security measure and a power optimization.

### 14. Pipelined Spectral Multiply with Mode Interleaving
Instead of processing modes 0,1,2,...,31 sequentially, interleave modes across two pipeline stages: stage A processes even modes (0,2,4,...), stage B processes odd modes (1,3,5,...). Both stages operate simultaneously on different modes. This **doubles spectral multiply throughput** without doubling area — only one extra set of accumulator registers needed.

**Security preserved:** Mode interleaving is a scheduling optimization. All modes are still processed in constant time. The ordering changes but the total work doesn't. No timing or power side-channel — both stages draw equal power simultaneously.

### 15. Adaptive Mode Count with Secure Reporting
Allow the host to configure the number of active modes k (from 8 to 32) via a Wishbone register. Fewer active modes = faster inference. The chip always processes exactly k modes in a fixed, configurable cycle count. The host reads the actual cycle count from a status register, but **this value is determined by the configuration, not the data** — no data-dependent timing leak.

**Security preserved:** The mode count is a configuration parameter set by the host, not derived from input data. Constant-time guarantee holds for any fixed k. An attacker observing the bus sees the configured k, which is not secret. The timing varies with k but not with data content — this is acceptable because k is public configuration.

---

## Pipeline & Throughput Optimizations (16–20)

### 16. Deep Pipeline with 8-Stage FFT
Increase the FFT pipeline depth from 4 stages to 8 stages (2 sub-stages per radix-4 stage: twiddle multiply + butterfly add). Each sub-stage is shorter, enabling a **higher clock frequency (65→80 MHz)**. The latency increases by 4 clock cycles but throughput remains 1 sample/clock, and the higher frequency more than compensates.

**Security preserved:** Deeper pipeline doesn't change what data is processed. Operand isolation (Improvement 10) applies to each sub-stage independently. Clock gating can gate individual sub-stages for finer-grained power control.

### 17. Two-Channel Parallel Datapath
Add a second identical spectral processing channel that processes a second stream of tokens in parallel. The two channels share the weight register file (weights are loaded once, used by both) but have independent FFT/MAC/IFFT pipelines. This **doubles throughput** for batch-2 inference (common in LLM speculative decoding) at the cost of ~2× FFT area but only 1× weight storage.

**Security preserved:** Both channels use the same encrypted weights. Power flattening decoy MACs are independent per channel. EM shield covers both channels. No additional side-channel — both channels compute simultaneously with identical power profiles.

### 18. Early IFFT Start with Overlap
Begin the IFFT stage as soon as the last 64 of 256 FFT modes are written, overlapping the IFFT input phase with the final FFT modes. The IFFT only needs the first k=32 modes (the rest are zero), so it can start after mode 32 is ready. This **reduces end-to-end latency by ~30%** (from 512 to ~360 clock cycles).

**Security preserved:** The overlap changes scheduling but not the set of operations performed. All modes are still processed. Constant-time guarantee holds — the total cycle count for a fixed k is deterministic. An external observer sees the same "busy" signal duration for any input of the same configured k.

### 19. Configurable FFT Size (128/256/512)
Make the FFT size configurable via a Wishbone register. For shorter token sequences (seq_len ≤ 128), use a 128-point FFT (fewer cycles). For longer sequences (seq_len > 256), use 512-point. The spectral weights are resolution-invariant (FNO property), so the same weights work at any FFT size. This **provides 2× speedup for short sequences** and 2× longer context for long ones.

**Security preserved:** FFT size is a public configuration parameter, not data-dependent. Integrity hash covers weights regardless of FFT size. Constant-time guarantee holds for any fixed N. No side-channel — the attacker can observe N from the bus but it's not secret.

### 20. Frequency Scaling with Secure Voltage Tracking
Implement dynamic voltage-frequency scaling (DVFS): when the chip is idle or processing at reduced mode count k, lower the clock frequency and core voltage (1.8V → 1.2V) to save power. A secure voltage tracker verifies that the voltage transition completed before enabling computation at the new frequency. This **reduces idle power by ~60%** and active power at low-k by ~40%.

**Security preserved:** DVFS is a known side-channel concern, but we mitigate it: (1) voltage transitions only occur between inference batches, never mid-computation, so intra-batch timing is constant. (2) The transition is triggered by the host's batch completion, not by data content. (3) Power flattening decoy MAC operates at the same voltage as the real MAC. (4) A secure tracker prevents operation at unsafe voltage/frequency combinations that could introduce faults (fault attack resistance).