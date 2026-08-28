# V6 Improvements (26-45) — Clock Speed & Architecture

## Goal: Push clock from 90 MHz → 150+ MHz without errors

### Clock Speed Boosters (26-32)

26. **Retimed Pipeline Registers** — Insert optimal register positions in the critical path of the radix-4 butterfly using retiming synthesis constraints. Moves registers to balance stage delays, targeting 100→120 MHz.

27. **Clock Tree Synthesis Optimizer** — Dedicated clock distribution module with balanced H-tree buffer sizing, minimizing clock skew to <50ps across the die. Reduces hold-time violations that limit max frequency.

28. **Multi-Stage Spectral Multiply Pipeline** — Split the complex spectral weight multiply (currently 1 cycle) into 3 micro-stages: (a) real part multiply, (b) imaginary part multiply, (c) accumulate + threshold. Each stage is shorter, enabling 90→110 MHz.

29. **Register File Banking for Weight Access** — Bank the spectral weight registers into 4 independent banks that can be read simultaneously, eliminating the 1-cycle read-after-read stall in the spectral multiply loop. Effectively doubles the weight fetch bandwidth.

30. **Operand Isolation with Clock Gating** — Gate the clock to inactive pipeline stages (e.g., IFFT during FFT phase) using explicit clock-enable cells. Eliminates unwanted switching in dormant logic, reducing dynamic power and noise injection that can cause timing margin loss.

31. **Wider Datapath (Q10.6 → Q12.4 Hybrid)** — Use 18-bit datapath for FFT intermediate stages with Q12.4 format, giving 12 integer bits (no overflow in multi-layer inference) and 4 fractional bits (sufficient with BFP). The wider path reduces overflow stalls that waste cycles.

32. **Static Timing Analysis Fixup Module** — A synthesis directive module that inserts buffer chains on long interconnect paths identified by STA, specifically for the cross-die routes between FFT engine and spectral multiply. Eliminates the 5 longest paths limiting clock to 90 MHz.

### Architecture & Throughput (33-40)

33. **Speculative IFFT with Rollback** — Begin IFFT computation speculatively when 80% of spectral modes are ready, with a rollback mechanism if the remaining modes are non-zero (rare for soft-thresholded models). Saves ~10% latency in the common case.

34. **Multi-Port Weight SRAM** — Replace single-port weight register file with a dual-port SRAM macro that allows simultaneous read of current-mode weight and prefetch of next-mode weight, eliminating the 1-cycle bubble between modes in the spectral multiply loop.

35. **Fused FFT+IFFT with Shared Datapath** — Go beyond sharing the butterfly network: share the entire pipeline including twiddle ROM, adder tree, and output buffer. The chip time-multiplexes FFT and IFFT through the same datapath with a mode pin, saving ~8K gates.

36. **Channel Interleaving with Double Buffering** — Instead of processing all D channels of one token sequentially, interleave channels from two consecutive tokens. While token N's channel d is in the IFFT phase, token N+1's channel d is in the FFT phase. Doubles throughput for autoregressive generation.

37. **Configurable Pipeline Depth (2/4/8 stages)** — Allow the host to select FFT pipeline depth based on the clock frequency. At 50 MHz, 2 stages suffice (less latency). At 120 MHz, 8 stages are needed (more latency but higher throughput). The configuration is set once per inference session.

38. **Wishbone Burst Write for Input Data** — Extend the DMA burst controller to handle input data bursts, not just weights. The host writes 256 complex samples in a single 64-cycle burst instead of 256 individual write cycles, reducing input load time by 4×.

39. **Output Streaming with Backpressure** — Instead of waiting for all 256 IFFT outputs before the host can read, stream outputs as they are produced with a FIFO and backpressure signal. The host can start reading results after the first 32 outputs, reducing latency by ~100 cycles.

40. **Layer Scheduler with Weight Swapping** — A hardware scheduler that automatically swaps spectral weights between layers without host intervention. When layer N completes, the controller prefetches layer N+1 weights from the shadow register while the IFFT runs. Eliminates inter-layer host round-trips.

### Error Reduction & Reliability (41-45)

41. **Error Detection with Parity Check on FFT Stages** — Add 1-bit parity to each FFT butterfly output, with a parity checker at each stage boundary. Detects single-bit errors from timing violations or soft errors, preventing silent corruption that would degrade output quality.

42. **Guard Bands on Fixed-Point Saturation** — Instead of hard saturation at the max value, use guard bands that saturate at 95% of max, leaving 5% headroom for subsequent operations. Prevents cascading overflow artifacts that produce garbage tokens.

43. **Sticky Overflow Counter** — A counter that tracks the total number of overflow events across all FFT stages and spectral multiplies in one inference pass. The host reads this after each token and can adjust the BFP exponent range or reduce k if overflow is excessive.

44. **Redundant Compute with Checksum** — Compute a running checksum of all spectral multiply outputs. After the IFFT, compare against a precomputed expected checksum. If they differ, the host can retry the token — a lightweight error recovery mechanism.

45. **Thermal Throttle with Graceful Degradation** — Monitor on-chip temperature via a ring-oscillator thermal sensor. If temperature exceeds a threshold, automatically reduce k (active modes) and clock frequency. Prevents thermal-induced timing errors that would corrupt output.