`default_nettype none
//==============================================================================
// triple_twiddle_rom.v — Triple-Port Twiddle Factor ROM for Radix-4 Butterfly
//==============================================================================
// Provides three independent read ports into the twiddle-factor lookup table
// so a radix-4 butterfly can fetch W1, W2, and W3 simultaneously instead of
// serially.  Each port is a separate instance of twiddle_rom with its own
// registered (1-cycle latency) output.
//
// -----------------------------------------------------------------------------
// Why this exists — the serial twiddle bottleneck
// -----------------------------------------------------------------------------
// A radix-4 butterfly consumes three twiddle factors per group:
//     W1 = W_N^{k},   W2 = W_N^{2k},   W3 = W_N^{3k}
// The original FFT engine (fft_256.v / fft_ifft_256.v) has a single twiddle_rom
// with one read port and 1-cycle latency.  It must therefore read the three
// factors one at a time:
//     cycle 0  issue W1 addr            cycle 1  capture W1
//     cycle 2  issue W2 addr            cycle 3  capture W2
//     cycle 4  issue W3 addr            cycle 5  capture W3
//   → 6 cycles per butterfly just for twiddle reads.
//
// With 64 groups × 4 stages = 256 butterflies per FFT, that is
//     256 × 6 = 1536 cycles of serial twiddle overhead.
// Each butterfly also needs ~2 cycles of arithmetic (MAC + accum), so the
// total per-FFT cost is roughly
//     256 × (6 twiddle + 2 calc) ≈ 2048 cycles.
//
// triple_twiddle_rom issues all three addresses in the same cycle and returns
// all three (cos, sin) pairs one cycle later:
//     cycle 0  issue addr1, addr2, addr3 simultaneously
//     cycle 1  capture W1, W2, W3 together
//   → 2 cycles per butterfly for twiddle reads (1 addr issue + 1 data return).
//
// New cost per butterfly: 2 (twiddle) + 2 (calc) = 4 cycles.
// New total per FFT: 256 × 4 ≈ 1024 cycles.
//
//   Speedup on twiddle reads:  6 → 2  cycles  = 3×
//   Overall FFT speedup:       ~2048 → ~1024 cycles = ~2×
//
// Area cost: 3× the ROM (two extra copies of the 256×16 cos and sin tables).
// In an ASIC flow the three ports can also be a single 3-port RAM macro
// (common in standard-cell/Foundry memory compilers), giving the same timing
// win at ~1× area.  This RTL instantiation form is portable across any
// simulator or synthesizer that already supports twiddle_rom.
//
// -----------------------------------------------------------------------------
// Interface
// -----------------------------------------------------------------------------
//   clk          — system clock
//   rst_n        — active-low reset (synchronous to clk; held low while the
//                  $readmemh initial blocks finish)
//   addr1[7:0]   — index k for W1  (issued on cycle T,  data valid cycle T+1)
//   addr2[7:0]   — index k for W2  (issued on cycle T,  data valid cycle T+1)
//   addr3[7:0]   — index k for W3  (issued on cycle T,  data valid cycle T+1)
//   cos1_out[15:0], sin1_out[15:0] — W1 cos/sin, Q8.8 signed
//   cos2_out[15:0], sin2_out[15:0] — W2 cos/sin, Q8.8 signed
//   cos3_out[15:0], sin3_out[15:0] — W3 cos/sin, Q8.8 signed
//
// All three ports have identical 1-cycle latency and are completely
// independent: each may address any of the N entries without interfering with
// the others.
//
// Parameters:
//   N         = 256     FFT size (number of twiddle entries)
//   WIDTH     = 16      Data width (Q8.8 fixed-point)
//   ADDR_BITS = 8       log2(N) address bits
//   COS_FILE  = "twiddle_data/twiddle_cos_256.hex"
//   SIN_FILE  = "twiddle_data/twiddle_sin_256.hex"
//
// The hex-file paths are identical to twiddle_rom.v so both modules read the
// same precomputed tables.  Paths are relative to the simulator/synthesis
// working directory (set via PLUSARGS or include path as needed).
//==============================================================================
module triple_twiddle_rom #(
    parameter N         = 256,
    parameter WIDTH     = 16,
    parameter ADDR_BITS = 8,
    parameter COS_FILE  = "twiddle_data/twiddle_cos_256.hex",
    parameter SIN_FILE  = "twiddle_data/twiddle_sin_256.hex"
) (
    input  wire                    clk,
    input  wire                    rst_n,

    // Port 1 — W1
    input  wire [ADDR_BITS-1:0]    addr1,
    output wire signed [WIDTH-1:0]  cos1_out,
    output wire signed [WIDTH-1:0]  sin1_out,

    // Port 2 — W2
    input  wire [ADDR_BITS-1:0]    addr2,
    output wire signed [WIDTH-1:0]  cos2_out,
    output wire signed [WIDTH-1:0]  sin2_out,

    // Port 3 — W3
    input  wire [ADDR_BITS-1:0]    addr3,
    output wire signed [WIDTH-1:0]  cos3_out,
    output wire signed [WIDTH-1:0]  sin3_out
);

    //------------------------------------------------------------------
    // Three independent twiddle_rom instances — one per read port.
    // Each is a registered ROM with 1-cycle latency; the three are fully
    // parallel so all six outputs (3 cos + 3 sin) are valid on the same
    // clock edge after the addresses are presented.
    //------------------------------------------------------------------
    twiddle_rom #(
        .N         (N),
        .WIDTH     (WIDTH),
        .ADDR_BITS (ADDR_BITS),
        .COS_FILE  (COS_FILE),
        .SIN_FILE  (SIN_FILE)
    ) u_rom1 (
        .clk      (clk),
        .addr     (addr1),
        .cos_out  (cos1_out),
        .sin_out  (sin1_out)
    );

    twiddle_rom #(
        .N         (N),
        .WIDTH     (WIDTH),
        .ADDR_BITS (ADDR_BITS),
        .COS_FILE  (COS_FILE),
        .SIN_FILE  (SIN_FILE)
    ) u_rom2 (
        .clk      (clk),
        .addr     (addr2),
        .cos_out  (cos2_out),
        .sin_out  (sin2_out)
    );

    twiddle_rom #(
        .N         (N),
        .WIDTH     (WIDTH),
        .ADDR_BITS (ADDR_BITS),
        .COS_FILE  (COS_FILE),
        .SIN_FILE  (SIN_FILE)
    ) u_rom3 (
        .clk      (clk),
        .addr     (addr3),
        .cos_out  (cos3_out),
        .sin_out  (sin3_out)
    );

endmodule

`default_nettype wire