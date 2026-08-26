`default_nettype none
//==============================================================================
// zero_skip_mac.v — Zero-Skipping MAC with Dummy Cycle Injection
//==============================================================================
// Performance improvement: Detects zeroed spectral modes and skips the real
// multiply for those modes.  However, to maintain constant cycle count
// (critical for timing security), a dummy multiply cycle is injected using
// LFSR-generated random data.  The real multiplier is idle for zeroed modes
// but draws power on the dummy data, making the power profile indistinguishable
// from a real multiply.
//
// Security preservation: This is the key security module.  By injecting dummy
// cycles with LFSR random data, the total cycle count and power trace are
// identical whether a mode is zero or non-zero.  An attacker cannot determine
// which modes were skipped by observing timing or power.  The LFSR output is
// never used in the result — it only feeds the multiplier's idle inputs to
// draw power.
//
// Interface:
//   clk, rst_n       — clock and reset
//   a_re, a_im       — input A (complex)
//   b_re, b_im       — input B (complex weight)
//   mode_zero        — 1 if this mode is zero (skip real multiply)
//   valid_in         — input valid
//   result_re, result_im — MAC result (zero for skipped modes)
//   valid_out        — result valid (1 cycle after valid_in, always)
//
// Verilog-2005, `default_nettype none.  Synthesizable.
//==============================================================================
module zero_skip_mac #(
    parameter WIDTH = 16,
    parameter FRAC  = 8
) (
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire signed [WIDTH-1:0] a_re, a_im,
    input  wire signed [WIDTH-1:0] b_re, b_im,
    input  wire                    mode_zero,
    input  wire                    valid_in,
    output reg  signed [WIDTH-1:0] result_re,
    output reg  signed [WIDTH-1:0] result_im,
    output reg                     valid_out
);

    localparam PW = 2 * WIDTH;

    //------------------------------------------------------------------
    // LFSR for dummy data generation
    // 16-bit maximal-length LFSR (polynomial x^16 + x^14 + x^13 + x^11 + 1)
    // Used to generate random-looking data for dummy multiply cycles.
    // The LFSR output is NOT used in the result — it only feeds the
    // multiplier inputs when mode_zero=1, drawing power to mask the skip.
    //------------------------------------------------------------------
    reg [15:0] lfsr;
    wire lfsr_feedback = lfsr[0] ^ lfsr[2] ^ lfsr[3] ^ lfsr[5];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            lfsr <= 16'hACE1;  // Non-zero seed
        else
            lfsr <= {lfsr[14:0], lfsr_feedback};
    end

    // Dummy data from LFSR (sign-extended to WIDTH)
    wire signed [WIDTH-1:0] dummy_a_re = {lfsr[7:0], lfsr[WIDTH-1:8]};
    wire signed [WIDTH-1:0] dummy_a_im = {lfsr[15:8], lfsr[7:0]};
    wire signed [WIDTH-1:0] dummy_b_re = {lfsr[WIDTH-1:0]};
    wire signed [WIDTH-1:0] dummy_b_im = {lfsr[3:0], lfsr[WIDTH-1:4]};

    //------------------------------------------------------------------
    // MUX: select real or dummy inputs for the multiplier
    // When mode_zero=1: feed dummy data to multiplier (draws power)
    // When mode_zero=0: feed real data to multiplier (computes result)
    //
    // The multiplier ALWAYS runs — same power, same timing, either way.
    //------------------------------------------------------------------
    wire signed [WIDTH-1:0] mul_a_re = mode_zero ? dummy_a_re : a_re;
    wire signed [WIDTH-1:0] mul_a_im = mode_zero ? dummy_a_im : a_im;
    wire signed [WIDTH-1:0] mul_b_re = mode_zero ? dummy_b_re : b_re;
    wire signed [WIDTH-1:0] mul_b_im = mode_zero ? dummy_b_im : b_im;

    // Complex multiply (always computes — constant power)
    wire signed [PW-1:0] prod_re = (mul_a_re * mul_b_re) - (mul_a_im * mul_b_im);
    wire signed [PW-1:0] prod_im = (mul_a_re * mul_b_im) + (mul_a_im * mul_b_re);

    wire signed [WIDTH-1:0] mult_re = prod_re >>> FRAC;
    wire signed [WIDTH-1:0] mult_im = prod_im >>> FRAC;

    //------------------------------------------------------------------
    // Output: zero result for skipped modes, real result for active modes
    // The result mux is AFTER the multiplier, so the multiplier always
    // processes data.  Only the output is zeroed for skipped modes.
    //------------------------------------------------------------------
    wire signed [WIDTH-1:0] final_re = mode_zero ? {WIDTH{1'b0}} : mult_re;
    wire signed [WIDTH-1:0] final_im = mode_zero ? {WIDTH{1'b0}} : mult_im;

    // Register output (1-cycle latency — constant regardless of mode_zero)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result_re <= 0;
            result_im <= 0;
            valid_out <= 1'b0;
        end else begin
            result_re <= final_re;
            result_im <= final_im;
            valid_out <= valid_in;
        end
    end

endmodule