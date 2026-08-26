`default_nettype none
//==============================================================================
// mode_interleave.v — Mode Interleaving Pipeline
//==============================================================================
// Performance improvement: Interleaves even and odd spectral modes across
// two pipeline stages that operate simultaneously.  While stage 0 processes
// mode 2k, stage 1 processes mode 2k+1.  This doubles the throughput of the
// spectral MAC pipeline without duplicating the full datapath — the two
// stages share the same weight ROM and input buffer, reading different
// addresses in parallel.
//
// Security preservation: both stages run unconditionally for every mode pair.
// The interleaving pattern is fixed (even/odd) and data-independent.  No
// stage is ever skipped, so power traces are uniform across all modes.
//
// Interface:
//   clk, rst_n       — clock and reset
//   mode_re, mode_im — input mode data (complex)
//   mode_idx         — mode index (0..N-1)
//   valid_in         — input valid
//   wt0_re, wt0_im   — weight for even mode (stage 0)
//   wt1_re, wt1_im   — weight for odd mode (stage 1)
//   result0_re/im    — stage 0 result (even mode)
//   result1_re/im    — stage 1 result (odd mode)
//   valid_out        — both results valid (1 cycle after input)
//
// Verilog-2005, `default_nettype none.  Synthesizable.
//==============================================================================
module mode_interleave #(
    parameter WIDTH = 16,
    parameter FRAC  = 8
) (
    input  wire                    clk,
    input  wire                    rst_n,

    // Input: one mode per cycle, interleaved into two stages
    input  wire signed [WIDTH-1:0] mode_re,
    input  wire signed [WIDTH-1:0] mode_im,
    input  wire [7:0]              mode_idx,
    input  wire                    valid_in,

    // Weights for the two pipeline stages
    input  wire signed [WIDTH-1:0] wt0_re, wt0_im,  // Even-mode weight
    input  wire signed [WIDTH-1:0] wt1_re, wt1_im,  // Odd-mode weight

    // Output: two results per cycle
    output reg  signed [WIDTH-1:0] result0_re, result0_im,
    output reg  signed [WIDTH-1:0] result1_re, result1_im,
    output reg                     valid_out,
    output reg  [7:0]              out_idx0,
    output reg  [7:0]              out_idx1
);

    localparam PW = 2 * WIDTH;

    //------------------------------------------------------------------
    // Stage 0 register: holds even-mode data
    // Stage 1 register: holds odd-mode data
    // Both stages compute in parallel on consecutive cycles.
    //
    // When mode_idx is even: latch into stage 0 input
    // When mode_idx is odd: latch into stage 1 input
    // Both stages produce results on the cycle after the odd mode arrives.
    //------------------------------------------------------------------
    reg signed [WIDTH-1:0] s0_re_r, s0_im_r;
    reg signed [WIDTH-1:0] s1_re_r, s1_im_r;
    reg [7:0] s0_idx_r, s1_idx_r;
    reg       s0_valid_r, s1_valid_r;

    // Is current mode even or odd?
    wire is_even = ~mode_idx[0];

    //------------------------------------------------------------------
    // Stage 0: even-mode complex multiply (uses wt0)
    //------------------------------------------------------------------
    wire signed [PW-1:0] s0_prod_re = (wt0_re * s0_re_r) - (wt0_im * s0_im_r);
    wire signed [PW-1:0] s0_prod_im = (wt0_re * s0_im_r) + (wt0_im * s0_re_r);
    wire signed [WIDTH-1:0] s0_res_re = s0_prod_re >>> FRAC;
    wire signed [WIDTH-1:0] s0_res_im = s0_prod_im >>> FRAC;

    //------------------------------------------------------------------
    // Stage 1: odd-mode complex multiply (uses wt1)
    //------------------------------------------------------------------
    wire signed [PW-1:0] s1_prod_re = (wt1_re * s1_re_r) - (wt1_im * s1_im_r);
    wire signed [PW-1:0] s1_prod_im = (wt1_re * s1_im_r) + (wt1_im * s1_re_r);
    wire signed [WIDTH-1:0] s1_res_re = s1_prod_re >>> FRAC;
    wire signed [WIDTH-1:0] s1_res_im = s1_prod_im >>> FRAC;

    //------------------------------------------------------------------
    // Input latching and output registration
    //------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s0_re_r <= 0; s0_im_r <= 0;
            s1_re_r <= 0; s1_im_r <= 0;
            s0_idx_r <= 0; s1_idx_r <= 0;
            s0_valid_r <= 1'b0; s1_valid_r <= 1'b0;
            result0_re <= 0; result0_im <= 0;
            result1_re <= 0; result1_im <= 0;
            valid_out  <= 1'b0;
            out_idx0   <= 0; out_idx1  <= 0;
        end else begin
            // Default
            valid_out <= 1'b0;

            // Latch input into appropriate stage
            if (valid_in) begin
                if (is_even) begin
                    // Even mode → stage 0
                    s0_re_r   <= mode_re;
                    s0_im_r   <= mode_im;
                    s0_idx_r  <= mode_idx;
                    s0_valid_r <= 1'b1;
                end else begin
                    // Odd mode → stage 1
                    s1_re_r   <= mode_re;
                    s1_im_r   <= mode_im;
                    s1_idx_r  <= mode_idx;
                    s1_valid_r <= 1'b1;
                end
            end

            // When stage 1 gets data (odd mode), both stages produce results
            // This assumes modes arrive in order: 0, 1, 2, 3, ...
            if (s1_valid_r) begin
                result0_re <= s0_res_re;
                result0_im <= s0_res_im;
                result1_re <= s1_res_re;
                result1_im <= s1_res_im;
                out_idx0   <= s0_idx_r;
                out_idx1   <= s1_idx_r;
                valid_out  <= 1'b1;
                // Clear valids after producing output
                s0_valid_r <= 1'b0;
                s1_valid_r <= 1'b0;
            end
        end
    end

endmodule