`default_nettype none
//==============================================================================
// parity_error_detect.v — Parity Checker for FFT Stage Outputs
//==============================================================================
// Computes even parity over the 32-bit complex data (16-bit real concatenated
// with 16-bit imaginary) at each FFT stage boundary.  The expected parity is
// supplied by the host on parity_in.  If the computed parity does not match the
// expected value, parity_error is asserted and held until the next valid stage
// boundary, allowing the host to read the flag after the inference pass.
//
// Even parity is used so that a single-bit flip in the 32-bit complex word
// (including a stage boundary corruption) changes the parity and is detected.
//
// Interface:
//   clk          — clock
//   rst          — active-high synchronous reset
//   data_re[15:0]— real part of complex FFT sample (Q8.8)
//   data_im[15:0]— imaginary part of complex FFT sample (Q8.8)
//   parity_in    — expected even-parity bit from host
//   stage_num[2:0]— current FFT stage index (0..STAGES-1)
//   parity_error — sticky error flag, asserted on mismatch
//
// Parameter STAGES — number of FFT pipeline stages (default 4 for 256-point).
//
// Verilog-2005, `default_nettype none.  Synthesizable.
//==============================================================================
module parity_error_detect #(
    parameter STAGES = 4
) (
    input  wire        clk,
    input  wire        rst,
    input  wire [15:0] data_re,
    input  wire [15:0] data_im,
    input  wire        parity_in,
    output reg         parity_error,
    input  wire [2:0]  stage_num
);

    //----------------------------------------------------------------------
    // Concatenate real and imaginary into a 32-bit word and XOR-reduce to get
    // even parity (parity bit = 1 if an odd number of bits are set, so that the
    // total number of 1s including the parity bit is even).
    //----------------------------------------------------------------------
    wire [31:0] complex_word = {data_re, data_im};
    wire        computed_parity;

    assign computed_parity = ^complex_word;  // XOR reduction = even parity

    //----------------------------------------------------------------------
    // Synchronous comparison: latch the error if computed parity differs from
    // the expected parity_in.  The error is sticky — once asserted it stays
    // high until reset, so the host can poll it after the full inference pass.
    //----------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst) begin
            parity_error <= 1'b0;
        end else if (stage_num < STAGES) begin
            // Only check at valid stage boundaries
            if (computed_parity != parity_in) begin
                parity_error <= 1'b1;
            end
        end
    end

endmodule

`default_nettype wire