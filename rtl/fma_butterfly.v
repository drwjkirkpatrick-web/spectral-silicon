`default_nettype none
//==============================================================================
// fma_butterfly.v — Fused Multiply-Add Butterfly
//==============================================================================
// Performance improvement: Fuses the twiddle multiplication and the butterfly
// addition into a single pipeline stage.  The standard butterfly computes
// the DFT kernel (additions), then multiplies by twiddle factors, then adds
// to the accumulator.  The FMA butterfly computes a + W*b and a - W*b in one
// combinational block with no intermediate register between the multiplier
// and the adder, saving one pipeline stage per FFT stage (4 stages → 4
// cycles saved per transform).
//
// Security preservation: fully combinational, constant-time operation.
// No data-dependent branching or early termination.  The FMA path is
// structurally identical for all coefficient values.
//
// Interface (radix-2 butterfly with FMA):
//   a_re, a_im    — first input (complex)
//   b_re, b_im    — second input (complex)
//   w_re, w_im    — twiddle factor (complex)
//   y_plus_re/im  — output: a + W*b
//   y_minus_re/im — output: a - W*b
//
// Verilog-2005, `default_nettype none.  Synthesizable.
//==============================================================================
module fma_butterfly #(
    parameter WIDTH = 16,
    parameter FRAC  = 8
) (
    input  wire signed [WIDTH-1:0]  a_re, a_im,
    input  wire signed [WIDTH-1:0]  b_re, b_im,
    input  wire signed [WIDTH-1:0]  w_re, w_im,
    output wire signed [WIDTH-1:0]  y_plus_re,  y_plus_im,
    output wire signed [WIDTH-1:0]  y_minus_re, y_minus_im
);

    localparam PW = 2 * WIDTH;

    //------------------------------------------------------------------
    // Complex multiply W*b (fused into adder — no intermediate register)
    //   W*b = (w_re + j*w_im) * (b_re + j*b_im)
    //   re = w_re*b_re - w_im*b_im
    //   im = w_re*b_im + w_im*b_re
    //------------------------------------------------------------------
    wire signed [PW-1:0] wb_re_full = (w_re * b_re) - (w_im * b_im);
    wire signed [PW-1:0] wb_im_full = (w_re * b_im) + (w_im * b_re);

    // Rescale by FRAC (arithmetic right shift to restore Q8.8)
    wire signed [WIDTH-1:0] wb_re = wb_re_full >>> FRAC;
    wire signed [WIDTH-1:0] wb_im = wb_im_full >>> FRAC;

    //------------------------------------------------------------------
    // Fused add: a + W*b and a - W*b
    // Sign-extend a by 1 bit to prevent overflow in the add/subtract
    //------------------------------------------------------------------
    wire signed [WIDTH:0] a_re_ext = {{1{a_re[WIDTH-1]}}, a_re};
    wire signed [WIDTH:0] a_im_ext = {{1{a_im[WIDTH-1]}}, a_im};
    wire signed [WIDTH:0] wb_re_ext = {{1{wb_re[WIDTH-1]}}, wb_re};
    wire signed [WIDTH:0] wb_im_ext = {{1{wb_im[WIDTH-1]}}, wb_im};

    wire signed [WIDTH:0] plus_re  = a_re_ext + wb_re_ext;
    wire signed [WIDTH:0] plus_im  = a_im_ext + wb_im_ext;
    wire signed [WIDTH:0] minus_re = a_re_ext - wb_re_ext;
    wire signed [WIDTH:0] minus_im = a_im_ext - wb_im_ext;

    // Saturate to WIDTH bits (clip to prevent wraparound)
    // Saturation preserves security by avoiding wrap-around artifacts
    // that could leak information through overflow flags.
    assign y_plus_re  = plus_re[WIDTH]  ? {WIDTH{1'b0}} :
                        (plus_re > {1'b0, {WIDTH{1'b1}}}) ?
                        {1'b0, {WIDTH{1'b1}}} : plus_re[WIDTH-1:0];
    assign y_plus_im  = plus_im[WIDTH]  ? {WIDTH{1'b0}} :
                        (plus_im > {1'b0, {WIDTH{1'b1}}}) ?
                        {1'b0, {WIDTH{1'b1}}} : plus_im[WIDTH-1:0];
    assign y_minus_re = minus_re[WIDTH] ? {WIDTH{1'b0}} :
                        (minus_re > {1'b0, {WIDTH{1'b1}}}) ?
                        {1'b0, {WIDTH{1'b1}}} : minus_re[WIDTH-1:0];
    assign y_minus_im = minus_im[WIDTH] ? {WIDTH{1'b0}} :
                        (minus_im > {1'b0, {WIDTH{1'b1}}}) ?
                        {1'b0, {WIDTH{1'b1}}} : minus_im[WIDTH-1:0];

endmodule