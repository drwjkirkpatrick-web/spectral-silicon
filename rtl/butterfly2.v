`default_nettype none
//==============================================================================
// butterfly2.v — Radix-2 FFT Butterfly Core
//==============================================================================
// Computes the radix-2 DIT FFT butterfly:
//   X = a + W*b     (sum branch)
//   Y = a - W*b     (difference branch)
// where a, b, W are complex numbers in Q8.8 fixed-point (parameterized width).
//
// All inputs/outputs are signed two's complement.
// W = w_re + j*w_im.  b is the "twiddle-multiplied" input.
//
// Complex multiply:  W*b = (w_re + j*w_im)*(b_re + j*b_im)
//   re_part = w_re*b_re - w_im*b_im
//   im_part = w_re*b_im + w_im*b_re
//
// Q8.8: one multiply gives 2*WIDTH bits; we right-shift by 8 (FRAC) to return
// to the original Q format.
//
// Prompt 11 specification.
//==============================================================================
module butterfly2 #(
    parameter WIDTH = 16,           // Total data width (signed)
    parameter FRAC  = 8             // Fractional bits (Q8.8 when WIDTH=16)
) (
    input  wire signed [WIDTH-1:0]  a_re,   // Top input, real
    input  wire signed [WIDTH-1:0]  a_im,   // Top input, imag
    input  wire signed [WIDTH-1:0]  b_re,   // Bottom input, real
    input  wire signed [WIDTH-1:0]  b_im,   // Bottom input, imag
    input  wire signed [WIDTH-1:0]  w_re,   // Twiddle factor, real
    input  wire signed [WIDTH-1:0]  w_im,   // Twiddle factor, imag
    output wire signed [WIDTH-1:0]  x_re,   // Output X = a + W*b, real
    output wire signed [WIDTH-1:0]  x_im,   // Output X = a + W*b, imag
    output wire signed [WIDTH-1:0]  y_re,   // Output Y = a - W*b, real
    output wire signed [WIDTH-1:0]  y_im    // Output Y = a - W*b, imag
);

    // Internal product width: 2*WIDTH to hold full-precision multiply
    localparam PW = 2 * WIDTH;

    // Complex multiply: W * b
    // re_part = w_re * b_re - w_im * b_im  (scaled by 2^FRAC after multiply)
    // im_part = w_re * b_im + w_im * b_re
    wire signed [PW-1:0] prod_re1 = w_re * b_re;
    wire signed [PW-1:0] prod_im1 = w_im * b_im;
    wire signed [PW-1:0] prod_re2 = w_re * b_im;
    wire signed [PW-1:0] prod_im2 = w_im * b_re;

    // Sum/difference of products, then right-shift by FRAC to rescale
    // The products are in Q(WIDTH+FRAC).(WIDTH-FRAC)... actually Q(2*FRAC).FRAC
    // After multiplying two Q8.8 numbers: result is Q16.16, shift right by FRAC=8
    // to get back to Q(WIDTH).FRAC.
    wire signed [PW-1:0] wb_re_full = prod_re1 - prod_im1;  // W*b real part
    wire signed [PW-1:0] wb_im_full = prod_re2 + prod_im2;  // W*b imag part

    // Rescale to original width: arithmetic right shift by FRAC, then take
    // the lower WIDTH bits.
    wire signed [WIDTH-1:0] wb_re = wb_re_full >>> FRAC;
    wire signed [WIDTH-1:0] wb_im = wb_im_full >>> FRAC;

    // Butterfly outputs
    // X = a + W*b
    assign x_re = a_re + wb_re;
    assign x_im = a_im + wb_im;

    // Y = a - W*b
    assign y_re = a_re - wb_re;
    assign y_im = a_im - wb_im;

endmodule