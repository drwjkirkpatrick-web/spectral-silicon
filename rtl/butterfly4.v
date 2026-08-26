`default_nettype none
//==============================================================================
// butterfly4.v — Radix-4 FFT Butterfly Core
//==============================================================================
// Computes the radix-4 DIT FFT butterfly: 4 complex inputs → 4 complex outputs
// with 3 twiddle multiplications (W1, W2, W3).
//
// The radix-4 DFT kernel for inputs x0..x3:
//   X0 = x0 + x1 + x2 + x3
//   X1 = x0 - j*x1 - x2 + j*x3    → then * W1
//   X2 = x0 - x1 + x2 - x3       → then * W2
//   X3 = x0 + j*x1 - x2 - j*x3    → then * W3
//
// where j = sqrt(-1). Multiplication by j swaps real/imag and negates:
//   j*(re + j*im) = -im + j*re
//
// All arithmetic in Q8.8 fixed-point (parameterized). Complex multiplies use
// the full 2*WIDTH product width and right-shift by FRAC to rescale.
//
// Prompt 12 specification.
//==============================================================================
module butterfly4 #(
    parameter WIDTH = 16,
    parameter FRAC  = 8
) (
    input  wire signed [WIDTH-1:0]  x0_re, x0_im,  // Input 0
    input  wire signed [WIDTH-1:0]  x1_re, x1_im,  // Input 1
    input  wire signed [WIDTH-1:0]  x2_re, x2_im,  // Input 2
    input  wire signed [WIDTH-1:0]  x3_re, x3_im,  // Input 3
    input  wire signed [WIDTH-1:0]  w1_re, w1_im,  // Twiddle for X1
    input  wire signed [WIDTH-1:0]  w2_re, w2_im,  // Twiddle for X2
    input  wire signed [WIDTH-1:0]  w3_re, w3_im,  // Twiddle for X3
    output wire signed [WIDTH-1:0]  y0_re, y0_im,  // Output 0
    output wire signed [WIDTH-1:0]  y1_re, y1_im,  // Output 1
    output wire signed [WIDTH-1:0]  y2_re, y2_im,  // Output 2
    output wire signed [WIDTH-1:0]  y3_re, y3_im   // Output 3
);

    localparam PW = 2 * WIDTH;

    //--- Step 1: Radix-4 DFT kernel (no twiddles yet) ---
    // X0 = x0 + x1 + x2 + x3
    wire signed [WIDTH+1:0] s0_re = x0_re + x1_re + x2_re + x3_re;
    wire signed [WIDTH+1:0] s0_im = x0_im + x1_im + x2_im + x3_im;

    // X1 = x0 - j*x1 - x2 + j*x3
    // -j*(x1) → (x1_im, -x1_re),  j*(x3) → (-x3_im, x3_re)
    wire signed [WIDTH+1:0] s1_re_pre = x0_re + x1_im - x2_re - x3_im;
    wire signed [WIDTH+1:0] s1_im_pre = x0_im - x1_re - x2_im + x3_re;

    // X2 = x0 - x1 + x2 - x3
    wire signed [WIDTH+1:0] s2_re_pre = x0_re - x1_re + x2_re - x3_re;
    wire signed [WIDTH+1:0] s2_im_pre = x0_im - x1_im + x2_im - x3_im;

    // X3 = x0 + j*x1 - x2 - j*x3
    // j*(x1) → (-x1_im, x1_re),  -j*(x3) → (x3_im, -x3_re)
    wire signed [WIDTH+1:0] s3_re_pre = x0_re - x1_im - x2_re + x3_im;
    wire signed [WIDTH+1:0] s3_im_pre = x0_im + x1_re - x2_im - x3_re;

    // Trim intermediate results back to WIDTH for twiddle multiply
    // (they have WIDTH+2 bits, but we saturate/take the lower WIDTH)
    wire signed [WIDTH-1:0] s1_re = s1_re_pre[WIDTH-1:0];
    wire signed [WIDTH-1:0] s1_im = s1_im_pre[WIDTH-1:0];
    wire signed [WIDTH-1:0] s2_re = s2_re_pre[WIDTH-1:0];
    wire signed [WIDTH-1:0] s2_im = s2_im_pre[WIDTH-1:0];
    wire signed [WIDTH-1:0] s3_re = s3_re_pre[WIDTH-1:0];
    wire signed [WIDTH-1:0] s3_im = s3_im_pre[WIDTH-1:0];

    //--- Step 2: Twiddle multiplications ---
    // W * s = (w_re + j*w_im)*(s_re + j*s_im)
    //   re = w_re*s_re - w_im*s_im
    //   im = w_re*s_im + w_im*s_re

    // Twiddle 1
    wire signed [PW-1:0] t1_re_full = (w1_re * s1_re) - (w1_im * s1_im);
    wire signed [PW-1:0] t1_im_full = (w1_re * s1_im) + (w1_im * s1_re);

    // Twiddle 2
    wire signed [PW-1:0] t2_re_full = (w2_re * s2_re) - (w2_im * s2_im);
    wire signed [PW-1:0] t2_im_full = (w2_re * s2_im) + (w2_im * s2_re);

    // Twiddle 3
    wire signed [PW-1:0] t3_re_full = (w3_re * s3_re) - (w3_im * s3_im);
    wire signed [PW-1:0] t3_im_full = (w3_re * s3_im) + (w3_im * s3_re);

    // Rescale: arithmetic right shift by FRAC, take lower WIDTH bits
    assign y1_re = t1_re_full >>> FRAC;
    assign y1_im = t1_im_full >>> FRAC;
    assign y2_re = t2_re_full >>> FRAC;
    assign y2_im = t2_im_full >>> FRAC;
    assign y3_re = t3_re_full >>> FRAC;
    assign y3_im = t3_im_full >>> FRAC;

    // y0 has no twiddle — just the sum
    assign y0_re = s0_re[WIDTH-1:0];
    assign y0_im = s0_im[WIDTH-1:0];

endmodule