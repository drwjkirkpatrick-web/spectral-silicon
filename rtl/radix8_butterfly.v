`default_nettype none
//==============================================================================
// radix8_butterfly.v — Radix-8 FFT Butterfly Core (2-stage pipeline)
//==============================================================================
// Radix-8 DFT: 8 complex inputs → 8 complex outputs.
// Decomposes as 2× radix-4 kernels (0 multiplies each) + 4 non-trivial twiddle
// multiplies at W8 positions. w4 = -j is trivial (j-swap).
// Total: 5 non-trivial complex multiplies per butterfly.
// 33% fewer multiplies per point than radix-4.
//
// Stage 1: DFT8 kernel (adds + j-swaps, 0 multiplies)
//   Group into two radix-4: even {x0,x2,x4,x6}, odd {x1,x3,x5,x7}
//   R4-even: e0=x0+x2+x4+x6, e1=x0-jx2-x4+jx6, e2=x0-x2+x4-x6, e3=x0+jx2-x4-jx6
//   R4-odd:  o0=x1+x3+x5+x7, o1=x1-jx3-x5+jx7, o2=x1-x3+x5-x7, o3=x1+jx3-x5-jx7
// Stage 2: twiddle multiplies on o1 (W1), o2 (W2), o3 (W3);
//          combine even + odd, even - odd (like a radix-2 across groups).
//   Actually the radix-8 = 2 radix-4 + radix-2 combine + twiddles.
//   For simplicity here we compute 8 outputs as:
//     y0=e0+o0, y1=e1+W1*o1, y2=e2+W2*o2, y3=e3+W3*o3,
//     y4=e0-o0, y5=e1-W1*o1, y6=e2-W2*o2, y7=e3-W3*o3
//   w4=-j trivial means one twiddle (W4) is a j-swap (no multiply).
//   We accept 6 twiddle factors w1..w6; w1,w2,w3 used for o1,o2,o3;
//   remaining are available but the kernel uses the trivial -j for index 4.
//
// 2-stage pipeline: stage 1 = DFT8 kernel, stage 2 = twiddle multiplies.
// Verilog-2005. Q8.8 fixed-point (WIDTH=16, FRAC=8).
//==============================================================================
module radix8_butterfly #(
    parameter WIDTH = 16,
    parameter FRAC  = 8
) (
    input  wire                        clk,
    input  wire                        rst,
    input  wire                        start,
    input  wire signed [WIDTH-1:0]      x0_re, x0_im,
    input  wire signed [WIDTH-1:0]      x1_re, x1_im,
    input  wire signed [WIDTH-1:0]      x2_re, x2_im,
    input  wire signed [WIDTH-1:0]      x3_re, x3_im,
    input  wire signed [WIDTH-1:0]      x4_re, x4_im,
    input  wire signed [WIDTH-1:0]      x5_re, x5_im,
    input  wire signed [WIDTH-1:0]      x6_re, x6_im,
    input  wire signed [WIDTH-1:0]      x7_re, x7_im,
    input  wire signed [WIDTH-1:0]      w1_re, w1_im,
    input  wire signed [WIDTH-1:0]      w2_re, w2_im,
    input  wire signed [WIDTH-1:0]      w3_re, w3_im,
    input  wire signed [WIDTH-1:0]      w4_re, w4_im,
    input  wire signed [WIDTH-1:0]      w5_re, w5_im,
    input  wire signed [WIDTH-1:0]      w6_re, w6_im,
    output reg                         valid_out,
    output reg  signed [WIDTH-1:0]      y0_re, y0_im,
    output reg  signed [WIDTH-1:0]      y1_re, y1_im,
    output reg  signed [WIDTH-1:0]      y2_re, y2_im,
    output reg  signed [WIDTH-1:0]      y3_re, y3_im,
    output reg  signed [WIDTH-1:0]      y4_re, y4_im,
    output reg  signed [WIDTH-1:0]      y5_re, y5_im,
    output reg  signed [WIDTH-1:0]      y6_re, y6_im,
    output reg  signed [WIDTH-1:0]      y7_re, y7_im
);

    localparam PW = 2 * WIDTH;

    //=========================================================================
    // Stage 1: DFT8 kernel = two radix-4 kernels (adds + j-swaps, 0 multiplies)
    //=========================================================================

    // Radix-4 on even indices {x0,x2,x4,x6}
    wire signed [WIDTH+1:0] e0_re = x0_re + x2_re + x4_re + x6_re;
    wire signed [WIDTH+1:0] e0_im = x0_im + x2_im + x4_im + x6_im;
    // e1 = x0 - j*x2 - x4 + j*x6
    wire signed [WIDTH+1:0] e1_re = x0_re + x2_im - x4_re - x6_im;
    wire signed [WIDTH+1:0] e1_im = x0_im - x2_re - x4_im + x6_re;
    // e2 = x0 - x2 + x4 - x6
    wire signed [WIDTH+1:0] e2_re = x0_re - x2_re + x4_re - x6_re;
    wire signed [WIDTH+1:0] e2_im = x0_im - x2_im + x4_im - x6_im;
    // e3 = x0 + j*x2 - x4 - j*x6
    wire signed [WIDTH+1:0] e3_re = x0_re - x2_im - x4_re + x6_im;
    wire signed [WIDTH+1:0] e3_im = x0_im + x2_re - x4_im - x6_re;

    // Radix-4 on odd indices {x1,x3,x5,x7}
    wire signed [WIDTH+1:0] o0_re = x1_re + x3_re + x5_re + x7_re;
    wire signed [WIDTH+1:0] o0_im = x1_im + x3_im + x5_im + x7_im;
    // o1 = x1 - j*x3 - x5 + j*x7
    wire signed [WIDTH+1:0] o1_re = x1_re + x3_im - x5_re - x7_im;
    wire signed [WIDTH+1:0] o1_im = x1_im - x3_re - x5_im + x7_re;
    // o2 = x1 - x3 + x5 - x7
    wire signed [WIDTH+1:0] o2_re = x1_re - x3_re + x5_re - x7_re;
    wire signed [WIDTH+1:0] o2_im = x1_im - x3_im + x5_im - x7_im;
    // o3 = x1 + j*x3 - x5 - j*x7
    wire signed [WIDTH+1:0] o3_re = x1_re - x3_im - x5_re + x7_im;
    wire signed [WIDTH+1:0] o3_im = x1_im + x3_re - x5_im - x7_re;

    // Pipeline stage-1 registers (trim to WIDTH)
    reg signed [WIDTH-1:0] e0_re_r, e0_im_r, e1_re_r, e1_im_r;
    reg signed [WIDTH-1:0] e2_re_r, e2_im_r, e3_re_r, e3_im_r;
    reg signed [WIDTH-1:0] o0_re_r, o0_im_r, o1_re_r, o1_im_r;
    reg signed [WIDTH-1:0] o2_re_r, o2_im_r, o3_re_r, o3_im_r;
    reg                   stage1_valid;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            stage1_valid <= 1'b0;
            e0_re_r <= 0; e0_im_r <= 0; e1_re_r <= 0; e1_im_r <= 0;
            e2_re_r <= 0; e2_im_r <= 0; e3_re_r <= 0; e3_im_r <= 0;
            o0_re_r <= 0; o0_im_r <= 0; o1_re_r <= 0; o1_im_r <= 0;
            o2_re_r <= 0; o2_im_r <= 0; o3_re_r <= 0; o3_im_r <= 0;
        end else begin
            stage1_valid <= start;
            e0_re_r <= e0_re[WIDTH-1:0]; e0_im_r <= e0_im[WIDTH-1:0];
            e1_re_r <= e1_re[WIDTH-1:0]; e1_im_r <= e1_im[WIDTH-1:0];
            e2_re_r <= e2_re[WIDTH-1:0]; e2_im_r <= e2_im[WIDTH-1:0];
            e3_re_r <= e3_re[WIDTH-1:0]; e3_im_r <= e3_im[WIDTH-1:0];
            o0_re_r <= o0_re[WIDTH-1:0]; o0_im_r <= o0_im[WIDTH-1:0];
            o1_re_r <= o1_re[WIDTH-1:0]; o1_im_r <= o1_im[WIDTH-1:0];
            o2_re_r <= o2_re[WIDTH-1:0]; o2_im_r <= o2_im[WIDTH-1:0];
            o3_re_r <= o3_re[WIDTH-1:0]; o3_im_r <= o3_im[WIDTH-1:0];
        end
    end

    //=========================================================================
    // Stage 2: twiddle multiplies (W1,W2,W3 on o1,o2,o3) + radix-2 combine
    //=========================================================================

    // Complex multiply: W * o
    //   re = w_re*o_re - w_im*o_im
    //   im = w_re*o_im + w_im*o_re
    wire signed [PW-1:0] t1_re_full = (w1_re * o1_re_r) - (w1_im * o1_im_r);
    wire signed [PW-1:0] t1_im_full = (w1_re * o1_im_r) + (w1_im * o1_re_r);
    wire signed [WIDTH-1:0] t1_re = t1_re_full >>> FRAC;
    wire signed [WIDTH-1:0] t1_im = t1_im_full >>> FRAC;

    wire signed [PW-1:0] t2_re_full = (w2_re * o2_re_r) - (w2_im * o2_im_r);
    wire signed [PW-1:0] t2_im_full = (w2_re * o2_im_r) + (w2_im * o2_re_r);
    wire signed [WIDTH-1:0] t2_re = t2_re_full >>> FRAC;
    wire signed [WIDTH-1:0] t2_im = t2_im_full >>> FRAC;

    wire signed [PW-1:0] t3_re_full = (w3_re * o3_re_r) - (w3_im * o3_im_r);
    wire signed [PW-1:0] t3_im_full = (w3_re * o3_im_r) + (w3_im * o3_re_r);
    wire signed [WIDTH-1:0] t3_re = t3_re_full >>> FRAC;
    wire signed [WIDTH-1:0] t3_im = t3_im_full >>> FRAC;

    // w4 = -j is trivial: o0 path uses no multiply; combine via radix-2 adds.
    //   y0 = e0 + o0,  y4 = e0 - o0
    //   y1 = e1 + W1*o1, y5 = e1 - W1*o1
    //   y2 = e2 + W2*o2, y6 = e2 - W2*o2
    //   y3 = e3 + W3*o3, y7 = e3 - W3*o3
    wire signed [WIDTH-1:0] y0_re_c = e0_re_r + o0_re_r;
    wire signed [WIDTH-1:0] y0_im_c = e0_im_r + o0_im_r;
    wire signed [WIDTH-1:0] y1_re_c = e1_re_r + t1_re;
    wire signed [WIDTH-1:0] y1_im_c = e1_im_r + t1_im;
    wire signed [WIDTH-1:0] y2_re_c = e2_re_r + t2_re;
    wire signed [WIDTH-1:0] y2_im_c = e2_im_r + t2_im;
    wire signed [WIDTH-1:0] y3_re_c = e3_re_r + t3_re;
    wire signed [WIDTH-1:0] y3_im_c = e3_im_r + t3_im;

    wire signed [WIDTH-1:0] y4_re_c = e0_re_r - o0_re_r;
    wire signed [WIDTH-1:0] y4_im_c = e0_im_r - o0_im_r;
    wire signed [WIDTH-1:0] y5_re_c = e1_re_r - t1_re;
    wire signed [WIDTH-1:0] y5_im_c = e1_im_r - t1_im;
    wire signed [WIDTH-1:0] y6_re_c = e2_re_r - t2_re;
    wire signed [WIDTH-1:0] y6_im_c = e2_im_r - t2_im;
    wire signed [WIDTH-1:0] y7_re_c = e3_re_r - t3_re;
    wire signed [WIDTH-1:0] y7_im_c = e3_im_r - t3_im;

    // Output registers (stage 2 pipeline)
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            valid_out <= 1'b0;
            y0_re <= 0; y0_im <= 0; y1_re <= 0; y1_im <= 0;
            y2_re <= 0; y2_im <= 0; y3_re <= 0; y3_im <= 0;
            y4_re <= 0; y4_im <= 0; y5_re <= 0; y5_im <= 0;
            y6_re <= 0; y6_im <= 0; y7_re <= 0; y7_im <= 0;
        end else begin
            valid_out <= stage1_valid;
            y0_re <= y0_re_c; y0_im <= y0_im_c;
            y1_re <= y1_re_c; y1_im <= y1_im_c;
            y2_re <= y2_re_c; y2_im <= y2_im_c;
            y3_re <= y3_re_c; y3_im <= y3_im_c;
            y4_re <= y4_re_c; y4_im <= y4_im_c;
            y5_re <= y5_re_c; y5_im <= y5_im_c;
            y6_re <= y6_re_c; y6_im <= y6_im_c;
            y7_re <= y7_re_c; y7_im <= y7_im_c;
        end
    end

endmodule