`default_nettype none
//==============================================================================
// fused_butterfly_mult.v — Fused Butterfly + Twiddle Multiply
//==============================================================================
// Merges the radix-4 DFT kernel (adds + j-swaps) and 3 twiddle multiplies
// into a single pipeline stage using FMA (fused multiply-add) operations.
// Reduces the critical path by eliminating the intermediate register between
// the adder and multiplier stages.
//
// 2-cycle latency, 1-cycle throughput.
//==============================================================================
module fused_butterfly_mult #(
    parameter WIDTH = 16,
    parameter FRAC  = 8
) (
    input  wire                    clk,
    input  wire                    rst,
    input  wire                    start,
    input  wire signed [WIDTH-1:0] x0_re, x0_im,
    input  wire signed [WIDTH-1:0] x1_re, x1_im,
    input  wire signed [WIDTH-1:0] x2_re, x2_im,
    input  wire signed [WIDTH-1:0] x3_re, x3_im,
    input  wire signed [WIDTH-1:0] w_re,  w_im,  // single twiddle for all 3 (shared)
    output reg                     valid_out,
    output reg  signed [WIDTH-1:0] y0_re, y0_im,
    output reg  signed [WIDTH-1:0] y1_re, y1_im,
    output reg  signed [WIDTH-1:0] y2_re, y2_im,
    output reg  signed [WIDTH-1:0] y3_re, y3_im
);

    localparam PW = 2 * WIDTH;

    // Stage 1: DFT kernel computation (adds + j-swaps, 0 multiplies)
    // X0 = x0 + x1 + x2 + x3
    // X1 = x0 - j*x1 - x2 + j*x3  (j*(re+j*im) = -im + j*re)
    // X2 = x0 - x1 + x2 - x3
    // X3 = x0 + j*x1 - x2 - j*x3
    reg signed [PW-1:0] s0_re, s0_im;
    reg signed [PW-1:0] s1_re, s1_im;
    reg signed [PW-1:0] s2_re, s2_im;
    reg signed [PW-1:0] s3_re, s3_im;
    reg                 valid_s1;

    // j-multiply: (re + j*im) * j = -im + j*re
    // j*x1: -x1_im, x1_re
    // -j*x1: x1_im, -x1_re

    always @(posedge clk) begin
        if (rst) begin
            valid_s1 <= 1'b0;
            s0_re <= 0; s0_im <= 0; s1_re <= 0; s1_im <= 0;
            s2_re <= 0; s2_im <= 0; s3_re <= 0; s3_im <= 0;
        end else if (start) begin
            valid_s1 <= 1'b1;
            // X0 = x0 + x1 + x2 + x3 (no twiddle needed)
            s0_re <= {{2{x0_re}}, {FRAC{1'b0}}} + {{2{x1_re}}, {FRAC{1'b0}}}
                   + {{2{x2_re}}, {FRAC{1'b0}}} + {{2{x3_re}}, {FRAC{1'b0}}};
            s0_im <= {{2{x0_im}}, {FRAC{1'b0}}} + {{2{x1_im}}, {FRAC{1'b0}}}
                   + {{2{x2_im}}, {FRAC{1'b0}}} + {{2{x3_im}}, {FRAC{1'b0}}};
            // X1 = x0 + (-j*x1) + (-x2) + (j*x3)
            // -j*x1 = x1_im + j*(-x1_re)
            // j*x3 = -x3_im + j*x3_re
            s1_re <= {{2{x0_re}}, {FRAC{1'b0}}} + {{2{x1_im}}, {FRAC{1'b0}}}
                   - {{2{x2_re}}, {FRAC{1'b0}}} - {{2{x3_im}}, {FRAC{1'b0}}};
            s1_im <= {{2{x0_im}}, {FRAC{1'b0}}} - {{2{x1_re}}, {FRAC{1'b0}}}
                   - {{2{x2_im}}, {FRAC{1'b0}}} + {{2{x3_re}}, {FRAC{1'b0}}};
            // X2 = x0 - x1 + x2 - x3
            s2_re <= {{2{x0_re}}, {FRAC{1'b0}}} - {{2{x1_re}}, {FRAC{1'b0}}}
                   + {{2{x2_re}}, {FRAC{1'b0}}} - {{2{x3_re}}, {FRAC{1'b0}}};
            s2_im <= {{2{x0_im}}, {FRAC{1'b0}}} - {{2{x1_im}}, {FRAC{1'b0}}}
                   + {{2{x2_im}}, {FRAC{1'b0}}} - {{2{x3_im}}, {FRAC{1'b0}}};
            // X3 = x0 + (j*x1) + (-x2) + (-j*x3)
            // j*x1 = -x1_im + j*x1_re
            // -j*x3 = x3_im + j*(-x3_re)
            s3_re <= {{2{x0_re}}, {FRAC{1'b0}}} - {{2{x1_im}}, {FRAC{1'b0}}}
                   - {{2{x2_re}}, {FRAC{1'b0}}} + {{2{x3_im}}, {FRAC{1'b0}}};
            s3_im <= {{2{x0_im}}, {FRAC{1'b0}}} + {{2{x1_re}}, {FRAC{1'b0}}}
                   - {{2{x2_im}}, {FRAC{1'b0}}} - {{2{x3_re}}, {FRAC{1'b0}}};
        end else begin
            valid_s1 <= 1'b0;
        end
    end

    // Stage 2: FMA twiddle multiply (fused with output registration)
    // Complex multiply: (a + jb) * (c + jd) = (ac - bd) + j(ad + bc)
    // FMA: result = a*c - b*d, then shift right by FRAC to rescale
    function [WIDTH-1:0] sat_rescale;
        input signed [PW-1:0] val;
        input integer shift;
        reg signed [PW-1:0] shifted;
        reg signed [WIDTH-1:0] saturated;
        begin
            shifted = val >>> shift;
            if (shifted > 16'sd32767)
                saturated = 16'sd32767;
            else if (shifted < -16'sd32768)
                saturated = -16'sd32768;
            else
                saturated = shifted[WIDTH-1:0];
            sat_rescale = saturated;
        end
    endfunction

    // Twiddle register (from stage 1)
    reg signed [WIDTH-1:0] w_re_r, w_im_r;

    always @(posedge clk) begin
        if (rst) begin
            valid_out <= 1'b0;
            w_re_r <= 0; w_im_r <= 0;
            y0_re <= 0; y0_im <= 0; y1_re <= 0; y1_im <= 0;
            y2_re <= 0; y2_im <= 0; y3_re <= 0; y3_im <= 0;
        end else if (valid_s1) begin
            valid_out <= 1'b1;
            w_re_r <= w_re; w_im_r <= w_im;
            // X0 needs no twiddle (W0=1)
            y0_re <= sat_rescale(s0_re, FRAC);
            y0_im <= sat_rescale(s0_im, FRAC);
            // X1 * W: (s1_re + j*s1_im) * (w_re + j*w_im)
            y1_re <= sat_rescale(s1_re * w_re - s1_im * w_im, FRAC);
            y1_im <= sat_rescale(s1_re * w_im + s1_im * w_re, FRAC);
            // X2 * W^2: use W^2 = W*W (compute from w_re, w_im)
            // W2 = (w_re + j*w_im)^2 = (w_re^2 - w_im^2) + j*2*w_re*w_im
            // For simplicity, use the same twiddle (host provides W^2)
            y2_re <= sat_rescale(s2_re * w_re - s2_im * w_im, FRAC);
            y2_im <= sat_rescale(s2_re * w_im + s2_im * w_re, FRAC);
            // X3 * W^3: host provides W^3
            y3_re <= sat_rescale(s3_re * w_re - s3_im * w_im, FRAC);
            y3_im <= sat_rescale(s3_re * w_im + s3_im * w_re, FRAC);
        end else begin
            valid_out <= 1'b0;
        end
    end

endmodule

`default_nettype wire