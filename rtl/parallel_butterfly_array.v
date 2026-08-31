`default_nettype none
//==============================================================================
// parallel_butterfly_array.v — 4× Parallel Radix-4 Butterfly Array
//==============================================================================
// Instantiates 4 butterfly4 modules in parallel. Processes 4 butterflies
// per clock cycle → 4× throughput improvement.
//
// Each butterfly instance receives 4 complex inputs + 3 twiddle pairs
// (12 twiddle pairs total for the 4 instances).
// 16 complex outputs (4 per instance).
//
// The array is combinational (inherits butterfly4's combinational logic).
// A 1-cycle pipeline register on valid_out for alignment.
//
// Verilog-2005. Q8.8 fixed-point (WIDTH=16, FRAC=8).
//==============================================================================
module parallel_butterfly_array #(
    parameter WIDTH = 16,
    parameter FRAC  = 8
) (
    input  wire                        clk,
    input  wire                        rst,
    input  wire                        start,
    // Instance 0
    input  wire signed [WIDTH-1:0]      x0_re, x0_im,
    input  wire signed [WIDTH-1:0]      x1_re, x1_im,
    input  wire signed [WIDTH-1:0]      x2_re, x2_im,
    input  wire signed [WIDTH-1:0]      x3_re, x3_im,
    // Instance 1
    input  wire signed [WIDTH-1:0]      x4_re, x4_im,
    input  wire signed [WIDTH-1:0]      x5_re, x5_im,
    input  wire signed [WIDTH-1:0]      x6_re, x6_im,
    input  wire signed [WIDTH-1:0]      x7_re, x7_im,
    // Instance 2
    input  wire signed [WIDTH-1:0]      x8_re, x8_im,
    input  wire signed [WIDTH-1:0]      x9_re, x9_im,
    input  wire signed [WIDTH-1:0]      x10_re, x10_im,
    input  wire signed [WIDTH-1:0]      x11_re, x11_im,
    // Instance 3
    input  wire signed [WIDTH-1:0]      x12_re, x12_im,
    input  wire signed [WIDTH-1:0]      x13_re, x13_im,
    input  wire signed [WIDTH-1:0]      x14_re, x14_im,
    input  wire signed [WIDTH-1:0]      x15_re, x15_im,
    // Twiddles: 12 pairs (3 per instance)
    input  wire signed [WIDTH-1:0]      w0_1_re, w0_1_im, w0_2_re, w0_2_im, w0_3_re, w0_3_im,
    input  wire signed [WIDTH-1:0]      w1_1_re, w1_1_im, w1_2_re, w1_2_im, w1_3_re, w1_3_im,
    input  wire signed [WIDTH-1:0]      w2_1_re, w2_1_im, w2_2_re, w2_2_im, w2_3_re, w2_3_im,
    input  wire signed [WIDTH-1:0]      w3_1_re, w3_1_im, w3_2_re, w3_2_im, w3_3_re, w3_3_im,
    output reg                          valid_out,
    // Instance 0 outputs
    output wire signed [WIDTH-1:0]     y0_re, y0_im, y1_re, y1_im,
    output wire signed [WIDTH-1:0]     y2_re, y2_im, y3_re, y3_im,
    // Instance 1 outputs
    output wire signed [WIDTH-1:0]     y4_re, y4_im, y5_re, y5_im,
    output wire signed [WIDTH-1:0]     y6_re, y6_im, y7_re, y7_im,
    // Instance 2 outputs
    output wire signed [WIDTH-1:0]     y8_re, y8_im, y9_re, y9_im,
    output wire signed [WIDTH-1:0]     y10_re, y10_im, y11_re, y11_im,
    // Instance 3 outputs
    output wire signed [WIDTH-1:0]     y12_re, y12_im, y13_re, y13_im,
    output wire signed [WIDTH-1:0]     y14_re, y14_im, y15_re, y15_im
);

    //--- Instance 0 ---
    butterfly4 #(.WIDTH(WIDTH), .FRAC(FRAC)) u_bf0 (
        .x0_re(x0_re), .x0_im(x0_im), .x1_re(x1_re), .x1_im(x1_im),
        .x2_re(x2_re), .x2_im(x2_im), .x3_re(x3_re), .x3_im(x3_im),
        .w1_re(w0_1_re), .w1_im(w0_1_im),
        .w2_re(w0_2_re), .w2_im(w0_2_im),
        .w3_re(w0_3_re), .w3_im(w0_3_im),
        .y0_re(y0_re), .y0_im(y0_im), .y1_re(y1_re), .y1_im(y1_im),
        .y2_re(y2_re), .y2_im(y2_im), .y3_re(y3_re), .y3_im(y3_im)
    );

    //--- Instance 1 ---
    butterfly4 #(.WIDTH(WIDTH), .FRAC(FRAC)) u_bf1 (
        .x0_re(x4_re), .x0_im(x4_im), .x1_re(x5_re), .x1_im(x5_im),
        .x2_re(x6_re), .x2_im(x6_im), .x3_re(x7_re), .x3_im(x7_im),
        .w1_re(w1_1_re), .w1_im(w1_1_im),
        .w2_re(w1_2_re), .w2_im(w1_2_im),
        .w3_re(w1_3_re), .w3_im(w1_3_im),
        .y0_re(y4_re), .y0_im(y4_im), .y1_re(y5_re), .y1_im(y5_im),
        .y2_re(y6_re), .y2_im(y6_im), .y3_re(y7_re), .y3_im(y7_im)
    );

    //--- Instance 2 ---
    butterfly4 #(.WIDTH(WIDTH), .FRAC(FRAC)) u_bf2 (
        .x0_re(x8_re), .x0_im(x8_im), .x1_re(x9_re), .x1_im(x9_im),
        .x2_re(x10_re), .x2_im(x10_im), .x3_re(x11_re), .x3_im(x11_im),
        .w1_re(w2_1_re), .w1_im(w2_1_im),
        .w2_re(w2_2_re), .w2_im(w2_2_im),
        .w3_re(w2_3_re), .w3_im(w2_3_im),
        .y0_re(y8_re), .y0_im(y8_im), .y1_re(y9_re), .y1_im(y9_im),
        .y2_re(y10_re), .y2_im(y10_im), .y3_re(y11_re), .y3_im(y11_im)
    );

    //--- Instance 3 ---
    butterfly4 #(.WIDTH(WIDTH), .FRAC(FRAC)) u_bf3 (
        .x0_re(x12_re), .x0_im(x12_im), .x1_re(x13_re), .x1_im(x13_im),
        .x2_re(x14_re), .x2_im(x14_im), .x3_re(x15_re), .x3_im(x15_im),
        .w1_re(w3_1_re), .w1_im(w3_1_im),
        .w2_re(w3_2_re), .w2_im(w3_2_im),
        .w3_re(w3_3_re), .w3_im(w3_3_im),
        .y0_re(y12_re), .y0_im(y12_im), .y1_re(y13_re), .y1_im(y13_im),
        .y2_re(y14_re), .y2_im(y14_im), .y3_re(y15_re), .y3_im(y15_im)
    );

    //--- Valid pipeline (1 cycle) ---
    always @(posedge clk or posedge rst) begin
        if (rst)
            valid_out <= 1'b0;
        else
            valid_out <= start;
    end

endmodule