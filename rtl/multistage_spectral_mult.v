`default_nettype none
//==============================================================================
// multistage_spectral_mult.v — 3-Stage Pipelined Spectral Multiply
//==============================================================================
// Computes the complex spectral product
//     result = mode * weight = (mode_re + j*mode_im) * (weight_re + j*weight_im)
// in three pipelined stages with a 3-cycle latency and 1-cycle throughput.
//
//   Stage 1 (real part):  mode_re * weight_re - mode_im * weight_im
//   Stage 2 (imag part):  mode_re * weight_im + mode_im * weight_re
//   Stage 3 (accumulate + soft-threshold):
//       Compare the approximate magnitude of the product against a fixed
//       threshold; pass the product through or zero it.
//
// All arithmetic is Q8.8 fixed-point (WIDTH=16, FRAC=8).  Products are formed
// at 2*WIDTH precision and rescaled with an arithmetic right shift by FRAC.
// Each rescale is saturating: if the value exceeds the signed Q8.8 range it is
// clamped to ±max instead of wrapping.
//
// The `start` pulse launches a computation on the current inputs; `valid_out`
// asserts exactly 3 cycles after `start`.
//
// Verilog-2005, `default_nettype none.  Synthesizable.
//==============================================================================
module multistage_spectral_mult #(
    parameter WIDTH   = 16,
    parameter FRAC    = 8
) (
    input  wire                   clk,
    input  wire                   rst,
    input  wire                   start,
    input  wire signed [WIDTH-1:0] mode_re,
    input  wire signed [WIDTH-1:0] mode_im,
    input  wire signed [WIDTH-1:0] weight_re,
    input  wire signed [WIDTH-1:0] weight_im,
    output reg                     valid_out,
    output reg  signed [WIDTH-1:0] result_re,
    output reg  signed [WIDTH-1:0] result_im
);

    localparam PW = 2 * WIDTH;           // product width = 32
    localparam signed [WIDTH-1:0] SAT_POS = {1'b0, {(WIDTH-1){1'b1}}};  // +127.996
    localparam signed [WIDTH-1:0] SAT_NEG = {1'b1, {(WIDTH-1){1'b0}}};  // -128.000

    // Soft-threshold constant in Q8.8 (e.g. 0.03125 → 8 → 0x0008).
    localparam signed [WIDTH-1:0] THRESH = 16'sd8;

    //----------------------------------------------------------------------
    // Saturation helper: rescale a 2*WIDTH product to WIDTH with clamping.
    //----------------------------------------------------------------------
    function [WIDTH-1:0] sat_rescale;
        input signed [PW-1:0] prod;
        reg  signed [PW-1:0]  shifted;
        begin
            shifted = prod >>> FRAC;     // arithmetic right shift
            if (shifted > {{(PW-WIDTH){1'b0}}, SAT_POS})
                sat_rescale = SAT_POS;
            else if (shifted < {{(PW-WIDTH){1'b1}}, SAT_NEG})
                sat_rescale = SAT_NEG;
            else
                sat_rescale = shifted[WIDTH-1:0];
        end
    endfunction

    //----------------------------------------------------------------------
    // Stage 1: real-part multiply
    //   re_full = mode_re * weight_re - mode_im * weight_im
    //----------------------------------------------------------------------
    wire signed [PW-1:0] re_prod =
        (mode_re * weight_re) - (mode_im * weight_im);

    reg signed [WIDTH-1:0] s1_re;
    reg                    s1_valid;

    always @(posedge clk) begin
        if (rst) begin
            s1_re    <= {WIDTH{1'b0}};
            s1_valid <= 1'b0;
        end else begin
            s1_re    <= sat_rescale(re_prod);
            s1_valid <= start;
        end
    end

    //----------------------------------------------------------------------
    // Stage 2: imaginary-part multiply
    //   im_full = mode_re * weight_im + mode_im * weight_re
    // The real part is carried forward unchanged.
    //----------------------------------------------------------------------
    wire signed [PW-1:0] im_prod =
        (mode_re * weight_im) + (mode_im * weight_re);

    reg signed [WIDTH-1:0] s2_re;
    reg signed [WIDTH-1:0] s2_im;
    reg                    s2_valid;

    always @(posedge clk) begin
        if (rst) begin
            s2_re    <= {WIDTH{1'b0}};
            s2_im    <= {WIDTH{1'b0}};
            s2_valid <= 1'b0;
        end else begin
            s2_re    <= s1_re;
            s2_im    <= sat_rescale(im_prod);
            s2_valid <= s1_valid;
        end
    end

    //----------------------------------------------------------------------
    // Stage 3: accumulate + soft-threshold
    // Approximate magnitude |z| ≈ max(|re|,|im|) + 0.5*min(|re|,|im|).
    // If |z| < THRESH the output is zeroed; otherwise pass through.
    //----------------------------------------------------------------------
    wire signed [WIDTH-1:0] abs_s2_re = s2_re[WIDTH-1] ? (~s2_re + 1'b1) : s2_re;
    wire signed [WIDTH-1:0] abs_s2_im = s2_im[WIDTH-1] ? (~s2_im + 1'b1) : s2_im;

    wire signed [WIDTH-1:0] max_abs = (abs_s2_re > abs_s2_im) ? abs_s2_re : abs_s2_im;
    wire signed [WIDTH-1:0] min_abs = (abs_s2_re > abs_s2_im) ? abs_s2_im : abs_s2_re;

    wire signed [WIDTH-1:0] mag_s2 = max_abs + (min_abs >>> 1);
    wire                   below_thresh = (mag_s2 < THRESH);

    wire signed [WIDTH-1:0] thr_re = below_thresh ? {WIDTH{1'b0}} : s2_re;
    wire signed [WIDTH-1:0] thr_im = below_thresh ? {WIDTH{1'b0}} : s2_im;

    always @(posedge clk) begin
        if (rst) begin
            valid_out <= 1'b0;
            result_re <= {WIDTH{1'b0}};
            result_im <= {WIDTH{1'b0}};
        end else begin
            valid_out <= s2_valid;
            result_re <= thr_re;
            result_im <= thr_im;
        end
    end

endmodule

`default_nettype wire