`default_nettype none
//==============================================================================
// modrelu.v — modReLU Activation Module
//==============================================================================
// modReLU(z, b) = z * sign(|z| + b)
//
// Given complex z = re + j*im and real bias b:
//   1. Compute approximate magnitude |z| ≈ max(|re|, |im|) + 0.5*min(|re|, |im|)
//      (avoids CORDIC for simplicity and area efficiency).
//   2. Compute |z| + b.
//   3. If (|z| + b) > 0, output = z (pass-through).
//      If (|z| + b) <= 0, output = 0.
//      This is equivalent to: out_re = re * sign(|z|+b), out_im = im * sign(|z|+b)
//
// In Q8.8 fixed-point, the magnitude computation uses signed arithmetic.
// The bias b is also in Q8.8.
//
// Parameters:
//   WIDTH = 16 (Q8.8)
//
// Prompt 18 specification.
//==============================================================================
module modrelu #(
    parameter WIDTH = 16,
    parameter FRAC  = 8
) (
    input  wire                    clk,
    input  wire                    rst_n,

    // Streaming input interface
    input  wire                    data_in_valid,
    output reg                     data_in_ready,
    input  wire signed [WIDTH-1:0] data_in_re,
    input  wire signed [WIDTH-1:0] data_in_im,
    input  wire signed [WIDTH-1:0] bias,           // modReLU bias (Q8.8, real)

    // Streaming output interface
    output reg                     data_out_valid,
    input  wire                    data_out_ready,
    output reg  signed [WIDTH-1:0] data_out_re,
    output reg  signed [WIDTH-1:0] data_out_im
);

    //----------------------------------------------------------------------
    // Approximate magnitude: |z| ≈ max(|re|, |im|) + 0.5 * min(|re|, |im|)
    //----------------------------------------------------------------------

    // Absolute values (two's complement negation)
    wire signed [WIDTH-1:0] abs_re = data_in_re[WIDTH-1] ? (~data_in_re + 1'b1) : data_in_re;
    wire signed [WIDTH-1:0] abs_im = data_in_im[WIDTH-1] ? (~data_in_im + 1'b1) : data_in_im;

    // max and min of |re|, |im|
    wire signed [WIDTH-1:0] max_val = (abs_re > abs_im) ? abs_re : abs_im;
    wire signed [WIDTH-1:0] min_val = (abs_re > abs_im) ? abs_im : abs_re;

    // min/2 (arithmetic right shift by 1)
    wire signed [WIDTH-1:0] half_min = min_val >>> 1;

    // Approximate magnitude
    wire signed [WIDTH-1:0] mag_z = max_val + half_min;

    //----------------------------------------------------------------------
    // Compute |z| + b and determine sign
    //----------------------------------------------------------------------
    wire signed [WIDTH-1:0] mag_plus_b = mag_z + bias;

    // sign(|z| + b): 1 if positive (> 0), 0 if <= 0
    // If |z| + b > 0, output = z (pass-through)
    // If |z| + b <= 0, output = 0
    wire pass_through = (mag_plus_b > 0);

    // Output: z * sign(|z|+b)
    wire signed [WIDTH-1:0] out_re_comb = pass_through ? data_in_re : {WIDTH{1'b0}};
    wire signed [WIDTH-1:0] out_im_comb = pass_through ? data_in_im : {WIDTH{1'b0}};

    //----------------------------------------------------------------------
    // Pipeline register (1-cycle latency for timing)
    //----------------------------------------------------------------------
    reg                   valid_r;
    reg signed [WIDTH-1:0] out_re_r;
    reg signed [WIDTH-1:0] out_im_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_r       <= 1'b0;
            out_re_r      <= 0;
            out_im_r      <= 0;
            data_in_ready <= 1'b0;
            data_out_valid<= 1'b0;
            data_out_re   <= 0;
            data_out_im   <= 0;
        end else begin
            // Accept input
            data_in_ready <= 1'b1;
            if (data_in_valid && data_in_ready) begin
                out_re_r <= out_re_comb;
                out_im_r <= out_im_comb;
                valid_r  <= 1'b1;
            end else begin
                valid_r  <= 1'b0;
            end

            // Output registered result
            if (valid_r) begin
                data_out_valid <= 1'b1;
                data_out_re    <= out_re_r;
                data_out_im    <= out_im_r;
            end else begin
                data_out_valid <= 1'b0;
            end
        end
    end

endmodule