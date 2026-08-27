`default_nettype none
//==============================================================================
// residual_add.v — Residual Connection Adder
//==============================================================================
// Adds the spectral mixer output back to the original input embeddings:
//   out = mixer_out + input    (complex addition, with saturation)
//
// This eliminates a host round-trip for every transformer layer — the residual
// sum now happens on-chip in 1 cycle instead of reading back, adding on the
// host CPU, and re-sending.
//
// ~200 gates.  Q8.8 fixed-point, WIDTH=16.
//==============================================================================
module residual_add #(
    parameter WIDTH = 16
) (
    input  wire                    clk,
    input  wire                    rst_n,

    // Original input embeddings (complex)
    input  wire                    data_in_valid,
    output reg                     data_in_ready,
    input  wire signed [WIDTH-1:0] in_re,
    input  wire signed [WIDTH-1:0] in_im,

    // Spectral mixer output (complex)
    input  wire                    mixer_valid,
    input  wire signed [WIDTH-1:0] mixer_re,
    input  wire signed [WIDTH-1:0] mixer_im,

    // Residual sum output (complex)
    output reg                     data_out_valid,
    input  wire                    data_out_ready,
    output reg  signed [WIDTH-1:0] out_re,
    output reg  signed [WIDTH-1:0] out_im
);

    // Saturation helper
    function [WIDTH-1:0] saturate;
        input signed [WIDTH:0] val;  // one extra bit for overflow detection
        begin
            if (val > {1'b0, {(WIDTH-1){1'b1}}})       // > +32767
                saturate = {(WIDTH-1){1'b1}};           // clamp to max
            else if (val < {1'b1, {(WIDTH-1){1'b0}}})   // < -32768
                saturate = {1'b1, {(WIDTH-1){1'b0}}};   // clamp to min
            else
                saturate = val[WIDTH-1:0];
        end
    endfunction

    // Combinational sum
    wire signed [WIDTH:0] sum_re = in_re + mixer_re;
    wire signed [WIDTH:0] sum_im = in_im + mixer_im;

    // Accept when both inputs valid and downstream can accept (or no output pending)
    wire both_valid = data_in_valid && mixer_valid;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_in_ready  <= 1'b0;
            data_out_valid <= 1'b0;
            out_re         <= 0;
            out_im         <= 0;
        end else begin
            data_in_ready <= data_out_ready || !data_out_valid;

            if (both_valid && (data_out_ready || !data_out_valid)) begin
                out_re         <= saturate(sum_re);
                out_im         <= saturate(sum_im);
                data_out_valid <= 1'b1;
            end else if (data_out_ready && data_out_valid) begin
                data_out_valid <= 1'b0;
            end
        end
    end

endmodule

`default_nettype wire