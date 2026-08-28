`default_nettype none
//==============================================================================
// cordic_twiddle.v — CORDIC-Based Twiddle Factor Generator
//==============================================================================
// Computes cos/sin on-the-fly using 12-stage CORDIC rotation, eliminating
// the need for twiddle ROM (~2KB storage saved).
//
// Input: angle in Q8.8 fixed-point, range [-pi, pi] (mapped to [-128, 128])
// Output: cos_out, sin_out in Q8.8
//
// 12-stage CORDIC: ~12-bit accuracy (sufficient for Q8.8).
// 12-cycle latency, pipelined for 1-cycle throughput.
//==============================================================================
module cordic_twiddle #(
    parameter STAGES = 12,
    parameter WIDTH  = 16,
    parameter FRAC   = 8
) (
    input  wire                    clk,
    input  wire                    rst,
    input  wire                    start,
    input  wire signed [WIDTH-1:0] angle,       // Q8.8 angle in [-pi, pi]
    output reg                     valid_out,
    output reg  signed [WIDTH-1:0] cos_out,
    output reg  signed [WIDTH-1:0] sin_out
);

    // CORDIC atan lookup table (Q8.8 values of atan(2^-i))
    // atan(2^0) = 45°     = 0x0032 (in units of pi/256 ≈ 0.703°)
    // Using a simpler encoding: angle in Q8.8 where pi = 0x0324 (804)
    // We use a scaled angle where pi/2 = 0x0192 (402)
    // Actually, let's use a straightforward approach:
    // The input angle is in Q8.8 radians. pi ≈ 3.14159 → 3.14159 * 256 = 804 = 0x0324
    // atan(2^-i) in Q8.8 radians:
    // i=0: atan(1) = pi/4 = 0x00C9 (201)
    // i=1: atan(0.5) = 0.4636 → 0x0077 (119)
    // i=2: atan(0.25) = 0.2450 → 0x003E (62)
    // i=3: atan(0.125) = 0.1244 → 0x001F (31)
    // i=4: atan(0.0625) = 0.0624 → 0x0010 (16)
    // i=5: atan(0.03125) = 0.0312 → 0x0008 (8)
    // i=6..11: decreasing by ~2× each

    // CORDIC gain for 12 stages: 1.6468 → 1/gain = 0.6073
    // In Q8.8: 0.6073 * 256 = 155 = 0x009B
    localparam CORDIC_GAIN_INV = 16'h009B;  // 1/1.6468 in Q8.8

    // Pipeline registers
    reg signed [WIDTH-1:0] x_pipe [0:STAGES];
    reg signed [WIDTH-1:0] y_pipe [0:STAGES];
    reg signed [WIDTH-1:0] z_pipe [0:STAGES];
    reg                   valid_pipe [0:STAGES];

    // Atan table (Q8.8 radians)
    // atan(2^-i) * 256 rounded
    function [WIDTH-1:0] atan_lut;
        input [3:0] idx;
        begin
            case (idx)
                4'd0:  atan_lut = 16'h00C9; // atan(1) = 0.7854 → 201
                4'd1:  atan_lut = 16'h0077; // atan(0.5) = 0.4636 → 119
                4'd2:  atan_lut = 16'h003E; // atan(0.25) = 0.2450 → 62
                4'd3:  atan_lut = 16'h001F; // atan(0.125) = 0.1244 → 31
                4'd4:  atan_lut = 16'h0010; // atan(0.0625) = 0.0624 → 16
                4'd5:  atan_lut = 16'h0008; // atan(0.03125) → 8
                4'd6:  atan_lut = 16'h0004; // → 4
                4'd7:  atan_lut = 16'h0002; // → 2
                4'd8:  atan_lut = 16'h0001; // → 1
                4'd9:  atan_lut = 16'h0001; // → 1 (rounded)
                4'd10: atan_lut = 16'h0000; // → 0 (negligible)
                4'd11: atan_lut = 16'h0000; // → 0
                default: atan_lut = 16'h0000;
            endcase
        end
    endfunction

    integer i;
    reg signed [WIDTH-1:0] atan_val;
    reg signed [2*WIDTH-1:0] x_shift, y_shift;
    reg                     dir;

    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i <= STAGES; i = i + 1) begin
                x_pipe[i] <= 0;
                y_pipe[i] <= 0;
                z_pipe[i] <= 0;
                valid_pipe[i] <= 1'b0;
            end
            valid_out <= 1'b0;
            cos_out <= 0;
            sin_out <= 0;
        end else begin
            // Stage 0: initialize
            x_pipe[0] <= CORDIC_GAIN_INV;  // 1/gain
            y_pipe[0] <= 0;
            z_pipe[0] <= angle;
            valid_pipe[0] <= start;

            // Iteration stages
            for (i = 0; i < STAGES; i = i + 1) begin
                if (valid_pipe[i]) begin
                    atan_val = atan_lut(i);
                    dir = z_pipe[i][WIDTH-1] == 1'b0; // positive z → +1
                    x_shift = $signed(x_pipe[i]) >>> i;
                    y_shift = $signed(y_pipe[i]) >>> i;
                    if (dir) begin
                        x_pipe[i+1] <= x_pipe[i] - y_shift[WIDTH-1:0];
                        y_pipe[i+1] <= y_pipe[i] + x_shift[WIDTH-1:0];
                        z_pipe[i+1] <= z_pipe[i] - atan_val;
                    end else begin
                        x_pipe[i+1] <= x_pipe[i] + y_shift[WIDTH-1:0];
                        y_pipe[i+1] <= y_pipe[i] - x_shift[WIDTH-1:0];
                        z_pipe[i+1] <= z_pipe[i] + atan_val;
                    end
                    valid_pipe[i+1] <= 1'b1;
                end else begin
                    valid_pipe[i+1] <= 1'b0;
                end
            end

            // Output stage
            valid_out <= valid_pipe[STAGES];
            cos_out <= x_pipe[STAGES];
            sin_out <= y_pipe[STAGES];
        end
    end

endmodule

`default_nettype wire