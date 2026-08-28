`default_nettype none
//==============================================================================
// guard_bands.v — Guard Band Saturation for Q8.8 Fixed-Point
//==============================================================================
// Saturates a Q8.8 signed value at 95% of full-scale instead of 100%, leaving
// a 5% headroom band so that subsequent additions / multiplications cannot
// overflow on the next pipeline stage.  For Q8.8 the signed range is
// [-128.00, +127.996]; 95% of the positive max (~127.996) is ~121.6, which
// rounds down to 121 (0x3D00 in Q8.8).  The negative limit is -121 (0xC300).
//
//   data_out = clamp(data_in, -121.0, +121.0)  in Q8.8
//
// The saturated flag is asserted for the cycle in which clamping actually
// occurred.  The near_limit flag is asserted when the input magnitude is
// within 5% of the guard band (i.e. in the outer headroom region), giving the
// controller an early warning before saturation happens.
//
// Constants (Q8.8):
//   GUARD_MAX =  121.0 = 16'sh3D00  (0x3D00)
//   GUARD_MIN = -121.0 = 16'shC300  (two's complement of 0x3D00)
//   NEAR_LO   =  114.0 = 16'sh7200  (121 * 0.95 ~ 114.75, floor to 114)
//   NEAR_HI   = -114.0 = 16'sh8E00
//   (near_limit asserted when |data_in| >= 114, i.e. within 7 of the ±121 limit)
//
// Verilog-2005, `default_nettype none.  Synthesizable.
//==============================================================================
module guard_bands (
    input  wire         clk,
    input  wire         rst,
    input  wire [15:0]  data_in,
    output reg  [15:0]  data_out,
    output reg          saturated,
    output reg          near_limit
);

    // Guard band limits in Q8.8
    localparam signed [15:0] GUARD_MAX = 16'sh3D00;  // +121.0
    localparam signed [15:0] GUARD_MIN = 16'shC300;  // -121.0

    // Near-limit thresholds (within 5% of the guard band -> ~114.0 Q8.8)
    localparam signed [15:0] NEAR_HI  = 16'sh7200;   // +114.0
    localparam signed [15:0] NEAR_LO  = 16'sh8E00;   // -114.0

    wire signed [15:0] din_s = data_in;

    // Combinational saturation decision
    reg        [15:0]  sat_val;
    reg                sat_flag;

    always @(*) begin
        if (din_s > GUARD_MAX) begin
            sat_val  = GUARD_MAX;
            sat_flag = 1'b1;
        end else if (din_s < GUARD_MIN) begin
            sat_val  = GUARD_MIN;
            sat_flag = 1'b1;
        end else begin
            sat_val  = data_in;
            sat_flag = 1'b0;
        end
    end

    // Near-limit flag: magnitude within the outer 5% headroom band
    reg near_flag;
    always @(*) begin
        if (din_s >= NEAR_HI || din_s <= NEAR_LO) begin
            near_flag = 1'b1;
        end else begin
            near_flag = 1'b0;
        end
    end

    // Registered outputs
    always @(posedge clk) begin
        if (rst) begin
            data_out   <= 16'b0;
            saturated  <= 1'b0;
            near_limit <= 1'b0;
        end else begin
            data_out   <= sat_val;
            saturated  <= sat_flag;
            near_limit <= near_flag;
        end
    end

endmodule

`default_nettype wire