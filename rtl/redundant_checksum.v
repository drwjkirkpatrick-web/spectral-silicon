`default_nettype none
//==============================================================================
// redundant_checksum.v — Redundant Compute Checksum for Error Detection
//==============================================================================
// Computes a running 16-bit checksum over spectral multiply outputs to detect
// computation errors via redundant recomputation.  For each valid input pair
// (real, imaginary) the checksum is updated:
//
//   checksum <= (checksum + (data_re XOR data_im)) mod 2^16
//
// The XOR of the two halves is a cheap mixing function that catches bit-level
// corruption in either the real or imaginary stream.  The 16-bit accumulator
// truncates naturally (mod 2^16), so no extra modulo logic is needed.
//
// After the IFFT completes, the host loads expected_checksum and asserts
// valid_in for the comparison (or the controller pulses valid_in on the last
// word).  The module compares the accumulated checksum against
// expected_checksum and drives `mismatch` for one cycle when they differ,
// allowing the host to retry the token.
//
// Interface:
//   clk              — clock
//   rst              — active-high synchronous reset
//   data_in_re[15:0] — real part of spectral multiply output (Q8.8)
//   data_in_im[15:0] — imaginary part of spectral multiply output (Q8.8)
//   valid_in         — input data valid (also used as compare strobe)
//   checksum_out[15:0]  — current running checksum
//   checksum_valid      — high while the checksum is being accumulated
//   expected_checksum[15:0] — host-supplied expected value
//   mismatch            — pulse: checksum != expected on compare cycle
//
// Verilog-2005, `default_nettype none.  Synthesizable.
//==============================================================================
module redundant_checksum (
    input  wire        clk,
    input  wire        rst,
    input  wire [15:0] data_in_re,
    input  wire [15:0] data_in_im,
    input  wire        valid_in,
    output reg  [15:0] checksum_out,
    output reg         checksum_valid,
    input  wire [15:0] expected_checksum,
    output reg         mismatch
);

    // Mixing term: XOR of real and imaginary halves
    wire [15:0] mix = data_in_re ^ data_in_im;

    //----------------------------------------------------------------------
    // Running 16-bit checksum accumulator.  On each valid input the mixed
    // value is added into the accumulator; the 16-bit register truncates
    // naturally so no explicit modulo is required.
    //----------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst) begin
            checksum_out   <= 16'h0000;
            checksum_valid <= 1'b0;
            mismatch       <= 1'b0;
        end else begin
            if (valid_in) begin
                checksum_out   <= checksum_out + mix;
                checksum_valid <= 1'b1;
                // Compare accumulated (pre-update) value against expected.
                // Using the value as-of this cycle gives the host a single
                // compare point at the last valid word of the pass.
                mismatch       <= (checksum_out != expected_checksum) ? 1'b1 : 1'b0;
            end else begin
                checksum_valid <= 1'b0;
                mismatch        <= 1'b0;
            end
        end
    end

endmodule

`default_nettype wire