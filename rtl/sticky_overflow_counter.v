`default_nettype none
//==============================================================================
// sticky_overflow_counter.v — Sticky Overflow Event Counter
//==============================================================================
// Counts overflow events across all FFT stages and spectral multiplies within
// a single inference pass (one token).  The counter is "sticky" — it only
// clears on an explicit `clear` strobe from the host at the start of a new
// inference, not automatically per cycle.  The host reads `count` after each
// token to assess numerical stability.
//
// If the running count exceeds a programmable threshold (default 16), the
// `threshold_exceeded` flag is asserted and held until the next clear.  This
// lets the host adjust the BFP (block floating-point) range or reduce the
// active mode count k to keep the datapath within safe numeric bounds.
//
// Interface:
//   clk              — clock
//   rst              — active-high synchronous reset
//   overflow_event   — pulse: an overflow was detected this cycle
//   clear            — pulse: start of a new inference pass (resets count)
//   count[15:0]      — running 16-bit event count (host-readable)
//   threshold_exceeded — sticky flag, high when count > THRESHOLD
//
// Parameter THRESHOLD — default 16.
//
// Verilog-2005, `default_nettype none.  Synthesizable.
//==============================================================================
module sticky_overflow_counter #(
    parameter THRESHOLD = 16
) (
    input  wire        clk,
    input  wire        rst,
    input  wire        overflow_event,
    input  wire        clear,
    output reg  [15:0] count,
    output reg         threshold_exceeded
);

    //----------------------------------------------------------------------
    // Sticky counter: increments on each overflow_event; clears on `clear`
    // (new inference pass) or reset.  Count saturates at 16'hFFFF to avoid
    // rollover that would mask a persistent overflow condition.
    //----------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst || clear) begin
            count             <= 16'h0000;
            threshold_exceeded<= 1'b0;
        end else begin
            if (overflow_event && (count != 16'hFFFF)) begin
                count <= count + 16'h0001;
            end
            // Sticky threshold flag: set once count exceeds THRESHOLD and
            // held until clear/reset (not de-asserted if count later holds).
            if (count >= THRESHOLD) begin
                threshold_exceeded <= 1'b1;
            end
        end
    end

endmodule

`default_nettype wire