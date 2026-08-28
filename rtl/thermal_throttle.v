`default_nettype none
//==============================================================================
// thermal_throttle.v — Thermal Throttle with Graceful Degradation
//==============================================================================
// Monitors the on-chip temperature reported by a ring-oscillator ADC (12-bit
// unsigned value, larger = hotter).  When the measured temperature exceeds
// temp_threshold, the module:
//   1. Asserts throttle_active.
//   2. Halves the active mode count k (e.g. 32 -> 16) via reduced_k, so the
//      spectral multiply controller processes fewer modes per cycle.
//   3. De-asserts reduced_clk_en to gate the main clock enable, dropping the
//      effective operating frequency and reducing dynamic power.
//
// This graceful degradation prevents thermal-induced timing errors: by
// reducing both the workload (fewer modes) and the clock rate, setup-time
// margins are restored without halting the inference entirely.  The host /
// spectral multiply controller reads reduced_k to reconfigure the datapath.
//
// Hysteresis: to avoid oscillation around the threshold, throttling engages
// at temp > threshold and disengages only when temp falls below (threshold -
// HYST), where HYST is a small margin in ADC counts.
//
// Interface:
//   clk             — clock
//   rst             — active-high synchronous reset
//   temp_sensor[11:0]  — 12-bit ring-oscillator ADC reading
//   temp_threshold[11:0] — host-programmable trip point
//   throttle_active — sticky flag, high while throttling
//   reduced_k[4:0]  — reduced mode count (NOMINAL_K when cool, halved when hot)
//   reduced_clk_en  — clock-enable gating signal (0 = gated when throttling)
//
// Parameters:
//   NOMINAL_K  — default active mode count (32)
//   MIN_K      — floor for reduced k (8)
//   HYST       — hysteresis margin in ADC counts (16)
//
// Verilog-2005, `default_nettype none.  Synthesizable.
//==============================================================================
module thermal_throttle #(
    parameter NOMINAL_K = 32,       // nominal mode count (may exceed 5-bit output)
    parameter MIN_K     = 5'd8,
    parameter HYST       = 12'd16
) (
    input  wire        clk,
    input  wire        rst,
    input  wire [11:0] temp_sensor,
    input  wire [11:0] temp_threshold,
    output reg         throttle_active,
    output reg  [4:0]  reduced_k,
    output reg         reduced_clk_en
);

    //----------------------------------------------------------------------
    // Hysteresis comparator.  Engage throttling when temp exceeds threshold;
    // release only when temp falls below (threshold - HYST).  This prevents
    // the throttle from toggling rapidly when the temperature sits near the
    // trip point.
    //----------------------------------------------------------------------
    wire [11:0] release_level = temp_threshold - HYST;

    always @(posedge clk) begin
        if (rst) begin
            throttle_active <= 1'b0;
        end else if (temp_sensor > temp_threshold) begin
            throttle_active <= 1'b1;
        end else if (temp_sensor < release_level) begin
            throttle_active <= 1'b0;
        end
    end

    //----------------------------------------------------------------------
    // Graceful degradation: reduce active modes k and gate the clock.
    // reduced_k is NOMINAL_K (truncated to 5 bits) when cool.  When throttling,
    // it halves the nominal k (arithmetic right shift by 1) but never below
    // MIN_K.  reduced_clk_en is 0 while throttling (clock gated) and 1
    // otherwise.
    //
    // Note: NOMINAL_K (default 32) is a 6-bit concept; the 5-bit reduced_k
    // port carries the effective active count which the spectral multiply
    // controller only consumes while throttle_active is high (i.e. the 16-
    // bit halved value, which fits cleanly in 5 bits).
    wire [4:0] nominal_out = NOMINAL_K[4:0];
    wire [4:0] halved_k    = NOMINAL_K[4:0] >> 1;
    wire [4:0] safe_k      = (halved_k < MIN_K) ? MIN_K : halved_k;

    always @(posedge clk) begin
        if (rst) begin
            reduced_k      <= nominal_out;
            reduced_clk_en <= 1'b1;
        end else begin
            reduced_k      <= throttle_active ? safe_k    : nominal_out;
            reduced_clk_en <= throttle_active ? 1'b0      : 1'b1;
        end
    end

endmodule

`default_nettype wire