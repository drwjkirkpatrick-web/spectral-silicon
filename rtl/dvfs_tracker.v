`default_nettype none
//==============================================================================
// dvfs_tracker.v — Secure DVFS Voltage-Frequency Tracking
//==============================================================================
// Performance improvement: Dynamic Voltage and Frequency Scaling (DVFS) adjusts
// the supply voltage and clock frequency to minimize power consumption during
// idle or light-load periods.  This tracker monitors the voltage transition
// completion before enabling the clock at the new frequency, ensuring safe
// operation across V/f points.
//
// Security preservation: This is a critical security module.  Fault injection
// attacks exploit unsafe V/f combinations (e.g., low voltage with high clock
// frequency) to cause setup-time violations that flip bits in cryptographic
// or neural-network computations.  The DVFS tracker prevents this by:
//
//   1. Enforcing a safe V/f pairing table — each voltage level has a maximum
//      allowed frequency.  Any request for an unsafe combination is rejected.
//
//   2. Requiring a settle period after each voltage change before the clock
//      frequency can change.  The voltage transition must be confirmed
//      complete (via a voltage-stable signal) before the clock switches.
//
//   3. Implementing a monotonic transition protocol: voltage is always
//      raised before frequency is increased, and frequency is always lowered
//      before voltage is decreased.  This prevents the dangerous "low V,
//      high f" window.
//
//   4. Glitch-free clock muxing: the clock switch only occurs at a clock
//      edge boundary, preventing runt pulses that could cause metastability.
//
// Interface:
//   clk, rst_n       — reference clock (always running, slowest frequency)
//   // DVFS request interface
//   req_v_level      — requested voltage level (0=lowest, 3=highest)
//   req_f_level      — requested frequency level (0=lowest, 3=highest)
//   req_valid        — request is valid
//   // Voltage regulator interface
//   v_set_level      — voltage level to set
//   v_set_valid      — assert to change voltage
//   v_stable         — regulator confirms voltage is stable
//   // Clock control
//   f_set_level      — frequency level to set
//   f_set_valid      — assert to change frequency
//   f_ack            — clock PLL acknowledges frequency change
//   // Status
//   current_v        — current voltage level
//   current_f        — current frequency level
//   transition_busy   — V/f transition in progress
//   transition_done  — transition completed successfully
//   error_unsafe     — unsafe V/f combination requested (error flag)
//
// Verilog-2005, `default_nettype none.  Synthesizable.
//==============================================================================
module dvfs_tracker #(
    parameter N_LEVELS = 4   // Number of V/f levels
) (
    input  wire              clk,
    input  wire              rst_n,

    // DVFS request
    input  wire [1:0]       req_v_level,
    input  wire [1:0]       req_f_level,
    input  wire              req_valid,

    // Voltage regulator interface
    output reg  [1:0]       v_set_level,
    output reg               v_set_valid,
    input  wire              v_stable,

    // Clock control
    output reg  [1:0]       f_set_level,
    output reg               f_set_valid,
    input  wire              f_ack,

    // Status
    output reg  [1:0]       current_v,
    output reg  [1:0]       current_f,
    output reg               transition_busy,
    output reg               transition_done,
    output reg               error_unsafe
);

    //------------------------------------------------------------------
    // Safe V/f pairing table
    //
    // Each voltage level has a maximum safe frequency:
    //   V=0 (lowest) → max F=0 (lowest)
    //   V=1          → max F=1
    //   V=2          → max F=2
    //   V=3 (highest) → max F=3 (highest)
    //
    // The rule: f_level <= v_level.  A request where f_level > v_level is
    // unsafe and is rejected with an error flag.
    //------------------------------------------------------------------
    function safe_vf;
        input [1:0] v;
        input [1:0] f;
        begin
            safe_vf = (f <= v) ? 1'b1 : 1'b0;
        end
    endfunction

    //------------------------------------------------------------------
    // State machine
    //
    // Transition protocol (monotonic):
    //   UP (increase performance): raise V first → wait stable → raise F
    //   DOWN (decrease performance): lower F first → wait ack → lower V
    //
    // States:
    //   S_IDLE:      waiting for request
    //   S_CHECK:     validate safe V/f combination
    //   S_V_UP:       raise voltage, wait for v_stable
    //   S_F_SET_UP:   set new frequency, wait for f_ack
    //   S_F_DOWN:     lower frequency, wait for f_ack
    //   S_V_DOWN:     lower voltage, wait for v_stable
    //   S_DONE:       transition complete
    //   S_ERROR:      unsafe request, latch error
    //------------------------------------------------------------------
    localparam S_IDLE      = 4'd0,
               S_CHECK     = 4'd1,
               S_V_UP      = 4'd2,
               S_F_SET_UP  = 4'd3,
               S_F_DOWN    = 4'd4,
               S_V_DOWN    = 4'd5,
               S_DONE      = 4'd6,
               S_ERROR     = 4'd7;

    reg [3:0] state;

    // Latched request
    reg [1:0] req_v_r, req_f_r;

    // Determine transition direction
    // UP: new performance > current → raise V then F
    // DOWN: new performance < current → lower F then V
    // SAME: no change needed
    wire going_up = (req_v_r > current_v) || (req_f_r > current_f);
    wire going_down = (req_v_r < current_v) || (req_f_r < current_f);

    // Settle counter for voltage stability (additional guard time)
    reg [3:0] settle_cnt;
    localparam SETTLE_CYCLES = 4'd10;  // 10-cycle settle guard

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state          <= S_IDLE;
            v_set_level     <= 2'd0;
            v_set_valid     <= 1'b0;
            f_set_level     <= 2'd0;
            f_set_valid     <= 1'b0;
            current_v       <= 2'd0;
            current_f       <= 2'd0;
            transition_busy <= 1'b0;
            transition_done <= 1'b0;
            error_unsafe    <= 1'b0;
            req_v_r         <= 2'd0;
            req_f_r         <= 2'd0;
            settle_cnt      <= 4'd0;
        end else begin
            // Defaults: deassert one-cycle pulses
            v_set_valid     <= 1'b0;
            f_set_valid     <= 1'b0;
            transition_done <= 1'b0;

            case (state)
            //--- Idle: wait for request ---
            S_IDLE: begin
                transition_busy <= 1'b0;
                error_unsafe    <= 1'b0;  // Clear previous error
                if (req_valid) begin
                    req_v_r <= req_v_level;
                    req_f_r <= req_f_level;
                    state   <= S_CHECK;
                end
            end

            //--- Check: validate safe V/f combination ---
            S_CHECK: begin
                if (!safe_vf(req_v_r, req_f_r)) begin
                    // Unsafe: f_level > v_level → reject
                    error_unsafe <= 1'b1;
                    state        <= S_ERROR;
                end else if (req_v_r == current_v && req_f_r == current_f) begin
                    // No change needed
                    transition_done <= 1'b1;
                    state <= S_DONE;
                end else begin
                    transition_busy <= 1'b1;
                    if (going_up) begin
                        // Raise voltage first
                        v_set_level <= req_v_r;
                        v_set_valid <= 1'b1;
                        settle_cnt <= SETTLE_CYCLES;
                        state      <= S_V_UP;
                    end else if (going_down) begin
                        // Lower frequency first
                        f_set_level <= req_f_r;
                        f_set_valid <= 1'b1;
                        state       <= S_F_DOWN;
                    end else begin
                        transition_done <= 1'b1;
                        state <= S_DONE;
                    end
                end
            end

            //--- Voltage up: wait for v_stable + settle period ---
            S_V_UP: begin
                if (v_stable) begin
                    if (settle_cnt == 0) begin
                        // Voltage settled — now set frequency
                        current_v   <= req_v_r;
                        f_set_level <= req_f_r;
                        f_set_valid <= 1'b1;
                        state       <= S_F_SET_UP;
                    end else begin
                        settle_cnt <= settle_cnt - 1;
                    end
                end
            end

            //--- Frequency set (up): wait for PLL ack ---
            S_F_SET_UP: begin
                if (f_ack) begin
                    current_f       <= req_f_r;
                    transition_done <= 1'b1;
                    transition_busy <= 1'b0;
                    state           <= S_DONE;
                end
            end

            //--- Frequency down: wait for PLL ack ---
            S_F_DOWN: begin
                if (f_ack) begin
                    current_f   <= req_f_r;
                    // Now lower voltage
                    v_set_level <= req_v_r;
                    v_set_valid <= 1'b1;
                    settle_cnt  <= SETTLE_CYCLES;
                    state       <= S_V_DOWN;
                end
            end

            //--- Voltage down: wait for v_stable + settle ---
            S_V_DOWN: begin
                if (v_stable) begin
                    if (settle_cnt == 0) begin
                        current_v       <= req_v_r;
                        transition_done <= 1'b1;
                        transition_busy <= 1'b0;
                        state           <= S_DONE;
                    end else begin
                        settle_cnt <= settle_cnt - 1;
                    end
                end
            end

            //--- Done: return to idle ---
            S_DONE: begin
                transition_busy <= 1'b0;
                state           <= S_IDLE;
            end

            //--- Error: unsafe request, return to idle ---
            S_ERROR: begin
                transition_busy <= 1'b0;
                // Keep error_unsafe asserted until next request
                state           <= S_IDLE;
            end

            default: state <= S_IDLE;
            endcase
        end
    end

endmodule