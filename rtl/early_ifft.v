`default_nettype none
//==============================================================================
// early_ifft.v — Early IFFT Start with Overlap Controller
//==============================================================================
// Performance improvement: Overlaps IFFT execution with the tail end of FFT
// mode computation.  Instead of waiting for all N spectral modes to be
// computed before starting the IFFT, the IFFT begins as soon as mode k is
// ready, overlapping with the remaining N-k modes.  This reduces total
// pipeline latency by approximately k cycles (e.g., ~12.5% for k=32 of 256).
//
// The controller monitors the FFT mode production counter.  When mode k is
// produced, it asserts `ifft_start_ready` to begin loading the IFFT input
// buffer.  Subsequent modes are written to the IFFT buffer as they become
// available, overlapping with the IFFT's initial loading phase.
//
// Security preservation: the overlap point (mode k) is a fixed configuration
// value, not data-dependent.  The IFFT always starts at the same mode count,
// regardless of input data.  This prevents timing-based information leakage
// through variable overlap amounts.
//
// Interface:
//   clk, rst_n       — clock and reset
//   fft_mode_cnt     — current FFT mode being produced (0..N-1)
//   fft_mode_valid   — FFT mode output is valid
//   fft_done         — all FFT modes complete
//   k_threshold      — mode count at which to start IFFT (configurable)
//   ifft_start       — assert to start IFFT (when mode k is ready)
//   ifft_busy        — IFFT is running
//   overlap_active    — IFFT and FFT are running simultaneously
//   all_done          — both FFT and IFFT complete
//
// Verilog-2005, `default_nettype none.  Synthesizable.
//==============================================================================
module early_ifft #(
    parameter WIDTH = 16,
    parameter N     = 256
) (
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire [7:0]              fft_mode_cnt,
    input  wire                    fft_mode_valid,
    input  wire                    fft_done,
    input  wire [7:0]              k_threshold,   // Start IFFT when this mode is ready
    output reg                     ifft_start,
    output wire                    ifft_busy,
    output reg                     overlap_active,
    output reg                     all_done,
    input  wire                    ifft_done      // IFFT completion signal (feedback)
);

    // State machine
    localparam O_IDLE      = 3'd0,
               O_FFT_RUN   = 3'd1,  // FFT running, waiting for mode k
               O_OVERLAP   = 3'd2,  // FFT tail + IFFT running
               O_IFFT_ONLY = 3'd3,  // FFT done, IFFT still running
               O_DONE      = 3'd4;

    reg [2:0] o_state;

    // Track whether IFFT has been started
    reg ifft_started;

    assign ifft_busy = ifft_started && !ifft_done;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            o_state        <= O_IDLE;
            ifft_start     <= 1'b0;
            overlap_active <= 1'b0;
            all_done        <= 1'b0;
            ifft_started    <= 1'b0;
        end else begin
            ifft_start <= 1'b0;  // Default: pulse
            all_done    <= 1'b0;  // Default

            case (o_state)
            //--- Idle ---
            O_IDLE: begin
                if (fft_mode_valid && fft_mode_cnt == 0) begin
                    // FFT has started producing modes
                    o_state <= O_FFT_RUN;
                end
            end

            //--- FFT running: wait for mode k to be ready ---
            O_FFT_RUN: begin
                if (fft_mode_valid && fft_mode_cnt >= k_threshold) begin
                    // Mode k is ready — start IFFT
                    ifft_start  <= 1'b1;
                    ifft_started <= 1'b1;
                    overlap_active <= 1'b1;
                    o_state <= O_OVERLAP;
                end
            end

            //--- Overlap: FFT tail and IFFT running simultaneously ---
            O_OVERLAP: begin
                if (fft_done) begin
                    // FFT complete, IFFT may still be running
                    overlap_active <= 1'b0;
                    o_state <= O_IFFT_ONLY;
                end
                if (ifft_done) begin
                    // IFFT finished (unlikely before FFT, but handle it)
                    ifft_started <= 1'b0;
                    if (fft_done) begin
                        all_done <= 1'b1;
                        o_state  <= O_DONE;
                    end
                end
            end

            //--- IFFT only: FFT done, waiting for IFFT ---
            O_IFFT_ONLY: begin
                overlap_active <= 1'b0;
                if (ifft_done) begin
                    ifft_started <= 1'b0;
                    all_done <= 1'b1;
                    o_state  <= O_DONE;
                end
            end

            //--- Done ---
            O_DONE: begin
                all_done <= 1'b0;
                o_state  <= O_IDLE;
            end

            default: o_state <= O_IDLE;
            endcase
        end
    end

endmodule