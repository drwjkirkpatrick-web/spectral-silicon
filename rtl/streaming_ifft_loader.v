`default_nettype none
//==============================================================================
// streaming_ifft_loader.v — Streaming IFFT Input Loader with Ping/Ppong Buffer
//==============================================================================
// Performance improvement: Overlaps IFFT input loading with spectral-multiply
// output streaming, hiding most of the IFFT load latency behind the spectral
// multiply phase.
//
// BACKGROUND
//   The baseline spectral_mixer pipeline is strictly sequential:
//     M_LOAD_FFT → M_FFT_RUN → M_SM_IFFT → M_MR_STORE
//   During M_SM_IFFT the spectral multiply (SM) produces 256 modes one per
//   cycle, and only then does the IFFT begin consuming them.  The IFFT itself
//   must load all 256 input samples before it can start computing, so the
//   SM+IFFT phase costs:
//     SM_output (256 cycles) + IFFT_load (256 cycles) = 512 cycles
//   plus the IFFT compute latency (~256 cycles), totalling ~768 cycles for the
//   SM+IFFT portion.
//
//   However, spectral truncation means only the first k=32 modes carry
//   non-zero data; modes 32..255 are zeroed by the spectral multiply.
//   The IFFT only needs those 32 non-zero modes to begin.  This module
//   collects the first n_modes (=32 by default) SM outputs into a small
//   ping/pong buffer, then immediately starts feeding them to the IFFT
//   while the remaining 224 (zero) modes stream into the second buffer
//   and are discarded.
//
//   With this overlap the SM+IFFT phase shrinks to roughly:
//     n_modes (32) + IFFT_compute (256) + drain ≈ 540 cycles
//   versus 512 + 256 = 768 cycles baseline — a ~30% latency reduction for
//   the SM+IFFT phase, and a proportionally smaller end-to-end improvement
//   for the full FFT→SM→IFFT→modReLU pipeline.
//
// DUAL-BUFFER (PING/PONG) ARCHITECTURE
//   Two small RAMs, each 32 entries × 32-bit complex (re[15:0] + im[15:0]):
//     • Buffer A (ping): collects the first n_modes SM outputs.
//     • Buffer B (pong): collects modes n_modes..255 (all zero, discarded).
//   Once Buffer A is full (n_modes entries), the controller:
//     1. Starts streaming Buffer A contents to the IFFT (ifft_data_valid).
//     2. Simultaneously routes incoming SM output to Buffer B (discarded).
//   This allows the IFFT to begin its input load immediately, overlapping
//   with the tail of the SM output stream.
//
// INTERFACE
//   clk, rst_n              — clock and active-low reset
//   start                   — pulse to arm the loader (begin collecting)
//   n_modes[7:0]            — config: modes to buffer before starting IFFT
//                            (default 32; 1..256)
//   sm_data_valid           — input: spectral multiply output valid
//   sm_data_re[15:0]        — input: spectral multiply real part (Q8.8)
//   sm_data_im[15:0]        — input: spectral multiply imag part (Q8.8)
//   sm_data_ready           — output: loader ready to accept SM data
//   ifft_data_valid         — output: data valid to IFFT
//   ifft_data_re[15:0]      — output: real part to IFFT (Q8.8)
//   ifft_data_im[15:0]      — output: imag part to IFFT (Q8.8)
//   ifft_data_ready         — input: IFFT ready to accept data
//   done                    — output: all n_modes forwarded to IFFT, idle
//
// Verilog-2005, `default_nettype none.  Synthesizable.
//==============================================================================
module streaming_ifft_loader #(
    parameter WIDTH   = 16,   // Q8.8 data width
    parameter FRAC    = 8,    // Fractional bits
    parameter N_TOTAL = 256,  // Total FFT/IFFT size
    parameter BUF_DEPTH = 32  // Per-buffer depth (== default n_modes)
) (
    input  wire                    clk,
    input  wire                    rst_n,

    // Configuration
    input  wire [7:0]              n_modes,    // Modes to buffer before IFFT start
    input  wire                    start,      // Arm / begin collecting

    // Spectral multiply input interface (streaming)
    input  wire                    sm_data_valid,
    input  wire signed [WIDTH-1:0] sm_data_re,
    input  wire signed [WIDTH-1:0] sm_data_im,
    output reg                     sm_data_ready,

    // IFFT output interface (streaming)
    output reg                     ifft_data_valid,
    output reg  signed [WIDTH-1:0] ifft_data_re,
    output reg  signed [WIDTH-1:0] ifft_data_im,
    input  wire                    ifft_data_ready,

    // Status
    output reg                     done
);

    //----------------------------------------------------------------------
    // Local parameters
    //----------------------------------------------------------------------
    // Address width for buffer (log2(BUF_DEPTH)); BUF_DEPTH=32 → 5 bits
    localparam BUF_AW = 5;

    // State machine
    localparam S_IDLE     = 3'd0,  // Waiting for start
               S_COLLECT  = 3'd1,  // Collecting first n_modes into Buffer A
               S_FEED     = 3'd2,  // Stream Buffer A to IFFT, discard tail to B
               S_DRAIN    = 3'd3,  // Finish streaming last modes to IFFT
               S_FINISH   = 3'd4;  // Done, return to idle

    //----------------------------------------------------------------------
    // Dual-port RAM buffers: 32 entries × {re[15:0], im[15:0]}
    // Buffer A: collects first n_modes, then read by IFFT
    // Buffer B: collects tail (zero modes), always discarded
    //----------------------------------------------------------------------
    reg signed [WIDTH-1:0] buf_a_re [0:BUF_DEPTH-1];
    reg signed [WIDTH-1:0] buf_a_im [0:BUF_DEPTH-1];
    reg signed [WIDTH-1:0] buf_b_re [0:BUF_DEPTH-1];
    reg signed [WIDTH-1:0] buf_b_im [0:BUF_DEPTH-1];

    //----------------------------------------------------------------------
    // State registers
    //----------------------------------------------------------------------
    reg [2:0]    state;
    reg [7:0]    sm_rcv_cnt;     // Count of SM samples received (0..255)
    reg [BUF_AW-1:0] wr_addr;     // Write address into active buffer
    reg [BUF_AW-1:0] rd_addr;    // Read address from Buffer A → IFFT
    reg [7:0]    ifft_sent_cnt;  // Count of modes sent to IFFT

    // Latched n_modes (captured at start)
    reg [7:0]    n_modes_latched;

    // Track whether we're writing to A (phase 1) or B (phase 2)
    // Phase 1: writing to A, reading nothing
    // Phase 2: writing to B (discard), reading from A → IFFT
    reg         writing_buf_b;   // 0 = writing A, 1 = writing B

    //----------------------------------------------------------------------
    // Combinational: effective n_modes (use parameter default if n_modes=0)
    //----------------------------------------------------------------------
    wire [7:0] effective_n = (n_modes == 8'd0) ? 8'd32 : n_modes;

    //----------------------------------------------------------------------
    // Main state machine
    //----------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= S_IDLE;
            sm_data_ready   <= 1'b0;
            ifft_data_valid <= 1'b0;
            ifft_data_re    <= 0;
            ifft_data_im    <= 0;
            done            <= 1'b0;
            sm_rcv_cnt      <= 8'd0;
            wr_addr         <= {BUF_AW{1'b0}};
            rd_addr         <= {BUF_AW{1'b0}};
            ifft_sent_cnt   <= 8'd0;
            n_modes_latched <= 8'd32;
            writing_buf_b   <= 1'b0;
        end else begin
            done <= 1'b0;  // Default pulse

            case (state)

            //--------------------------------------------------------------
            // IDLE: wait for start signal
            //--------------------------------------------------------------
            S_IDLE: begin
                sm_data_ready   <= 1'b0;
                ifft_data_valid <= 1'b0;
                if (start) begin
                    state           <= S_COLLECT;
                    sm_data_ready   <= 1'b1;   // Ready to accept SM data
                    sm_rcv_cnt      <= 8'd0;
                    wr_addr         <= {BUF_AW{1'b0}};
                    rd_addr         <= {BUF_AW{1'b0}};
                    ifft_sent_cnt   <= 8'd0;
                    n_modes_latched <= effective_n;
                    writing_buf_b   <= 1'b0;
                    done            <= 1'b0;
                end
            end

            //--------------------------------------------------------------
            // COLLECT: buffer first n_modes into Buffer A
            //--------------------------------------------------------------
            S_COLLECT: begin
                sm_data_ready <= 1'b1;  // Always ready to accept
                if (sm_data_valid && sm_data_ready) begin
                    // Write to Buffer A
                    buf_a_re[wr_addr] <= sm_data_re;
                    buf_a_im[wr_addr] <= sm_data_im;
                    wr_addr    <= wr_addr + 1'b1;
                    sm_rcv_cnt <= sm_rcv_cnt + 8'd1;

                    // Check if we've collected n_modes
                    if (sm_rcv_cnt == (n_modes_latched - 8'd1)) begin
                        // Buffer A is full; switch to feeding IFFT
                        // while remaining modes go to Buffer B
                        writing_buf_b <= 1'b1;
                        wr_addr       <= {BUF_AW{1'b0}};  // Reset for Buffer B
                        state         <= S_FEED;
                        // Start reading from Buffer A (address 0 next cycle)
                        rd_addr       <= {BUF_AW{1'b0}};
                    end
                end
            end

            //--------------------------------------------------------------
            // FEED: stream Buffer A → IFFT, simultaneously collect tail → B
            // (Buffer B contents are discarded since tail modes are zero)
            //--------------------------------------------------------------
            S_FEED: begin
                // Continue accepting SM output into Buffer B (discarded)
                sm_data_ready <= 1'b1;

                // Write tail modes to Buffer B (circular, discarded)
                if (sm_data_valid && sm_data_ready) begin
                    buf_b_re[wr_addr] <= sm_data_re;
                    buf_b_im[wr_addr] <= sm_data_im;
                    wr_addr    <= wr_addr + 1'b1;
                    sm_rcv_cnt <= sm_rcv_cnt + 8'd1;

                    // When all N_TOTAL modes received, stop accepting
                    if (sm_rcv_cnt == 8'd255) begin
                        sm_data_ready <= 1'b0;
                    end
                end

                // Read from Buffer A and stream to IFFT
                // Buffer A has n_modes entries; we send exactly n_modes
                if (ifft_sent_cnt < n_modes_latched) begin
                    ifft_data_valid <= 1'b1;
                    ifft_data_re    <= buf_a_re[rd_addr];
                    ifft_data_im    <= buf_a_im[rd_addr];

                    if (ifft_data_valid && ifft_data_ready) begin
                        rd_addr       <= rd_addr + 1'b1;
                        ifft_sent_cnt <= ifft_sent_cnt + 8'd1;
                    end
                end else begin
                    // All n_modes sent; check if SM stream is done
                    ifft_data_valid <= 1'b0;
                    if (sm_rcv_cnt >= 8'd255 || !sm_data_ready) begin
                        state <= S_DRAIN;
                    end
                end
            end

            //--------------------------------------------------------------
            // DRAIN: ensure IFFT has consumed all forwarded modes
            // (In case IFFT was slow to accept during S_FEED)
            //--------------------------------------------------------------
            S_DRAIN: begin
                sm_data_ready   <= 1'b0;
                ifft_data_valid <= 1'b0;

                if (ifft_sent_cnt >= n_modes_latched) begin
                    state <= S_FINISH;
                end
            end

            //--------------------------------------------------------------
            // FINISH: assert done, return to idle
            //--------------------------------------------------------------
            S_FINISH: begin
                done  <= 1'b1;
                state <= S_IDLE;
            end

            default: state <= S_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire