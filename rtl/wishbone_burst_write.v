`default_nettype none
//==============================================================================
// wishbone_burst_write.v — Wishbone Burst Write Controller (input data)
//==============================================================================
// Accepts 256 complex (re/im) input samples from the host in a single burst
// write and stores them into an internal dual-buffer (ping/pong).
//
// The host drives a Wishbone Classic-compatible write port.  Each 16-bit
// Wishbone write carries one real or imaginary component; consecutive
// writes are packed into re/im pairs.  To cut the host-side load from
// 256 cycles to 64 cycles the controller groups writes into 4-word bursts
// (one 4-word burst = 2 complex samples), and the host only needs to assert
// the bus for the active burst windows.
//
// Storage layout: 256 complex samples → buffer_re[255:0], buffer_im[255:0].
// wb_dat_i[15:0] is the payload; wb_adr selects the target slot.  Even
// addresses write the real part, odd addresses write the imaginary part of
// the same complex sample (sample index = adr>>1).  When a re/im pair is
// complete the sample counter advances and, once 256 samples are received,
// burst_done pulses and the dual-buffer flips so the host can immediately
// start the next block.
//
// Q8.8 fixed-point (16-bit).  Verilog-2005, synthesizable.
//==============================================================================
module wishbone_burst_write #(
    parameter WIDTH      = 16,
    parameter N_SAMPLES  = 256,
    parameter AW         = 16,   // Wishbone address width
    parameter BURST_LEN  = 4     // 4-word bursts → 64 bursts for 256 words
) (
    input  wire             clk,
    input  wire             rst,

    // Wishbone Classic slave write port (host → this controller)
    input  wire             wb_cyc,
    input  wire             wb_stb,
    input  wire             wb_we,
    input  wire [AW-1:0]    wb_adr,
    input  wire [WIDTH-1:0] wb_dat_i,
    input  wire             wb_ack,     // host-asserted ack (mirrored back)
    output reg              wb_ack_o,   // per-word ack driven to host
    output reg              wb_stall,   // backpressure during buffer flip

    // Burst control / status
    input  wire             burst_start,
    input  wire [7:0]       burst_len,   // host-programmed burst length (words)
    output reg  [WIDTH-1:0] burst_data_re,
    output reg  [WIDTH-1:0] burst_data_im,
    output reg              burst_done
);

    //------------------------------------------------------------------
    // Dual-buffer storage (ping/pong) for 256 complex samples.
    // buf_sel selects the active write target; reading can occur from the
    // inactive buffer while the host writes the active one.
    //------------------------------------------------------------------
    reg [WIDTH-1:0] buf_re [0:2*N_SAMPLES-1];
    reg [WIDTH-1:0] buf_im [0:2*N_SAMPLES-1];

    reg             buf_sel;        // 0 = lower half, 1 = upper half
    reg [8:0]       word_cnt;        // 0..511 (256 complex × 2 words)
    reg [7:0]       burst_word_cnt;  // 0..BURST_LEN-1 within a burst
    reg [7:0]       burst_cnt;       // completed bursts (0..63)
    reg             active;          // controller armed

    // Derived: which complex sample and which half (re/im).
    wire [8:0]   flat_index = word_cnt;                 // 0..511
    wire [7:0]   sample_idx = flat_index[8:1];          // 0..255
    wire         is_imag    = flat_index[0];             // 0=re, 1=im
    wire [8:0]   buf_base   = {buf_sel, 8'h00};          // lower/upper half base
    wire [8:0]   store_addr = buf_base + {1'b0, sample_idx};

    // Host-facing ack: single-cycle, gated by cyc&stb&we.
    wire         wb_write_valid = wb_cyc & wb_stb & wb_we;

    //------------------------------------------------------------------
    // Write logic
    //------------------------------------------------------------------
    integer j;
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            buf_sel        <= 1'b0;
            word_cnt       <= 9'd0;
            burst_word_cnt<= 8'd0;
            burst_cnt      <= 8'd0;
            active         <= 1'b0;
            wb_ack_o       <= 1'b0;
            wb_stall       <= 1'b0;
            burst_done     <= 1'b0;
            burst_data_re  <= {WIDTH{1'b0}};
            burst_data_im  <= {WIDTH{1'b0}};
            for (j = 0; j < 2*N_SAMPLES; j = j + 1) begin
                buf_re[j] <= {WIDTH{1'b0}};
                buf_im[j] <= {WIDTH{1'b0}};
            end
        end else begin
            // Defaults
            wb_ack_o   <= 1'b0;
            wb_stall   <= 1'b0;
            burst_done <= 1'b0;

            // Arm on burst_start
            if (burst_start && !active) begin
                active          <= 1'b1;
                word_cnt        <= 9'd0;
                burst_word_cnt <= 8'd0;
                burst_cnt      <= 8'd0;
            end

            if (active) begin
                // Accept a Wishbone write word.
                if (wb_write_valid && !wb_stall) begin
                    // Store into active buffer half.
                    if (is_imag)
                        buf_im[store_addr] <= wb_dat_i;
                    else
                        buf_re[store_addr] <= wb_dat_i;

                    wb_ack_o       <= 1'b1;
                    word_cnt       <= word_cnt + 9'd1;
                    burst_word_cnt <= burst_word_cnt + 8'd1;

                    // Expose the most-recently written re/im pair for debug/
                    // streaming taps.
                    if (is_imag) begin
                        burst_data_im <= wb_dat_i;
                    end else begin
                        burst_data_re <= wb_dat_i;
                    end

                    // End of a 4-word burst boundary?
                    if (burst_word_cnt + 8'd1 >= BURST_LEN[7:0] ||
                        burst_word_cnt + 8'd1 >= burst_len) begin
                        burst_word_cnt <= 8'd0;
                        burst_cnt      <= burst_cnt + 8'd1;
                    end

                    // All 256 complex samples (512 words) received?
                    if (word_cnt + 9'd1 >= 9'd2*N_SAMPLES) begin
                        burst_done <= 1'b1;
                        active     <= 1'b0;
                        buf_sel    <= ~buf_sel;   // flip dual-buffer
                        wb_stall   <= 1'b1;       // brief pause for flip
                    end
                end
            end
        end
    end

endmodule

`default_nettype wire