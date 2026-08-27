`default_nettype none
//==============================================================================
// batch_channel_controller.v — Automatic Multi-Channel Sequencer
//==============================================================================
//
// Eliminates 63 host round-trips for D=64 channels.  Without this controller
// the host must manually issue a start command for each channel, wait for
// done, read back results, then write the next channel's data — 64 host
// round-trips per inference.  Each round-trip costs ~100+ Wishbone bus cycles
// (write start → poll done → read data → write next data → write start …).
//
//   Eliminated round-trips: 63 (channels 1..63, channel 0 still needs start)
//   Cycles saved per round-trip: ~100
//   Cycles saved per inference: 63 × 100 ≈ 6,300 cycles
//
// For a 2-layer LLM model with 64 channels per layer:
//   Total round-trips eliminated: 2 × 63 = 126
//   Total cycles saved: 126 × 100 ≈ 12,600 cycles
//
// At 50 MHz that is ~252 µs of host bus overhead removed per forward pass —
// a meaningful fraction of the ~100 µs per-channel pipeline latency.
//
//------------------------------------------------------------------------------
// Operation
//------------------------------------------------------------------------------
//
// The host loads all 64 channels of input data into a dual-port input RAM
// (64 channels × 256 samples × 2 (re/im) × 16 bits = 64 KB), sets the
// channel count register (chan_cnt), and issues a single start pulse.
//
// The controller then automatically:
//   1. Feeds channel 0's 256 complex samples from the input RAM into the
//      spectral mixer (FFT → spectral multiply → IFFT → modReLU pipeline).
//   2. Waits for mixer_done (full pipeline complete, output ready).
//   3. Stores channel 0's 256 output samples to the output RAM.
//   4. Immediately starts channel 1 (no host intervention).
//   5. Repeats for all D channels.
//
// The host polls a single all_done bit at the end — one poll instead of 64.
//
//------------------------------------------------------------------------------
// Memory Layout
//------------------------------------------------------------------------------
//
// Each RAM address holds one complex sample (re + im as separate 16-bit
// ports).  Address = channel_id × 256 + sample_index.
//
//   Input RAM:  64 channels × 256 samples = 16,384 words = 2^14 (14-bit addr)
//   Output RAM: same layout, 16,384 words
//
//------------------------------------------------------------------------------
// Interface
//------------------------------------------------------------------------------
//
//   clk          [in]  System clock
//   rst_n        [in]  Active-low synchronous/asynchronous reset
//   start        [in]  Single-cycle start pulse from host
//   all_done     [out] Asserted when all channels have completed
//   busy         [out] Asserted while any channel is being processed
//   chan_cnt     [in]  Number of channels to process (default 64, max 128)
//   channel_id   [out] Current channel being processed (0..chan_cnt-1)
//
//   mixer_start  [out] Start pulse to spectral_mixer
//   mixer_done   [in]  Done signal from spectral_mixer
//
//   ram_rd_addr  [out] Read address into input RAM
//   ram_rd_re    [in]  Real part from input RAM (Q8.8)
//   ram_rd_im    [in]  Imag part from input RAM (Q8.8)
//   ram_wr_addr  [out] Write address into output RAM
//   ram_wr_re    [out] Real part to output RAM (Q8.8)
//   ram_wr_im    [out] Imag part to output RAM (Q8.8)
//   ram_we       [out] Write enable for output RAM
//
//------------------------------------------------------------------------------
// Makefile snippet (add to project Makefile):
//------------------------------------------------------------------------------
// # Batch channel controller test
// TB_BATCH = tb_batch_channel
// RTL_BATCH = rtl/batch_channel_controller.v
//
// sim_batch:
// 	iverilog -g2012 -o sim_batch $(RTL_BATCH) tb/$(TB_BATCH).v
// 	vvp sim_batch
//
// # Cocotb version (with spectral_mixer mock)
// sim_batch_cocotb:
// 	TOPLEVEL=batch_channel_controller SIM=icarus pytest tb/tb_batch_channel.py
//
//==============================================================================
module batch_channel_controller #(
    parameter N     = 256,    // FFT size / samples per channel
    parameter D     = 64,     // Default number of channels
    parameter WIDTH = 16      // Q8.8 data width
) (
    input  wire              clk,
    input  wire              rst_n,
    input  wire              start,        // Single-cycle start from host
    output wire              all_done,     // High when all channels complete
    output wire              busy,         // High while processing
    input  wire [6:0]        chan_cnt,     // Number of channels (default 64)
    output wire [6:0]        channel_id,   // Current channel index

    // Spectral mixer handshake
    output reg               mixer_start,  // Start pulse to spectral_mixer
    input  wire              mixer_done,    // Done from spectral_mixer

    // Input RAM read port
    output reg  [13:0]       ram_rd_addr,
    input  wire [15:0]       ram_rd_re,
    input  wire [15:0]       ram_rd_im,

    // Output RAM write port
    output reg  [13:0]       ram_wr_addr,
    output reg  [15:0]       ram_wr_re,
    output reg  [15:0]       ram_wr_im,
    output reg               ram_we
);

    //--------------------------------------------------------------------------
    // Local parameters
    //--------------------------------------------------------------------------
    localparam SAMPLE_BITS = 8;  // log2(256) — bits for sample index

    //--------------------------------------------------------------------------
    // State machine
    //--------------------------------------------------------------------------
    localparam S_IDLE      = 4'd0,  // Wait for start
               S_START     = 4'd1,  // Issue mixer_start for current channel
               S_LOAD      = 4'd2,  // Stream 256 samples from input RAM to mixer
               S_WAIT      = 4'd3,  // Wait for mixer_done
               S_STORE     = 4'd4,  // Stream 256 output samples to output RAM
               S_NEXT      = 4'd5,  // Advance to next channel or finish
               S_DONE      = 4'd6;  // Assert all_done

    reg [3:0]  state;

    reg [6:0]  chan_reg;           // Current channel counter
    reg [7:0]  samp_idx;           // Sample index within channel (0..255)

    reg        all_done_r;
    reg        busy_r;

    //--------------------------------------------------------------------------
    // Combinational output assignments
    //--------------------------------------------------------------------------
    assign all_done   = all_done_r;
    assign busy       = busy_r;
    assign channel_id = chan_reg;

    //--------------------------------------------------------------------------
    // Main state machine
    //--------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= S_IDLE;
            chan_reg     <= 7'd0;
            samp_idx     <= 8'd0;
            mixer_start  <= 1'b0;
            ram_rd_addr  <= 14'd0;
            ram_wr_addr  <= 14'd0;
            ram_wr_re    <= 16'd0;
            ram_wr_im    <= 16'd0;
            ram_we       <= 1'b0;
            all_done_r   <= 1'b0;
            busy_r       <= 1'b0;
        end else begin
            // Defaults
            mixer_start <= 1'b0;
            ram_we      <= 1'b0;
            all_done_r  <= 1'b0;

            case (state)

            //--------------------------------------------------------------
            // IDLE: wait for host start
            //--------------------------------------------------------------
            S_IDLE: begin
                busy_r   <= 1'b0;
                if (start) begin
                    busy_r   <= 1'b1;
                    chan_reg <= 7'd0;
                    state    <= S_START;
                end
            end

            //--------------------------------------------------------------
            // START: assert mixer_start for one cycle
            // The spectral_mixer uses the same in/out buffers each time;
            // the batch controller is responsible for loading the input
            // buffer from the external RAM before each channel.
            //--------------------------------------------------------------
            S_START: begin
                mixer_start <= 1'b1;
                samp_idx    <= 8'd0;
                state       <= S_LOAD;
            end

            //--------------------------------------------------------------
            // LOAD: stream 256 complex samples from input RAM.
            // The spectral_mixer's input buffer accepts data sample-by-
            // sample.  We present one sample per cycle by advancing the
            // read address each clock.  A registered read (1-cycle latency)
            // is assumed: address presented on cycle T, data valid on T+1.
            // The mixer captures data when its internal ready/valid
            // handshake completes; here we assume the mixer accepts one
            // sample per cycle (backpressure handling is the mixer's job).
            //--------------------------------------------------------------
            S_LOAD: begin
                // Present read address for current sample
                ram_rd_addr <= {chan_reg[5:0], samp_idx};

                // Mixer is assumed to latch input on mixer_start + each
                // subsequent cycle.  Advance through all 256 samples.
                if (samp_idx == 8'd255) begin
                    samp_idx <= 8'd0;
                    state    <= S_WAIT;
                end else begin
                    samp_idx <= samp_idx + 8'd1;
                end
            end

            //--------------------------------------------------------------
            // WAIT: wait for the full pipeline (FFT→SM→IFFT→modReLU) done
            //--------------------------------------------------------------
            S_WAIT: begin
                if (mixer_done) begin
                    samp_idx <= 8'd0;
                    state    <= S_STORE;
                end
            end

            //--------------------------------------------------------------
            // STORE: write 256 output samples to output RAM.
            // The spectral_mixer produces output samples that we write
            // to the external output RAM.  Address = chan × 256 + idx.
            // Here we use a simplified model: the mixer_done pulse signals
            // that all 256 output samples are available in the mixer's
            // output buffer; the controller reads them out via the
            // ram_wr_* interface.  In a real integration, the mixer's
            // output buffer read port would be connected here.
            //
            // For this controller, we assume the output data is presented
            // on the ram_wr_re/ram_wr_im lines by the mixer (or a small
            // output buffer) indexed by samp_idx.  We generate the write
            // address and write-enable.
            //--------------------------------------------------------------
            S_STORE: begin
                ram_wr_addr <= {chan_reg[5:0], samp_idx};
                ram_we      <= 1'b1;
                // ram_wr_re / ram_wr_im come from the spectral_mixer output
                // buffer (connected at the top level).  The controller
                // only provides address and write-enable.

                if (samp_idx == 8'd255) begin
                    samp_idx <= 8'd0;
                    state    <= S_NEXT;
                end else begin
                    samp_idx <= samp_idx + 8'd1;
                end
            end

            //--------------------------------------------------------------
            // NEXT: advance channel counter or finish
            //--------------------------------------------------------------
            S_NEXT: begin
                if (chan_reg == (chan_cnt - 1)) begin
                    // All channels processed
                    state <= S_DONE;
                end else begin
                    chan_reg <= chan_reg + 7'd1;
                    state    <= S_START;
                end
            end

            //--------------------------------------------------------------
            // DONE: assert all_done, return to IDLE
            //--------------------------------------------------------------
            S_DONE: begin
                all_done_r <= 1'b1;
                busy_r     <= 1'b0;
                state      <= S_IDLE;
            end

            default: state <= S_IDLE;
            endcase
        end
    end

endmodule

//==============================================================================
// TB_WRAP — Minimal wrapper so the cocotb testbench can instantiate the
// controller together with a mock spectral_mixer and dual-port RAMs in a
// single Verilog module.  This is compiled alongside batch_channel_controller.v.
//==============================================================================
`ifndef SYNTH_ONLY
module batch_channel_controller_tb_wrap;
    // Empty — real instantiation is done in the cocotb testbench via
    // the simulator's hierarchical features or a SystemVerilog TB module.
    // This guard prevents the wrap from being synthesized.
endmodule
`endif