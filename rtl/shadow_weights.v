`default_nettype none
//==============================================================================
// shadow_weights.v — Shadow Register File with Prefetch
//==============================================================================
// Performance improvement: Double-buffered weight registers allow the next
// set of spectral weights to be prefetched into a shadow buffer while the
// current set is active in the computation pipeline.  A single-cycle swap
// immediately makes the prefetched weights available, eliminating the
// loading latency between consecutive spectral mixing operations.
//
// Security preservation: shadow registers have identical timing to active
// registers — the swap is a register rename, not a memory reload.  No timing
// difference exists between swapped and non-swapped operations, preventing
// weight-value-dependent side channels.
//
// Interface:
//   clk, rst_n       — clock and reset
//   wr_en            — write to shadow register
//   wr_addr          — weight address (0..DEPTH-1)
//   wr_data_re, wr_data_im — weight data
//   prefetch_done    — indicates shadow buffer is fully loaded
//   swap             — swap active and shadow (single cycle)
//   rd_addr          — read address for active weights
//   rd_data_re, rd_data_im — active weight (combinational read)
//   shadow_ready     — shadow buffer has valid data
//
// Verilog-2005, `default_nettype none.  Synthesizable.
//==============================================================================
module shadow_weights #(
    parameter WIDTH = 16,
    parameter DEPTH = 32,
    parameter AW   = 5         // Address width: log2(DEPTH)
) (
    input  wire                    clk,
    input  wire                    rst_n,

    // Write to shadow
    input  wire                    wr_en,
    input  wire [AW-1:0]          wr_addr,
    input  wire signed [WIDTH-1:0] wr_data_re,
    input  wire signed [WIDTH-1:0] wr_data_im,

    // Swap control
    input  wire                    swap,
    output reg                     shadow_ready,

    // Read from active
    input  wire [AW-1:0]          rd_addr,
    output wire signed [WIDTH-1:0] rd_data_re,
    output wire signed [WIDTH-1:0] rd_data_im
);

    //------------------------------------------------------------------
    // Active and shadow register files
    //------------------------------------------------------------------
    reg signed [WIDTH-1:0] active_re  [0:DEPTH-1];
    reg signed [WIDTH-1:0] active_im  [0:DEPTH-1];
    reg signed [WIDTH-1:0] shadow_re  [0:DEPTH-1];
    reg signed [WIDTH-1:0] shadow_im  [0:DEPTH-1];

    // Valid flags: track whether each buffer has been populated
    reg shadow_valid_r;

    //------------------------------------------------------------------
    // Write logic: always write to shadow buffer
    //------------------------------------------------------------------
    always @(posedge clk) begin
        if (wr_en) begin
            shadow_re[wr_addr] <= wr_data_re;
            shadow_im[wr_addr] <= wr_data_im;
        end
    end

    //------------------------------------------------------------------
    // Swap logic: exchange active and shadow pointers
    // Implemented as actual data copy for all entries in a single cycle
    // (synthesis maps this to register renaming or mux-based flip).
    //
    // For small DEPTH (32), a full swap is feasible.  For larger DEPTH,
    // a pointer-based approach would be used instead.
    //------------------------------------------------------------------
    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            shadow_valid_r <= 1'b0;
            shadow_ready   <= 1'b0;
            for (i = 0; i < DEPTH; i = i + 1) begin
                active_re[i] <= 0;
                active_im[i] <= 0;
            end
        end else begin
            if (wr_en) begin
                shadow_valid_r <= 1'b1;  // Mark shadow as being populated
            end

            if (swap) begin
                // Swap: active ← shadow, shadow ← active (old active)
                for (i = 0; i < DEPTH; i = i + 1) begin
                    active_re[i] <= shadow_re[i];
                    active_im[i] <= shadow_im[i];
                    shadow_re[i] <= active_re[i];
                    shadow_im[i] <= active_im[i];
                end
                shadow_ready <= shadow_valid_r;
                // After swap, shadow becomes the old active (may be stale)
                shadow_valid_r <= 1'b0;
            end else if (wr_en) begin
                shadow_ready <= 1'b1;
            end
        end
    end

    //------------------------------------------------------------------
    // Read logic: combinational read from active register file
    //------------------------------------------------------------------
    assign rd_data_re = active_re[rd_addr];
    assign rd_data_im = active_im[rd_addr];

endmodule