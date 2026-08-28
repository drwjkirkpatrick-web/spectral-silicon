`default_nettype none
//==============================================================================
// multiport_weight_sram.v — Dual-Port Weight SRAM Macro
//==============================================================================
// 128-entry dual-port weight SRAM for spectral weight storage.  Port 1 reads
// the current mode weight while port 2 simultaneously prefetches the next
// mode weight, enabling zero-bubble spectral multiply with streaming mode
// processing.
//
// Architecture:
//   • Single write port (wr_addr, wr_data, wr_en) for weight loading.
//   • Two independent read ports (rd1_addr→rd1_data, rd2_addr→rd2_data).
//   • Single-cycle read latency (registered output for timing closure).
//   • 128 entries × 16-bit (Q8.8 fixed-point).
//
// The two read ports operate independently: rd1 can read address A while rd2
// reads address B in the same cycle.  This enables the spectral multiply unit
// to fetch weight[k] and weight[k+1] in the same cycle, overlapping the
// current-mode multiply with next-mode weight prefetch.
//
// Security preservation: both read ports have constant 1-cycle latency
// regardless of address or data.  No data-dependent access patterns.
//
// Interface:
//   clk, rst         — clock and active-high reset
//   wr_addr[6:0]     — write address (0..127)
//   wr_data[15:0]    — write data (Q8.8)
//   wr_en            — write enable
//   rd1_addr[6:0]    — read port 1 address (current mode weight)
//   rd1_data[15:0]   — read port 1 data (1-cycle latency)
//   rd2_addr[6:0]    — read port 2 address (prefetch next mode weight)
//   rd2_data[15:0]   — read port 2 data (1-cycle latency)
//
// Q8.8 fixed-point, 16-bit total, 8-bit fraction.
// Verilog-2005, `default_nettype none.  Synthesizable.
//==============================================================================
module multiport_weight_sram #(
    parameter DEPTH = 128,
    parameter AW    = 7,         // log2(128) = 7
    parameter WIDTH = 16
) (
    input  wire                    clk,
    input  wire                    rst,

    // Write port
    input  wire [AW-1:0]          wr_addr,
    input  wire signed [WIDTH-1:0] wr_data,
    input  wire                    wr_en,

    // Read port 1 (current mode weight)
    input  wire [AW-1:0]          rd1_addr,
    output reg  signed [WIDTH-1:0] rd1_data,

    // Read port 2 (prefetch next mode weight)
    input  wire [AW-1:0]          rd2_addr,
    output reg  signed [WIDTH-1:0] rd2_data
);

    //----------------------------------------------------------------------
    // SRAM array: 128 entries × 16-bit signed (Q8.8)
    //----------------------------------------------------------------------
    reg signed [WIDTH-1:0] mem [0:DEPTH-1];

    integer i;
    initial begin
        for (i = 0; i < DEPTH; i = i + 1) begin
            mem[i] = {WIDTH{1'b0}};
        end
    end

    //----------------------------------------------------------------------
    // Write logic (synchronous)
    //----------------------------------------------------------------------
    always @(posedge clk) begin
        if (wr_en) begin
            mem[wr_addr] <= wr_data;
        end
    end

    //----------------------------------------------------------------------
    // Read port 1: registered output, single-cycle latency
    //----------------------------------------------------------------------
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            rd1_data <= {WIDTH{1'b0}};
        end else begin
            rd1_data <= mem[rd1_addr];
        end
    end

    //----------------------------------------------------------------------
    // Read port 2: registered output, single-cycle latency
    //----------------------------------------------------------------------
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            rd2_data <= {WIDTH{1'b0}};
        end else begin
            rd2_data <= mem[rd2_addr];
        end
    end

endmodule