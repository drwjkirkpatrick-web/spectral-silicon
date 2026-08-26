`default_nettype none
//==============================================================================
// pingpong_ram.v — Dual-Bank Ping-Pong Memory
//==============================================================================
// Performance improvement: Double-buffered memory allows one bank to be read
// while the other is written, enabling continuous data streaming without
// bubbles.  For FFT-based processing, the current frame is read from bank A
// while the next frame is being written to bank B.  A single-cycle bank swap
// switches the roles.  This eliminates the idle gap between frames, doubling
// throughput for streaming workloads.
//
// Security preservation: both banks are always accessible — no data-dependent
// bank selection.  The bank swap signal is control-driven, not data-driven,
// so access patterns are constant regardless of payload content.
//
// Interface:
//   clk, rst_n       — clock and reset
//   wr_en            — write enable
//   wr_addr          — write address
//   wr_data_re, wr_data_im — complex write data
//   rd_en            — read enable
//   rd_addr          — read address
//   rd_data_re, rd_data_im — complex read data (1-cycle latency)
//   bank_swap        — swap read/write banks (single cycle)
//   active_bank      — indicates which bank is being written (0=A, 1=B)
//
// Verilog-2005, `default_nettype none.  Synthesizable.
//==============================================================================
module pingpong_ram #(
    parameter WIDTH = 16,
    parameter DEPTH = 256,
    parameter AW    = 8       // Address width: log2(DEPTH)
) (
    input  wire                    clk,
    input  wire                    rst_n,

    // Write port
    input  wire                    wr_en,
    input  wire [AW-1:0]          wr_addr,
    input  wire signed [WIDTH-1:0] wr_data_re,
    input  wire signed [WIDTH-1:0] wr_data_im,

    // Read port
    input  wire                    rd_en,
    input  wire [AW-1:0]          rd_addr,
    output reg  signed [WIDTH-1:0] rd_data_re,
    output reg  signed [WIDTH-1:0] rd_data_im,

    // Bank swap
    input  wire                    bank_swap,
    output reg                     active_bank
);

    //------------------------------------------------------------------
    // Two memory banks (each stores complex samples: re + im)
    //------------------------------------------------------------------
    reg signed [WIDTH-1:0] bank_a_re [0:DEPTH-1];
    reg signed [WIDTH-1:0] bank_a_im [0:DEPTH-1];
    reg signed [WIDTH-1:0] bank_b_re [0:DEPTH-1];
    reg signed [WIDTH-1:0] bank_b_im [0:DEPTH-1];

    // active_bank: 0 = writing to A / reading from B
    //              1 = writing to B / reading from A
    // wr_bank = active_bank, rd_bank = ~active_bank

    //------------------------------------------------------------------
    // Bank swap (synchronous)
    //------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            active_bank <= 1'b0;
        else if (bank_swap)
            active_bank <= ~active_bank;
    end

    //------------------------------------------------------------------
    // Write logic: write to the active (write) bank
    //------------------------------------------------------------------
    always @(posedge clk) begin
        if (wr_en) begin
            if (active_bank == 1'b0) begin
                bank_a_re[wr_addr] <= wr_data_re;
                bank_a_im[wr_addr] <= wr_data_im;
            end else begin
                bank_b_re[wr_addr] <= wr_data_re;
                bank_b_im[wr_addr] <= wr_data_im;
            end
        end
    end

    //------------------------------------------------------------------
    // Read logic: read from the inactive (read) bank
    // Registered read with 1-cycle latency for timing closure.
    //------------------------------------------------------------------
    always @(posedge clk) begin
        if (rd_en) begin
            if (active_bank == 1'b0) begin
                // Reading from bank B
                rd_data_re <= bank_b_re[rd_addr];
                rd_data_im <= bank_b_im[rd_addr];
            end else begin
                // Reading from bank A
                rd_data_re <= bank_a_re[rd_addr];
                rd_data_im <= bank_a_im[rd_addr];
            end
        end
    end

endmodule