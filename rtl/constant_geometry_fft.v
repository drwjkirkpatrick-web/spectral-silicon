`default_nettype none
//==============================================================================
// constant_geometry_fft.v — Constant-Geometry FFT Addressing Module
//==============================================================================
// Generates butterfly addressing for a constant-geometry FFT where the
// butterfly connections are the same for every stage (no bit-reversal needed).
//
// For N=256, RADIX=4: 4 stages, each with N/RADIX = 64 butterflies.
// The addressing is stage-independent: the same butterfly index maps to the
// same pair of memory positions in every stage, simplifying interconnect.
//
// Given a butterfly_idx and stage, produces read/write addresses for the
// two operands (a, b) of a radix-4 butterfly. For RADIX=4, 4 reads and 4
// writes occur; we expose addr_a and addr_b as representative operand
// addresses (the full set is derived by striding).
//
// Constant-geometry: addr_a = butterfly_idx, addr_b = butterfly_idx XOR
// stride_pattern. The stride pattern is constant (stage-independent) so the
// interconnect is fixed.
//
// Verilog-2005. Parameter N=256, RADIX=4.
//==============================================================================
module constant_geometry_fft #(
    parameter N     = 256,
    parameter RADIX = 4
) (
    input  wire             clk,
    input  wire             rst,
    input  wire             start,
    input  wire [2:0]       stage,        // stage index (0..log4(N)-1)
    input  wire [5:0]       butterfly_idx, // butterfly within stage (0..63)
    input  wire signed [15:0] data_re, data_im,  // data sample
    output reg  [7:0]       rd_addr_a,     // read address, operand a
    output reg  [7:0]       rd_addr_b,     // read address, operand b
    output reg  [7:0]       wr_addr_a,     // write address, operand a
    output reg  [7:0]       wr_addr_b,     // write address, operand b
    output reg              data_valid     // data output is valid
);

    // log4(N) = log2(N)/2 stages. For N=256: log4(256) = 4 stages.
    localparam STAGES = 4;     // ceil(log2(N) / log2(RADIX))
    localparam LOGN   = 8;      // log2(N)

    // For constant geometry with RADIX=4, each butterfly reads from positions
    //   group = butterfly_idx / (N/RADIX/?) ...
    // The constant-geometry address for a radix-4 butterfly:
    //   Let stride = N / RADIX = 64 (constant across stages for const-geometry).
    //   operand index within butterfly: k = 0..3, addr = base + k*stride
    //   base = butterfly_idx mod stride, group = butterfly_idx / stride
    //   For constant geometry, the butterfly reads:
    //     addr_a = base + group * stride * RADIX  ... but simplified:
    //   The pair (a, b) for a radix-4 butterfly:
    //     a_addr = butterfly_idx
    //     b_addr = {butterfly_idx[1:0], butterfly_idx[5:2]}  (digit-reversed)
    //   Constant geometry keeps the same physical read/write pattern every
    //   stage, so addresses depend only on butterfly_idx, not stage.

    // Constant-geometry address generation (stage-independent)
    //   stride = N / RADIX = 64
    //   a = butterfly_idx
    //   b = butterfly_idx XOR stride_mask  (constant-geometry: fixed partners)
    // For a 256-point radix-4 constant-geometry FFT, the butterfly partner
    // index is derived by bit-reversing within the stage group. Because it's
    // constant geometry, the pattern is fixed.
    wire [7:0] addr_a = {2'b0, butterfly_idx};
    wire [7:0] addr_b = {2'b0, butterfly_idx} ^ 8'h40;  // XOR with stride=64

    // Pipeline register (1 cycle) to register addresses and valid
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            rd_addr_a <= 8'h00;
            rd_addr_b <= 8'h00;
            wr_addr_a <= 8'h00;
            wr_addr_b <= 8'h00;
            data_valid <= 1'b0;
        end else begin
            rd_addr_a <= addr_a;
            rd_addr_b <= addr_b;
            wr_addr_a <= addr_a;  // in-place: write back to same address
            wr_addr_b <= addr_b;
            data_valid <= start;
        end
    end

endmodule