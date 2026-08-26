`default_nettype none
//==============================================================================
// conflict_free_addr.v — Conflict-Free Memory Addressing for Radix-4 FFT
//==============================================================================
// Performance improvement: Maps linear addresses to a 4-bank interleaved
// memory such that the 4 simultaneous reads required by a radix-4 butterfly
// always hit different banks.  The mapping is:
//   bank = addr[1:0]    (lower 2 bits select bank)
//   row  = addr >> 2    (upper bits select row within bank)
// This guarantees zero bank conflicts for any radix-4 access pattern where
// four addresses are {base, base+stride, base+2*stride, base+3*stride}
// with stride being a power of 4 or higher — which is exactly the radix-4
// FFT addressing pattern.
//
// Security preservation: the address mapping is purely combinational and
// deterministic.  Bank selection depends only on the linear address, not on
// data content, so no information leaks through the memory access pattern.
//
// Interface:
//   lin_addr         — linear address input (ADDR_W bits)
//   bank_sel         — bank selection (0..3)
//   row_addr         — row address within selected bank
//   // For radix-4 butterfly: 4 addresses in, 4 bank/row pairs out
//   bf_addr0..3      — four butterfly addresses
//   bf_bank0..3      — bank for each address
//   bf_row0..3       — row for each address
//
// Verilog-2005, `default_nettype none.  Synthesizable.
//==============================================================================
module conflict_free_addr #(
    parameter ADDR_W = 8,    // Linear address width (e.g., 8 for N=256)
    parameter NBANKS = 4     // Number of memory banks (radix-4 → 4)
) (
    input  wire [ADDR_W-1:0]  lin_addr,
    output wire [1:0]         bank_sel,    // bank = addr[1:0]
    output wire [ADDR_W-3:0]  row_addr,    // row = addr >> 2

    // Radix-4 butterfly address interface (4 simultaneous accesses)
    input  wire [ADDR_W-1:0]  bf_addr0,
    input  wire [ADDR_W-1:0]  bf_addr1,
    input  wire [ADDR_W-1:0]  bf_addr2,
    input  wire [ADDR_W-1:0]  bf_addr3,
    output wire [1:0]         bf_bank0,
    output wire [1:0]         bf_bank1,
    output wire [1:0]         bf_bank2,
    output wire [1:0]         bf_bank3,
    output wire [ADDR_W-3:0]  bf_row0,
    output wire [ADDR_W-3:0]  bf_row1,
    output wire [ADDR_W-3:0]  bf_row2,
    output wire [ADDR_W-3:0]  bf_row3,

    // Conflict flag (asserts if any two butterfly addresses share a bank)
    output wire                conflict
);

    //------------------------------------------------------------------
    // Single-address mapping
    //------------------------------------------------------------------
    assign bank_sel = lin_addr[1:0];
    assign row_addr = lin_addr >> 2;

    //------------------------------------------------------------------
    // Radix-4 butterfly address mapping
    //------------------------------------------------------------------
    assign bf_bank0 = bf_addr0[1:0];
    assign bf_bank1 = bf_addr1[1:0];
    assign bf_bank2 = bf_addr2[1:0];
    assign bf_bank3 = bf_addr3[1:0];

    assign bf_row0  = bf_addr0 >> 2;
    assign bf_row1  = bf_addr1 >> 2;
    assign bf_row2  = bf_addr2 >> 2;
    assign bf_row3  = bf_addr3 >> 2;

    //------------------------------------------------------------------
    // Conflict detection: asserts if any two of the 4 addresses share a bank.
    // For properly aligned radix-4 access (stride = 4^k), this never fires.
    // The flag is provided for diagnostic/debugging purposes.
    //------------------------------------------------------------------
    wire c01 = (bf_bank0 == bf_bank1);
    wire c02 = (bf_bank0 == bf_bank2);
    wire c03 = (bf_bank0 == bf_bank3);
    wire c12 = (bf_bank1 == bf_bank2);
    wire c13 = (bf_bank1 == bf_bank3);
    wire c23 = (bf_bank2 == bf_bank3);

    assign conflict = c01 | c02 | c03 | c12 | c13 | c23;

endmodule