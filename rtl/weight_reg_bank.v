`default_nettype none
//==============================================================================
// weight_reg_bank.v — Banked Weight Register File for Spectral Weights
//==============================================================================
// Stores 128 spectral weights organised as 4 banks of 32 entries each
// (32 modes × 4 blocks).  Each entry is a single Q8.8 word; the parent
// instantiates two banks (re / im) or packs re|im into the 16-bit word.
//
// Interface:
//   Write port:  wr_en, wr_bank[1:0], wr_addr[4:0], wr_data[15:0]
//   Read  port:  rd_bank[1:0], rd_addr[4:0] → rd_data[15:0] (single-cycle)
//
// The read port is combinational (single-cycle read from any bank).  The
// write port is synchronous and may target a different bank than the read
// port in the same cycle (dual-port operation: write one bank while reading
// another).  If the read and write target the same bank and address in the
// same cycle, the read returns the old value (read-before-write).
//
// Reset clears all entries to zero.
//
// Verilog-2005, `default_nettype none.  Synthesizable.
//==============================================================================
module weight_reg_bank #(
    parameter WIDTH    = 16,
    parameter N_BANKS  = 4,
    parameter DEPTH    = 32
) (
    input  wire             clk,
    input  wire             rst,

    // Write port
    input  wire             wr_en,
    input  wire [1:0]       wr_bank,
    input  wire [4:0]       wr_addr,
    input  wire [WIDTH-1:0] wr_data,

    // Read port (combinational, single-cycle)
    input  wire [1:0]       rd_bank,
    input  wire [4:0]       rd_addr,
    output wire [WIDTH-1:0] rd_data
);

    //----------------------------------------------------------------------
    // Storage: 4 banks × 32 entries
    //----------------------------------------------------------------------
    reg [WIDTH-1:0] mem [0:N_BANKS-1][0:DEPTH-1];

    integer bi;
    integer mi;

    //----------------------------------------------------------------------
    // Synchronous write + reset
    //----------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst) begin
            for (bi = 0; bi < N_BANKS; bi = bi + 1) begin
                for (mi = 0; mi < DEPTH; mi = mi + 1) begin
                    mem[bi][mi] <= {WIDTH{1'b0}};
                end
            end
        end else if (wr_en) begin
            mem[wr_bank][wr_addr] <= wr_data;
        end
    end

    //----------------------------------------------------------------------
    // Combinational read (single-cycle).  Read-before-write semantics are
    // preserved because the write occurs on the clock edge and the read is
    // taken from the current (pre-edge) register state.
    //----------------------------------------------------------------------
    assign rd_data = mem[rd_bank][rd_addr];

endmodule

`default_nettype wire