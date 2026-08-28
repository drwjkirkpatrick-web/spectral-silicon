`default_nettype none
//==============================================================================
// clock_tree_optimizer.v — Clock Tree Synthesis Helper Module
//==============================================================================
// Produces a low-skew clock distribution network by chaining BUF_STAGES clock
// buffers with balanced fanout.  The output clk_out is clk_in delayed through
// the buffer chain.
//
// In synthesis the buffer chain is replaced by the tool's clock-tree
// buffers; in simulation the delay is purely functional (zero-delay buf
// primitives keep the model race-free).
//
// Parameter BUF_STAGES defaults to 8.
//
// Verilog-2005, `default_nettype none.  Synthesizable.
//==============================================================================
module clock_tree_optimizer #(
    parameter BUF_STAGES = 8
) (
    input  wire clk_in,
    output wire clk_out,
    input  wire rst
);

    //----------------------------------------------------------------------
    // Internal buffer-chain nets.  buf_chain[0] is driven by clk_in;
    // buf_chain[BUF_STAGES] drives clk_out.
    //----------------------------------------------------------------------
    wire clk_buf [0:BUF_STAGES];

    assign clk_buf[0] = clk_in;

    //----------------------------------------------------------------------
    // Buffer chain: BUF_STAGES buf primitives in series.
    // Each buf drives one load (the next buf) — balanced fanout of 1.
    //----------------------------------------------------------------------
    genvar gi;
    generate
        for (gi = 0; gi < BUF_STAGES; gi = gi + 1) begin : buf_gen
            buf (clk_buf[gi+1], clk_buf[gi]);
        end
    endgenerate

    assign clk_out = clk_buf[BUF_STAGES];

    // rst is accepted for interface completeness; the buffer chain has no
    // state to reset.  Tie it off to prevent lint warnings about an unused
    // input when `default_nettype none is active.
    wire unused_rst = rst;

endmodule

`default_nettype wire