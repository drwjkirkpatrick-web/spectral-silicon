`default_nettype none
//==============================================================================
// retimed_pipeline_regs.v — Retimed Pipeline Registers for Butterfly Critical Path
//==============================================================================
// Inserts balanced register stages along the radix-4 butterfly critical path
// to break up combinational delay and enable a higher clock frequency.
//
// The module is a parameterised shift register of STAGES D flip-flops.  Each
// stage registers its input on the rising clock edge and resets to zero.  The
// total latency through the module is STAGES clock cycles; throughput is one
// sample per cycle (1-cycle throughput).
//
// Parameter STAGES defaults to 3, matching the typical radix-4 butterfly
// split (DFT kernel → twiddle multiply → rescale/saturate).
//
// Verilog-2005, `default_nettype none.  Synthesizable.
//==============================================================================
module retimed_pipeline_regs #(
    parameter WIDTH  = 16,
    parameter STAGES = 3
) (
    input  wire                 clk,
    input  wire                 rst,
    input  wire [WIDTH-1:0]     data_in,
    output wire [WIDTH-1:0]     data_out
);

    //----------------------------------------------------------------------
    // Pipeline register array: pipe_r[0] is stage 1, pipe_r[STAGES-1] is last
    //----------------------------------------------------------------------
    reg [WIDTH-1:0] pipe_r [0:STAGES-1];

    integer k;

    always @(posedge clk) begin
        if (rst) begin
            for (k = 0; k < STAGES; k = k + 1) begin
                pipe_r[k] <= {WIDTH{1'b0}};
            end
        end else begin
            // Stage 0 captures the input
            pipe_r[0] <= data_in;
            // Subsequent stages shift forward
            for (k = 1; k < STAGES; k = k + 1) begin
                pipe_r[k] <= pipe_r[k-1];
            end
        end
    end

    // Output from the final stage
    assign data_out = pipe_r[STAGES-1];

endmodule

`default_nettype wire