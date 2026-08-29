`default_nettype none
//==============================================================================
// timing_margin_buffer.v — Timing Margin Buffer for 120 MHz Operation
//==============================================================================
// Inserts pipeline registers on long interconnect paths between the bus
// interface and the FFT engine. At 120 MHz (8.3 ns cycle), the cross-die
// route from wishbone_b4 to spectral_mixer can exceed the timing budget.
//
// This module adds 1-2 pipeline stages on the control/data path to
// break long routes, with a simple valid/ready handshake.
//
// Latency: 2 cycles (configurable via STAGES parameter)
// Throughput: 1 word per cycle
//==============================================================================
module timing_margin_buffer #(
    parameter WIDTH  = 32,
    parameter STAGES = 2
) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  in_valid,
    input  wire [WIDTH-1:0]      in_data,
    output wire                  in_ready,
    output reg                   out_valid,
    output reg  [WIDTH-1:0]      out_data,
    input  wire                  out_ready
);

    // Pipeline registers
    reg [WIDTH-1:0] pipe_data [0:STAGES-1];
    reg             pipe_valid [0:STAGES-1];
    reg             stalled;

    // in_ready: accept when not stalled
    assign in_ready = !stalled;

    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < STAGES; i = i + 1) begin
                pipe_data[i]  <= {WIDTH{1'b0}};
                pipe_valid[i] <= 1'b0;
            end
            out_valid <= 1'b0;
            out_data  <= {WIDTH{1'b0}};
            stalled   <= 1'b0;
        end else begin
            // Shift pipeline when not stalled by downstream
            if (out_ready || !out_valid) begin
                stalled <= 1'b0;

                // Stage 0: input
                if (in_valid && in_ready) begin
                    pipe_data[0]  <= in_data;
                    pipe_valid[0] <= 1'b1;
                end else begin
                    pipe_valid[0] <= 1'b0;
                end

                // Middle stages
                for (i = 1; i < STAGES; i = i + 1) begin
                    pipe_data[i]  <= pipe_data[i-1];
                    pipe_valid[i] <= pipe_valid[i-1];
                end

                // Output
                if (STAGES > 0) begin
                    out_data  <= pipe_data[STAGES-1];
                    out_valid <= pipe_valid[STAGES-1];
                end
            end else begin
                // Downstream not ready — stall the pipeline
                stalled <= 1'b1;
            end
        end
    end

endmodule

`default_nettype wire