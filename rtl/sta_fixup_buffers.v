// sta_fixup_buffers.v
// Static timing analysis buffer insertion on long interconnect paths.
// Inserts N_BUFFERS register stages, each adding 1 cycle of latency
// to break long combinational paths (fixes the 5 longest STA paths).
// Verilog-2005 (iverilog -g2005).
module sta_fixup_buffers #(
    parameter N_BUFFERS = 2
) (
    input  wire        clk,
    input  wire        rst,
    input  wire [15:0] data_in,
    output wire [15:0] data_out
);

    // Pipeline register chain of N_BUFFERS stages.
    reg [15:0] buf_reg [0:N_BUFFERS-1];
    integer i;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            for (i = 0; i < N_BUFFERS; i = i + 1)
                buf_reg[i] <= 16'b0;
        end else begin
            buf_reg[0] <= data_in;
            for (i = 1; i < N_BUFFERS; i = i + 1)
                buf_reg[i] <= buf_reg[i-1];
        end
    end

    // Output from the final buffer stage.
    assign data_out = buf_reg[N_BUFFERS-1];

endmodule