// operand_isolation.v
// Clock gating / operand isolation for inactive pipeline stages.
// When en=0, outputs forced to 0 (isolated) and internal registers clock-gated.
// When en=1, data passes through with a register stage.
// Uses latch-based clock-enable gate to prevent wasted switching power.
// Verilog-2005 (iverilog -g2005).
module operand_isolation #(
    parameter WIDTH = 16
) (
    input  wire                  clk,
    input  wire                  rst,
    input  wire                  en,
    input  wire [WIDTH-1:0]      data_in,
    output wire [WIDTH-1:0]      data_out
);

    // Latch-based clock-enable gate.
    // When clk is low, the enable latch is transparent and captures en.
    // When clk is high, the latch holds its value -> gated_clk only toggles
    // when en was high at the clock edge, preventing wasted switching.
    reg clk_en;
    always @(*) begin
        if (!clk)
            clk_en = en;  // transparent on low phase
        // else hold (implicit latch)
    end

    wire gated_clk;
    assign gated_clk = clk & clk_en;

    // Registered datapath, only updates when enabled (clock-gated).
    reg [WIDTH-1:0] data_reg;
    always @(posedge gated_clk or posedge rst) begin
        if (rst)
            data_reg <= {WIDTH{1'b0}};
        else if (en)
            data_reg <= data_in;
    end

    // Operand isolation: zero output when disabled.
    assign data_out = en ? data_reg : {WIDTH{1'b0}};

endmodule