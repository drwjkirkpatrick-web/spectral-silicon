// wider_datapath.v
// Wider 18-bit datapath (Q12.4 hybrid) for FFT intermediate stages.
// mode_sel=0: Q8.8 passthrough (zero-extend 16-bit -> 18-bit).
// mode_sel=1: Q12.4 wide — sign-extend 16-bit Q8.8 to 18-bit Q12.4.
//   Q8.8  = 8 int bits, 8 frac bits
//   Q12.4 = 12 int bits, 4 frac bits
//   To convert Q8.8 -> Q12.4: keep the 12 MSBs (8 int + top 4 frac),
//   truncating the bottom 4 frac bits, then sign-extend the integer
//   field by 4 bits. Net: take [15:4] (12 bits) and sign-extend to 18.
// Verilog-2005 (iverilog -g2005).
module wider_datapath (
    input  wire        clk,
    input  wire        rst,
    input  wire        mode_sel,       // 0=Q8.8 passthrough, 1=Q12.4 wide
    input  wire [15:0] data_in,
    output reg  [17:0] data_out
);

    // Combinational format conversion.
    wire [11:0] q12_4_core;       // [15:4] of Q8.8 input
    wire [17:0] q12_4_ext;        // sign-extended to 18 bits
    wire [17:0] q8_8_ext;         // zero-extended to 18 bits

    assign q12_4_core = data_in[15:4];
    assign q12_4_ext  = {{6{q12_4_core[11]}}, q12_4_core};
    assign q8_8_ext   = {2'b00, data_in};

    wire [17:0] selected;
    assign selected = mode_sel ? q12_4_ext : q8_8_ext;

    // Registered output.
    always @(posedge clk or posedge rst) begin
        if (rst)
            data_out <= 18'b0;
        else
            data_out <= selected;
    end

endmodule