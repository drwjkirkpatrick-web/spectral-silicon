`default_nettype none
//==============================================================================
// twiddle_bank_r8.v — Twiddle Factor Bank for Radix-8 Butterfly
//==============================================================================
// Provides 7 twiddle factors simultaneously for the radix-8 butterfly.
// W0=1 (trivial: 0x0100, 0x0000) and W4=-j (trivial: 0x0000, 0xFF00).
// The other 5 are non-trivial, stored in ROM indexed by stage+group+index.
//
// Parameterized for N=256, 4 radix-8 stages (2 r8 + 1 r4 tail handled externally).
//==============================================================================
module twiddle_bank_r8 #(
    parameter WIDTH = 16,
    parameter FRAC  = 8,
    parameter N     = 256
) (
    input  wire                    clk,
    input  wire                    rst,
    input  wire [2:0]              stage,       // FFT stage (0..3)
    input  wire [4:0]              group,       // butterfly group index
    input  wire [2:0]              index,       // twiddle index (0..6)
    output wire signed [WIDTH-1:0] w0_re, w0_im,  // W^0 = 1
    output wire signed [WIDTH-1:0] w1_re, w1_im,  // W^1
    output wire signed [WIDTH-1:0] w2_re, w2_im,  // W^2
    output wire signed [WIDTH-1:0] w3_re, w3_im,  // W^3
    output wire signed [WIDTH-1:0] w4_re, w4_im,  // W^4 = -j
    output wire signed [WIDTH-1:0] w5_re, w5_im,  // W^5
    output wire signed [WIDTH-1:0] w6_re, w6_im   // W^6
);

    // Trivial twiddles: W0 = 1+0j, W4 = 0-1j
    localparam W0_RE = {1'b0, {(WIDTH-1){1'b0}}} + (1 << FRAC);  // 1.0 in Q8.8
    localparam W0_IM = {WIDTH{1'b0}};
    localparam W4_RE = {WIDTH{1'b0}};
    localparam W4_IM = ~({1'b0, {(WIDTH-1){1'b0}}} + (1 << FRAC)) + 1'b1;  // -1.0

    // ROM for non-trivial twiddles (W1, W2, W3, W5, W6)
    // 5 twiddles × 32 groups × 4 stages = 640 entries
    reg signed [WIDTH-1:0] rom_re [0:639];
    reg signed [WIDTH-1:0] rom_im [0:639];

    // ROM address: stage * 160 + group * 5 + twiddle_index
    wire [9:0] rom_addr;
    assign rom_addr = stage * 160 + group * 5 + index;

    // Initialize ROM with twiddle factors
    // W256^k = cos(-2*pi*k/256) + j*sin(-2*pi*k/256)
    integer i;
    initial begin
        for (i = 0; i < 640; i = i + 1) begin
            rom_re[i] = 16'h0100;  // placeholder: 1.0
            rom_im[i] = 16'h0000;
        end
    end

    // Trivial outputs (combinational — constants)
    assign w0_re = 16'h0100;  // 1.0 in Q8.8
    assign w0_im = 16'h0000;  // 0.0
    assign w4_re = 16'h0000;  // 0.0
    assign w4_im = 16'hFF00;  // -1.0 in Q8.8 (two's complement)

    // Non-trivial outputs (registered read)
    reg signed [WIDTH-1:0] w1_re_r, w1_im_r;
    reg signed [WIDTH-1:0] w2_re_r, w2_im_r;
    reg signed [WIDTH-1:0] w3_re_r, w3_im_r;
    reg signed [WIDTH-1:0] w5_re_r, w5_im_r;
    reg signed [WIDTH-1:0] w6_re_r, w6_im_r;

    always @(posedge clk) begin
        if (rst) begin
            w1_re_r <= 0; w1_im_r <= 0;
            w2_re_r <= 0; w2_im_r <= 0;
            w3_re_r <= 0; w3_im_r <= 0;
            w5_re_r <= 0; w5_im_r <= 0;
            w6_re_r <= 0; w6_im_r <= 0;
        end else begin
            w1_re_r <= rom_re[rom_addr]; w1_im_r <= rom_im[rom_addr];
            w2_re_r <= rom_re[rom_addr + 1]; w2_im_r <= rom_im[rom_addr + 1];
            w3_re_r <= rom_re[rom_addr + 2]; w3_im_r <= rom_im[rom_addr + 2];
            w5_re_r <= rom_re[rom_addr + 3]; w5_im_r <= rom_im[rom_addr + 3];
            w6_re_r <= rom_re[rom_addr + 4]; w6_im_r <= rom_im[rom_addr + 4];
        end
    end

    assign w1_re = w1_re_r; assign w1_im = w1_im_r;
    assign w2_re = w2_re_r; assign w2_im = w2_im_r;
    assign w3_re = w3_re_r; assign w3_im = w3_im_r;
    assign w5_re = w5_re_r; assign w5_im = w5_im_r;
    assign w6_re = w6_re_r; assign w6_im = w6_im_r;

endmodule

`default_nettype wire