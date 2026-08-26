`default_nettype none
//==============================================================================
// twiddle_rom.v — Twiddle Factor ROM for N=256 FFT
//==============================================================================
// Stores cos/sin twiddle factors for a 256-point FFT in Q8.8 fixed-point.
//
//   addr[7:0]  → index k = 0..255
//   cos_out    = round( cos(2*pi*k/256) * 256 )  (Q8.8, signed 16-bit)
//   sin_out    = round( sin(2*pi*k/256) * 256 )
//
// The ROM is initialized via $readmemh from two hex files (one for cos, one
// for sin), each containing 256 lines of 16-bit hex values.  The file paths
// are parameterized so synthesis and simulation can point to the correct
// location.
//
// For a radix-4 FFT of N=256, twiddle factors are needed at various strides.
// The full 256-entry table covers all stages — just index differently.
//
// Prompt 13 specification.
//==============================================================================
module twiddle_rom #(
    parameter N         = 256,                       // FFT size
    parameter WIDTH     = 16,                         // Data width (Q8.8)
    parameter ADDR_BITS = 8,                          // Address bits: log2(256)=8
    parameter COS_FILE  = "twiddle_data/twiddle_cos_256.hex",
    parameter SIN_FILE  = "twiddle_data/twiddle_sin_256.hex"
) (
    input  wire                    clk,
    input  wire [ADDR_BITS-1:0]    addr,              // Index 0..N-1
    output reg  signed [WIDTH-1:0] cos_out,           // cos(2*pi*k/N) in Q8.8
    output reg  signed [WIDTH-1:0] sin_out            // sin(2*pi*k/N) in Q8.8
);

    // ROM arrays: cos and sin tables, each N entries of WIDTH-bit signed values.
    // Initialized at elaboration time via $readmemh.
    reg signed [WIDTH-1:0] cos_mem [0:N-1];
    reg signed [WIDTH-1:0] sin_mem [0:N-1];

    initial begin
        $readmemh(COS_FILE, cos_mem);
        $readmemh(SIN_FILE, sin_mem);
    end

    // Registered read (1-cycle latency) — improves timing for synthesis.
    always @(posedge clk) begin
        cos_out <= cos_mem[addr];
        sin_out <= sin_mem[addr];
    end

endmodule