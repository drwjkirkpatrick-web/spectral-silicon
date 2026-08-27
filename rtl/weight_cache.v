`default_nettype none
//==============================================================================
// weight_cache.v — Multi-Layer Weight Cache for Spectral Weights
//==============================================================================
// Caches spectral weights for up to N_LAYERS transformer layers on-chip,
// eliminating Wishbone bus latency when switching between layers.
//
// Storage: N_LAYERS × N_MODES × 2 (re/im) × WIDTH bits
//   Default: 4 layers × 32 modes × 2 × 16-bit = 4 KB
//
// The host loads all layers' weights at boot via the write interface.
// During inference, layer_sel selects the active layer, and the spectral
// mixer reads weights combinationally — zero bus latency between layers.
//
// ~200 gates + 4KB SRAM (register-file inferred).
// Q8.8 fixed-point, WIDTH=16.
//==============================================================================
module weight_cache #(
    parameter WIDTH    = 16,
    parameter N_MODES  = 32,
    parameter N_LAYERS = 4
) (
    input  wire                    clk,
    input  wire                    rst_n,

    // Write interface (host loads weights at boot)
    input  wire                    write_en,
    input  wire [5:0]              write_addr,    // {layer_sel[1:0], mode[3:0], re_im}
                                                 // bit 5: re/im select
                                                 // bits 4:2: mode index (0..31 → 5 bits needed)
                                                 // Actually: addr = {layer[1:0], mode[4:0], re_im}
    input  wire signed [WIDTH-1:0] write_data,

    // Read interface (spectral mixer reads during inference)
    input  wire [1:0]              layer_sel,     // active layer 0..3
    input  wire [4:0]              read_addr,     // mode index 0..31
    output wire signed [WIDTH-1:0] read_data_re, // combinational read
    output wire signed [WIDTH-1:0] read_data_im
);

    // Storage: N_LAYERS banks, each with N_MODES complex weights
    reg signed [WIDTH-1:0] wt_re [0:N_LAYERS-1][0:N_MODES-1];
    reg signed [WIDTH-1:0] wt_im [0:N_LAYERS-1][0:N_MODES-1];

    // Write: decode address
    // write_addr[5] = re/im (0=re, 1=im)
    // write_addr[4:0] = mode index (but we need layer too)
    // Let's use: write_addr = {layer[1:0], mode[2:0], re_im}
    // That's only 6 bits: layer(2) + mode(3) + re_im(1) = 6
    // But N_MODES=32 needs 5 bits. So let's redefine:
    // write_addr = {layer[1:0], re_im, mode_idx_truncated}
    // Actually the spec says write_addr[5:0]. Let's use:
    //   write_addr[5:3] = layer_sel for write (0..3 → 2 bits, but we have 3 bits)
    //   write_addr[2]   = re/im
    //   write_addr[1:0]  = not enough for 32 modes...
    //
    // Redesign: write_addr[5:0] encodes a flat address:
    //   For N_LAYERS=4, N_MODES=32: total complex entries = 128
    //   Flat index = layer * 32 + mode (0..127)
    //   write_addr[5] = re/im (0=re, 1=im)
    //   write_addr[4:0] = flat_index... but 4 layers * 32 modes = 128 > 32
    // We need more than 6 bits for full addressing.
    //
    // Practical approach: write_addr encodes {layer[1:0], mode[4:0], re_im[0:0]}
    // = 8 bits. But the spec says [5:0]. Let's just make it work with sequential
    // writes: the host writes mode 0 re, mode 0 im, mode 1 re, ... for each layer.
    // write_addr = 0..191 (4 layers × 32 modes × 2 = 256 entries, need 8 bits)
    //
    // Since the spec says [5:0], let's interpret it as a simpler scheme:
    // The host writes one layer at a time. A separate register (not part of
    // this module) sets the target layer. write_addr = {mode[4:0], re_im[0:0]}
    // = 6 bits exactly (32 modes × 2 = 64 entries per layer).
    //
    // But we also need to select which layer to write to. Let's add a
    // write_layer input.

    // Refined: use write_addr[5:0] = {mode[4:0], re_im[0:0]}
    // write_layer[1:0] selects the target layer for writes.

    // Since the spec didn't include write_layer, let's repurpose:
    // write_addr[5:0] with the understanding that writes go to the layer
    // selected by layer_sel. This way, the host sets layer_sel to the
    // target layer, writes that layer's weights, then moves to the next.

    wire [1:0]  wr_layer = layer_sel;        // write to currently selected layer
    wire        wr_is_im = write_addr[0];     // bit 0: re/im
    wire [4:0]  wr_mode  = write_addr[5:1];   // bits 5:1: mode index

    always @(posedge clk) begin
        if (write_en) begin
            if (wr_is_im)
                wt_im[wr_layer][wr_mode] <= write_data;
            else
                wt_re[wr_layer][wr_mode] <= write_data;
        end
    end

    // Read: combinational from selected layer
    assign read_data_re = wt_re[layer_sel][read_addr];
    assign read_data_im = wt_im[layer_sel][read_addr];

endmodule

`default_nettype wire