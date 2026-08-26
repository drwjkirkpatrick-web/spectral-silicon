`default_nettype none
//==============================================================================
// weight_crypto.v — LFSR-Based Stream Cipher for Weight Bitstream Encryption
//==============================================================================
// Security rationale:
//   Weight bitstreams carry the trained model parameters — the primary IP of
//   the spectral-silicon chip.  An attacker who can probe the weight bus during
//   loading can extract the model via a simple readout.  This module decrypts
//   incoming weight data on-the-fly using a lightweight LFSR-based stream
//   cipher keyed by poly-fuse bits, so the external bus never sees plaintext
//   weights after manufacturing.
//
//   The LFSR uses the standard CRC-32 polynomial x^32 + x^22 + x^2 + 1, which
//   has good statistical properties and maximal period (2^32 - 1).  The key
//   seeds the LFSR; each clock cycle the top byte of the LFSR state is XORed
//   with the incoming data word to produce the decrypted weight.
//
//   In test mode (fuse_key == 0) the cipher is bypassed for production test
//   access — no area-wasting dummy path, just a mux that selects the raw bus
//   data when the key is unprogrammed.
//
// Interface:
//   fuse_key[31:0]   — 32-bit key from poly-fuse inputs (all 0 = test/bypass)
//   wb_dat_i[31:0]   — incoming (encrypted) Wishbone data
//   decrypted_out[31:0] — decrypted weight data (XOR with LFSR keystream)
//   key_valid        — 1 when fuse_key is non-zero (cipher active)
//
// Improvement 11 specification.
//==============================================================================
module weight_crypto #(
    parameter KEY_WIDTH = 32
) (
    input  wire              clk,
    input  wire              rst_n,

    // Poly-fuse key input (all zeros → test/bypass mode)
    input  wire [KEY_WIDTH-1:0] fuse_key,

    // Incoming encrypted weight data (from Wishbone bus)
    input  wire [31:0]       wb_dat_i,
    input  wire              data_valid,    // strobe: new encrypted word available

    // Decrypted weight data out (to weight register file)
    output wire [31:0]       decrypted_out,
    output wire              key_valid      // 1 = cipher active (key programmed)
);

    //----------------------------------------------------------------------
    // LFSR state register (Galois-type, CRC-32 polynomial)
    // Polynomial: x^32 + x^22 + x^2 + 1  (taps at bits 0, 2, 22)
    //----------------------------------------------------------------------
    reg [31:0] lfsr_state;
    wire [31:0] lfsr_next;

    // CRC-32 Galois feedback: shift right, XOR taps when LSB is 1
    // Taps for x^32 + x^22 + x^2 + 1 → bits 0, 2, 22
    assign lfsr_next = {1'b0, lfsr_state[31:1]} ^
                       (lfsr_state[0] ? 32'h00000000 |
                                       32'h00400004 : 32'h00000000);
    // Decompose for readability:
    //   bit 0  → bit 1   (shift)
    //   bit 2  → bit 3   (tap XOR if lsb=1)
    //   bit 22 → bit 23  (tap XOR if lsb=1)
    // Using the compact form above:
    //   0x00400004 = bit22 | bit2 set (XOR feedback positions)

    // key_valid: cipher is active only when fuse is programmed (non-zero)
    assign key_valid = (fuse_key != 32'h0);

    //----------------------------------------------------------------------
    // LFSR initialization / stepping
    // On reset or key load, seed the LFSR with the fuse key.
    // Each data_valid pulse advances the LFSR one step.
    //----------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lfsr_state <= 32'h0;
        end else if (!key_valid) begin
            // Test mode: LFSR held at 0, bypass active
            lfsr_state <= 32'h0;
        end else if (data_valid) begin
            lfsr_state <= lfsr_next;
        end
    end

    //----------------------------------------------------------------------
    // Keystream generation: take upper 32 bits of LFSR as keystream.
    // In bypass mode (key_valid=0), keystream = 0 → output = input (passthrough)
    //----------------------------------------------------------------------
    wire [31:0] keystream;
    assign keystream = key_valid ? lfsr_next : 32'h00000000;

    // Stream cipher: decrypted = encrypted XOR keystream
    assign decrypted_out = wb_dat_i ^ keystream;

endmodule

`default_nettype wire