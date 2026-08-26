`default_nettype none
//==============================================================================
// integrity_hash.v — FNV-1a Rolling Hash for Weight Tamper Detection
//==============================================================================
// Security rationale:
//   After weights are loaded into the register file, an attacker with physical
//   access (or a fault injection probe) could modify individual weight cells
//   to alter the model behavior.  A rolling hash computed during loading
//   provides runtime integrity verification: the hash is computed over all
//   weight data as it streams in, then compared against an expected value
//   loaded via Wishbone.  Any tampering with weights after loading will be
//   detected on the next verification cycle.
//
//   We use FNV-1a (Fowler-Noll-Vo 1a) rather than full SHA-256 for area
//   efficiency.  FNV-1a requires only a multiply and XOR per word — a fraction
//   of the gate count of a SHA-256 core — while still providing strong
//   detection of accidental and many malicious modifications.  The 32-bit
//   hash has a collision space of ~4 billion, which is sufficient for the
//   32-mode × 32-bit weight register file (64 words).
//
//   FNV-1a algorithm:
//     hash = FNV_OFFSET_BASIS
//     for each byte b in data:
//       hash = hash XOR b
//       hash = hash * FNV_PRIME
//
//   We process one 32-bit word per cycle (4 bytes), applying FNV-1a to each
//   byte of the word in sequence within a single clock using unrolled logic.
//
// Interface:
//   data_in[31:0]  — weight data word being loaded
//   data_valid     — strobe: new weight word available
//   hash_start     — reset hash to FNV offset basis (begin loading session)
//   expected_hash  — expected hash value (loaded via Wishbone before verify)
//   hash_verify    — strobe: compare computed vs expected, update tamper_flag
//   computed_hash  — current hash value (for readback/debug)
//   tamper_flag    — 1 = hash mismatch detected (sticky until reset)
//
// Improvement 17 specification.
//==============================================================================
module integrity_hash #(
    parameter FNV_OFFSET_BASIS = 32'h811C9DC5,
    parameter FNV_PRIME        = 32'h01000193
) (
    input  wire       clk,
    input  wire       rst_n,

    // Weight data input (streaming during load)
    input  wire [31:0] data_in,
    input  wire       data_valid,     // new word available

    // Control
    input  wire       hash_start,     // reset hash to offset basis
    input  wire       hash_verify,    // compare computed vs expected

    // Expected hash (loaded via Wishbone register)
    input  wire [31:0] expected_hash,

    // Outputs
    output wire [31:0] computed_hash,  // current rolling hash
    output reg         tamper_flag     // sticky: 1 = mismatch detected
);

    //----------------------------------------------------------------------
    // Rolling hash register
    //----------------------------------------------------------------------
    reg [31:0] hash_reg;

    //----------------------------------------------------------------------
    // FNV-1a per-byte update (unrolled for 4 bytes of a 32-bit word)
    // For each byte: hash = (hash XOR byte) * FNV_PRIME
    // We compute all 4 byte-steps combinationally, then register the result.
    //----------------------------------------------------------------------
    function [31:0] fnv1a_word;
        input [31:0] current_hash;
        input [31:0] word_in;
        reg [31:0] h;
        reg [7:0]  b;
        begin
            h = current_hash;
            // Byte 0 (LSB)
            b = word_in[7:0];
            h = (h ^ {24'h0, b}) * FNV_PRIME;
            // Byte 1
            b = word_in[15:8];
            h = (h ^ {24'h0, b}) * FNV_PRIME;
            // Byte 2
            b = word_in[23:16];
            h = (h ^ {24'h0, b}) * FNV_PRIME;
            // Byte 3 (MSB)
            b = word_in[31:24];
            h = (h ^ {24'h0, b}) * FNV_PRIME;
            fnv1a_word = h;
        end
    endfunction

    wire [31:0] hash_next;
    assign hash_next = fnv1a_word(hash_reg, data_in);

    //----------------------------------------------------------------------
    // Hash update logic
    //----------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            hash_reg <= FNV_OFFSET_BASIS;
            tamper_flag <= 1'b0;
        end else if (hash_start) begin
            // Begin new loading session: reset hash
            hash_reg <= FNV_OFFSET_BASIS;
            tamper_flag <= 1'b0;  // clear on new session
        end else if (data_valid) begin
            // Rolling update with each weight word
            hash_reg <= hash_next;
        end else if (hash_verify) begin
            // Compare computed hash against expected value
            if (hash_reg != expected_hash) begin
                tamper_flag <= 1'b1;  // sticky: stays high until reset/start
            end
        end
    end

    assign computed_hash = hash_reg;

endmodule

`default_nettype wire