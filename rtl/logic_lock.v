`default_nettype none
//==============================================================================
// logic_lock.v — Logic Locking with Spectral-Mode Key
//==============================================================================
// Security rationale:
//   Logic locking protects the chip against reverse engineering and IP theft
//   by inserting key-gated multiplexers that select between real and decoy
//   functional paths.  Without the correct key, the chip produces incorrect
//   twiddle factors, causing the FFT to produce garbage output — the device
//   appears to function but produces wrong results.
//
//   The key is stored in poly-fuse bits and read on every cycle.  A simple
//   comparator checks the applied key against KEY_CONST.  When matched, real
//   twiddle factors pass through; otherwise, decoy (scrambled) values are
//   selected, corrupting all spectral computations.
//
//   This module locks the twiddle factor path, which is central to the FFT
//   pipeline — locking twiddles effectively locks the entire spectral MAC.
//
// Interface:
//   lock_key[31:0]  — key from poly-fuse or external key input
//   twiddle_re[WIDTH-1:0] — real twiddle factor (real path)
//   twiddle_im[WIDTH-1:0] — imag twiddle factor (real path)
//   decoy_re[WIDTH-1:0]  — decoy real value (wrong path)
//   decoy_im[WIDTH-1:0]  — decoy imag value (wrong path)
//   out_re / out_im       — selected twiddle factor output
//   lock_active          — 1 when key mismatch (decoy path active)
//
// Improvement 12 specification.
//==============================================================================
module logic_lock #(
    parameter KEY_CONST = 32'hA5A5_5A5A,   // Expected key value
    parameter WIDTH     = 16                 // Q8.8 data width
) (
    input  wire                    clk,
    input  wire                    rst_n,

    // Key input (from poly-fuse or external key register)
    input  wire [31:0]             lock_key,

    // Real twiddle factors (correct values)
    input  wire signed [WIDTH-1:0] twiddle_re,
    input  wire signed [WIDTH-1:0] twiddle_im,

    // Decoy twiddle factors (scrambled values for wrong key)
    input  wire signed [WIDTH-1:0] decoy_re,
    input  wire signed [WIDTH-1:0] decoy_im,

    // Selected output (real if key matches, decoy otherwise)
    output reg  signed [WIDTH-1:0] out_re,
    output reg  signed [WIDTH-1:0] out_im,

    // Status: 1 = lock active (key mismatch, decoy path selected)
    output wire                   lock_active
);

    //----------------------------------------------------------------------
    // Key comparator — combinational
    // When key matches KEY_CONST, select real twiddles; else decoy.
    // This is the simplest form of logic locking: a single comparison gate
    // controls the functional correctness of the entire datapath.
    //----------------------------------------------------------------------
    wire key_match;
    assign key_match = (lock_key == KEY_CONST);
    assign lock_active = ~key_match;

    //----------------------------------------------------------------------
    // Key-gated mux: registered output for clean timing
    // The register ensures the mux output is glitch-free even if the key
    // input has brief metastability.  On reset, output zeros (safe default).
    //----------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_re <= {WIDTH{1'b0}};
            out_im <= {WIDTH{1'b0}};
        end else begin
            if (key_match) begin
                out_re <= twiddle_re;
                out_im <= twiddle_im;
            end else begin
                // Key mismatch: route decoy values to corrupt computation.
                // The decoy values are typically bit-reversed or negated
                // versions of the real twiddles, ensuring the FFT produces
                // structurally-valid but numerically-wrong output.
                out_re <= decoy_re;
                out_im <= decoy_im;
            end
        end
    end

endmodule

`default_nettype wire