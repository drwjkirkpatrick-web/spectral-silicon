`default_nettype none
//==============================================================================
// scan_lockout.v — Scan Chain Lockout via Poly-Fuse
//==============================================================================
// Security rationale:
//   Scan chains (JTAG / boundary scan) are a primary attack vector for IP
//   extraction and debug-mode exploitation.  After manufacturing test, the
//   scan chain must be permanently disabled to prevent attackers from
//   reading out internal registers and weight memories through the test
//   access port (TAP).
//
//   A poly-fuse is blown after manufacturing test.  When fuse_blown=1, this
//   module forces scan_en to 0 regardless of the external input, and forces
//   scan_out to 0 as well — effectively cutting off all scan data flow.
//   Before the fuse is blown (during manufacturing test), scan operates
//   normally.
//
//   This is purely combinational logic (~20 gates): a handful of AND/OR
//   gates gated by the fuse bit.  No clock needed — the fuse state is
//   permanent and the gating is instantaneous.
//
// Interface:
//   scan_en_in   — external scan enable (from TAP controller)
//   scan_in      — scan data input (from previous scan cell)
//   scan_out     — scan data output (to next scan cell or TAP)
//   fuse_blown   — 1 = fuse blown (production), 0 = pre-blow (test)
//   scan_en_out  — gated scan enable (always 0 when fuse blown)
//   scan_en_internal — internal scan enable (same as scan_en_out)
//
// Improvement 15 specification.
//==============================================================================
module scan_lockout (
    // External scan control
    input  wire scan_en_in,      // From TAP controller / external pin
    input  wire scan_in,         // Scan data from upstream cell
    input  wire fuse_blown,      // Poly-fuse status: 1 = blown (lockout active)

    // Gated scan outputs
    output wire scan_en_out,     // Gated scan enable to internal scan chain
    output wire scan_en_internal,// Gated scan enable (same as scan_en_out)
    output wire scan_out         // Gated scan data output
);

    //----------------------------------------------------------------------
    // Combinational gating logic:
    //
    //   scan_en_out = scan_en_in & ~fuse_blown
    //   scan_out    = scan_in    & ~fuse_blown
    //
    // When fuse_blown=1: both outputs forced to 0 → scan chain dead.
    // When fuse_blown=0: pass-through → normal scan operation (test mode).
    //
    // The ~fuse_blown inversion uses ~2 gates; each AND gate is 1 gate;
    // total ≈ 4-6 gates plus buffering.  Well within the ~20 gate budget.
    //----------------------------------------------------------------------
    wire fuse_ok;
    assign fuse_ok = ~fuse_blown;  // 1 = fuse intact (test mode)

    // Scan enable: only active when fuse is NOT blown
    assign scan_en_out      = scan_en_in & fuse_ok;
    assign scan_en_internal = scan_en_in & fuse_ok;

    // Scan data output: forced to 0 when fuse is blown
    assign scan_out = scan_in & fuse_ok;

endmodule

`default_nettype wire