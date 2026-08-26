`default_nettype none
//==============================================================================
// em_shield.v — Top-Metal EM Shield Control
//==============================================================================
// Security rationale:
//   Electromagnetic (EM) emanations from switching transistors can be captured
//   by nearby antennas to reconstruct internal data values (EM side-channel
//   attack).  A dedicated top-metal fill pattern, when driven to ground,
//   acts as a Faraday shield that absorbs and reflects EM radiation from
//   the active logic below.
//
//   This module provides the control signal for the top-metal shield pad.
//   When shield_en=1, the shield metal is driven to ground potential,
//   creating a low-impedance return path for EM radiation.  The shield_en
//   signal is registered for clean drive and glitch-free operation.
//
//   In practice, the shield_pad output connects to a top-metal fill pattern
//   via a standard I/O pad.  The register ensures the pad output is stable
//   and does not toggle accidentally during power-up or reset.
//
//   On reset, the shield is enabled by default (fail-safe: shield on).
//
// Interface:
//   shield_en  — control: 1 = drive shield to ground (shield active)
//   shield_pad — output to top-metal EM shield pad (1 = shield grounded)
//   shield_status — readback: mirrors shield_pad
//
// Improvement 18 specification.
//==============================================================================
module em_shield (
    input  wire clk,
    input  wire rst_n,

    // Control input (from Wishbone config register or dedicated pin)
    input  wire shield_en,

    // Shield pad output (connects to top-metal fill pattern via I/O pad)
    output reg  shield_pad,

    // Status readback (mirrors shield_pad for register polling)
    output wire shield_status
);

    //----------------------------------------------------------------------
    // Registered shield control — fail-safe: shield ON at reset
    //
    // On reset (rst_n=0), shield_pad defaults to 1 (shield active/grounded).
    // This ensures the chip boots with EM protection enabled — the shield
    // is only disabled by explicit software action after secure boot.
    //
    // The register prevents glitches on the shield pad during power-up
    // sequencing or metastability on shield_en.
    //----------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            shield_pad <= 1'b1;  // fail-safe: shield ON at reset
        end else begin
            shield_pad <= shield_en;
        end
    end

    assign shield_status = shield_pad;

endmodule

`default_nettype wire