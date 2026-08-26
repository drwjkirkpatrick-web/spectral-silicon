`default_nettype none
//==============================================================================
// power_flattening.v — Decoy MAC for Power-Trace Flattening
//==============================================================================
// Security rationale:
//   Differential Power Analysis (DPA) correlates power consumption with
//   internal data values to extract secret keys or weight values.  A real
//   MAC unit's power draw varies with the Hamming weight of its operands,
//   creating exploitable power signatures on the supply rail.
//
//   This module runs a "decoy" MAC unit in parallel with the real MAC,
//   consuming similar power regardless of the real MAC's activity.  The
//   decoy generates pseudo-random operands via an LFSR and performs complex
//   multiply-accumulate operations continuously.  The combined power
//   profile (real + decoy) is dominated by the decoy's constant switching
//   activity, flattening the observable power envelope.
//
//   Enable/disable is controlled by a register bit (decoy_en).  When
//   disabled, the decoy unit is clock-gated to save power during
//   non-sensitive operations.
//
//   Q8.8 signed complex MAC with LFSR-generated operands:
//     LFSR → pseudo-random 16-bit re/im operands
//     decoy_acc_re += (rand_re * rand_re - rand_im * rand_im) >>> FRAC
//     decoy_acc_im += (rand_re * rand_im + rand_im * rand_re) >>> FRAC
//
// Interface:
//   decoy_en       — control register bit: 1 = decoy active (power flatten)
//   mac_active     — 1 when real MAC is running (for correlation/timing)
//   decoy_running  — 1 = decoy MAC currently cycling
//
// Improvement 14 specification.
//==============================================================================
module power_flattening #(
    parameter WIDTH   = 16,   // Q8.8 data width
    parameter FRAC    = 8,    // fractional bits
    parameter ACC_W   = 32    // accumulator width
) (
    input  wire                    clk,
    input  wire                    rst_n,

    // Control
    input  wire                    decoy_en,    // 1 = enable decoy MAC
    input  wire                    mac_active,  // real MAC active (optional sync)

    // Status
    output wire                    decoy_running // 1 = decoy MAC cycling
);

    //----------------------------------------------------------------------
    // LFSR for pseudo-random operand generation
    // Maximal-length 16-bit LFSR, polynomial x^16 + x^14 + x^13 + x^11 + 1
    // Period = 2^16 - 1 = 65535 cycles
    //----------------------------------------------------------------------
    reg [15:0] lfsr;

    // Galois LFSR feedback: taps at 0, 11, 13, 14
    wire lfsr_fb;
    assign lfsr_fb = lfsr[0] ^ lfsr[11] ^ lfsr[13] ^ lfsr[14];

    wire [15:0] lfsr_next;
    assign lfsr_next = {lfsr[14:0], lfsr_fb};  // shift left, feedback in LSB

    //----------------------------------------------------------------------
    // Derived pseudo-random operands (Q8.8 signed)
    // Use different LFSR taps to create decorrelated re/im values.
    //----------------------------------------------------------------------
    wire signed [WIDTH-1:0] rand_re;
    wire signed [WIDTH-1:0] rand_im;

    // re = upper 16 bits of a 32-bit shift register view
    // im = lower 16 bits XOR upper bits (decorrelate)
    assign rand_re = lfsr[15:0];                       // direct LFSR state
    assign rand_im = lfsr[15:0] ^ {lfsr[7:0], lfsr[15:8]};  // bit-swapped XOR

    //----------------------------------------------------------------------
    // Decoy complex MAC (same structure as real MAC for similar power)
    //----------------------------------------------------------------------
    wire signed [2*WIDTH-1:0] d_prod_rr;
    wire signed [2*WIDTH-1:0] d_prod_ii;
    wire signed [2*WIDTH-1:0] d_prod_ri;
    wire signed [2*WIDTH-1:0] d_prod_ir;

    assign d_prod_rr = rand_re * rand_re;
    assign d_prod_ii = rand_im * rand_im;
    assign d_prod_ri = rand_re * rand_im;
    assign d_prod_ir = rand_im * rand_re;

    wire signed [ACC_W-1:0] d_mac_re;
    wire signed [ACC_W-1:0] d_mac_im;

    assign d_mac_re = {{(ACC_W-2*WIDTH){d_prod_rr[2*WIDTH-1]}}, (d_prod_rr - d_prod_ii) >>> FRAC};
    assign d_mac_im = {{(ACC_W-2*WIDTH){d_prod_ri[2*WIDTH-1]}}, (d_prod_ri + d_prod_ir) >>> FRAC};

    // Decoy accumulator (continuously wraps — we only care about power draw,
    // not the accumulated value)
    reg signed [ACC_W-1:0] decoy_acc_re;
    reg signed [ACC_W-1:0] decoy_acc_im;

    //----------------------------------------------------------------------
    // Clock-gated execution: when decoy_en=1, LFSR advances and MAC runs
    // every cycle.  When disabled, hold state (clock gating saves power).
    //----------------------------------------------------------------------
    assign decoy_running = decoy_en;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lfsr         <= 16'hACE1;  // non-zero seed
            decoy_acc_re <= {ACC_W{1'b0}};
            decoy_acc_im <= {ACC_W{1'b0}};
        end else if (decoy_en) begin
            // Advance LFSR and run decoy MAC every cycle
            lfsr         <= lfsr_next;
            decoy_acc_re <= decoy_acc_re + d_mac_re;
            decoy_acc_im <= decoy_acc_im + d_mac_im;
        end
        // When decoy_en=0: hold (clock gating effect via enable check)
    end

endmodule

`default_nettype wire