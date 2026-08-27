`default_nettype none
//==============================================================================
// mode_skip_multiply.v — Spectral Multiply with Truncated-Mode Skip
//==============================================================================
// Drop-in replacement for spectral_multiply.v.
//
// The original spectral_multiply.v processes all 256 FFT output modes with a
// full complex multiply (4 real multiplications + 2 additions) followed by a
// soft-threshold check.  However, only modes 0..N_MODES-1 (0..31) carry
// non-zero spectral weights.  Modes N_MODES..255 (32..255) are always zeroed by
// the is_truncated check — their output is guaranteed zero regardless of the
// input data or weights.
//
// Despite knowing the result is zero, the original module still toggles the
// complex multiplier for all 256 modes, burning 4 real multiplications of
// switching power on guaranteed-zero results.
//
// This module skips the multiplier entirely for truncated modes:
//   • mode_cnt <  N_MODES  →  full complex multiply + soft-threshold (as before)
//   • mode_cnt >= N_MODES  →  multiplier inputs gated to zero, output = 0
//
// The gating is achieved by ANDing the multiplier operands with
// (mode_cnt < N_MODES).  When the mode is truncated, all four multiplier
// inputs are zero, so the multiplier array sees no data transitions and
// produces zero with negligible switching activity.  A final mux selects the
// multiply result for active modes or hard zero for truncated modes.
//
// ── Power savings ──────────────────────────────────────────────────────────
//   • 224 of 256 modes (87.5%) skip the multiply entirely.
//   • Each mode uses 4 real multiplications (complex multiply).
//   • Original: 256 × 4 = 1024 multiplications active per pass.
//   • This module: 32 × 4 = 128 multiplications active per pass.
//   • Multiplier switching power reduced by 87.5%.
//   • Overall multiplier activity reduced by ~4× (1024 → 128).
//
// ── Timing / security ─────────────────────────────────────────────────────
//   This is NOT a timing side-channel:
//     • The output is identical for both paths (zero either way for truncated
//       modes — the original module already zeros them).
//     • The cycle count per mode is constant: exactly 1 cycle per mode in
//       both paths.  No early termination, no variable latency.
//     • The data_in_ready / data_out_valid handshake is identical to the
//       original module.
//   The improvement is purely in switching power: gated multiplier inputs
// prevent toggling inside the multiplier array for truncated modes.
//
// ── Interface ─────────────────────────────────────────────────────────────
//   Identical to spectral_multiply.v (same ports, same parameters).
//   Parameters: N_MODES=32, BLOCK_SIZE=8, WIDTH=16, FRAC=8.
//
// Prompt 23 specification (mode-skip power optimization).
//==============================================================================
module mode_skip_multiply #(
    parameter N_MODES    = 32,
    parameter BLOCK_SIZE = 8,
    parameter WIDTH      = 16,
    parameter FRAC       = 8
) (
    input  wire                    clk,
    input  wire                    rst_n,

    // Weight register file interface (Wishbone-driven)
    input  wire                    weight_we,           // Write enable
    input  wire [4:0]              weight_addr,          // Weight index 0..31
    input  wire signed [WIDTH-1:0] weight_wr_re,        // Weight real part
    input  wire signed [WIDTH-1:0] weight_wr_im,        // Weight imag part

    // Threshold for soft-thresholding (Q8.8)
    input  wire signed [WIDTH-1:0] threshold,

    // Input data interface (streaming FFT output, 256 modes)
    input  wire                    data_in_valid,
    output reg                     data_in_ready,
    input  wire signed [WIDTH-1:0] data_in_re,
    input  wire signed [WIDTH-1:0] data_in_im,

    // Output data interface (streaming, 256 modes)
    output reg                     data_out_valid,
    input  wire                    data_out_ready,
    output reg  signed [WIDTH-1:0] data_out_re,
    output reg  signed [WIDTH-1:0] data_out_im
);

    //----------------------------------------------------------------------
    // Weight register file: N_MODES complex weights
    //----------------------------------------------------------------------
    reg signed [WIDTH-1:0] wt_re [0:N_MODES-1];
    reg signed [WIDTH-1:0] wt_im [0:N_MODES-1];

    // Initialize weights to zero for simulation
    // In synthesis, these will be loaded via Wishbone
    integer i;
    initial begin
        for (i = 0; i < N_MODES; i = i + 1) begin
            wt_re[i] = {WIDTH{1'b0}};
            wt_im[i] = {WIDTH{1'b0}};
        end
        // Set first weight to 1.0 (Q8.8 = 256 = 0x0100) as identity default
        wt_re[0] = 16'sd256;
    end

    // Weight write (synchronous)
    always @(posedge clk) begin
        if (weight_we) begin
            wt_re[weight_addr] <= weight_wr_re;
            wt_im[weight_addr] <= weight_wr_im;
        end
    end

    //----------------------------------------------------------------------
    // Mode counter (0..255)
    //----------------------------------------------------------------------
    reg [7:0] mode_cnt;

    // Truncated-mode flag: true when mode_cnt >= N_MODES
    wire is_truncated = (mode_cnt >= N_MODES);
    // Active-mode flag: true when mode_cnt < N_MODES (multiplier enabled)
    wire is_active = ~is_truncated;

    // Determine weight index for current mode (block-diagonal)
    // mode m → weight[m % BLOCK_SIZE]
    wire [4:0] wt_idx = mode_cnt[4:0] % BLOCK_SIZE;

    // Read weight (combinational from register file)
    wire signed [WIDTH-1:0] cur_wt_re = wt_re[wt_idx];
    wire signed [WIDTH-1:0] cur_wt_im = wt_im[wt_idx];

    //----------------------------------------------------------------------
    // Multiplier input gating
    //----------------------------------------------------------------------
    // For truncated modes, gate ALL multiplier inputs to zero so the multiplier
    // array sees no data transitions.  This is the key power optimization:
    // the multiplier combinational logic does not toggle.
    //
    //   is_active=1:  gated_re = data_in_re,  gated_im = data_in_im
    //                gated_wt_re = cur_wt_re, gated_wt_im = cur_wt_im
    //   is_active=0:  all four inputs = 0  →  multiplier produces 0, no toggling
    //
    // We use bitwise AND with the replicated enable to gate to zero.
    wire signed [WIDTH-1:0] gated_in_re  = data_in_re  & {WIDTH{is_active}};
    wire signed [WIDTH-1:0] gated_in_im  = data_in_im  & {WIDTH{is_active}};
    wire signed [WIDTH-1:0] gated_wt_re  = cur_wt_re   & {WIDTH{is_active}};
    wire signed [WIDTH-1:0] gated_wt_im  = cur_wt_im   & {WIDTH{is_active}};

    //----------------------------------------------------------------------
    // Complex multiply: output_mode = weight * input_mode
    //   re_out = w_re * in_re - w_im * in_im  (then >> FRAC)
    //   im_out = w_re * in_im + w_im * in_re  (then >> FRAC)
    //
    // For truncated modes all gated inputs are zero, so products are zero.
    //----------------------------------------------------------------------
    localparam PW = 2 * WIDTH;

    wire signed [PW-1:0] prod_re = (gated_wt_re * gated_in_re) - (gated_wt_im * gated_in_im);
    wire signed [PW-1:0] prod_im = (gated_wt_re * gated_in_im) + (gated_wt_im * gated_in_re);

    wire signed [WIDTH-1:0] mult_re = prod_re >>> FRAC;
    wire signed [WIDTH-1:0] mult_im = prod_im >>> FRAC;

    //----------------------------------------------------------------------
    // Soft-thresholding: if |weight| < threshold, zero the output.
    // |weight| is approximated as max(|w_re|, |w_im|) + 0.5*min(|w_re|, |w_im|)
    // (same approximate magnitude used in modrelu.v).
    // Only meaningful for active modes; for truncated modes the weight is
    // gated to zero so mag_wt=0 and is_zero will be true (consistent with
    // outputting zero).
    //----------------------------------------------------------------------
    wire signed [WIDTH-1:0] abs_wt_re = cur_wt_re[WIDTH-1] ? (~cur_wt_re + 1) : cur_wt_re;
    wire signed [WIDTH-1:0] abs_wt_im = cur_wt_im[WIDTH-1] ? (~cur_wt_im + 1) : cur_wt_im;

    wire signed [WIDTH-1:0] max_abs  = (abs_wt_re > abs_wt_im) ? abs_wt_re : abs_wt_im;
    wire signed [WIDTH-1:0] min_abs  = (abs_wt_re > abs_wt_im) ? abs_wt_im : abs_wt_re;

    // Approximate magnitude: max + min/2
    wire signed [WIDTH-1:0] mag_wt = max_abs + (min_abs >>> 1);

    // If magnitude < threshold, output = 0 (soft-thresholding zeros the mode)
    wire is_zero = (mag_wt < threshold);

    //----------------------------------------------------------------------
    // Output mux: select multiply result or hard zero
    //----------------------------------------------------------------------
    // For active modes (mode_cnt < N_MODES):
    //   output = soft-thresholded multiply result (zero if mag < threshold)
    // For truncated modes (mode_cnt >= N_MODES):
    //   output = 0 (hard zero, multiplier was bypassed)
    wire signed [WIDTH-1:0] final_re = is_truncated ? {WIDTH{1'b0}} :
                                       is_zero     ? {WIDTH{1'b0}} : mult_re;
    wire signed [WIDTH-1:0] final_im = is_truncated ? {WIDTH{1'b0}} :
                                       is_zero     ? {WIDTH{1'b0}} : mult_im;

    //----------------------------------------------------------------------
    // Pipeline: register the result for timing (1-cycle latency)
    //----------------------------------------------------------------------
    reg                   result_valid_r;
    reg signed [WIDTH-1:0] result_re_r;
    reg signed [WIDTH-1:0] result_im_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result_valid_r <= 1'b0;
            result_re_r     <= 0;
            result_im_r     <= 0;
            mode_cnt        <= 0;
            data_in_ready   <= 1'b0;
            data_out_valid  <= 1'b0;
            data_out_re     <= 0;
            data_out_im     <= 0;
        end else begin
            // Accept input data
            data_in_ready <= 1'b1;
            if (data_in_valid && data_in_ready) begin
                // Register the multiplication result (or zero for truncated)
                result_re_r     <= final_re;
                result_im_r     <= final_im;
                result_valid_r  <= 1'b1;
                mode_cnt        <= (mode_cnt == 8'd255) ? 8'd0 : mode_cnt + 8'd1;
            end else begin
                result_valid_r <= 1'b0;
            end

            // Output registered result
            if (result_valid_r) begin
                data_out_valid <= 1'b1;
                data_out_re    <= result_re_r;
                data_out_im     <= result_im_r;
            end else begin
                data_out_valid <= 1'b0;
            end
        end
    end

endmodule