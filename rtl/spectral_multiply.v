`default_nettype none
//==============================================================================
// spectral_multiply.v — Spectral Weight Multiply + Soft Thresholding
//==============================================================================
// Stores k=32 complex spectral weights in a register file (Q8.8 fixed-point).
// Reads FFT output modes, multiplies each mode by its corresponding weight,
// applies soft-thresholding (|w| < threshold → 0), and outputs the result.
//
// Block-diagonal structure: the N_MODES=32 weights are shared across BLOCK_SIZE=8
// channel blocks.  Mode index m (0..31) maps to weight[m % BLOCK_SIZE] in
// block-diagonal fashion.  For a 256-point FFT, modes 0..31 are kept and the
// remaining modes (32..255) are zeroed (spectral truncation).
//
// The weight register file is Wishbone-loadable: the parent module exposes
// weight write access through the Wishbone interface.  Weight address and
// data come from the wishbone_if module ports.
//
// Parameters:
//   N_MODES    = 32   (number of spectral modes / weights)
//   BLOCK_SIZE = 8    (block-diagonal block size)
//   WIDTH      = 16   (Q8.8 data width)
//
// Prompt 16 specification.
//==============================================================================
module spectral_multiply #(
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

    // Initialize weights to identity (1 + 0j) for simulation
    // In synthesis, these will be loaded via Wishbone
    integer i;
    initial begin
        for (i = 0; i < N_MODES; i = i + 1) begin
            wt_re[i] = {1'b0, {(WIDTH-1){1'b0}}};  // 0
            wt_im[i] = {1'b0, {(WIDTH-1){1'b0}}};  // 0
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

    // Determine weight index for current mode (block-diagonal)
    // mode m → weight[m % BLOCK_SIZE]
    wire [4:0] wt_idx = mode_cnt[4:0] % BLOCK_SIZE;

    // Read weight (combinational from register file)
    wire signed [WIDTH-1:0] cur_wt_re = wt_re[wt_idx];
    wire signed [WIDTH-1:0] cur_wt_im = wt_im[wt_idx];

    //----------------------------------------------------------------------
    // Complex multiply: output_mode = weight * input_mode
    //   re_out = w_re * in_re - w_im * in_im  (then >> FRAC)
    //   im_out = w_re * in_im + w_im * in_re  (then >> FRAC)
    //----------------------------------------------------------------------
    localparam PW = 2 * WIDTH;

    wire signed [PW-1:0] prod_re = (cur_wt_re * data_in_re) - (cur_wt_im * data_in_im);
    wire signed [PW-1:0] prod_im = (cur_wt_re * data_in_im) + (cur_wt_im * data_in_re);

    wire signed [WIDTH-1:0] mult_re = prod_re >>> FRAC;
    wire signed [WIDTH-1:0] mult_im = prod_im >>> FRAC;

    //----------------------------------------------------------------------
    // Soft-thresholding: if |weight| < threshold, zero the output.
    // |weight| is approximated as max(|w_re|, |w_im|) + 0.5*min(|w_re|, |w_im|)
    // (same approximate magnitude used in modrelu.v).
    //----------------------------------------------------------------------
    wire signed [WIDTH-1:0] abs_wt_re = cur_wt_re[WIDTH-1] ? (~cur_wt_re + 1) : cur_wt_re;
    wire signed [WIDTH-1:0] abs_wt_im = cur_wt_im[WIDTH-1] ? (~cur_wt_im + 1) : cur_wt_im;

    wire signed [WIDTH-1:0] max_abs  = (abs_wt_re > abs_wt_im) ? abs_wt_re : abs_wt_im;
    wire signed [WIDTH-1:0] min_abs  = (abs_wt_re > abs_wt_im) ? abs_wt_im : abs_wt_re;

    // Approximate magnitude: max + min/2
    wire signed [WIDTH-1:0] mag_wt = max_abs + (min_abs >>> 1);

    // If magnitude < threshold, output = 0 (soft-thresholding zeros the mode)
    wire is_zero = (mag_wt < threshold);

    // Also zero modes beyond N_MODES (spectral truncation)
    wire is_truncated = (mode_cnt >= N_MODES);

    // Final output: multiply result, zeroed if below threshold or truncated
    wire signed [WIDTH-1:0] final_re = (is_zero || is_truncated) ? {WIDTH{1'b0}} : mult_re;
    wire signed [WIDTH-1:0] final_im = (is_zero || is_truncated) ? {WIDTH{1'b0}} : mult_im;

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
                // Register the multiplication result
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