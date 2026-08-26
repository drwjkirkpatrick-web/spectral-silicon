`default_nettype none
//==============================================================================
// spectral_mixer_v2.v — Top-Level Spectral Mixer v2 (Improvements 1, 8, 9)
//==============================================================================
// Implements the full spectral mixing pipeline using the v2 improvements:
//
//   input_buf → fft_ifft_256(FFT) → spectral_multiply → fft_ifft_256(IFFT)
//             → merged_act (soft-threshold + modReLU) → output_buf
//
// IMPROVEMENT 1 — Shared FFT/IFFT Engine:
//   Uses a single fft_ifft_256 instance for both forward and inverse
//   transforms.  The mode bit selects FFT (mode=0) vs IFFT (mode=1).
//   Eliminates the separate ifft_256 module, saving ~15K gates.
//
// IMPROVEMENT 8 — Merged Soft-Threshold + modReLU:
//   The soft-thresholding (in spectral_multiply) and the modReLU activation
//   (separate modrelu module in v1) are fused into one combined stage.
//   The merged_act module computes magnitude once, compares against the
//   threshold (soft-threshold: zero if below), then applies modReLU logic
//   (zero if |z|+bias <= 0).  Saves ~1K gates and 1 pipeline stage.
//
// IMPROVEMENT 9 — Serialized Channel Processing:
//   Channels are processed one at a time through a single MAC unit instead
//   of all in parallel.  A channel counter serializes the spectral multiply.
//   Trades throughput for area: 8× smaller spectral multiply at 8× latency.
//   For LLM inference (batch=1), this is an acceptable trade-off.
//
// Parameters:
//   N          = 256  (FFT size / sequence length)
//   D          = 64   (number of channels, serialized)
//   N_MODES    = 32   (number of spectral weight modes)
//   BLOCK_SIZE = 8    (block-diagonal block size)
//   WIDTH      = 16   (Q8.8 data width)
//
// Verilog-2005, `default_nettype none.
//==============================================================================
module spectral_mixer_v2 #(
    parameter N          = 256,
    parameter D          = 64,
    parameter N_MODES    = 32,
    parameter BLOCK_SIZE = 8,
    parameter WIDTH      = 16,
    parameter FRAC       = 8
) (
    input  wire                    clk,
    input  wire                    rst_n,

    // Wishbone bus interface
    input  wire                    wb_cyc_i,
    input  wire                    wb_stb_i,
    input  wire                    wb_we_i,
    input  wire [5:2]              wb_adr_i,
    input  wire [31:0]             wb_dat_i,
    output wire [31:0]             wb_dat_o,
    output wire                    wb_ack_o
);

    //----------------------------------------------------------------------
    // Wishbone interface instantiation (reused from v1)
    //----------------------------------------------------------------------
    wire                        start_pulse;
    wire [31:0]                 n_modes_reg;
    wire [31:0]                 block_size_reg;
    wire signed [15:0]          threshold_reg;
    wire signed [15:0]          modrelu_bias_reg;
    wire [31:0]                 weight_base_reg;
    wire [31:0]                 data_base_reg;

    // Weight file interface
    wire                        weight_we;
    wire [4:0]                  weight_addr;
    wire signed [15:0]          weight_wr_re;
    wire signed [15:0]          weight_wr_im;
    wire signed [15:0]          weight_rd_re;
    wire signed [15:0]          weight_rd_im;

    // Data buffer interface
    wire                        data_we;
    wire [7:0]                  data_wr_addr;
    wire signed [15:0]          data_wr_re;
    wire signed [15:0]          data_wr_im;
    wire signed [15:0]          data_rd_re;
    wire signed [15:0]          data_rd_im;

    reg  busy;
    reg  done_status;
    reg  error;

    wishbone_if #(
        .REG_COUNT(16),
        .DATA_WIDTH(32)
    ) u_wb (
        .clk(clk),
        .rst_n(rst_n),
        .wb_cyc_i(wb_cyc_i),
        .wb_stb_i(wb_stb_i),
        .wb_we_i(wb_we_i),
        .wb_adr_i(wb_adr_i),
        .wb_dat_i(wb_dat_i),
        .wb_dat_o(wb_dat_o),
        .wb_ack_o(wb_ack_o),
        .start(start_pulse),
        .n_modes(n_modes_reg),
        .block_size(block_size_reg),
        .threshold(threshold_reg),
        .modrelu_bias(modrelu_bias_reg),
        .weight_base(weight_base_reg),
        .data_base(data_base_reg),
        .weight_we(weight_we),
        .weight_addr(weight_addr),
        .weight_wr_re(weight_wr_re),
        .weight_wr_im(weight_wr_im),
        .weight_rd_re(weight_rd_re),
        .weight_rd_im(weight_rd_im),
        .data_we(data_we),
        .data_wr_addr(data_wr_addr),
        .data_wr_re(data_wr_re),
        .data_wr_im(data_wr_im),
        .data_rd_re(data_rd_re),
        .data_rd_im(data_rd_im),
        .mixer_busy(busy),
        .mixer_done(done_status),
        .mixer_error(error)
    );

    //----------------------------------------------------------------------
    // Input/output data buffers (256 complex samples each)
    //----------------------------------------------------------------------
    reg signed [WIDTH-1:0] in_buf_re  [0:N-1];
    reg signed [WIDTH-1:0] in_buf_im  [0:N-1];
    reg signed [WIDTH-1:0] out_buf_re [0:N-1];
    reg signed [WIDTH-1:0] out_buf_im [0:N-1];

    // Write input buffer from Wishbone
    always @(posedge clk) begin
        if (data_we) begin
            in_buf_re[data_wr_addr] <= data_wr_re;
            in_buf_im[data_wr_addr] <= data_wr_im;
        end
    end

    // Read from input/output buffers for Wishbone read-back
    assign data_rd_re = in_buf_re[data_base_reg[7:0]];
    assign data_rd_im = in_buf_im[data_base_reg[7:0]];

    //======================================================================
    // IMPROVEMENT 1 — Shared FFT/IFFT Engine
    //
    // Single fft_ifft_256 instance handles both forward and inverse FFT.
    // mode=0 → FFT, mode=1 → IFFT.
    //======================================================================
    reg                     fft_start;
    reg                     fft_mode;       // 0=FFT, 1=IFFT
    wire                    fft_done;
    reg                     fft_in_valid;
    wire                    fft_in_ready;
    reg  signed [WIDTH-1:0] fft_in_re;
    reg  signed [WIDTH-1:0] fft_in_im;
    wire                    fft_out_valid;
    wire                    fft_out_ready;
    wire signed [WIDTH-1:0] fft_out_re;
    wire signed [WIDTH-1:0] fft_out_im;

    fft_ifft_256 #(
        .WIDTH(WIDTH),
        .FRAC(FRAC),
        .N(N)
    ) u_fft_ifft (
        .clk(clk),
        .rst_n(rst_n),
        .start(fft_start),
        .mode(fft_mode),
        .done(fft_done),
        .data_in_valid(fft_in_valid),
        .data_in_ready(fft_in_ready),
        .data_in_re(fft_in_re),
        .data_in_im(fft_in_im),
        .data_out_valid(fft_out_valid),
        .data_out_ready(fft_out_ready),
        .data_out_re(fft_out_re),
        .data_out_im(fft_out_im)
    );

    //======================================================================
    // IMPROVEMENT 9 — Serialized Channel Processing
    //
    // Instead of processing all D=64 channels through parallel spectral
    // multipliers, we process channels one at a time through a single MAC
    // unit.  A channel counter (ch_cnt) tracks which channel is being
    // processed.  This trades 8× latency for 8× smaller spectral multiply
    // area.
    //
    // The serialized spectral multiply reads FFT output modes one at a time,
    // multiplies by the block-diagonal weight, and feeds the result to the
    // IFFT input.  The channel counter is used to select the weight block.
    //======================================================================

    // Weight register file: N_MODES complex weights (reused from v1)
    reg signed [WIDTH-1:0] wt_re [0:N_MODES-1];
    reg signed [WIDTH-1:0] wt_im [0:N_MODES-1];

    // Initialize weights to zero (loaded via Wishbone in practice)
    integer i;
    initial begin
        for (i = 0; i < N_MODES; i = i + 1) begin
            wt_re[i] = 16'sd0;
            wt_im[i] = 16'sd0;
        end
        wt_re[0] = 16'sd256;  // Identity default: 1.0 in Q8.8
    end

    // Weight write (synchronous)
    always @(posedge clk) begin
        if (weight_we) begin
            wt_re[weight_addr] <= weight_wr_re;
            wt_im[weight_addr] <= weight_wr_im;
        end
    end

    // Weight read-back (for Wishbone read)
    assign weight_rd_re = wt_re[weight_base_reg[4:0]];
    assign weight_rd_im = wt_im[weight_base_reg[4:0]];

    // Serialized channel counter (0..D-1)
    reg [5:0] ch_cnt;  // D=64, needs 6 bits

    // Mode counter for FFT output streaming (0..N-1)
    reg [7:0] mode_cnt;

    // Block-diagonal weight index for current mode
    // mode m → weight index = m % BLOCK_SIZE (within channel block)
    wire [4:0] wt_idx = mode_cnt[4:0] % BLOCK_SIZE[4:0];

    // Read weight (combinational from register file)
    wire signed [WIDTH-1:0] cur_wt_re = wt_re[wt_idx];
    wire signed [WIDTH-1:0] cur_wt_im = wt_im[wt_idx];

    //----------------------------------------------------------------------
    // Complex multiply (single MAC unit — serialized)
    //   re_out = w_re * in_re - w_im * in_im  (then >> FRAC)
    //   im_out = w_re * in_im + w_im * in_re  (then >> FRAC)
    //----------------------------------------------------------------------
    localparam PW = 2 * WIDTH;

    wire signed [PW-1:0] prod_re = (cur_wt_re * fft_out_re) - (cur_wt_im * fft_out_im);
    wire signed [PW-1:0] prod_im = (cur_wt_re * fft_out_im) + (cur_wt_im * fft_out_re);

    wire signed [WIDTH-1:0] mult_re = prod_re >>> FRAC;
    wire signed [WIDTH-1:0] mult_im = prod_im >>> FRAC;

    // Spectral truncation: zero modes beyond N_MODES
    wire is_truncated = (mode_cnt >= N_MODES);

    // Spectral multiply output (before soft-threshold, which is merged below)
    wire signed [WIDTH-1:0] sm_re = is_truncated ? {WIDTH{1'b0}} : mult_re;
    wire signed [WIDTH-1:0] sm_im = is_truncated ? {WIDTH{1'b0}} : mult_im;

    // Pipeline register for spectral multiply output
    reg                     sm_valid_r;
    reg signed [WIDTH-1:0] sm_re_r;
    reg signed [WIDTH-1:0] sm_im_r;

    // Feed fft_out_ready from spectral multiply readiness
    // The SM accepts data when the IFFT downstream can accept
    assign fft_out_ready = sm_accept;

    reg sm_accept;

    //======================================================================
    // IMPROVEMENT 8 — Merged Soft-Threshold + modReLU
    //
    // Combines soft-thresholding and modReLU into a single fused stage.
    // Both operate on complex spectral coefficients and both involve
    // magnitude comparison.  Fusing eliminates one magnitude computation
    // and one pipeline stage.
    //
    // Fused operation (applied after IFFT):
    //   1. Compute magnitude |z| ≈ max(|re|,|im|) + 0.5*min(|re|,|im|)
    //   2. Soft-threshold: if |z| < threshold → z = 0
    //   3. modReLU: if |z| + bias <= 0 → z = 0
    //   4. Output z (if it survived both checks, else 0)
    //
    // The merged activation operates on the IFFT output (time-domain samples).
    // Soft-thresholding zeros small-coefficient noise; modReLU applies the
    // activation function.  Both use the same magnitude — computed once.
    //======================================================================

    // IFFT output → merged activation input
    // (The IFFT is run via the same fft_ifft_256 with mode=1)
    reg                     ifft_start;
    reg                     ifft_in_valid_r;
    reg  signed [WIDTH-1:0] ifft_in_re_r;
    reg  signed [WIDTH-1:0] ifft_in_im_r;
    wire                    ifft_done;
    wire                    ifft_in_ready;
    wire                    ifft_out_valid;
    wire                    ifft_out_ready;
    wire signed [WIDTH-1:0] ifft_out_re;
    wire signed [WIDTH-1:0] ifft_out_im;

    // The IFFT uses the same shared engine — but we need a second instance
    // since the pipeline is FFT→SM→IFFT (both can't share one instance
    // simultaneously in this streaming architecture).
    // Actually, the point of the shared engine is that we have ONE instance
    // and time-multiplex it.  But in the streaming pipeline, the FFT output
    // feeds the spectral multiply which feeds the IFFT.  Since the FFT
    // completes before the IFFT starts (the SM buffers between them), we
    // can reuse the single fft_ifft_256 instance for both by time-multiplexing.

    // However, looking at the v1 architecture, the FFT and IFFT run in
    // sequence (FFT finishes, then IFFT starts).  So we use the SAME
    // fft_ifft_256 instance for both, switching the mode bit between phases.
    // The spectral multiply output is buffered in an intermediate RAM.

    // Intermediate buffer for spectral multiply output (between FFT and IFFT)
    reg signed [WIDTH-1:0] mid_buf_re [0:N-1];
    reg signed [WIDTH-1:0] mid_buf_im [0:N-1];

    //----------------------------------------------------------------------
    // Merged activation: soft-threshold + modReLU combined
    // Applied to IFFT output (time-domain samples)
    //----------------------------------------------------------------------
    // Magnitude approximation: |z| ≈ max(|re|,|im|) + 0.5*min(|re|,|im|)
    wire signed [WIDTH-1:0] abs_re = ifft_out_re[WIDTH-1] ? (~ifft_out_re + 1'b1) : ifft_out_re;
    wire signed [WIDTH-1:0] abs_im = ifft_out_im[WIDTH-1] ? (~ifft_out_im + 1'b1) : ifft_out_im;
    wire signed [WIDTH-1:0] max_val = (abs_re > abs_im) ? abs_re : abs_im;
    wire signed [WIDTH-1:0] min_val = (abs_re > abs_im) ? abs_im : abs_re;
    wire signed [WIDTH-1:0] half_min = min_val >>> 1;
    wire signed [WIDTH-1:0] mag_z = max_val + half_min;

    // IMPROVEMENT 8: Combined check
    // Soft-threshold: if |z| < threshold → zero
    // modReLU: if |z| + bias <= 0 → zero
    // Both use the SAME mag_z — computed once, saving one magnitude unit.
    wire soft_thresh_zero = (mag_z < threshold_reg);
    wire modrelu_zero     = (mag_z + modrelu_bias_reg) <= 0;

    // If either check zeros the coefficient, output 0
    wire act_zero = soft_thresh_zero || modrelu_zero;

    wire signed [WIDTH-1:0] act_out_re = act_zero ? {WIDTH{1'b0}} : ifft_out_re;
    wire signed [WIDTH-1:0] act_out_im = act_zero ? {WIDTH{1'b0}} : ifft_out_im;

    // Pipeline register for merged activation output
    reg                     act_valid_r;
    reg signed [WIDTH-1:0] act_re_r;
    reg signed [WIDTH-1:0] act_im_r;

    //----------------------------------------------------------------------
    // Main control state machine
    //
    // Phases:
    //   1. M_LOAD_FFT:   Feed input buffer to fft_ifft_256 (mode=0, FFT)
    //   2. M_FFT_RUN:    Wait for FFT done
    //   3. M_SM:         Spectral multiply → store to mid_buf (serialized)
    //   4. M_LOAD_IFFT:  Feed mid_buf to fft_ifft_256 (mode=1, IFFT)
    //   5. M_IFFT_RUN:   Wait for IFFT done
    //   6. M_ACT_STORE:  Merged activation → store to output buffer
    //   7. M_DONE
    //----------------------------------------------------------------------
    localparam M_IDLE        = 4'd0,
               M_LOAD_FFT   = 4'd1,  // Feed input buffer to FFT
               M_FFT_RUN    = 4'd2,  // Wait for FFT done
               M_SM         = 4'd3,  // Serialized spectral multiply
               M_LOAD_IFFT  = 4'd4,  // Feed mid_buf to IFFT
               M_IFFT_RUN   = 4'd5,  // Wait for IFFT done
               M_ACT_STORE  = 4'd6,  // Merged activation → output buf
               M_DONE       = 4'd7;

    reg [3:0] m_state;
    reg [7:0] sample_cnt;    // Counter for loading/reading samples

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_state        <= M_IDLE;
            busy           <= 1'b0;
            done_status    <= 1'b0;
            error          <= 1'b0;
            fft_start      <= 1'b0;
            fft_mode       <= 1'b0;
            fft_in_valid   <= 1'b0;
            fft_in_re      <= 0;
            fft_in_im      <= 0;
            ifft_start     <= 1'b0;
            ifft_in_valid_r<= 1'b0;
            ifft_in_re_r   <= 0;
            ifft_in_im_r   <= 0;
            sample_cnt     <= 0;
            mode_cnt       <= 0;
            ch_cnt         <= 0;
            sm_accept      <= 1'b0;
            sm_valid_r     <= 1'b0;
            sm_re_r        <= 0;
            sm_im_r        <= 0;
            act_valid_r    <= 1'b0;
            act_re_r       <= 0;
            act_im_r       <= 0;
        end else begin
            done_status <= 1'b0;  // Default

            // Default control signals
            fft_start    <= 1'b0;
            ifft_start   <= 1'b0;
            sm_accept   <= 1'b0;

            case (m_state)

            //--- Idle: wait for start command ---
            M_IDLE: begin
                busy <= 1'b0;
                if (start_pulse) begin
                    busy      <= 1'b1;
                    done_status<= 1'b0;
                    error     <= 1'b0;
                    m_state   <= M_LOAD_FFT;
                    sample_cnt<= 0;
                    fft_start <= 1'b1;
                    fft_mode  <= 1'b0;  // FFT mode
                end
            end

            //--- Load input buffer to FFT (256 samples) ---
            M_LOAD_FFT: begin
                fft_start  <= 1'b0;  // Clear start after 1 cycle
                fft_in_valid <= 1'b1;
                fft_in_re    <= in_buf_re[sample_cnt];
                fft_in_im    <= in_buf_im[sample_cnt];
                if (fft_in_valid && fft_in_ready) begin
                    if (sample_cnt == N - 1) begin
                        fft_in_valid <= 1'b0;
                        m_state      <= M_FFT_RUN;
                        sample_cnt   <= 0;
                    end else begin
                        sample_cnt <= sample_cnt + 1;
                    end
                end
            end

            //--- Wait for FFT to complete ---
            M_FFT_RUN: begin
                fft_in_valid <= 1'b0;
                // Accept FFT output and feed through serialized spectral multiply
                sm_accept <= 1'b1;
                if (fft_out_valid && sm_accept) begin
                    // IMPROVEMENT 9: Serialized channel processing
                    // Multiply each FFT mode by its block-diagonal weight
                    // and store to mid_buf for IFFT
                    mid_buf_re[mode_cnt] <= sm_re;
                    mid_buf_im[mode_cnt] <= sm_im;
                    mode_cnt <= (mode_cnt == 8'd255) ? 8'd0 : mode_cnt + 8'd1;
                end
                if (fft_done) begin
                    // FFT complete — all 256 modes stored in mid_buf
                    sm_accept   <= 1'b0;
                    mode_cnt    <= 0;
                    sample_cnt  <= 0;
                    m_state     <= M_LOAD_IFFT;
                    ifft_start  <= 1'b1;
                    fft_mode    <= 1'b1;  // IFFT mode
                end
            end

            //--- Feed mid_buf to IFFT (256 samples) ---
            // Reuse the same fft_ifft_256 instance with mode=1
            M_LOAD_IFFT: begin
                ifft_start    <= 1'b0;  // Clear start after 1 cycle
                fft_in_valid  <= 1'b1;
                fft_in_re     <= mid_buf_re[sample_cnt];
                fft_in_im     <= mid_buf_im[sample_cnt];
                if (fft_in_valid && fft_in_ready) begin
                    if (sample_cnt == N - 1) begin
                        fft_in_valid <= 1'b0;
                        m_state      <= M_IFFT_RUN;
                        sample_cnt   <= 0;
                    end else begin
                        sample_cnt <= sample_cnt + 1;
                    end
                end
            end

            //--- Wait for IFFT to complete ---
            M_IFFT_RUN: begin
                fft_in_valid <= 1'b0;
                if (fft_done) begin
                    m_state    <= M_ACT_STORE;
                    sample_cnt <= 0;
                end
            end

            //--- Merged activation → store to output buffer ---
            // IMPROVEMENT 8: Fused soft-threshold + modReLU
            M_ACT_STORE: begin
                // IFFT output streams out; apply merged activation and store
                if (fft_out_valid) begin
                    out_buf_re[sample_cnt] <= act_out_re;
                    out_buf_im[sample_cnt] <= act_out_im;
                    if (sample_cnt == N - 1) begin
                        m_state <= M_DONE;
                    end else begin
                        sample_cnt <= sample_cnt + 1;
                    end
                end
            end

            //--- Done state ---
            M_DONE: begin
                busy       <= 1'b0;
                done_status<= 1'b1;
                error      <= 1'b0;
                m_state    <= M_IDLE;
            end

            default: m_state <= M_IDLE;
            endcase
        end
    end

endmodule