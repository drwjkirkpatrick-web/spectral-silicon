`default_nettype none
//==============================================================================
// spectral_mixer.v — Top-Level Spectral Mixer Module
//==============================================================================
// Implements the full spectral mixing pipeline:
//
//   input_buf → FFT(256) → spectral_multiply → IFFT(256) → modReLU → output_buf
//
// D=64 channels are serialized: the pipeline processes one channel at a time,
// so the same hardware handles all channels sequentially.  Input data is
// loaded via the Wishbone interface into an input buffer (256 complex samples),
// and results are read back from an output buffer after computation completes.
//
// Parameters:
//   N          = 256  (FFT size / sequence length)
//   D          = 64   (number of channels, serialized through the pipeline)
//   N_MODES    = 32   (number of spectral weight modes)
//   BLOCK_SIZE = 8    (block-diagonal block size)
//   WIDTH      = 16   (Q8.8 data width)
//
// The Wishbone interface provides:
//   - Control (start, done status)
//   - Configuration (n_modes, block_size, threshold, modrelu_bias)
//   - Weight loading (32 complex weights via WEIGHT_WR register)
//   - Data I/O (256 complex input/output samples via DATA_WR/DATA_RD)
//
// Prompt 19 specification.
//==============================================================================
module spectral_mixer #(
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
    // Wishbone interface instantiation
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
        .mixer_done(done),
        .mixer_error(error)
    );

    //----------------------------------------------------------------------
    // Input/output data buffers (256 complex samples each)
    // Input buffer: written via Wishbone, read into FFT
    // Output buffer: written from modReLU output, read via Wishbone
    //----------------------------------------------------------------------
    reg signed [WIDTH-1:0] in_buf_re  [0:N-1];
    reg signed [WIDTH-1:0] in_buf_im  [0:N-1];
    reg signed [WIDTH-1:0] out_buf_re [0:N-1];
    reg signed [WIDTH-1:0] out_buf_im [0:N-1];

    // Write input buffer from Wishbone data write
    always @(posedge clk) begin
        if (data_we) begin
            in_buf_re[data_wr_addr] <= data_wr_re;
            in_buf_im[data_wr_addr] <= data_wr_im;
        end
    end

    // Read from input/output buffers for Wishbone read-back
    assign data_rd_re = in_buf_re[data_base_reg[7:0]];
    assign data_rd_im = in_buf_im[data_base_reg[7:0]];

    //----------------------------------------------------------------------
    // FFT instance
    //----------------------------------------------------------------------
    reg                     fft_start;
    wire                    fft_done;
    reg                     fft_in_valid;
    wire                    fft_in_ready;
    reg  signed [WIDTH-1:0] fft_in_re;
    reg  signed [WIDTH-1:0] fft_in_im;
    wire                    fft_out_valid;
    wire                    fft_out_ready;
    wire signed [WIDTH-1:0] fft_out_re;
    wire signed [WIDTH-1:0] fft_out_im;

    fft_256 #(
        .WIDTH(WIDTH),
        .FRAC(FRAC),
        .N(N)
    ) u_fft (
        .clk(clk),
        .rst_n(rst_n),
        .start(fft_start),
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

    //----------------------------------------------------------------------
    // Spectral multiply instance
    //----------------------------------------------------------------------
    wire                    sm_in_ready;
    wire                    sm_out_valid;
    wire                    sm_out_ready;
    wire signed [WIDTH-1:0] sm_out_re;
    wire signed [WIDTH-1:0] sm_out_im;

    spectral_multiply #(
        .N_MODES(N_MODES),
        .BLOCK_SIZE(BLOCK_SIZE),
        .WIDTH(WIDTH),
        .FRAC(FRAC)
    ) u_sm (
        .clk(clk),
        .rst_n(rst_n),
        .weight_we(weight_we),
        .weight_addr(weight_addr),
        .weight_wr_re(weight_wr_re),
        .weight_wr_im(weight_wr_im),
        .threshold(threshold_reg),
        .data_in_valid(fft_out_valid),
        .data_in_ready(sm_in_ready),
        .data_in_re(fft_out_re),
        .data_in_im(fft_out_im),
        .data_out_valid(sm_out_valid),
        .data_out_ready(sm_out_ready),
        .data_out_re(sm_out_re),
        .data_out_im(sm_out_im)
    );

    // Ready signal chain: each stage's ready is driven by the downstream stage.
    // fft_out_ready ← sm_in_ready (spectral multiply can accept)
    // sm_out_ready  ← ifft_in_ready (IFFT can accept)
    // ifft_out_ready ← mr_data_in_ready (modReLU can accept)
    assign fft_out_ready  = sm_in_ready;
    assign sm_out_ready   = ifft_in_ready;

    // Weight read-back (for Wishbone read)
    assign weight_rd_re = u_sm.wt_re[weight_base_reg[4:0]];
    assign weight_rd_im = u_sm.wt_im[weight_base_reg[4:0]];

    //----------------------------------------------------------------------
    // IFFT instance
    //----------------------------------------------------------------------
    reg                     ifft_start;
    wire                    ifft_done;
    reg                     ifft_in_valid;
    wire                    ifft_in_ready;
    reg  signed [WIDTH-1:0] ifft_in_re;
    reg  signed [WIDTH-1:0] ifft_in_im;
    wire                    ifft_out_valid;
    wire                    ifft_out_ready;
    wire signed [WIDTH-1:0] ifft_out_re;
    wire signed [WIDTH-1:0] ifft_out_im;

    ifft_256 #(
        .WIDTH(WIDTH),
        .FRAC(FRAC),
        .N(N)
    ) u_ifft (
        .clk(clk),
        .rst_n(rst_n),
        .start(ifft_start),
        .done(ifft_done),
        .data_in_valid(ifft_in_valid),
        .data_in_ready(ifft_in_ready),
        .data_in_re(ifft_in_re),
        .data_in_im(ifft_in_im),
        .data_out_valid(ifft_out_valid),
        .data_out_ready(ifft_out_ready),
        .data_out_re(ifft_out_re),
        .data_out_im(ifft_out_im)
    );

    // Feed spectral_multiply output to IFFT input
    always @(*) begin
        ifft_in_re    = sm_out_re;
        ifft_in_im    = sm_out_im;
        ifft_in_valid = sm_out_valid;
    end

    //----------------------------------------------------------------------
    // modReLU instance
    //----------------------------------------------------------------------
    wire                    mr_out_valid;
    wire                    mr_out_ready;
    wire signed [WIDTH-1:0] mr_out_re;
    wire signed [WIDTH-1:0] mr_out_im;

    // modReLU output is always accepted (stored to output buffer)
    assign mr_out_ready = (m_state == M_MR_STORE);

    modrelu #(
        .WIDTH(WIDTH),
        .FRAC(FRAC)
    ) u_modrelu (
        .clk(clk),
        .rst_n(rst_n),
        .data_in_valid(ifft_out_valid),
        .data_in_ready(ifft_out_ready),
        .data_in_re(ifft_out_re),
        .data_in_im(ifft_out_im),
        .bias(modrelu_bias_reg),
        .data_out_valid(mr_out_valid),
        .data_out_ready(mr_out_ready),
        .data_out_re(mr_out_re),
        .data_out_im(mr_out_im)
    );

    //----------------------------------------------------------------------
    // Main control state machine
    //----------------------------------------------------------------------
    localparam M_IDLE        = 3'd0,
               M_LOAD_FFT   = 3'd1,  // Feed input buffer to FFT
               M_FFT_RUN    = 3'd2,  // Wait for FFT done
               M_SM_IFFT    = 3'd3,  // Spectral multiply → IFFT runs automatically
               M_MR_STORE   = 3'd4,  // modReLU → store to output buffer
               M_DONE       = 3'd5;

    reg [2:0] m_state;
    reg [7:0] sample_cnt;    // Counter for loading/reading samples

    // Status outputs to Wishbone
    reg busy;
    reg done;
    reg error;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_state        <= M_IDLE;
            busy           <= 1'b0;
            done           <= 1'b0;
            error          <= 1'b0;
            fft_start      <= 1'b0;
            fft_in_valid   <= 1'b0;
            fft_in_re      <= 0;
            fft_in_im      <= 0;
            ifft_start     <= 1'b0;
            sample_cnt     <= 0;
        end else begin
            done <= 1'b0;  // Default

            case (m_state)

            //--- Idle: wait for start command ---
            M_IDLE: begin
                busy <= 1'b0;
                if (start_pulse) begin
                    busy     <= 1'b1;
                    done     <= 1'b0;
                    error    <= 1'b0;
                    m_state  <= M_LOAD_FFT;
                    sample_cnt <= 0;
                    fft_start <= 1'b1;
                end
            end

            //--- Load input buffer to FFT (256 samples) ---
            M_LOAD_FFT: begin
                fft_start <= 1'b0;  // Clear start after 1 cycle
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
                if (fft_done) begin
                    // FFT done; spectral multiply and IFFT are pipelined
                    // Start IFFT (it will wait for spectral multiply output)
                    ifft_start <= 1'b1;
                    m_state    <= M_SM_IFFT;
                    sample_cnt <= 0;
                end
            end

            //--- Spectral multiply + IFFT pipeline running ---
            M_SM_IFFT: begin
                ifft_start <= 1'b0;
                if (ifft_done) begin
                    // IFFT complete, modReLU output streaming
                    m_state <= M_MR_STORE;
                    sample_cnt <= 0;
                end
            end

            //--- Store modReLU output to output buffer ---
            M_MR_STORE: begin
                if (mr_out_valid) begin
                    out_buf_re[sample_cnt] <= mr_out_re;
                    out_buf_im[sample_cnt] <= mr_out_im;
                    if (sample_cnt == N - 1) begin
                        m_state <= M_DONE;
                    end else begin
                        sample_cnt <= sample_cnt + 1;
                    end
                end
            end

            //--- Done state ---
            M_DONE: begin
                busy  <= 1'b0;
                done  <= 1'b1;
                error <= 1'b0;
                m_state <= M_IDLE;
            end

            default: m_state <= M_IDLE;
            endcase
        end
    end

endmodule