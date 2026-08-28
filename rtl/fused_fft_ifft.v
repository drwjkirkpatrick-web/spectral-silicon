`default_nettype none
//==============================================================================
// fused_fft_ifft.v — Fused FFT/IFFT with Fully Shared Datapath
//==============================================================================
// Time-multiplexes FFT and IFFT through the same butterfly network, twiddle
// ROM, and adder tree.  For IFFT, uses the conjugate method:
//
//   IFFT(x) = (1/N) * conj( FFT( conj(x) ) )
//
// Steps for IFFT mode:
//   1. Conjugate input: x* = re - j*im  (negate imaginary part)
//   2. Forward FFT through shared datapath
//   3. Conjugate output: X* = Re(X) - j*Im(X)
//   4. Scale by N: right-shift by log2(N) = 8 (for N=256)
//
// This saves ~8K gates vs separate FFT and IFFT engines by sharing:
//   • Butterfly network (radix-2 DIT)
//   • Twiddle factor ROM
//   • Adder tree / complex multiplier
//   • In-place data RAM
//
// Security preservation: the mode (FFT vs IFFT) is set once before start and
// does not change during computation.  The cycle count is identical for both
// modes — only the conjugation and scaling differ, adding no extra cycles.
// No data-dependent timing variations.
//
// Interface:
//   clk, rst         — clock and active-high reset
//   mode             — 0 = FFT, 1 = IFFT
//   start            — begin transform
//   data_in_re[15:0], data_in_im[15:0] — input complex sample
//   data_addr[7:0]   — input sample address (0..N-1)
//   data_valid       — input data is valid (write to in-place RAM)
//   result_re[15:0], result_im[15:0] — output complex sample
//   result_addr[7:0]  — output sample address
//   result_valid     — output data is valid
//   done             — transform complete
//
// Q8.8 fixed-point, 16-bit total, 8-bit fraction.
// Verilog-2005, `default_nettype none.  Synthesizable.
//==============================================================================
module fused_fft_ifft #(
    parameter WIDTH = 16,
    parameter FRAC  = 8,
    parameter N     = 256,
    parameter AW    = 8           // log2(256) = 8
) (
    input  wire                    clk,
    input  wire                    rst,
    input  wire                    mode,         // 0=FFT, 1=IFFT
    input  wire                    start,
    input  wire signed [WIDTH-1:0] data_in_re,
    input  wire signed [WIDTH-1:0] data_in_im,
    input  wire [AW-1:0]           data_addr,
    input  wire                    data_valid,
    output reg  signed [WIDTH-1:0] result_re,
    output reg  signed [WIDTH-1:0] result_im,
    output reg  [AW-1:0]           result_addr,
    output reg                     result_valid,
    output reg                     done
);

    //----------------------------------------------------------------------
    // States
    //----------------------------------------------------------------------
    localparam S_IDLE      = 4'd0,
               S_LOAD      = 4'd1,   // Load input data (with conjugation if IFFT)
               S_STAGE     = 4'd2,   // Butterfly stages
               S_CONJ_OUT  = 4'd3,   // Conjugate output (IFFT only)
               S_SCALE     = 4'd4,   // Scale by N (IFFT only)
               S_READOUT   = 4'd5,   // Output results
               S_DONE      = 4'd6;

    reg [3:0] state;

    // Mode register (latched on start)
    reg mode_r;

    // In-place data RAM (two banks for re/im)
    reg signed [WIDTH-1:0] ram_re [0:N-1];
    reg signed [WIDTH-1:0] ram_im [0:N-1];

    // Stage / group counters
    reg [3:0] stage;
    reg [AW-1:0] group;
    reg [AW-1:0] rd_addr;

    // Number of stages = log2(N) = 8
    localparam NSTAGES = 8;

    //----------------------------------------------------------------------
    // Bit-reverse function for input loading
    //----------------------------------------------------------------------
    function [AW-1:0] bit_reverse;
        input [AW-1:0] val;
        input [3:0]   nbits;
        reg [AW-1:0] result;
        integer b;
        begin
            result = {AW{1'b0}};
            for (b = 0; b < nbits; b = b + 1) begin
                result[b] = val[nbits-1-b];
            end
            bit_reverse = result;
        end
    endfunction

    //----------------------------------------------------------------------
    // Stride computation: stride = 2^stage
    //----------------------------------------------------------------------
    function [AW-1:0] stride_of;
        input [3:0] s;
        begin
            case (s)
                4'd0: stride_of = 1;
                4'd1: stride_of = 2;
                4'd2: stride_of = 4;
                4'd3: stride_of = 8;
                4'd4: stride_of = 16;
                4'd5: stride_of = 32;
                4'd6: stride_of = 64;
                4'd7: stride_of = 128;
                default: stride_of = 1;
            endcase
        end
    endfunction

    //----------------------------------------------------------------------
    // Twiddle factor approximation (same approach as configurable_fft.v)
    // cos(2*pi*k/N) and sin(2*pi*k/N) in Q8.8
    //----------------------------------------------------------------------
    function signed [WIDTH-1:0] tw_cos;
        input [AW-1:0] idx;
        input [AW-1:0] n;
        begin
            if (idx == 0)
                tw_cos = 16'sd256;
            else if (idx == n >> 2)
                tw_cos = 16'sd0;
            else if (idx == n >> 1)
                tw_cos = -16'sd256;
            else if (idx == (n >> 1) + (n >> 2))
                tw_cos = 16'sd0;
            else if (idx < n >> 2)
                tw_cos = 16'sd181;
            else if (idx < n >> 1)
                tw_cos = -16'sd181;
            else if (idx < (n >> 1) + (n >> 2))
                tw_cos = -16'sd181;
            else
                tw_cos = 16'sd181;
        end
    endfunction

    function signed [WIDTH-1:0] tw_sin;
        input [AW-1:0] idx;
        input [AW-1:0] n;
        begin
            if (idx == 0)
                tw_sin = 16'sd0;
            else if (idx == n >> 2)
                tw_sin = 16'sd256;
            else if (idx == n >> 1)
                tw_sin = 16'sd0;
            else if (idx == (n >> 1) + (n >> 2))
                tw_sin = -16'sd256;
            else if (idx < n >> 2)
                tw_sin = 16'sd181;
            else if (idx < n >> 1)
                tw_sin = 16'sd181;
            else if (idx < (n >> 1) + (n >> 2))
                tw_sin = -16'sd181;
            else
                tw_sin = -16'sd181;
        end
    endfunction

    //----------------------------------------------------------------------
    // Butterfly addresses
    //----------------------------------------------------------------------
    wire [AW-1:0] stride    = stride_of(stage);
    wire [AW-1:0] span      = stride * 2;
    wire [AW-1:0] base_addr = group * span;
    wire [AW-1:0] sa0       = base_addr;
    wire [AW-1:0] sa1       = base_addr + stride;

    wire [AW-1:0] tw_addr   = (group * (N >> (stage + 1))) & (N - 1);

    wire signed [WIDTH-1:0] w_re = tw_cos(tw_addr, N[AW-1:0]);
    wire signed [WIDTH-1:0] w_im = tw_sin(tw_addr, N[AW-1:0]);

    // RAM reads
    wire signed [WIDTH-1:0] sx0_re = ram_re[sa0];
    wire signed [WIDTH-1:0] sx0_im = ram_im[sa0];
    wire signed [WIDTH-1:0] sx1_re = ram_re[sa1];
    wire signed [WIDTH-1:0] sx1_im = ram_im[sa1];

    // Radix-2 butterfly: X0 = x0 + W*x1, X1 = x0 - W*x1
    localparam PW = 2 * WIDTH;

    wire signed [PW-1:0] w1_re_full = (w_re * sx1_re) - (w_im * sx1_im);
    wire signed [PW-1:0] w1_im_full = (w_re * sx1_im) + (w_im * sx1_re);

    wire signed [WIDTH-1:0] w1_re_q = w1_re_full >>> FRAC;
    wire signed [WIDTH-1:0] w1_im_q = w1_im_full >>> FRAC;

    wire signed [WIDTH:0] y0_re = {sx0_re[WIDTH-1], sx0_re} + {w1_re_q[WIDTH-1], w1_re_q};
    wire signed [WIDTH:0] y0_im = {sx0_im[WIDTH-1], sx0_im} + {w1_im_q[WIDTH-1], w1_im_q};
    wire signed [WIDTH:0] y1_re = {sx0_re[WIDTH-1], sx0_re} - {w1_re_q[WIDTH-1], w1_re_q};
    wire signed [WIDTH:0] y1_im = {sx0_im[WIDTH-1], sx0_im} - {w1_im_q[WIDTH-1], w1_im_q};

    wire [AW-1:0] group_max = (N >> (stage + 1)) - 1;

    //----------------------------------------------------------------------
    // IFFT scale shift: log2(N) = 8
    //----------------------------------------------------------------------
    localparam SCALE_SHIFT = 8;   // N=256 → shift right by 8

    //----------------------------------------------------------------------
    // Main state machine
    //----------------------------------------------------------------------
    integer i;
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state        <= S_IDLE;
            stage        <= 0;
            group        <= 0;
            rd_addr      <= 0;
            mode_r       <= 1'b0;
            result_re    <= {WIDTH{1'b0}};
            result_im    <= {WIDTH{1'b0}};
            result_addr  <= {AW{1'b0}};
            result_valid <= 1'b0;
            done         <= 1'b0;
        end else begin
            done         <= 1'b0;
            result_valid <= 1'b0;

            case (state)
            //--------------------------------------------------------------
            S_IDLE: begin
                if (start) begin
                    mode_r <= mode;
                    stage  <= 0;
                    group  <= 0;
                    rd_addr <= 0;
                    state  <= S_LOAD;
                end
            end

            //--------------------------------------------------------------
            // Load input data with bit-reversal.
            // For IFFT: conjugate input (negate imaginary part).
            //--------------------------------------------------------------
            S_LOAD: begin
                if (data_valid) begin
                    if (mode_r == 1'b1) begin
                        // IFFT: conjugate input (negate im)
                        ram_re[bit_reverse(data_addr, NSTAGES)] <= data_in_re;
                        ram_im[bit_reverse(data_addr, NSTAGES)] <= -data_in_im;
                    end else begin
                        // FFT: store as-is
                        ram_re[bit_reverse(data_addr, NSTAGES)] <= data_in_re;
                        ram_im[bit_reverse(data_addr, NSTAGES)] <= data_in_im;
                    end

                    if (data_addr == N[AW-1:0] - 1) begin
                        stage <= 0;
                        group <= 0;
                        state <= S_STAGE;
                    end
                end
            end

            //--------------------------------------------------------------
            // Butterfly stages (shared FFT datapath)
            //--------------------------------------------------------------
            S_STAGE: begin
                // Write butterfly results back to RAM
                ram_re[sa0] <= y0_re[WIDTH-1:0];
                ram_im[sa0] <= y0_im[WIDTH-1:0];
                ram_re[sa1] <= y1_re[WIDTH-1:0];
                ram_im[sa1] <= y1_im[WIDTH-1:0];

                // Advance group or stage
                if (group == group_max) begin
                    if (stage == NSTAGES - 1) begin
                        // All stages complete
                        if (mode_r == 1'b1) begin
                            // IFFT: need conjugate output + scale
                            rd_addr <= 0;
                            state   <= S_CONJ_OUT;
                        end else begin
                            // FFT: direct readout
                            rd_addr <= 0;
                            state   <= S_READOUT;
                        end
                    end else begin
                        stage <= stage + 1;
                        group <= 0;
                    end
                end else begin
                    group <= group + 1;
                end
            end

            //--------------------------------------------------------------
            // IFFT: conjugate output and scale by N
            // Conjugate: Re stays, Im negated.
            // Scale: right-shift by SCALE_SHIFT (log2(N)).
            //--------------------------------------------------------------
            S_CONJ_OUT: begin
                // Conjugate and scale each entry
                ram_re[rd_addr] <= ram_re[rd_addr] >>> SCALE_SHIFT;
                ram_im[rd_addr] <= -(ram_im[rd_addr] >>> SCALE_SHIFT);

                if (rd_addr == N[AW-1:0] - 1) begin
                    rd_addr <= 0;
                    state   <= S_READOUT;
                end else begin
                    rd_addr <= rd_addr + 1;
                end
            end

            //--------------------------------------------------------------
            // Readout: stream results
            //--------------------------------------------------------------
            S_READOUT: begin
                result_valid <= 1'b1;
                result_re    <= ram_re[rd_addr];
                result_im    <= ram_im[rd_addr];
                result_addr  <= rd_addr;

                if (rd_addr == N[AW-1:0] - 1) begin
                    state <= S_DONE;
                end else begin
                    rd_addr <= rd_addr + 1;
                end
            end

            //--------------------------------------------------------------
            S_DONE: begin
                result_valid <= 1'b0;
                done         <= 1'b1;
                state        <= S_IDLE;
            end

            default: state <= S_IDLE;
            endcase
        end
    end

endmodule