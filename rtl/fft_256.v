`default_nettype none
//==============================================================================
// fft_256.v — Complete 256-Point FFT (4 Radix-4 Stages)
//==============================================================================
// Implements a 256-point FFT using 4 radix-4 stages (256 = 4^4).
//
// Architecture (iterative, resource-shared for ASIC area efficiency):
//   1. Load 256 complex samples into internal RAM with base-4 digit-reversal
//      (bit-reversal for radix-4 DIT).
//   2. Iteratively process 4 stages.  Each stage performs 64 radix-4 butterfly
//      operations, reading/writing from/to the same in-place RAM.
//   3. Read out 256 complex results in natural order.
//
// Radix-4 DIT stage addressing (stage s = 0..3):
//   Butterfly span  L = 4^(s+1)
//   Stride          = 4^s  (distance between butterfly inputs)
//   Groups          = N / L = 4^(3-s)
//   Group g inputs:  base + j*stride  for j=0,1,2,3
//     where base = g * L
//   Twiddle for element j in group g:  W_N^{j * g * (N/L)}
//     twiddle_addr = j * g * (N/L) mod N = j * g * 4^(3-s) mod 256
//
// Parameters:
//   WIDTH = 16 (Q8.8 fixed-point)
//
// Prompt 15 specification.
//==============================================================================
module fft_256 #(
    parameter WIDTH = 16,
    parameter FRAC  = 8,
    parameter N     = 256
) (
    input  wire                    clk,
    input  wire                    rst_n,

    // Control
    input  wire                    start,        // Assert to begin FFT
    output reg                     done,         // Asserted when result ready

    // Input data interface (streaming, 256 samples)
    input  wire                    data_in_valid,
    output reg                     data_in_ready,
    input  wire signed [WIDTH-1:0] data_in_re,
    input  wire signed [WIDTH-1:0] data_in_im,

    // Output data interface (streaming, 256 samples)
    output reg                     data_out_valid,
    input  wire                    data_out_ready,
    output reg  signed [WIDTH-1:0] data_out_re,
    output reg  signed [WIDTH-1:0] data_out_im
);

    localparam NLOG = 8;               // log2(256) = 8
    localparam GROUPS_PER_STAGE = 64;  // 256/4

    //----------------------------------------------------------------------
    // State machine states
    //----------------------------------------------------------------------
    localparam ST_IDLE        = 3'd0,
               ST_LOAD       = 3'd1,
               ST_TW_ADDR    = 3'd2,
               ST_TW_WAIT    = 3'd3,
               ST_BUTTERFLY  = 3'd4,
               ST_READ_OUT   = 3'd5,
               ST_FINISH     = 3'd6;

    reg [2:0] state;

    //----------------------------------------------------------------------
    // In-place RAM: 256 complex entries
    //----------------------------------------------------------------------
    reg signed [WIDTH-1:0] ram_re [0:N-1];
    reg signed [WIDTH-1:0] ram_im [0:N-1];

    // Counters
    reg [NLOG-1:0] wr_addr;     // Input write address
    reg [NLOG-1:0] rd_addr;    // Output read address
    reg [1:0]     stage;        // Current stage 0..3
    reg [5:0]     group;        // Group within stage 0..63
    reg [1:0]     tw_idx;       // Twiddle read index 0..2

    //----------------------------------------------------------------------
    // Base-4 digit reversal (bit-reversal for radix-4 DIT)
    // 8-bit address split into 4 base-4 digits (2 bits each), reversed.
    //----------------------------------------------------------------------
    function [NLOG-1:0] digit_rev;
        input [NLOG-1:0] addr;
        begin
            digit_rev = {addr[1:0], addr[3:2], addr[5:4], addr[7:6]};
        end
    endfunction

    //----------------------------------------------------------------------
    // Address computation for current stage/group
    // stride = 4^stage, L = 4 * stride, base = group * L
    //----------------------------------------------------------------------
    function [NLOG-1:0] stride_of;       // 4^stage
        input [1:0] s;
        begin
            case (s)
                2'd0: stride_of = 8'd1;
                2'd1: stride_of = 8'd4;
                2'd2: stride_of = 8'd16;
                2'd3: stride_of = 8'd64;
                default: stride_of = 8'd1;
            endcase
        end
    endfunction

    function [NLOG-1:0] tw_stride_of;    // 4^(3-stage) = N / (4 * stride)
        input [1:0] s;
        begin
            case (s)
                2'd0: tw_stride_of = 8'd64;
                2'd1: tw_stride_of = 8'd16;
                2'd2: tw_stride_of = 8'd4;
                2'd3: tw_stride_of = 8'd1;
                default: tw_stride_of = 8'd64;
            endcase
        end
    endfunction

    // Current addresses (combinational from stage, group)
    wire [NLOG-1:0] stride    = stride_of(stage);
    wire [NLOG-1:0] span      = stride * 4;       // L = 4 * stride
    wire [NLOG-1:0] base_addr = group * span;      // base = group * L
    wire [NLOG-1:0] sa0 = base_addr;
    wire [NLOG-1:0] sa1 = base_addr + stride;
    wire [NLOG-1:0] sa2 = base_addr + 2 * stride;
    wire [NLOG-1:0] sa3 = base_addr + 3 * stride;

    // Twiddle addresses: W_j addr = j * group * tw_stride mod 256
    wire [7:0] tw_stride = tw_stride_of(stage);
    wire [7:0] tw_a1 = (group * tw_stride) & 8'hFF;
    wire [7:0] tw_a2 = (group * tw_stride * 2) & 8'hFF;
    wire [7:0] tw_a3 = (group * tw_stride * 3) & 8'hFF;

    //----------------------------------------------------------------------
    // Twiddle ROM (single instance, time-multiplexed for 3 reads)
    //----------------------------------------------------------------------
    reg  [7:0] tw_rom_addr;
    wire signed [WIDTH-1:0] tw_cos, tw_sin;

    twiddle_rom #(
        .N(N),
        .WIDTH(WIDTH),
        .ADDR_BITS(8),
        .COS_FILE("twiddle_data/twiddle_cos_256.hex"),
        .SIN_FILE("twiddle_data/twiddle_sin_256.hex")
    ) u_twiddle (
        .clk(clk),
        .addr(tw_rom_addr),
        .cos_out(tw_cos),
        .sin_out(tw_sin)
    );

    // Registered twiddle values (captured after ROM latency)
    reg signed [WIDTH-1:0] tw1_re_r, tw1_im_r;
    reg signed [WIDTH-1:0] tw2_re_r, tw2_im_r;
    reg signed [WIDTH-1:0] tw3_re_r, tw3_im_r;

    //----------------------------------------------------------------------
    // Radix-4 butterfly
    //----------------------------------------------------------------------
    wire signed [WIDTH-1:0] bf_y0_re, bf_y0_im, bf_y1_re, bf_y1_im;
    wire signed [WIDTH-1:0] bf_y2_re, bf_y2_im, bf_y3_re, bf_y3_im;

    // Latched sample values for butterfly input (registered for timing)
    reg signed [WIDTH-1:0] sx0_re, sx0_im, sx1_re, sx1_im;
    reg signed [WIDTH-1:0] sx2_re, sx2_im, sx3_re, sx3_im;

    butterfly4 #(
        .WIDTH(WIDTH),
        .FRAC(FRAC)
    ) u_bf (
        .x0_re(sx0_re), .x0_im(sx0_im),
        .x1_re(sx1_re), .x1_im(sx1_im),
        .x2_re(sx2_re), .x2_im(sx2_im),
        .x3_re(sx3_re), .x3_im(sx3_im),
        .w1_re(tw1_re_r), .w1_im(tw1_im_r),
        .w2_re(tw2_re_r), .w2_im(tw2_im_r),
        .w3_re(tw3_re_r), .w3_im(tw3_im_r),
        .y0_re(bf_y0_re), .y0_im(bf_y0_im),
        .y1_re(bf_y1_re), .y1_im(bf_y1_im),
        .y2_re(bf_y2_re), .y2_im(bf_y2_im),
        .y3_re(bf_y3_re), .y3_im(bf_y3_im)
    );

    //----------------------------------------------------------------------
    // Main state machine
    //----------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state          <= ST_IDLE;
            done           <= 1'b0;
            data_in_ready  <= 1'b0;
            data_out_valid <= 1'b0;
            data_out_re   <= 0;
            data_out_im   <= 0;
            wr_addr        <= 0;
            rd_addr        <= 0;
            stage          <= 0;
            group          <= 0;
            tw_idx         <= 0;
            tw_rom_addr    <= 0;
            tw1_re_r <= 0; tw1_im_r <= 0;
            tw2_re_r <= 0; tw2_im_r <= 0;
            tw3_re_r <= 0; tw3_im_r <= 0;
            sx0_re <= 0; sx0_im <= 0;
            sx1_re <= 0; sx1_im <= 0;
            sx2_re <= 0; sx2_im <= 0;
            sx3_re <= 0; sx3_im <= 0;
        end else begin
            done <= 1'b0;  // Default: clear done each cycle

            case (state)

            //--- Idle: wait for start, accept input data ---
            ST_IDLE: begin
                data_in_ready <= 1'b1;
                if (start) begin
                    state    <= ST_LOAD;
                    wr_addr  <= 0;
                    data_in_ready <= 1'b0;
                end
            end

            //--- Load input: store 256 samples with digit-reversal ---
            ST_LOAD: begin
                if (data_in_valid) begin
                    ram_re[digit_rev(wr_addr)] <= data_in_re;
                    ram_im[digit_rev(wr_addr)] <= data_in_im;
                    if (wr_addr == N - 1) begin
                        state  <= ST_TW_ADDR;
                        stage  <= 2'd0;
                        group  <= 6'd0;
                        tw_idx <= 2'd0;
                        tw_rom_addr <= 8'd0;  // First twiddle (W^0)
                    end else begin
                        wr_addr <= wr_addr + 1;
                    end
                end
            end

            //--- Issue twiddle addresses sequentially (3 reads) ---
            // ROM has 1-cycle latency: address on cycle T → data on cycle T+1.
            // We issue W1 addr, then W2 addr, then W3 addr, capturing each.
            ST_TW_ADDR: begin
                case (tw_idx)
                    2'd0: begin
                        tw_rom_addr <= tw_a1;      // Issue W1
                        tw_idx      <= 2'd1;
                        state       <= ST_TW_WAIT;
                    end
                    2'd1: begin
                        tw_rom_addr <= tw_a2;      // Issue W2
                        tw_idx      <= 2'd2;
                        state       <= ST_TW_WAIT;
                    end
                    2'd2: begin
                        tw_rom_addr <= tw_a3;      // Issue W3
                        tw_idx      <= 2'd0;
                        state       <= ST_TW_WAIT;
                    end
                    default: tw_idx <= 2'd0;
                endcase
            end

            //--- Wait for ROM latency, capture twiddle value ---
            ST_TW_WAIT: begin
                // Capture the twiddle value that just came back from ROM
                case (tw_idx)
                    2'd1: begin  // W1 just read
                        tw1_re_r <= tw_cos;
                        tw1_im_r <= tw_sin;
                        state    <= ST_TW_ADDR;  // Go issue W2
                    end
                    2'd2: begin  // W2 just read
                        tw2_re_r <= tw_cos;
                        tw2_im_r <= tw_sin;
                        state    <= ST_TW_ADDR;  // Go issue W3
                    end
                    2'd0: begin  // W3 just read
                        tw3_re_r <= tw_cos;
                        tw3_im_r <= tw_sin;
                        // Latch butterfly input samples from RAM
                        sx0_re <= ram_re[sa0];  sx0_im <= ram_im[sa0];
                        sx1_re <= ram_re[sa1];  sx1_im <= ram_im[sa1];
                        sx2_re <= ram_re[sa2];  sx2_im <= ram_im[sa2];
                        sx3_re <= ram_re[sa3];  sx3_im <= ram_im[sa3];
                        state  <= ST_BUTTERFLY;
                    end
                    default: state <= ST_TW_ADDR;
                endcase
            end

            //--- Execute butterfly and write results back to RAM ---
            ST_BUTTERFLY: begin
                ram_re[sa0] <= bf_y0_re;  ram_im[sa0] <= bf_y0_im;
                ram_re[sa1] <= bf_y1_re;  ram_im[sa1] <= bf_y1_im;
                ram_re[sa2] <= bf_y2_re;  ram_im[sa2] <= bf_y2_im;
                ram_re[sa3] <= bf_y3_re;  ram_im[sa3] <= bf_y3_im;

                // Advance to next group or next stage
                if (group == GROUPS_PER_STAGE - 1) begin
                    if (stage == 2'd3) begin
                        // All 4 stages complete
                        state   <= ST_READ_OUT;
                        rd_addr <= 0;
                    end else begin
                        stage <= stage + 2'd1;
                        group <= 6'd0;
                        state <= ST_TW_ADDR;
                        tw_idx <= 2'd0;
                        tw_rom_addr <= 8'd0;
                    end
                end else begin
                    group  <= group + 6'd1;
                    state  <= ST_TW_ADDR;
                    tw_idx <= 2'd0;
                    tw_rom_addr <= 8'd0;
                end
            end

            //--- Read out 256 results sequentially ---
            ST_READ_OUT: begin
                data_out_valid <= 1'b1;
                if (data_out_ready) begin
                    data_out_re <= ram_re[rd_addr];
                    data_out_im <= ram_im[rd_addr];
                    if (rd_addr == N - 1) begin
                        state <= ST_FINISH;
                    end else begin
                        rd_addr <= rd_addr + 1;
                    end
                end
            end

            //--- Finish: assert done, return to idle ---
            ST_FINISH: begin
                data_out_valid <= 1'b0;
                done           <= 1'b1;
                state          <= ST_IDLE;
            end

            default: state <= ST_IDLE;
            endcase
        end
    end

endmodule