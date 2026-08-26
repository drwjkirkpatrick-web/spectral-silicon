`default_nettype none
//==============================================================================
// fft_stage.v — Pipelined FFT Stage with Valid/Stall Handshaking
//==============================================================================
// A single radix-4 pipelined FFT stage.  Processes 4 complex samples at a time
// through a radix-4 butterfly, applies twiddle factors from ROM, and provides
// valid/stall handshaking for pipeline flow control.
//
// Pipeline structure:
//   Input buffer (4 complex samples) → Butterfly → Twiddle multiply → Output buffer
//
// The stage operates on groups of 4 complex values.  In a 256-point FFT with
// 4 radix-4 stages, each stage processes 64 groups of 4.
//
// Parameters:
//   STAGE_IDX : 0..3 for the 4 stages of a 256-point FFT (256 = 4^4)
//   N         : FFT size (256)
//   WIDTH     : Data width (16 bits, Q8.8)
//
// Handshaking:
//   valid_in  / stall_out : upstream provides valid data; we assert stall_out
//                           when our input buffer is full and we can't accept.
//   valid_out / stall_in  : downstream provides stall_in; we hold valid_out
//                           when data is ready.
//
// Prompt 14 specification.
//==============================================================================
module fft_stage #(
    parameter STAGE_IDX = 0,       // 0..3
    parameter N         = 256,      // FFT size
    parameter WIDTH     = 16,      // Q8.8 data width
    parameter FRAC      = 8
) (
    input  wire                    clk,
    input  wire                    rst_n,

    // Input interface (4 complex samples: x0, x1, x2, x3)
    input  wire                    valid_in,
    output wire                    stall_out,
    input  wire signed [WIDTH-1:0] x0_re, x0_im,
    input  wire signed [WIDTH-1:0] x1_re, x1_im,
    input  wire signed [WIDTH-1:0] x2_re, x2_im,
    input  wire signed [WIDTH-1:0] x3_re, x3_im,

    // Output interface (4 complex samples: y0, y1, y2, y3)
    output reg                     valid_out,
    input  wire                    stall_in,
    output reg  signed [WIDTH-1:0] y0_re, y0_im,
    output reg  signed [WIDTH-1:0] y1_re, y1_im,
    output reg  signed [WIDTH-1:0] y2_re, y2_im,
    output reg  signed [WIDTH-1:0] y3_re, y3_im,

    // Twiddle ROM interface (this stage provides addresses)
    // In a complete design, twiddle ROMs are instantiated inside or shared.
    // For this module we instantiate twiddle ROM internally.
    input  wire [7:0]              twiddle_base_addr  // Optional: base addr override
);

    //----------------------------------------------------------------------
    // Twiddle factor addresses for this stage
    // For radix-4, stage s operates on groups of 4.  The twiddle stride for
    // stage s is N / 4^(s+1).  Stage 0 stride = 64, stage 1 = 16, etc.
    // We use a simple counter to generate addresses.
    //----------------------------------------------------------------------
    // Stride = N / (4**(STAGE_IDX+1))
    // For N=256: stage0=64, stage1=16, stage2=4, stage3=1
    // We need twiddle indices: 0, k*stride, 2*k*stride, 3*k*stride
    // for k = group counter (0..N/4-1)

    localparam GROUPS = N / 4;   // 64 groups per stage

    // Group counter
    reg [7:0] group_cnt;

    // Twiddle addresses: W1 = group*stride, W2 = 2*group*stride, W3 = 3*group*stride
    // stride = N / 4^(STAGE_IDX+1)
    // For N=256, stage 0: stride=64, stage 1: stride=16, stage 2: stride=4, stage 3: stride=1
    // We compute stride as a function of STAGE_IDX.
    // Use a function for clarity.
    function [7:0] stride_calc;
        input [7:0] s_idx;
        begin
            case (s_idx)
                0: stride_calc = 64;  // 256/4
                1: stride_calc = 16;  // 256/16
                2: stride_calc = 4;   // 256/64
                3: stride_calc = 1;   // 256/256
                default: stride_calc = 1;
            endcase
        end
    endfunction

    wire [7:0] stride = stride_calc(STAGE_IDX[7:0]);
    wire [7:0] w1_addr = (group_cnt * stride) & 8'hFF;
    wire [7:0] w2_addr = (group_cnt * stride * 2) & 8'hFF;
    wire [7:0] w3_addr = (group_cnt * stride * 3) & 8'hFF;

    //----------------------------------------------------------------------
    // Twiddle ROM instances
    //----------------------------------------------------------------------
    wire signed [WIDTH-1:0] w1_cos, w1_sin;
    wire signed [WIDTH-1:0] w2_cos, w2_sin;
    wire signed [WIDTH-1:0] w3_cos, w3_sin;

    twiddle_rom #(
        .N(N),
        .WIDTH(WIDTH),
        .ADDR_BITS(8),
        .COS_FILE("twiddle_data/twiddle_cos_256.hex"),
        .SIN_FILE("twiddle_data/twiddle_sin_256.hex")
    ) u_tw1 (
        .clk(clk),
        .addr(w1_addr),
        .cos_out(w1_cos),
        .sin_out(w1_sin)
    );

    twiddle_rom u_tw2 (
        .clk(clk),
        .addr(w2_addr),
        .cos_out(w2_cos),
        .sin_out(w2_sin)
    );

    twiddle_rom u_tw3 (
        .clk(clk),
        .addr(w3_addr),
        .cos_out(w3_cos),
        .sin_out(w3_sin)
    );

    //----------------------------------------------------------------------
    // Pipeline registers: 3 stages
    //   Stage 0: Input latch (capture 4 complex inputs)
    //   Stage 1: Butterfly + twiddle address issue
    //   Stage 2: Twiddle multiply output latch
    //----------------------------------------------------------------------

    // Input buffer registers
    reg signed [WIDTH-1:0] ib0_re, ib0_im, ib1_re, ib1_im;
    reg signed [WIDTH-1:0] ib2_re, ib2_im, ib3_re, ib3_im;
    reg                   ib_valid;

    // Butterfly outputs (combinational)
    wire signed [WIDTH-1:0] bf0_re, bf0_im, bf1_re, bf1_im;
    wire signed [WIDTH-1:0] bf2_re, bf2_im, bf3_re, bf3_im;

    butterfly4 #(
        .WIDTH(WIDTH),
        .FRAC(FRAC)
    ) u_bf (
        .x0_re(ib0_re), .x0_im(ib0_im),
        .x1_re(ib1_re), .x1_im(ib1_im),
        .x2_re(ib2_re), .x2_im(ib2_im),
        .x3_re(ib3_re), .x3_im(ib3_im),
        .w1_re(w1_cos), .w1_im(w1_sin),
        .w2_re(w2_cos), .w2_im(w2_sin),
        .w3_re(w3_cos), .w3_im(w3_sin),
        .y0_re(bf0_re), .y0_im(bf0_im),
        .y1_re(bf1_re), .y1_im(bf1_im),
        .y2_re(bf2_re), .y2_im(bf2_im),
        .y3_re(bf3_re), .y3_im(bf3_im)
    );

    // Output buffer registers
    reg signed [WIDTH-1:0] ob0_re, ob0_im, ob1_re, ob1_im;
    reg signed [WIDTH-1:0] ob2_re, ob2_im, ob3_re, ob3_im;
    reg                   ob_valid;

    // Pipeline valid bits
    reg pipe1_valid;  // butterfly stage valid

    //----------------------------------------------------------------------
    // Pipeline control logic
    //----------------------------------------------------------------------
    // Stall computation: stall_out asserted when input buffer is full
    // and we can't move data forward (output buffer also full).
    wire input_can_accept = !ib_valid || (ob_valid && !stall_in);
    wire output_can_push  = ib_valid && (!ob_valid || !stall_in);

    assign stall_out = !input_can_accept;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset all pipeline registers
            ib_valid <= 1'b0;
            pipe1_valid <= 1'b0;
            ob_valid <= 1'b0;
            valid_out <= 1'b0;
            group_cnt <= 8'd0;

            ib0_re <= 0; ib0_im <= 0;
            ib1_re <= 0; ib1_im <= 0;
            ib2_re <= 0; ib2_im <= 0;
            ib3_re <= 0; ib3_im <= 0;

            ob0_re <= 0; ob0_im <= 0;
            ob1_re <= 0; ob1_im <= 0;
            ob2_re <= 0; ob2_im <= 0;
            ob3_re <= 0; ob3_im <= 0;

            y0_re <= 0; y0_im <= 0;
            y1_re <= 0; y1_im <= 0;
            y2_re <= 0; y2_im <= 0;
            y3_re <= 0; y3_im <= 0;
        end else begin
            //--- Output stage: push to output if valid and not stalled ---
            if (ob_valid && !stall_in) begin
                y0_re <= ob0_re; y0_im <= ob0_im;
                y1_re <= ob1_re; y1_im <= ob1_im;
                y2_re <= ob2_re; y2_im <= ob2_im;
                y3_re <= ob3_re; y3_im <= ob3_im;
                valid_out <= 1'b1;
                ob_valid <= 1'b0;  // Free output buffer
            end else if (!ob_valid) begin
                valid_out <= 1'b0;  // No data to push
            end

            //--- Butterfly → Output buffer: move butterfly result to output ---
            if (pipe1_valid && !ob_valid) begin
                // Twiddle ROM has 1-cycle latency, butterfly is combinational.
                // The twiddle values were registered in ROM, so they arrive
                // one cycle after address issue.  We capture the butterfly
                // output (which uses registered twiddle values) here.
                ob0_re <= bf0_re; ob0_im <= bf0_im;
                ob1_re <= bf1_re; ob1_im <= bf1_im;
                ob2_re <= bf2_re; ob2_im <= bf2_im;
                ob3_re <= bf3_re; ob3_im <= bf3_im;
                ob_valid <= 1'b1;
                pipe1_valid <= 1'b0;
            end

            //--- Input → Butterfly: latch input into butterfly stage ---
            if (ib_valid && !pipe1_valid && (!ob_valid || !stall_in)) begin
                pipe1_valid <= 1'b1;
                // Issue twiddle addresses (combinational, registered in ROM)
                // The butterfly sees the input buffer values and the
                // registered twiddle outputs from the previous cycle.
            end

            //--- Input capture: accept new data when buffer is free ---
            if (valid_in && input_can_accept) begin
                ib0_re <= x0_re; ib0_im <= x0_im;
                ib1_re <= x1_re; ib1_im <= x1_im;
                ib2_re <= x2_re; ib2_im <= x2_im;
                ib3_re <= x3_re; ib3_im <= x3_im;
                ib_valid <= 1'b1;

                // Increment group counter for twiddle addressing
                if (group_cnt == GROUPS - 1)
                    group_cnt <= 8'd0;
                else
                    group_cnt <= group_cnt + 8'd1;
            end else if (input_can_accept && !valid_in) begin
                ib_valid <= 1'b0;  // No new data, bubble
            end
        end
    end

endmodule