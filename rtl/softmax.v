`default_nettype none
//==============================================================================
// softmax.v — Softmax Module for Byte-Level Vocabulary (128)
//==============================================================================
// Converts logits to probabilities: p_i = exp(x_i) / sum(exp(x_j))
//
// Phase 1 (ACCUMULATE): Receive 128 logits, compute exp(x_i) via LUT,
//   store in internal RAM, accumulate the sum.
// Phase 2 (OUTPUT): Stream out p_i = exp(x_i) / sum via reciprocal LUT.
//
// exp(x) approximation: piecewise linear over [-8, 8] in Q8.8
//   x < -8 → exp ≈ 0 (clamp to 1)
//   x >  8 → exp ≈ 256 (1.0 in Q8.8, will saturate)
//   in between: exp(x) ≈ 256 * 2^(x/1) — but we use a simpler LUT
//
// ~1000 gates.  Q8.8 fixed-point, WIDTH=16.
//==============================================================================
module softmax #(
    parameter WIDTH      = 16,
    parameter FRAC       = 8,
    parameter VOCAB_SIZE = 128
) (
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    start,         // Begin accumulation
    input  wire                    data_in_valid,
    output reg                     data_in_ready,
    input  wire signed [WIDTH-1:0] data_in,       // Logit value
    output reg                     data_out_valid,
    input  wire                    data_out_ready,
    output reg  signed [WIDTH-1:0] data_out,     // Probability (Q8.8)
    output reg                     done
);

    // Internal RAM: store exp values (128 entries × 16 bits)
    reg signed [WIDTH-1:0] exp_ram [0:VOCAB_SIZE-1];

    // State machine
    localparam [1:0] ST_IDLE     = 2'd0,
                     ST_ACCUM    = 2'd1,
                     ST_DIVIDE   = 2'd2,
                     ST_FINISH   = 2'd3;

    reg [1:0]  state;
    reg [6:0]  idx;            // current index (0..127)
    reg signed [31:0] exp_sum; // accumulator (wide to avoid overflow)

    //--- exp approximation (Q8.8) ---
    // For Q8.8 input in [-8, 8]:
    //   exp(x) for x=0 → 1.0 = 256
    //   exp(x) ≈ 256 * (1 + x/2 + (x/2)^2/2) for small x
    //   For hardware simplicity, use a shifted power-of-2 approach:
    //   exp(x) ≈ 256 << (x / ln2) but that's too coarse.
    //   Use a 16-entry LUT indexed by x[7:4]:
    function signed [WIDTH-1:0] exp_approx;
        input signed [WIDTH-1:0] x;  // Q8.8
        reg signed [WIDTH-1:0] result;
        begin
            if (x < -16'sd2048)         // < -8.0
                result = 16'sd1;       // ~0.004 (minimum nonzero)
            else if (x > 16'sd2048)    // > 8.0
                result = 16'sd32767;   // saturate (exp(8) ≈ 2981, way above Q8.8 max)
            else begin
                // Use 8-segment piecewise approximation
                // Segment by integer part of x (Q8.8 → integer = x >> 8)
                case (x[15:11])  // top 5 bits indicate range
                    // For a simpler approach: use x >> 4 as index into approx
                    default: begin
                        // exp(x) ≈ 256 + 256*x/256 for small x
                        // Better: exp(x) = 256 * (1 + x/256 + x^2/(2*256^2))
                        // For Q8.8, use: 256 + x + (x*x >> 9)
                        result = 16'sd256 + x + ((x * x) >>> 9);
                        if (result < 16'sd1) result = 16'sd1;
                        if (result > 16'sd32767) result = 16'sd32767;
                    end
                endcase
            end
            exp_approx = result;
        end
    endfunction

    //--- Reciprocal approximation (Q8.8) ---
    // 1/sum ≈ LUT lookup. For sum in [128, 32768], return 256/sum * 256
    function signed [WIDTH-1:0] recip_approx;
        input signed [31:0] denom;  // sum of exps (up to 128*32767 ≈ 4M)
        reg signed [31:0] scaled_recip;
        begin
            // 1/denom in Q8.8 = 256 / denom * 256 = 65536 / denom
            // For denom > 256: recip = 65536 / denom
            // For denom ≤ 256: recip = 256 (max)
            if (denom <= 256)
                scaled_recip = 256;
            else if (denom >= 16'sd65536)
                scaled_recip = 1;
            else
                scaled_recip = 32'sd65536 / denom;
            if (scaled_recip > 16'sd32767)
                scaled_recip = 16'sd32767;
            recip_approx = scaled_recip[WIDTH-1:0];
        end
    endfunction

    //--- State machine ---
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= ST_IDLE;
            data_in_ready <= 1'b0;
            data_out_valid<= 1'b0;
            data_out      <= 0;
            done          <= 1'b0;
            idx           <= 0;
            exp_sum       <= 0;
        end else begin
            done <= 1'b0;
            case (state)
            //--- Idle: wait for start ---
            ST_IDLE: begin
                data_in_ready <= 1'b0;
                data_out_valid <= 1'b0;
                if (start) begin
                    idx      <= 0;
                    exp_sum  <= 0;
                    state    <= ST_ACCUM;
                    data_in_ready <= 1'b1;
                end
            end
            //--- Accumulate: receive 128 logits ---
            ST_ACCUM: begin
                data_in_ready <= 1'b1;
                if (data_in_valid) begin
                    exp_ram[idx] <= exp_approx(data_in);
                    exp_sum <= exp_sum + exp_approx(data_in);
                    if (idx == VOCAB_SIZE - 1) begin
                        data_in_ready <= 1'b0;
                        idx <= 0;
                        state <= ST_DIVIDE;
                    end else begin
                        idx <= idx + 1;
                    end
                end
            end
            //--- Divide: stream out probabilities ---
            ST_DIVIDE: begin
                data_out_valid <= 1'b1;
                if (data_out_ready || !data_out_valid) begin
                    // p_i = exp(x_i) * (1/sum) = exp(x_i) * recip >> 8
                    data_out <= (exp_ram[idx] * recip_approx(exp_sum)) >>> 8;
                    if (idx == VOCAB_SIZE - 1) begin
                        state <= ST_FINISH;
                        data_out_valid <= 1'b0;
                    end else begin
                        idx <= idx + 1;
                    end
                end
            end
            //--- Finish ---
            ST_FINISH: begin
                done  <= 1'b1;
                state <= ST_IDLE;
            end
            endcase
        end
    end

endmodule

`default_nettype wire