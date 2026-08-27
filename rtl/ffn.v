`default_nettype none
//==============================================================================
// ffn.v — Feed-Forward Network (d_model=64, d_ffn=128)
//==============================================================================
// FFN(x) = W2 · GELU(W1 · x) + b2
//
// Two-phase computation:
//   Phase 1 (EXPAND): Compute 128 intermediate values h[j] = sum_i(W1[j][i] * x[i])
//     for j=0..127, then apply GELU.
//   Phase 2 (PROJECT): Compute 64 output values y[j] = sum_i(W2[j][i] * h[i])
//     for j=0..63.
//
// Uses a single MAC unit + internal RAM for intermediate values.
// Weight loading via Wishbone-driven interface.
//
// Weight storage: W1 (128×64) + W2 (64×128) = 16,384 weights = 32 KB
//
// ~3000 gates + 32 KB weight storage.  Q8.8 fixed-point, WIDTH=16.
//==============================================================================
module ffn #(
    parameter WIDTH  = 16,
    parameter FRAC   = 8,
    parameter D_MODEL = 64,
    parameter D_FFN   = 128
) (
    input  wire                    clk,
    input  wire                    rst_n,

    // Weight loading (host writes at boot)
    input  wire                    weight_we,
    input  wire [13:0]             weight_addr,  // see address map below
    input  wire signed [WIDTH-1:0] weight_data,

    // Input data interface (d_model values streamed in)
    input  wire                    data_in_valid,
    output reg                     data_in_ready,
    input  wire signed [WIDTH-1:0] data_in,

    // Output data interface (d_model values streamed out)
    output reg                     data_out_valid,
    input  wire                    data_out_ready,
    output reg  signed [WIDTH-1:0] data_out,
    output reg                     done
);

    //--- Weight address map ---
    // W1: 0 .. (D_FFN * D_MODEL - 1)          = 0..8191
    // W2: D_FFN*D_MODEL .. (2*D_FFN*D_MODEL-1) = 8192..16383

    localparam W1_BASE = 0;
    localparam W2_BASE = D_FFN * D_MODEL;  // 8192

    //--- Weight storage ---
    reg signed [WIDTH-1:0] weights [0:2*D_FFN*D_MODEL-1];

    //--- Input buffer ---
    reg signed [WIDTH-1:0] in_buf [0:D_MODEL-1];

    //--- Intermediate values (after GELU) ---
    reg signed [WIDTH-1:0] hidden [0:D_FFN-1];

    //--- State machine ---
    localparam [3:0] ST_IDLE     = 4'd0,
                     ST_LOAD_IN  = 4'd1,
                     ST_EXPAND   = 4'd2,   // MAC for W1
                     ST_GELU     = 4'd3,    // Apply GELU to all hidden
                     ST_PROJECT  = 4'd4,   // MAC for W2
                     ST_STREAM   = 4'd5,   // Wait for downstream
                     ST_FINISH   = 4'd6;

    reg [3:0] state;
    reg [6:0] out_idx;     // 0..127 (expand) or 0..63 (project)
    reg [5:0] in_idx;      // 0..63 (inner MAC loop)
    reg signed [31:0] acc; // MAC accumulator

    //--- GELU approximation (same as gelu_silu.v) ---
    function signed [WIDTH-1:0] gelu_approx;
        input signed [WIDTH-1:0] x;
        begin
            if (x < -16'sd768)         // < -3.0
                gelu_approx = 0;
            else if (x >= 16'sd768)    // >= 3.0
                gelu_approx = x;
            else if (x < -16'sd384)    // [-3, -1.5)
                gelu_approx = 0;
            else if (x < -16'sd128)    // [-1.5, -0.5)
                gelu_approx = (x >>> 4) + 16'sd8;
            else if (x < 16'sd128)     // [-0.5, 0.5)
                gelu_approx = x;
            else if (x < 16'sd384)     // [0.5, 1.5)
                gelu_approx = (x * 16'sd205) >>> 8;
            else                       // [1.5, 3.0)
                gelu_approx = x;
        end
    endfunction

    //--- Weight write ---
    always @(posedge clk) begin
        if (weight_we) begin
            weights[weight_addr] <= weight_data;
        end
    end

    //--- Main state machine ---
    integer k;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state          <= ST_IDLE;
            data_in_ready  <= 1'b0;
            data_out_valid <= 1'b0;
            data_out       <= 0;
            done           <= 1'b0;
            out_idx        <= 0;
            in_idx         <= 0;
            acc            <= 0;
        end else begin
            done <= 1'b0;
            case (state)
            //--- Idle: wait for input ---
            ST_IDLE: begin
                data_in_ready  <= 1'b1;
                data_out_valid <= 1'b0;
                if (data_in_valid) begin
                    in_buf[0] <= data_in;
                    in_idx   <= 1;
                    state    <= ST_LOAD_IN;
                end
            end
            //--- Load input: receive remaining 63 values ---
            ST_LOAD_IN: begin
                data_in_ready <= 1'b1;
                if (data_in_valid) begin
                    in_buf[in_idx] <= data_in;
                    if (in_idx == D_MODEL - 1) begin
                        data_in_ready <= 1'b0;
                        state    <= ST_EXPAND;
                        out_idx  <= 0;
                        in_idx   <= 0;
                        acc      <= 0;
                    end else begin
                        in_idx <= in_idx + 1;
                    end
                end
            end
            //--- Expand: h[j] = sum_i(W1[j][i] * x[i]) ---
            ST_EXPAND: begin
                acc <= acc + (weights[W1_BASE + out_idx*D_MODEL + in_idx] * in_buf[in_idx]);
                if (in_idx == D_MODEL - 1) begin
                    // Store GELU(acc + final product)
                    hidden[out_idx] <= gelu_approx(
                        (acc + (weights[W1_BASE + out_idx*D_MODEL + in_idx] * in_buf[in_idx])) >>> FRAC
                    );
                    in_idx <= 0;
                    acc   <= 0;
                    if (out_idx == D_FFN - 1) begin
                        state   <= ST_PROJECT;
                        out_idx <= 0;
                        in_idx  <= 0;
                        acc     <= 0;
                    end else begin
                        out_idx <= out_idx + 1;
                    end
                end else begin
                    in_idx <= in_idx + 1;
                end
            end
            //--- Project: y[j] = sum_i(W2[j][i] * h[i]) ---
            ST_PROJECT: begin
                acc <= acc + (weights[W2_BASE + out_idx*D_FFN + in_idx] * hidden[in_idx]);
                if (in_idx == D_FFN - 1) begin
                    // Output this value
                    data_out <= (acc + (weights[W2_BASE + out_idx*D_FFN + in_idx] * hidden[in_idx])) >>> FRAC;
                    data_out_valid <= 1'b1;
                    in_idx <= 0;
                    acc   <= 0;
                    if (out_idx == D_MODEL - 1) begin
                        state <= ST_FINISH;
                    end else begin
                        out_idx <= out_idx + 1;
                        state   <= ST_STREAM;
                    end
                end else begin
                    in_idx <= in_idx + 1;
                end
            end
            //--- Stream: wait for downstream to accept ---
            ST_STREAM: begin
                if (data_out_ready || !data_out_valid) begin
                    data_out_valid <= 1'b0;
                    state <= ST_PROJECT;
                end
            end
            //--- Finish ---
            ST_FINISH: begin
                if (data_out_ready || !data_out_valid) begin
                    data_out_valid <= 1'b0;
                    done  <= 1'b1;
                    state <= ST_IDLE;
                end
            end
            endcase
        end
    end

endmodule

`default_nettype wire