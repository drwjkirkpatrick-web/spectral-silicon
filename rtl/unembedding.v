`default_nettype none
//==============================================================================
// unembedding.v — Unembedding (Logits Projection) Module
//==============================================================================
// Projects from d_model=64 to vocab_size=128:
//   logits[j] = sum_i(W[j][i] * x[i]) >> FRAC  for j=0..127
//
// Weight storage: 128 × 64 × 16-bit = 16 KB (register file)
// Weight loading via Wishbone-driven interface.
//
// Serialized MAC: one MAC unit, cycling through 64 input dimensions for
// each of 128 output logits. At 50 MHz: 128×64 = 8,192 cycles ≈ 164 µs.
//
// ~3000 gates + 16 KB weight storage.  Q8.8 fixed-point, WIDTH=16.
//==============================================================================
module unembedding #(
    parameter WIDTH      = 16,
    parameter FRAC       = 8,
    parameter D_MODEL    = 64,
    parameter VOCAB_SIZE = 128
) (
    input  wire                    clk,
    input  wire                    rst_n,

    // Weight loading (host writes at boot)
    input  wire                    weight_we,
    input  wire [12:0]             weight_addr,  // {logit_idx[6:0], dim_idx[5:0]}
    input  wire signed [WIDTH-1:0] weight_data,

    // Input data interface (d_model values streamed in)
    input  wire                    data_in_valid,
    output reg                     data_in_ready,
    input  wire signed [WIDTH-1:0] data_in,

    // Output data interface (vocab_size logits streamed out)
    output reg                     data_out_valid,
    input  wire                    data_out_ready,
    output reg  signed [WIDTH-1:0] data_out,
    output reg                     done
);

    // Weight storage: 128 × 64 = 8192 entries × 16 bits
    reg signed [WIDTH-1:0] weights [0:VOCAB_SIZE*D_MODEL-1];

    // Input buffer: store 64 input values
    reg signed [WIDTH-1:0] in_buf [0:D_MODEL-1];

    // State machine
    localparam [2:0] ST_IDLE    = 3'd0,
                     ST_LOAD   = 3'd1,
                     ST_MAC    = 3'd2,
                     ST_STREAM = 3'd3,
                     ST_FINISH = 3'd4;

    reg [2:0] state;
    reg [6:0] logit_idx;    // 0..127 (current output logit)
    reg [5:0] dim_idx;      // 0..63 (current input dimension)
    reg signed [31:0] acc;  // MAC accumulator (wide)

    //--- Weight write (synchronous) ---
    always @(posedge clk) begin
        if (weight_we) begin
            weights[weight_addr] <= weight_data;
        end
    end

    //--- Main state machine ---
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state          <= ST_IDLE;
            data_in_ready  <= 1'b0;
            data_out_valid <= 1'b0;
            data_out       <= 0;
            done           <= 1'b0;
            logit_idx      <= 0;
            dim_idx        <= 0;
            acc            <= 0;
        end else begin
            done <= 1'b0;
            case (state)
            //--- Idle: wait for start ---
            ST_IDLE: begin
                data_in_ready <= 1'b1;
                data_out_valid<= 1'b0;
                if (data_in_valid) begin
                    // First value arriving
                    in_buf[0] <= data_in;
                    dim_idx  <= 1;
                    if (D_MODEL == 1) begin
                        state <= ST_MAC;
                        logit_idx <= 0;
                        dim_idx <= 0;
                        acc <= 0;
                    end else begin
                        state <= ST_LOAD;
                    end
                end
            end
            //--- Load: receive remaining d_model-1 input values ---
            ST_LOAD: begin
                data_in_ready <= 1'b1;
                if (data_in_valid) begin
                    in_buf[dim_idx] <= data_in;
                    if (dim_idx == D_MODEL - 1) begin
                        data_in_ready <= 1'b0;
                        state    <= ST_MAC;
                        logit_idx<= 0;
                        dim_idx  <= 0;
                        acc      <= 0;
                    end else begin
                        dim_idx <= dim_idx + 1;
                    end
                end
            end
            //--- MAC: compute logits[j] = sum_i(W[j][i] * x[i]) ---
            ST_MAC: begin
                // Accumulate one product per cycle
                acc <= acc + (weights[{logit_idx, dim_idx}] * in_buf[dim_idx]);
                if (dim_idx == D_MODEL - 1) begin
                    // This logit is done — output it
                    data_out       <= (acc + (weights[{logit_idx, dim_idx}] * in_buf[dim_idx])) >>> FRAC;
                    data_out_valid <= 1'b1;
                    dim_idx        <= 0;
                    acc            <= 0;
                    if (logit_idx == VOCAB_SIZE - 1) begin
                        state <= ST_FINISH;
                    end else begin
                        logit_idx <= logit_idx + 1;
                        state    <= ST_STREAM;
                    end
                end else begin
                    dim_idx <= dim_idx + 1;
                end
            end
            //--- Stream: wait for downstream to accept, then continue MAC ---
            ST_STREAM: begin
                if (data_out_ready || !data_out_valid) begin
                    data_out_valid <= 1'b0;
                    state <= ST_MAC;
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