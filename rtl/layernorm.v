`default_nettype none
//==============================================================================
// layernorm.v — Layer Normalization Module (d_model=64)
//==============================================================================
// Normalizes across D_MODEL channels for each token position:
//   mean = sum(x_i) / D
//   var  = sum((x_i - mean)^2) / D
//   out_i = gamma_i * (x_i - mean) / sqrt(var + eps) + beta_i
//
// Phase 1 (ACCUMULATE): Receive 64 real values sequentially, store in RAM,
//   accumulate sum and sum_of_squares.
// Phase 2 (NORMALIZE): Compute mean and variance, stream out normalized +
//   affine-transformed values using reciprocal and rsqrt lookup tables.
//
// ~1500 gates.  Q8.8 fixed-point, WIDTH=16, D_MODEL=64.
//==============================================================================
module layernorm #(
    parameter WIDTH   = 16,
    parameter FRAC    = 8,
    parameter D_MODEL = 64
) (
    input  wire                    clk,
    input  wire                    rst_n,

    // Gamma/Beta weight loading (host writes at boot)
    input  wire                    gamma_we,
    input  wire [5:0]              gamma_addr,
    input  wire signed [WIDTH-1:0] gamma_data,
    input  wire                    beta_we,
    input  wire [5:0]              beta_addr,
    input  wire signed [WIDTH-1:0] beta_data,

    // Data interface
    input  wire                    start,       // Begin accumulation phase
    input  wire                    data_in_valid,
    output reg                     data_in_ready,
    input  wire signed [WIDTH-1:0] data_in,     // Real input values

    output reg                     data_out_valid,
    input  wire                    data_out_ready,
    output reg  signed [WIDTH-1:0] data_out,   // Normalized output
    output reg                     done
);

    //--- Gamma and beta parameter storage ---
    reg signed [WIDTH-1:0] gamma_arr [0:D_MODEL-1];
    reg signed [WIDTH-1:0] beta_arr  [0:D_MODEL-1];

    //--- Input value RAM (store for normalize phase) ---
    reg signed [WIDTH-1:0] val_ram [0:D_MODEL-1];

    //--- State machine ---
    localparam [2:0] ST_IDLE    = 3'd0,
                     ST_ACCUM   = 3'd1,
                     ST_COMPUTE = 3'd2,
                     ST_NORM    = 3'd3,
                     ST_FINISH  = 3'd4;

    reg [2:0] state;
    reg [5:0] idx;              // 0..63

    // Accumulators (wide to prevent overflow)
    reg signed [31:0] sum_val;      // sum of all values
    reg signed [31:0] sum_sq_val;  // sum of squares

    // Computed statistics
    reg signed [WIDTH-1:0] mean_r;     // mean in Q8.8
    reg signed [WIDTH-1:0] rsqrt_var;  // 1/sqrt(var+eps) in Q8.8

    //--- Weight loading (synchronous) ---
    integer i;
    always @(posedge clk) begin
        if (gamma_we) gamma_arr[gamma_addr] <= gamma_data;
        if (beta_we)  beta_arr[beta_addr]  <= beta_data;
    end

    //--- Reciprocal square root LUT ---
    // rsqrt(x) for x in Q8.8: returns 1/sqrt(x) in Q8.8
    // For var in [0.001, 16] → rsqrt in [0.25, 31.6]
    function signed [WIDTH-1:0] rsqrt_lut;
        input signed [WIDTH-1:0] var_q88;  // variance in Q8.8
        reg signed [31:0] v;
        begin
            v = var_q88;
            if (v <= 0 || v < 1)
                rsqrt_lut = 16'sd32767;   // 1/sqrt(~0) → large
            else if (v >= 16'sd4096)      // var >= 16.0
                rsqrt_lut = 16'sd64;      // 1/sqrt(16) = 0.25 → 64 in Q8.8
            else begin
                // Newton iteration: r = 1/sqrt(v)
                // Start: r0 = 256 / sqrt(v/256) = 256 * rsqrt(v/256)
                // Use integer approximation: rsqrt(v) ≈ 256 / isqrt(v/256)
                // For simplicity, use a piecewise approximation:
                // rsqrt(var) where var is Q8.8:
                //   result = 256 / sqrt(var / 256) = 256 * 256 / isqrt(var)
                //   ≈ 65536 / sqrt(var)
                rsqrt_lut = 65536 / (v > 0 ? v : 1);
                if (rsqrt_lut > 16'sd32767)
                    rsqrt_lut = 16'sd32767;
            end
        end
    endfunction

    //--- Main state machine ---
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state          <= ST_IDLE;
            data_in_ready  <= 1'b0;
            data_out_valid <= 1'b0;
            data_out       <= 0;
            done           <= 1'b0;
            idx            <= 0;
            sum_val        <= 0;
            sum_sq_val     <= 0;
            mean_r         <= 0;
            rsqrt_var      <= 0;
        end else begin
            done <= 1'b0;
            case (state)
            //--- Idle: wait for start ---
            ST_IDLE: begin
                data_in_ready  <= 1'b0;
                data_out_valid <= 1'b0;
                if (start) begin
                    idx       <= 0;
                    sum_val   <= 0;
                    sum_sq_val<= 0;
                    state     <= ST_ACCUM;
                    data_in_ready <= 1'b1;
                end
            end
            //--- Accumulate: receive D_MODEL values ---
            ST_ACCUM: begin
                data_in_ready <= 1'b1;
                if (data_in_valid) begin
                    val_ram[idx] <= data_in;
                    sum_val    <= sum_val + data_in;
                    sum_sq_val <= sum_sq_val + (data_in * data_in);
                    if (idx == D_MODEL - 1) begin
                        data_in_ready <= 1'b0;
                        state <= ST_COMPUTE;
                        idx   <= 0;
                    end else begin
                        idx <= idx + 1;
                    end
                end
            end
            //--- Compute: mean = sum/D, var = sum_sq/D - mean^2 ---
            ST_COMPUTE: begin
                // mean = sum_val / 64 = sum_val >> 6
                mean_r <= sum_val >>> 6;

                // var = (sum_sq / 64) - mean^2
                // = (sum_sq >>> 6) - (mean_r * mean_r >>> 8)
                // We compute this in the next state for timing
                begin : compute_var
                    reg signed [31:0] mean_sq;
                    reg signed [31:0] var_full;
                    mean_sq  = (sum_val >>> 6) * (sum_val >>> 6);
                    var_full = (sum_sq_val >>> 6) - (mean_sq >>> 8);
                    if (var_full < 1) var_full = 1;  // eps clamp
                    rsqrt_var <= rsqrt_lut(var_full[WIDTH-1:0]);
                end
                state <= ST_NORM;
                idx   <= 0;
            end
            //--- Normalize: stream out normalized values ---
            ST_NORM: begin
                data_out_valid <= 1'b1;
                if (data_out_ready || !data_out_valid) begin
                    // out_i = gamma_i * (x_i - mean) * rsqrt_var + beta_i
                    // = gamma_i * ((x_i - mean) * rsqrt_var >> 8) + beta_i
                    begin : norm_calc
                        reg signed [31:0] centered;
                        reg signed [31:0] normalized;
                        reg signed [31:0] affine;
                        centered  = val_ram[idx] - mean_r;
                        normalized = (centered * rsqrt_var) >>> 8;
                        affine    = (gamma_arr[idx] * normalized) >>> 8;
                        affine    = affine + beta_arr[idx];
                        // Saturate to 16-bit
                        if (affine > 16'sd32767)
                            data_out <= 16'sd32767;
                        else if (affine < -16'sd32768)
                            data_out <= -16'sd32768;
                        else
                            data_out <= affine[WIDTH-1:0];
                    end
                    if (idx == D_MODEL - 1) begin
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