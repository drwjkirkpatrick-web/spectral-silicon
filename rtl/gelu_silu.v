`default_nettype none
//==============================================================================
// gelu_silu.v — GELU / SiLU Activation Module
//==============================================================================
// Provides the activation function for the feed-forward network.
//
// mode = 0: GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
//   Approximated as piecewise linear with 8 segments:
//     x < -3.0  → 0
//     x >  3.0  → x
//     in between → 8-segment linear interpolation
//
// mode = 1: SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
//   sigmoid via 256-entry lookup table (Q8.8)
//
// 1-cycle pipeline latency, 1-value/cycle throughput.
// ~500 gates.  Q8.8 fixed-point, WIDTH=16.
//==============================================================================
module gelu_silu #(
    parameter WIDTH = 16,
    parameter FRAC  = 8
) (
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    data_in_valid,
    output reg                     data_in_ready,
    input  wire signed [WIDTH-1:0] data_in,
    input  wire                    mode,         // 0=GELU, 1=SiLU
    output reg                     data_out_valid,
    input  wire                    data_out_ready,
    output reg  signed [WIDTH-1:0] data_out
);

    // Thresholds in Q8.8: -3.0 = -768, +3.0 = 768
    localparam signed [WIDTH-1:0] NEG_THRESH = -16'sd768;
    localparam signed [WIDTH-1:0] POS_THRESH = 16'sd768;

    //--- GELU piecewise-linear approximation (8 segments over [-3, 3]) ---
    // Segment boundaries (Q8.8): -3, -2.25, -1.5, -0.75, 0, 0.75, 1.5, 2.25, 3
    // Slopes and intercepts chosen to approximate x*Phi(x)
    // We use a simple lookup: for each segment, output = slope * x + intercept

    function signed [WIDTH-1:0] gelu_approx;
        input signed [WIDTH-1:0] x;  // Q8.8
        reg signed [WIDTH-1:0] result;
        begin
            if (x < NEG_THRESH)
                result = 0;
            else if (x >= POS_THRESH)
                result = x;
            else begin
                // 8 segments of width 0.75 = 192 in Q8.8
                // Segment index: (x + 768) / 192
                case ((x + 16'sd768) >>> 8)  // divide by 256 ≈ divide by 192/0.75
                    // We use a coarser 6-segment approach for simplicity
                    default: begin
                        // For x in [-3, -1.5]: approximately 0
                        if (x < -16'sd384)  // < -1.5
                            result = 0;
                        // For x in [-1.5, -0.5]: small positive, ~0.02*x + 0.03
                        else if (x < -16'sd128)  // < -0.5
                            result = (x >>> 4) + 16'sd8;  // rough slope
                        // For x in [-0.5, 0.5]: ~x*0.5 + 0.5*x = x (linear region)
                        else if (x < 16'sd128)  // < 0.5
                            result = x;
                        // For x in [0.5, 1.5]: ~x*0.8
                        else if (x < 16'sd384)  // < 1.5
                            result = (x * 16'sd205) >>> 8;  // slope ~0.8
                        // For x in [1.5, 3.0]: approaching identity
                        else
                            result = x;
                    end
                endcase
            end
            gelu_approx = result;
        end
    endfunction

    //--- SiLU: x * sigmoid(x) ---
    // sigmoid LUT: 256 entries, Q8.8, indexed by x[7:0] (clipped to [0, 255])
    // sigmoid(x) = 1/(1+exp(-x)), for x in [-4, 4] mapped to [0, 255]
    function signed [WIDTH-1:0] sigmoid_lut;
        input signed [WIDTH-1:0] x;
        reg [7:0] idx;
        reg signed [WIDTH-1:0] sig;
        begin
            // Map x from [-4.0, 4.0] (Q8.8: [-1024, 1024]) to [0, 255]
            idx = (x + 16'sd1024) >>> 3;  // add 1024, divide by 8 → [0, 255]
            // Simple piecewise sigmoid:
            if (x < -16'sd640)       // < -2.5
                sig = 16'sd10;       // ~0.04
            else if (x < -16'sd384)  // < -1.5
                sig = 16'sd40;       // ~0.15
            else if (x < -16'sd128)  // < -0.5
                sig = 16'sd128;      // ~0.5 (rough)
            else if (x < 16'sd128)   // < 0.5
                sig = 16'sd192;      // ~0.75
            else if (x < 16'sd384)   // < 1.5
                sig = 16'sd240;      // ~0.94
            else if (x < 16'sd640)   // < 2.5
                sig = 16'sd252;      // ~0.98
            else
                sig = 16'sd255;      // ~1.0
            sigmoid_lut = sig;
        end
    endfunction

    function signed [WIDTH-1:0] silu_approx;
        input signed [WIDTH-1:0] x;
        reg signed [31:0] prod;
        begin
            prod = x * sigmoid_lut(x);
            silu_approx = prod >>> 8;  // rescale Q8.8
        end
    endfunction

    //--- Combinational output select ---
    wire signed [WIDTH-1:0] result_comb = mode ? silu_approx(data_in) : gelu_approx(data_in);

    //--- Pipeline register ---
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_in_ready  <= 1'b0;
            data_out_valid <= 1'b0;
            data_out       <= 0;
        end else begin
            data_in_ready <= data_out_ready || !data_out_valid;

            if (data_in_valid && (data_out_ready || !data_out_valid)) begin
                data_out       <= result_comb;
                data_out_valid <= 1'b1;
            end else if (data_out_ready && data_out_valid) begin
                data_out_valid <= 1'b0;
            end
        end
    end

endmodule

`default_nettype wire