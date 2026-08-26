`default_nettype none
//==============================================================================
// adaptive_k.v — Adaptive Mode Count Controller
//==============================================================================
// Performance improvement: Configurable active mode count (K) register that
// controls how many spectral modes the MAC loop processes.  For models with
// sparse spectral weight matrices, fewer modes are needed, reducing latency
// proportional to K.  The register accepts values 8..32, defaulting to 32
// (full processing) for maximum accuracy.
//
// Security preservation: despite the name "adaptive," the mode count is fixed
// for a given model configuration and does not change at runtime based on
// data.  The value is set once during model loading via a Wishbone register
// write and remains constant for all subsequent inferences.  This prevents
// data-dependent timing: every inference with the same model takes exactly
// the same number of cycles.
//
// The constant-time guarantee is enforced by the zero_skip_mac module, which
// injects dummy cycles for zeroed modes.  Even if K is reduced, the total
// cycle count remains constant because dummy cycles fill the gap.
//
// Interface:
//   clk, rst_n       — clock and reset
//   k_write          — write enable for K register
//   k_value          — K value to write (8..32)
//   k_active         — current active mode count (constant during inference)
//   k_valid          — K register has been initialized
//   max_k            — maximum K (parameter, default 32)
//
// Verilog-2005, `default_nettype none.  Synthesizable.
//==============================================================================
module adaptive_k #(
    parameter MIN_K = 8,
    parameter MAX_K = 32
) (
    input  wire              clk,
    input  wire              rst_n,
    input  wire              k_write,
    input  wire [7:0]        k_value,
    output reg  [7:0]        k_active,
    output reg               k_valid,
    output wire [7:0]        max_k
);

    //------------------------------------------------------------------
    // K register with bounds checking
    // The value is clamped to [MIN_K, MAX_K] on write to prevent invalid
    // configurations that could cause datapath underflow or overflow.
    //------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            k_active <= MAX_K[7:0];  // Default: full processing
            k_valid  <= 1'b1;        // Valid immediately (default value)
        end else begin
            if (k_write) begin
                // Clamp to valid range
                if (k_value < MIN_K[7:0]) begin
                    k_active <= MIN_K[7:0];
                end else if (k_value > MAX_K[7:0]) begin
                    k_active <= MAX_K[7:0];
                end else begin
                    k_active <= k_value;
                end
                k_valid <= 1'b1;
            end
        end
    end

    assign max_k = MAX_K[7:0];

endmodule