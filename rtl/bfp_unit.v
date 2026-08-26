`default_nettype none
//==============================================================================
// bfp_unit.v — Block Floating-Point Scaling Unit
//==============================================================================
// Performance improvement: Block floating-point (BFP) maintains precision
// across FFT stages by tracking a shared exponent for a block of 64 samples.
// Instead of scaling every sample independently, a single 4-bit exponent
// captures the dynamic range of the entire block, normalizing all samples
// to a 12-bit mantissa.  This prevents overflow in deep FFT pipelines while
// keeping the datapath width constant.
//
// Security preservation: the exponent computation is data-independent in
// timing — the OR-tree and leading-one detector have fixed depth regardless
// of input values.  No information leaks through the scaling path.
//
// Interface:
//   clk, rst_n       — clock and active-low reset
//   samples_in       — 64 samples of WIDTH bits each (flattened to a single
//                      packed bus: N*WIDTH bits)
//   valid_in         — input data valid
//   mantissa_out     — 64 mantissas of MANT_W bits (flattened: N*MANT_W bits)
//   exponent_out     — 4-bit shared exponent
//   valid_out        — output valid (1 cycle after input)
//
// Verilog-2005, `default_nettype none.  Synthesizable.
//==============================================================================
module bfp_unit #(
    parameter WIDTH    = 16,     // Input sample width
    parameter N        = 64,     // Number of samples per block
    parameter MANT_W   = 12,     // Output mantissa width
    parameter EXP_W    = 4       // Exponent width (4 bits → 0..15)
) (
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire [N*WIDTH-1:0]            samples_in,    // Flattened: 64×16 = 1024 bits
    input  wire                          valid_in,
    output reg  [N*MANT_W-1:0]            mantissa_out,  // Flattened: 64×12 = 768 bits
    output reg  [EXP_W-1:0]              exponent_out,
    output reg                           valid_out
);

    //------------------------------------------------------------------
    // Step 1: Find maximum absolute value across all N samples
    // Uses an OR-tree on the absolute values — the bit-OR of all |x|
    // gives the position of the most significant set bit across the block.
    //------------------------------------------------------------------
    // Compute |x| for each sample (extract from flattened input)
    wire signed [WIDTH-1:0] sample_val [0:N-1];
    wire [WIDTH-1:0]        abs_val [0:N-1];

    genvar gi;
    generate
        for (gi = 0; gi < N; gi = gi + 1) begin : abs_gen
            // Extract sample gi from flattened input
            assign sample_val[gi] = samples_in[gi*WIDTH +: WIDTH];
            // Absolute value
            assign abs_val[gi] = sample_val[gi][WIDTH-1]
                                ? (~sample_val[gi] + 1'b1)
                                : sample_val[gi];
        end
    endgenerate

    // OR all absolute values together to find the max bit position
    reg [WIDTH-1:0] or_result;

    integer i;
    always @(*) begin
        or_result = {WIDTH{1'b0}};
        for (i = 0; i < N; i = i + 1) begin
            or_result = or_result | abs_val[i];
        end
    end

    //------------------------------------------------------------------
    // Step 2: Leading-one detector on or_result
    // The exponent is the number of leading zeros, capped at EXP_W bits.
    // This determines how much to shift left to normalize to MANT_W.
    //------------------------------------------------------------------
    reg [EXP_W-1:0] leading_zeros;

    always @(*) begin
        leading_zeros = {EXP_W{1'b0}};
        if      (or_result[WIDTH-1])  leading_zeros = 4'd0;
        else if (or_result[WIDTH-2])  leading_zeros = 4'd1;
        else if (or_result[WIDTH-3])  leading_zeros = 4'd2;
        else if (or_result[WIDTH-4])  leading_zeros = 4'd3;
        else if (or_result[WIDTH-5])  leading_zeros = 4'd4;
        else if (or_result[WIDTH-6])  leading_zeros = 4'd5;
        else if (or_result[WIDTH-7])  leading_zeros = 4'd6;
        else if (or_result[WIDTH-8])  leading_zeros = 4'd7;
        else if (or_result[WIDTH-9])  leading_zeros = 4'd8;
        else if (or_result[WIDTH-10]) leading_zeros = 4'd9;
        else if (or_result[WIDTH-11]) leading_zeros = 4'd10;
        else if (or_result[WIDTH-12]) leading_zeros = 4'd11;
        else if (or_result[WIDTH-13]) leading_zeros = 4'd12;
        else if (or_result[WIDTH-14]) leading_zeros = 4'd13;
        else if (or_result[WIDTH-15]) leading_zeros = 4'd14;
        else                          leading_zeros = 4'd15;
    end

    //------------------------------------------------------------------
    // Step 3: Normalize all samples
    // Shift each sample left by leading_zeros, then take the top MANT_W bits.
    // This normalizes the largest sample to fill the mantissa width.
    //------------------------------------------------------------------
    wire [EXP_W-1:0] shift_amt = leading_zeros;
    wire signed [WIDTH-1:0] shifted [0:N-1];
    wire signed [MANT_W-1:0] mant [0:N-1];

    generate
        for (gi = 0; gi < N; gi = gi + 1) begin : shift_gen
            assign shifted[gi] = sample_val[gi] << shift_amt;
            // Take upper MANT_W bits of the shifted value as the mantissa
            assign mant[gi] = shifted[gi][WIDTH-1 -: MANT_W];
        end
    endgenerate

    //------------------------------------------------------------------
    // Step 4: Register outputs (1-cycle latency)
    // Pack mantissas back into flattened output bus.
    //------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            exponent_out <= {EXP_W{1'b0}};
            mantissa_out <= {(N*MANT_W){1'b0}};
        end else begin
            valid_out <= valid_in;
            if (valid_in) begin
                exponent_out <= leading_zeros;
                for (i = 0; i < N; i = i + 1) begin
                    mantissa_out[i*MANT_W +: MANT_W] <= mant[i];
                end
            end
        end
    end

endmodule