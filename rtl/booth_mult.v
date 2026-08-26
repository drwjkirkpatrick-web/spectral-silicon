`default_nettype none
//==============================================================================
// booth_mult.v — Radix-4 Booth Multiplier with Carry-Save Accumulation
//==============================================================================
// Performance improvement: Radix-4 Booth encoding halves the number of partial
// products (WIDTH/2 instead of WIDTH), and a Wallace-style carry-save tree
// defers the final carry-propagate addition to the very end.  This reduces
// critical-path latency from O(WIDTH) to O(log2(WIDTH/2)) for the partial-
// product reduction stage, plus one CPA at the output.
//
// Security preservation: deterministic combinational logic — no data-dependent
// timing.  The carry-save tree has constant depth regardless of operand values,
// preventing timing side-channels in the multiply path.
//
// Interface:
//   a, b  — signed WIDTH-bit operands (Q8.8 when WIDTH=16)
//   product — 2*WIDTH-bit signed result
//
// Verilog-2005, `default_nettype none.  Synthesizable.
//==============================================================================
module booth_mult #(
    parameter WIDTH = 16
) (
    input  wire  signed [WIDTH-1:0]  a,
    input  wire  signed [WIDTH-1:0]  b,
    output wire  signed [2*WIDTH-1:0] product
);

    // Number of Booth groups (radix-4 → groups of 2 bits, plus sign extension)
    localparam NPP = WIDTH / 2;  // partial product count

    //------------------------------------------------------------------
    // Radix-4 Booth encoding
    //
    // Booth radix-4 examines 3-bit windows: {b[2i+1], b[2i], b[2i-1]}
    // (with b[-1] = 0 for i=0) and encodes one of {-2a, -a, 0, +a, +2a}.
    //
    // Encoding table (b2 b1 b0):
    //   000 → 0,  001 → +a,  010 → +a,  011 → +2a
    //   100 → -2a, 101 → -a, 110 → -a, 111 → 0
    //------------------------------------------------------------------
    wire [2:0] booth_code [0:NPP-1];

    genvar gi;
    generate
        for (gi = 0; gi < NPP; gi = gi + 1) begin : booth_enc
            if (gi == 0)
                assign booth_code[gi] = {b[2*gi+1], b[2*gi], 1'b0};
            else
                assign booth_code[gi] = {b[2*gi+1], b[2*gi], b[2*gi-1]};
        end
    endgenerate

    //------------------------------------------------------------------
    // Partial product generation
    //
    // For each Booth group i, produce a (2*WIDTH)-bit partial product:
    //   pp[i] = encode(a) << (2*i)
    //
    // encode(a) is one of: 0, +a, -a, +2a, -2a  (sign-extended to 2*WIDTH).
    //------------------------------------------------------------------
    wire signed [2*WIDTH-1:0] pp [0:NPP-1];

    generate
        for (gi = 0; gi < NPP; gi = gi + 1) begin : pp_gen
            // Sign-extend a to 2*WIDTH bits, then shift left by 2*gi
            wire signed [2*WIDTH-1:0] a_ext   = {{WIDTH{a[WIDTH-1]}}, a};
            wire signed [2*WIDTH-1:0] a2_ext  = a_ext << 1;  // 2a
            wire signed [2*WIDTH-1:0] neg_a  = -a_ext;
            wire signed [2*WIDTH-1:0] neg_a2 = -a2_ext;

            reg signed [2*WIDTH-1:0] pp_val;

            always @(*) begin
                case (booth_code[gi])
                    3'b000: pp_val = {2*WIDTH{1'b0}};
                    3'b001: pp_val = a_ext;
                    3'b010: pp_val = a_ext;
                    3'b011: pp_val = a2_ext;
                    3'b100: pp_val = neg_a2;
                    3'b101: pp_val = neg_a;
                    3'b110: pp_val = neg_a;
                    3'b111: pp_val = {2*WIDTH{1'b0}};
                    default: pp_val = {2*WIDTH{1'b0}};
                endcase
            end

            assign pp[gi] = pp_val;
        end
    endgenerate

    //------------------------------------------------------------------
    // Carry-save accumulation tree (Wallace-style)
    //
    // We reduce all NPP partial products using a sequence of carry-save
    // adders (3:2 compressors).  The result is kept in carry-save form
    // (sum, carry) and a single carry-propagate adder produces the final
    // product.
    //
    // We implement the Wallace tree as a behavioral combinational reduction.
    // Synthesis tools map this to a CSA tree efficiently.
    //------------------------------------------------------------------

    // Intermediate arrays for the reduction
    reg signed [2*WIDTH:0] tmp_s [0:NPP-1];
    reg signed [2*WIDTH:0] tmp_c [0:NPP-1];
    reg signed [2*WIDTH:0] new_s [0:NPP-1];
    reg signed [2*WIDTH:0] new_c [0:NPP-1];

    // Final carry-propagate result
    reg signed [2*WIDTH:0] final_cpa;

    integer idx;
    integer remaining;
    integer next_remaining;

    always @(*) begin
        // Initialize: all partial products into sum array, carries = 0
        for (idx = 0; idx < NPP; idx = idx + 1) begin
            tmp_s[idx] = {{1'b0}, pp[idx]};  // zero-extend for carry-save
            tmp_c[idx] = 0;
        end

        remaining = NPP;

        // Reduce: while more than 2 values remain, apply CSA to triplets
        while (remaining > 2) begin
            next_remaining = 0;

            // Process triplets
            idx = 0;
            while (idx + 2 < remaining) begin
                // CSA: a, b, c → sum, carry
                new_s[next_remaining] = tmp_s[idx]   ^ tmp_s[idx+1] ^ tmp_s[idx+2];
                new_c[next_remaining] = ((tmp_s[idx] & tmp_s[idx+1]) |
                                        (tmp_s[idx] & tmp_s[idx+2]) |
                                        (tmp_s[idx+1] & tmp_s[idx+2])) << 1;
                next_remaining = next_remaining + 1;
                idx = idx + 3;
            end

            // Pass remaining 1 or 2 values through
            while (idx < remaining) begin
                new_s[next_remaining] = tmp_s[idx];
                new_c[next_remaining] = tmp_c[idx];
                next_remaining = next_remaining + 1;
                idx = idx + 1;
            end

            // Copy back
            for (idx = 0; idx < next_remaining; idx = idx + 1) begin
                tmp_s[idx] = new_s[idx];
                tmp_c[idx] = new_c[idx];
            end

            remaining = next_remaining;
        end

        // Final CPA: add remaining values
        if (remaining == 2) begin
            final_cpa = tmp_s[0] + tmp_s[1] + tmp_c[0] + tmp_c[1];
        end else begin
            final_cpa = tmp_s[0] + tmp_c[0];
        end
    end

    // Output: take lower 2*WIDTH bits
    assign product = final_cpa[2*WIDTH-1:0];

endmodule