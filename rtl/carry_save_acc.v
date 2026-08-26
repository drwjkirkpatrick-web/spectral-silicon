`default_nettype none
//==============================================================================
// carry_save_acc.v — Carry-Save Accumulator for Spectral MAC
//==============================================================================
// Performance improvement: Maintains the running accumulation in carry-save
// (redundant) form throughout the MAC loop, avoiding a carry-propagate add on
// every cycle.  Only a single CPA is performed at the end after k cycles,
// reducing accumulator critical path from O(WIDTH) per cycle to O(1) per
// cycle (just a full adder) plus one O(WIDTH) CPA at completion.
//
// Security preservation: the number of accumulation cycles (k) is fixed and
// data-independent.  The carry-save add operates in constant time regardless
// of operand values.  The final CPA fires after exactly k cycles — no early
// termination based on data content.
//
// Interface:
//   clk, rst_n       — clock and reset
//   product_re, product_im — complex product to accumulate (PW-bit signed)
//   valid_in         — product valid (assert for k cycles)
//   k                — number of accumulation cycles (parameter or input)
//   acc_re, acc_im   — accumulated sum (after k cycles, valid_out asserted)
//   valid_out        — result valid
//   busy             — accumulation in progress
//
// Verilog-2005, `default_nettype none.  Synthesizable.
//==============================================================================
module carry_save_acc #(
    parameter WIDTH = 16,       // Q8.8 data width
    parameter PW    = 2*WIDTH,   // Product width
    parameter AW    = PW + 8,   // Accumulator width (extra guard bits)
    parameter MAX_K = 64        // Maximum accumulation cycles
) (
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire signed [PW-1:0]    product_re,
    input  wire signed [PW-1:0]    product_im,
    input  wire                    valid_in,
    input  wire [7:0]              k,          // Number of cycles to accumulate
    output reg  signed [AW-1:0]    acc_re,
    output reg  signed [AW-1:0]    acc_im,
    output reg                     valid_out,
    output reg                     busy
);

    //------------------------------------------------------------------
    // Carry-save state: (sum, carry) for both real and imaginary parts.
    // During accumulation: acc = sum + carry (redundant representation).
    // Final CPA: acc_re = sum_re + carry_re (one adder at the end).
    //------------------------------------------------------------------
    reg signed [AW-1:0] sum_re,  carry_re;
    reg signed [AW-1:0] sum_im,  carry_im;

    // Cycle counter
    reg [7:0] cycle_cnt;

    // State machine
    localparam S_IDLE = 2'd0,
               S_ACC  = 2'd1,
               S_CPA = 2'd2,
               S_DONE = 2'd3;

    reg [1:0] state;

    // Sign-extend product to accumulator width
    wire signed [AW-1:0] prod_re_ext = {{(AW-PW){product_re[PW-1]}}, product_re};
    wire signed [AW-1:0] prod_im_ext = {{(AW-PW){product_im[PW-1]}}, product_im};

    //------------------------------------------------------------------
    // Carry-save adder (CSA): 3:2 compressor
    //   new_sum   = sum ^ carry ^ product
    //   new_carry = ((sum & carry) | (sum & product) | (carry & product)) << 1
    //------------------------------------------------------------------
    wire signed [AW-1:0] csa_sum_re   = sum_re   ^ carry_re   ^ prod_re_ext;
    wire signed [AW-1:0] csa_carry_re = ((sum_re & carry_re) |
                                         (sum_re & prod_re_ext) |
                                         (carry_re & prod_re_ext)) << 1;

    wire signed [AW-1:0] csa_sum_im   = sum_im   ^ carry_im   ^ prod_im_ext;
    wire signed [AW-1:0] csa_carry_im = ((sum_im & carry_im) |
                                         (sum_im & prod_im_ext) |
                                         (carry_im & prod_im_ext)) << 1;

    //------------------------------------------------------------------
    // Main control
    //------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            sum_re     <= 0;  carry_re <= 0;
            sum_im     <= 0;  carry_im <= 0;
            acc_re     <= 0;  acc_im   <= 0;
            valid_out  <= 1'b0;
            busy       <= 1'b0;
            cycle_cnt  <= 8'd0;
        end else begin
            valid_out <= 1'b0;  // Default

            case (state)
            //--- Idle: wait for first valid product ---
            S_IDLE: begin
                busy <= 1'b0;
                if (valid_in) begin
                    // Initialize: first product goes directly into sum
                    sum_re    <= prod_re_ext;
                    carry_re  <= 0;
                    sum_im    <= prod_im_ext;
                    carry_im  <= 0;
                    cycle_cnt <= 8'd1;
                    busy      <= 1'b1;
                    if (k <= 8'd1) begin
                        state <= S_CPA;
                    end else begin
                        state <= S_ACC;
                    end
                end
            end

            //--- Accumulate: CSA each cycle, no CPA ---
            S_ACC: begin
                if (valid_in) begin
                    sum_re   <= csa_sum_re;
                    carry_re <= csa_carry_re;
                    sum_im   <= csa_sum_im;
                    carry_im <= csa_carry_im;
                    cycle_cnt <= cycle_cnt + 8'd1;
                    if (cycle_cnt + 8'd1 >= k) begin
                        state <= S_CPA;
                    end
                end
            end

            //--- Final carry-propagate add ---
            S_CPA: begin
                acc_re    <= sum_re + carry_re;
                acc_im    <= sum_im + carry_im;
                valid_out <= 1'b1;
                busy      <= 1'b0;
                state     <= S_DONE;
            end

            //--- Done: return to idle ---
            S_DONE: begin
                state <= S_IDLE;
            end

            default: state <= S_IDLE;
            endcase
        end
    end

endmodule