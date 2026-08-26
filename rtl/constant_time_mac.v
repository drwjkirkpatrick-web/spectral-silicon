`default_nettype none
//==============================================================================
// constant_time_mac.v — Constant-Time Spectral MAC (Timing Side-Channel Defense)
//==============================================================================
// Security rationale:
//   A naive spectral MAC implementation skips modes that are zeroed by
//   soft-thresholding, creating a data-dependent timing signature: the
//   number of cycles to complete depends on how many modes are active.
//   An attacker measuring execution time can infer the spectral structure
//   of the input, leaking information about the model's activation pattern.
//
//   This module processes ALL k modes unconditionally in a fixed number of
//   cycles, regardless of input data.  Even modes zeroed by soft-thresholding
//   are multiplied through (by zero) — the multiply happens, the result is
//   just zero.  No early termination, no conditional skips, no data-dependent
//   branches.
//
//   Fixed latency: exactly N_MODES + 2 cycles (1 for input reg, N_MODES for
//   MAC accumulation, 1 for output reg).  This is independent of input values.
//
//   Q8.8 signed complex multiply-accumulate:
//     acc_re += (w_re * x_re - w_im * x_im) >> FRAC
//     acc_im += (w_re * x_im + w_im * x_re) >> FRAC
//
// Interface:
//   start         — strobe to begin MAC operation
//   weight_re/im  — weight memory read interface (addr from mode counter)
//   data_re/im    — input spectral data (one mode per cycle)
//   done          — completion strobe (fixed cycle after start)
//   result_re/im  — accumulated MAC result
//   mode_idx      — current mode index (for weight/data addressing)
//
// Improvement 13 specification.
//==============================================================================
module constant_time_mac #(
    parameter N_MODES = 32,    // number of spectral modes (k)
    parameter WIDTH   = 16,    // Q8.8 data width
    parameter FRAC    = 8,     // fractional bits
    parameter ACC_W   = 32     // accumulator width (guard bits)
) (
    input  wire                    clk,
    input  wire                    rst_n,

    // Control
    input  wire                    start,       // begin MAC operation

    // Weight memory interface (combinational read)
    output reg  [5:0]              weight_addr, // mode index → weight addr
    input  wire signed [WIDTH-1:0] weight_re,  // weight real (Q8.8)
    input  wire signed [WIDTH-1:0] weight_im,  // weight imag (Q8.8)

    // Input spectral data (one mode per cycle, synchronous to start)
    input  wire signed [WIDTH-1:0] data_re,    // data real (Q8.8)
    input  wire signed [WIDTH-1:0] data_im,    // data imag (Q8.8)

    // Outputs
    output reg                     done,        // completion strobe
    output reg  signed [ACC_W-1:0] result_re,  // accumulated real
    output reg  signed [ACC_W-1:0] result_im,  // accumulated imag
    output reg  [5:0]              mode_idx     // current mode index
);

    //----------------------------------------------------------------------
    // States: IDLE → MAC → DONE
    // Fixed cycle count: N_MODES cycles in MAC state, deterministic.
    //----------------------------------------------------------------------
    localparam ST_IDLE = 2'd0;
    localparam ST_MAC  = 2'd1;
    localparam ST_DONE = 2'd2;

    reg [1:0]  state;
    reg [5:0]  mode_cnt;

    // Accumulators (wide to prevent overflow)
    reg signed [ACC_W-1:0] acc_re;
    reg signed [ACC_W-1:0] acc_im;

    // Complex multiply intermediates (WIDTH*2 bits to hold product)
    wire signed [2*WIDTH-1:0] prod_rr;  // w_re * x_re
    wire signed [2*WIDTH-1:0] prod_ii;  // w_im * x_im
    wire signed [2*WIDTH-1:0] prod_ri;  // w_re * x_im
    wire signed [2*WIDTH-1:0] prod_ir;  // w_im * x_re

    assign prod_rr = weight_re * data_re;
    assign prod_ii = weight_im * data_im;
    assign prod_ri = weight_re * data_im;
    assign prod_ir = weight_im * data_re;

    // Complex multiply result (shifted right by FRAC to maintain Q8.8 scaling)
    // Sign-extend products to ACC_W before adding
    wire signed [ACC_W-1:0] mac_re;
    wire signed [ACC_W-1:0] mac_im;

    assign mac_re = {{(ACC_W-2*WIDTH){prod_rr[2*WIDTH-1]}}, (prod_rr - prod_ii) >>> FRAC};
    assign mac_im = {{(ACC_W-2*WIDTH){prod_ri[2*WIDTH-1]}}, (prod_ri + prod_ir) >>> FRAC};

    //----------------------------------------------------------------------
    // Main FSM — fixed latency, no data-dependent skips
    //----------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= ST_IDLE;
            mode_cnt    <= 6'd0;
            acc_re      <= {ACC_W{1'b0}};
            acc_im      <= {ACC_W{1'b0}};
            done        <= 1'b0;
            result_re   <= {ACC_W{1'b0}};
            result_im   <= {ACC_W{1'b0}};
            weight_addr <= 6'd0;
            mode_idx    <= 6'd0;
        end else begin
            done <= 1'b0;  // default: deassert done

            case (state)
                //--------------------------------------------------------------
                ST_IDLE: begin
                    if (start) begin
                        state    <= ST_MAC;
                        mode_cnt <= 6'd0;
                        acc_re   <= {ACC_W{1'b0}};
                        acc_im   <= {ACC_W{1'b0}};
                        // Pre-assert weight address for mode 0
                        weight_addr <= 6'd0;
                        mode_idx    <= 6'd0;
                    end
                end

                //--------------------------------------------------------------
                // MAC state: process ALL modes unconditionally.
                // Even if data_re/im == 0 (soft-thresholded), the multiply
                // still executes (× 0 = 0) — no skip, no early exit.
                // Latency is exactly N_MODES cycles, data-independent.
                //--------------------------------------------------------------
                ST_MAC: begin
                    // Accumulate complex product for current mode
                    acc_re <= acc_re + mac_re;
                    acc_im <= acc_im + mac_im;

                    mode_idx    <= mode_cnt;
                    weight_addr <= mode_cnt;

                    if (mode_cnt == (N_MODES - 1)) begin
                        state <= ST_DONE;
                    end else begin
                        mode_cnt <= mode_cnt + 1'b1;
                    end
                end

                //--------------------------------------------------------------
                ST_DONE: begin
                    // Latch final results, assert done for one cycle
                    result_re <= acc_re + mac_re;  // include last mode
                    result_im <= acc_im + mac_im;
                    done      <= 1'b1;
                    state     <= ST_IDLE;
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire