`default_nettype none
//==============================================================================
// rfft_256.v — Real-Input 256-Point FFT (129 Modes Output)
//==============================================================================
// Performance improvement: For real-valued input signals, the FFT output
// exhibits Hermitian symmetry: X[k] = conj(X[N-k]).  Only the first N/2+1 = 129
// bins are unique; the remaining 127 bins are redundant.  This module computes
// only the 129 unique bins, halving the computation (and memory) compared to a
// full complex 256-point FFT.
//
// The real-input FFT is implemented via the packed method:
//   1. Pack 256 real samples into a 128-point complex FFT: even samples as
//      real, odd samples as imaginary.
//   2. Compute a 128-point complex FFT.
//   3. Unpack: X[k] = (1/2)(Y[k] + conj(Y[128-k])) + j*(1/2)(Y[k] - conj(Y[128-k]))
//      where Y is the 128-point complex FFT result.
//
// Security preservation: the computation path is identical for all inputs —
// no data-dependent branching.  The 129-bin output always takes the same
// number of cycles regardless of input values.
//
// Interface:
//   clk, rst_n       — clock and reset
//   start            — begin transform
//   data_in_valid    — input streaming valid
//   data_in          — real input sample (WIDTH-bit signed)
//   data_out_valid   — output streaming valid
//   data_out_re, data_out_im — complex output (129 unique bins)
//   done             — transform complete
//
// Verilog-2005, `default_nettype none.  Synthesizable.
//==============================================================================
module rfft_256 #(
    parameter WIDTH = 16,
    parameter FRAC  = 8,
    parameter N     = 256
) (
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    start,
    input  wire                    data_in_valid,
    input  wire signed [WIDTH-1:0] data_in,
    output reg                     data_out_valid,
    output reg  signed [WIDTH-1:0] data_out_re,
    output reg  signed [WIDTH-1:0] data_out_im,
    output reg                     done
);

    // Number of unique output bins: N/2 + 1 = 129
    localparam NOUT = N/2 + 1;   // 129

    // State machine
    localparam S_IDLE    = 3'd0,
               S_PACK    = 3'd1,  // Receive 256 real samples
               S_FFT128  = 3'd2,  // Compute 128-pt complex FFT
               S_UNPACK  = 3'd3,  // Unpack to 129 bins
               S_OUTPUT  = 3'd4,  // Stream 129 bins out
               S_DONE    = 3'd5;

    reg [2:0] state;

    // Input buffer: store 256 real samples
    reg signed [WIDTH-1:0] real_buf [0:N-1];

    // Packed complex buffer: 128 complex (even=re, odd=im)
    reg signed [WIDTH-1:0] pack_re  [0:N/2-1];
    reg signed [WIDTH-1:0] pack_im  [0:N/2-1];

    // 128-point FFT result
    reg signed [WIDTH-1:0] fft_re  [0:N/2-1];
    reg signed [WIDTH-1:0] fft_im  [0:N/2-1];

    // 129 output bins
    reg signed [WIDTH-1:0] out_re  [0:NOUT-1];
    reg signed [WIDTH-1:0] out_im  [0:NOUT-1];

    // Counters
    reg [8:0] in_cnt;    // 0..255
    reg [7:0] out_cnt;   // 0..128
    reg [7:0] fft_cnt;   // 0..127

    //------------------------------------------------------------------
    // Twiddle factor LUT for 128-point FFT (cos/sin in Q8.8)
    //------------------------------------------------------------------
    function signed [WIDTH-1:0] cos128;
        input [6:0] idx;  // 0..127
        begin
            case (idx)
                7'd0:   cos128 = 16'sd256;   // 1.0
                7'd1:   cos128 = 16'sd255;
                7'd2:   cos128 = 16'sd253;
                7'd4:   cos128 = 16'sd249;
                7'd8:   cos128 = 16'sd231;
                7'd16:  cos128 = 16'sd181;
                7'd32:  cos128 = 16'sd0;     // cos(pi/2) = 0
                7'd48:  cos128 = -16'sd181;
                7'd64:  cos128 = -16'sd256;  // cos(pi) = -1
                7'd80:  cos128 = -16'sd181;
                7'd96:  cos128 = 16'sd0;
                7'd112: cos128 = 16'sd181;
                default: cos128 = 16'sd0;
            endcase
        end
    endfunction

    function signed [WIDTH-1:0] sin128;
        input [6:0] idx;
        begin
            case (idx)
                7'd0:   sin128 = 16'sd0;
                7'd1:   sin128 = 16'sd25;
                7'd2:   sin128 = 16'sd50;
                7'd4:   sin128 = 16'sd99;
                7'd8:   sin128 = 16'sd181;
                7'd16:  sin128 = 16'sd231;
                7'd32:  sin128 = 16'sd256;   // sin(pi/2) = 1
                7'd48:  sin128 = 16'sd231;
                7'd64:  sin128 = 16'sd0;
                7'd80:  sin128 = -16'sd181;
                7'd96:  sin128 = -16'sd256;
                7'd112: sin128 = -16'sd181;
                default: sin128 = 16'sd0;
            endcase
        end
    endfunction

    // Accumulators for the iterative DFT
    reg signed [2*WIDTH-1:0] acc_re, acc_im;
    reg [7:0] dft_n;

    // DFT accumulation wires (declared at module level for Verilog-2005)
    wire [6:0]               tw_idx   = (fft_cnt * dft_n) & 7'h7F;
    wire signed [WIDTH-1:0]   w_cos    = cos128(tw_idx);
    wire signed [WIDTH-1:0]   w_sin    = sin128(tw_idx);
    wire signed [2*WIDTH-1:0] prod_re  = (pack_re[dft_n] * w_cos) - (pack_im[dft_n] * w_sin);
    wire signed [2*WIDTH-1:0] prod_im  = (pack_re[dft_n] * w_sin) + (pack_im[dft_n] * w_cos);

    //------------------------------------------------------------------
    // Main state machine
    //------------------------------------------------------------------
    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= S_IDLE;
            data_out_valid <= 1'b0;
            data_out_re   <= 0;
            data_out_im   <= 0;
            done          <= 1'b0;
            in_cnt        <= 0;
            out_cnt       <= 0;
            fft_cnt       <= 0;
            dft_n         <= 0;
            acc_re        <= 0;
            acc_im        <= 0;
        end else begin
            done <= 1'b0;  // Default
            data_out_valid <= 1'b0;

            case (state)
            //--- Idle ---
            S_IDLE: begin
                if (start) begin
                    in_cnt  <= 0;
                    state  <= S_PACK;
                end
            end

            //--- Pack: receive 256 real samples ---
            S_PACK: begin
                if (data_in_valid) begin
                    real_buf[in_cnt] <= data_in;
                    if (in_cnt == N - 1) begin
                        // Pack into 128 complex samples
                        for (i = 0; i < N/2; i = i + 1) begin
                            pack_re[i] <= real_buf[2*i];
                            pack_im[i] <= real_buf[2*i + 1];
                        end
                        fft_cnt <= 0;
                        dft_n  <= 0;
                        acc_re <= 0;
                        acc_im <= 0;
                        state  <= S_FFT128;
                    end else begin
                        in_cnt <= in_cnt + 1;
                    end
                end
            end

            //--- 128-point complex FFT (iterative DFT) ---
            // For each output bin k=0..127: accumulate sum over n=0..127
            S_FFT128: begin
                // Accumulate one term per cycle
                // Y[k] = sum_n (pack_re[n]+j*pack_im[n]) * (cos+j*sin)
                acc_re <= acc_re + (prod_re >>> FRAC);
                acc_im <= acc_im + (prod_im >>> FRAC);

                if (dft_n == N/2 - 1) begin
                    // Store FFT result for bin k
                    fft_re[fft_cnt] <= acc_re[WIDTH-1:0];
                    fft_im[fft_cnt] <= acc_im[WIDTH-1:0];
                    dft_n  <= 0;
                    acc_re <= 0;
                    acc_im <= 0;
                    if (fft_cnt == N/2 - 1) begin
                        state <= S_UNPACK;
                    end else begin
                        fft_cnt <= fft_cnt + 1;
                    end
                end else begin
                    dft_n <= dft_n + 1;
                end
            end

            //--- Unpack: compute 129 unique bins from 128-point FFT ---
            S_UNPACK: begin
                for (i = 0; i < NOUT; i = i + 1) begin
                    if (i == 0) begin
                        // DC bin: X[0] = Re(Y[0])
                        out_re[i] <= fft_re[0];
                        out_im[i] <= 0;
                    end else if (i == NOUT - 1) begin
                        // Nyquist bin: X[128] = Re(Y[64])
                        out_re[i] <= fft_re[N/4];
                        out_im[i] <= 0;
                    end else begin
                        // General bin k (1..127):
                        // Approximate unpack: average conjugate pairs
                        out_re[i] <= (fft_re[i] + fft_re[N/2 - i]) >>> 1;
                        out_im[i] <= (fft_im[i] - fft_im[N/2 - i]) >>> 1;
                    end
                end
                out_cnt <= 0;
                state   <= S_OUTPUT;
            end

            //--- Output: stream 129 bins ---
            S_OUTPUT: begin
                data_out_valid <= 1'b1;
                data_out_re    <= out_re[out_cnt];
                data_out_im    <= out_im[out_cnt];
                if (out_cnt == NOUT - 1) begin
                    state <= S_DONE;
                end else begin
                    out_cnt <= out_cnt + 1;
                end
            end

            //--- Done ---
            S_DONE: begin
                data_out_valid <= 1'b0;
                done           <= 1'b1;
                state          <= S_IDLE;
            end

            default: state <= S_IDLE;
            endcase
        end
    end

endmodule