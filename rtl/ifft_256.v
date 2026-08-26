`default_nettype none
//==============================================================================
// ifft_256.v — 256-Point IFFT via Conjugate-FFT Method
//==============================================================================
// Implements the inverse FFT using the conjugate property:
//
//   IFFT(x) = (1/N) * conj( FFT( conj(x) ) )
//
// Steps:
//   1. Conjugate the input: x* = re - j*im
//   2. Forward FFT: X = FFT(x*)
//   3. Conjugate output: X* = Re(X) - j*Im(X)
//   4. Scale by 1/N: right-shift by 8 (since N=256 = 2^8)
//
// This reuses the fft_256 module, saving area (no separate IFFT engine).
//
// Parameters:
//   WIDTH = 16 (Q8.8 fixed-point)
//
// Prompt 17 specification.
//==============================================================================
module ifft_256 #(
    parameter WIDTH = 16,
    parameter FRAC  = 8,
    parameter N     = 256
) (
    input  wire                    clk,
    input  wire                    rst_n,

    // Control
    input  wire                    start,        // Assert to begin IFFT
    output reg                     done,        // Asserted when result ready

    // Input data interface (streaming, 256 samples)
    input  wire                    data_in_valid,
    output reg                     data_in_ready,
    input  wire signed [WIDTH-1:0] data_in_re,
    input  wire signed [WIDTH-1:0] data_in_im,

    // Output data interface (streaming, 256 samples)
    output reg                     data_out_valid,
    input  wire                    data_out_ready,
    output reg  signed [WIDTH-1:0] data_out_re,
    output reg  signed [WIDTH-1:0] data_out_im
);

    //----------------------------------------------------------------------
    // Internal state machine for the conjugate-FFT-conjugate pipeline
    //----------------------------------------------------------------------
    localparam I_IDLE       = 3'd0,
               I_CONJ_IN    = 3'd1,  // Conjugate input, feed to FFT
               I_FFT_RUN    = 3'd2,  // Wait for FFT to finish
               I_CONJ_OUT   = 3'd3,  // Conjugate + scale FFT output
               I_FINISH     = 3'd4;

    reg [2:0] state;

    // FFT instance
    reg                     fft_start;
    wire                    fft_done;
    reg                     fft_data_in_valid;
    wire                    fft_data_in_ready;
    reg  signed [WIDTH-1:0] fft_data_in_re;
    reg  signed [WIDTH-1:0] fft_data_in_im;
    wire                    fft_data_out_valid;
    reg                     fft_data_out_ready;
    wire signed [WIDTH-1:0] fft_data_out_re;
    wire signed [WIDTH-1:0] fft_data_out_im;

    fft_256 #(
        .WIDTH(WIDTH),
        .FRAC(FRAC),
        .N(N)
    ) u_fft (
        .clk(clk),
        .rst_n(rst_n),
        .start(fft_start),
        .done(fft_done),
        .data_in_valid(fft_data_in_valid),
        .data_in_ready(fft_data_in_ready),
        .data_in_re(fft_data_in_re),
        .data_in_im(fft_data_in_im),
        .data_out_valid(fft_data_out_valid),
        .data_out_ready(fft_data_out_ready),
        .data_out_re(fft_data_out_re),
        .data_out_im(fft_data_out_im)
    );

    // Sample counters
    reg [7:0] in_cnt;    // Input sample counter
    reg [7:0] out_cnt;   // Output sample counter

    // LOG2(N) = 8 for scaling
    localparam SCALE_SHIFT = 8;  // log2(256)

    //----------------------------------------------------------------------
    // Main state machine
    //----------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state              <= I_IDLE;
            done               <= 1'b0;
            data_in_ready      <= 1'b0;
            data_out_valid     <= 1'b0;
            data_out_re        <= 0;
            data_out_im        <= 0;
            fft_start          <= 1'b0;
            fft_data_in_valid  <= 1'b0;
            fft_data_in_re     <= 0;
            fft_data_in_im     <= 0;
            fft_data_out_ready <= 1'b0;
            in_cnt             <= 0;
            out_cnt             <= 0;
        end else begin
            done <= 1'b0;  // Default

            case (state)

            //--- Idle: wait for start ---
            I_IDLE: begin
                data_in_ready <= 1'b1;
                if (start) begin
                    state         <= I_CONJ_IN;
                    data_in_ready <= 1'b0;
                    in_cnt        <= 0;
                    fft_start     <= 1'b1;  // Start FFT
                end
            end

            //--- Conjugate input and feed to FFT ---
            // x* = re - j*im → pass (re, -im) to FFT
            I_CONJ_IN: begin
                fft_start <= 1'b0;  // Clear start after 1 cycle
                if (in_cnt < N) begin
                    // Conjugate: negate imaginary part
                    fft_data_in_re    <= data_in_re;
                    fft_data_in_im    <= -data_in_im;
                    fft_data_in_valid <= data_in_valid;
                    if (data_in_valid && fft_data_in_ready) begin
                        in_cnt <= in_cnt + 1;
                    end
                end else begin
                    fft_data_in_valid <= 1'b0;
                    state             <= I_FFT_RUN;
                end
            end

            //--- Wait for FFT to complete ---
            I_FFT_RUN: begin
                fft_data_in_valid  <= 1'b0;
                fft_data_out_ready <= 1'b1;
                if (fft_done) begin
                    // FFT is done; now read its output, conjugate and scale
                    state    <= I_CONJ_OUT;
                    out_cnt  <= 0;
                end
            end

            //--- Conjugate + scale FFT output ---
            // FFT output is Re(X) + j*Im(X)
            // Conjugate: Re(X) - j*Im(X)
            // Scale: right shift by 8 (1/256)
            I_CONJ_OUT: begin
                data_out_valid <= 1'b1;
                if (data_out_ready) begin
                    if (fft_data_out_valid) begin
                        // Conjugate: negate imaginary part
                        // Scale: arithmetic right shift by SCALE_SHIFT
                        data_out_re <= fft_data_out_re >>> SCALE_SHIFT;
                        data_out_im <= (-fft_data_out_im) >>> SCALE_SHIFT;
                    end
                    if (out_cnt == N - 1) begin
                        state <= I_FINISH;
                    end else begin
                        out_cnt <= out_cnt + 1;
                    end
                end
            end

            //--- Finish ---
            I_FINISH: begin
                data_out_valid     <= 1'b0;
                fft_data_out_ready <= 1'b0;
                done               <= 1'b1;
                state              <= I_IDLE;
            end

            default: state <= I_IDLE;
            endcase
        end
    end

endmodule