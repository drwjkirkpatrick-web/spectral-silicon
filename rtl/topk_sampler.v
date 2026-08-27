`default_nettype none
//==============================================================================
// topk_sampler.v — Top-k Token Sampler
//==============================================================================
// Selects the k largest probability values from VOCAB_SIZE=128 inputs and
// randomly samples one token index from the top-k set.
//
// The module maintains a register array of MAX_K (value, index) pairs.
// For each incoming probability, it compares against the smallest entry in
// the top-k and replaces it if larger, then re-sorts. After all values
// received, a simple LFSR picks one of the top-k entries.
//
// k=1 → greedy decoding (always picks argmax).
// k=5 → standard top-k sampling.
//
// ~500 gates.  Q8.8 fixed-point, WIDTH=16.
//==============================================================================
module topk_sampler #(
    parameter WIDTH      = 16,
    parameter VOCAB_SIZE = 128,
    parameter MAX_K      = 8
) (
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    start,         // Begin receiving probabilities
    input  wire                    data_in_valid,
    output reg                     data_in_ready,
    input  wire signed [WIDTH-1:0] data_in,       // Probability value
    input  wire [3:0]              k,             // k=1..8 (default 5)
    input  wire signed [WIDTH-1:0] temperature,  // Q8.8 (stored for reference)
    output reg                     data_out_valid,
    input  wire                    data_out_ready,
    output reg  [6:0]              selected_idx,  // Sampled token index
    output reg                     done
);

    // State machine
    localparam [1:0] ST_SCAN   = 2'd0,
                     ST_SAMPLE = 2'd1,
                     ST_IDLE   = 2'd2,
                     ST_DONE   = 2'd3;

    reg [1:0] state;

    // Top-k storage: MAX_K entries of (value, index)
    reg signed [WIDTH-1:0] tk_val  [0:MAX_K-1];
    reg [6:0]             tk_idx  [0:MAX_K-1];

    // Input counter
    reg [6:0] in_cnt;

    // LFSR for random selection
    reg [15:0] lfsr;

    //--- Find minimum entry in top-k (for replacement comparison) ---
    // We compare against tk_val[k-1] which we keep as the smallest entry
    // via insertion-sort: new entries are inserted in sorted order.

    integer j;
    reg signed [WIDTH-1:0] new_val;
    reg [6:0]             new_idx;
    reg [3:0]             effective_k;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state          <= ST_IDLE;
            data_in_ready   <= 1'b0;
            data_out_valid  <= 1'b0;
            selected_idx    <= 0;
            done            <= 1'b0;
            in_cnt          <= 0;
            lfsr            <= 16'hACE1;  // seed
            for (j = 0; j < MAX_K; j = j + 1) begin
                tk_val[j] <= -16'sd32768;  // init to minimum
                tk_idx[j] <= 0;
            end
        end else begin
            done <= 1'b0;
            lfsr <= {lfsr[14:0], lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10]};

            effective_k = (k == 0) ? 4'd5 : k;  // default k=5

            case (state)
            //--- Idle: wait for start ---
            ST_IDLE: begin
                if (start) begin
                    in_cnt <= 0;
                    state  <= ST_SCAN;
                    data_in_ready <= 1'b1;
                    // Reset top-k to minimum
                    for (j = 0; j < MAX_K; j = j + 1) begin
                        tk_val[j] <= -16'sd32768;
                        tk_idx[j] <= 0;
                    end
                end
            end
            //--- Scan: receive 128 probabilities, maintain top-k ---
            ST_SCAN: begin
                data_in_ready <= 1'b1;
                if (data_in_valid) begin
                    new_val = data_in;
                    new_idx = in_cnt;

                    // Insertion sort: find position and shift
                    // Compare against the smallest (last) entry
                    if (new_val > tk_val[effective_k - 1]) begin
                        // Replace the smallest, then bubble up
                        tk_val[effective_k - 1] <= new_val;
                        tk_idx[effective_k - 1] <= new_idx;
                        // Simple shift sort: each cycle we can only do one swap
                        // For a hardware-friendly approach, we compare each
                        // pair and swap if out of order (one pass per input)
                        for (j = MAX_K - 1; j > 0; j = j - 1) begin
                            if (j <= effective_k - 1) begin
                                if (tk_val[j] > tk_val[j-1]) begin
                                    // Swap
                                    tk_val[j]   <= tk_val[j-1];
                                    tk_val[j-1] <= tk_val[j];
                                    tk_idx[j]   <= tk_idx[j-1];
                                    tk_idx[j-1] <= tk_idx[j];
                                end
                            end
                        end
                    end

                    if (in_cnt == VOCAB_SIZE - 1) begin
                        data_in_ready <= 1'b0;
                        state <= ST_SAMPLE;
                    end else begin
                        in_cnt <= in_cnt + 1;
                    end
                end
            end
            //--- Sample: pick a random entry from top-k ---
            ST_SAMPLE: begin
                // Use LFSR to pick index 0..k-1
                // selected = lfsr % effective_k (compute with division)
                begin : sample_blk
                    reg [3:0] pick;
                    pick = lfsr[3:0] % effective_k;
                    selected_idx <= tk_idx[pick];
                end
                data_out_valid <= 1'b1;
                state <= ST_DONE;
            end
            //--- Done ---
            ST_DONE: begin
                if (data_out_ready) begin
                    data_out_valid <= 1'b0;
                    done <= 1'b1;
                    state <= ST_IDLE;
                end
            end
            endcase
        end
    end

endmodule

`default_nettype wire