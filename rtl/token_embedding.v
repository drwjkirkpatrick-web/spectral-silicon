`default_nettype none
//==============================================================================
// token_embedding.v — Token Embedding Lookup Module
//==============================================================================
// Maps a token_id (0..VOCAB_SIZE-1) to a D_MODEL-dimensional embedding vector.
//
// Storage: VOCAB_SIZE × D_MODEL × WIDTH bits
//   Default: 128 × 64 × 16 = 131,072 bits = 16,384 bytes = 16 KB
//
// The host loads embedding weights at boot via the write interface.
// During inference, the host sends a token_id and the module streams out
// D_MODEL consecutive values (one per clock cycle).
//
// ~500 gates logic + 16 KB storage.  Q8.8 fixed-point, WIDTH=16.
//==============================================================================
module token_embedding #(
    parameter WIDTH      = 16,
    parameter FRAC       = 8,
    parameter VOCAB_SIZE = 128,
    parameter D_MODEL    = 64
) (
    input  wire                    clk,
    input  wire                    rst_n,

    // Weight loading interface (host writes at boot)
    input  wire                    emb_we,
    input  wire [12:0]             emb_addr,   // {token_id[6:0], dim[5:0]}
    input  wire signed [WIDTH-1:0] emb_data,

    // Inference interface
    input  wire                    start,       // Assert to begin streaming
    input  wire [6:0]              token_id,    // Which token to look up
    output reg                     data_out_valid,
    input  wire                    data_out_ready,
    output reg  signed [WIDTH-1:0] data_out,
    output reg                     done
);

    // Embedding storage: 128 × 64 = 8192 entries × 16 bits
    reg signed [WIDTH-1:0] emb_table [0:VOCAB_SIZE*D_MODEL-1];

    // State machine
    localparam ST_IDLE = 1'b0,
               ST_STREAM = 1'b1;

    reg        state;
    reg [6:0]  cur_dim;     // current dimension counter (0..63)

    // Weight write (synchronous)
    always @(posedge clk) begin
        if (emb_we) begin
            emb_table[emb_addr] <= emb_data;
        end
    end

    // Inference state machine
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= ST_IDLE;
            data_out_valid<= 1'b0;
            data_out      <= 0;
            done          <= 1'b0;
            cur_dim       <= 0;
        end else begin
            done <= 1'b0;
            case (state)
            ST_IDLE: begin
                data_out_valid <= 1'b0;
                if (start) begin
                    cur_dim <= 0;
                    state  <= ST_STREAM;
                end
            end
            ST_STREAM: begin
                data_out_valid <= 1'b1;
                if (data_out_ready || !data_out_valid) begin
                    // Compute flat index: token_id * D_MODEL + cur_dim
                    data_out <= emb_table[{token_id, cur_dim}];
                    if (cur_dim == D_MODEL - 1) begin
                        state <= ST_IDLE;
                        done  <= 1'b1;
                        data_out_valid <= 1'b0;
                    end else begin
                        cur_dim <= cur_dim + 1;
                    end
                end
            end
            endcase
        end
    end

endmodule

`default_nettype wire