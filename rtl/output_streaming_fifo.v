`default_nettype none
//==============================================================================
// output_streaming_fifo.v — Output Streaming FIFO with Backpressure
//==============================================================================
// 32-entry FIFO that collects complex (re/im) IFFT outputs as they are
// produced and streams them to the host under flow control.
//
// The IFFT producer writes one complex sample per cycle (wr_en).  The host
// may begin reading after the first 32 outputs have been buffered.  A
// standard FIFO interface with almost_empty provides early-warning
// backpressure so the host read FSM can avoid underflow.  overflow latches
// if the producer writes a full FIFO (a dropped-sample flag the host can
// poll and clear).
//
// Storage: DEPTH complex entries, each WIDTH bits re + WIDTH bits im.
//   Q8.8 → WIDTH=16.  Verilog-2005, synthesizable (inferred dual-port RAM).
//==============================================================================
module output_streaming_fifo #(
    parameter DEPTH = 32,
    parameter WIDTH = 16,
    parameter AW    = 6,            // ceil(log2(DEPTH)) for DEPTH=32
    parameter ALMOST_EMPTY_THRESH = 4
) (
    input  wire             clk,
    input  wire             rst,

    // Write port (IFFT producer)
    input  wire             wr_en,
    input  wire [WIDTH-1:0] wr_data_re,
    input  wire [WIDTH-1:0] wr_data_im,

    // Read port (host consumer)
    input  wire             rd_en,
    output reg  [WIDTH-1:0] rd_data_re,
    output reg  [WIDTH-1:0] rd_data_im,

    // Flags
    output wire             empty,
    output wire             full,
    output wire             almost_empty,
    output reg              overflow
);

    //------------------------------------------------------------------
    // Storage
    //------------------------------------------------------------------
    reg [WIDTH-1:0] mem_re [0:DEPTH-1];
    reg [WIDTH-1:0] mem_im [0:DEPTH-1];

    // Pointers
    reg [AW-1:0] wr_ptr;
    reg [AW-1:0] rd_ptr;
    reg [AW:0]   count;        // 0..DEPTH (extra bit for full/empty disambig)

    // Effective write/read enables (gated by capacity/occupancy)
    wire do_write = wr_en & ~full;
    wire do_read  = rd_en & ~empty;

    //------------------------------------------------------------------
    // Pointer / count management
    //------------------------------------------------------------------
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            wr_ptr  <= {AW{1'b0}};
            rd_ptr  <= {AW{1'b0}};
            count   <= {(AW+1){1'b0}};
            overflow<= 1'b0;
        end else begin
            // Overflow: write attempted while full
            if (wr_en && full)
                overflow <= 1'b1;

            // Write
            if (do_write) begin
                mem_re[wr_ptr] <= wr_data_re;
                mem_im[wr_ptr] <= wr_data_im;
                wr_ptr         <= wr_ptr + 1'b1;
            end

            // Read
            if (do_read) begin
                rd_ptr <= rd_ptr + 1'b1;
            end

            // Count update (write + read in same cycle → net 0)
            case ({do_write, do_read})
                2'b10:   count <= count + 1'b1;
                2'b01:   count <= count - 1'b1;
                default: count <= count;
            endcase
        end
    end

    //------------------------------------------------------------------
    // Combinational read (synchronous-read FIFO: data appears next cycle).
    // For a look-ahead style we register the read data when do_read is 1.
    //------------------------------------------------------------------
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            rd_data_re <= {WIDTH{1'b0}};
            rd_data_im <= {WIDTH{1'b0}};
        end else if (do_read) begin
            rd_data_re <= mem_re[rd_ptr];
            rd_data_im <= mem_im[rd_ptr];
        end
    end

    //------------------------------------------------------------------
    // Status flags
    //------------------------------------------------------------------
    assign empty        = (count == {(AW+1){1'b0}});
    assign full         = (count == DEPTH[AW:0]);
    assign almost_empty = (count <= ALMOST_EMPTY_THRESH[AW:0]);

endmodule

`default_nettype wire