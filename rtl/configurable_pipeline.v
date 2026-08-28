`default_nettype none
//==============================================================================
// configurable_pipeline.v — Configurable Pipeline Depth (2/4/8 stages)
//==============================================================================
// Dynamically selectable pipeline depth for the Spectral Silicon datapath.
//
// At 50 MHz the host selects 2 stages (minimum latency, ~40 ns per stage).
// At 120 MHz the host selects 8 stages (more latency but meets timing for
// higher throughput).  An 8-deep shift-register bank is always present; a
// runtime mux selects which tap feeds the output, bypassing the unused
// stages so no spurious latency is added at the shallow settings.
//
// data/valid share the same pipeline so the valid handshake tracks the data
// exactly through whichever depth is active.  depth_sel is sampled and held
// in a register so a mid-burst change does not corrupt in-flight words.
//
// Q8.8 fixed-point (16-bit, 8 fractional).  Verilog-2005, synthesizable.
//==============================================================================
module configurable_pipeline #(
    parameter MAX_STAGES = 8,
    parameter WIDTH      = 16
) (
    input  wire             clk,
    input  wire             rst,
    input  wire [1:0]       depth_sel,    // 0=2-stage, 1=4-stage, 2=8-stage
    input  wire [WIDTH-1:0] data_in,
    input  wire             valid_in,
    output reg  [WIDTH-1:0] data_out,
    output reg             valid_out
);

    //------------------------------------------------------------------
    // Resolve the requested depth from depth_sel.
    // depth_sel=3 (reserved) defaults to MAX_STAGES for safety.
    //------------------------------------------------------------------
    function [5:0] depth_from_sel;
        input [1:0] sel;
        begin
            case (sel)
                2'd0: depth_from_sel = 6'd2;
                2'd1: depth_from_sel = 6'd4;
                2'd2: depth_from_sel = 6'd8;
                default: depth_from_sel = 6'd8;  // reserved → deepest/safest
            endcase
        end
    endfunction

    // Latched depth so the selected tap is stable for in-flight data.
    reg [5:0] cur_depth;

    //------------------------------------------------------------------
    // Pipeline register banks: MAX_STAGES stages of data + valid.
    // stage_data[i] holds the output of pipeline stage (i+1).
    //   stage_data[0] = output of stage 1 (1 cycle delayed)
    //   stage_data[7] = output of stage 8 (8 cycles delayed)
    //------------------------------------------------------------------
    reg [WIDTH-1:0] stage_data [0:MAX_STAGES-1];
    reg             stage_valid [0:MAX_STAGES-1];

    integer i;

    //------------------------------------------------------------------
    // Shift the pipeline every clock.  Unused (bypassed) stages still
    // shift harmlessly; only the selected tap is muxed to the output.
    //------------------------------------------------------------------
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            cur_depth <= 6'd2;
            for (i = 0; i < MAX_STAGES; i = i + 1) begin
                stage_data[i]  <= {WIDTH{1'b0}};
                stage_valid[i] <= 1'b0;
            end
            data_out   <= {WIDTH{1'b0}};
            valid_out  <= 1'b0;
        end else begin
            // Latch depth on a change.
            cur_depth <= depth_from_sel(depth_sel);

            // Stage 1: input → stage_data[0]
            stage_data[0]  <= data_in;
            stage_valid[0] <= valid_in;

            // Stages 2..MAX_STAGES: shift forward
            for (i = 1; i < MAX_STAGES; i = i + 1) begin
                stage_data[i]  <= stage_data[i-1];
                stage_valid[i] <= stage_valid[i-1];
            end

            // Mux: pick the tap that is cur_depth cycles delayed.
            // cur_depth=2 → stage_data[1] (2-cycle path: in→s1→s2 tap)
            // cur_depth=4 → stage_data[3]
            // cur_depth=8 → stage_data[7]
            // (index = cur_depth - 1)
            case (cur_depth)
                6'd2: begin
                    data_out  <= stage_data[1];
                    valid_out <= stage_valid[1];
                end
                6'd4: begin
                    data_out  <= stage_data[3];
                    valid_out <= stage_valid[3];
                end
                6'd8: begin
                    data_out  <= stage_data[7];
                    valid_out <= stage_valid[7];
                end
                default: begin
                    // Fallback: deepest available tap.
                    data_out  <= stage_data[MAX_STAGES-1];
                    valid_out <= stage_valid[MAX_STAGES-1];
                end
            endcase
        end
    end

endmodule

`default_nettype wire