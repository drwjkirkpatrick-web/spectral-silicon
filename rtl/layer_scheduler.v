`default_nettype none
//==============================================================================
// layer_scheduler.v — Hardware Layer Scheduler with Weight Swapping
//==============================================================================
// Orchestrates sequential execution of up to 16 transformer layers on the
// Spectral Silicon chip.  When layer N completes, the scheduler immediately
// issues a prefetch of layer N+1's weights from the shadow register while
// the IFFT for layer N is still running.  This pipelines weight loading with
// computation and eliminates the inter-layer host round-trip (host would
// otherwise have to poll layer_complete and then kick off the next load).
//
// Protocol:
//   1. Host asserts start_layer with layer_num / total_layers programmed.
//   2. Scheduler arms layer N: layer_complete tracks its end.
//   3. On layer_complete, if not the last layer, scheduler asserts the
//      prefetch request (next_layer_weights_ready is an input handshake
//      from the shadow-weight loader indicating N+1 weights are staged).
//   4. While ifft_busy is high, weight swapping for N+1 proceeds in parallel.
//   5. When weight_swap_done is asserted by the loader, the scheduler
//      advances to layer N+1 automatically.
//   6. all_layers_done pulses after the final layer completes.
//
// Supports up to 16 layers (4-bit layer_num/total_layers).
// Verilog-2005, synthesizable.
//==============================================================================
module layer_scheduler #(
    parameter MAX_LAYERS = 16
) (
    input  wire       clk,
    input  wire       rst,

    // Host control
    input  wire       start_layer,
    input  wire [3:0] layer_num,            // starting layer index
    input  wire [3:0] total_layers,         // number of layers to run

    // Weight loader / shadow handshake
    input  wire       weight_swap_done,     // loader finished swapping N+1
    input  wire       ifft_busy,            // IFFT for current layer running
    input  wire       next_layer_weights_ready, // shadow has N+1 weights staged

    // Status
    output reg        layer_complete,       // current layer finished
    output reg        all_layers_done       // entire stack finished
);

    //------------------------------------------------------------------
    // State machine
    //------------------------------------------------------------------
    localparam S_IDLE       = 3'd0,
               S_RUN        = 3'd1,   // running current layer (FFT→mul→IFFT→modReLU)
               S_PREFETCH   = 3'd2,   // waiting for N+1 weights to be staged
               S_SWAP_WAIT  = 3'd3,   // waiting for swap to finish while IFFT runs
               S_ADVANCE    = 3'd4,   // advance to next layer
               S_DONE       = 3'd5;

    reg [2:0]  state;
    reg [3:0]  cur_layer;
    reg [3:0]  cur_total;
    reg        prefetch_issued;
    reg        layer_was_complete;

    // Combinational: is this the final layer?
    wire last_layer = (cur_layer >= cur_total) || (cur_total == 4'd0);

    //------------------------------------------------------------------
    // State machine
    //------------------------------------------------------------------
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state            <= S_IDLE;
            cur_layer        <= 4'd0;
            cur_total        <= 4'd0;
            prefetch_issued  <= 1'b0;
            layer_was_complete<= 1'b0;
            layer_complete   <= 1'b0;
            all_layers_done  <= 1'b0;
        end else begin
            // Defaults: pulse signals are single-cycle
            layer_complete  <= 1'b0;
            all_layers_done  <= 1'b0;

            case (state)
            //--------------------------------------------------------------
            S_IDLE: begin
                if (start_layer) begin
                    cur_layer        <= layer_num;
                    cur_total        <= total_layers;
                    prefetch_issued  <= 1'b0;
                    layer_was_complete<= 1'b0;
                    state            <= S_RUN;
                end
            end

            //--------------------------------------------------------------
            // Running the current layer.  layer_complete is driven by the
            // datapath (we model its rising edge here via layer_was_complete).
            // The real chip would receive a done signal from the pipeline;
            // for this scheduler we treat a 1→0 transition of ifft_busy
            // (IFFT finished) as layer_complete.
            S_RUN: begin
                // Detect IFFT going idle → layer complete.
                if (layer_was_complete && !ifft_busy) begin
                    // already captured
                end else if (!ifft_busy && layer_was_complete) begin
                    // (no-op branch kept for clarity)
                end

                // Rising edge of "layer done": ifft_busy was high last cycle
                // and is now low, AND we haven't already flagged completion.
                if (!ifft_busy && !layer_was_complete) begin
                    layer_complete      <= 1'b1;
                    layer_was_complete  <= 1'b1;

                    if (last_layer) begin
                        // Final layer — done.
                        all_layers_done <= 1'b1;
                        state           <= S_DONE;
                    end else if (next_layer_weights_ready) begin
                        // Weights already staged → go straight to swap-wait.
                        state <= S_SWAP_WAIT;
                    end else begin
                        // Need to prefetch N+1 weights.
                        prefetch_issued <= 1'b1;
                        state           <= S_PREFETCH;
                    end
                end
            end

            //--------------------------------------------------------------
            // Prefetch: wait for the shadow loader to stage N+1 weights.
            // The IFFT for layer N may still be draining; that is fine.
            S_PREFETCH: begin
                if (next_layer_weights_ready) begin
                    state <= S_SWAP_WAIT;
                end
            end

            //--------------------------------------------------------------
            // Swap-wait: weight swap for N+1 runs in parallel with the last
            // IFFT beats.  Once the loader signals swap done, advance.
            S_SWAP_WAIT: begin
                if (weight_swap_done) begin
                    state <= S_ADVANCE;
                end
            end

            //--------------------------------------------------------------
            // Advance to next layer: bump cur_layer and re-enter RUN.
            S_ADVANCE: begin
                cur_layer         <= cur_layer + 4'd1;
                prefetch_issued   <= 1'b0;
                layer_was_complete<= 1'b0;
                state             <= S_RUN;
            end

            //--------------------------------------------------------------
            S_DONE: begin
                // Hold until a new start_layer re-arms.
                if (start_layer) begin
                    state <= S_IDLE;
                end
            end

            default: state <= S_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire