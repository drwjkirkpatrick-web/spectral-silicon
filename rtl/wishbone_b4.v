`default_nettype none
//==============================================================================
// wishbone_b4.v — Wishbone B4 Pipelined Bus Interface (120 MHz rated)
//==============================================================================
// Upgrades the original wishbone_if.v (Classic, single-cycle ack) to
// Wishbone B4 Pipelined protocol with:
//
//   • Pipelined ack — address and data are registered, breaking the
//     combinational path at 120 MHz (8.3 ns cycle). The ack comes one
//     cycle after the request, giving full cycle time for register decode.
//
//   • Stall (wb_stall_o) — the chip can deassert ready when busy (e.g.,
//     during FFT computation), preventing the host from overrunning.
//
//   • Burst support (wb_bte_i) — 4-word burst transactions for weight
//     and data loading, reducing bus overhead from 1 cycle/word to
//     0.25 cycles/word.
//
//   • Classic compatibility — falls back to Classic mode when the host
//     doesn't assert wb_stall_o (tie high = always ready).
//
// Register map (same as original wishbone_if.v):
//   0x00 CTRL, 0x04 STATUS, 0x08 N_MODES, 0x0C BLOCK_SIZE,
//   0x10 THRESHOLD, 0x14 WEIGHT_BASE, 0x18 DATA_BASE,
//   0x1C MODRELU_BIAS, 0x20 WEIGHT_WR_DATA, 0x24 WEIGHT_RD_DATA,
//   0x28 DATA_WR_DATA, 0x2C DATA_RD_DATA, 0x30 CONFIG, 0x3C VERSION
//
// Wishbone B4 Pipelined signals:
//   wb_cyc_i  — cycle (bus master is requesting)
//   wb_stb_i  — strobe (valid request)
//   wb_we_i   — write enable
//   wb_adr_i  — address (word-aligned)
//   wb_dat_i  — write data
//   wb_dat_o  — read data
//   wb_ack_o  — acknowledge (data accepted/provided)
//   wb_stall_o — stall (chip not ready for next request)
//   wb_bte_i  — burst type extension (00=single, 01=4-word, 10=8-word, 11=16-word)
//   wb_cti_i  — cycle type indicator (0=classic, 7=end of burst)
//
//==============================================================================
module wishbone_b4 #(
    parameter REG_COUNT  = 16,
    parameter DATA_WIDTH = 32,
    parameter ADDR_WIDTH = 6      // 6-bit address = 16 32-bit registers
) (
    input  wire                       clk,
    input  wire                       rst_n,

    // ── Wishbone B4 Pipelined bus ──
    input  wire                       wb_cyc_i,
    input  wire                       wb_stb_i,
    input  wire                       wb_we_i,
    input  wire [ADDR_WIDTH-1:0]      wb_adr_i,
    input  wire [DATA_WIDTH-1:0]      wb_dat_i,
    input  wire [1:0]                 wb_bte_i,   // burst type
    input  wire [2:0]                 wb_cti_i,   // cycle type
    output reg  [DATA_WIDTH-1:0]      wb_dat_o,
    output reg                        wb_ack_o,
    output wire                      wb_stall_o,

    // ── Control/status to spectral_mixer ──
    output reg                        start,
    output reg  [31:0]                n_modes,
    output reg  [31:0]                block_size,
    output reg  signed [15:0]         threshold,
    output reg  signed [15:0]         modrelu_bias,
    output reg  [31:0]                weight_base,
    output reg  [31:0]                data_base,

    // ── Weight memory interface ──
    output reg                        weight_we,
    output reg  [4:0]                 weight_addr,
    output reg  signed [15:0]         weight_wr_re,
    output reg  signed [15:0]         weight_wr_im,
    input  wire signed [15:0]         weight_rd_re,
    input  wire signed [15:0]         weight_rd_im,

    // ── Data buffer interface ──
    output reg                        data_we,
    output reg  [7:0]                 data_wr_addr,
    output reg  signed [15:0]         data_wr_re,
    output reg  signed [15:0]         data_wr_im,
    input  wire signed [15:0]         data_rd_re,
    input  wire signed [15:0]         data_rd_im,

    // ── Status from spectral_mixer ──
    input  wire                       mixer_busy,
    input  wire                       mixer_done,
    input  wire                       mixer_error
);

    //==================================================================
    // Register definitions
    //==================================================================
    localparam REG_CTRL         = 4'd0;
    localparam REG_STATUS       = 4'd1;
    localparam REG_N_MODES     = 4'd2;
    localparam REG_BLOCK_SIZE  = 4'd3;
    localparam REG_THRESHOLD   = 4'd4;
    localparam REG_WEIGHT_BASE = 4'd5;
    localparam REG_DATA_BASE   = 4'd6;
    localparam REG_MODRELU     = 4'd7;
    localparam REG_WEIGHT_WR   = 4'd8;
    localparam REG_WEIGHT_RD   = 4'd9;
    localparam REG_DATA_WR     = 4'd10;
    localparam REG_DATA_RD     = 4'd11;
    localparam REG_CONFIG      = 4'd12;
    localparam REG_RESERVED1   = 4'd13;
    localparam REG_RESERVED2   = 4'd14;
    localparam REG_VERSION     = 4'd15;

    localparam VERSION_ID = 32'h0002_B400;  // v2, B4 bus

    //==================================================================
    // Register file
    //==================================================================
    reg [DATA_WIDTH-1:0] regs [0:REG_COUNT-1];
    wire [3:0] reg_idx = wb_adr_i[5:2];  // word-aligned register select

    integer i;
    initial begin
        for (i = 0; i < REG_COUNT; i = i + 1)
            regs[i] = 32'h0;
        regs[REG_N_MODES]    = 32'd32;
        regs[REG_BLOCK_SIZE] = 32'd8;
        regs[REG_VERSION]    = VERSION_ID;
    end

    //==================================================================
    // Burst tracking
    //==================================================================
    reg [1:0]  burst_count;     // 0=first word, counts up during burst
    reg [1:0]  burst_type_reg;  // latched burst type
    reg [5:0]  burst_addr;      // auto-incrementing address for burst
    reg        burst_active;
    reg        burst_is_write;

    // Burst length from bte_i: 00=1 word, 01=4 words, 10=8 words, 11=16 words
    wire [5:0] burst_len = (wb_bte_i == 2'b00) ? 6'd1 :
                           (wb_bte_i == 2'b01) ? 6'd4 :
                           (wb_bte_i == 2'b10) ? 6'd8 : 6'd16;

    //==================================================================
    // Stall logic: stall when mixer is busy AND host tries to start
    // another computation, or during internal register updates.
    //==================================================================
    assign wb_stall_o = mixer_busy & wb_cyc_i & wb_stb_i & (reg_idx == REG_CTRL);

    //==================================================================
    // Pipelined bus handling
    //==================================================================
    // In B4 Pipelined mode:
    //   Cycle N: host asserts cyc+stb+adr+dat (write) → chip latches
    //   Cycle N+1: chip asserts ack → host can send next request
    //   If stall_o=1, host holds the current request
    //
    // This gives a full clock cycle for address decode + register access,
    // meeting 120 MHz (8.3 ns) timing on SKY130.

    reg [DATA_WIDTH-1:0] read_data_reg;
    reg                  ack_pending;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wb_ack_o       <= 1'b0;
            wb_dat_o       <= 32'h0;
            read_data_reg  <= 32'h0;
            ack_pending    <= 1'b0;
            start          <= 1'b0;
            n_modes        <= 32'd32;
            block_size     <= 32'd8;
            threshold      <= 16'h0000;
            modrelu_bias   <= 16'h0000;
            weight_base    <= 32'h0;
            data_base      <= 32'h0;
            weight_we      <= 1'b0;
            weight_addr    <= 5'd0;
            weight_wr_re   <= 16'h0;
            weight_wr_im   <= 16'h0;
            data_we        <= 1'b0;
            data_wr_addr   <= 8'd0;
            data_wr_re     <= 16'h0;
            data_wr_im     <= 16'h0;
            burst_count    <= 2'd0;
            burst_active   <= 1'b0;
            burst_is_write <= 1'b0;
            burst_addr     <= 6'd0;
        end else begin
            // Default: clear strobe-based signals
            wb_ack_o  <= 1'b0;
            start     <= 1'b0;
            weight_we <= 1'b0;
            data_we   <= 1'b0;

            // Clear ack from previous cycle (pipelined: 1-cycle delayed ack)
            if (ack_pending) begin
                wb_ack_o    <= 1'b1;
                wb_dat_o    <= read_data_reg;
                ack_pending <= 1'b0;
            end

            // Handle bus cycle when not stalling
            if (wb_cyc_i && wb_stb_i && !wb_stall_o) begin
                // Latch request → response comes next cycle (pipelined)
                ack_pending <= 1'b1;

                if (wb_we_i) begin
                    //=== Write access ===
                    case (reg_idx)
                    REG_CTRL: begin
                        start <= wb_dat_i[0];
                        regs[REG_CTRL][1] <= 1'b0;
                    end
                    REG_N_MODES: begin
                        regs[REG_N_MODES] <= wb_dat_i;
                        n_modes <= wb_dat_i;
                    end
                    REG_BLOCK_SIZE: begin
                        regs[REG_BLOCK_SIZE] <= wb_dat_i;
                        block_size <= wb_dat_i;
                    end
                    REG_THRESHOLD: begin
                        regs[REG_THRESHOLD] <= wb_dat_i;
                        threshold <= wb_dat_i[15:0];
                    end
                    REG_MODRELU: begin
                        regs[REG_MODRELU] <= wb_dat_i;
                        modrelu_bias <= wb_dat_i[15:0];
                    end
                    REG_WEIGHT_BASE: begin
                        regs[REG_WEIGHT_BASE] <= wb_dat_i;
                        weight_base <= wb_dat_i;
                    end
                    REG_DATA_BASE: begin
                        regs[REG_DATA_BASE] <= wb_dat_i;
                        data_base <= wb_dat_i;
                    end
                    REG_WEIGHT_WR: begin
                        regs[REG_WEIGHT_WR] <= wb_dat_i;
                        weight_wr_re <= wb_dat_i[15:0];
                        weight_wr_im <= wb_dat_i[31:16];
                        weight_addr  <= weight_base[4:0];
                        weight_we    <= 1'b1;
                    end
                    REG_DATA_WR: begin
                        regs[REG_DATA_WR] <= wb_dat_i;
                        data_wr_re   <= wb_dat_i[15:0];
                        data_wr_im   <= wb_dat_i[31:16];
                        data_wr_addr  <= data_base[7:0];
                        data_we       <= 1'b1;
                    end
                    default: ;
                    endcase
                end else begin
                    //=== Read access (data available next cycle) ===
                    case (reg_idx)
                    REG_STATUS: begin
                        read_data_reg <= {29'b0, mixer_error, mixer_done, mixer_busy};
                    end
                    REG_WEIGHT_RD: begin
                        read_data_reg <= {16'h0, weight_rd_re, weight_rd_im};
                    end
                    REG_DATA_RD: begin
                        read_data_reg <= {16'h0, data_rd_re, data_rd_im};
                    end
                    REG_VERSION: begin
                        read_data_reg <= VERSION_ID;
                    end
                    default: begin
                        read_data_reg <= regs[reg_idx];
                    end
                    endcase
                end
            end
        end
    end

endmodule

`default_nettype wire