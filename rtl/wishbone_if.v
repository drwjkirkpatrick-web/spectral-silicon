`default_nettype none
//==============================================================================
// wishbone_if.v — Wishbone-Compatible Register Interface
//==============================================================================
// Implements a Classic Wishbone B4 register interface with 16 32-bit registers.
//
// Wishbone signals:
//   wb_cyc_i  — bus cycle
//   wb_stb_i  — strobe (request)
//   wb_we_i   — write enable
//   wb_adr_i  — address (word-aligned, 4-bit register select)
//   wb_dat_i  — data input (write data)
//   wb_dat_o  — data output (read data)
//   wb_ack_o  — acknowledge
//
// Register map (32-bit registers, addressed by wb_adr_i[5:2]):
//   0x00 (0)  CTRL      — [0]=start (write 1 to trigger), [1]=done (read)
//   0x04 (1)  STATUS    — [0]=busy, [1]=done, [2]=error
//   0x08 (2)  N_MODES   — number of spectral modes (default 32)
//   0x0C (3)  BLOCK_SIZE— block-diagonal block size (default 8)
//   0x10 (4)  THRESHOLD — soft-thresholding threshold (Q8.8)
//   0x14 (5)  WEIGHT_BASE — base address for weight memory access
//   0x18 (6)  DATA_BASE  — base address for data buffer access
//   0x1C (7)  MODRELU_BIAS — modReLU bias value (Q8.8)
//   0x20 (8)  WEIGHT_WR_DATA — write data for weight (re[15:0], im[31:16])
//   0x24 (9)  WEIGHT_RD_DATA — read data from weight (re[15:0], im[31:16])
//   0x28 (10) DATA_WR_DATA   — write data for input/output buffer
//   0x2C (11) DATA_RD_DATA   — read data from input/output buffer
//   0x30 (12) CONFIG       — [7:0]=WIDTH, [15:8]=FRAC, [31:16]=N
//   0x34 (13) RESERVED
//   0x38 (14) RESERVED
//   0x3C (15) VERSION      — hardware version ID (read-only)
//
// Prompt 20 specification.
//==============================================================================
module wishbone_if #(
    parameter REG_COUNT = 16,       // 16 registers
    parameter DATA_WIDTH = 32        // 32-bit Wishbone data
) (
    input  wire                    clk,
    input  wire                    rst_n,

    // Wishbone Classic bus
    input  wire                    wb_cyc_i,
    input  wire                    wb_stb_i,
    input  wire                    wb_we_i,
    input  wire [5:2]              wb_adr_i,   // Register select (word address)
    input  wire [DATA_WIDTH-1:0]   wb_dat_i,
    output reg  [DATA_WIDTH-1:0]   wb_dat_o,
    output reg                     wb_ack_o,

    // Control/status interface to spectral_mixer
    output reg                     start,          // Start computation
    output reg  [31:0]             n_modes,        // Number of modes
    output reg  [31:0]             block_size,     // Block size
    output reg  signed [15:0]      threshold,      // Soft-threshold value
    output reg  signed [15:0]      modrelu_bias,   // modReLU bias
    output reg  [31:0]             weight_base,    // Weight memory base addr
    output reg  [31:0]             data_base,      // Data buffer base addr

    // Weight memory access (for spectral_multiply weight loading)
    output reg                     weight_we,       // Write enable to weight file
    output reg  [4:0]              weight_addr,    // Weight index 0..31
    output reg  signed [15:0]      weight_wr_re,   // Weight real part
    output reg  signed [15:0]      weight_wr_im,  // Weight imag part
    input  wire signed [15:0]      weight_rd_re,   // Weight read-back real
    input  wire signed [15:0]      weight_rd_im,  // Weight read-back imag

    // Data buffer access (for input/output streaming)
    output reg                     data_we,
    output reg  [7:0]              data_wr_addr,    // Data write address
    output reg  signed [15:0]      data_wr_re,
    output reg  signed [15:0]      data_wr_im,
    input  wire signed [15:0]      data_rd_re,
    input  wire signed [15:0]      data_rd_im,

    // Status from spectral_mixer
    input  wire                    mixer_busy,
    input  wire                    mixer_done,
    input  wire                    mixer_error
);

    //----------------------------------------------------------------------
    // Register storage
    //----------------------------------------------------------------------
    reg [DATA_WIDTH-1:0] regs [0:REG_COUNT-1];

    // Register indices
    localparam REG_CTRL          = 4'd0;
    localparam REG_STATUS        = 4'd1;
    localparam REG_N_MODES       = 4'd2;
    localparam REG_BLOCK_SIZE    = 4'd3;
    localparam REG_THRESHOLD     = 4'd4;
    localparam REG_WEIGHT_BASE   = 4'd5;
    localparam REG_DATA_BASE     = 4'd6;
    localparam REG_MODRELU_BIAS  = 4'd7;
    localparam REG_WEIGHT_WR     = 4'd8;
    localparam REG_WEIGHT_RD     = 4'd9;
    localparam REG_DATA_WR       = 4'd10;
    localparam REG_DATA_RD       = 4'd11;
    localparam REG_CONFIG        = 4'd12;
    localparam REG_RESERVED1     = 4'd13;
    localparam REG_RESERVED2     = 4'd14;
    localparam REG_VERSION       = 4'd15;

    // Register index from Wishbone address (bits [5:2])
    wire [3:0] reg_idx = wb_adr_i[5:2];

    // Default values for read-only registers
    localparam [31:0] VERSION_ID = 32'h0001_0000;  // v1.0
    localparam [31:0] CONFIG_DEFAULT = 32'h0100_0008; // N=256(will be 16 bits), FRAC=8, WIDTH=16

    integer j;
    initial begin
        for (j = 0; j < REG_COUNT; j = j + 1)
            regs[j] = 32'h0;
        regs[REG_N_MODES]    = 32'd32;
        regs[REG_BLOCK_SIZE] = 32'd8;
        regs[REG_CONFIG]     = CONFIG_DEFAULT;
        regs[REG_VERSION]    = VERSION_ID;
    end

    //----------------------------------------------------------------------
    // Wishbone bus handling (Classic protocol, single-cycle ack)
    //----------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wb_ack_o       <= 1'b0;
            wb_dat_o       <= 32'h0;
            start          <= 1'b0;
            n_modes        <= 32'd32;
            block_size     <= 32'd8;
            threshold      <= 16'h0000;
            modrelu_bias   <= 16'h0000;
            weight_base    <= 32'h0;
            data_base     <= 32'h0;
            weight_we      <= 1'b0;
            weight_addr    <= 5'd0;
            weight_wr_re   <= 16'h0;
            weight_wr_im   <= 16'h0;
            data_we         <= 1'b0;
            data_wr_addr    <= 8'd0;
            data_wr_re      <= 16'h0;
            data_wr_im      <= 16'h0;
        end else begin
            // Default: clear strobe-based signals
            wb_ack_o  <= 1'b0;
            start     <= 1'b0;
            weight_we <= 1'b0;
            data_we   <= 1'b0;

            // Handle Wishbone cycle
            if (wb_cyc_i && wb_stb_i) begin
                wb_ack_o <= 1'b1;  // Single-cycle acknowledge

                if (wb_we_i) begin
                    //--- Write access ---
                    case (reg_idx)
                    REG_CTRL: begin
                        // Bit 0 = start
                        start <= wb_dat_i[0];
                        regs[REG_CTRL][1] <= 1'b0; // Clear done on write
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
                    REG_MODRELU_BIAS: begin
                        regs[REG_MODRELU_BIAS] <= wb_dat_i;
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
                        // Write weight: [15:0]=re, [31:16]=im
                        regs[REG_WEIGHT_WR] <= wb_dat_i;
                        weight_wr_re <= wb_dat_i[15:0];
                        weight_wr_im <= wb_dat_i[31:16];
                        weight_addr  <= weight_base[4:0];
                        weight_we    <= 1'b1;
                    end
                    REG_DATA_WR: begin
                        // Write data: [15:0]=re, [31:16]=im
                        regs[REG_DATA_WR] <= wb_dat_i;
                        data_wr_re   <= wb_dat_i[15:0];
                        data_wr_im   <= wb_dat_i[31:16];
                        data_wr_addr  <= data_base[7:0];
                        data_we       <= 1'b1;
                    end
                    // Read-only and reserved registers: ignore writes
                    default: ; // no-op
                    endcase
                end else begin
                    //--- Read access ---
                    case (reg_idx)
                    REG_STATUS: begin
                        wb_dat_o <= {29'b0, mixer_error, mixer_done, mixer_busy};
                    end
                    REG_WEIGHT_RD: begin
                        // Read current weight from weight_base address
                        wb_dat_o <= {16'h0, weight_rd_re, weight_rd_im};
                    end
                    REG_DATA_RD: begin
                        wb_dat_o <= {16'h0, data_rd_re, data_rd_im};
                    end
                    REG_VERSION: begin
                        wb_dat_o <= VERSION_ID;
                    end
                    default: begin
                        wb_dat_o <= regs[reg_idx];
                    end
                    endcase
                end
            end
        end
    end

endmodule