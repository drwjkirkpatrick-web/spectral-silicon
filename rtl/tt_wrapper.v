`default_nettype none
//==============================================================================
// tt_wrapper.v — Tiny Tapeout Wrapper for Spectral Silicon
//==============================================================================
// Wraps the spectral_mixer top-level module in the Tiny Tapeout (TT) template.
// Module name: tt_um_spectral_silicon
//
// Tiny Tapeout pin mapping:
//   ui_in[7:0]   — 8-bit bidirectional input from host (SPI-like protocol)
//   uo_out[7:0]  — 8-bit output to host
//   uio_in[7:0]  — 8-bit bidirectional input (upper byte)
//   uio_out[7:0] — 8-bit bidirectional output
//   uio_oe[7:0]  — output enable for bidirectional pins
//
// SPI-like protocol over ui_in[7:0]:
//   The host sends commands as serial bytes.  A simple SPI-like protocol:
//   - ui_in[7]    = SCK  (clock)
//   - ui_in[6]    = CS_n (chip select, active low)
//   - ui_in[5]    = MOSI (master out, slave in)
//   - ui_in[4:0]  = mode/command bits (direct parallel command)
//   - uo_out[7:0] = MISO + status bits
//
//   Protocol phases:
//   1. Command: host writes 8-bit command + 8-bit address via SPI
//   2. Data:    host writes or reads 32-bit data via SPI (4 bytes)
//
//   Commands:
//   - WRITE_REG: write 32-bit data to Wishbone register at address
//   - READ_REG:  read 32-bit data from Wishbone register at address
//
//   The wrapper translates SPI serial data into Wishbone bus transactions.
//
// Prompt 26 specification.
//==============================================================================
module tt_um_spectral_silicon (
    input  wire [7:0]  ui_in,      // TT input pins
    output reg  [7:0]  uo_out,     // TT output pins
    input  wire [7:0]  uio_in,     // TT bidirectional input
    output reg  [7:0]  uio_out,    // TT bidirectional output
    output wire [7:0]  uio_oe,     // TT bidirectional output enable
    input  wire        ena,        // TT enable (always high in normal operation)
    input  wire        clk,        // TT clock
    input  wire        rst_n      // TT reset (active low)
);

    //----------------------------------------------------------------------
    // SPI-like protocol signals (from ui_in)
    //----------------------------------------------------------------------
    wire spi_sck   = ui_in[7];     // SPI clock
    wire spi_cs_n  = ui_in[6];     // Chip select (active low)
    wire spi_mosi  = ui_in[5];     // Master Out Slave In
    wire [4:0] cmd_bits = ui_in[4:0]; // Direct command bits

    // MISO output
    wire spi_miso;

    //----------------------------------------------------------------------
    // SPI receiver state machine
    // Shifts in 8 bits per byte on SCK rising edge when CS_n is low.
    // Byte sequence: [cmd_byte] [addr_byte] [data3] [data2] [data1] [data0]
    // For reads: after cmd+addr, shift out 4 data bytes on MISO.
    //----------------------------------------------------------------------
    localparam S_IDLE      = 3'd0,
               S_CMD       = 3'd1,   // Receive command byte
               S_ADDR      = 3'd2,   // Receive address byte
               S_WR_DATA   = 3'd3,   // Receive 4 data bytes (write)
               S_WB_ACCESS = 3'd4,   // Execute Wishbone transaction
               S_RD_DATA   = 3'd5;   // Shift out 4 data bytes (read)

    reg [2:0]  spi_state;
    reg [2:0]  byte_cnt;           // Which byte (0..3 for 32-bit data)
    reg [7:0]  shift_reg;           // SPI shift register (input)
    reg [2:0]  bit_cnt;            // Bit counter within byte
    reg        prev_sck;           // Previous SCK for edge detection

    reg [7:0]  cmd_byte;           // Received command
    reg [7:0]  addr_byte;          // Received address
    reg [31:0] data_rx;            // Received data (write)
    reg [31:0] data_tx;            // Data to transmit (read)

    // SPI output shift register
    reg [7:0]  tx_shift;
    reg [2:0]  tx_bit_cnt;

    // Commands
    localparam CMD_WRITE_REG = 8'h01;
    localparam CMD_READ_REG  = 8'h02;

    //----------------------------------------------------------------------
    // Wishbone bus signals
    //----------------------------------------------------------------------
    reg         wb_cyc;
    reg         wb_stb;
    reg         wb_we;
    reg  [5:2]  wb_adr;
    reg  [31:0] wb_dat_i;
    wire [31:0] wb_dat_o;
    wire        wb_ack;

    //----------------------------------------------------------------------
    // Spectral mixer instance
    //----------------------------------------------------------------------
    spectral_mixer #(
        .N(256),
        .D(64),
        .N_MODES(32),
        .BLOCK_SIZE(8),
        .WIDTH(16),
        .FRAC(8)
    ) u_mixer (
        .clk(clk),
        .rst_n(rst_n),
        .wb_cyc_i(wb_cyc),
        .wb_stb_i(wb_stb),
        .wb_we_i(wb_we),
        .wb_adr_i(wb_adr),
        .wb_dat_i(wb_dat_i),
        .wb_dat_o(wb_dat_o),
        .wb_ack_o(wb_ack),
        .busy(mixer_busy),
        .done(mixer_done),
        .error(mixer_error)
    );

    // Internal status wires from mixer
    wire mixer_busy, mixer_done, mixer_error;

    //----------------------------------------------------------------------
    // SPI-like protocol handler
    //----------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            spi_state   <= S_IDLE;
            byte_cnt    <= 0;
            shift_reg   <= 0;
            bit_cnt     <= 0;
            prev_sck    <= 1'b0;
            cmd_byte    <= 0;
            addr_byte   <= 0;
            data_rx     <= 0;
            data_tx     <= 0;
            tx_shift    <= 0;
            tx_bit_cnt  <= 0;
            wb_cyc      <= 1'b0;
            wb_stb      <= 1'b0;
            wb_we       <= 1'b0;
            wb_adr      <= 4'b0;
            wb_dat_i    <= 32'b0;
        end else begin
            prev_sck <= spi_sck;

            case (spi_state)

            //--- Idle: wait for CS_n low ---
            S_IDLE: begin
                if (!spi_cs_n) begin
                    spi_state <= S_CMD;
                    bit_cnt   <= 0;
                    shift_reg <= 0;
                end
            end

            //--- Receive command byte ---
            S_CMD: begin
                if (spi_cs_n) begin
                    // CS deasserted: abort
                    spi_state <= S_IDLE;
                end else if (spi_sck && !prev_sck) begin
                    // Rising edge: shift in MOSI
                    shift_reg <= {shift_reg[6:0], spi_mosi};
                    if (bit_cnt == 3'd7) begin
                        cmd_byte  <= {shift_reg[6:0], spi_mosi};
                        bit_cnt   <= 0;
                        spi_state <= S_ADDR;
                    end else begin
                        bit_cnt <= bit_cnt + 1;
                    end
                end
            end

            //--- Receive address byte ---
            S_ADDR: begin
                if (spi_cs_n) begin
                    spi_state <= S_IDLE;
                end else if (spi_sck && !prev_sck) begin
                    shift_reg <= {shift_reg[6:0], spi_mosi};
                    if (bit_cnt == 3'd7) begin
                        addr_byte <= {shift_reg[6:0], spi_mosi};
                        bit_cnt   <= 0;
                        // Decide write or read based on command
                        if (cmd_byte == CMD_WRITE_REG) begin
                            spi_state <= S_WR_DATA;
                            byte_cnt  <= 0;
                        end else if (cmd_byte == CMD_READ_REG) begin
                            spi_state <= S_WB_ACCESS;
                            // Issue Wishbone read
                            wb_cyc   <= 1'b1;
                            wb_stb   <= 1'b1;
                            wb_we    <= 1'b0;
                            wb_adr   <= addr_byte[5:2];
                        end else begin
                            // Unknown command: return to idle
                            spi_state <= S_IDLE;
                        end
                    end else begin
                        bit_cnt <= bit_cnt + 1;
                    end
                end
            end

            //--- Receive 4 data bytes (write) ---
            S_WR_DATA: begin
                if (spi_cs_n) begin
                    spi_state <= S_IDLE;
                end else if (spi_sck && !prev_sck) begin
                    shift_reg <= {shift_reg[6:0], spi_mosi};
                    if (bit_cnt == 3'd7) begin
                        // Assemble 32-bit data (MSB first: byte 3 → byte 0)
                        data_rx[(31 - byte_cnt*8) -: 8] <= {shift_reg[6:0], spi_mosi};
                        bit_cnt <= 0;
                        if (byte_cnt == 2'd3) begin
                            // All 4 bytes received: issue Wishbone write
                            spi_state <= S_WB_ACCESS;
                            wb_cyc    <= 1'b1;
                            wb_stb    <= 1'b1;
                            wb_we     <= 1'b1;
                            wb_adr    <= addr_byte[5:2];
                            wb_dat_i  <= {shift_reg[6:0], spi_mosi, data_rx[23:0]};
                            byte_cnt  <= 0;
                        end else begin
                            byte_cnt <= byte_cnt + 1;
                        end
                    end else begin
                        bit_cnt <= bit_cnt + 1;
                    end
                end
            end

            //--- Execute Wishbone transaction (wait for ack) ---
            S_WB_ACCESS: begin
                if (wb_ack) begin
                    wb_cyc <= 1'b0;
                    wb_stb <= 1'b0;
                    if (wb_we) begin
                        // Write complete: return to idle
                        spi_state <= S_IDLE;
                    end else begin
                        // Read complete: prepare to shift out data
                        data_tx   <= wb_dat_o;
                        tx_shift  <= wb_dat_o[31:24];  // MSB first
                        tx_bit_cnt <= 0;
                        spi_state  <= S_RD_DATA;
                        byte_cnt   <= 0;
                    end
                end
            end

            //--- Shift out 4 data bytes (read response) ---
            S_RD_DATA: begin
                if (spi_cs_n) begin
                    spi_state <= S_IDLE;
                end else if (spi_sck && !prev_sck) begin
                    // On rising edge: shift out next bit on MISO (falling edge)
                    // (We output data on rising edge for simplicity)
                    if (tx_bit_cnt == 3'd7) begin
                        tx_bit_cnt <= 0;
                        if (byte_cnt == 2'd3) begin
                            spi_state <= S_IDLE;  // All 4 bytes sent
                        end else begin
                            byte_cnt <= byte_cnt + 1;
                            // Load next byte
                            case (byte_cnt)
                                2'd0: tx_shift <= data_tx[23:16];
                                2'd1: tx_shift <= data_tx[15:8];
                                2'd2: tx_shift <= data_tx[7:0];
                                default: tx_shift <= 0;
                            endcase
                        end
                    end else begin
                        tx_shift   <= {tx_shift[6:0], 1'b0};
                        tx_bit_cnt <= tx_bit_cnt + 1;
                    end
                end
            end

            default: spi_state <= S_IDLE;
            endcase
        end
    end

    //----------------------------------------------------------------------
    // Output pin assignments
    //----------------------------------------------------------------------
    // MISO is the MSB of the current TX shift register
    assign spi_miso = tx_shift[7];

    // uo_out: MISO + status
    always @(*) begin
        uo_out[7] = spi_miso;              // MISO
        uo_out[6] = wb_ack;                // Wishbone ack
        uo_out[5] = ~spi_cs_n;             // CS active indicator
        uo_out[4] = (spi_state != S_IDLE);  // Busy
        uo_out[3:0] = 4'h0;                // Unused
    end

    // uio_out: status and debug
    always @(*) begin
        uio_out[7] = mixer_busy;
        uio_out[6] = mixer_done;
        uio_out[5] = mixer_error;
        uio_out[4:0] = 5'h0;               // Unused
    end

    // uio_oe: output enable for bidirectional pins — always all outputs
    assign uio_oe = 8'hFF;

endmodule