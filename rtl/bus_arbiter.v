`default_nettype none
//==============================================================================
// bus_arbiter.v — Bus Arbiter for Shared Chip Resources
//==============================================================================
// Arbitrates between multiple masters (host SPI, DMA controller)
// accessing the Wishbone B4 bus. Round-robin priority scheme.
// At 120 MHz, prevents bus contention that would cause data corruption.
//
// Verilog-2005 compliant: no unpacked array ports — uses flat buses.
//==============================================================================
module bus_arbiter #(
    parameter N_MASTERS = 2
) (
    input  wire                    clk,
    input  wire                    rst_n,

    // Master 0 (host)
    input  wire                    m0_cyc,
    input  wire                    m0_stb,
    input  wire                    m0_we,
    input  wire [5:0]             m0_adr,
    input  wire [31:0]            m0_dat,

    // Master 1 (DMA)
    input  wire                    m1_cyc,
    input  wire                    m1_stb,
    input  wire                    m1_we,
    input  wire [5:0]             m1_adr,
    input  wire [31:0]            m1_dat,

    // Grant to selected master
    output reg                     gnt_cyc,
    output reg                     gnt_stb,
    output reg                     gnt_we,
    output reg  [5:0]              gnt_adr,
    output reg  [31:0]             gnt_dat,
    output reg                     gnt_sel,    // 0=master0, 1=master1

    // Ack back from bus
    input  wire                    ack,
    input  wire [31:0]             dat_o,
    output reg                     ack_m0,
    output reg                     ack_m1,
    output reg  [31:0]             dat_o_m0,
    output reg  [31:0]             dat_o_m1
);

    reg current_master;  // 0 or 1

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_master <= 1'b0;
            gnt_cyc <= 1'b0;
            gnt_stb <= 1'b0;
            gnt_we  <= 1'b0;
            gnt_adr <= 6'd0;
            gnt_dat <= 32'd0;
            gnt_sel <= 1'b0;
            ack_m0  <= 1'b0;
            ack_m1  <= 1'b0;
            dat_o_m0 <= 32'd0;
            dat_o_m1 <= 32'd0;
        end else begin
            // Default: clear acks
            ack_m0 <= 1'b0;
            ack_m1 <= 1'b0;

            // Grant to current master if it's requesting
            if (current_master == 1'b0) begin
                // Master 0 has priority
                if (m0_cyc && m0_stb) begin
                    gnt_cyc <= 1'b1;
                    gnt_stb <= 1'b1;
                    gnt_we  <= m0_we;
                    gnt_adr <= m0_adr;
                    gnt_dat <= m0_dat;
                    gnt_sel <= 1'b0;
                end else begin
                    gnt_cyc <= 1'b0;
                    gnt_stb <= 1'b0;
                    // Switch to master 1 if it's requesting
                    if (m1_cyc && m1_stb) begin
                        current_master <= 1'b1;
                    end
                end

                // Route ack
                if (ack) begin
                    ack_m0 <= 1'b1;
                    dat_o_m0 <= dat_o;
                end
            end else begin
                // Master 1 has priority
                if (m1_cyc && m1_stb) begin
                    gnt_cyc <= 1'b1;
                    gnt_stb <= 1'b1;
                    gnt_we  <= m1_we;
                    gnt_adr <= m1_adr;
                    gnt_dat <= m1_dat;
                    gnt_sel <= 1'b1;
                end else begin
                    gnt_cyc <= 1'b0;
                    gnt_stb <= 1'b0;
                    // Switch to master 0 if it's requesting
                    if (m0_cyc && m0_stb) begin
                        current_master <= 1'b0;
                    end
                end

                // Route ack
                if (ack) begin
                    ack_m1 <= 1'b1;
                    dat_o_m1 <= dat_o;
                end
            end
        end
    end

endmodule

`default_nettype wire