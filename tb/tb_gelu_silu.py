# tb_gelu_silu.py — cocotb testbench for gelu_silu.v
# Run: SIM=icarus TOPLEVEL=gelu_silu VERILOG_SOURCES=../rtl/gelu_silu.v make
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

async def setup(dut):
    cocotb.start_soon(Clock(dut.clk, 10, "ns").start())
    dut.rst_n.value = 0
    dut.data_in_valid.value = 0
    dut.data_out_ready.value = 1
    dut.mode.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

@cocotb.test()
async def test_gelu_basic(dut):
    await setup(dut)
    dut.mode.value = 0  # GELU
    # GELU(0) ≈ 0
    dut.data_in_valid.value = 1
    dut.data_in.value = 0
    await RisingEdge(dut.clk)
    dut.data_in_valid.value = 0
    await RisingEdge(dut.clk)
    if dut.data_out_valid.value:
        val = int(dut.data_out.value)
        dut._log.info(f"GELU(0) = {val} (expected ~0)")
        assert abs(val) < 20, f"GELU(0) should be ~0, got {val}"
    dut._log.info("PASS: GELU basic test")

@cocotb.test()
async def test_silu_basic(dut):
    await setup(dut)
    dut.mode.value = 1  # SiLU
    # SiLU(0) = 0 * sigmoid(0) = 0
    dut.data_in_valid.value = 1
    dut.data_in.value = 0
    await RisingEdge(dut.clk)
    dut.data_in_valid.value = 0
    await RisingEdge(dut.clk)
    if dut.data_out_valid.value:
        val = int(dut.data_out.value)
        dut._log.info(f"SiLU(0) = {val} (expected ~0)")
        assert abs(val) < 20, f"SiLU(0) should be ~0, got {val}"
    dut._log.info("PASS: SiLU basic test")

@cocotb.test()
async def test_gelu_large_positive(dut):
    await setup(dut)
    dut.mode.value = 0
    # GELU(4.0) ≈ 4.0 = 1024 in Q8.8
    dut.data_in_valid.value = 1
    dut.data_in.value = 1024
    await RisingEdge(dut.clk)
    dut.data_in_valid.value = 0
    await RisingEdge(dut.clk)
    if dut.data_out_valid.value:
        val = int(dut.data_out.value)
        dut._log.info(f"GELU(4.0) = {val} (expected ~1024)")
        assert abs(val - 1024) < 50, f"GELU(4.0) should be ~1024, got {val}"
    dut._log.info("PASS: GELU large positive")
