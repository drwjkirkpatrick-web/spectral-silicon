# tb_residual_add.py — cocotb testbench for residual_add.v
# Run: SIM=icarus TOPLEVEL=residual_add VERILOG_SOURCES=../rtl/residual_add.v make
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

async def setup(dut):
    cocotb.start_soon(Clock(dut.clk, 10, "ns").start())
    dut.rst_n.value = 0
    dut.data_in_valid.value = 0
    dut.mixer_valid.value = 0
    dut.data_out_ready.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

@cocotb.test()
async def test_addition(dut):
    await setup(dut)
    dut.in_re.value = 100
    dut.in_im.value = 50
    dut.mixer_re.value = 200
    dut.mixer_im.value = 75
    dut.data_in_valid.value = 1
    dut.mixer_valid.value = 1
    await RisingEdge(dut.clk)
    dut.data_in_valid.value = 0
    dut.mixer_valid.value = 0
    await RisingEdge(dut.clk)
    assert dut.data_out_valid.value == 1, "output should be valid"
    assert int(dut.out_re.value) == 300, f"re: expected 300, got {int(dut.out_re.value)}"
    assert int(dut.out_im.value) == 125, f"im: expected 125, got {int(dut.out_im.value)}"
    dut._log.info("PASS: correct addition")

@cocotb.test()
async def test_saturation(dut):
    await setup(dut)
    dut.in_re.value = 30000
    dut.in_im.value = 0
    dut.mixer_re.value = 30000
    dut.mixer_im.value = 0
    dut.data_in_valid.value = 1
    dut.mixer_valid.value = 1
    await RisingEdge(dut.clk)
    dut.data_in_valid.value = 0
    dut.mixer_valid.value = 0
    await RisingEdge(dut.clk)
    assert int(dut.out_re.value) == 32767, f"saturated re: expected 32767, got {int(dut.out_re.value)}"
    dut._log.info("PASS: saturation works")

@cocotb.test()
async def test_latency(dut):
    await setup(dut)
    dut.in_re.value = 10
    dut.in_im.value = 0
    dut.mixer_re.value = 20
    dut.mixer_im.value = 0
    dut.data_in_valid.value = 1
    dut.mixer_valid.value = 1
    await RisingEdge(dut.clk)  # input latched
    dut.data_in_valid.value = 0
    dut.mixer_valid.value = 0
    await RisingEdge(dut.clk)  # output should be valid
    assert dut.data_out_valid.value == 1
    assert int(dut.out_re.value) == 30
    dut._log.info("PASS: 1-cycle latency confirmed")
