# tb_weight_cache.py — cocotb testbench for weight_cache.v
# Run: SIM=icarus TOPLEVEL=weight_cache VERILOG_SOURCES=../rtl/weight_cache.v make
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

async def setup(dut):
    cocotb.start_soon(Clock(dut.clk, 10, "ns").start())
    dut.rst_n.value = 0
    dut.write_en.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

@cocotb.test()
async def test_write_read(dut):
    """Write to layer 0, read back"""
    await setup(dut)
    # Select layer 0
    dut.layer_sel.value = 0
    # Write mode 0 re = 100
    dut.write_en.value = 1
    dut.write_addr.value = 0  # {mode=0, re_im=0}
    dut.write_data.value = 100
    await RisingEdge(dut.clk)
    # Write mode 0 im = 200
    dut.write_addr.value = 1  # {mode=0, re_im=1}
    dut.write_data.value = 200
    await RisingEdge(dut.clk)
    dut.write_en.value = 0
    # Read back
    dut.read_addr.value = 0
    await RisingEdge(dut.clk)
    assert int(dut.read_data_re.value) == 100, f"re: expected 100, got {int(dut.read_data_re.value)}"
    assert int(dut.read_data_im.value) == 200, f"im: expected 200, got {int(dut.read_data_im.value)}"
    dut._log.info("PASS: write and readback correct")

@cocotb.test()
async def test_layer_isolation(dut):
    """Write to layer 0 doesn't affect layer 1"""
    await setup(dut)
    # Write to layer 0, mode 0 re = 111
    dut.layer_sel.value = 0
    dut.write_en.value = 1
    dut.write_addr.value = 0
    dut.write_data.value = 111
    await RisingEdge(dut.clk)
    # Write to layer 1, mode 0 re = 222
    dut.layer_sel.value = 1
    dut.write_addr.value = 0
    dut.write_data.value = 222
    await RisingEdge(dut.clk)
    dut.write_en.value = 0
    # Read layer 0
    dut.layer_sel.value = 0
    dut.read_addr.value = 0
    await RisingEdge(dut.clk)
    assert int(dut.read_data_re.value) == 111, f"layer 0: expected 111, got {int(dut.read_data_re.value)}"
    # Read layer 1
    dut.layer_sel.value = 1
    await RisingEdge(dut.clk)
    assert int(dut.read_data_re.value) == 222, f"layer 1: expected 222, got {int(dut.read_data_re.value)}"
    dut._log.info("PASS: layer isolation works")
