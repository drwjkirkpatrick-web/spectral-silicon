# tb_topk_sampler.py — cocotb testbench for topk_sampler.v
# Run: SIM=icarus TOPLEVEL=topk_sampler VERILOG_SOURCES=../rtl/topk_sampler.v make
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

async def setup(dut):
    cocotb.start_soon(Clock(dut.clk, 10, "ns").start())
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.data_in_valid.value = 0
    dut.data_out_ready.value = 1
    dut.k.value = 5
    dut.temperature.value = 256
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

@cocotb.test()
async def test_greedy(dut):
    """k=1 always picks argmax"""
    await setup(dut)
    dut.k.value = 1
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    for i in range(128):
        dut.data_in_valid.value = 1
        dut.data_in.value = 500 if i == 42 else 10  # token 42 is largest
        await RisingEdge(dut.clk)
    dut.data_in_valid.value = 0

    for _ in range(20):
        await RisingEdge(dut.clk)
        if dut.done.value:
            break

    assert dut.data_out_valid.value == 1, "should have output"
    idx = int(dut.selected_idx.value)
    dut._log.info(f"Greedy selected token {idx} (expected 42)")
    assert idx == 42, f"Greedy should pick 42, got {idx}"
    dut._log.info("PASS: greedy (k=1) picks argmax")

@cocotb.test()
async def test_topk_selection(dut):
    """k=5: selected index should be one of the top 5"""
    await setup(dut)
    dut.k.value = 5
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Top 5 tokens: 0, 1, 2, 3, 4 with values 500, 499, 498, 497, 496
    for i in range(128):
        dut.data_in_valid.value = 1
        dut.data_in.value = (500 - i) if i < 5 else 10
        await RisingEdge(dut.clk)
    dut.data_in_valid.value = 0

    for _ in range(20):
        await RisingEdge(dut.clk)
        if dut.done.value:
            break

    assert dut.data_out_valid.value == 1
    idx = int(dut.selected_idx.value)
    dut._log.info(f"Top-5 selected token {idx}")
    assert idx < 5, f"Should be in top-5 (0-4), got {idx}"
    dut._log.info("PASS: top-k selection within bounds")
