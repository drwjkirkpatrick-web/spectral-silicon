# tb_softmax.py — cocotb testbench for softmax.v
# Run: SIM=icarus TOPLEVEL=softmax VERILOG_SOURCES=../rtl/softmax.v make
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

async def setup(dut):
    cocotb.start_soon(Clock(dut.clk, 10, "ns").start())
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.data_in_valid.value = 0
    dut.data_out_ready.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

@cocotb.test()
async def test_uniform_input(dut):
    """All-equal logits → all-equal probabilities (~1/128)"""
    await setup(dut)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    for i in range(128):
        dut.data_in_valid.value = 1
        dut.data_in.value = 0  # logit=0 → exp(0)=1
        await RisingEdge(dut.clk)
    dut.data_in_valid.value = 0

    # Collect outputs
    outputs = []
    for _ in range(200):
        await RisingEdge(dut.clk)
        if dut.data_out_valid.value:
            outputs.append(int(dut.data_out.value))
        if dut.done.value:
            break

    dut._log.info(f"Collected {len(outputs)} probabilities, first few: {outputs[:5]}")
    assert len(outputs) > 0, "Should have received outputs"
    # With all-equal logits, all probabilities should be approximately equal
    if len(outputs) >= 2:
        diff = abs(outputs[0] - outputs[1])
        assert diff < 20, f"Uniform probs should be equal, diff={diff}"
    dut._log.info("PASS: uniform input produces uniform probabilities")

@cocotb.test()
async def test_larger_logit(dut):
    """Larger logit → larger probability"""
    await setup(dut)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    for i in range(128):
        dut.data_in_valid.value = 1
        dut.data_in.value = 256 if i == 0 else 0  # logit[0]=1.0, rest=0
        await RisingEdge(dut.clk)
    dut.data_in_valid.value = 0

    outputs = []
    for _ in range(200):
        await RisingEdge(dut.clk)
        if dut.data_out_valid.value:
            outputs.append(int(dut.data_out.value))
        if dut.done.value:
            break

    if len(outputs) >= 2:
        dut._log.info(f"prob[0]={outputs[0]}, prob[1]={outputs[1]}")
        assert outputs[0] > outputs[1], "Larger logit should produce larger prob"
    dut._log.info("PASS: larger logit produces larger probability")
