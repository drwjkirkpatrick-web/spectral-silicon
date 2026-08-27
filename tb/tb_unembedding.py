# tb_unembedding.py — cocotb testbench for unembedding.v
# Run: SIM=icarus TOPLEVEL=unembedding VERILOG_SOURCES=../rtl/unembedding.v make
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

async def setup(dut):
    cocotb.start_soon(Clock(dut.clk, 10, "ns").start())
    dut.rst_n.value = 0
    dut.weight_we.value = 0
    dut.data_in_valid.value = 0
    dut.data_out_ready.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

@cocotb.test()
async def test_identity_weights(dut):
    """With identity-like weights (W[j][0]=256, rest=0), logit[j] = x[0]"""
    await setup(dut)
    # Load weights: W[j][0] = 256 (1.0), all others = 0
    for j in range(128):
        dut.weight_we.value = 1
        dut.weight_addr.value = j * 64  # W[j][0]
        dut.weight_data.value = 256
        await RisingEdge(dut.clk)
    dut.weight_we.value = 0

    # Send input: x[0] = 100, x[1..63] = 0
    dut.data_in_valid.value = 1
    dut.data_in.value = 100
    await RisingEdge(dut.clk)
    for i in range(63):
        dut.data_in.value = 0
        await RisingEdge(dut.clk)
    dut.data_in_valid.value = 0

    # Collect 128 logits
    outputs = []
    for _ in range(200):
        await RisingEdge(dut.clk)
        if dut.data_out_valid.value:
            outputs.append(int(dut.data_out.value))
        if dut.done.value:
            break

    dut._log.info(f"Collected {len(outputs)} logits, first 3: {outputs[:3]}")
    assert len(outputs) == 128, f"Should output 128 logits, got {len(outputs)}"
    # All logits should be 100 (256*100 >> 8 = 100)
    assert outputs[0] == 100, f"logit[0] should be 100, got {outputs[0]}"
    dut._log.info("PASS: identity weights produce correct logits")
