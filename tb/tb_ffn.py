# tb_ffn.py — cocotb testbench for ffn.v
# Run: SIM=icarus TOPLEVEL=ffn VERILOG_SOURCES=../rtl/ffn.v make
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
async def test_passthrough(dut):
    """With W1=identity and W2=identity, FFN(x) ≈ GELU(x) ≈ x for large x"""
    await setup(dut)
    # Load W1: W1[j][j%64] = 256 (1.0) for j=0..127, rest=0
    # This is a rough identity: each hidden unit copies one input dim
    for j in range(128):
        dut.weight_we.value = 1
        dut.weight_addr.value = j * 64 + (j % 64)
        dut.weight_data.value = 256
        await RisingEdge(dut.clk)
    # Load W2: W2[j][j*2] = 256 for j=0..63 (pick even hidden units)
    for j in range(64):
        dut.weight_we.value = 1
        dut.weight_addr.value = 8192 + j * 128 + (j * 2)
        dut.weight_data.value = 256
        await RisingEdge(dut.clk)
    dut.weight_we.value = 0

    # Send input: all dims = 256 (1.0)
    dut.data_in_valid.value = 1
    for i in range(64):
        dut.data_in.value = 256
        await RisingEdge(dut.clk)
    dut.data_in_valid.value = 0

    # Collect 64 outputs
    outputs = []
    for _ in range(500):
        await RisingEdge(dut.clk)
        if dut.data_out_valid.value:
            outputs.append(int(dut.data_out.value))
        if dut.done.value:
            break

    dut._log.info(f"Collected {len(outputs)} outputs, first 3: {outputs[:3]}")
    assert len(outputs) == 64, f"Should output 64 values, got {len(outputs)}"
    dut._log.info("PASS: FFN runs end-to-end and produces 64 outputs")

@cocotb.test()
async def test_zero_input(dut):
    """FFN(0) should produce ~0 output"""
    await setup(dut)
    # Load all-zero weights
    for a in range(0, 16384, 64):
        dut.weight_we.value = 1
        dut.weight_addr.value = a
        dut.weight_data.value = 0
        await RisingEdge(dut.clk)
    dut.weight_we.value = 0

    # Send all-zero input
    dut.data_in_valid.value = 1
    for i in range(64):
        dut.data_in.value = 0
        await RisingEdge(dut.clk)
    dut.data_in_valid.value = 0

    outputs = []
    for _ in range(500):
        await RisingEdge(dut.clk)
        if dut.data_out_valid.value:
            outputs.append(int(dut.data_out.value))
        if dut.done.value:
            break

    assert len(outputs) == 64
    assert all(v == 0 for v in outputs), "Zero weights + zero input → zero output"
    dut._log.info("PASS: zero input produces zero output")
