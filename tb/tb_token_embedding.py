# tb_token_embedding.py — cocotb testbench for token_embedding.v
# Run: SIM=icarus TOPLEVEL=token_embedding VERILOG_SOURCES=../rtl/token_embedding.v make
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

async def setup(dut):
    cocotb.start_soon(Clock(dut.clk, 10, "ns").start())
    dut.rst_n.value = 0
    dut.emb_we.value = 0
    dut.start.value = 0
    dut.data_out_ready.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

@cocotb.test()
async def test_embedding_lookup(dut):
    await setup(dut)
    # Load embedding for token 0: dim i = i*10
    for i in range(64):
        dut.emb_we.value = 1
        dut.emb_addr.value = i  # token_id=0, dim=i → flat addr = 0*64+i
        dut.emb_data.value = i * 10
        await RisingEdge(dut.clk)
    dut.emb_we.value = 0

    # Look up token 0
    dut.token_id.value = 0
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    outputs = []
    for _ in range(80):
        await RisingEdge(dut.clk)
        if dut.data_out_valid.value:
            outputs.append(int(dut.data_out.value))
        if dut.done.value:
            break

    dut._log.info(f"Collected {len(outputs)} dims, first 5: {outputs[:5]}")
    assert len(outputs) == 64, f"Should output 64 dims, got {len(outputs)}"
    assert outputs[0] == 0, f"dim 0 should be 0, got {outputs[0]}"
    assert outputs[1] == 10, f"dim 1 should be 10, got {outputs[1]}"
    assert outputs[5] == 50, f"dim 5 should be 50, got {outputs[5]}"
    dut._log.info("PASS: correct embedding lookup")

@cocotb.test()
async def test_different_tokens(dut):
    await setup(dut)
    # Load token 0: all dims = 100
    for i in range(64):
        dut.emb_we.value = 1
        dut.emb_addr.value = i
        dut.emb_data.value = 100
        await RisingEdge(dut.clk)
    # Load token 1: all dims = 200
    for i in range(64):
        dut.emb_we.value = 1
        dut.emb_addr.value = 64 + i  # token_id=1, flat addr = 1*64+i
        dut.emb_data.value = 200
        await RisingEdge(dut.clk)
    dut.emb_we.value = 0

    # Look up token 1
    dut.token_id.value = 1
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    outputs = []
    for _ in range(80):
        await RisingEdge(dut.clk)
        if dut.data_out_valid.value:
            outputs.append(int(dut.data_out.value))
        if dut.done.value:
            break

    assert len(outputs) == 64
    assert outputs[0] == 200, f"token 1 dim 0 should be 200, got {outputs[0]}"
    dut._log.info("PASS: different tokens produce different embeddings")
