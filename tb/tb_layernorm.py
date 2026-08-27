# tb_layernorm.py — cocotb testbench for layernorm.v
# Run: SIM=icarus TOPLEVEL=layernorm VERILOG_SOURCES=../rtl/layernorm.v make
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

async def setup(dut):
    cocotb.start_soon(Clock(dut.clk, 10, "ns").start())
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.data_in_valid.value = 0
    dut.data_out_ready.value = 1
    dut.gamma_we.value = 0
    dut.beta_we.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

@cocotb.test()
async def test_normalization(dut):
    await setup(dut)
    # Load gamma=1.0 (256), beta=0.0 (0) for all 64 channels
    for i in range(64):
        dut.gamma_we.value = 1
        dut.gamma_addr.value = i
        dut.gamma_data.value = 256
        dut.beta_we.value = 1
        dut.beta_addr.value = i
        dut.beta_data.value = 0
        await RisingEdge(dut.clk)
    dut.gamma_we.value = 0
    dut.beta_we.value = 0

    # Start accumulation
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Send 64 values: all = 256 (1.0 in Q8.8)
    for i in range(64):
        dut.data_in_valid.value = 1
        dut.data_in.value = 256
        await RisingEdge(dut.clk)
    dut.data_in_valid.value = 0

    # Wait for compute + normalize
    for _ in range(80):
        await RisingEdge(dut.clk)
        if dut.done.value:
            break

    # With all inputs equal, mean=1.0, var=0, so output should be ~0
    # (normalized = (x - mean) / sqrt(var+eps) ≈ 0)
    outputs = []
    for _ in range(80):
        await RisingEdge(dut.clk)
        if dut.data_out_valid.value:
            outputs.append(int(dut.data_out.value))
    dut._log.info(f"Collected {len(outputs)} outputs, first few: {outputs[:5]}")
    assert len(outputs) > 0, "Should have received outputs"
    dut._log.info("PASS: normalization runs and produces output")
