import torch
import torch_atb
import sys
import os
import re

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from utils import check_float, run_perf_test
import unittest
import logging

def is910B():
    device_name = torch.npu.get_device_name()
    return (re.search("Ascend910B", device_name, re.I) and len(device_name) > 10)

def run_test():
    if not is910B():
        print("This test case only supports 910B")
        return True
    print("----------- cumsum test begin ------------")
    axes = [1]
    intensor_cpu = torch.randn(2, 16, dtype=torch.float16)
    intensor_npu = intensor_cpu.npu()
    cumsum_param = torch_atb.CumsumParam(axes = axes)
    cumsum = torch_atb.Operation(cumsum_param)
    logging.info(cumsum_param)

    def cumsum_run():
        cumsum_outputs = cumsum.forward([intensor_npu])
        return cumsum_outputs

    def golden():
        return [torch.cumsum(intensor_npu, dim = axes[0]).cpu()]

    cpu_goldens = golden()
    logging.info("cpu_goldens: ", cpu_goldens)

    npu_outputs = cumsum_run()
    logging.info("npu_outputs: ", npu_outputs)
    
    assert check_float(npu_outputs, cpu_goldens), "Test failed"
    run_perf_test(cumsum, [intensor_npu])
    print("----------- cumsum test success ------------")

class TestGraph(unittest.TestCase):
    def test(self):
        run_test()

if __name__ == "__main__":
    unittest.main()