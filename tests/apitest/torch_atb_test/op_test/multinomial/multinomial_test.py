import torch
import torch_atb
import sys
import os
import re

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from utils import run_perf_test
import unittest
import logging

def is910B():
    device_name = torch.npu.get_device_name()
    return (re.search("Ascend910B", device_name, re.I) and len(device_name) > 10)

def run_test():
    if not is910B():
        print("This test case only supports 910B")
        return True
    print("----------- multinomial test begin ------------")
    rand_seed = 123
    intensor = torch.rand(3, 3, dtype=torch.float16)
    normalized_tensor = intensor / intensor.sum()
    normalized_tensor_npu = normalized_tensor.npu()
    multinomial_param = torch_atb.MultinomialParam(rand_seed = rand_seed)
    multinomial = torch_atb.Operation(multinomial_param)
    logging.info(multinomial_param)

    def multinomial_run():
        multinomial_outputs = multinomial.forward([normalized_tensor_npu])
        return multinomial_outputs

    npu_outputs = multinomial_run()
    logging.info("npu_outputs: ", npu_outputs)

    run_perf_test(multinomial, [normalized_tensor_npu])
    print("----------- multinomial test success ------------")

class TestGraph(unittest.TestCase):
    def test(self):
        run_test()

if __name__ == "__main__":
    unittest.main()