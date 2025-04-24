import torch
import torch.nn as nn
import torch_atb
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from utils import check_float, run_perf_test

def run_test():
    print("----------- linear test begin ------------")
    m, k, n = 3, 4, 5
    input_tensor = torch.randn(m, k, dtype=torch.float16)
    weight_tensor = torch.randn(k, n, dtype=torch.float16)
    input = input_tensor.npu()
    weight = weight_tensor.npu()

    linear_param = torch_atb.LinearParam()
    linear_param.has_bias = False
    linear_param.transpose_b = False
    linear = torch_atb.Operation(linear_param)

    def linear_run():
        linear_outputs = linear.forward([input, weight])
        return [linear_outputs[0].to(torch.float32)]
    
    def golden():
        cpu_golden = torch.matmul(input_tensor.to(torch.float32), weight_tensor.to(torch.float32))
        return [cpu_golden]

    cpu_goldens = golden()
    print("cpu_goldens: ", cpu_goldens)

    npu_outputs = linear_run()
    print("npu_outputs: ", npu_outputs)

    assert check_float(npu_outputs, cpu_goldens), "Test failed"

    run_perf_test(linear, [input, weight])
    print("----------- linear test success ------------")

if __name__ == "__main__":
    run_test()