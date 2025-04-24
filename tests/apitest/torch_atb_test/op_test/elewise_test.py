import torch
import torch.nn as nn
import torch_atb
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from utils import check_float, run_perf_test

def run_test():
    print("----------- elewise test begin ------------")
    input1_cpu = torch.randn(2, 3, dtype=torch.float16)
    input2_cpu = torch.randn(2, 3, dtype=torch.float16)
    input1_npu = input1_cpu.npu()
    input2_npu = input2_cpu.npu()
    elewise_param = torch_atb.ElewiseParam()
    elewise_param.elewise_type = torch_atb.ElewiseParam.ElewiseType.ELEWISE_ADD
    elewise = torch_atb.Operation(elewise_param)

    def elewise_run():
        elewise_outputs = elewise.forward([input1_npu, input2_npu])
        return elewise_outputs

    def golden():
        return [input1_cpu+input2_cpu]

    cpu_goldens = golden()
    print("cpu_goldens: ", cpu_goldens)

    npu_outputs = elewise_run()
    print("npu_outputs: ", npu_outputs)
    
    assert check_float(npu_outputs, cpu_goldens), "Test failed"

    run_perf_test(elewise, [input1_npu, input2_npu])
    print("----------- elewise test success ------------")

if __name__ == "__main__":
    run_test()