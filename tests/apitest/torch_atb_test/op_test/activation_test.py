import torch
import torch.nn as nn
import torch_atb
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from utils import check_float, run_perf_test

def run_test():
    print("----------- activation test begin ------------")
    activation_param = torch_atb.ActivationParam()
    activation_param.activation_type = torch_atb.ActivationType.ACTIVATION_SWISH
    activation_param.scale = 1.0
    activation = torch_atb.Operation(activation_param)

    intensor = torch.rand(2, 3, 5).bfloat16()
    intensor_npu = intensor.npu()

    def activation_run():
        activation_outputs = activation.forward([intensor_npu])
        return activation_outputs

    def golden():
        intensor_float = intensor.float()
        outtensor = intensor_float / (1 + torch.exp(-intensor_float * 1.0))
        return [outtensor.bfloat16()]

    cpu_goldens = golden()
    print("cpu_goldens: ", cpu_goldens)

    npu_outputs = activation_run()
    print("npu_outputs: ", npu_outputs)
    
    assert check_float(npu_outputs, cpu_goldens), "Test failed"

    run_perf_test(activation, [intensor_npu])
    print("----------- activation test success ------------")

if __name__ == "__main__":
    run_test()