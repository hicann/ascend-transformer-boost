import torch
import torch.nn as nn
import torch_atb  
from utils import check_move, run_perf_test

def run_test():
    print("----------- split test begin ------------")
    splitDim = 0
    splitNum = 2
    input_cpu = torch.randn(6, 6, dtype=torch.float16)
    input_npu = input_cpu.npu()
    split_param = torch_atb.SplitParam()
    split_param.split_dim = splitDim
    split_param.split_num = splitNum
    print(split_param)
    split = torch_atb.Operation(split_param)
    
    def split_run():
        split_outputs = split.forward([input_npu])
        return split_outputs

    split_output = torch.chunk(input_cpu, chunks=splitNum, dim=splitDim)
    cpu_goldens = torch.stack(split_output)
    print("cpu_goldens: ", cpu_goldens)

    npu_goldens = split_run()
    print("npu_goldens: ", cpu_goldens)
    
    assert check_move(npu_goldens, cpu_goldens), "Test failed"
    run_perf_test(split, [input_npu])
    print("----------- split test success ------------")

if __name__ == "__main__":
    run_test()