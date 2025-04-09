import torch
import torch_atb  
from utils import check_float, run_perf_test

def run_test():
    print("----------- as_strided test begin ------------")
    size = [2, 2]
    stride = [2, 3]
    offset = [0]
    intensor_cpu = torch.randn(3, 3, dtype=torch.float16)
    intensor_npu = intensor_cpu.npu()
    as_strided_param = torch_atb.AsStridedParam()
    as_strided_param.size = size
    as_strided_param.stride = stride
    as_strided_param.offset = offset
    as_strided = torch_atb.Operation(as_strided_param)
    print(as_strided_param)

    def as_strided_run():
        as_strided_outputs = as_strided.forward([intensor_npu])
        return as_strided_outputs

    def golden():
        return [torch.as_strided(intensor_cpu, size, stride, offset[0])]

    cpu_goldens = golden()
    print("cpu_goldens: ", cpu_goldens)

    npu_outputs = as_strided_run()
    print("npu_outputs: ", npu_outputs)
    
    assert check_float(npu_outputs, cpu_goldens), "Test failed"
    run_perf_test(as_strided, [intensor_npu])
    print("----------- as_strided test success ------------")

if __name__ == "__main__":
    run_test()