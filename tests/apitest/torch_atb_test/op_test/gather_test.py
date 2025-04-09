import torch
import torch.nn as nn
import torch_atb  
from utils import check_move, run_perf_test

def run_test():
    print("----------- gather test begin ------------")
    axis = 1
    gather_param = torch_atb.GatherParam()
    gather_param.axis = axis
    gather = torch_atb.Operation(gather_param)
    intensor0 = torch.randn([3,5],dtype=torch.float16)
    intensor0_npu = intensor0.npu()
    intensor1 = torch.randint(0, 5, [3,4],dtype=torch.int64)
    intensor1_npu = intensor1.npu()

    def gather_run():
        gather_outputs = gather.forward([intensor0_npu, intensor1_npu])
        return gather_outputs

    def golden():
        outputSize = []
        dim0 = 1
        for i in range(0,axis):
            outputSize.append(intensor0.shape[i])
            dim0 *= intensor0.shape[i]
        dim1 = intensor0.shape[axis]
        for i in range(0,intensor1.ndim):
            outputSize.append(intensor1.shape[i])
        dim2 = 1
        for i in range(axis + 1,intensor0.ndim):
            outputSize.append(intensor0.shape[i])
            dim2 *= intensor0.shape[i]
        inputFlatten = intensor0.clone().reshape(-1)
        indicesFlatten = intensor1.clone().reshape(-1)
        golden_result_np = torch.zeros(outputSize,dtype=torch.float16).reshape(-1).numpy()
        idx = 0
        for i in range(0,dim0):
            inputIdx = i * dim1 * dim2
            for indice in indicesFlatten:
                for k in range(0,dim2):
                    golden_result_np[idx] = inputFlatten[inputIdx + indice * dim2 + k]
                    idx+=1
        golden_result = torch.from_numpy(golden_result_np).reshape(outputSize)
        print(intensor0.dtype)
        if intensor0.dtype == torch.bfloat16:
            golden_result = golden_result.bfloat16()
        return [golden_result]

    cpu_goldens = golden()
    print("cpu_goldens: ", cpu_goldens)

    npu_outputs = gather_run()
    print("npu_outputs: ", npu_outputs)
    
    assert check_move(npu_outputs, cpu_goldens), "Test failed"

    run_perf_test(gather, [intensor0_npu, intensor1_npu])
    print("----------- gather test success ------------")

if __name__ == "__main__":
    run_test()