import torch
import torch.nn as nn
import torch_atb  
import numpy as np
from utils import check_float, run_perf_test

def run_test():
    print("----------- rms_norm test begin ------------")
    rms_norm_param = torch_atb.RmsNormParam()
    rms_norm_param.layer_type = torch_atb.RmsNormParam.RmsNormType.RMS_NORM_NORM
    rms_norm_param.norm_param.rstd = True
    rms_norm = torch_atb.Operation(rms_norm_param)
    epsilon = 1e-5
    shape=[8, 8, 8]
    shape_gamma=[8]
    shape_rstd=[8, 8, 1]
    x = torch.from_numpy(np.random.uniform(low=0, high=100, size=shape).astype(np.float32))
    gamma = torch.from_numpy(np.random.uniform(low=0, high=100, size=shape_gamma).astype(np.float32))
    in_tensors = [x.npu(), gamma.npu()]

    def rms_norm_run():
        rms_norm_outputs = rms_norm.forward(in_tensors)
        return rms_norm_outputs

    def golden():
        reduceDims=[]
        edim = x.dim()-gamma.dim()
        for i in range(gamma.dim()):
            reduceDims.append(edim + i)
        rstd = torch.rsqrt(x.pow(2).mean(reduceDims, keepdim=True) + epsilon)
        result = x * rstd
        result = result * gamma
        return [result, rstd]

    cpu_goldens = golden()
    print("cpu_goldens: ", cpu_goldens)
    npu_outputs = rms_norm_run()
    print("npu_outputs: ", npu_outputs)
    assert check_float(npu_outputs, cpu_goldens), "Test failed"

    run_perf_test(rms_norm, in_tensors)
    print("----------- rms_norm test success ------------")

if __name__ == "__main__":
    run_test()