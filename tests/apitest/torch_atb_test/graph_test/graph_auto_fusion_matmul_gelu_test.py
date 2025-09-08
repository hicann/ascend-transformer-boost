
import os
import torch
import torch.nn as nn
import torch_atb
import numpy as np
import torch.nn.functional as F


os.environ['ATB_LAUNCH_KERNEL_WITH_TILING']="0"
ATB_LAUNCH_KERNEL_WITH_TILING = os.environ.get("ATB_LAUNCH_KERNEL_WITH_TILING")

def run_test():
    print("----------- graph test begin ------------")
    m, n, k = 512, 512, 512
    linear_param = torch_atb.LinearParam()
    linear_param.has_bias = False
    linear_param.transpose_b = True
    linear = torch_atb.Operation(linear_param)

    activate_param = torch_atb.ActivationParam()
    activate_param.activation_type = torch_atb.ActivationType.ACTIVATION_GELU
    activate_gelu = torch_atb.Operation(activate_param)


    linear_param_1 = torch_atb.LinearParam()
    linear_param_1.has_bias = False
    linear_param_1.transpose_b = True
    linear_1 = torch_atb.Operation(linear_param_1)

    activate_param_1 = torch_atb.ActivationParam()
    activate_param_1.activation_type = torch_atb.ActivationType.ACTIVATION_GELU
    activate_gelu_1 = torch_atb.Operation(activate_param_1)

    elewise_param_2 = torch_atb.ElewiseParam()
    elewise_param_2.elewise_type = torch_atb.ElewiseParam.ElewiseType.ELEWISE_ADD
    elewise_add_2 = torch_atb.Operation(elewise_param_2)


    graph = torch_atb.GraphBuilder("matmul_add_fuse_test") \
        .set_input_output(["a", "b", "f", "g"], ["k"]) \
        .add_operation(linear, ["a", "b"], ["c"]) \
        .add_operation(activate_gelu, ["c"], ["e"]) \
        .add_operation(linear_1, ["f", "g"], ["h"]) \
        .add_operation(activate_gelu_1, ["h"], ["j"]) \
        .add_operation(elewise_add_2, ["e", "j"], ["k"]) \
        .build({"matmul_gelu"})
    a = torch.randn(m, n, dtype=torch.float16)
    b = torch.ones(k, n, dtype=torch.float16)
    # d = torch.randn(m, k, dtype=torch.float16)

    f = torch.ones(m, n, dtype=torch.float16)
    g = torch.randn(k, n, dtype=torch.float16)
    # i = torch.randn(m, k, dtype=torch.float16)

    

    tensors_npu = [tensor.npu() for tensor in [a, b, f, g]]

    def graph_run():
        return graph.forward(tensors_npu, True)

    def golden():
        result_1 = torch.matmul(a, b.transpose(0, 1))
        result_1 = F.gelu(result_1)
        result_2 = torch.matmul(f, g.transpose(0, 1))
        result_2 = F.gelu(result_2)
        result = result_1 + result_2
        return [result]

    cpu_goldens = golden()
    print("cpu_goldens", cpu_goldens)

    npu_outputs = graph_run()
    print("cpu_goldens: ", cpu_goldens[0])
    print("npu_outputs: ", npu_outputs[0].cpu())
    print("cpu_goldens: ", npu_outputs[0].shape)
    print("npu_outputs: ", npu_outputs[0].cpu().shape)


    difference = (cpu_goldens[0] - npu_outputs[0].cpu()).numpy()
    print("difference = ", difference)
    mean = np.mean(difference)
    print("差异均值", mean)
    std = np.std(difference)
    print("差异标准差", std)
    fake = (cpu_goldens[0] == npu_outputs[0].cpu())
    print("fake = ", fake)
if __name__ == "__main__":
    run_test()

