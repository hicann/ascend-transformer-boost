
import os
import torch
import torch.nn as nn
import torch_atb
import numpy as np

os.environ['ATB_LAUNCH_KERNEL_WITH_TILING']="0"
ATB_LAUNCH_KERNEL_WITH_TILING = os.environ.get("ATB_LAUNCH_KERNEL_WITH_TILING")

def run_test():
    print("----------- graph test begin ------------")
    m, n, k = 512, 512, 512
    linear_param = torch_atb.LinearParam()
    linear_param.has_bias = False
    linear_param.transpose_b = True
    linear = torch_atb.Operation(linear_param)

    elewise_param = torch_atb.ElewiseParam()
    elewise_param.elewise_type = torch_atb.ElewiseParam.ElewiseType.ELEWISE_ADD
    elewise_add = torch_atb.Operation(elewise_param)

    linear_param_1 = torch_atb.LinearParam()
    linear_param_1.has_bias = False
    linear_param_1.transpose_b = True
    linear_1 = torch_atb.Operation(linear_param_1)

    elewise_param_1 = torch_atb.ElewiseParam()
    elewise_param_1.elewise_type = torch_atb.ElewiseParam.ElewiseType.ELEWISE_ADD
    elewise_add_1 = torch_atb.Operation(elewise_param_1)

    linear_param_2 = torch_atb.LinearParam()
    linear_param_2.has_bias = False
    linear_param_2.transpose_b = True
    linear_2 = torch_atb.Operation(linear_param_2)

    elewise_param_2 = torch_atb.ElewiseParam()
    elewise_param_2.elewise_type = torch_atb.ElewiseParam.ElewiseType.ELEWISE_ADD
    elewise_add_2 = torch_atb.Operation(elewise_param_2)


    linear_param_3_ = torch_atb.LinearParam()
    linear_param_3_.has_bias = False
    linear_param_3_.transpose_b = True
    linear_3_ = torch_atb.Operation(linear_param_3_)

    elewise_param_3_ = torch_atb.ElewiseParam()
    elewise_param_3_.elewise_type = torch_atb.ElewiseParam.ElewiseType.ELEWISE_ADD
    elewise_add_3_ = torch_atb.Operation(elewise_param_3_)


    elewise_param_3 = torch_atb.ElewiseParam()
    elewise_param_3.elewise_type = torch_atb.ElewiseParam.ElewiseType.ELEWISE_ADD
    elewise_add_3 = torch_atb.Operation(elewise_param_3)


    elewise_param_4 = torch_atb.ElewiseParam()
    elewise_param_4.elewise_type = torch_atb.ElewiseParam.ElewiseType.ELEWISE_ADD
    elewise_add_4 = torch_atb.Operation(elewise_param_4)


    elewise_param_5 = torch_atb.ElewiseParam()
    elewise_param_5.elewise_type = torch_atb.ElewiseParam.ElewiseType.ELEWISE_ADD
    elewise_add_5 = torch_atb.Operation(elewise_param_5)

    


    graph = torch_atb.GraphBuilder("matmul_add_fuse_test") \
        .set_input_output(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"], ["k"]) \
        .add_operation(linear, ["0", "1"], ["a"]) \
        .add_operation(elewise_add, ["2", "a"], ["b"]) \
        .add_operation(linear_1, ["3", "4"], ["c"]) \
        .add_operation(elewise_add_1, ["5", "c"], ["d"]) \
        .add_operation(linear_2, ["6", "7"], ["e"]) \
        .add_operation(elewise_add_2, ["8", "e"], ["f"]) \
        .add_operation(linear_3_, ["9", "10"], ["g"]) \
        .add_operation(elewise_add_3_, ["11", "g"], ["h"]) \
        .add_operation(elewise_add_3, ["b", "d"], ["i"]) \
        .add_operation(elewise_add_4, ["f", "h"], ["j"]) \
        .add_operation(elewise_add_5, ["i", "j"], ["k"]) \
        .build({"matmul_add"})
    a = torch.randn(m, n, dtype=torch.float16)
    b = torch.randn(k, n, dtype=torch.float16)
    d = torch.randn(m, k, dtype=torch.float16)

    f = torch.randn(m, n, dtype=torch.float16)
    g = torch.randn(k, n, dtype=torch.float16)
    i = torch.randn(m, k, dtype=torch.float16)

    a1 = torch.randn(m, n, dtype=torch.float16)
    b1 = torch.randn(k, n, dtype=torch.float16)
    d1 = torch.randn(m, k, dtype=torch.float16)

    f1 = torch.randn(m, n, dtype=torch.float16)
    g1 = torch.randn(k, n, dtype=torch.float16)
    i1 = torch.randn(m, k, dtype=torch.float16)

    tensors_npu = [tensor.npu() for tensor in [a, b, d, f, g, i, a1, b1, d1, f1, g1, i1]]

    def graph_run():
        return graph.forward(tensors_npu, True)

    def golden():
        result_1 = torch.matmul(a, b.transpose(1, 0))
        result_1 = torch.add(result_1, d)
        result_2 = torch.matmul(f, g.transpose(1, 0))
        result_2 = torch.add(result_2, i)

        result_3 = torch.matmul(a1, b1.transpose(1, 0))
        result_3 = torch.add(result_3, d1)
        result_3_ = torch.matmul(f1, g1.transpose(1, 0))
        result_3_ = torch.add(result_3_, i1)


        result_4 = result_1 + result_2
        result_5 = result_3_ + result_3
        result_6 = result_4 + result_5
        return [result_6]

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

