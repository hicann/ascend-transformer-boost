import torch
import torch.nn as nn
import torch_atb  

def run_test():
    print("----------- graph test begin ------------")
    elewise_param = torch_atb.ElewiseParam()
    elewise_param.elewise_type = torch_atb.ElewiseParam.ElewiseType.ELEWISE_ADD
    elewise_add = torch_atb.Operation(elewise_param)
    elewise_param.elewise_type = torch_atb.ElewiseParam.ElewiseType.ELEWISE_MUL
    elewise_mul = torch_atb.Operation(elewise_param)
    graph = torch_atb.GraphBuilder("Graph") \
        .set_input_output(["x", "y", "z"], ["out"]) \
        .add_operation(elewise_add, ["x", "y"], ["add_out"]) \
        .reshape("add_out", lambda shape: [1, shape[0] * shape[1]], "add_out_") \
        .add_operation(elewise_mul, ["add_out_", "z"], ["out"]) \
        .build()

    x = torch.ones(2, 3, dtype=torch.float16)
    y = torch.ones(2, 3, dtype=torch.float16)
    z = torch.ones(1, 6, dtype=torch.float16)
    tensors_npu = [tensor.npu() for tensor in [x, y, z]]

    def graph_run():
        return graph.forward(tensors_npu)

    def golden():
        sum_xy = x + y
        sum_xy_reshaped = sum_xy.view(1, 6)
        result = sum_xy_reshaped * z
        return [result]

    cpu_goldens = golden()
    print("cpu_goldens: ", cpu_goldens)

    npu_outputs = graph_run()
    print("npu_outputs: ", npu_outputs)
    
    assert torch.allclose(npu_outputs[0].cpu(), cpu_goldens[0]), "Test failed"
    print("----------- graph test success ------------")

if __name__ == "__main__":
    run_test()