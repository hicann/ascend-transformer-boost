import torch
import torch.nn as nn
import torch_atb
import numbers
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from utils import check_float, run_perf_test

def run_test():
    print("----------- layernorm test begin ------------")
    eps=1e-05
    batch, sentence_length, embedding_dim = 20, 5, 10
    embedding = torch.randn(batch, sentence_length, embedding_dim)
    embedding_npu = embedding.npu()
    # torch_atb
    normalized_shape = (embedding_dim,) if isinstance(embedding_dim, numbers.Integral) else tuple(embedding_dim)
    weight = torch.ones(normalized_shape, dtype=torch.float32).npu()
    bias = torch.zeros(normalized_shape, dtype=torch.float32).npu()
    layer_norm_param = torch_atb.LayerNormParam()
    layer_norm_param.layer_type = torch_atb.LayerNormParam.LayerNormType.LAYER_NORM_NORM
    layer_norm_param.norm_param.epsilon = eps
    layer_norm_param.norm_param.begin_norm_axis = len(normalized_shape) * -1
    layernorm = torch_atb.Operation(layer_norm_param)

    def layernorm_run():
        layernorm_outputs = layernorm.forward([embedding_npu, weight, bias])
        return layernorm_outputs

    def golden():
        nn_layer_norm = torch.nn.LayerNorm(embedding_dim)
        return [nn_layer_norm(embedding)]

    cpu_goldens = golden()

    print("cpu_goldens: ", cpu_goldens)

    npu_outputs = layernorm_run()
    print("npu_outputs: ", npu_outputs)
    
    assert check_float(npu_outputs, cpu_goldens), "Test failed"

    run_perf_test(layernorm, [embedding_npu, weight, bias])
    print("----------- layernorm test success ------------")

if __name__ == "__main__":
    run_test()