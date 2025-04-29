import torch_atb
import torch
import acl
import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "./"))
from utils import ret_check

s = 128 # Sequence Length
h = 16 # Number of Heads
d_k = 64 # Head Dimension
d_v = 64 # Value Dimension (vHiddenSize)
output_dim = 64
output_dim_1 = 128

def single_graph_build(n):
    print(f"------------ single graph {n} build begin ------------")
    graph = torch_atb.Builder("Graph")
    
    query = graph.add_input("query")
    key = graph.add_input("key")
    value = graph.add_input("value")
    seqLen = graph.add_input("seqLen")
    self_attention_param = torch_atb.SelfAttentionParam()
    self_attention_param.head_num = 16
    self_attention_param.kv_head_num = 16
    self_attention_param.calc_type = torch_atb.SelfAttentionParam.CalcType.PA_ENCODER

    # float16: query, key, value,
    # int32: seqLen
    # -> float16 (s, 16, d_k)
    self_attention = graph.add_node([query, key, value, seqLen], self_attention_param)
    self_attention_out = self_attention.get_output(0)

    input_0 = graph.add_input("input_0")
    elewise_add_param = torch_atb.ElewiseParam()
    elewise_add_param.elewise_type = torch_atb.ElewiseParam.ElewiseType.ELEWISE_ADD

    elewise_add_0 = graph.add_node([self_attention_out, input_0], elewise_add_param)
    elewise_add_0_out = elewise_add_0.get_output(0)

    gamma = graph.add_input("gamma") # weight in layernorm, (Hadamard product)
    beta = graph.add_input("beta") # bias in layernorm
    layernorm_param = torch_atb.LayerNormParam()
    layernorm_param.layer_type = torch_atb.LayerNormParam.LayerNormType.LAYER_NORM_NORM
    layernorm_param.norm_param.begin_norm_axis = 0
    layernorm_param.norm_param.begin_params_axis = 0

    # # Just fot reshape test
    # graph.reshape(gamma, reshape_transpose, "gamma_")

    # x, gamma, beta, float16 -> float16
    layernorm_0 = graph.add_node([elewise_add_0_out, gamma, beta], layernorm_param)
    layernorm_0_out = layernorm_0.get_output(0)

    weight_0 = graph.add_input("weight_0") # weight in linear
    bias_0 = graph.add_input("bias_0") # bias in linear
    linear_param = torch_atb.LinearParam()

    # x, weight, bias， float 16 -> float16
    linear_0 = graph.add_node([layernorm_0_out, weight_0, bias_0], linear_param) 
    linear_0_out = linear_0.get_output(0)

    elewise_tanh_param = torch_atb.ElewiseParam()
    elewise_tanh_param.elewise_type = torch_atb.ElewiseParam.ElewiseType.ELEWISE_TANH

    elewise_tanh = graph.add_node([linear_0_out], elewise_tanh_param)
    elewise_tanh_out = elewise_tanh.get_output(0)

    weight_1 = graph.add_input("weight_1")
    bias_1 = graph.add_input("bias_1")

    # x, weight, bias， float 16 -> float16
    linear_1 = graph.add_node([elewise_tanh_out, weight_1, bias_1], linear_param)
    linear_1_out = linear_1.get_output(0)

    graph.mark_output(linear_1_out)
    Graph = graph.build()
    print(f"----------- single graph {n} build success -----------")
    return Graph

def reshape_transpose(old_shape):
    new_shape = [old_shape[1], old_shape[0]]
    return new_shape 

def get_inputs():
    torch.manual_seed(233)

    print("------------ generate inputs begin ------------")
    query = (torch.randn((s, 16, d_k), dtype=torch.float16)).npu()
    key = (torch.randn((s, 16, d_k), dtype=torch.float16)).npu()
    value = (torch.randn((s, 16, d_k), dtype=torch.float16)).npu()
    seqLen = (torch.tensor([s], dtype = torch.int32))
    # (s, 16, d_k) == (128, ,16 , 64)

    input_0 = (torch.randn((16, d_k), dtype=torch.float16)).npu()

    gamma = (torch.randn((s, 16, d_k), dtype=torch.float16)).npu()
    beta = (torch.zeros((s, 16, d_k), dtype=torch.float16)).npu()
    # (s, 16, d_k) == (128, 16, 64)

    weight_0 = (torch.randn((output_dim_1, output_dim), dtype=torch.float16)).npu()
    bias_0 = (torch.randn((output_dim_1,), dtype=torch.float16)).npu()
    # (s, 16, output_dim1) == (128, 16, 128)

    weight_1 = (torch.randn((output_dim_1, output_dim_1), dtype=torch.float16)).npu()
    bias_1 = (torch.randn((output_dim_1,), dtype=torch.float16)).npu()
    # (s, 16, output_dim1) == (128, 16, 128)

    inputs = [query, key, value, seqLen, input_0, gamma, beta, weight_0, bias_0, weight_1, bias_1]
    print("------------ generate inputs success ------------")
    return inputs

def run():
    Graph_0 = single_graph_build(0)

    print("------------ bigger graph build begin ------------")
    bigger_graph = torch_atb.Builder("BiggerGraph")

    query = bigger_graph.add_input("query")
    key = bigger_graph.add_input("key")
    value = bigger_graph.add_input("value")
    seqLen = bigger_graph.add_input("seqLen")
    input_0 = bigger_graph.add_input("input_0")
    gamma = bigger_graph.add_input("gamma")
    beta = bigger_graph.add_input("beta")
    weight_0 = bigger_graph.add_input("weight_0")
    bias_0 = bigger_graph.add_input("bias_0")
    weight_1 = bigger_graph.add_input("weight_1")
    bias_1 = bigger_graph.add_input("bias_1")

    node_graph0 = bigger_graph.add_node([query, key, value, seqLen, input_0, gamma, beta, 
                                            weight_0, bias_0, weight_1, bias_1], Graph_0)
    node_graph0_out = node_graph0.get_output(0)

    x = bigger_graph.add_input("x")
    elewise_add = torch_atb.ElewiseParam()
    elewise_add.elewise_type = torch_atb.ElewiseParam.ElewiseType.ELEWISE_ADD
    layer = bigger_graph.add_node([x, node_graph0_out], elewise_add)

    bigger_graph.mark_output(layer.get_output(0))

    BiggerGraph = bigger_graph.build()

    print("------------ bigger graph build success ------------")
    print(BiggerGraph.__repr__)
    print("------------ bigger graph forward begin ------------")
    inputs = get_inputs()
    x = (torch.ones((128, 16, 128), dtype=torch.float16)).npu()
    inputs.append(x)
    result = BiggerGraph.forward(inputs)
    print("------------ bigger graph forward success ------------")

class TestMultiGraph(unittest.TestCase):
    def test(self):
        ret = acl.rt.set_device(0)
        ret_check(ret)
        run()

if __name__ == "__main__":
    unittest.main()