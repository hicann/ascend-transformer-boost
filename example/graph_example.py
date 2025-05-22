import torch
import torch.nn.functional as F
import torch_atb
import math
import acl
import logging

s = 128 # Sequence Length
h = 16 # Number of Heads
d_k = 64 # Head Dimension
d_v = 64 # Value Dimension (vHiddenSize)
output_dim = 64
output_dim_1 = 128

def get_inputs():
    torch.manual_seed(233)

    print("------------ generate inputs begin ------------")
    query = (torch.randn((s, 16, d_k), dtype=torch.float16)).npu()
    key = (torch.randn((s, 16, d_k), dtype=torch.float16)).npu()
    value = (torch.randn((s, 16, d_k), dtype=torch.float16)).npu()
    seqLen = (torch.tensor([s], dtype = torch.int32))

    input_0 = (torch.randn((16, d_k), dtype=torch.float16)).npu()

    gamma = (torch.randn((s, 16, d_k), dtype=torch.float16)).npu()
    beta = (torch.zeros((s, 16, d_k), dtype=torch.float16)).npu()

    weight_0 = (torch.randn((output_dim_1, output_dim), dtype=torch.float16)).npu()
    bias_0 = (torch.randn((output_dim_1,), dtype=torch.float16)).npu()

    weight_1 = (torch.randn((output_dim_1, output_dim_1), dtype=torch.float16)).npu()
    bias_1 = (torch.randn((output_dim_1,), dtype=torch.float16)).npu()

    inputs = [query, key, value, seqLen, input_0, gamma, beta, weight_0, bias_0, weight_1, bias_1]
    print("------------ generate inputs success ------------")
    return inputs

def graph_build():
    print("------------ graph build begin ------------")
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
    elewise_add_param = torch_atb.ElewiseParam(elewise_type = torch_atb.ElewiseParam.ElewiseType.ELEWISE_ADD)

    elewise_add_0 = graph.add_node([self_attention_out, input_0], elewise_add_param)
    elewise_add_0_out = elewise_add_0.get_output(0)

    gamma = graph.add_input("gamma") # weight in layernorm, (Hadamard product)
    beta = graph.add_input("beta") # bias in layernorm
    layernorm_param = torch_atb.LayerNormParam(layer_type = torch_atb.LayerNormParam.LayerNormType.LAYER_NORM_NORM)
    layernorm_param.norm_param.begin_norm_axis = 0
    layernorm_param.norm_param.begin_params_axis = 0

    # x, gamma, beta, float16 -> float16
    layernorm_0 = graph.add_node([elewise_add_0_out, gamma, beta], layernorm_param)
    layernorm_0_out = layernorm_0.get_output(0)

    weight_0 = graph.add_input("weight_0") # weight in linear
    bias_0 = graph.add_input("bias_0") # bias in linear
    linear_param = torch_atb.LinearParam(out_data_type=torch_atb.AclDataType.ACL_DT_UNDEFINED)

    # x, weight, bias， float 16 -> float16
    linear_0 = graph.add_node([layernorm_0_out, weight_0, bias_0], linear_param) 
    linear_0_out = linear_0.get_output(0)

    elewise_tanh_param = torch_atb.ElewiseParam(elewise_type = torch_atb.ElewiseParam.ElewiseType.ELEWISE_TANH)

    elewise_tanh = graph.add_node([linear_0_out], elewise_tanh_param)
    elewise_tanh_out = elewise_tanh.get_output(0)

    weight_1 = graph.add_input("weight_1")
    bias_1 = graph.add_input("bias_1")

    # x, weight, bias， float 16 -> float16
    linear_1 = graph.add_node([elewise_tanh_out, weight_1, bias_1], linear_param)
    linear_1_out = linear_1.get_output(0)

    graph.mark_output(linear_1_out)
    Graph = graph.build()
    print("----------- graph build success -----------")
    return Graph

def golden(inputs):
    query, key, value, seqLen, input_0, gamma, beta, w0, b0, w1, b1 = inputs

    # 1. Self-Attention
    #   q,k,v: [s, heads, d_k] → permute到 [heads, s, d_k]
    Q = query.permute(1, 0, 2)
    K = key.permute(1, 0, 2)
    V = value.permute(1, 0, 2)

    # 计算 scaled dot-product attention
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, V)              # [heads, s, d_k]
    A = out.permute(1, 0, 2)                # back to [s, heads, d_k]

    # 2. 残差相加
    B = A + input_0                              # broadcast [16,d_k] → [s,16,d_k]

    # 3. LayerNorm
    #    等价于 normalize B 上所有元素，然后逐元素乘 gamma 加 beta
    C = F.layer_norm(B, (s, 16, d_k), weight=gamma, bias=beta, eps=1e-5)

    # 4. 前馈网络：Linear0 → Tanh → Linear1
    D = torch.matmul(C, w0.T) + b0               # [s,16,d_k] @ [d_k,output_dim1] → [s,16,output_dim1]
    E = torch.tanh(D)
    F_ = torch.matmul(E, w1.T) + b1              # [s,16,output_dim1]

    return F_

def run():
    Graph = graph_build()
    inputs = get_inputs()
    print("----------- single graph forward begin -----------")
    results = Graph.forward(inputs)
    golden = golden(inputs)
    logging.info(golden)
    logging.info(results)
    print("----------- single graph forward success -----------")

if __name__ == "__main__":
    run()