import torch
import torch_npu
import torch_atb
import argparse
import time

hn = head_num = 8
hd = head_dim = 128
b = 1
s = 512
h = hn * hd
max_s = 1024
bn = 1024
bs = 128
width = 0.2

 
def reshape_handler(old_shape):
    print(f"Old shape: {old_shape}")
    new_shape = [1, old_shape[0] * old_shape[1]]
    print(f"New shape: {new_shape}")
    return new_shape 
 
class LlamaRMSNorm():
    def __init__(self):
        print("---LlamaRMSNorm builder---")
 
    def forward(self, inputs): 
        return

class LlamaLayer():
    def __init__(self):
        def reshape_qkv(org_shape):
            return [org_shape[0], head_num, head_dim]

        def reshape_2d(org_shape):
            return [org_shape[0], org_shape[1] * org_shape[2]]

        rms_norm_param = torch_atb.RmsNormParam()
        rms_norm_param.layerType = torch_atb.RmsNormParam.RmsNormType.RMS_NORM_NORM
        rms_norm = torch_atb.Operation(rms_norm_param)

        linear_param = torch_atb.LinearParam(hasBias=False)
        qkv_linear = torch_atb.Operation(linear_param)

        split_param = torch_atb.SplitParam()
        split_param.splitDim = 1
        split_param.splitNum = 3
        split = torch_atb.Operation(split_param)

        rope_param = torch_atb.RopeParam()
        rope_param.rotaryCoeff = 2
        rope = torch_atb.Operation(rope_param)

        reshape_and_cache = torch_atb.Operation(torch_atb.ReshapeAndCacheParam())

        self_attention_param = torch_atb.SelfAttentionParam()
        self_attention_param.headNum = 4
        self_attention_param.kvHeadNum = 4
        self_attention_param.calcType = torch_atb.SelfAttentionParam.CalcType.PA_ENCODER
        self_attention = torch_atb.Operation(self_attention_param)

        attention_linear = torch_atb.Operation(linear_param)

        elewise_param = torch_atb.ElewiseParam()
        elewise_param.elewise_type = torch_atb.ElewiseParam.ElewiseType.ELEWISE_ADD
        elewise_add = torch_atb.Operation(elewise_param)

        rms_norm2 = torch_atb.Operation(rms_norm_param)

        qkv_linear2 = torch_atb.Operation(linear_param)

        split_param = torch_atb.SplitParam()
        split_param.splitDim = 1
        split_param.splitNum = 2
        split2 = torch_atb.Operation(split_param)

        activation_param = torch_atb.ActivationParam()
        activation_param.activationType = torch_atb.ActivationType.ACTIVATION_SWISH
        swish = torch_atb.Operation(activation_param)

        elewise_param = torch_atb.ElewiseParam()
        elewise_param.elewise_type = torch_atb.ElewiseParam.ElewiseType.ELEWISE_MUL
        elewise_mul = torch_atb.Operation(elewise_param)

        down_linear = torch_atb.Operation(linear_param)

        mlp_linear = torch_atb.Operation(linear_param)

        elewise_param.elewise_type = torch_atb.ElewiseParam.ElewiseType.ELEWISE_ADD
        mlp_add = torch_atb.Operation(elewise_param)

        inputs = ["hidden_states", "norm_weight_1", "qkv_weight", "cos", "sin", "k_cache", 
        "v_cache", "slots_mapping", "seq_len", "atten_weight", "norm_weight_2", 
        "mlp_up_gate_weight", "mlp_down_weight", "mlp_weight"]

        self.graph = torch_atb.GraphBuilder("LlamaAttention") \
            .set_input_output(inputs, ["llama_atten_out"]) \
            .add_operation(rms_norm, ["hidden_states", "norm_weight_1"], ["norm1_out"]) \
            .add_operation(qkv_linear, ["norm1_out", "qkv_weight"], ["qkv"]) \
            .add_operation(split, ["qkv"], ["q", "k", "v"]) \
            .add_operation(rope, ["q", "k", "cos", "sin", "seq_len"], ["q_embed", "k_embed"]) \
            .reshape("q_embed", reshape_qkv, "q_embed_") \
            .reshape("k_embed", reshape_qkv, "k_embed_") \
            .reshape("v", reshape_qkv, "v_") \
            .add_operation(reshape_and_cache, ["k_embed_", "v_", "k_cache", "v_cache", "slots_mapping"], ["k_cache", "v_cache"]) \
            .add_operation(self_attention, ["q_embed_", "k_embed_", "v_", "seq_len"], ["atten_out"]) \
            .reshape("atten_out", reshape_2d, "atten_out_") \
            .add_operation(attention_linear, ["atten_out_", "atten_weight"], ["atten_linear_out"]) \
            .add_operation(elewise_add, ["hidden_states", "atten_linear_out"], ["llama_atten_out"]) \
            .add_operation(rms_norm2, ["llama_atten_out", "norm_weight_2"], ["norm2_out"]) \
            .add_operation(qkv_linear2, ["norm2_out", "mlp_up_gate_weight"], ["up_gate_out"]) \
            .add_operation(split2, ["up_gate_out"], ["up_out", "gate_out"]) \
            .add_operation(swish, ["gate_out"], ["swish_out"]) \
            .add_operation(elewise_mul, ["up_out", "swish_out"], ["swish_out"]) \
            .add_operation(down_linear, ["swish_out", "mlp_down_weight"], ["mlp_out"]) \
            .add_operation(mlp_linear, ["mlp_out", "mlp_weight"], ["mlp_linear_out"]) \
            .add_operation(mlp_add, ["atten_linear_out", "mlp_linear_out"], ["llama_mlp_out"]) \
            .build()

    def forward(self, inputs):   
        return self.graph.forward(inputs)

class LlamaModel():
    def __init__(self):
        return
 
def get_inputs():
    hidden_states = (torch.rand(b * s, h).half() * width - width / 2).npu()
    norm_weight_1 = (torch.rand(h).half() * width - width / 2).npu()
    qkv_weight = (torch.rand(3 * h, h).half() * width - width / 2).npu()
    qkv = (torch.rand(b * s, 3 * h).half() * width - width / 2).npu()
    cos = (torch.rand(max_s, hd).half() * width - width / 2).npu()
    sin = (torch.rand(max_s, hd).half() * width - width / 2).npu()
    k_cache = torch.zeros(bn, bs, hn, hd).half().npu()
    v_cache = torch.zeros(bn, bs, hn, hd).half().npu()
    slots_mapping = torch.zeros(b * s, dtype=torch.int).npu()
    seqlen = (torch.ones(b, dtype=torch.int) * s).cpu()

    atten_weight = (torch.rand(h, h).half() * width - width / 2).npu()
    norm_weight_2 = (torch.rand(h).half() * width - width / 2).npu()
    mlp_up_gate_weight = (torch.rand(8 * h, h).half() * width - width / 2).npu()
    mlp_down_weight = (torch.rand(h, 4 * h).half() * width - width / 2).npu()
    mlp_weight = (torch.rand(h, h).half() * width - width / 2).npu()

    return [hidden_states, norm_weight_1, qkv_weight, cos, sin, k_cache, v_cache, slots_mapping, seqlen, 
    atten_weight, norm_weight_2, mlp_up_gate_weight, mlp_down_weight, mlp_weight]

def main():
    args = parse_arguments()
    device = f"npu:{args.device}"
    layer = LlamaLayer()
    layer_inputs = get_inputs()
    start_time = time.time()
    layer_outputs = layer.forward(layer_inputs)
    end_time = time.time()
    duration = (end_time - start_time) * 1_000_000 
    print("layer_outputs: ", layer_outputs)
    return 
        
    torch.npu.synchronize()
    if args.performance:
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            l2_cache=False,
            record_op_args=False,
            data_simplification=False,
        )
        with torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                dir_name=f"/home/yinqiran/torch/profiling/",
                worker_name="torch_npu",
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            with_flops=False,
            with_modules=False,
            experimental_config=experimental_config,
        ) as prof:
            torch.npu.synchronize() 
            for _ in range(3):
                layer_outputs = layer.forward(layer_inputs)
            torch.npu.synchronize()

            torch.npu.synchronize() 
            for _ in range(10):
                layer_outputs = layer.forward(layer_inputs)
            torch.npu.synchronize()

    print("layer_outputs: ", layer_outputs)
    
 
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--performance", type=int, default=1)
    parser.add_argument("--warmup_time", type=int, default=0)
    parser.add_argument("--infer_time", type=int, default=0)
    return parser.parse_args()
 
 
if __name__ == "__main__":
    main()