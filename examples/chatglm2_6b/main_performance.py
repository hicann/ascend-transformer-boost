import platform
import os
import torch

# 适配昇腾NPU
import torch_npu
from torch_npu.contrib import transfer_to_npu
import transformers
from transformers import AutoTokenizer, AutoModel
DEVICE_ID = os.environ.get("SET_NPU_DEVICE")
device_id = 1
if DEVICE_ID is not None:
    device_id = int(DEVICE_ID)
print(f"user npu:{device_id}")
torch.npu.set_device(torch.device(f"npu:{device_id}"))

# 使用二进制优化，消除动态shape的编译问题
torch.npu.set_compile_mode(jit_compile=False)
option = {}
option["NPU_FUZZY_COMPILE_BLACKLIST"] = "Tril"
torch.npu.set_option(option)
ACLTRANSFORMER_HOME_PATH = os.environ.get("ACLTRANSFORMER_HOME_PATH")
if ACLTRANSFORMER_HOME_PATH is None:
    raise RuntimeError(
        "env ACLTRANSFORMER_HOME_PATH not exist, source set_env.sh")

LIB_PATH = os.path.join(ACLTRANSFORMER_HOME_PATH,
                        "lib/libacltransformer_torch.so")
torch.classes.load_library(LIB_PATH)
tokenizer = AutoTokenizer.from_pretrained("./", trust_remote_code=True)
model = AutoModel.from_pretrained(
    "./", trust_remote_code=True).half().npu()

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False

import numpy as np
# from npu_quantization import quantize_model
input_scale_dict = np.load("./chatglm_quant_param/input_scale.npy", allow_pickle=True).item()
input_offset_dict = np.load("./chatglm_quant_param/input_offset.npy", allow_pickle=True).item()
weight_scale_dict = np.load("./chatglm_quant_param/weight_scale.npy", allow_pickle=True).item()
quant_weight_dict = np.load("./chatglm_quant_param/quant_weight.npy", allow_pickle=True).item()
deq_scale_dict = np.load("./chatglm_quant_param/deq_scale.npy", allow_pickle=True).item()
fp_bias_dict = np.load("./chatglm_quant_param/fp_bias.npy", allow_pickle=True).item()

def bias_correction(fp_bias, quant_weight, input_offset, deq_scale):
    correction = quant_weight.to(torch.float32).npu().sum(dim=1) * float(input_offset) * deq_scale.npu()
    bias_correction = fp_bias.npu() - correction
    return bias_correction

transdata_operation = torch.classes.OperationTorch.OperationTorch("TransDataInt8Operation")
transdata_operation.set_param("{}")

# 量化模型适配
for name, mod in model.named_modules():
    if type(mod).__name__ == "GLMBlock":
        if hasattr(mod, "fhtoh_deq_scale"):
            print(name)
            query_key_value_name = "{}.self_attention.query_key_value".format(name)
            dense_name = "{}.self_attention.dense".format(name)
            dense_h_to_4h_name = "{}.mlp.dense_h_to_4h".format(name)
            dense_4h_to_h_name = "{}.mlp.dense_4h_to_h".format(name)

            # mod.qkv_weight = torch.nn.Parameter(quant_weight_dict[query_key_value_name].to(torch.int8).npu(), requires_grad=False)
            mod.qkv_weight = torch.nn.Parameter(transdata_operation.execute([quant_weight_dict[query_key_value_name].to(torch.int8).npu()])[0], requires_grad=False)
            mod.qkv_deq_scale = torch.nn.Parameter(deq_scale_dict[query_key_value_name].to(torch.float32).npu(),requires_grad=False)
            fp_qkv_bias2 = bias_correction(fp_bias_dict[query_key_value_name], quant_weight_dict[query_key_value_name], int(input_offset_dict[query_key_value_name]), deq_scale_dict[query_key_value_name])
            mod.qkv_bias = torch.nn.Parameter(fp_qkv_bias2.half().npu(), requires_grad=False)

            # mod.dense_weight = torch.nn.Parameter(quant_weight_dict[dense_name].to(torch.int8).npu(), requires_grad=False)
            mod.dense_weight = torch.nn.Parameter(transdata_operation.execute([quant_weight_dict[dense_name].to(torch.int8).npu()])[0], requires_grad=False)
            mod.dense_deq_scale = torch.nn.Parameter(deq_scale_dict[dense_name].to(torch.float32).npu(), requires_grad=False)
            fp_dense_bais2 = bias_correction(fp_bias_dict[dense_name], quant_weight_dict[dense_name], int(input_offset_dict[dense_name]), deq_scale_dict[dense_name])
            mod.dense_bais = torch.nn.Parameter(fp_dense_bais2.half().npu(), requires_grad=False)

            # mod.hto4h_weight = torch.nn.Parameter(quant_weight_dict[dense_h_to_4h_name].to(torch.int8).npu(), requires_grad=False)
            mod.hto4h_weight = torch.nn.Parameter(transdata_operation.execute([quant_weight_dict[dense_h_to_4h_name].to(torch.int8).npu()])[0], requires_grad=False)
            mod.hto4h_deq_scale = torch.nn.Parameter(deq_scale_dict[dense_h_to_4h_name].to(torch.float32).npu(), requires_grad=False)
            fp_hto4h_bais2 = bias_correction(fp_bias_dict[dense_h_to_4h_name], quant_weight_dict[dense_h_to_4h_name], int(input_offset_dict[dense_h_to_4h_name]), deq_scale_dict[dense_h_to_4h_name])
            mod.hto4h_bais = torch.nn.Parameter(fp_hto4h_bais2.half().npu(), requires_grad=False)

            # mod.fhtoh_weight = torch.nn.Parameter(quant_weight_dict[dense_4h_to_h_name].to(torch.int8).npu(), requires_grad=False)
            mod.fhtoh_weight = torch.nn.Parameter(transdata_operation.execute([quant_weight_dict[dense_4h_to_h_name].to(torch.int8).npu()])[0], requires_grad=False)
            mod.fhtoh_deq_scale = torch.nn.Parameter(deq_scale_dict[dense_4h_to_h_name].to(torch.float32).npu(), requires_grad=False)
            fp_fhtoh_bais2 = bias_correction(fp_bias_dict[dense_4h_to_h_name], quant_weight_dict[dense_4h_to_h_name], int(input_offset_dict[dense_4h_to_h_name]), deq_scale_dict[dense_4h_to_h_name])
            mod.fhtoh_bais = torch.nn.Parameter(fp_fhtoh_bais2.half().npu(), requires_grad=False)



# 修改transformers的TopKLogitsWarper
def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    top_k = min(self.top_k, scores.size(-1))
    indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
    filter_value = torch.finfo(scores.dtype).min
    scores = scores.masked_fill(indices_to_remove, filter_value)
    # scores = scores.masked_fill(indices_to_remove, self.filter_value)
    return scores

transformers.generation.TopKLogitsWarper.__call__ = __call__

# 修改transformers的TopPLogitsWarper
def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    sorted_logits, sorted_indices = torch.sort(scores, descending=False)
    # cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    cumulative_probs = sorted_logits.softmax(
        dim=-1).cpu().float().cumsum(dim=-1).to(sorted_logits.device).to(sorted_logits.dtype)

    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
    if self.min_tokens_to_keep > 1:
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., -self.min_tokens_to_keep:] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove)
    filter_value = torch.finfo(scores.dtype).min
    scores = scores.masked_fill(indices_to_remove, filter_value)
    # scores = scores.masked_fill(indices_to_remove, self.filter_value)
    return scores

transformers.generation.TopPLogitsWarper.__call__ = __call__

# 优化ND NZ排布，消除transdata
soc_version = torch_npu._C._npu_get_soc_version()
print("soc_version", soc_version)
if soc_version in [104, 220, 221, 222, 223]:
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.weight.data = module.weight.data.npu_format_cast(2)
    print("soc_version:", soc_version, " is 910B, support ND")
else:
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if name == "lm_head":
                module.weight = torch.nn.parameter.Parameter(module.weight.data)
            module.weight.data = module.weight.data.npu_format_cast(29)


def build_prompt(history):
    prompt = "欢迎使用 ChatGLM2-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM-6B：{response}"
    return prompt


def run_query():
    history = []
    global stop_stream
    print("欢迎使用 ChatGLM2-6B 模型")

    querys = [
        "中国的首都在哪里？",
        # "请做一首诗歌：",
        # "我想要学习python，该怎么学习？",
        # "请帮我写一篇关于大模型推理优化的任职报告：",
    ]
    
    for query in querys:
        history = []
        model.count = 0
        model.input_generate = 0
        model.model_total = 0
        model.token_total = 0
        model.model_time = 0
        model.token_time = 0
        model.model_first = 0
        model.token_first = 0
        model.pre_processing = 0
        model.post_processing = 0
        for response, history in model.stream_chat(tokenizer, query, history=history):
            if stop_stream:
                stop_stream = False
                break

        print(build_prompt(history), flush=True)


def main():
    history = []
    global stop_stream
    print("欢迎使用 ChatGLM2-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            history = []
            # os.system(clear_command)
            print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        model.count = 0
        model.input_generate = 0
        model.model_total = 0
        model.token_total = 0
        model.model_time = 0
        model.token_time = 0
        model.model_first = 0
        model.token_first = 0
        model.pre_processing = 0
        model.post_processing = 0
        for response, history in model.stream_chat(tokenizer, query, history=history):
            if stop_stream:
                stop_stream = False
                break
        print(build_prompt(history), flush=True)


if __name__ == "__main__":
    run_query()
    # main()

