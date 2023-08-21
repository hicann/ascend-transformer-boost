import transformers
from transformers import AutoTokenizer, AutoModel
import platform
import os
import torch

# 适配昇腾NPU
import torch_npu
from torch_npu.contrib import transfer_to_npu
DEVICE_ID = os.environ.get("SET_NPU_DEVICE")
device_id = 0
if DEVICE_ID is not None:
    device_id = int(DEVICE_ID)
print(f"user npu:{device_id}")
torch.npu.set_device(torch.device(f"npu:{device_id}"))

# 使用二进制优化，消除动态shape的编译问题
torch.npu.set_compile_mode(jit_compile=False)
option = {}
option["NPU_FUZZY_COMPILE_BLACKLIST"] = "Tril"
torch.npu.set_option(option)

tokenizer = AutoTokenizer.from_pretrained("./", trust_remote_code=True)
model = AutoModel.from_pretrained(
    "./", trust_remote_code=True).half().npu()

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False

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
if soc_version in [104, 220, 221, 222, 223]:
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.weight.data = module.weight.data.npu_format_cast(2)
    print("soc_version:", soc_version, " is 910B, support ND")
else:
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.weight.data = module.weight.data.npu_format_cast(29)
    print("soc_version:", soc_version, " is not 910B, support NZ")

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Embedding):
        module.weight.data = module.weight.data.npu_format_cast(2)


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
        "请做一首诗歌：",
        "我想要学习python，该怎么学习？",
        "请帮我写一篇关于大模型推理优化的任职报告：",
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

