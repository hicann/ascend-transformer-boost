import transformers
import readline
from transformers import AutoTokenizer, AutoModel
import signal
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

# 量化模型适配
for name, mod in model.named_modules():
    if type(mod).__name__ == "GLMBlock":
        if hasattr(mod, "qkv_deq_scale"):
            mod.qkv_deq_scale = torch.nn.Parameter(
                mod.qkv_deq_scale.data.to(torch.float32))
            mod.dense_deq_scale = torch.nn.Parameter(
                mod.dense_deq_scale.data.to(torch.float32))
            mod.hto4h_deq_scale = torch.nn.Parameter(
                mod.hto4h_deq_scale.data.to(torch.float32))
            mod.fhtoh_deq_scale = torch.nn.Parameter(
                mod.fhtoh_deq_scale.data.to(torch.float32))

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False


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
    scores = scores.masked_fill(indices_to_remove, self.filter_value)
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
            if name == "lm_head":
                module.weight = torch.nn.parameter.Parameter(module.weight.data)
            module.weight.data = module.weight.data.npu_format_cast(29)
    print("soc_version:", soc_version, " is not 910B, support NZ")

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Embedding):
        module.weight.data = module.weight.data.npu_format_cast(2)


def build_prompt(history):
    prompt = "欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM-6B：{response}"
    return prompt


def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True


def main():
    history = []
    question = 0
    test_tokens_num = 64
    performance_file_name = os.getenv('TEMP_TEST_PERFORMANCE_SAVE_PATH')
    if performance_file_name is None:
        performance_file_name = "performance.txt"
    else:
        performance_file_name = os.path.join(performance_file_name, "performance.txt")
    output_file = open(performance_file_name, 'w')
    global stop_stream
    print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            history = []
            # os.system(clear_command)
            print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        count = 0
        question += 1
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
            else:
                count += 1
                if count % 10 == 0:
                    # os.system(clear_command)
                    print(build_prompt(history), flush=True)
                    # signal.signal(signal.SIGINT, signal_handler)
                if question > 1:
                    if model.count == 1:
                        output_file.write(f"pre_processing: {model.post_processing}ms\n" +
                                          f"first_token: input generate: {model.input_generate}ms, model time: {model.model_first}ms, " +
                                          f"post processing: {model.post_processing}ms, token time: {model.token_first}ms\n")
                        output_file.write("Per token time\n")
                    elif model.count < test_tokens_num:
                        output_file.write(f"token_{model.count}: input generate: {model.input_generate}ms, " +
                                          f"model time: {model.model_time}ms, " +
                                          f"post processing: {model.post_processing}ms, token time: {model.token_time}ms\n")
                    else:
                        output_file.write(f"token_{model.count}: input generate: {model.input_generate}ms, " +
                                          f"model time: {model.model_time}ms, " +
                                          f"post processing: {model.post_processing}ms, token time: {model.token_time}ms\n")
                        output_file.write(
                            "Average time without first token\n")
                        output_file.write(
                            f"model time: {model.model_total / (test_tokens_num - 1)}ms, " +
                            f"token time: {model.token_total / (test_tokens_num - 1)}ms\n")
                        output_file.write(
                            f"Response time\nmodel time: {model.model_first + model.model_total}ms\n" +
                            f"token time: {model.token_first + model.token_total}ms\n")
                        output_file.close()
                        exit()
        # os.system(clear_command)
        print(build_prompt(history), flush=True)


if __name__ == "__main__":
    main()
