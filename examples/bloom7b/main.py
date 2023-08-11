import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import signal
import platform
import os
import torch

# 适配昇腾NPU
import torch_npu
from torch_npu.contrib import transfer_to_npu

DEVICE_ID = os.environ.get("SET_NPU_DEVICE")
device = "npu:0"
if DEVICE_ID is not None:
    device = "npu:" + DEVICE_ID
print("use " + device)
torch.npu.set_device(device)

# 使用二进制优化，消除动态shape的编译问题
torch.npu.set_compile_mode(jit_compile=False)

tokenizer = AutoTokenizer.from_pretrained("./", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("./").half().to(device)

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
    global stop_stream
    print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            history = []
            print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        count = 0
        for response, history in model.stream_chat(tokenizer, query, history=history):
            if stop_stream:
                stop_stream = False
                break
            else:
                count += 1
                if count % 3 == 0:
                    print(build_prompt(history), flush=True)
                    signal.signal(signal.SIGINT, signal_handler)
        print(build_prompt(history), flush=True)


if __name__ == "__main__":
    main()
