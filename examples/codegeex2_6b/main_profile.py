import transformers
from transformers import AutoTokenizer, AutoModel
import signal
import platform
import os
import torch
import time


# 适配昇腾NPU
import torch_npu
from torch_npu.contrib import transfer_to_npu
torch.npu.set_device(torch.device("npu:1"))

# 使用二进制优化，消除动态shape的编译问题
torch.npu.set_compile_mode(jit_compile=False)
option = {}
option["NPU_FUZZY_COMPILE_BLACKLIST"] = "Tril,ReduceProd"
torch.npu.set_option(option)

tokenizer = AutoTokenizer.from_pretrained("./", trust_remote_code=True)
model = AutoModel.from_pretrained("./", trust_remote_code=True).half().npu()

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
    # history = []
    # test_tokens_num = 64
    # output_file = open('performance.txt', 'a')
    warm_up_query = "你好"*512
    warm_up_querys = [warm_up_query] * 2
    # prompt = tokenizer.build_prompt(warm_up_query)
    inputs_warm = tokenizer(warm_up_querys, return_tensors="pt").to('npu')
    output_ids_warm = model.generate(**inputs_warm, do_sample=True, max_new_tokens=10)
    # inputs = inputs.to('npu')
    # model(**inputs)
    print("================================= finish warm up ===============================")
    
    print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    query = "一"*(2046*2-1)
    for bs in (1, 16):
        querys = [query] * bs
        for encoder_seqlen in (256, 512, 1024):
            print(f"================================= {bs}:encoder_seqlen {encoder_seqlen} ===============================")
            start_time = time.time()
            
            # inputs = tokenizer(querys, return_tensors='pt').to('npu')
            inputs = tokenizer(querys, return_tensors='pt', padding='max_length', max_length=encoder_seqlen, truncation=True).to('npu')
            print(len(inputs.input_ids[0]))

            stream = torch.npu.current_stream()
            stream.synchronize()
            prof = torch.npu.profile(f"./profiling/seq{encoder_seqlen}_bs{bs}_out32")
            prof.__enter__()

            output_ids = model.generate(**inputs, do_sample=True,top_p=0.95,top_k=0,temperature=0.2, max_new_tokens=32)

            stream = torch.npu.current_stream()
            stream.synchronize()
            prof.__exit__(None, None, None)
            
            new_tokens = len(output_ids[0]) - len(inputs.input_ids[0])
            answers = tokenizer.batch_decode(output_ids)
            print(f"END to END Output generated in {(time.time()-start_time):.2f} s ({new_tokens/(time.time()-start_time):.2f} tokens/s, {new_tokens} tokens)")

    max_memory_allocated = torch.npu.max_memory_allocated(device='npu') / (1024 * 1024 * 1024)
    memory_allocated = torch.npu.memory_allocated(device='npu') / (1024 * 1024 * 1024)
    max_memory_reserved = torch.npu.max_memory_reserved(device='npu') / (1024 * 1024 * 1024)
    memory_reserved = torch.npu.memory_reserved(device='npu') / (1024 * 1024 * 1024)

    print("allocated memory")
    print(max_memory_allocated)

    memory_max = max(max_memory_allocated, memory_allocated, max_memory_reserved, memory_reserved)
    print("batch_size", 4, "seq_len", inputs['input_ids'].shape[0], "max_memory", memory_max)


    print("======================================inference end======================================\n")




if __name__ == "__main__":
    main()
