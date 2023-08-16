import torch
import torch_npu
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch_npu.contrib import transfer_to_npu
import os

SEQ_LEN_IN = 32
SEQ_LEN_OUT = 32

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

batch_sizes = [1]
seq_lens = [32]
test_cases = [(bs, sq) for sq in seq_lens for bs in batch_sizes]
for batch_size, seq_len in test_cases:
    warmup_start_time=time.time()

    print("---------------warm-up---------------")
    test_prompt = "Common sense questions and answers\n\nQuestion: Why do people need sleep\nFactual answer:"
    inputs_warm_up = tokenizer(test_prompt, return_tensors="pt", max_length=seq_len, padding='max_length', truncation=True)

    with torch.no_grad():
        output = model.generate(
            inputs_warm_up.input_ids.to(device),
            max_new_tokens=seq_len,
            attention_mask = inputs_warm_up.attention_mask.to(device)
        )
    print(model.device)
    print(time.time()-warmup_start_time,"s")


    print("---------------inference---------------")
    first_token_sum = 0
    sum = 0

    torch.cuda.empty_cache() 
    torch.cuda.reset_peak_memory_stats(device=device)
    prompt = [
        "Common sense questions and answers\n\nQuestion: How to learn a new language\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: Why do we need to learn a new language\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: What is the best way to study\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: What is the capital of France\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: Should I take notes during class\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: How to use photoshop\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: What is depression\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: Who was the first president of the United States\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: What is the name of the vice president of the United States\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: How to learn programming\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: How does artificial intelligence work\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: Why should we learn math\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: How does machine learning work\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: How to live a happy life\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: How to run a company\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: Who is the CEO of Google\nFactual answer:"
	]

#  first_token_start_time = time.time()
#  #   tokenize
#  inputs = tokenizer(prompt[:batch_size], return_tensors="pt", padding='max_length', truncation=True, max_length=seq_len)
#  # generate
#  with torch.no_grad():
#	  generate_ids = model.generate(inputs.input_ids.to(device), attention_mask = inputs.attention_mask.to(device), max_new_tokens=1)
#          #generate_ids = model.generate(inputs.input_ids, attention_mask = inputs.attention_mask, max_new_tokens=1)
#  # decode
#  res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
#  first_token_end_time = time.time()
#  total_time = first_token_end_time - first_token_start_time
#  first_token_sum += total_time
#  new_tokens = len(generate_ids[0]) - len(inputs.input_ids[0])
#  print(f"first token generated in {(total_time * 1000.0):.2f} ms ")
#  torch.cuda.empty_cache() 
#  torch.cuda.reset_peak_memory_stats(device=device)
    start_time = time.time()
    # tokenize
    inputs = tokenizer(prompt[:batch_size], return_tensors="pt", padding='max_length', truncation=True, max_length=seq_len)
    # generate
    # optimal parameters
    # torch.save(inputs, "./pre/inputs_npu.pt")
    with torch.no_grad():
        generate_ids = model.generate(inputs.input_ids.to(device), attention_mask = inputs.attention_mask.to(device), max_new_tokens=32)
            #generate_ids = model.generate(inputs.input_ids, attention_mask = inputs.attention_mask, max_new_tokens=seq_len)
    # decode
    res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    # torch.save(res, "./pre/outputs_npu.pt")
    print(res)
    end_time = time.time()
    total_time = end_time - start_time
    sum += total_time/seq_len
    # output
    # print(res)

    # time analysis
    new_tokens = len(generate_ids[0]) - len(inputs.input_ids[0])
    print(f"Output tokens number: {len(generate_ids[0])},\nInput tokens number: {len(inputs.input_ids[0])},\ntotal new tokens generated: {new_tokens}")
    print(f"Output generated in {(total_time * 1000.0):.2f} ms ({new_tokens/total_time:.2f} tokens/s, {new_tokens} tokens)")

#print(f"First token generated in {(first_token_sum/len(seq_lens) * 1000.0):.2f} ms ")
#print(f"Output token generated in {(sum/len(seq_lens) * 1000.0):.2f} ms ")
# memory analysis
#print("max_memory_allocated: ", torch.cuda.max_memory_allocated()) # get max memory allocated
#print("memory_allocated: ", torch.cuda.memory_allocated()) # get max memory currently allocated
#print("max_memory_reserved: ", torch.cuda.max_memory_reserved()) # max memory reserved in torch pool
#print("memory_reserved: ", torch.cuda.memory_reserved()) # memory reserved in torch pool
