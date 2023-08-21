import sys
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch_npu
from torch_npu.contrib import transfer_to_npu

 
SEQ_LEN_IN = 128
SEQ_LEN_OUT = 128
DEVICE_ID = 4

torch.npu.set_device(torch.device(f"npu:{DEVICE_ID}"))

torch.npu.set_compile_mode(jit_compile=False)
option = {}
option["NPU_FUZZY_COMPILE_BLACKLIST"] = "ReduceProd"
torch.npu.set_option(option)

tokenizer = AutoTokenizer.from_pretrained("./", trust_remote_code=True, use_fast=False)
model = AutoModelForCausalLM.from_pretrained("./", trust_remote_code=True).half().npu()

# padding
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))

soc_version = torch_npu._C._npu_get_soc_version()
if soc_version in [104, 220, 221, 222, 223]:
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.weight.data = module.weight.data.npu_format_cast(2)
    print("soc version: ", soc_version, " is 910B, support ND")
else: 
    # if on 910A or 310P chip, eliminate the TransData and Transpose ops by converting weight data types 
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if name == 'lm_head':
                # eliminate TransData op before lm_head calculation
                module.weight.data = torch.nn.parameter.Parameter(module.weight.data)
            module.weight.data = module.weight.data.npu_format_cast(29)
    print("soc version: ", soc_version, " is not 910B, support NZ")

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Embedding):
        module.weight.data = module.weight.data.npu_format_cast(2)

# warm-up using huggingface's generate api
print("--------------warm up--------------")
test_prompt = "Common sense questions and answers\n\nQuestion: Why do people need sleep\nFactual answer:"
inputs_warm_up = tokenizer(test_prompt, return_tensors="pt", padding="max_length", max_length=SEQ_LEN_IN)
with torch.no_grad():
    _ = model.generate(inputs_warm_up.input_ids.npu(), attention_mask=inputs_warm_up.attention_mask.npu(), max_new_tokens=SEQ_LEN_OUT)

# inference using huggingface's generate api
print("--------------inference--------------")
prompt = ["Common sense questions and answers\n\nQuestion: What is the capital of France\nFactual answer:"]
inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=SEQ_LEN_IN)

start_time = time.time()
with torch.no_grad():
    generate_ids = model.generate(inputs.input_ids.npu(), attention_mask=inputs.attention_mask.npu(), max_new_tokens=SEQ_LEN_OUT)
end_time = time.time()

# decode
res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
for item in res:
    print(item)

# time analysis
new_tokens = len(generate_ids[0]) - len(inputs.input_ids[0])
elapse = end_time - start_time
print(f"Output generated in {elapse:.2f}s, {(new_tokens/elapse):.2f} tokens/s, {new_tokens} new tokens generated.")                                                                                                                                                                                     ")