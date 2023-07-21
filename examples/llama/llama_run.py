import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch_npu
from fastchat.conversation import  get_default_conv_template

torch.npu.set_device(torch.device("npu:2"))

model_path = "/data/acltransformer_testdata/weights/llama/vicuna-7b/"
conv = get_default_conv_template(model_path).copy()
prompt = conv.get_prompt()

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_path).half().npu()
input_ids = tokenizer(prompt).input_ids
out = model(torch.as_tensor([input_ids]).npu(), use_cache=True)
past_key_values = out.past_key_values
last_token_logits = out.logits[0][-1]
probs = torch.softmax(last_token_logits / 0.7, dim=-1)
token = int(torch.multinomial(probs, num_samples=1))
print("finish first")

out = model(input_ids=torch.as_tensor([[token]]).npu(), use_cache=True, past_key_values=past_key_values)
print("finish second")
