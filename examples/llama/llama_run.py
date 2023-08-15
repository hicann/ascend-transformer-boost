import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch_npu
from fastchat.conversation import  get_default_conv_template
import time
import numpy as np

# 适配昇腾NPU
from torch_npu.contrib import transfer_to_npu

# 使用二进制优化，消除动态shape的编译问题
#torch.npu.set_compile_mode(jit_compile=False)
# option = {}
# option["NPU_FUZZY_COMPILE_BLACKLIST"] = "Tril"
#torch.npu.set_option(option)

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

torch.npu.set_device(torch.device("npu:6"))

model_path = "/data/acltransformer_testdata/weights/llama/vicuna-7b/"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_path).half().npu()

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        module.weight.data = module.weight.data.npu_format_cast(2)
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Embedding):
        module.weight.data = module.weight.data.npu_format_cast(2)

SEQ_LEN_IN = 128
BATCH_SIZE = 1
VOCAB_SIZE = 32000
NUM_HEADS = 32
EMBED_SIZE_PER = 128
dummy_input_ids_full = torch.randint(0, VOCAB_SIZE, [BATCH_SIZE, SEQ_LEN_IN], dtype=torch.int).npu()
dummy_attention_mask_full = torch.ones([BATCH_SIZE, SEQ_LEN_IN], dtype=torch.bool).npu()
dummy_position_ids_full = torch.arange(0, SEQ_LEN_IN, dtype=torch.long).repeat(BATCH_SIZE,1).npu()
dummy_position_ids_full = dummy_position_ids_full.view(-1, SEQ_LEN_IN)
dummy_past_key_values = None

# torch.npu.synchronize()
start = time.time()
model_inputs = {
    "input_ids": dummy_input_ids_full,
    "past_key_values": None,
    "position_ids": dummy_position_ids_full,
    # "attention_mask": dummy_attention_mask_full
}
out = model(**model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False, 
            use_cache=True)
# torch.npu.synchronize()
end = time.time()
sum_time = 0
first_time = (end - start) * 1000
print(f"first token: {first_time}ms")

warm_up_cycle = 3
test_cycle = 50
dummy_past_key_values = out.past_key_values
dummy_input_ids_nxt = torch.randint(0, VOCAB_SIZE, [BATCH_SIZE, 1], dtype=torch.int).npu()
seq_length = dummy_input_ids_nxt.shape[1]
pkv_length = dummy_past_key_values[0][0].shape[2]
position_ids = torch.arange(pkv_length, seq_length + pkv_length, dtype=torch.long).repeat(BATCH_SIZE, 1).npu()
dummy_position_ids_nxt = position_ids.view(-1, seq_length)
attention_mask_nxt = torch.ones([BATCH_SIZE, seq_length], dtype=torch.bool).npu()
dummy_attention_mask_nxt = torch.cat([dummy_attention_mask_full, attention_mask_nxt], dim =-1)

for i in range(warm_up_cycle):
    # torch.npu.synchronize()
    model_inputs = {
    "input_ids": dummy_input_ids_nxt,
    "past_key_values": dummy_past_key_values,
    "position_ids": dummy_position_ids_nxt,
    # "attention_mask": dummy_attention_mask_nxt
    }
    out = model(**model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
                use_cache=True)
    # torch.npu.synchronize()
    dummy_past_key_values = out.past_key_values
    print(f"warm up {i}")

for i in range(test_cycle):
    # torch.npu.synchronize()
    start = time.time()
    model_inputs = {
    "input_ids": dummy_input_ids_nxt,
    "past_key_values": dummy_past_key_values,
    "position_ids": dummy_position_ids_nxt,
    # "attention_mask": dummy_attention_mask_nxt
    }
    out = model(**model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False, 
                use_cache=True)
    # torch.npu.synchronize()
    end = time.time()
    cur_time = (end - start) * 1000
    sum_time += cur_time
    print(f"【token{i}】")
    print(cur_time)
    dummy_past_key_values = out.past_key_values
avg_time = sum_time / test_cycle
print(f"average token: {sum_time / test_cycle}ms")
print(f"response time: {first_time + sum_time}ms")
