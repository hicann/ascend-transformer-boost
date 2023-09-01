import transformers
from transformers import AutoTokenizer, AutoModel
import platform
import os
import torch
import sys

# 适配昇腾NPU
import torch_npu
from torch_npu.contrib import transfer_to_npu

ACLTRANSFORMER_HOME_PATH = os.environ.get("ACLTRANSFORMER_HOME_PATH")
if ACLTRANSFORMER_HOME_PATH is None:
    raise RuntimeError(
        "env ACLTRANSFORMER_HOME_PATH not exist, source set_env.sh")
CHATGLM2_6B_PATH = os.path.join(ACLTRANSFORMER_HOME_PATH, "../../examples/chatglm2_6b")
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

def generate_data(batch, seq_len):
    past_key_values = None
    input_ids = torch.randint(3, 65024, (batch, seq_len)).npu()
    input_ids[:, -4] = 13
    input_ids[:, -3] = 13
    input_ids[:, -2] = 55437
    input_ids[:, -1] = 31211
    # past_key_values = ()
    # for i in range(28):
    #     k_cache = torch.rand(seq_len, batch, 32, 128)
    #     v_cache = torch.rand(seq_len, batch, 32, 128)
    #     past_key_values = past_key_values + ((k_cache, v_cache),)
    input = {
        "input_ids":input_ids,
        "past_key_values":past_key_values,
    }
    return input

def get_random_input(batch, seq_len):
    input_ids_path = os.path.join(CHATGLM2_6B_PATH, "random_input_ids.pth")
    past_key_path = []
    past_values_path = []
    for i in range(28):
        past_key_path.append(os.path.join(CHATGLM2_6B_PATH, f"random_past_key{i}.pth"))
    for i in range(28):
        past_values_path.append(os.path.join(CHATGLM2_6B_PATH, f"random_past_value{i}.pth"))
    if os.path.exists(input_ids_path):
        input_ids = torch.load(input_ids_path).npu()
        # past_key_values = ()
        # for i in range(28):
        #     k_cache = torch.load(past_key_path[i]).npu()
        #     v_cache = torch.load(past_values_path[i]).npu()
        #     past_key_values = past_key_values + ((k_cache, v_cache),)
        input = {
            "input_ids":input_ids,
            "past_key_values":None,
        }
    else:
        input = generate_data(batch, seq_len)
        torch.save(input["input_ids"].cpu(), input_ids_path)
        # for i in range(28):
        #     torch.save(input["past_key_values"][i][0].cpu(), past_key_path[i])
        # for i in range(28):
        #     torch.save(input["past_key_values"][i][1].cpu(), past_values_path[i])
    return input



if __name__ == "__main__":
    batch = 1
    seq_len = 1024

    args = sys.argv
    output_file_name = "hidden_states.pth"
    # Access individual arguments
    if len(args) > 1:
        output_file_name = args[1]

    model_inputs = get_random_input(batch, seq_len)
    torch.npu.synchronize()
    outputs = model(
        **model_inputs,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
    )
    torch.npu.synchronize()
    
    torch.save(outputs.logits.cpu(), output_file_name)
