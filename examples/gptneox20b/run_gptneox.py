import torch
import random
import numpy as np
import torch_npu
from transformers import AutoTokenizer
from configuration_gpt_neox import GPTNeoXConfig
from patches.operations.modeling_gpt_neox_only_rope_ops import GPTNeoXForCausalLM

device_id = 4
torch_device = torch.device(f'npu:{device_id}')
torch.npu.set_device(torch_device)

torch.npu.set_compile_mode(jit_compile=False)
option = {}
option["NPU_FUZZY_COMPILE_BLACKLIST"] = "Tril,SoftmaxV2,LayerNormGrad,ReduceProd"
torch.npu.set_option(option)

seed = 128
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch_npu.npu.is_available():
        torch_npu.npu.manual_seed_all(seed)

set_random_seed(seed)

tokenizer = AutoTokenizer.from_pretrained("./", trust_remote_code=True)
model_config = GPTNeoXConfig.from_pretrained("./")

def main_full_model():
    print("model config", model_config)
    model = GPTNeoXForCausalLM.from_pretrained("./", trust_remote_code=True,
                                                 config=model_config).half().npu()
    prompt = "hello, please introduce yourself."
    prompts = [prompt] * 1
    inputs = tokenizer(prompts, return_tensors='pt').to(torch_device)

    # ----do warm up---
    output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=10)

    # ----do generate---
    output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=256)
    answers = tokenizer.batch_decode(output_ids)
    print("answer is", answers)

def main_test_small_model(num_layers=2):
    model_config.num_hidden_layers = num_layers
    print("model config", model_config)
    model = GPTNeoXForCausalLM.from_pretrained("./", trust_remote_code=True,
                                                 config=model_config).half().npu()
    prompt = "hello, please introduce yourself."
    prompts = [prompt] * 1
    inputs = tokenizer(prompts, return_tensors='pt').to(torch_device)
    output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=12)
    answers = tokenizer.batch_decode(output_ids)
    print("answer is", answers)

if __name__ == '__main__':
    main_full_model()