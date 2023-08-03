import time
import torch
from transformers import AutoTokenizer, AutoModel
from transformers import GPTNeoXConfig

import os
import random
import numpy as np

import torch_npu

device_id = 1
torch.npu.set_device(torch.device(f'npu:{device_id}'))

torch.npu.set_compile_mode(jit_compile=False)
option = {}
option["NPU_FUZZY_COMPILE_BLACKLIST"] = "Tril, SoftmaxV2,LayerNormGrad,ReduceProd"
torch.npu.set_option(option)

seed = 128

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch_npu.npu.is_available():
        torch_npu.npu.manual_seed_all(seed)

set_random_seed(seed)

model_path = "/data/models/gptneox_20b/"
tokenizer = AutoTokenizer.from_pretrained(model_path)

def main_full_model(is_prof: bool = False):
    from modeling_gpt_neox import GPTNeoXForCausalLM

    print("Done load tokenizer", time.time())

    config = GPTNeoXConfig.from_pretrained(model_path)
    config.is_decoder = True
    print("Done load model config in cpu", config)
    model = GPTNeoXForCausalLM.from_pretrained(model_path, config=config)
    print("Done load model in cpu", time.time())

    # clear npu cache
    torch_npu.npu.empty_cache()
    torch_npu.npu.reset_peak_memory_stats()

    model.gradient_checkpointing_disable()
    model.eval()
    model.half().npu()

    # now_memory = torch_npu.npu.memory_stats()
    # print("After init model on npu memory stats is", now_memory)
    torch_npu.npu.synchronize()
    peak_memory = torch_npu.npu.max_memory_allocated()
    print("Done load model to device", time.time(), "peak_memory", peak_memory)

    input_token = ["a"] * 128
    prompt = " ".join(input_token)
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.npu()

    print("Start warm up....")
    start = time.time()
    output = model(inputs)

    torch_npu.npu.synchronize()
    peak_memory = torch_npu.npu.max_memory_allocated()
    print("Done warm up", time.time() - start, "peak_memory", peak_memory)

    torch_npu.npu.empty_cache()
    torch_npu.npu.reset_peak_memory_stats()

    start = time.time()
    output_ids = model.generate(inputs, do_sample=False, max_new_tokens=128)
    torch_npu.npu.synchronize()
    peak_memory = torch_npu.npu.max_memory_allocated()
    print("generate", time.time() - start, "peak_memory", peak_memory)

    output_str = tokenizer.batch_decode(output_ids)[0]

def main_small_model():
    from modeling_gpt_neox import GPTNeoXForCausalLM

    config = GPTNeoXConfig(num_hidden_layers=2, is_decoder=True)
    print("==config", config)

    model = GPTNeoXForCausalLM(config)
    print("==Done init model", time.time())
    model.gradient_checkpointing_disable()
    model.eval()
    model.half().npu()
    print("==Done to device", time.time())

    inputs = tokenizer("My favorite food is fired fished, but it hash lots of fat.", return_tensors="pt").input_ids.npu()

    output_ids = model.generate(inputs, do_sample=False, max_new_tokens=5)

def main_chat():
    from modeling_gpt_neox import GPTNeoXForCausalLM

    print("Done load tokenizer", time.time())

    config = GPTNeoXConfig.from_pretrained(model_path)
    config.is_decoder = True
    # config.num_hidden_layers = 12
    print("Done load model config in cpu", config)
    model = GPTNeoXForCausalLM.from_pretrained(model_path, config=config)
    print("Done load model in cpu", time.time())

    model.gradient_checkpointing_disable()
    model.eval()
    model.half().npu()

    prompt = "hello, please introduce yourself."
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.npu()
    print("inputs is", inputs)

    output_ids = model.generate(inputs, do_sample=False, max_new_tokens=128)
    print("output ids is", output_ids)
    answers = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    print("answer is", answers)

if __name__ == '__main__':
    main_full_model()
    # main_small_model()
    # main_chat()

