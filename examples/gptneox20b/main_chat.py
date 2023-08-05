import time
import torch
from transformers import AutoTokenizer, AutoModel
from configuration_gpt_neox import GPTNeoXConfig

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
def main():
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
    print("Done load model in device", time.time())

    print("欢迎使用GPT-NEOX20B模型，输入内容即可进行对话，仅支持单轮对话，输入'stop'终止程序")
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        inputs = tokenizer(query, return_tensors="pt").input_ids.npu()
        print("inputs is", inputs)
        output_ids = model.generate(inputs, do_sample=False, max_new_tokens=128)
        print("output ids is", output_ids)
        answers = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        print("回答：", answers[0])

if __name__ == '__main__':
    main()
    
