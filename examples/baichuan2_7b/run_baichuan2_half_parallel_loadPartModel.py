import time
import argparse

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu

from transformers import AutoTokenizer
# from modeling_baichuan2_parallel import LlamaForCausalLM


def setup_model_parallel():
    torch.distributed.init_process_group("hccl")
    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    if local_rank == 0:
        torch_npu.npu.set_device(torch.device("npu:0"))
    elif local_rank == 1:
        torch_npu.npu.set_device(torch.device("npu:1"))
    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="load Model weights and run.")
    parser.add_argument(
        "--load_path",
        default = "/data/models/baichuan2/baichuan2-7b-part",
        help="Location of Model weights, which contains model folders"
    )
    args = parser.parse_args()
    # initialize parallel
    local_rank, world_size = setup_model_parallel()

    from modeling_baichuan2_model_v2_parallel_performance import LlamaForCausalLM

    torch.npu.set_compile_mode(jit_compile=False)
    option = {}
    option["NPU_FUZZY_COMPILE_BLACKLIST"] = "ReduceProd"
    torch.npu.set_option(option)

    SEQ_LEN_IN = 128
    SEQ_LEN_OUT = 32

    tokenizer_path = args.load_path + '/tokenizer'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, local_files_only=True, use_fast=False)

    part_model_path = args.load_path + '/part_model/' + str(local_rank) + '/'
    model = LlamaForCausalLM.from_pretrained(part_model_path, trust_remote_code=True, low_cpu_mem_usage=True)
    model = model.half().npu()

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

    print("-------------model_device---------------")
    print(model.device)
    print("---------------warm-up---------------")
    test_prompt = "Hamlet->Shakespeare\nOne Hundred Years of Solitude->"
    inputs_warm_up = tokenizer(test_prompt, return_tensors="pt")

    with torch.no_grad():
        _ = model.generate(inputs_warm_up.input_ids.npu(), 
                           attention_mask=inputs_warm_up.attention_mask.npu(), 
                           max_new_tokens=SEQ_LEN_OUT)

    print("---------------inference---------------")
    # tokenize
    prompt = "登鹳雀楼->王之涣\n夜雨寄北->"
    inputs = tokenizer(prompt, return_tensors="pt")

    # generate
    start_time = time.time()
    with torch.no_grad():
        generate_ids = model.generate(inputs.input_ids.npu(), attention_mask=inputs.attention_mask.npu(), max_new_tokens=SEQ_LEN_OUT)
    end_time = time.time()
    
    # decode
    print(tokenizer.decode(generate_ids[0], skip_special_tokens=True))

    # time analysis
    new_tokens = len(generate_ids[0]) - len(inputs.input_ids[0])
    elapse = end_time - start_time
    print(f"[Output tokens number]: {len(generate_ids[0])}, \n[Input tokens number]: {len(inputs.input_ids[0])}, \n[total new tokens generated]: {new_tokens}")
    print(f"Output generated in {elapse:.2f}s, {(new_tokens/elapse):.2f} tokens/s, {new_tokens} new tokens generated.")