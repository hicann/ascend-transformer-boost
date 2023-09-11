import torch
import time
import os
import datetime
from transformers import LlamaTokenizer, pipeline, LlamaForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
import torch_npu
from torch_npu.contrib import transfer_to_npu
import argparse
def setup_model_parallel():
    torch.distributed.init_process_group("hccl")
    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    # torch_npu.npu.set_device(local_rank+2)
    if local_rank==0:
        torch_npu.npu.set_device(0)
    elif local_rank==1:
        torch_npu.npu.set_device(1)
    elif local_rank==2:
        torch_npu.npu.set_device(2)
    elif local_rank==3:
        torch_npu.npu.set_device(3)
    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="load Model weights and run.")
    parser.add_argument(
        "--load_path",
        default = "/data/models/llama2-70B-parallel-80layer",
        help="Location of Model weights, which contains model folders",
    )
    args = parser.parse_args()
    #initialize parallel
    local_rank, world_size = setup_model_parallel()

    torch.npu.set_compile_mode(jit_compile=False)
    option = {}
    option["NPU_FUZZY_COMPILE_BLACKLIST"] = "Tril,SoftmaxV2,LayerNormGrad,ReduceProd"
    torch.npu.set_option(option)

    SEQ_LEN_IN = 128
    SEQ_LEN_OUT = 128
    config = LlamaConfig(
        architectures="LlamaForCausalLM",
        bos_token_id=1,
        eos_token_id=2,
        hidden_act="silu",
        hidden_size=8192,
        initializer_range=0.02,
        intermediate_size=28672,
        max_position_embeddings=4096,
        model_type="llama",
        num_attention_heads=64,
        num_hidden_layers=1,
        num_key_value_heads=8,
        pad_token_id=0,
        pretraining_tp=1,
        rms_norm_eps=1e-05,
        rope_scaling=None,
        tie_word_embedding=False,
        torch_dtype="float32",
        transformers_version="4.29.0",
        use_cache=True,
        vocab_size=32000,
        world_size=4
    )

    tokenizer_path = args.load_path+'/tokenizer'
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # pad or not
    part_model_path=args.load_path+'/part_model/'+str(local_rank)+'/'
    model = LlamaForCausalLM.from_pretrained(part_model_path, config=config, torch_dtype=torch.float16).npu()
    model.resize_token_embeddings(len(tokenizer))  # pad or not

    soc_version = torch_npu._C._npu_get_soc_version()
    if soc_version in [104, 220, 221, 222, 223]:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.data = module.weight.data.npu_format_cast(2)
        print("soc version: ", soc_version, " is 910B, support ND.")
    else:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if name == 'lm_head':
                    module.weight.data = torch.nn.parameter.Parameter(module.weight.data)
                module.weight.data = module.weight.data.npu_format_cast(29)
        print("soc version: ", soc_version, " is not 910B, support NZ.")

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding):
            module.weight.data = module.weight.data.npu_format_cast(2)

    print('***********************model_device*******************')
    print(model.device)
    print("---------------warm-up---------------")
    test_prompt = "Common sense questions and answers\n\nQuestion: Why do people need sleep\nFactual answer:"
    inputs_warm_up = tokenizer(test_prompt, return_tensors="pt", max_length=SEQ_LEN_IN, truncation=True)
    with torch.no_grad():
        _ = model.generate(
            inputs_warm_up.input_ids.npu(),
            attention_mask=inputs_warm_up.attention_mask.npu(),
            max_new_tokens=SEQ_LEN_OUT
        )

    print("---------------inference---------------")
    prompt = ["Common sense questions and answers\n\nQuestion: What is the capital of France\nFactual answer:"]
    # tokenize
    inputs = tokenizer(prompt, return_tensors="pt", padding='max_length', max_length=SEQ_LEN_IN)
    start_time = time.time()

    with torch.no_grad():
        generate_ids = model.generate(inputs.input_ids.npu(), attention_mask = inputs.attention_mask.npu(),
                                      max_new_tokens=SEQ_LEN_OUT)
    end_time = time.time()
    # decode
    res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    # output
    for item in res:
        print(item)

    # time analysis
    new_tokens = len(generate_ids[0]) - len(inputs.input_ids[0])
    print(f"Output tokens number: {len(generate_ids[0])},\nInput tokens number: {len(inputs.input_ids[0])},\ntotal new tokens generated: {new_tokens}")
    print(f"Output generated in {(end_time-start_time):.2f} s ({new_tokens/(end_time-start_time):.2f} tokens/s, {new_tokens} tokens)")


