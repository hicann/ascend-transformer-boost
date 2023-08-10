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
        torch_npu.npu.set_device(4)
    elif local_rank==1:
        torch_npu.npu.set_device(5)
    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="load Model weights and run.")
    parser.add_argument(
        "--load_path",
        default = "/data/models/llama-13b-part_model_2",
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

    tokenizer_path = args.load_path+'/tokenizer'
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # pad or not
    part_model_path=args.load_path+'/part_model/'+str(local_rank)+'/'
    model = LlamaForCausalLM.from_pretrained(part_model_path, torch_dtype=torch.float16).npu()
    model.resize_token_embeddings(len(tokenizer))  # pad or not

    print('***********************model_device*******************')
    print(model.device)
    print("---------------warm-up---------------")
    test_prompt = "Common sense questions and answers\n\nQuestion: Why do people need sleep\nFactual answer:"
    inputs_warm_up = tokenizer(test_prompt, return_tensors="pt", max_length=SEQ_LEN_IN, truncation=True)

    _ = model.generate(
        inputs_warm_up.input_ids.npu(),
        max_new_tokens=SEQ_LEN_OUT
    )

    print("---------------inference---------------")
    prompt = ["Common sense questions and answers\n\nQuestion: What is the capital of France\nFactual answer:"]
    # prompt = [
    # 			"Common sense questions and answers\n\nQuestion: How to learn a new language\nFactual answer:",
    # 		  	"Common sense questions and answers\n\nQuestion: Why do we need to learn a new language\nFactual answer:",
    # 		  	"Common sense questions and answers\n\nQuestion: What is the best way to study\nFactual answer:"
    # 		  	"Common sense questions and answers\n\nQuestion: What is the capital of France\nFactual answer:",
    # 		  	"Common sense questions and answers\n\nQuestion: Should I take notes during class\nFactual answer:",
    # 		  	"Common sense questions and answers\n\nQuestion: How to use photoshop\nFactual answer:",
    # 		  	"Common sense questions and answers\n\nQuestion: What is depression\nFactual answer:",
    # 		  	"Common sense questions and answers\n\nQuestion: Who was the first president of the United States\nFactual answer:",
    # 		  	"Common sense questions and answers\n\nQuestion: What is the name of the vice president of the United States\nFactual answer:",
    # 		  	"Common sense questions and answers\n\nQuestion: How to learn programming\nFactual answer:",
    # 		  	"Common sense questions and answers\n\nQuestion: How does artificial intelligence work\nFactual answer:",
    # 		  	"Common sense questions and answers\n\nQuestion: Why should we learn math\nFactual answer:",
    # 		  	"Common sense questions and answers\n\nQuestion: How does machine learning work\nFactual answer:",
    # 		  	"Common sense questions and answers\n\nQuestion: How to live a happy life\nFactual answer:",
    # 		  	"Common sense questions and answers\n\nQuestion: How to run a company\nFactual answer:",
    # 		  	"Common sense questions and answers\n\nQuestion: Who is the CEO of Google\nFactual answer:"
    # 		 ]
    start_time = time.time()
    # tokenize
    inputs = tokenizer(prompt, return_tensors="pt", padding='max_length', max_length=SEQ_LEN_IN)
    print("---------------inputs shape---------------")
    print(inputs.input_ids.shape)
    with torch.no_grad():
        generate_ids = model.generate(inputs.input_ids.npu(), attention_mask = inputs.attention_mask.npu(), max_new_tokens=SEQ_LEN_OUT)
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


