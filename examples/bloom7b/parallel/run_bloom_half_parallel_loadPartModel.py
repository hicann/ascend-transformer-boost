import torch
import time
import os
import datetime
from transformers import BloomTokenizerFast, pipeline, BloomForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.bloom.configuration_bloom import BloomConfig
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
    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="load Model weights and run.")
    parser.add_argument(
        "--load_path",
        default = "./model_cut",
        help="Location of Model weights, which contains model folders",
    )
    args = parser.parse_args()
    #initialize parallel
    local_rank, world_size = setup_model_parallel()

    torch.npu.set_compile_mode(jit_compile=False)

    batch_size = [1]
    seq_lens = [32,64,128,256,512,1024]
    out_lens = [32,64,128,256,512,1024]

    test_cases = [(bs, seq, out) for sq in seq_lens for bs in batch_sizes for out in out_lens]
    for batch_size, seq_len, out_len in test_cases:
        tokenizer_path = args.load_path+'/tokenizer'
        tokenizer = BloomTokenizerFast.from_pretrained(tokenizer_path, use_fast=False)
        #tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # pad or not
        part_model_path=args.load_path+'/part_model/'+str(local_rank)+'/'
        model = AutoModelForCausalLM.from_pretrained(part_model_path, torch_dtype=torch.float16).npu()
        # model.resize_token_embeddings(len(tokenizer))  # pad or not
        
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

        print('***********************model_device*******************')
        print(model.device)
        print("---------------warm-up---------------")
        test_prompt = "Common sense questions and answers\n\nQuestion: Why do people need sleep\nFactual answer:"
        inputs_warm_up = tokenizer(test_prompt, return_tensors="pt", max_length=seq_len, truncation=True)

        _ = model.generate(
            inputs_warm_up.input_ids.npu(),
            max_new_tokens=out_len
        )

        print("---------------inference---------------")
        # prompt = ["Common sense questions and answers\n\nQuestion: What is the capital of France\nFactual answer:"]
        prompt = [
        			"Common sense questions and answers\n\nQuestion: How to learn a new language\nFactual answer:",
        		  	"Common sense questions and answers\n\nQuestion: Why do we need to learn a new language\nFactual answer:",
        		  	"Common sense questions and answers\n\nQuestion: What is the best way to study\nFactual answer:"
        		  	"Common sense questions and answers\n\nQuestion: What is the capital of France\nFactual answer:",
        		  	"Common sense questions and answers\n\nQuestion: Should I take notes during class\nFactual answer:",
        		  	"Common sense questions and answers\n\nQuestion: How to use photoshop\nFactual answer:",
        		  	"Common sense questions and answers\n\nQuestion: What is depression\nFactual answer:",
        		  	"Common sense questions and answers\n\nQuestion: Who was the first president of the United States\nFactual answer:",
        		  	"Common sense questions and answers\n\nQuestion: What is the name of the vice president of the United States\nFactual answer:",
        		  	"Common sense questions and answers\n\nQuestion: How to learn programming\nFactual answer:",
        		  	"Common sense questions and answers\n\nQuestion: How does artificial intelligence work\nFactual answer:",
        		  	"Common sense questions and answers\n\nQuestion: Why should we learn math\nFactual answer:",
        		  	"Common sense questions and answers\n\nQuestion: How does machine learning work\nFactual answer:",
        		  	"Common sense questions and answers\n\nQuestion: How to live a happy life\nFactual answer:",
        		  	"Common sense questions and answers\n\nQuestion: How to run a company\nFactual answer:",
        		  	"Common sense questions and answers\n\nQuestion: Who is the CEO of Google\nFactual answer:"
        			"Common sense questions and answers\n\nQuestion: How to learn a new language\nFactual answer:",
        		  	"Common sense questions and answers\n\nQuestion: Why do we need to learn a new language\nFactual answer:",
        		  	"Common sense questions and answers\n\nQuestion: What is the best way to study\nFactual answer:"
        		  	"Common sense questions and answers\n\nQuestion: What is the capital of France\nFactual answer:",
        		  	"Common sense questions and answers\n\nQuestion: Should I take notes during class\nFactual answer:",
        		  	"Common sense questions and answers\n\nQuestion: How to use photoshop\nFactual answer:",
        		  	"Common sense questions and answers\n\nQuestion: What is depression\nFactual answer:",
        		  	"Common sense questions and answers\n\nQuestion: Who was the first president of the United States\nFactual answer:",
        		  	"Common sense questions and answers\n\nQuestion: What is the name of the vice president of the United States\nFactual answer:",
        		  	"Common sense questions and answers\n\nQuestion: How to learn programming\nFactual answer:",
        		  	"Common sense questions and answers\n\nQuestion: How does artificial intelligence work\nFactual answer:",
        		  	"Common sense questions and answers\n\nQuestion: Why should we learn math\nFactual answer:",
        		  	"Common sense questions and answers\n\nQuestion: How does machine learning work\nFactual answer:",
        		  	"Common sense questions and answers\n\nQuestion: How to live a happy life\nFactual answer:",
        		  	"Common sense questions and answers\n\nQuestion: How to run a company\nFactual answer:",
        		  	"Common sense questions and answers\n\nQuestion: Who is the CEO of Google\nFactual answer:"
        		 ]
        start_time = time.time()
        # tokenize
        inputs = tokenizer(prompt[batch_size], return_tensors="pt", padding='max_length', max_length=seq_len, truncation=True)
        print("---------------inputs shape---------------")
        print(inputs.input_ids.shape)
        with torch.no_grad():
            generate_ids = model.generate(inputs.input_ids.npu(), attention_mask = inputs.attention_mask.npu(), max_new_tokens=out_len)
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


