#!/bin/bash
input_dir="/data/models/llama-13b"
output_dir="/data/models/llama-13b-part_model_2_test"
world_size_=2
cut_row_keys_=['q_proj','k_proj','v_proj','gate_proj','up_proj']
cut_col_keys_=['o_proj','down_proj']
if test -d "$output_dir";
then
    echo "Weight directory exists, runing......"
    torchrun --nproc_per_node 2 run_llama_half_parallel_loadPartModel.py --load_path $output_dir
else 
    echo "Cutted Weight directory does not exist, cuting the weight......"
    python ./cut_model_util.py --input_path $input_dir --output_path $output_dir --world_size $world_size_ --cut_row_keys $cut_row_keys_ --cut_col_keys $cut_col_keys_
fi
