#!/bin/bash
input_dir="/data/models/baichuan2"
output_dir="/data/models/baichuan2/baichuan2-7b-part"
world_size_=2
cut_W_pack_keys_=['W_pack']
cut_row_keys_=['gate_proj','up_proj']
cut_col_keys_=['o_proj','down_proj']
if test -d "$output_dir";
then
    echo "Weight directory exists, runing......"
    torchrun --nproc_per_node 2 run_baichuan2_half_parallel_loadPartModel.py --load_path $output_dir
else 
    echo "Cutted Weight directory does not exist, cuting the weight......"
    python ./cut_model_util.py --input_path $input_dir --output_path $output_dir --world_size $world_size_ \
           --cut_W_pack_keys $cut_W_pack_keys_ --cut_row_keys $cut_row_keys_ --cut_col_keys $cut_col_keys_
fi