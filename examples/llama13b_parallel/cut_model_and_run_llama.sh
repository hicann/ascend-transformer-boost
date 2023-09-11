#!/bin/bash
# input_dir="/data/acltransformer_testdata/weights/llama/llama1-13b"

declare -A map
map["0"]="3"
map["1"]="3"
map["2"]="2"
map["3"]="2"
map["4"]="0"
map["5"]="0"
map["6"]="1"
map["7"]="1"
export HCCL_WHITELIST_DISABLE=1
RANK_ID_START=0
WORLD_SIZE=8

output_dir="/data/acltransformer_testdata/weights/llama/llama1-65b-model-8/"
world_size_=8
cut_row_keys_=['q_proj','k_proj','v_proj','gate_proj','up_proj']
cut_col_keys_=['o_proj','down_proj']

TRANSFORMER_PACKAGE_PATH=$(python3 -c 'import transformers; import os; print(os.path.dirname(transformers.__file__))')
MODELING_SCRIPT_PATH="/data/acltransformer_testdata/wmj/linear/examples/llama13b_parallel/modeling_llama_parallel_enc_dec_fusion_perf.py"
UTIL_SCRIPT_PATH="/data/acltransformer_testdata/wmj/0904/fusion/examples/llama13b_parallel/utils_182.py"

export ACLTRANSFORMER_CONVERT_NCHW_TO_ND=1
# export ASDOPS_LOG_TO_FILE=1

if test -d "$output_dir";
then
    echo "Weight directory exists, runing......"
    cp $MODELING_SCRIPT_PATH $TRANSFORMER_PACKAGE_PATH/models/llama/modeling_llama.py
    # cp $UTIL_SCRIPT_PATH $TRANSFORMER_PACKAGE_PATH/generation/utils.py
    # torchrun --nproc_per_node 8 run_llama_half_parallel_loadPartModel.py --load_path $output_dir
    for((RANK_ID=$RANK_ID_START;RANK_ID<$((WORLD_SIZE+RANK_ID_START));RANK_ID++));
    do
    export LOCAL_RANK=$RANK_ID
    export WORLD_SIZE=$WORLD_SIZE
    bind=${map["$RANK_ID"]}
    echo "Device ID: $RANK_ID, bind to NUMA node: $bind"
    numactl --cpunodebind=$bind --membind $bind python3 run_llama_half_parallel_loadPartModel.py --load_path $output_dir &
done
wait 
else 
    echo "Cutted Weight directory does not exist, cuting the weight......"
    python ./cut_model_util.py --input_path $input_dir --output_path $output_dir --world_size $world_size_ --cut_row_keys $cut_row_keys_ --cut_col_keys $cut_col_keys_
fi
