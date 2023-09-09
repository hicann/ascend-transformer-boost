#! /bin/bash
export HCCL_BUFFSIZE=110
input_dir="./model"
output_dir="./model_cut"
world_size_=2
TRANSFORMER_PACKAGE_PATH=$(python3 -c 'import transformers; import os; print(os.path.dirname(transformers.__file__))')/models/bloom

if [ ! -d "$output_dir" ];
then
    cp ./patches/modeling_bloom_parallel.py $TRANSFORMER_PACKAGE_PATH/modeling_bloom.py
    echo "Cutted Weight directory does not exist, cuting the weight......"
    python ./cut_model_util.py --input_path $input_dir --output_path $output_dir --world_size $world_size_
fi 
echo "Weight directory exists, runing......"
cp ./patches/modeling_bloom_layer_parallel_accerate.py $TRANSFORMER_PACKAGE_PATH/modeling_bloom.py
torchrun --nproc_per_node 2 run_bloom_half_parallel_loadPartModel.py --load_path $output_dir

