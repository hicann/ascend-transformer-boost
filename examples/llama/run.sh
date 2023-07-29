#! /bin/bash

#=========================================#
# IN [scripts/build.sh] row78, add [git checkout Feature_730]
# if already compiled 3rdparty, remember to delete 3rdparty file [rm -rf 3rdparty/]
#=========================================#

SCRIPT_DIR=$(cd $(dirname $0); pwd)
cd $SCRIPT_DIR

if [ ! -d "$ACLTRANSFORMER_TESTDATA/weights/llama/vicuna-7b" ];then
    echo "$ACLTRANSFORMER_TESTDATA/weights/llama/vicuna-7b dir not exist, can't run"
    exit
fi

# python3 -m fastchat.serve.cli --model-path $ACLTRANSFORMER_TESTDATA/weights/llama/vicuna-7b --num-gpus 1

# Adjust for MATMUL
export ACLTRANSFORMER_CONVERT_NCHW_TO_ND=1

# Replace modeling_llama.py
cd ./transformers_patch/layer/
bash modeling_llama_layer.sh
cd ../../
python3 llama_run.py