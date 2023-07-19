#! /bin/bash

# activate Ascend environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh

if [ ! -d "vicuna-7b" ];then
    echo "vicuna-7b dir not exist, can't run"
    exit
fi

python3 -m fastchat.serve.cli --model-path ./vicuna-7b --num-gpus 1