#! /bin/bash

# activate Ascend environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python3 -m fastchat.serve.cli --model-path path/to/FastChat/13B-vicuna --num-gpus 1