#!/bin/bash

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)

source "${main_dir}/configs/model_glm_130b.sh"

SEED=1234
MAX_OUTPUT_LENGTH=256
MIN_GEN_LENGTH=0
# BeamSearchStrategy args
NUM_BEAMS=4
LENGTH_PENALTY=1.0
NO_REPEAT_NGRAM=3
# BaseStrategy args
TEMP=1.0
TOPK=40
# TOPK=0
TOPP=0.7

ARGS="${main_dir}/generate.py \
       --seed $SEED \
       --mode inference \
       --sampling-strategy BaseStrategy \
       --out-seq-length $MAX_OUTPUT_LENGTH \
       --min-gen-length $MIN_GEN_LENGTH \
       --num-beams $NUM_BEAMS \
       --length-penalty $LENGTH_PENALTY \
       --no-repeat-ngram-size $NO_REPEAT_NGRAM \
       --temperature $TEMP \
       --top_k $TOPK \
       --top_p $TOPP \
       --output-path samples \
       $MODEL_ARGS \
       $*"

# run_cmd="torchrun --nproc_per_node $MP_SIZE --master_port 12233 ${ARGS}"
# echo ${run_cmd}
# eval ${run_cmd}

# 请根据机器上的NUMA亲和性配置每个芯片对应的NUMA node映射,并通过numactl进行绑核
# 查询NUMA node命令示例：lspci -vs c1:00.0
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
for((RANK_ID=$RANK_ID_START;RANK_ID<$((WORLD_SIZE+RANK_ID_START));RANK_ID++));
do
    export LOCAL_RANK=$RANK_ID
    export WORLD_SIZE=$WORLD_SIZE
    bind=${map["$RANK_ID"]}
    echo "Device ID: $RANK_ID, bind to NUMA node: $bind"
    numactl --cpunodebind=$bind --membind $bind python3 ${ARGS} &
done
wait
