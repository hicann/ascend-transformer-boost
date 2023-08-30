#!/bin/bash
# file="lcal.o"
# if [ ! -e "$file" ]; then
#     echo "lcal.o not exist, copy..."
#     cp ${ACLTRANSFORMER_HOME_PATH}/lib/lcal.o ./
# fi
export ACLTRANSFORMER_PLAN_EXECUTE_ASYNC=1
export TASK_QUEUE_ENABLE=1
cp -f SwissArmyTransformer/model/transformer_encoder_decoder_fusion_performance.py SwissArmyTransformer/model/transformer.py
cp -f SwissArmyTransformer/model/official/glm130B_model_decoder.py SwissArmyTransformer/model/official/glm130B_model.py
cp -f SwissArmyTransformer/generation/autoregressive_sampling_performance.py SwissArmyTransformer/generation/autoregressive_sampling.py
cp -f evaluation/model_performance.py evaluation/model.py

bash scripts/generate.sh --input-source input.txt
