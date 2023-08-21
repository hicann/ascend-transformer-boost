#!/bin/bash
# file="lcal.o"
# if [ ! -e "$file" ]; then
#     echo "lcal.o not exist, copy..."
#     cp ${ACLTRANSFORMER_HOME_PATH}/lib/lcal.o ./
# fi
export ACLTRANSFORMER_PLAN_EXECUTE_ASYNC=1
cp -f SwissArmyTransformer/model/transformer_decoder.py SwissArmyTransformer/model/transformer.py
cp -f SwissArmyTransformer/model/official/glm130B_model_decoder.py SwissArmyTransformer/model/official/glm130B_model.py

bash scripts/generate.sh --input-source input.txt
