#!/bin/bash
file="lccl.o"
if [ ! -e "$file" ]; then
    echo "lccl.o not exist, copy..."
    cp ${ACLTRANSFORMER_HOME_PATH}/lib/lccl.o ./
fi

cp -f SwissArmyTransformer/model/transformer_decoder.py SwissArmyTransformer/model/transformer.py
cp -f SwissArmyTransformer/model/official/glm130B_model_decoder.py SwissArmyTransformer/model/official/glm130B_model.py

bash scripts/generate.sh --input-source input.txt
