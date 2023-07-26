#!/bin/bash
file="lccl.o"
if [ ! -e "$file" ]; then
    echo "lccl.o not exist, copy..."
    cp ${ACLTRANSFORMER_HOME_PATH}/lib/lccl.o ./
fi

cp -f SwissArmyTransformer/model/transformer_layer.py SwissArmyTransformer/model/transformer.py
cp -f SwissArmyTransformer/model/official/glm130B_model_layer.py SwissArmyTransformer/model/official/glm130B_model.py

bash scripts/generate.sh --input-source input.txt
