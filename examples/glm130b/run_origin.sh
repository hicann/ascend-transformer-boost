#!/bin/bash
cp -f SwissArmyTransformer/model/transformer_origin.py SwissArmyTransformer/model/transformer.py
cp -f SwissArmyTransformer/model/official/glm130B_model_origin.py SwissArmyTransformer/model/official/glm130B_model.py

bash scripts/generate.sh --input-source input.txt
