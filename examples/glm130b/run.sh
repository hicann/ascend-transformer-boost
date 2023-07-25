#!/bin/bash
file="lccl.o"
if [ ! -e "$file" ]; then
    echo "lccl.o not exist, copy..."
    cp ${ACLTRANSFORMER_HOME_PATH}/lib/lccl.o ./
fi
bash scripts/generate.sh --input-source input.txt
