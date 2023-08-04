SCRIPT_DIR=$(cd $(dirname $0); pwd)
cd $SCRIPT_DIR
pwd
if [ ! -f "./pytorch_model-00001-of-00008.bin" ];then
    ln -s $ACLTRANSFORMER_TESTDATA/weights/chatglm6b/pytorch_model-00001-of-00008.bin ./pytorch_model-00001-of-00008.bin
    ln -s $ACLTRANSFORMER_TESTDATA/weights/chatglm6b/pytorch_model-00002-of-00008.bin ./pytorch_model-00002-of-00008.bin
    ln -s $ACLTRANSFORMER_TESTDATA/weights/chatglm6b/pytorch_model-00003-of-00008.bin ./pytorch_model-00003-of-00008.bin
    ln -s $ACLTRANSFORMER_TESTDATA/weights/chatglm6b/pytorch_model-00004-of-00008.bin ./pytorch_model-00004-of-00008.bin
    ln -s $ACLTRANSFORMER_TESTDATA/weights/chatglm6b/pytorch_model-00005-of-00008.bin ./pytorch_model-00005-of-00008.bin
    ln -s $ACLTRANSFORMER_TESTDATA/weights/chatglm6b/pytorch_model-00006-of-00008.bin ./pytorch_model-00006-of-00008.bin
    ln -s $ACLTRANSFORMER_TESTDATA/weights/chatglm6b/pytorch_model-00007-of-00008.bin ./pytorch_model-00007-of-00008.bin
    ln -s $ACLTRANSFORMER_TESTDATA/weights/chatglm6b/pytorch_model-00008-of-00008.bin ./pytorch_model-00008-of-00008.bin
fi


SCRIPT_TYPE=$1
SCRIPT_NAME=""

if [[ "$SCRIPT_TYPE" == "layers" ]]; then
    echo "tesing a specific layer"
    SCRIPT_NAME=$2
elif [[ "$SCRIPT_TYPE" == "models" ]]; then
    echo "tesing a specific model"
    SCRIPT_NAME=$2
elif [[ "$SCRIPT_TYPE" == "operations" ]]; then
    echo "tesing a specific operation"
    SCRIPT_NAME=$2
else
    echo "running the original model"
    SCRIPT_NAME="modeling_chatglm"
fi

if [ ! -f "./patches/$SCRIPT_TYPE/pytorch_model-00001-of-00008.bin" ];then
    ln -s $ACLTRANSFORMER_TESTDATA/weights/chatglm6b/pytorch_model-00001-of-00008.bin ./patches/$SCRIPT_TYPE/pytorch_model-00001-of-00008.bin
    ln -s $ACLTRANSFORMER_TESTDATA/weights/chatglm6b/pytorch_model-00002-of-00008.bin ./patches/$SCRIPT_TYPE/pytorch_model-00002-of-00008.bin
    ln -s $ACLTRANSFORMER_TESTDATA/weights/chatglm6b/pytorch_model-00003-of-00008.bin ./patches/$SCRIPT_TYPE/pytorch_model-00003-of-00008.bin
    ln -s $ACLTRANSFORMER_TESTDATA/weights/chatglm6b/pytorch_model-00004-of-00008.bin ./patches/$SCRIPT_TYPE/pytorch_model-00004-of-00008.bin
    ln -s $ACLTRANSFORMER_TESTDATA/weights/chatglm6b/pytorch_model-00005-of-00008.bin ./patches/$SCRIPT_TYPE/pytorch_model-00005-of-00008.bin
    ln -s $ACLTRANSFORMER_TESTDATA/weights/chatglm6b/pytorch_model-00006-of-00008.bin ./patches/$SCRIPT_TYPE/pytorch_model-00006-of-00008.bin
    ln -s $ACLTRANSFORMER_TESTDATA/weights/chatglm6b/pytorch_model-00007-of-00008.bin ./patches/$SCRIPT_TYPE/pytorch_model-00007-of-00008.bin
    ln -s $ACLTRANSFORMER_TESTDATA/weights/chatglm6b/pytorch_model-00008-of-00008.bin ./patches/$SCRIPT_TYPE/pytorch_model-00008-of-00008.bin
fi



if [ "$SCRIPT_NAME" != "modeling_chatglm" ]; then
    if [ -f "./patches/$SCRIPT_TYPE/config.json" ];then
        rm ./patches/$SCRIPT_TYPE/config.json
    fi
fi

if [ "$SCRIPT_NAME" != "modeling_chatglm" ]; then
    python3 json_modifier.py $SCRIPT_TYPE $SCRIPT_NAME
fi



if [ "$SCRIPT_NAME" == "modeling_chatglm" ]; then
    python3 main.py ./
else 
    python3 main.py ./patches/$SCRIPT_TYPE/
fi