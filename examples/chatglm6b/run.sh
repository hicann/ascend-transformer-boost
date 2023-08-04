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

SCRIPT_PATH=$1

if [ $# -ne 1 ];then
    echo "running the original model"
    SCRIPT_PATH="./modeling_chatglm.py"
fi

if [ -f "./modeling_target.py" ];then
    rm -rf ./modeling_target.py
fi

if [ ! -f $SCRIPT_PATH ];then
    echo "cannot find the file to be tested"
    exit 1
fi

ln -s $SCRIPT_PATH ./modeling_target.py 

python3 main.py