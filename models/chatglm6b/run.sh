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

python3 main.py