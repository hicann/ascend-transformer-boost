# 性能测试脚本
# 测试时，第一次输入一个短句对模型进行warm up
# warm up完成后，输入clear清除history
# 之后开始性能测试，输入测试用例（五条直线相交，最多能有多少个交点？）后等待测试结果即可
# 测试结果将输出到当前目录下的performance.txt文件中

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
if [ ! -d "./chatglm_quant_param" ];then
    ln -s $ACLTRANSFORMER_TESTDATA/quant_param/chatglm6b/no_ft ./chatglm_quant_param
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
python3 main_performance.py
