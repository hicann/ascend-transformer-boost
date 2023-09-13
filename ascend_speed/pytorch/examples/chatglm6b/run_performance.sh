# 性能测试脚本
# 测试时，第一次输入一个短句对模型进行warm up
# warm up完成后，输入clear清除history
# 之后开始性能测试，输入测试用例（五条直线相交，最多能有多少个交点？）后等待测试结果即可
# 测试结果将输出到当前目录下的performance.txt文件中
# 运行方式:bash run_performance.sh 模型脚本路径
# 参数1:模型脚本路径, 参数2:是否开启profiling

SCRIPT_DIR=$(cd $(dirname $0); pwd)
SCRIPT_PATH=$(cd $(dirname $1); pwd)/$(basename $1)
CURRENT_DIR=$(pwd)
cd $SCRIPT_DIR
MODEL_TARGET_DIR=$SCRIPT_DIR
OPTION_ARG=$2
TRANSFORMER_PACKAGE_PATH=$(python3 -c 'import transformers; import os; print(os.path.dirname(transformers.__file__))')
pwd

if [ ! -f "$MODEL_TARGET_DIR/pytorch_model-00001-of-00008.bin" ];then
    ln -s $ACLTRANSFORMER_TESTDATA/weights/chatglm6b/pytorch_model-00001-of-00008.bin $MODEL_TARGET_DIR/pytorch_model-00001-of-00008.bin
    ln -s $ACLTRANSFORMER_TESTDATA/weights/chatglm6b/pytorch_model-00002-of-00008.bin $MODEL_TARGET_DIR/pytorch_model-00002-of-00008.bin
    ln -s $ACLTRANSFORMER_TESTDATA/weights/chatglm6b/pytorch_model-00003-of-00008.bin $MODEL_TARGET_DIR/pytorch_model-00003-of-00008.bin
    ln -s $ACLTRANSFORMER_TESTDATA/weights/chatglm6b/pytorch_model-00004-of-00008.bin $MODEL_TARGET_DIR/pytorch_model-00004-of-00008.bin
    ln -s $ACLTRANSFORMER_TESTDATA/weights/chatglm6b/pytorch_model-00005-of-00008.bin $MODEL_TARGET_DIR/pytorch_model-00005-of-00008.bin
    ln -s $ACLTRANSFORMER_TESTDATA/weights/chatglm6b/pytorch_model-00006-of-00008.bin $MODEL_TARGET_DIR/pytorch_model-00006-of-00008.bin
    ln -s $ACLTRANSFORMER_TESTDATA/weights/chatglm6b/pytorch_model-00007-of-00008.bin $MODEL_TARGET_DIR/pytorch_model-00007-of-00008.bin
    ln -s $ACLTRANSFORMER_TESTDATA/weights/chatglm6b/pytorch_model-00008-of-00008.bin $MODEL_TARGET_DIR/pytorch_model-00008-of-00008.bin
fi

if [ ! -z $OPTION_ARG ];then
    if [ $OPTION_ARG == "--profiling" ]
    then
        export TEMP_MODEL_PROFILING=ON
    else
        echo "error argument!"
        exit 1
    fi
fi

if [ -z $SCRIPT_PATH ];then
    echo "running the original model"
    SCRIPT_PATH="$MODEL_TARGET_DIR/modeling_chatglm.py"
fi

if [ -f "$MODEL_TARGET_DIR/modeling_target.py" ];then
    rm -rf $MODEL_TARGET_DIR/modeling_target.py
fi

if [ -f "$TRANSFORMER_PACKAGE_PATH/configuration_chatglm.py" ];then
    rm -rf $TRANSFORMER_PACKAGE_PATH/configuration_chatglm.py
fi

if [ -f "$TRANSFORMER_PACKAGE_PATH/modeling_target.py" ];then
    rm -rf $TRANSFORMER_PACKAGE_PATH/modeling_target.py
fi

if [ ! -f $SCRIPT_PATH ];then
    echo "cannot find the file to be tested"
    exit 1
fi

cp $SCRIPT_PATH $MODEL_TARGET_DIR/modeling_target.py
cp $MODEL_TARGET_DIR/modeling_target.py $TRANSFORMER_PACKAGE_PATH/modeling_target.py
cp $MODEL_TARGET_DIR/configuration_chatglm.py $TRANSFORMER_PACKAGE_PATH/configuration_chatglm.py

python3 $SCRIPT_DIR/main_performance.py

rm $TRANSFORMER_PACKAGE_PATH/modeling_target.py
rm $TRANSFORMER_PACKAGE_PATH/configuration_chatglm.py
if [[ $OPTION_ARG == "--profiling" ]];then
    python3 $ASCEND_HOME_PATH/ascend_speed/tools/profiler/profiler_tool/analysis/msprof/msprof.py export timeline -dir $SCRIPT_DIR/PROF_*
    python3 $ASCEND_HOME_PATH/ascend_speed/tools/profiler/profiler_tool/analysis/msprof/msprof.py export summary -dir $SCRIPT_DIR/PROF_*
fi