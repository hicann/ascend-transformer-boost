# 总入口脚本
# 性能测试时，第一次输入一个短句对模型进行warm up
# warm up完成后，输入clear清除history
# 之后开始性能测试，输入测试用例（五条直线相交，最多能有多少个交点？）后等待测试结果即可
# 测试结果将输出到当前目录下的performance.txt文件中

SCRIPT_DIR=$(cd $(dirname $0); pwd)
MODEL_TARGET_DIR=$SCRIPT_DIR
TRANSFORMER_PACKAGE_PATH=$(python3 -c 'import transformers; import os; print(os.path.dirname(transformers.__file__))')
RUN_OPTION_LIST="--run --performance --webdemo --zhipu --profiling --prepare"
RUN_OPTION="--run"
SCRIPT_PATH=$SCRIPT_DIR/modeling_chatglm.py

function fn_prepare()
{
    echo "$RUN_OPTION $SCRIPT_PATH"

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
    if [ ! -d "./chatglm_quant_param" ];then
        ln -s $ACLTRANSFORMER_TESTDATA/quant_param/chatglm6b/no_ft ./chatglm_quant_param
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
}

function fn_clean()
{
    if [ -f "$TRANSFORMER_PACKAGE_PATH/configuration_chatglm.py" ];then
        rm -rf $TRANSFORMER_PACKAGE_PATH/configuration_chatglm.py
    fi

    if [ -f "$TRANSFORMER_PACKAGE_PATH/modeling_target.py" ];then
        rm -rf $TRANSFORMER_PACKAGE_PATH/modeling_target.py
    fi
}

function fn_main()
{
    
    if [ ! -z $1 ];then
        TEMP_SCRIPT_PATH=$(cd $(dirname $1); pwd)/$(basename $1)
        if [ -f $TEMP_SCRIPT_PATH ];then
            SCRIPT_PATH=$TEMP_SCRIPT_PATH
            shift
        fi
    fi
    
    if [[ ! -z "$1" ]];then
        RUN_OPTION=$1
    fi
    cd $SCRIPT_DIR

    case "${RUN_OPTION}" in
        "--run")
            fn_prepare
            python3 $SCRIPT_DIR/main.py
            fn_clean
            ;;
        "--performance")
            fn_prepare
            python3 $SCRIPT_DIR/main_performance.py
            fn_clean
            ;;
        "--webdemo")
            fn_prepare
            unset https_proxy
            unset http_proxy
            python3 $SCRIPT_DIR/main_web.py
            fn_clean
            ;;
        "--zhipu")
            fn_prepare
            python3 $SCRIPT_DIR/zhipu_test.py
            fn_clean
            ;;
        "--profiling")
            
            ;;
        "--prepare")
            fn_prepare
            ;;
        "--precision")
            fn_prepare
            export TEMP_TENSOR_SAVE_SWITCH="ON"
            export TEMP_TENSOR_SAVE_PATH=$SCRIPT_DIR
            python3 $SCRIPT_DIR/precision_test.py
            fn_clean
            ;;
        "--help")
            echo "run.sh [model script path] [--run|--performance|--webdemo|--zhipu|--profiling|--precision]"
            ;;
        *)
            echo "unknown build type:${RUN_OPTION}"
            echo "run.sh [model script path] [--run|--performance|--webdemo|--zhipu|--profiling|--precision]"
            ;;
    esac
}

fn_main "$@"
