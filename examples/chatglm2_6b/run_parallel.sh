# 总入口脚本
# 性能测试时，第一次输入一个短句对模型进行warm up
# warm up完成后，输入clear清除history
# 之后开始性能测试，输入测试用例（五条直线相交，最多能有多少个交点？）后等待测试结果即可
# 测试结果将输出到当前目录下的performance.txt文件中

SCRIPT_DIR=$(cd $(dirname $0); pwd)
MODEL_TARGET_DIR=$SCRIPT_DIR
TRANSFORMER_PACKAGE_PATH=$(python3 -c 'import transformers; import os; print(os.path.dirname(transformers.__file__))')
RUN_OPTION_LIST="--run --performance --webdemo --zhipu --profiling"
RUN_OPTION="--run"
SCRIPT_PATH=$SCRIPT_DIR/modeling_chatglm.py
OUTPUT_DIR="tensor_parallel"
export HCCL_BUFFSIZE=110
export ACLTRANSFORMER_PLAN_EXECUTE_ASYNC=1
export TASK_QUEUE_ENABLE=1
export HCCL_OP_BASE_FFTS_MODE_ENABLE=1
export ACLTRANSFORMER_CONVERT_NCHW_TO_ND=1

function fn_prepare()
{
    echo "$RUN_OPTION $SCRIPT_PATH"

    if [ ! -f "$MODEL_TARGET_DIR/pytorch_model-00001-of-00007.bin" ];then
        ln -s $ACLTRANSFORMER_TESTDATA/weights/chatglm2_6b/pytorch_model-00001-of-00007.bin $MODEL_TARGET_DIR/pytorch_model-00001-of-00007.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/chatglm2_6b/pytorch_model-00002-of-00007.bin $MODEL_TARGET_DIR/pytorch_model-00002-of-00007.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/chatglm2_6b/pytorch_model-00003-of-00007.bin $MODEL_TARGET_DIR/pytorch_model-00003-of-00007.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/chatglm2_6b/pytorch_model-00004-of-00007.bin $MODEL_TARGET_DIR/pytorch_model-00004-of-00007.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/chatglm2_6b/pytorch_model-00005-of-00007.bin $MODEL_TARGET_DIR/pytorch_model-00005-of-00007.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/chatglm2_6b/pytorch_model-00006-of-00007.bin $MODEL_TARGET_DIR/pytorch_model-00006-of-00007.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/chatglm2_6b/pytorch_model-00007-of-00007.bin $MODEL_TARGET_DIR/pytorch_model-00007-of-00007.bin
    fi


    if [ ! -f $SCRIPT_PATH ];then
        echo "cannot find the file to be tested"
        exit 1
    fi

    cp $MODEL_TARGET_DIR/configuration_chatglm.py $TRANSFORMER_PACKAGE_PATH/configuration_chatglm.py
}

function fn_prepare_parallel()
{
    echo "$RUN_OPTION $SCRIPT_PATH"

    if [ ! -f $SCRIPT_PATH ];then
        echo "cannot find the file to be tested"
        exit 1
    fi

    cp $MODEL_TARGET_DIR/configuration_chatglm.py $OUTPUT_DIR/part_model/0/configuration_chatglm.py
    cp $MODEL_TARGET_DIR/configuration_chatglm.py $OUTPUT_DIR/part_model/1/configuration_chatglm.py
    cp $MODEL_TARGET_DIR/quantization.py $OUTPUT_DIR/part_model/0/quantization.py
    cp $MODEL_TARGET_DIR/quantization.py $OUTPUT_DIR/part_model/1/quantization.py
    cp $MODEL_TARGET_DIR/modeling_chatglm.py $OUTPUT_DIR/part_model/0/modeling_chatglm.py
    cp $MODEL_TARGET_DIR/modeling_chatglm.py $OUTPUT_DIR/part_model/1/modeling_chatglm.py
    cp $MODEL_TARGET_DIR/tokenization_chatglm.py $OUTPUT_DIR/tokenizer/tokenization_chatglm.py
}

function fn_clean()
{
    if [ -f $TRANSFORMER_PACKAGE_PATH/configuration_chatglm.py ];then
        rm $TRANSFORMER_PACKAGE_PATH/configuration_chatglm.py
    fi
}

function fn_main()
{
    # check first parameter is a path
    if [ ! -z $1 ];then
        TEMP_SCRIPT_PATH=$(cd $(dirname $1); pwd)/$(basename $1)
        if [ -f $TEMP_SCRIPT_PATH ];then
            cp $TEMP_SCRIPT_PATH $SCRIPT_PATH
            shift
        fi
    fi
    
    # get option
    if [[ ! -z "$1" ]];then
        RUN_OPTION=$1
    fi
    cd $SCRIPT_DIR

    # cut weights first
    if [ ! -d "$OUTPUT_DIR" ] && [ "$RUN_OPTION" != "--cut" ];then
        echo "The weight has not cut, please cut weight first:"
        echo "run_parallel.sh [model script path] [--cut]"
        exit
    fi

    if [ -d "$OUTPUT_DIR" ];then
        if [ "$RUN_OPTION" == "--cut" ]; then
            echo "The cut weights has exist, please remove weights or choose other options:"
            echo "run_parallel.sh [model script path] [--performance|--precision <output_name>|--zhipu]"
            exit
        else
            fn_prepare_parallel
        fi
    fi

    case "${RUN_OPTION}" in
        "--cut")
            fn_prepare
            python3 $SCRIPT_DIR/cut_model_util.py --output_path $OUTPUT_DIR
            ;;
        "--performance")
            torchrun --nproc_per_node 2 --master_port 20000 $SCRIPT_DIR/main_performance.py --parallel --model_path $OUTPUT_DIR
            ;;

        "--multi-batch")
            torchrun --nproc_per_node 2 --master_port 20000 $SCRIPT_DIR/main_batch_performance.py --parallel --model_path $OUTPUT_DIR
            ;;

        # TODO
        # "--webdemo")
        #     unset https_proxy
        #     unset http_proxy
        #     python3 $SCRIPT_DIR/main_web.py
        #     ;;

        "--zhipu")
            torchrun --nproc_per_node 2 --master_port 20000 $SCRIPT_DIR/zhipu_test.py --parallel --model_path $OUTPUT_DIR
            ;;
        
        # TODO
        # "--profiling")  
        #     ;;

        "--precision")
            python3 $SCRIPT_DIR/main_precision.py $2
            ;;
        
        "--help")
            echo "export ACLTRANSFORMER_TESTDATA=<path>"
            echo "run_parallel.sh [model script path] [--cut|--performance|--precision <output_name>|--zhipu]"
            ;;
        *)
            echo "unknown build type:${RUN_OPTION}"
            echo "run_parallel.sh [model script path] [--cut|performance|--precision <output_name>|--zhipu]"
            ;;
    esac

    fn_clean
}

fn_main "$@"
