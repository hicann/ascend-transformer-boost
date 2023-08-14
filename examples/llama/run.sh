SCRIPT_DIR=$(cd $(dirname $0); pwd)
MODEL_TARGET_DIR=$SCRIPT_DIR
TRANSFORMER_PACKAGE_PATH=$(python3 -c 'import transformers; import os; print(os.path.dirname(transformers.__file__))')
RUN_OPTION_LIST="--run --performance --webdemo --zhipu --profiling"
SCRIPT_PATH=$SCRIPT_DIR/transformers_patch/modeling_llama.py

function fn_prepare_llama1_7b()
{
    echo "$RUN_OPTION $SCRIPT_PATH"
    echo "$TRANSFORMER_PACKAGE_PATH"

    if [ ! -f "$MODEL_TARGET_DIR/pytorch_model-00001-of-00033.bin" ];then
        ln -s /data/acltransformer_testdata/weights/llama/llama1-7b/config.json $MODEL_TARGET_DIR/config.json
        ln -s /data/acltransformer_testdata/weights/llama/llama1-7b/generation_config.json $MODEL_TARGET_DIR/generation_config.json
        ln -s /data/acltransformer_testdata/weights/llama/llama1-7b/pytorch_model.bin.index.json $MODEL_TARGET_DIR/pytorch_model.bin.index.json
        ln -s /data/acltransformer_testdata/weights/llama/llama1-7b/special_tokens_map.json $MODEL_TARGET_DIR/special_tokens_map.json
        ln -s /data/acltransformer_testdata/weights/llama/llama1-7b/tokenizer_config.json $MODEL_TARGET_DIR/tokenizer_config.json
        ln -s /data/acltransformer_testdata/weights/llama/llama1-7b/tokenizer.model $MODEL_TARGET_DIR/tokenizer.model
        ln -s /data/acltransformer_testdata/weights/llama/llama1-7b/pytorch_model-00001-of-00033.bin $MODEL_TARGET_DIR/pytorch_model-00001-of-00033.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-7b/pytorch_model-00002-of-00033.bin $MODEL_TARGET_DIR/pytorch_model-00002-of-00033.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-7b/pytorch_model-00003-of-00033.bin $MODEL_TARGET_DIR/pytorch_model-00003-of-00033.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-7b/pytorch_model-00004-of-00033.bin $MODEL_TARGET_DIR/pytorch_model-00004-of-00033.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-7b/pytorch_model-00005-of-00033.bin $MODEL_TARGET_DIR/pytorch_model-00005-of-00033.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-7b/pytorch_model-00006-of-00033.bin $MODEL_TARGET_DIR/pytorch_model-00006-of-00033.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-7b/pytorch_model-00007-of-00033.bin $MODEL_TARGET_DIR/pytorch_model-00007-of-00033.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-7b/pytorch_model-00008-of-00033.bin $MODEL_TARGET_DIR/pytorch_model-00008-of-00033.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-7b/pytorch_model-00009-of-00033.bin $MODEL_TARGET_DIR/pytorch_model-00009-of-00033.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-7b/pytorch_model-00010-of-00033.bin $MODEL_TARGET_DIR/pytorch_model-00010-of-00033.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-7b/pytorch_model-00011-of-00033.bin $MODEL_TARGET_DIR/pytorch_model-00011-of-00033.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-7b/pytorch_model-00012-of-00033.bin $MODEL_TARGET_DIR/pytorch_model-00012-of-00033.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-7b/pytorch_model-00013-of-00033.bin $MODEL_TARGET_DIR/pytorch_model-00013-of-00033.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-7b/pytorch_model-00014-of-00033.bin $MODEL_TARGET_DIR/pytorch_model-00014-of-00033.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-7b/pytorch_model-00015-of-00033.bin $MODEL_TARGET_DIR/pytorch_model-00015-of-00033.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-7b/pytorch_model-00016-of-00033.bin $MODEL_TARGET_DIR/pytorch_model-00016-of-00033.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-7b/pytorch_model-00017-of-00033.bin $MODEL_TARGET_DIR/pytorch_model-00017-of-00033.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-7b/pytorch_model-00018-of-00033.bin $MODEL_TARGET_DIR/pytorch_model-00018-of-00033.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-7b/pytorch_model-00019-of-00033.bin $MODEL_TARGET_DIR/pytorch_model-00019-of-00033.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-7b/pytorch_model-00020-of-00033.bin $MODEL_TARGET_DIR/pytorch_model-00020-of-00033.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-7b/pytorch_model-00021-of-00033.bin $MODEL_TARGET_DIR/pytorch_model-00021-of-00033.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-7b/pytorch_model-00022-of-00033.bin $MODEL_TARGET_DIR/pytorch_model-00022-of-00033.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-7b/pytorch_model-00023-of-00033.bin $MODEL_TARGET_DIR/pytorch_model-00023-of-00033.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-7b/pytorch_model-00024-of-00033.bin $MODEL_TARGET_DIR/pytorch_model-00024-of-00033.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-7b/pytorch_model-00025-of-00033.bin $MODEL_TARGET_DIR/pytorch_model-00025-of-00033.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-7b/pytorch_model-00026-of-00033.bin $MODEL_TARGET_DIR/pytorch_model-00026-of-00033.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-7b/pytorch_model-00027-of-00033.bin $MODEL_TARGET_DIR/pytorch_model-00027-of-00033.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-7b/pytorch_model-00028-of-00033.bin $MODEL_TARGET_DIR/pytorch_model-00028-of-00033.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-7b/pytorch_model-00029-of-00033.bin $MODEL_TARGET_DIR/pytorch_model-00029-of-00033.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-7b/pytorch_model-00030-of-00033.bin $MODEL_TARGET_DIR/pytorch_model-00030-of-00033.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-7b/pytorch_model-00031-of-00033.bin $MODEL_TARGET_DIR/pytorch_model-00031-of-00033.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-7b/pytorch_model-00032-of-00033.bin $MODEL_TARGET_DIR/pytorch_model-00032-of-00033.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-7b/pytorch_model-00033-of-00033.bin $MODEL_TARGET_DIR/pytorch_model-00033-of-00033.bin
    fi

    cp $SCRIPT_PATH $TRANSFORMER_PACKAGE_PATH/models/llama/modeling_llama.py
}

function fn_main()
{
    if [ ! -z $1 ];then
        TEMP_SCRIPT_PATH=$1
        echo "$TEMP_SCRIPT_PATH"
        if [ -f $TEMP_SCRIPT_PATH ];then
            SCRIPT_PATH=$TEMP_SCRIPT_PATH
            shift
        fi
    fi

    if [[ ! -z "$1" ]];then
        RUN_OPTION=$1
    fi
    cd $SCRIPT_DIR

    fn_prepare_llama1_7b

    case "${RUN_OPTION}" in
        "--run")
            python3 $SCRIPT_DIR/run_llama_performance.py
            ;;
        "--performance")
            ;;
        "--webdemo")
            ;;
        "--zhipu")
            python3 $SCRIPT_DIR/zhipu_test.py
            ;;
        "--profiling")
            ;;
        "--help")
            echo "run.sh [--run|--performance|--webdemo|--zhipu|--profiling] [model script path]"
            ;;
        *)
            echo "unknown build type:${RUN_OPTION}"
            echo "run.sh [--run|--performance|--webdemo|--zhipu|--profiling] [model script path]"
            ;;
    esac

}

fn_main "$@"