SCRIPT_DIR=$(cd $(dirname $0); pwd)
MODEL_TARGET_DIR=$SCRIPT_DIR
SCRIPT_PATH=$SCRIPT_DIR/transformers_patch/layer/modeling_llama_layer_performance.py
TRANSFORMER_PACKAGE_PATH=$(python3 -c 'import transformers; import os; print(os.path.dirname(transformers.__file__))')
RUN_OPTION_LIST="--run --performance --webdemo --zhipu --profiling"
MODEL_LIST="--llama1-7b --llama1-13b --llama2-7b --llama2-13b"

function fn_prepare_llama1_7b()
{
    if [ ! -f "$MODEL_TARGET_DIR/pytorch_model-00033-of-00033.bin" ];then
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
}

function fn_prepare_llama1_13b()
{
    if [ ! -f "$MODEL_TARGET_DIR/pytorch_model-00041-of-00041.bin" ];then
        ln -s /data/acltransformer_testdata/weights/llama/llama1-13b/config.json $MODEL_TARGET_DIR/config.json
        ln -s /data/acltransformer_testdata/weights/llama/llama1-13b/generation_config.json $MODEL_TARGET_DIR/generation_config.json
        ln -s /data/acltransformer_testdata/weights/llama/llama1-13b/pytorch_model.bin.index.json $MODEL_TARGET_DIR/pytorch_model.bin.index.json
        ln -s /data/acltransformer_testdata/weights/llama/llama1-13b/special_tokens_map.json $MODEL_TARGET_DIR/special_tokens_map.json
        ln -s /data/acltransformer_testdata/weights/llama/llama1-13b/tokenizer_config.json $MODEL_TARGET_DIR/tokenizer_config.json
        ln -s /data/acltransformer_testdata/weights/llama/llama1-13b/tokenizer.model $MODEL_TARGET_DIR/tokenizer.model
        ln -s /data/acltransformer_testdata/weights/llama/llama1-13b/pytorch_model-00001-of-00041.bin $MODEL_TARGET_DIR/pytorch_model-00001-of-00041.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-13b/pytorch_model-00002-of-00041.bin $MODEL_TARGET_DIR/pytorch_model-00002-of-00041.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-13b/pytorch_model-00003-of-00041.bin $MODEL_TARGET_DIR/pytorch_model-00003-of-00041.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-13b/pytorch_model-00004-of-00041.bin $MODEL_TARGET_DIR/pytorch_model-00004-of-00041.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-13b/pytorch_model-00005-of-00041.bin $MODEL_TARGET_DIR/pytorch_model-00005-of-00041.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-13b/pytorch_model-00006-of-00041.bin $MODEL_TARGET_DIR/pytorch_model-00006-of-00041.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-13b/pytorch_model-00007-of-00041.bin $MODEL_TARGET_DIR/pytorch_model-00007-of-00041.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-13b/pytorch_model-00008-of-00041.bin $MODEL_TARGET_DIR/pytorch_model-00008-of-00041.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-13b/pytorch_model-00009-of-00041.bin $MODEL_TARGET_DIR/pytorch_model-00009-of-00041.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-13b/pytorch_model-00010-of-00041.bin $MODEL_TARGET_DIR/pytorch_model-00010-of-00041.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-13b/pytorch_model-00011-of-00041.bin $MODEL_TARGET_DIR/pytorch_model-00011-of-00041.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-13b/pytorch_model-00012-of-00041.bin $MODEL_TARGET_DIR/pytorch_model-00012-of-00041.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-13b/pytorch_model-00013-of-00041.bin $MODEL_TARGET_DIR/pytorch_model-00013-of-00041.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-13b/pytorch_model-00014-of-00041.bin $MODEL_TARGET_DIR/pytorch_model-00014-of-00041.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-13b/pytorch_model-00015-of-00041.bin $MODEL_TARGET_DIR/pytorch_model-00015-of-00041.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-13b/pytorch_model-00016-of-00041.bin $MODEL_TARGET_DIR/pytorch_model-00016-of-00041.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-13b/pytorch_model-00017-of-00041.bin $MODEL_TARGET_DIR/pytorch_model-00017-of-00041.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-13b/pytorch_model-00018-of-00041.bin $MODEL_TARGET_DIR/pytorch_model-00018-of-00041.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-13b/pytorch_model-00019-of-00041.bin $MODEL_TARGET_DIR/pytorch_model-00019-of-00041.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-13b/pytorch_model-00020-of-00041.bin $MODEL_TARGET_DIR/pytorch_model-00020-of-00041.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-13b/pytorch_model-00021-of-00041.bin $MODEL_TARGET_DIR/pytorch_model-00021-of-00041.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-13b/pytorch_model-00022-of-00041.bin $MODEL_TARGET_DIR/pytorch_model-00022-of-00041.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-13b/pytorch_model-00023-of-00041.bin $MODEL_TARGET_DIR/pytorch_model-00023-of-00041.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-13b/pytorch_model-00024-of-00041.bin $MODEL_TARGET_DIR/pytorch_model-00024-of-00041.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-13b/pytorch_model-00025-of-00041.bin $MODEL_TARGET_DIR/pytorch_model-00025-of-00041.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-13b/pytorch_model-00026-of-00041.bin $MODEL_TARGET_DIR/pytorch_model-00026-of-00041.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-13b/pytorch_model-00027-of-00041.bin $MODEL_TARGET_DIR/pytorch_model-00027-of-00041.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-13b/pytorch_model-00028-of-00041.bin $MODEL_TARGET_DIR/pytorch_model-00028-of-00041.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-13b/pytorch_model-00029-of-00041.bin $MODEL_TARGET_DIR/pytorch_model-00029-of-00041.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-13b/pytorch_model-00030-of-00041.bin $MODEL_TARGET_DIR/pytorch_model-00030-of-00041.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-13b/pytorch_model-00031-of-00041.bin $MODEL_TARGET_DIR/pytorch_model-00031-of-00041.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-13b/pytorch_model-00032-of-00041.bin $MODEL_TARGET_DIR/pytorch_model-00032-of-00041.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-13b/pytorch_model-00033-of-00041.bin $MODEL_TARGET_DIR/pytorch_model-00033-of-00041.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-13b/pytorch_model-00034-of-00041.bin $MODEL_TARGET_DIR/pytorch_model-00034-of-00041.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-13b/pytorch_model-00035-of-00041.bin $MODEL_TARGET_DIR/pytorch_model-00035-of-00041.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-13b/pytorch_model-00036-of-00041.bin $MODEL_TARGET_DIR/pytorch_model-00036-of-00041.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-13b/pytorch_model-00037-of-00041.bin $MODEL_TARGET_DIR/pytorch_model-00037-of-00041.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-13b/pytorch_model-00038-of-00041.bin $MODEL_TARGET_DIR/pytorch_model-00038-of-00041.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-13b/pytorch_model-00039-of-00041.bin $MODEL_TARGET_DIR/pytorch_model-00039-of-00041.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-13b/pytorch_model-00040-of-00041.bin $MODEL_TARGET_DIR/pytorch_model-00040-of-00041.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama1-13b/pytorch_model-00041-of-00041.bin $MODEL_TARGET_DIR/pytorch_model-00041-of-00041.bin
    fi
}

function fn_prepare_llama2_7b()
{
    if [ ! -f "$MODEL_TARGET_DIR/pytorch_model-00003-of-00003.bin" ];then
        ln -s /data/acltransformer_testdata/weights/llama/llama2-7b/added_tokens.json $MODEL_TARGET_DIR/added_tokens.json
        ln -s /data/acltransformer_testdata/weights/llama/llama2-7b/config.json $MODEL_TARGET_DIR/config.json
        ln -s /data/acltransformer_testdata/weights/llama/llama2-7b/generation_config.json $MODEL_TARGET_DIR/generation_config.json
        ln -s /data/acltransformer_testdata/weights/llama/llama2-7b/pytorch_model.bin.index.json $MODEL_TARGET_DIR/pytorch_model.bin.index.json
        ln -s /data/acltransformer_testdata/weights/llama/llama2-7b/special_tokens_map.json $MODEL_TARGET_DIR/special_tokens_map.json
        ln -s /data/acltransformer_testdata/weights/llama/llama2-7b/tokenizer_config.json $MODEL_TARGET_DIR/tokenizer_config.json
        ln -s /data/acltransformer_testdata/weights/llama/llama2-7b/tokenizer.json $MODEL_TARGET_DIR/tokenizer.json
        ln -s /data/acltransformer_testdata/weights/llama/llama2-7b/tokenizer.model $MODEL_TARGET_DIR/tokenizer.model
        ln -s /data/acltransformer_testdata/weights/llama/llama2-7b/pytorch_model-00001-of-00003.bin $MODEL_TARGET_DIR/pytorch_model-00001-of-00003.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama2-7b/pytorch_model-00002-of-00003.bin $MODEL_TARGET_DIR/pytorch_model-00002-of-00003.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama2-7b/pytorch_model-00003-of-00003.bin $MODEL_TARGET_DIR/pytorch_model-00003-of-00003.bin
    fi
}


function fn_prepare_llama2_13b()
{
    if [ ! -f "$MODEL_TARGET_DIR/pytorch_model-00006-of-00006.bin" ];then
        ln -s /data/acltransformer_testdata/weights/llama/llama2-13b/added_tokens.json $MODEL_TARGET_DIR/added_tokens.json
        ln -s /data/acltransformer_testdata/weights/llama/llama2-13b/config.json $MODEL_TARGET_DIR/config.json
        ln -s /data/acltransformer_testdata/weights/llama/llama2-13b/generation_config.json $MODEL_TARGET_DIR/generation_config.json
        ln -s /data/acltransformer_testdata/weights/llama/llama2-13b/pytorch_model.bin.index.json $MODEL_TARGET_DIR/pytorch_model.bin.index.json
        ln -s /data/acltransformer_testdata/weights/llama/llama2-13b/special_tokens_map.json $MODEL_TARGET_DIR/special_tokens_map.json
        ln -s /data/acltransformer_testdata/weights/llama/llama2-13b/tokenizer_config.json $MODEL_TARGET_DIR/tokenizer_config.json
        ln -s /data/acltransformer_testdata/weights/llama/llama2-13b/tokenizer.model $MODEL_TARGET_DIR/tokenizer.model
        ln -s /data/acltransformer_testdata/weights/llama/llama2-13b/pytorch_model-00001-of-00006.bin $MODEL_TARGET_DIR/pytorch_model-00001-of-00006.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama2-13b/pytorch_model-00002-of-00006.bin $MODEL_TARGET_DIR/pytorch_model-00002-of-00006.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama2-13b/pytorch_model-00003-of-00006.bin $MODEL_TARGET_DIR/pytorch_model-00003-of-00006.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama2-13b/pytorch_model-00004-of-00006.bin $MODEL_TARGET_DIR/pytorch_model-00004-of-00006.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama2-13b/pytorch_model-00005-of-00006.bin $MODEL_TARGET_DIR/pytorch_model-00005-of-00006.bin
        ln -s /data/acltransformer_testdata/weights/llama/llama2-13b/pytorch_model-00006-of-00006.bin $MODEL_TARGET_DIR/pytorch_model-00006-of-00006.bin
    fi
}

function fn_clean()
{
    rm $MODEL_TARGET_DIR/pytorch_model*
    rm $MODEL_TARGET_DIR/*.json
    rm $MODEL_TARGET_DIR/*.model
}

function fn_main()
{
    echo "-----run.sh-----"
    fn_clean
    
    if [[ ! -z "$1" ]];then
        RUN_OPTION=$1
        echo "[RUN_OPTION]: $RUN_OPTION"
        shift
    fi
    
    if [[ ! -z "$1" ]];then
        MODEL=$1
        echo "[MODEL]: $MODEL"
        shift
    fi
    
    # if [[ ! -z "$1" ]];then
    #     TEMP_SCRIPT_PATH="$1"
    #     if [[ ! -e $TEMP_SCRIPT_PATH ]];then
    #         SCRIPT_PATH=$TEMP_SCRIPT_PATH
    #         echo "[MODEL_SCRIPT_PATH]: $SCRIPT_PATH"
    #         shift
    #     else
    #         echo "WRONG dir"
    #         exit -1
    #     fi
    # fi

    cd $SCRIPT_DIR
    # echo "[TRANSFORMER_PACKAGE_PATH]: $TRANSFORMER_PACKAGE_PATH"

    case "${MODEL}" in
        "--llama1-7b")
            fn_prepare_llama1_7b
            ;;
        "--llama1-13b")
            fn_prepare_llama1_13b
            ;;
        "--llama2-7b")
            fn_prepare_llama2_7b
            ;;
        "--llama2-13b")
            fn_prepare_llama2_13b
            ;;
        "--help")
            echo "run.sh [--run|--performance|--webdemo|--zhipu|--profiling] [--llama1-7b|--llama1-13b|--llama2-7b|--llama2-13b] [model script path]"
            ;;
        *)
            echo "unknown build type:${MODEL}"
            echo "run.sh [--run|--performance|--webdemo|--zhipu|--profiling] [--llama1-7b|--llama1-13b|--llama2-7b|--llama2-13b] [model script path]"
            exit -1
            ;;
    esac

    # cp $SCRIPT_PATH $TRANSFORMER_PACKAGE_PATH/models/llama/modeling_llama.py

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
            echo "run.sh [--run|--performance|--webdemo|--zhipu|--profiling] [--llama1-7b|--llama1-13b|--llama2-7b|--llama2-13b] [model script path]"
            ;;
        *)
            echo "unknown build type:${RUN_OPTION}"
            echo "run.sh [--run|--performance|--webdemo|--zhipu|--profiling] [--llama1-7b|--llama1-13b|--llama2-7b|--llama2-13b] [model script path]"
            exit -1
            ;;
    esac

    fn_clean
}

fn_main "$@"