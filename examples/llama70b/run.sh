SCRIPT_DIR=$(cd $(dirname $0); pwd)
MODEL_TARGET_DIR=$SCRIPT_DIR
TRANSFORMER_PACKAGE_PATH=$(python3 -c 'import transformers; import os; print(os.path.dirname(transformers.__file__))')
RUN_OPTION_LIST="--run --performance --zhipu --profiling"
RUN_OPTION="--run"

ACLTRANSFORMER_TESTDATA='/data/models/llama2-70B-parallel-80layer'

export ACLTRANSFORMER_CONVERT_NCHW_TO_ND=1

function fn_prepare()
{
    echo "$RUN_OPTION $SCRIPT_PATH"

    if [ ! -f "$MODEL_TARGET_DIR/tokenizer/tokenizer.model" ];then
        mkdir $MODEL_TARGET_DIR/part_model
        mkdir $MODEL_TARGET_DIR/part_model/0
        mkdir $MODEL_TARGET_DIR/part_model/1
        mkdir $MODEL_TARGET_DIR/part_model/2
        mkdir $MODEL_TARGET_DIR/part_model/3
        mkdir $MODEL_TARGET_DIR/tokenizer

        ln -s $ACLTRANSFORMER_TESTDATA/part_model/0/config.json $MODEL_TARGET_DIR/part_model/0/config.json
        ln -s $ACLTRANSFORMER_TESTDATA/part_model/0/generation_config.json $MODEL_TARGET_DIR/part_model/0/generation_config.json
        ln -s $ACLTRANSFORMER_TESTDATA/part_model/0/pytorch_model.bin.index.json $MODEL_TARGET_DIR/part_model/0/pytorch_model.bin.index.json
        ln -s $ACLTRANSFORMER_TESTDATA/part_model/0/pytorch_model-00001-of-00008.bin $MODEL_TARGET_DIR/part_model/0/pytorch_model-00001-of-00008.bin
        ln -s $ACLTRANSFORMER_TESTDATA/part_model/0/pytorch_model-00002-of-00008.bin $MODEL_TARGET_DIR/part_model/0/pytorch_model-00002-of-00008.bin
        ln -s $ACLTRANSFORMER_TESTDATA/part_model/0/pytorch_model-00003-of-00008.bin $MODEL_TARGET_DIR/part_model/0/pytorch_model-00003-of-00008.bin
        ln -s $ACLTRANSFORMER_TESTDATA/part_model/0/pytorch_model-00004-of-00008.bin $MODEL_TARGET_DIR/part_model/0/pytorch_model-00004-of-00008.bin
        ln -s $ACLTRANSFORMER_TESTDATA/part_model/0/pytorch_model-00005-of-00008.bin $MODEL_TARGET_DIR/part_model/0/pytorch_model-00005-of-00008.bin
        ln -s $ACLTRANSFORMER_TESTDATA/part_model/0/pytorch_model-00006-of-00008.bin $MODEL_TARGET_DIR/part_model/0/pytorch_model-00006-of-00008.bin
        ln -s $ACLTRANSFORMER_TESTDATA/part_model/0/pytorch_model-00007-of-00008.bin $MODEL_TARGET_DIR/part_model/0/pytorch_model-00007-of-00008.bin
        ln -s $ACLTRANSFORMER_TESTDATA/part_model/0/pytorch_model-00008-of-00008.bin $MODEL_TARGET_DIR/part_model/0/pytorch_model-00008-of-00008.bin

        ln -s $ACLTRANSFORMER_TESTDATA/part_model/1/config.json $MODEL_TARGET_DIR/part_model/1/config.json
        ln -s $ACLTRANSFORMER_TESTDATA/part_model/1/generation_config.json $MODEL_TARGET_DIR/part_model/1/generation_config.json
        ln -s $ACLTRANSFORMER_TESTDATA/part_model/1/pytorch_model.bin.index.json $MODEL_TARGET_DIR/part_model/1/pytorch_model.bin.index.json
        ln -s $ACLTRANSFORMER_TESTDATA/part_model/1/pytorch_model-00001-of-00008.bin $MODEL_TARGET_DIR/part_model/1/pytorch_model-00001-of-00008.bin
        ln -s $ACLTRANSFORMER_TESTDATA/part_model/1/pytorch_model-00002-of-00008.bin $MODEL_TARGET_DIR/part_model/1/pytorch_model-00002-of-00008.bin
        ln -s $ACLTRANSFORMER_TESTDATA/part_model/1/pytorch_model-00003-of-00008.bin $MODEL_TARGET_DIR/part_model/1/pytorch_model-00003-of-00008.bin
        ln -s $ACLTRANSFORMER_TESTDATA/part_model/1/pytorch_model-00004-of-00008.bin $MODEL_TARGET_DIR/part_model/1/pytorch_model-00004-of-00008.bin
        ln -s $ACLTRANSFORMER_TESTDATA/part_model/1/pytorch_model-00005-of-00008.bin $MODEL_TARGET_DIR/part_model/1/pytorch_model-00005-of-00008.bin
        ln -s $ACLTRANSFORMER_TESTDATA/part_model/1/pytorch_model-00006-of-00008.bin $MODEL_TARGET_DIR/part_model/1/pytorch_model-00006-of-00008.bin
        ln -s $ACLTRANSFORMER_TESTDATA/part_model/1/pytorch_model-00007-of-00008.bin $MODEL_TARGET_DIR/part_model/1/pytorch_model-00007-of-00008.bin
        ln -s $ACLTRANSFORMER_TESTDATA/part_model/1/pytorch_model-00008-of-00008.bin $MODEL_TARGET_DIR/part_model/1/pytorch_model-00008-of-00008.bin

        ln -s $ACLTRANSFORMER_TESTDATA/part_model/2/config.json $MODEL_TARGET_DIR/part_model/2/config.json
        ln -s $ACLTRANSFORMER_TESTDATA/part_model/2/generation_config.json $MODEL_TARGET_DIR/part_model/2/generation_config.json
        ln -s $ACLTRANSFORMER_TESTDATA/part_model/2/pytorch_model.bin.index.json $MODEL_TARGET_DIR/part_model/2/pytorch_model.bin.index.json
        ln -s $ACLTRANSFORMER_TESTDATA/part_model/2/pytorch_model-00001-of-00003.bin $MODEL_TARGET_DIR/part_model/2/pytorch_model-00001-of-00008.bin
        ln -s $ACLTRANSFORMER_TESTDATA/part_model/2/pytorch_model-00002-of-00003.bin $MODEL_TARGET_DIR/part_model/2/pytorch_model-00002-of-00008.bin
        ln -s $ACLTRANSFORMER_TESTDATA/part_model/2/pytorch_model-00003-of-00003.bin $MODEL_TARGET_DIR/part_model/2/pytorch_model-00003-of-00008.bin
        ln -s $ACLTRANSFORMER_TESTDATA/part_model/2/pytorch_model-00004-of-00003.bin $MODEL_TARGET_DIR/part_model/2/pytorch_model-00004-of-00008.bin
        ln -s $ACLTRANSFORMER_TESTDATA/part_model/2/pytorch_model-00005-of-00003.bin $MODEL_TARGET_DIR/part_model/2/pytorch_model-00005-of-00008.bin
        ln -s $ACLTRANSFORMER_TESTDATA/part_model/2/pytorch_model-00006-of-00003.bin $MODEL_TARGET_DIR/part_model/2/pytorch_model-00006-of-00008.bin
        ln -s $ACLTRANSFORMER_TESTDATA/part_model/2/pytorch_model-00007-of-00003.bin $MODEL_TARGET_DIR/part_model/2/pytorch_model-00007-of-00008.bin
        ln -s $ACLTRANSFORMER_TESTDATA/part_model/2/pytorch_model-00008-of-00003.bin $MODEL_TARGET_DIR/part_model/2/pytorch_model-00008-of-00008.bin

        ln -s $ACLTRANSFORMER_TESTDATA/part_model/3/config.json $MODEL_TARGET_DIR/part_model/3/config.json
        ln -s $ACLTRANSFORMER_TESTDATA/part_model/3/generation_config.json $MODEL_TARGET_DIR/part_model/3/generation_config.json
        ln -s $ACLTRANSFORMER_TESTDATA/part_model/3/pytorch_model.bin.index.json $MODEL_TARGET_DIR/part_model/3/pytorch_model.bin.index.json
        ln -s $ACLTRANSFORMER_TESTDATA/part_model/3/pytorch_model-00001-of-00003.bin $MODEL_TARGET_DIR/part_model/3/pytorch_model-00001-of-00008.bin
        ln -s $ACLTRANSFORMER_TESTDATA/part_model/3/pytorch_model-00002-of-00003.bin $MODEL_TARGET_DIR/part_model/3/pytorch_model-00002-of-00008.bin
        ln -s $ACLTRANSFORMER_TESTDATA/part_model/3/pytorch_model-00003-of-00003.bin $MODEL_TARGET_DIR/part_model/3/pytorch_model-00003-of-00008.bin
        ln -s $ACLTRANSFORMER_TESTDATA/part_model/3/pytorch_model-00004-of-00003.bin $MODEL_TARGET_DIR/part_model/3/pytorch_model-00004-of-00008.bin
        ln -s $ACLTRANSFORMER_TESTDATA/part_model/3/pytorch_model-00005-of-00003.bin $MODEL_TARGET_DIR/part_model/3/pytorch_model-00005-of-00008.bin
        ln -s $ACLTRANSFORMER_TESTDATA/part_model/3/pytorch_model-00006-of-00003.bin $MODEL_TARGET_DIR/part_model/3/pytorch_model-00006-of-00008.bin
        ln -s $ACLTRANSFORMER_TESTDATA/part_model/3/pytorch_model-00007-of-00003.bin $MODEL_TARGET_DIR/part_model/3/pytorch_model-00007-of-00008.bin
        ln -s $ACLTRANSFORMER_TESTDATA/part_model/3/pytorch_model-00008-of-00003.bin $MODEL_TARGET_DIR/part_model/3/pytorch_model-00008-of-00008.bin

        ln -s $ACLTRANSFORMER_TESTDATA/tokenizer/added_tokens.json $MODEL_TARGET_DIR/tokenizer/added_tokens.json
        ln -s $ACLTRANSFORMER_TESTDATA/tokenizer/special_tokens_map.json $MODEL_TARGET_DIR/tokenizer/special_tokens_map.json
        ln -s $ACLTRANSFORMER_TESTDATA/tokenizer/tokenizer_config.json $MODEL_TARGET_DIR/tokenizer/tokenizer_config.json
        ln -s $ACLTRANSFORMER_TESTDATA/tokenizer/tokenizer.model $MODEL_TARGET_DIR/tokenizer/tokenizer.model
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

    cd $SCRIPT_DIR

    case "${MODEL}" in
        "--llama2-70B-parallel-80layer")
            fn_prepare
            ;;
        "--help")
            echo "run.sh [--run|--performance|--webdemo|--zhipu|--profiling] [--llama2-70B-parallel-80layer] [model script path]"
            ;;
        *)
            echo "unknown build type:${MODEL}"
            echo "run.sh [--run|--performance|--webdemo|--zhipu|--profiling] [--llama2-70B-parallel-80layer] [model script path]"
            exit -1
            ;;
    esac

    # cp $SCRIPT_PATH $TRANSFORMER_PACKAGE_PATH/models/llama/modeling_llama.py

    case "${RUN_OPTION}" in
        "--run")
            torchrun --nproc_per_node 4 $SCRIPT_DIR/run_llama70b_parallel_performance.py
            ;;
        "--performance")
            ;;
        "--webdemo")
            ;;
        "--zhipu")
            case "${MODEL}" in
                "--llama2-70B-parallel-80layer")
                    echo "start llama2-70B-parallel-80layer"
                    torchrun --nproc_per_node 4 $SCRIPT_DIR/zhipu_test.py
                    ;;
                "--help")
                    echo "run.sh [--run|--performance|--webdemo|--zhipu|--profiling] [--llama2-70B-parallel-80layer] [model script path]"
                    ;;
                *)
                    echo "unknown build type:${MODEL}"
                    echo "run.sh [--run|--performance|--webdemo|--zhipu|--profiling] [--llama2-70B-parallel-80layer] [model script path]"
                    exit -1
                    ;;
            esac
            ;;
        "--profiling")
            ;;
        "--precision")
            ;;
        "--help")
            echo "run.sh [--run|--performance|--webdemo|--zhipu|--profiling] [--llama2-70B-parallel-80layer] [model script path]"
            ;;
        *)
            echo "unknown build type:${RUN_OPTION}"
            echo "run.sh [--run|--performance|--webdemo|--zhipu|--profiling] [--llama2-70B-parallel-80layer] [model script path]"
            exit -1
            ;;
    esac

    fn_clean
}

fn_main "$@"
