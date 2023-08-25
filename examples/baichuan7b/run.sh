SCRIPT_DIR=$(cd $(dirname $0); pwd)
MODEL_TARGET_DIR=$SCRIPT_DIR
SCRIPT_PATH=$SCRIPT_DIR/transformers_patch/layer/modeling_baichuan_decoder_layer_performance.py
TRANSFORMER_PACKAGE_PATH=$(python3 -c 'import transformers; import os; print(os.path.dirname(transformers.__file__))')
RUN_OPTION_LIST="--run --performance --webdemo --zhipu --profiling"
MODEL_LIST="--baichuan-7b"

function fn_prepare()
{
    if [ ! -f "$MODEL_TARGET_DIR/pytorch_model.bin" ];then
        ln -s /data/acltransformer_testdata/weights/baichuan/baichuan-7b/config.json $MODEL_TARGET_DIR/config.json
        ln -s /data/acltransformer_testdata/weights/baichuan/baichuan-7b/configuration_baichuan.py $MODEL_TARGET_DIR/configuration_baichuan.py
        ln -s /data/acltransformer_testdata/weights/baichuan/baichuan-7b/generation_config.json $MODEL_TARGET_DIR/generation_config.json
        ln -s /data/acltransformer_testdata/weights/baichuan/baichuan-7b/handler.py $MODEL_TARGET_DIR/handler.py
        ln -s /data/acltransformer_testdata/weights/baichuan/baichuan-7b/special_tokens_map.json $MODEL_TARGET_DIR/special_tokens_map.json
        ln -s /data/acltransformer_testdata/weights/baichuan/baichuan-7b/tokenizer_config.json $MODEL_TARGET_DIR/tokenizer_config.json
        ln -s /data/acltransformer_testdata/weights/baichuan/baichuan-7b/tokenizer.model $MODEL_TARGET_DIR/tokenizer.model
        ln -s /data/acltransformer_testdata/weights/baichuan/baichuan-7b/tokenization_baichuan.py $MODEL_TARGET_DIR/tokenization_baichuan.py
        ln -s /data/acltransformer_testdata/weights/baichuan/baichuan-7b/modeling_baichuan.py $MODEL_TARGET_DIR/modeling_baichuan.py
        ln -s /data/acltransformer_testdata/weights/baichuan/baichuan-7b/pytorch_model.bin $MODEL_TARGET_DIR/pytorch_model.bin
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
    # echo "[TRANSFORMER_PACKAGE_PATH]: $TRANSFORMER_PACKAGE_PATH"

    case "${MODEL}" in
        "--baichuan-7b")
            fn_prepar
            ;;
    esac

    # cp $SCRIPT_PATH $TRANSFORMER_PACKAGE_PATH/models/llama/modeling_llama.py

    case "${RUN_OPTION}" in
        "--run")
            python3 $SCRIPT_DIR/run_baichuan7b.py
            ;;
        "--performance")
            python3 $SCRIPT_DIR/run_baichuan7b.py
            ;;
        # "--webdemo")
        #     ;;
        # "--profiling")
        #     ;;
        "--percision")
            python3 $SCRIPT_DIR/run_baichuan7b.py
            ;;
        "--help")
            echo "run.sh [--run|--performance|--webdemo|--profiling]"
            ;;
        *)
            echo "unknown build type:${RUN_OPTION}"
            echo "run.sh [--run|--performance|--webdemo|--profiling]"
            exit -1
            ;;
    esac

    fn_clean
}

fn_main "$@"