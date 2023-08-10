SCRIPT_DIR=$(cd $(dirname $0); pwd)
RESULT_DIR=$SCRIPT_DIR/test_results
PYTHON_TOOL_PATH=$ACLTRANSFORMER_HOME_PATH/tools/python_tools
export TEMP_COMPARE_MODEL_PRICISION=ON

if [ -d $RESULT_DIR ];then
    rm -rf $RESULT_DIR
fi

mkdir -p $RESULT_DIR

export TEMP_TEST_TENSOR_SAVE_PATH=$RESULT_DIR/golden/precision/data
export TEMP_TEST_PERFORMANCE_SAVE_PATH=$RESULT_DIR/golden/performance
mkdir -p $RESULT_DIR/golden/precision/data
mkdir -p $RESULT_DIR/golden/performance

echo "Start execute golden!"
bash $SCRIPT_DIR/run_performance.sh < $SCRIPT_DIR/input.txt
echo "Golden execute finished!"

cat $SCRIPT_DIR/test_files.txt | while read filename
do
    export TEMP_TEST_TENSOR_SAVE_PATH=$RESULT_DIR/$filename/precision/data
    export TEMP_TEST_PERFORMANCE_SAVE_PATH=$RESULT_DIR/$filename/performance
    mkdir -p $RESULT_DIR/$filename/precision/data
    mkdir -p $RESULT_DIR/$filename/performance
    echo "Start execute $filename"
    bash $SCRIPT_DIR/run_performance.sh $SCRIPT_DIR/patches/$filename < $SCRIPT_DIR/input.txt
done

echo "Start data compare"

for outer_dir in $(ls $RESULT_DIR)
do
    if [ $outer_dir != "golden" ];then
        OUTER_DIR=$RESULT_DIR/$outer_dir
        for dir_name in $(ls $OUTER_DIR)
        do
            DIR_NAME=$OUTER_DIR/$dir_name
            for tensor_name in $(ls $RESULT_DIR/golden/precision/data)
            do
                mkdir -p $DIR_NAME/precision/compile_log
                if [ -f $DIR_NAME/precision/data/$tensor_name -a -f $RESULT_DIR/golden/precision/data/$tensor_name ];then
                    echo "Compare $DIR_NAME/precision/data/$tensor_name"
                    python3 $PYTHON_TOOL_PATH/tensor_compare.py $DIR_NAME/precision/data/$tensor_name $RESULT_DIR/golden/precision/data/$tensor_name > $DIR_NAME/precision/compile_log/$tensor_name.txt
                fi
            done
        done
    fi
done

echo "Batch test finished!"